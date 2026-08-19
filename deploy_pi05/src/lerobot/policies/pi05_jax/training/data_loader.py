"""Adapted from openpi's `src/openpi/training/data_loader.py` (commit 15a9616).

The protocols (`Dataset` / `IterableDataset` / `DataLoader`), `TransformedDataset`, `FakeDataset`,
`transform_dataset`, `TorchDataLoader`, `_collate_fn`, `_worker_init_fn` and `DataLoaderImpl` are
upstream's, unchanged. Two things differ:

  * **Dataset construction.** Upstream does
    `import lerobot.common.datasets.lerobot_dataset as lerobot_dataset` -- the *official* LeRobot
    package, which collides by name with this repo's own `lerobot` package (see ../README.md for
    why openpi is vendored rather than installed). `create_torch_dataset` below therefore builds
    this repo's `lerobot.datasets.LeRobotDataset` instead, and concatenates one per
    `DataConfig.sources` entry, applying that source's `rename_map` so all sources land in a
    single key space before the shared repack/data transforms run.
  * **No RLDS path.** Upstream's `create_rlds_data_loader` / `RLDSDataLoader` exist for DROID and
    need `openpi.training.droid_rlds_dataset` plus TensorFlow. Not vendored, not needed here.
"""

from collections.abc import Iterator, Mapping, Sequence
import dataclasses
import logging
import multiprocessing
import os
import typing
from typing import Protocol, SupportsIndex, TypeVar

import jax
import jax.numpy as jnp
import numpy as np
import torch

from lerobot.datasets import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.datasets.dataset_sources import DatasetSource, resolve_source_visual_keys
from lerobot.datasets.sample_utils import action_delta_timestamps, resolve_action_key

from .. import model as _model
from .. import transforms as _transforms
from . import config as _config

T_co = TypeVar("T_co", covariant=True)


class Dataset(Protocol[T_co]):
    """Interface for a dataset with random access."""

    def __getitem__(self, index: SupportsIndex) -> T_co:
        raise NotImplementedError("Subclasses of Dataset should implement __getitem__.")

    def __len__(self) -> int:
        raise NotImplementedError("Subclasses of Dataset should implement __len__.")


class IterableDataset(Protocol[T_co]):
    """Interface for an iterable dataset."""

    def __iter__(self) -> Iterator[T_co]:
        raise NotImplementedError("Subclasses of IterableDataset should implement __iter__.")

    def __len__(self) -> int:
        raise NotImplementedError("Subclasses of Dataset should implement __len__.")


class DataLoader(Protocol[T_co]):
    """Interface for a data loader."""

    def data_config(self) -> _config.DataConfig:
        """Get the data config for this data loader."""
        raise NotImplementedError("Subclasses of DataLoader should implement data_config.")

    def __iter__(self) -> Iterator[T_co]:
        raise NotImplementedError("Subclasses of DataLoader should implement __iter__.")


class TransformedDataset(Dataset[T_co]):
    def __init__(self, dataset: Dataset, transforms: Sequence[_transforms.DataTransformFn]):
        self._dataset = dataset
        self._transform = _transforms.compose(transforms)

    def __getitem__(self, index: SupportsIndex) -> T_co:
        return self._transform(self._dataset[index])

    def __len__(self) -> int:
        return len(self._dataset)


class FakeDataset(Dataset):
    def __init__(self, model_config: _model.BaseModelConfig, num_samples: int):
        self._num_samples = num_samples
        self._observation_spec, self._action_spec = model_config.inputs_spec()

    def __getitem__(self, index: SupportsIndex) -> dict:
        rng = jax.random.key(index.__index__())

        def make_from_spec(spec: jax.ShapeDtypeStruct):
            nonlocal rng
            rng, data_rng = jax.random.split(rng)
            # Remove the batch dimension.
            shape = spec.shape[1:]
            if spec.dtype == jnp.float32:
                return jax.random.uniform(data_rng, shape=shape, minval=-1.0, maxval=1.0)
            if spec.dtype == jnp.int32:
                return jax.random.randint(data_rng, shape=shape, minval=0, maxval=2048)
            return jnp.zeros(shape=shape, dtype=spec.dtype)

        observation = jax.tree.map(make_from_spec, self._observation_spec)
        action = jax.tree.map(make_from_spec, self._action_spec)

        return {
            **observation.to_dict(),
            "actions": action,
        }

    def __len__(self) -> int:
        return self._num_samples


# --- Not in upstream openpi. -------------------------------------------------------------------
# Upstream's datasets already agree on column names, because a TrainConfig names exactly one
# `repo_id`. Here several datasets are concatenated, and the pick_tube captures number their
# cameras from zero while the rest of this repo's configs number from one, so each source declares
# a `rename_map`. Applying it as a transform (rather than teaching `RepackTransform` about it)
# keeps ../transforms.py a verbatim copy of upstream.


@dataclasses.dataclass(frozen=True)
class RenameKeys(_transforms.DataTransformFn):
    """Rename top-level sample keys, passing everything else through untouched.

    Collisions are an error rather than last-write-wins: pick_tube's map shifts the whole camera
    numbering (`camera0`->`camera1`, `camera1`->`camera2`), so a half-written map would quietly
    drop one of the two wrist views and train on a duplicated camera.
    """

    rename_map: Mapping[str, str]

    def __call__(self, data: dict) -> dict:
        if not self.rename_map:
            return data
        renamed: dict = {}
        for key, value in data.items():
            new_key = self.rename_map.get(key, key)
            if new_key in renamed:
                raise ValueError(
                    f"rename_map collision: {key!r} and another key both map to {new_key!r}; "
                    f"rename_map={dict(self.rename_map)}"
                )
            renamed[new_key] = value
        return renamed


@dataclasses.dataclass(frozen=True)
class PromptFromTask(_transforms.DataTransformFn):
    """Copy this repo's `task` column into openpi's `prompt` key.

    Upstream's `transforms.PromptFromLeRobotTask` looks a `task_index` up in the dataset metadata;
    this repo's `LeRobotDataset` hands back the task string directly, so no lookup is needed.
    """

    def __call__(self, data: dict) -> dict:
        if (task := data.get("task")) is None:
            raise ValueError("prompt_from_task is set but the sample has no 'task' key")
        return {**data, "prompt": np.asarray(task)}


def _create_source_dataset(
    source: DatasetSource,
    data_config: _config.DataConfig,
    action_horizon: int,
) -> Dataset:
    """One `LeRobotDataset`, renamed into the common key space the repack transform expects."""
    metadata = LeRobotDatasetMetadata(source.repo_id, root=source.root, revision=source.revision)
    action_key = resolve_action_key(metadata.features, source.action_key)
    rename_map = dict(source.rename_map or {})

    # `image_keys` is written in post-rename space; map it back to this dataset's own camera names
    # so only the cameras the model consumes get decoded.
    visual_keys = resolve_source_visual_keys(tuple(data_config.image_keys), rename_map, metadata.camera_keys)

    dataset = LeRobotDataset(
        source.repo_id,
        root=source.root,
        revision=source.revision,
        episodes=None if source.episodes is None else list(source.episodes),
        delta_timestamps=action_delta_timestamps(action_key, action_horizon, metadata.fps),
        visual_keys=visual_keys,
        video_backend=data_config.video_backend,
        # `PickTubeInputs._parse_image` wants uint8; skipping the decoder's float32 normalization
        # saves a copy per frame. (It also accepts float CHW, so image-column datasets still work.)
        return_uint8=True,
    )
    logging.info(
        f"dataset={source.repo_id} frames={len(dataset)} episodes={dataset.num_episodes} "
        f"action_key={action_key} visual_keys={visual_keys}"
    )

    # Rename the source's camera columns *and* its action column, so every source presents the
    # same keys to the shared repack transform.
    renames = {**rename_map, action_key: data_config.action_sequence_keys[0]}
    steps: list[_transforms.DataTransformFn] = [RenameKeys(renames)]
    if data_config.prompt_from_task:
        steps.append(PromptFromTask())
    return TransformedDataset(dataset, steps)


def create_torch_dataset(
    data_config: _config.DataConfig, action_horizon: int, model_config: _model.BaseModelConfig
) -> Dataset:
    """Create a dataset for training."""
    repo_id = data_config.repo_id
    if repo_id is None:
        raise ValueError("Repo ID is not set. Cannot create dataset.")
    if repo_id == "fake":
        return FakeDataset(model_config, num_samples=1024)

    if not data_config.sources:
        raise ValueError(
            f"data config for {repo_id!r} declares no `sources`; see LeRobotPickTubeDataConfig in config.py"
        )

    datasets = [_create_source_dataset(source, data_config, action_horizon) for source in data_config.sources]
    if len(datasets) == 1:
        return datasets[0]
    return typing.cast(Dataset, torch.utils.data.ConcatDataset(datasets))


def transform_dataset(dataset: Dataset, data_config: _config.DataConfig, *, skip_norm_stats: bool = False) -> Dataset:
    """Transform the dataset by applying the data transforms."""
    norm_stats = {}
    if data_config.repo_id != "fake" and not skip_norm_stats:
        if data_config.norm_stats is None:
            raise ValueError(
                "Normalization stats not found. "
                "Make sure to run `python tools/compute_pi05_norm_stats.py --config-name=<your-config>`."
            )
        norm_stats = data_config.norm_stats

    return TransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            _transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.model_transforms.inputs,
        ],
    )


def create_data_loader(
    config: _config.TrainConfig,
    *,
    sharding: jax.sharding.Sharding | None = None,
    shuffle: bool = False,
    num_batches: int | None = None,
    skip_norm_stats: bool = False,
) -> DataLoader[tuple[_model.Observation, _model.Actions]]:
    """Create a data loader for training.

    Args:
        config: The training configuration.
        sharding: The sharding to use for the data loader.
        shuffle: Whether to shuffle the data.
        num_batches: Determines the number of batches to return.
        skip_norm_stats: Whether to skip data normalization.
    """
    data_config = config.data.create(config.assets_dirs, config.model)
    logging.info(f"data_config: {data_config}")

    return create_torch_data_loader(
        data_config,
        model_config=config.model,
        action_horizon=config.model.action_horizon,
        batch_size=config.batch_size,
        sharding=sharding,
        shuffle=shuffle,
        num_batches=num_batches,
        num_workers=config.num_workers,
        seed=config.seed,
        skip_norm_stats=skip_norm_stats,
    )


def create_torch_data_loader(
    data_config: _config.DataConfig,
    model_config: _model.BaseModelConfig,
    action_horizon: int,
    batch_size: int,
    *,
    sharding: jax.sharding.Sharding | None = None,
    skip_norm_stats: bool = False,
    shuffle: bool = False,
    num_batches: int | None = None,
    num_workers: int = 0,
    seed: int = 0,
) -> DataLoader[tuple[_model.Observation, _model.Actions]]:
    """Create a data loader for training.

    Args:
        data_config: The data configuration.
        model_config: The model configuration (used to build fake data, if requested).
        action_horizon: The action horizon.
        batch_size: The batch size.
        sharding: The sharding to use for the data loader. If None, the data loader will
            use a single device sharding.
        skip_norm_stats: Whether to skip data normalization.
        shuffle: Whether to shuffle the data.
        num_batches: Determines the number of batches to return. If the number exceeds the
            number of batches in the dataset, the data loader will loop over the dataset.
            If not provided, will iterate over the dataset indefinitely.
        num_workers: The number of worker processes to use. If zero, the data loader will
            execute in the main process.
        seed: The seed to use for shuffling the data.
    """
    dataset = create_torch_dataset(data_config, action_horizon, model_config)
    dataset = transform_dataset(dataset, data_config, skip_norm_stats=skip_norm_stats)

    local_batch_size = batch_size // jax.process_count()
    logging.info(f"local_batch_size: {local_batch_size}")

    data_loader = TorchDataLoader(
        dataset,
        local_batch_size=local_batch_size,
        sharding=sharding,
        shuffle=shuffle,
        num_batches=num_batches,
        num_workers=num_workers,
        seed=seed,
    )

    return DataLoaderImpl(data_config, data_loader)


class TorchDataLoader:
    """Torch data loader implementation."""

    def __init__(
        self,
        dataset,
        local_batch_size: int,
        *,
        sharding: jax.sharding.Sharding | None = None,
        shuffle: bool = False,
        num_batches: int | None = None,
        num_workers: int = 0,
        seed: int = 0,
    ):
        """Create a PyTorch data loader.

        Args:
            dataset: The dataset to load.
            local_batch_size: The local batch size for each process.
            sharding: The sharding to use for the data loader.
            shuffle: Whether to shuffle the data.
            num_batches: If provided, determines the number of returned batches. If the
                number is larger than the number of batches in the dataset, the data loader
                will loop over the dataset. If not provided, will iterate over the dataset
                indefinitely.
            num_workers: The number of worker processes to use. If zero, the data loader will
                execute in the main process.
            seed: The seed to use for shuffling the data.
        """
        if jax.process_count() > 1:
            raise NotImplementedError("Data loading with multiple processes is not supported.")

        if len(dataset) < local_batch_size:
            raise ValueError(f"Local batch size ({local_batch_size}) is larger than the dataset size ({len(dataset)}).")

        if sharding is None:
            # Use data parallel sharding by default.
            sharding = jax.sharding.NamedSharding(
                jax.sharding.Mesh(jax.devices(), ("B",)),
                jax.sharding.PartitionSpec("B"),
            )
        self._sharding = sharding
        self._num_batches = num_batches

        mp_context = None
        if num_workers > 0:
            mp_context = multiprocessing.get_context("spawn")

        generator = torch.Generator()
        generator.manual_seed(seed)
        self._data_loader = torch.utils.data.DataLoader(
            typing.cast(torch.utils.data.Dataset, dataset),
            batch_size=local_batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            multiprocessing_context=mp_context,
            persistent_workers=num_workers > 0,
            collate_fn=_collate_fn,
            worker_init_fn=_worker_init_fn,
            drop_last=True,
            generator=generator,
        )

    @property
    def torch_loader(self) -> torch.utils.data.DataLoader:
        return self._data_loader

    def __iter__(self):
        num_items = 0
        while True:
            data_iter = iter(self._data_loader)
            while True:
                if self._num_batches is not None and num_items >= self._num_batches:
                    return
                try:
                    batch = next(data_iter)
                except StopIteration:
                    break  # We've exhausted the dataset. Create a new iterator and start over.
                num_items += 1
                yield jax.tree.map(lambda x: jax.make_array_from_process_local_data(self._sharding, x), batch)


def _collate_fn(items):
    """Collate the batch elements into batched numpy arrays."""
    # Make sure to convert to numpy arrays before stacking since some of the incoming elements
    # may be JAX arrays.
    return jax.tree.map(lambda *xs: np.stack([np.asarray(x) for x in xs], axis=0), *items)


def _worker_init_fn(worker_id: int) -> None:
    """Tell JAX inside the worker process not to preallocate the GPU memory."""
    # NOTE: This is called after jax is imported inside the worker process. This
    # means that this approach will not work for selecting the backend.
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"


class DataLoaderImpl(DataLoader):
    def __init__(self, data_config: _config.DataConfig, data_loader: TorchDataLoader):
        self._data_config = data_config
        self._data_loader = data_loader

    def data_config(self) -> _config.DataConfig:
        return self._data_config

    def __iter__(self):
        for batch in self._data_loader:
            yield _model.Observation.from_dict(batch), batch["actions"]
