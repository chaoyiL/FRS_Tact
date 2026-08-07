from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Subset, default_collate

from lerobot.datasets import LeRobotDataset, LeRobotDatasetMetadata, aggregate_stats

from .configuration import JaxSmolVLAConfig
from .preprocessing import JaxSmolVLAPreprocessor

Array = jax.Array
CANONICAL_ACTION_KEY = "action"


class DeterministicEpochBatchSampler:
    """Epoch-addressable batches so resume can jump to the exact data position."""

    def __init__(
        self,
        dataset_size: int,
        *,
        batch_size: int,
        drop_last: bool,
        shuffle: bool,
        seed: int,
        sample_weights: Sequence[float] | None = None,
    ) -> None:
        self.dataset_size = int(dataset_size)
        self.batch_size = int(batch_size)
        self.drop_last = bool(drop_last)
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.sample_weights = (
            None if sample_weights is None else torch.as_tensor(sample_weights, dtype=torch.double)
        )
        if self.dataset_size <= 0 or self.batch_size <= 0:
            raise ValueError("dataset_size and batch_size must be positive")
        if self.sample_weights is not None and len(self.sample_weights) != self.dataset_size:
            raise ValueError("sample_weights length must match dataset_size")
        self.epoch = 0
        self.start_batch = 0

    @property
    def batches_per_epoch(self) -> int:
        if self.drop_last:
            return self.dataset_size // self.batch_size
        return (self.dataset_size + self.batch_size - 1) // self.batch_size

    def set_position(self, *, epoch: int, start_batch: int = 0) -> None:
        if epoch < 0 or not 0 <= start_batch <= self.batches_per_epoch:
            raise ValueError(
                f"invalid sampler position epoch={epoch} start_batch={start_batch}"
            )
        self.epoch = int(epoch)
        self.start_batch = int(start_batch)

    def __iter__(self) -> Iterator[list[int]]:
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        if self.sample_weights is not None:
            order = torch.multinomial(
                self.sample_weights,
                self.dataset_size,
                replacement=True,
                generator=generator,
            )
        elif self.shuffle:
            order = torch.randperm(self.dataset_size, generator=generator)
        else:
            order = torch.arange(self.dataset_size)
        start = self.start_batch * self.batch_size
        for offset in range(start, self.dataset_size, self.batch_size):
            batch = order[offset : offset + self.batch_size]
            if len(batch) < self.batch_size and self.drop_last:
                break
            yield batch.tolist()

    def __len__(self) -> int:
        return self.batches_per_epoch - self.start_batch


@dataclass(frozen=True)
class DatasetSource:
    """One LeRobot dataset to mix into training."""

    repo_id: str
    root: str | Path | None = None
    revision: str | None = None
    episodes: Sequence[int] | None = None
    action_key: str | None = None
    rename_map: Mapping[str, str] | None = None
    weight: float = 1.0


def resolve_action_key(features: Mapping[str, Any], action_key: str | None = None) -> str:
    """Resolve both current ``action`` and legacy/custom ``actions`` feature names."""

    if action_key is not None:
        if action_key not in features:
            raise KeyError(f"action feature {action_key!r} is absent from the dataset")
        return action_key
    matches = [key for key in ("action", "actions") if key in features]
    if len(matches) != 1:
        raise ValueError(
            "could not unambiguously find the dataset action feature; pass action_key explicitly"
        )
    return matches[0]


def action_delta_timestamps(action_key: str, chunk_size: int, fps: int) -> dict[str, list[float]]:
    if fps <= 0:
        raise ValueError(f"dataset FPS must be positive, got {fps}")
    return {action_key: [index / fps for index in range(chunk_size)]}


def canonicalize_dataset_stats(
    stats: Mapping[str, Mapping[str, Any]] | None,
    action_key: str,
) -> dict[str, Mapping[str, Any]]:
    if not stats:
        raise ValueError("the LeRobot dataset has no normalization statistics")
    canonical = dict(stats)
    if action_key not in canonical:
        raise KeyError(f"dataset statistics do not contain action feature {action_key!r}")
    if action_key != CANONICAL_ACTION_KEY:
        canonical[CANONICAL_ACTION_KEY] = canonical.pop(action_key)
    for key in ("observation.state", CANONICAL_ACTION_KEY):
        missing = {"mean", "std"} - set(canonical.get(key, {}))
        if missing:
            raise KeyError(f"dataset statistics for {key!r} are missing {sorted(missing)}")
    return canonical


def rename_dataset_stats(
    stats: Mapping[str, Mapping[str, Any]],
    rename_map: Mapping[str, str] | None,
) -> dict[str, Mapping[str, Any]]:
    rename_map = dict(rename_map or {})
    renamed: dict[str, Mapping[str, Any]] = {}
    for key, value in stats.items():
        renamed[rename_map.get(key, key)] = value
    return renamed


def ensure_stats_counts(
    stats: Mapping[str, Mapping[str, Any]],
    *,
    frame_count: int,
) -> dict[str, dict[str, Any]]:
    """Guarantee each feature has a ``count`` so ``aggregate_stats`` can merge datasets."""
    if frame_count <= 0:
        raise ValueError(f"frame_count must be positive, got {frame_count}")
    out: dict[str, dict[str, Any]] = {}
    for key, feature_stats in stats.items():
        feature = dict(feature_stats)
        if "count" not in feature:
            feature["count"] = np.asarray([frame_count], dtype=np.int64)
        out[key] = feature
    return out


def _collate_lerobot_samples(samples: list[dict[str, Any]]) -> dict[str, Any]:
    keep = {
        key
        for key in samples[0]
        if key.startswith("observation.") or key in (CANONICAL_ACTION_KEY, "action_is_pad", "task")
    }
    return default_collate([{key: sample[key] for key in keep} for sample in samples])


def _to_numpy(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def lerobot_sample_to_observation(sample: Mapping[str, Any]) -> dict[str, np.ndarray]:
    return {key: _to_numpy(value) for key, value in sample.items() if key.startswith("observation.")}


def prepare_lerobot_batch(
    raw_batch: Mapping[str, Any],
    preprocessor: JaxSmolVLAPreprocessor,
    config: JaxSmolVLAConfig,
    action_key: str = CANONICAL_ACTION_KEY,
) -> dict[str, Array]:
    observation = lerobot_sample_to_observation(raw_batch)
    tasks = [str(task) for task in raw_batch["task"]]
    prepared = preprocessor.prepare(observation, tasks)

    actions = _to_numpy(raw_batch[action_key]).astype(np.float32, copy=False)
    expected_prefix = (prepared["state"].shape[0], config.chunk_size)
    if actions.shape[:2] != expected_prefix:
        raise ValueError(f"dataset actions must have shape [B,{config.chunk_size},A], got {actions.shape}")
    prepared["actions"] = preprocessor.normalize_actions(jnp.asarray(actions))

    padding_key = "action_is_pad" if action_key == CANONICAL_ACTION_KEY else f"{action_key}_is_pad"
    if padding_key in raw_batch:
        prepared["action_is_pad"] = jnp.asarray(_to_numpy(raw_batch[padding_key]), dtype=jnp.bool_)
    return prepared


class _KeyMappedLeRobotDataset(Dataset):
    """Normalize keys and apply image augmentation before concatenation."""

    def __init__(
        self,
        dataset: LeRobotDataset,
        *,
        action_key: str,
        rename_map: Mapping[str, str] | None,
        image_transforms: Callable | None = None,
        image_transform_keys: Sequence[str] = (),
    ):
        self.dataset = dataset
        self.action_key = action_key
        self.rename_map = dict(rename_map or {})
        self.image_transforms = image_transforms
        self.image_transform_keys = frozenset(image_transform_keys)
        self.padding_key = f"{action_key}_is_pad"

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.dataset[index]
        mapped: dict[str, Any] = {}
        for key, value in sample.items():
            if key == self.action_key:
                mapped[CANONICAL_ACTION_KEY] = value
            elif key == self.padding_key:
                mapped["action_is_pad"] = value
            elif key.startswith("observation."):
                mapped[self.rename_map.get(key, key)] = value
            elif key == "task":
                mapped["task"] = value
        if self.image_transforms is not None:
            for key in self.image_transform_keys:
                if key in mapped:
                    mapped[key] = self.image_transforms(mapped[key])
        if CANONICAL_ACTION_KEY not in mapped:
            raise KeyError(f"sample is missing action feature {self.action_key!r}")
        return mapped


def resolve_source_visual_keys(
    model_image_keys: Sequence[str],
    rename_map: Mapping[str, str] | None,
    available_cameras: Sequence[str],
    *,
    allow_missing: int = 0,
) -> list[str]:
    """Map model ``image_keys`` back to dataset camera names before rename."""

    if allow_missing < 0:
        raise ValueError(f"allow_missing must be non-negative, got {allow_missing}")
    rename_map = dict(rename_map or {})
    inverse = {dst: src for src, dst in rename_map.items()}
    available = set(available_cameras)
    resolved: list[str] = []
    missing: list[str] = []
    for key in model_image_keys:
        source = inverse.get(key, key)
        if source in available:
            resolved.append(source)
        elif key in available:
            resolved.append(key)
        else:
            missing.append(key)
    if len(missing) > allow_missing:
        raise KeyError(
            f"could not resolve model image keys {missing} via rename_map={rename_map} "
            f"against cameras={list(available_cameras)}; allow_missing={allow_missing}"
        )
    if not resolved:
        raise KeyError(
            f"none of model image keys {list(model_image_keys)} resolve via "
            f"rename_map={rename_map} against cameras={list(available_cameras)}"
        )
    return list(dict.fromkeys(resolved))


def resolve_model_visual_keys(config: JaxSmolVLAConfig) -> tuple[str, ...]:
    """All dataset image/video columns needed by the configured model."""

    return tuple(dict.fromkeys(config.image_keys))


def parse_dataset_sources(cfg: Mapping[str, Any]) -> list[DatasetSource]:
    """Build dataset sources from YAML ``datasets: [{repo_id, ...}, ...]``."""
    raw_datasets = cfg.get("datasets")
    if not isinstance(raw_datasets, list) or not raw_datasets:
        raise ValueError("datasets must be a non-empty list of dataset mappings")
    sources: list[DatasetSource] = []
    for index, item in enumerate(raw_datasets):
        if not isinstance(item, Mapping):
            raise ValueError(f"datasets[{index}] must be a mapping")
        if "repo_id" not in item or not item["repo_id"]:
            raise ValueError(f"datasets[{index}].repo_id is required")
        weight = float(item.get("weight", 1.0))
        if weight <= 0:
            raise ValueError(f"datasets[{index}].weight must be positive")
        rename_map = item.get("rename_map") or {}
        if not isinstance(rename_map, Mapping):
            raise ValueError(f"datasets[{index}].rename_map must be a mapping")
        action_key = item.get("action_key")
        sources.append(
            DatasetSource(
                repo_id=str(item["repo_id"]),
                root=item.get("root"),
                revision=item.get("revision"),
                episodes=item.get("episodes"),
                action_key=None if action_key is None else str(action_key),
                rename_map=dict(rename_map),
                weight=weight,
            )
        )
    return sources


def split_sources_train_val(
    sources: Sequence[DatasetSource],
    *,
    val_fraction: float,
    seed: int,
) -> tuple[list[DatasetSource], list[DatasetSource]]:
    """Hold out a fraction of episodes per dataset for FM validation."""

    if not 0.0 < val_fraction < 1.0:
        raise ValueError(f"val_fraction must be in (0, 1), got {val_fraction}")

    train_sources: list[DatasetSource] = []
    val_sources: list[DatasetSource] = []
    rng = np.random.default_rng(seed)

    for source in sources:
        metadata = LeRobotDatasetMetadata(
            repo_id=source.repo_id,
            root=source.root,
            revision=source.revision,
        )
        if source.episodes is not None:
            episode_ids = np.asarray(list(source.episodes), dtype=np.int64)
        else:
            episode_ids = np.arange(metadata.total_episodes, dtype=np.int64)
        if episode_ids.size == 0:
            raise ValueError(f"dataset {source.repo_id!r} has no episodes to split")
        if episode_ids.size == 1:
            train_sources.append(source)
            continue

        shuffled = episode_ids.copy()
        rng.shuffle(shuffled)
        n_val = max(1, int(round(float(val_fraction) * shuffled.size)))
        n_val = min(n_val, shuffled.size - 1)
        val_ids = sorted(int(x) for x in shuffled[:n_val])
        train_ids = sorted(int(x) for x in shuffled[n_val:])
        train_sources.append(
            DatasetSource(
                repo_id=source.repo_id,
                root=source.root,
                revision=source.revision,
                episodes=train_ids,
                action_key=source.action_key,
                rename_map=source.rename_map,
                weight=source.weight,
            )
        )
        val_sources.append(
            DatasetSource(
                repo_id=source.repo_id,
                root=source.root,
                revision=source.revision,
                episodes=val_ids,
                action_key=source.action_key,
                rename_map=source.rename_map,
                weight=source.weight,
            )
        )
    return train_sources, val_sources


def fixed_stratified_subset_indices(
    dataset_lengths: Sequence[int],
    *,
    sample_count: int,
    seed: int,
) -> tuple[int, ...]:
    """Choose one fixed random subset while retaining coverage across datasets."""

    lengths = np.asarray(dataset_lengths, dtype=np.int64)
    if lengths.ndim != 1 or lengths.size == 0 or np.any(lengths <= 0):
        raise ValueError(f"dataset_lengths must contain positive values, got {list(dataset_lengths)}")
    total_frames = int(lengths.sum())
    if sample_count <= 0:
        raise ValueError(f"sample_count must be positive, got {sample_count}")
    target = min(int(sample_count), total_frames)
    if target == total_frames:
        return tuple(range(total_frames))

    ideal = target * lengths.astype(np.float64) / total_frames
    quotas = np.floor(ideal).astype(np.int64)
    if target >= lengths.size:
        quotas = np.maximum(quotas, 1)
    quotas = np.minimum(quotas, lengths)

    while int(quotas.sum()) < target:
        candidates = np.flatnonzero(quotas < lengths)
        index = int(candidates[np.argmax(ideal[candidates] - quotas[candidates])])
        quotas[index] += 1
    while int(quotas.sum()) > target:
        minimum = 1 if target >= lengths.size else 0
        candidates = np.flatnonzero(quotas > minimum)
        index = int(candidates[np.argmax(quotas[candidates] - ideal[candidates])])
        quotas[index] -= 1

    rng = np.random.default_rng(seed)
    selected: list[int] = []
    offset = 0
    for length, quota in zip(lengths.tolist(), quotas.tolist(), strict=True):
        local = rng.choice(length, size=quota, replace=False)
        selected.extend(offset + int(index) for index in local)
        offset += length
    rng.shuffle(selected)
    return tuple(selected)


class LeRobotJaxDataLoader:
    """JAX batch stream backed by one or more LeRobot datasets."""

    def __init__(
        self,
        checkpoint: str | Path,
        config: JaxSmolVLAConfig,
        *,
        sources: Sequence[DatasetSource],
        batch_size: int = 8,
        num_workers: int = 4,
        prefetch_factor: int = 2,
        video_backend: str | None = None,
        return_uint8: bool = True,
        seed: int = 0,
        local_files_only: bool = True,
        shuffle: bool = True,
        infinite: bool = True,
        drop_last: bool | None = None,
        preprocessor: JaxSmolVLAPreprocessor | None = None,
        image_transforms: Callable | None = None,
        fixed_subset_size: int | None = None,
        fixed_subset_seed: int = 0,
        subset_indices: Sequence[int] | None = None,
    ):
        if batch_size <= 0:
            raise ValueError(f"batch size must be positive, got {batch_size}")
        if num_workers < 0:
            raise ValueError(f"number of workers cannot be negative, got {num_workers}")
        if not sources:
            raise ValueError("at least one dataset source is required")

        self.sources = list(sources)
        self.config = config
        self.action_key = CANONICAL_ACTION_KEY
        self.infinite = bool(infinite)
        self.shuffle = bool(shuffle)
        self.image_transforms = image_transforms

        mapped_datasets: list[_KeyMappedLeRobotDataset] = []
        stats_list: list[dict[str, dict[str, Any]]] = []
        sample_weights: list[float] = []
        self.dataset_summaries: list[dict[str, Any]] = []

        for source in self.sources:
            metadata = LeRobotDatasetMetadata(
                repo_id=source.repo_id,
                root=source.root,
                revision=source.revision,
            )
            resolved_action_key = resolve_action_key(metadata.features, source.action_key)
            delta_timestamps = action_delta_timestamps(
                resolved_action_key,
                config.chunk_size,
                metadata.fps,
            )
            source_rename = dict(source.rename_map or {})
            visual_keys = resolve_source_visual_keys(
                config.image_keys,
                source_rename,
                metadata.camera_keys,
                # Match JaxSmolVLAPreprocessor: at least one RGB camera must
                # resolve, while missing model/placeholder keys are handled
                # by its empty-camera policy.
                allow_missing=len(config.image_keys),
            )
            dataset = LeRobotDataset(
                repo_id=source.repo_id,
                root=metadata.root,
                revision=metadata.revision,
                episodes=list(source.episodes) if source.episodes is not None else None,
                delta_timestamps=delta_timestamps,
                image_transforms=None,
                video_backend=video_backend,
                # Keep decoded frames compact across worker boundaries.
                return_uint8=return_uint8,
                download_videos=True,
                visual_keys=visual_keys,
            )
            if len(dataset) == 0:
                raise ValueError(f"dataset {source.repo_id!r} contains no frames")

            self._validate_features(
                config,
                dataset.features,
                resolved_action_key,
                source_rename,
                repo_id=source.repo_id,
            )
            mapped = _KeyMappedLeRobotDataset(
                dataset,
                action_key=resolved_action_key,
                rename_map=source_rename,
                image_transforms=image_transforms,
                image_transform_keys=config.image_keys,
            )
            mapped_datasets.append(mapped)
            sample_weights.extend([float(source.weight)] * len(mapped))

            canonical_stats = rename_dataset_stats(
                canonicalize_dataset_stats(dataset.meta.stats, resolved_action_key),
                source_rename,
            )
            stats_list.append(ensure_stats_counts(canonical_stats, frame_count=len(dataset)))
            self.dataset_summaries.append(
                {
                    "repo_id": source.repo_id,
                    "frames": len(dataset),
                    "episodes": dataset.num_episodes,
                    "fps": dataset.fps,
                    "action_key": resolved_action_key,
                    "weight": source.weight,
                    "visual_keys": list(visual_keys),
                }
            )

        full_dataset: Dataset
        if len(mapped_datasets) == 1:
            full_dataset = mapped_datasets[0]
        else:
            full_dataset = ConcatDataset(mapped_datasets)
        self.full_dataset_size = len(full_dataset)
        if fixed_subset_size is not None and subset_indices is not None:
            raise ValueError("pass either fixed_subset_size or subset_indices, not both")
        if (fixed_subset_size is not None or subset_indices is not None) and self.shuffle:
            raise ValueError("fixed validation subsets require shuffle=False")

        if subset_indices is not None:
            selected = tuple(int(index) for index in subset_indices)
            if not selected:
                raise ValueError("subset_indices cannot be empty")
            if len(set(selected)) != len(selected):
                raise ValueError("subset_indices must be unique")
            if min(selected) < 0 or max(selected) >= self.full_dataset_size:
                raise ValueError(
                    f"subset index outside [0, {self.full_dataset_size}): "
                    f"min={min(selected)} max={max(selected)}"
                )
        elif fixed_subset_size is not None:
            selected = fixed_stratified_subset_indices(
                [len(dataset) for dataset in mapped_datasets],
                sample_count=int(fixed_subset_size),
                seed=int(fixed_subset_seed),
            )
        else:
            selected = ()

        self.subset_indices = selected
        self.dataset = Subset(full_dataset, list(selected)) if selected else full_dataset
        dataset_size = len(self.dataset)
        if dataset_size <= 0:
            raise ValueError("dataset is empty")
        if drop_last is None:
            drop_last = bool(infinite)
        if drop_last and dataset_size < batch_size:
            raise ValueError(
                f"combined datasets contain {dataset_size} frames, "
                f"smaller than requested batch size {batch_size}"
            )
        effective_batch_size = min(batch_size, dataset_size)
        self.batch_size = int(effective_batch_size)

        if preprocessor is not None:
            self.preprocessor = preprocessor
        else:
            merged_stats = aggregate_stats(stats_list) if len(stats_list) > 1 else stats_list[0]
            # Sample keys are already remapped; keep preprocessor rename_map empty.
            self.preprocessor = JaxSmolVLAPreprocessor(
                checkpoint,
                config,
                rename_map={},
                stats=merged_stats,
                local_files_only=local_files_only,
            )

        use_weighted = (
            self.shuffle
            and len(self.sources) > 1
            and any(source.weight != 1.0 for source in self.sources)
        )
        batch_sampler = DeterministicEpochBatchSampler(
            len(self.dataset),
            batch_size=self.batch_size,
            drop_last=bool(drop_last),
            shuffle=self.shuffle,
            seed=seed,
            sample_weights=sample_weights if use_weighted else None,
        )
        self._batch_sampler = batch_sampler
        self._worker_generator = torch.Generator().manual_seed(seed)
        loader_kwargs: dict[str, Any] = {
            "batch_sampler": batch_sampler,
            "num_workers": num_workers,
            "persistent_workers": num_workers > 0 and self.infinite,
            "collate_fn": _collate_lerobot_samples,
            "generator": self._worker_generator,
        }
        if num_workers > 0:
            loader_kwargs["prefetch_factor"] = prefetch_factor
            # JAX owns background threads in the training process; forking that
            # process can deadlock. Spawn keeps data workers isolated from the
            # JAX runtime while they decode Parquet/images/videos.
            loader_kwargs["multiprocessing_context"] = "spawn"
        self.loader = DataLoader(self.dataset, **loader_kwargs)

    def _validate_features(
        self,
        config: JaxSmolVLAConfig,
        features: Mapping[str, Any],
        action_key: str,
        rename_map: Mapping[str, str] | None,
        *,
        repo_id: str,
    ) -> None:
        state_shape = tuple(features.get("observation.state", {}).get("shape", ()))
        action_shape = tuple(features[action_key].get("shape", ()))
        if not state_shape or state_shape[-1] > config.max_state_dim:
            raise ValueError(
                f"dataset {repo_id!r} state shape {state_shape} is incompatible with "
                f"max_state_dim={config.max_state_dim}"
            )
        if action_shape != (config.action_dim,):
            raise ValueError(
                f"dataset {repo_id!r} action shape {action_shape} does not match "
                f"checkpoint action_dim={config.action_dim}"
            )
        rename_map = dict(rename_map or {})
        dataset_cameras = {
            rename_map.get(key, key)
            for key, feature in features.items()
            if feature.get("dtype") in ("image", "video")
        }
        if not dataset_cameras.intersection(config.image_keys):
            raise ValueError(
                f"dataset {repo_id!r}: none of the cameras match checkpoint image features "
                f"after renaming: dataset={sorted(dataset_cameras)}, "
                f"checkpoint={sorted(config.image_keys)}"
            )

    def batches(self, *, start_batch: int = 0) -> Iterator[dict[str, Array]]:
        if start_batch < 0:
            raise ValueError(f"start_batch must be non-negative, got {start_batch}")
        batches_per_epoch = self._batch_sampler.batches_per_epoch
        if batches_per_epoch <= 0:
            raise ValueError("data loader produces no complete batches")
        epoch, batch_in_epoch = divmod(int(start_batch), batches_per_epoch)
        while True:
            self._batch_sampler.set_position(epoch=epoch, start_batch=batch_in_epoch)
            self._worker_generator.manual_seed(self._batch_sampler.seed + epoch)
            for raw_batch in self.loader:
                yield prepare_lerobot_batch(
                    raw_batch,
                    self.preprocessor,
                    self.config,
                    self.action_key,
                )
            if not self.infinite:
                break
            epoch += 1
            batch_in_epoch = 0
