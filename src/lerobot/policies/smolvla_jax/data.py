from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from functools import partial
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import torch
from torch.utils.data import DataLoader, default_collate

from lerobot.datasets import LeRobotDataset, LeRobotDatasetMetadata

from .configuration import JaxSmolVLAConfig
from .preprocessing import JaxSmolVLAPreprocessor

Array = jax.Array


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
    canonical["action"] = canonical.pop(action_key)
    for key in ("observation.state", "action"):
        missing = {"mean", "std"} - set(canonical.get(key, {}))
        if missing:
            raise KeyError(f"dataset statistics for {key!r} are missing {sorted(missing)}")
    return canonical


def _collate_lerobot_samples(samples: list[dict[str, Any]], *, action_key: str) -> dict[str, Any]:
    padding_key = f"{action_key}_is_pad"
    keep = {
        key
        for key in samples[0]
        if key.startswith("observation.") or key in (action_key, padding_key, "task")
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
    action_key: str,
) -> dict[str, Array]:
    observation = lerobot_sample_to_observation(raw_batch)
    tasks = [str(task) for task in raw_batch["task"]]
    prepared = preprocessor.prepare(observation, tasks)

    actions = _to_numpy(raw_batch[action_key]).astype(np.float32, copy=False)
    expected_prefix = (prepared["state"].shape[0], config.chunk_size)
    if actions.shape[:2] != expected_prefix:
        raise ValueError(f"dataset actions must have shape [B,{config.chunk_size},A], got {actions.shape}")
    prepared["actions"] = preprocessor.normalize_actions(jnp.asarray(actions))

    padding_key = f"{action_key}_is_pad"
    if padding_key in raw_batch:
        prepared["action_is_pad"] = jnp.asarray(_to_numpy(raw_batch[padding_key]), dtype=jnp.bool_)
    return prepared


class LeRobotJaxDataLoader:
    """Infinite JAX batch stream backed directly by a LeRobotDataset."""

    def __init__(
        self,
        checkpoint: str | Path,
        config: JaxSmolVLAConfig,
        *,
        repo_id: str,
        root: str | Path | None = None,
        revision: str | None = None,
        episodes: Sequence[int] | None = None,
        action_key: str | None = None,
        rename_map: Mapping[str, str] | None = None,
        batch_size: int = 8,
        num_workers: int = 4,
        prefetch_factor: int = 2,
        video_backend: str | None = None,
        seed: int = 0,
        local_files_only: bool = True,
    ):
        if batch_size <= 0:
            raise ValueError(f"batch size must be positive, got {batch_size}")
        if num_workers < 0:
            raise ValueError(f"number of workers cannot be negative, got {num_workers}")

        # Metadata is loaded first so the action feature can be resolved before
        # constructing the temporal window used by LeRobotDataset.
        metadata = LeRobotDatasetMetadata(
            repo_id=repo_id,
            root=root,
            revision=revision,
        )
        self.action_key = resolve_action_key(metadata.features, action_key)
        delta_timestamps = action_delta_timestamps(
            self.action_key,
            config.chunk_size,
            metadata.fps,
        )

        self.dataset = LeRobotDataset(
            repo_id=repo_id,
            root=metadata.root,
            revision=metadata.revision,
            episodes=list(episodes) if episodes is not None else None,
            delta_timestamps=delta_timestamps,
            video_backend=video_backend,
            download_videos=True,
        )
        if len(self.dataset) < batch_size:
            raise ValueError(
                f"dataset contains {len(self.dataset)} frames, smaller than batch size {batch_size}"
            )

        stats = canonicalize_dataset_stats(self.dataset.meta.stats, self.action_key)
        self.preprocessor = JaxSmolVLAPreprocessor(
            checkpoint,
            config,
            rename_map=rename_map,
            stats=stats,
            local_files_only=local_files_only,
        )
        self._validate_features(config, self.preprocessor.rename_map)
        loader_kwargs: dict[str, Any] = {
            "batch_size": batch_size,
            "shuffle": True,
            "drop_last": True,
            "num_workers": num_workers,
            "persistent_workers": num_workers > 0,
            "generator": torch.Generator().manual_seed(seed),
            "collate_fn": partial(_collate_lerobot_samples, action_key=self.action_key),
        }
        if num_workers > 0:
            loader_kwargs["prefetch_factor"] = prefetch_factor
            # JAX owns background threads in the training process; forking that
            # process can deadlock. Spawn keeps data workers isolated from the
            # JAX runtime while they decode Parquet/images/videos.
            loader_kwargs["multiprocessing_context"] = "spawn"
        self.loader = DataLoader(self.dataset, **loader_kwargs)
        self.config = config

    def _validate_features(
        self,
        config: JaxSmolVLAConfig,
        rename_map: Mapping[str, str] | None,
    ) -> None:
        features = self.dataset.features
        state_shape = tuple(features.get("observation.state", {}).get("shape", ()))
        action_shape = tuple(features[self.action_key].get("shape", ()))
        if not state_shape or state_shape[-1] > config.max_state_dim:
            raise ValueError(
                f"dataset state shape {state_shape} is incompatible with max_state_dim={config.max_state_dim}"
            )
        if action_shape != (config.action_dim,):
            raise ValueError(
                f"dataset action shape {action_shape} does not match checkpoint action_dim={config.action_dim}"
            )
        rename_map = dict(rename_map or {})
        dataset_cameras = {
            rename_map.get(key, key)
            for key, feature in features.items()
            if feature.get("dtype") in ("image", "video")
        }
        if not dataset_cameras.intersection(config.image_keys):
            raise ValueError(
                "none of the dataset cameras match the checkpoint image features after renaming: "
                f"dataset={sorted(dataset_cameras)}, checkpoint={sorted(config.image_keys)}"
            )

    def batches(self) -> Iterator[dict[str, Array]]:
        while True:
            for raw_batch in self.loader:
                yield prepare_lerobot_batch(
                    raw_batch,
                    self.preprocessor,
                    self.config,
                    self.action_key,
                )
