"""Read-only, aligned PyTorch datasets over frozen action and tactile caches."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from train_baseline_pi05.action_cache import SPLIT_IDS
from train_baseline_pi05.config import TACTILE_KEYS


class BaselineCacheDataset(Dataset[dict[str, torch.Tensor]]):
    """Pair each cached action chunk with tactile tokens at its absolute frame."""

    def __init__(self, action_cache: Any, tactile_cache: Any, split: str) -> None:
        if split not in SPLIT_IDS:
            raise ValueError(f"unknown split: {split}")
        self.action_cache, self.tactile_cache = action_cache, tactile_cache
        action_manifest = getattr(action_cache, "manifest", None)
        tactile_manifest = getattr(tactile_cache, "metadata", None)
        if not isinstance(action_manifest, Mapping) or not isinstance(tactile_manifest, Mapping):
            raise ValueError("action and tactile cache manifests are required")
        if tuple(tactile_manifest.get("tactile_keys", ())) != TACTILE_KEYS:
            raise ValueError("tactile cache key order does not match the decoder contract")
        action_identity = action_manifest.get("dataset_identity")
        tactile_identity = tactile_manifest.get("dataset_identity")
        if not isinstance(action_identity, Mapping) or not isinstance(tactile_identity, Mapping):
            raise ValueError("action and tactile cache dataset provenance is invalid")
        action_root = action_identity.get("root")
        tactile_root = tactile_identity.get("root")
        if not isinstance(action_root, (str, Path)) or not isinstance(tactile_root, (str, Path)):
            raise ValueError("action and tactile cache dataset provenance is invalid")
        identities_differ = any(
            action_identity.get(key) != tactile_identity.get(key)
            for key in ("repo_id", "revision")
        ) or Path(action_root).expanduser().resolve() != Path(tactile_root).expanduser().resolve()
        if identities_differ:
            raise ValueError("action and tactile cache dataset provenance does not match")
        indices = np.asarray(action_cache.dataset_indices, dtype=np.int64)
        total_frames = int(tactile_manifest.get("total_frames", -1))
        if total_frames < 0 or np.any(indices < 0) or np.any(indices >= total_frames):
            raise IndexError("action cache tactile alignment index is out of range")
        self.rows = np.asarray(action_cache.indices(split), dtype=np.int64)

    def __len__(self) -> int:
        return int(self.rows.size)

    def __getitem__(self, position: int) -> dict[str, torch.Tensor]:
        row = int(self.rows[position])
        frame = int(self.action_cache.dataset_indices[row])
        tactile = self.tactile_cache.get_many([frame])[0]
        return {
            "coarse": torch.from_numpy(np.array(self.action_cache.coarse_actions[row], copy=True)),
            "target": torch.from_numpy(np.array(self.action_cache.expert_actions[row], copy=True)),
            "valid": torch.from_numpy(np.array(self.action_cache.valid_masks[row], copy=True)),
            "tactile": torch.from_numpy(np.array(tactile, copy=True)),
            "dataset_index": torch.tensor(frame, dtype=torch.int64),
            "episode_index": torch.tensor(int(self.action_cache.episode_indices[row]), dtype=torch.int64),
        }


def make_loader(
    dataset: Dataset[dict[str, torch.Tensor]], *, batch_size: int, shuffle: bool, seed: int,
    workers: int = 0, pin_memory: bool = False,
) -> DataLoader[dict[str, torch.Tensor]]:
    """Make a deterministic loader; callers use ``shuffle=False`` for validation/test."""
    if batch_size <= 0 or workers < 0:
        raise ValueError("batch_size must be positive and workers non-negative")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, generator=generator,
        num_workers=workers, pin_memory=pin_memory,
    )
