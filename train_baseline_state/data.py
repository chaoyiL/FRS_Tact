"""Read-only state/structured-noise training pairs without sensory streams."""

from __future__ import annotations
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from .action_cache import SPLIT_IDS


class BaselineCacheDataset(Dataset):
    def __init__(self, action_cache, split):
        if split not in SPLIT_IDS:
            raise ValueError("unknown split")
        if action_cache.manifest.get("origin") != "structured_noise":
            raise ValueError("state-only training requires reverse-integrated structured_noise cache")
        self.action_cache = action_cache
        self.rows = np.asarray(action_cache.indices(split), dtype=np.int64)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, position):
        row = int(self.rows[position])
        cache = self.action_cache
        return {
            name: torch.from_numpy(np.array(value[row], copy=True))
            for name, value in {
                "coarse": cache.coarse_actions,
                "structured_noise": cache.structured_noise,
                "target": cache.expert_actions,
                "state": cache.states,
                "valid": cache.valid_masks,
            }.items()
        }


def make_loader(dataset, *, batch_size, shuffle, seed, workers=0, pin_memory=False):
    if batch_size <= 0 or workers < 0:
        raise ValueError("invalid loader parameters")
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=torch.Generator().manual_seed(seed),
        num_workers=workers,
        pin_memory=pin_memory,
    )
