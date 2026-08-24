"""Episode-disjoint 8:1:1 split over a CachedPairs cache."""

from __future__ import annotations

import dataclasses
import pathlib
from collections.abc import Sequence
from typing import Any

import numpy as np

from utils.cache import CachedPairs
from utils.cache import atomic_write_json

TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1


@dataclasses.dataclass(frozen=True)
class EpisodeSplit:
    train_episodes: tuple[int, ...]
    val_episodes: tuple[int, ...]
    test_episodes: tuple[int, ...]
    train_indices: np.ndarray
    val_indices: np.ndarray
    test_indices: np.ndarray
    seed: int

    def counts(self) -> dict[str, int]:
        return {
            "train_episodes": len(self.train_episodes),
            "val_episodes": len(self.val_episodes),
            "test_episodes": len(self.test_episodes),
            "train_samples": int(self.train_indices.shape[0]),
            "val_samples": int(self.val_indices.shape[0]),
            "test_samples": int(self.test_indices.shape[0]),
        }


def split_episodes_three_way(
    episode_indices: Sequence[int],
    *,
    train_ratio: float = TRAIN_RATIO,
    val_ratio: float = VAL_RATIO,
    seed: int = 0,
) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
    """Shuffle unique episodes and allocate train / val / test, all non-empty."""
    episodes = np.asarray(sorted({int(index) for index in episode_indices}), dtype=np.int64)
    count = int(episodes.shape[0])
    if count < 3:
        raise ValueError(
            f"At least three episodes are required for an episode-disjoint 8:1:1 split, got {count}."
        )
    if not 0.0 < train_ratio < 1.0 or not 0.0 < val_ratio < 1.0:
        raise ValueError("train_ratio and val_ratio must be in (0, 1).")
    if train_ratio + val_ratio >= 1.0:
        raise ValueError("train_ratio + val_ratio must be < 1 so the test split is non-empty.")

    rng = np.random.default_rng(seed)
    shuffled = episodes.copy()
    rng.shuffle(shuffled)

    train_count = min(count - 2, max(1, int(round(count * train_ratio))))
    val_count = min(count - train_count - 1, max(1, int(round(count * val_ratio))))
    test_count = count - train_count - val_count
    if min(train_count, val_count, test_count) <= 0:
        raise ValueError(
            f"Failed to allocate a non-empty 8:1:1 split from {count} episodes "
            f"(train={train_count}, val={val_count}, test={test_count})."
        )

    train = tuple(sorted(int(index) for index in shuffled[:train_count]))
    val = tuple(
        sorted(int(index) for index in shuffled[train_count : train_count + val_count])
    )
    test = tuple(sorted(int(index) for index in shuffled[train_count + val_count :]))
    return train, val, test


def indices_for_episodes(episode_array: np.ndarray, episodes: Sequence[int]) -> np.ndarray:
    episode_set = {int(index) for index in episodes}
    values = np.asarray(episode_array)
    return np.flatnonzero(np.isin(values, list(episode_set))).astype(np.int64)


def build_episode_split(pairs: CachedPairs, *, seed: int) -> EpisodeSplit:
    episode_array = np.asarray(pairs.arrays["episode_index"])
    train_episodes, val_episodes, test_episodes = split_episodes_three_way(
        episode_array.tolist(),
        seed=seed,
    )
    train_indices = indices_for_episodes(episode_array, train_episodes)
    val_indices = indices_for_episodes(episode_array, val_episodes)
    test_indices = indices_for_episodes(episode_array, test_episodes)
    if min(len(train_indices), len(val_indices), len(test_indices)) == 0:
        raise ValueError("Episode split left an empty sample set; check the cache episode labels.")
    return EpisodeSplit(
        train_episodes=train_episodes,
        val_episodes=val_episodes,
        test_episodes=test_episodes,
        train_indices=train_indices,
        val_indices=val_indices,
        test_indices=test_indices,
        seed=int(seed),
    )


def write_split_json(
    path: pathlib.Path,
    split: EpisodeSplit,
    *,
    cache_dir: pathlib.Path,
    records_sha256: str,
) -> None:
    payload: dict[str, Any] = {
        "seed": split.seed,
        "ratios": [TRAIN_RATIO, VAL_RATIO, TEST_RATIO],
        "train_episodes": list(split.train_episodes),
        "val_episodes": list(split.val_episodes),
        "test_episodes": list(split.test_episodes),
        "train_sample_count": int(split.train_indices.shape[0]),
        "val_sample_count": int(split.val_indices.shape[0]),
        "test_sample_count": int(split.test_indices.shape[0]),
        "cache_dir": str(cache_dir.resolve()),
        "records_sha256": records_sha256,
    }
    atomic_write_json(path, payload)
