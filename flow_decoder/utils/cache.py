from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import pathlib
from collections.abc import Iterator, Sequence
from typing import Any, Literal

import numpy as np

CACHE_VERSION = 1
MANIFEST_NAME = "manifest.json"
X_BASE_NAME = "x_base.npy"
TARGET_NAME = "predicted_actions.npy"
DATASET_INDEX_NAME = "dataset_indices.npy"
EPISODE_INDEX_NAME = "episode_indices.npy"
SPLIT_NAME = "split.npy"
INVERSION_MSE_NAME = "inversion_mse.npy"
ARRAY_FILENAMES = {
    "x_base": X_BASE_NAME,
    "target": TARGET_NAME,
    "dataset_index": DATASET_INDEX_NAME,
    "episode_index": EPISODE_INDEX_NAME,
    "split": SPLIT_NAME,
    "inversion_mse": INVERSION_MSE_NAME,
}


@dataclasses.dataclass(frozen=True)
class SampleRecord:
    dataset_index: int
    episode_index: int
    split: Literal["train", "val"]


def atomic_write_json(path: pathlib.Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as file:
        json.dump(value, file, indent=2, sort_keys=True)
        file.write("\n")
        file.flush()
        os.fsync(file.fileno())
    temporary.replace(path)


def records_digest(records: Sequence[SampleRecord]) -> str:
    digest = hashlib.sha256()
    for record in records:
        digest.update(f"{record.dataset_index}:{record.episode_index}:{record.split}\n".encode())
    return digest.hexdigest()


def split_episodes(
    episode_indices: Sequence[int], *, val_fraction: float = 0.2, seed: int = 0
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    episodes = np.asarray(sorted(set(int(index) for index in episode_indices)), dtype=np.int64)
    if len(episodes) < 2:
        raise ValueError("At least two episodes are required for an episode-disjoint train/validation split.")
    if not 0.0 < val_fraction < 1.0:
        raise ValueError(f"val_fraction must be in (0, 1), got {val_fraction}.")

    rng = np.random.default_rng(seed)
    shuffled = episodes.copy()
    rng.shuffle(shuffled)
    val_count = min(len(shuffled) - 1, max(1, int(round(len(shuffled) * val_fraction))))
    val = tuple(sorted(int(index) for index in shuffled[:val_count]))
    train = tuple(sorted(int(index) for index in shuffled[val_count:]))
    return train, val


def limit_records(
    records: Sequence[SampleRecord], *, max_samples: int | None, seed: int
) -> list[SampleRecord]:
    records = list(records)
    if max_samples is None or max_samples >= len(records):
        return records
    if max_samples <= 1:
        raise ValueError("max_samples must be at least 2 so both train and validation remain represented.")

    rng = np.random.default_rng(seed)
    train = [record for record in records if record.split == "train"]
    val = [record for record in records if record.split == "val"]
    if not train or not val:
        raise ValueError("Both train and validation records are required before applying max_samples.")

    val_count = min(len(val), max(1, int(round(max_samples * len(val) / len(records)))))
    train_count = min(len(train), max_samples - val_count)
    if train_count == 0:
        train_count = 1
        val_count = max_samples - 1
    # Fill unused quota when one split is smaller than its proportional allocation.
    remaining = max_samples - train_count - val_count
    train_count += min(remaining, len(train) - train_count)
    remaining = max_samples - train_count - val_count
    val_count += min(remaining, len(val) - val_count)

    selected_train = rng.choice(len(train), size=train_count, replace=False)
    selected_val = rng.choice(len(val), size=val_count, replace=False)
    selected = [train[int(i)] for i in selected_train] + [val[int(i)] for i in selected_val]
    return sorted(selected, key=lambda record: record.dataset_index)


def create_cache_arrays(
    cache_dir: pathlib.Path,
    records: Sequence[SampleRecord],
    *,
    action_horizon: int,
    action_dim: int,
) -> dict[str, np.memmap]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    count = len(records)
    shape = (count, action_horizon, action_dim)
    arrays: dict[str, np.memmap] = {
        "x_base": np.lib.format.open_memmap(cache_dir / X_BASE_NAME, mode="w+", dtype=np.float32, shape=shape),
        "target": np.lib.format.open_memmap(cache_dir / TARGET_NAME, mode="w+", dtype=np.float32, shape=shape),
        "dataset_index": np.lib.format.open_memmap(
            cache_dir / DATASET_INDEX_NAME, mode="w+", dtype=np.int64, shape=(count,)
        ),
        "episode_index": np.lib.format.open_memmap(
            cache_dir / EPISODE_INDEX_NAME, mode="w+", dtype=np.int64, shape=(count,)
        ),
        "split": np.lib.format.open_memmap(cache_dir / SPLIT_NAME, mode="w+", dtype=np.uint8, shape=(count,)),
        "inversion_mse": np.lib.format.open_memmap(
            cache_dir / INVERSION_MSE_NAME, mode="w+", dtype=np.float32, shape=(count,)
        ),
    }
    arrays["dataset_index"][:] = [record.dataset_index for record in records]
    arrays["episode_index"][:] = [record.episode_index for record in records]
    arrays["split"][:] = [0 if record.split == "train" else 1 for record in records]
    flush_arrays(arrays)
    return arrays


def open_cache_arrays(cache_dir: pathlib.Path, *, mode: str = "r") -> dict[str, np.ndarray]:
    return {
        "x_base": np.load(cache_dir / X_BASE_NAME, mmap_mode=mode),
        "target": np.load(cache_dir / TARGET_NAME, mmap_mode=mode),
        "dataset_index": np.load(cache_dir / DATASET_INDEX_NAME, mmap_mode=mode),
        "episode_index": np.load(cache_dir / EPISODE_INDEX_NAME, mmap_mode=mode),
        "split": np.load(cache_dir / SPLIT_NAME, mmap_mode=mode),
        "inversion_mse": np.load(cache_dir / INVERSION_MSE_NAME, mmap_mode=mode),
    }


def flush_arrays(arrays: dict[str, np.ndarray]) -> None:
    for array in arrays.values():
        flush = getattr(array, "flush", None)
        if flush is not None:
            flush()


def records_from_arrays(arrays: dict[str, np.ndarray], *, count: int) -> list[SampleRecord]:
    records: list[SampleRecord] = []
    for index in range(count):
        split = "val" if int(arrays["split"][index]) == 1 else "train"
        records.append(
            SampleRecord(
                int(arrays["dataset_index"][index]),
                int(arrays["episode_index"][index]),
                split,
            )
        )
    return records


def _atomic_save_array(path: pathlib.Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as file:
        np.save(file, array, allow_pickle=False)
    temporary.replace(path)


def truncate_cache_arrays(cache_dir: pathlib.Path, *, count: int) -> None:
    arrays = open_cache_arrays(cache_dir)
    if arrays["x_base"].shape[0] < count:
        raise ValueError(
            f"Cannot truncate cache to {count} samples; arrays only contain {arrays['x_base'].shape[0]} rows."
        )
    if arrays["x_base"].shape[0] == count:
        return
    for key, filename in ARRAY_FILENAMES.items():
        _atomic_save_array(cache_dir / filename, np.asarray(arrays[key][:count]))


def finalize_partial_cache(
    cache_dir: pathlib.Path,
    *,
    resplit: bool = True,
) -> dict[str, Any]:
    manifest = load_manifest(cache_dir, require_complete=False)
    completed = int(manifest.get("completed_samples", 0))
    sample_count = int(manifest.get("sample_count", 0))
    if completed <= 0:
        raise ValueError("completed_samples must be positive before finalizing a partial cache.")
    if manifest.get("status") == "complete" and completed == sample_count:
        return manifest

    truncate_cache_arrays(cache_dir, count=completed)
    arrays = open_cache_arrays(cache_dir, mode="r+")

    configuration = manifest.get("configuration", {})
    if resplit:
        episodes = sorted({int(index) for index in arrays["episode_index"]})
        train_episodes, val_episodes = split_episodes(
            episodes,
            val_fraction=float(configuration.get("val_fraction", 0.2)),
            seed=int(configuration.get("split_seed", 0)),
        )
        val_set = set(val_episodes)
        arrays["split"][:] = np.asarray(
            [1 if int(episode_index) in val_set else 0 for episode_index in arrays["episode_index"]],
            dtype=np.uint8,
        )
        flush_arrays(arrays)
    else:
        train_episodes = tuple(int(index) for index in manifest.get("train_episodes", ()))
        val_episodes = tuple(int(index) for index in manifest.get("val_episodes", ()))

    records = records_from_arrays(arrays, count=completed)
    if not any(record.split == "train" for record in records):
        raise ValueError("Finalized cache has no training samples.")
    if not any(record.split == "val" for record in records):
        raise ValueError("Finalized cache has no validation samples.")

    manifest.update(
        {
            "status": "complete",
            "sample_count": completed,
            "completed_samples": completed,
            "train_sample_count": sum(record.split == "train" for record in records),
            "val_sample_count": sum(record.split == "val" for record in records),
            "train_episodes": list(train_episodes),
            "val_episodes": list(val_episodes),
            "records_sha256": records_digest(records),
            "mean_source_inversion_mse": float(np.mean(np.asarray(arrays["inversion_mse"]))),
        }
    )
    atomic_write_json(cache_dir / MANIFEST_NAME, manifest)
    return manifest


def load_manifest(cache_dir: pathlib.Path, *, require_complete: bool = True) -> dict[str, Any]:
    manifest_path = cache_dir / MANIFEST_NAME
    if not manifest_path.exists():
        raise FileNotFoundError(f"Cache manifest not found: {manifest_path}")
    with manifest_path.open(encoding="utf-8") as file:
        manifest = json.load(file)
    if manifest.get("version") != CACHE_VERSION:
        raise ValueError(f"Unsupported cache version {manifest.get('version')}; expected {CACHE_VERSION}.")
    if require_complete and manifest.get("status") != "complete":
        raise ValueError(
            f"Cache is not complete ({manifest.get('completed_samples', 0)}/{manifest.get('sample_count')}). "
            "Resume flow_decoder.prepare first."
        )
    return manifest


class CachedPairs:
    def __init__(self, cache_dir: str | pathlib.Path):
        self.cache_dir = pathlib.Path(cache_dir)
        self.manifest = load_manifest(self.cache_dir)
        self.arrays = open_cache_arrays(self.cache_dir)

    def indices(self, split: Literal["train", "val"]) -> np.ndarray:
        split_value = 0 if split == "train" else 1
        return np.flatnonzero(np.asarray(self.arrays["split"]) == split_value)

    def batches(
        self,
        split: Literal["train", "val"],
        *,
        batch_size: int,
        shuffle: bool,
        seed: int,
    ) -> Iterator[tuple[np.ndarray, np.ndarray, np.ndarray]]:
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}.")
        indices = self.indices(split)
        if shuffle:
            indices = np.random.default_rng(seed).permutation(indices)
        for start in range(0, len(indices), batch_size):
            batch_indices = indices[start : start + batch_size]
            yield (
                batch_indices,
                np.asarray(self.arrays["x_base"][batch_indices], dtype=np.float32),
                np.asarray(self.arrays["target"][batch_indices], dtype=np.float32),
            )
