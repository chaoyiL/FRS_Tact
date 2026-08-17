from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import pathlib
from collections.abc import Iterator, Sequence
from typing import Any, Literal

import numpy as np

CACHE_VERSION = 3
MANIFEST_NAME = "manifest.json"
X_BASE_NAME = "x_base.npy"
TARGET_NAME = "predicted_actions.npy"
GT_ACTION_NAME = "gt_actions.npy"
DATASET_INDEX_NAME = "dataset_indices.npy"
EPISODE_INDEX_NAME = "episode_indices.npy"
SPLIT_NAME = "split.npy"
INVERSION_MSE_NAME = "inversion_mse.npy"
STATE_NAME = "states.npy"
ARRAY_FILENAMES = {
    "x_base": X_BASE_NAME,
    "target": TARGET_NAME,
    "gt_action": GT_ACTION_NAME,
    "dataset_index": DATASET_INDEX_NAME,
    "episode_index": EPISODE_INDEX_NAME,
    "split": SPLIT_NAME,
    "inversion_mse": INVERSION_MSE_NAME,
    "state": STATE_NAME,
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


def trim_episode_tail(
    dataset_indices: Sequence[int],
    *,
    drop_tail_action_chunks: int,
    action_horizon: int,
) -> tuple[int, ...]:
    """Drop the last ``drop_tail_action_chunks * action_horizon`` frames from an episode."""
    if drop_tail_action_chunks < 0:
        raise ValueError(f"drop_tail_action_chunks must be >= 0, got {drop_tail_action_chunks}.")
    if action_horizon <= 0:
        raise ValueError(f"action_horizon must be positive, got {action_horizon}.")
    indices = tuple(int(index) for index in dataset_indices)
    drop_frames = int(drop_tail_action_chunks) * int(action_horizon)
    if drop_frames <= 0:
        return indices
    if len(indices) <= drop_frames:
        return ()
    return indices[:-drop_frames]


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


def _as_scalar(value: Any) -> Any:
    array = np.asarray(value)
    if array.shape == ():
        return array.item()
    if array.size == 1:
        return array.reshape(()).item()
    return array


def _episode_bounds(metadata: Any, episode_index: int) -> tuple[int, int]:
    if episode_index < 0 or episode_index >= metadata.total_episodes:
        raise ValueError(
            f"Episode {episode_index} is out of range for this dataset. "
            f"Available episode indices are 0..{metadata.total_episodes - 1}."
        )
    episode = metadata.episodes[episode_index]
    start = int(_as_scalar(episode["dataset_from_index"]))
    end = int(_as_scalar(episode["dataset_to_index"]))
    if end <= start:
        raise ValueError(f"Episode {episode_index} has an empty frame range [{start}, {end}).")
    return start, end


def _indices_for_episode(metadata: Any, episode_index: int) -> tuple[int, ...]:
    start, end = _episode_bounds(metadata, episode_index)
    return tuple(range(start, end))


def build_records(
    metadata: Any,
    *,
    val_fraction: float,
    split_seed: int,
    frame_stride: int,
    max_episodes: int | None,
    max_samples: int | None,
    action_horizon: int,
    drop_tail_action_chunks: int = 1,
) -> tuple[list[SampleRecord], tuple[int, ...], tuple[int, ...]]:
    """Select an episode-disjoint train/val sample set from any LeRobot-metadata-like object.

    ``metadata`` only needs ``total_episodes`` and ``episodes[i]["dataset_from_index"/"dataset_to_index"]``,
    so this works with this repo's ``LeRobotDatasetMetadata`` as well as an independently installed
    upstream ``lerobot`` package's metadata class (e.g. from an openpi environment), keeping every
    action-cache producer (regardless of base model / Python environment) on identical record selection.
    """
    if frame_stride <= 0:
        raise ValueError(f"frame_stride must be positive, got {frame_stride}.")
    episode_count = int(metadata.total_episodes)
    if episode_count < 2:
        raise ValueError("At least two episodes are required for an episode-disjoint train/validation split.")
    episodes = list(range(episode_count))
    if max_episodes is not None:
        if max_episodes < 2:
            raise ValueError("max_episodes must be at least 2 for an episode-disjoint split.")
        episodes = episodes[:max_episodes]
    train_episodes, val_episodes = split_episodes(episodes, val_fraction=val_fraction, seed=split_seed)
    val_set = set(val_episodes)

    records: list[SampleRecord] = []
    for episode_index in episodes:
        split = "val" if episode_index in val_set else "train"
        trimmed = trim_episode_tail(
            _indices_for_episode(metadata, episode_index),
            drop_tail_action_chunks=drop_tail_action_chunks,
            action_horizon=action_horizon,
        )
        dataset_indices = trimmed[::frame_stride]
        records.extend(SampleRecord(int(index), episode_index, split) for index in dataset_indices)
    records = limit_records(records, max_samples=max_samples, seed=split_seed)
    if not records:
        raise ValueError(
            "Dataset selection produced no samples. "
            "Try lowering --drop-tail-action-chunks or using longer episodes."
        )
    present = {record.episode_index for record in records}
    train_episodes = tuple(episode for episode in train_episodes if episode in present)
    val_episodes = tuple(episode for episode in val_episodes if episode in present)
    if not train_episodes or not val_episodes:
        raise ValueError(
            "After dropping episode tails, train or val split is empty. "
            "Try lowering --drop-tail-action-chunks."
        )
    return records, train_episodes, val_episodes


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
    state_dim: int = 1,
) -> dict[str, np.memmap]:
    if state_dim <= 0:
        raise ValueError(f"state_dim must be positive, got {state_dim}.")
    cache_dir.mkdir(parents=True, exist_ok=True)
    count = len(records)
    shape = (count, action_horizon, action_dim)
    arrays: dict[str, np.memmap] = {
        "x_base": np.lib.format.open_memmap(cache_dir / X_BASE_NAME, mode="w+", dtype=np.float32, shape=shape),
        "target": np.lib.format.open_memmap(cache_dir / TARGET_NAME, mode="w+", dtype=np.float32, shape=shape),
        "gt_action": np.lib.format.open_memmap(
            cache_dir / GT_ACTION_NAME, mode="w+", dtype=np.float32, shape=shape
        ),
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
        "state": np.lib.format.open_memmap(
            cache_dir / STATE_NAME, mode="w+", dtype=np.float32, shape=(count, state_dim)
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
        "gt_action": np.load(cache_dir / GT_ACTION_NAME, mmap_mode=mode),
        "dataset_index": np.load(cache_dir / DATASET_INDEX_NAME, mmap_mode=mode),
        "episode_index": np.load(cache_dir / EPISODE_INDEX_NAME, mmap_mode=mode),
        "split": np.load(cache_dir / SPLIT_NAME, mmap_mode=mode),
        "inversion_mse": np.load(cache_dir / INVERSION_MSE_NAME, mmap_mode=mode),
        "state": np.load(cache_dir / STATE_NAME, mmap_mode=mode),
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
    # Copy and close every mmap before replacing files; Windows forbids replacing an open mmap.
    truncated = {key: np.array(arrays[key][:count], copy=True) for key in ARRAY_FILENAMES}
    for array in arrays.values():
        mmap = getattr(array, "_mmap", None)
        if mmap is not None:
            mmap.close()
    arrays.clear()
    for key, filename in ARRAY_FILENAMES.items():
        _atomic_save_array(cache_dir / filename, truncated[key])


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
            "action_horizon": int(arrays["x_base"].shape[1]),
            "action_dim": int(arrays["x_base"].shape[2]),
            "state_dim": int(arrays["state"].shape[1]),
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
            "Resume prepare_pi05.py first."
        )
    return manifest


def sample_pred_gt_mse(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    """Per-sample MSE between predicted and GT action chunks. Shape [N]."""
    if pred.shape != gt.shape:
        raise ValueError(f"pred/gt shape mismatch: {pred.shape} vs {gt.shape}")
    if pred.ndim != 3:
        raise ValueError(f"Expected actions [N, H, A], got {pred.shape}")
    diff = pred.astype(np.float64) - gt.astype(np.float64)
    return np.mean(np.square(diff), axis=(1, 2)).astype(np.float64)


def filter_cache_by_mse(
    cache_dir: str | pathlib.Path,
    output_dir: str | pathlib.Path,
    *,
    max_mse: float = 1.0,
) -> dict[str, Any]:
    """Write a new cache keeping only samples with MSE(pred, gt) <= max_mse."""
    if max_mse < 0:
        raise ValueError(f"max_mse must be >= 0, got {max_mse}.")

    pairs = CachedPairs(cache_dir)
    pred = np.asarray(pairs.arrays["target"], dtype=np.float32)
    gt = np.asarray(pairs.arrays["gt_action"], dtype=np.float32)
    mse = sample_pred_gt_mse(pred, gt)
    keep = np.flatnonzero(mse <= float(max_mse)).astype(np.int64)
    dropped = int(mse.shape[0] - keep.shape[0])

    if keep.size == 0:
        raise ValueError(
            f"No samples remain after filtering with max_mse={max_mse}. "
            f"Source MSE: min={float(mse.min()):.6f} median={float(np.median(mse)):.6f} "
            f"max={float(mse.max()):.6f}."
        )

    kept_mse = mse[keep]
    return write_cache_subset(
        cache_dir,
        output_dir,
        keep,
        extra_manifest={
            "filter": {
                "type": "pred_gt_mse",
                "max_mse": float(max_mse),
                "source_sample_count": int(mse.shape[0]),
                "kept_sample_count": int(keep.shape[0]),
                "dropped_sample_count": dropped,
                "kept_mse_mean": float(kept_mse.mean()),
                "kept_mse_median": float(np.median(kept_mse)),
                "kept_mse_max": float(kept_mse.max()),
                "source_mse_mean": float(mse.mean()),
                "source_mse_median": float(np.median(mse)),
                "source_mse_max": float(mse.max()),
            }
        },
    )


def write_cache_subset(
    source_dir: str | pathlib.Path,
    output_dir: str | pathlib.Path,
    keep_indices: Sequence[int] | np.ndarray,
    *,
    extra_manifest: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Copy selected rows from a complete cache into a new complete cache directory."""
    source_dir = pathlib.Path(source_dir)
    output_dir = pathlib.Path(output_dir)
    keep = np.asarray(keep_indices, dtype=np.int64)
    if keep.ndim != 1:
        raise ValueError(f"keep_indices must be 1-D, got shape {keep.shape}.")
    if keep.size == 0:
        raise ValueError("keep_indices is empty; refusing to write an empty cache.")
    if np.any(keep[1:] < keep[:-1]):
        keep = np.sort(keep)

    source_manifest = load_manifest(source_dir, require_complete=True)
    source = open_cache_arrays(source_dir)
    sample_count = int(source_manifest["sample_count"])
    if int(keep.min()) < 0 or int(keep.max()) >= sample_count:
        raise ValueError(
            f"keep_indices out of range for cache with {sample_count} samples "
            f"(min={int(keep.min())}, max={int(keep.max())})."
        )

    records = [
        SampleRecord(
            int(source["dataset_index"][index]),
            int(source["episode_index"][index]),
            "val" if int(source["split"][index]) == 1 else "train",
        )
        for index in keep.tolist()
    ]
    if not any(record.split == "train" for record in records):
        raise ValueError("Filtered cache would have no training samples.")
    if not any(record.split == "val" for record in records):
        raise ValueError("Filtered cache would have no validation samples.")

    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"Output cache directory is not empty: {output_dir}. "
            "Choose a new directory to avoid mixing caches."
        )

    action_horizon = int(source_manifest["action_horizon"])
    action_dim = int(source_manifest["action_dim"])
    state_dim = int(source_manifest["state_dim"])
    arrays = create_cache_arrays(
        output_dir,
        records,
        action_horizon=action_horizon,
        action_dim=action_dim,
        state_dim=state_dim,
    )
    for key in ("x_base", "target", "gt_action", "inversion_mse", "state"):
        arrays[key][:] = np.asarray(source[key][keep], dtype=arrays[key].dtype)
    flush_arrays(arrays)

    train_episodes = sorted({record.episode_index for record in records if record.split == "train"})
    val_episodes = sorted({record.episode_index for record in records if record.split == "val"})
    manifest = {
        "version": CACHE_VERSION,
        "status": "complete",
        "completed_samples": len(records),
        "sample_count": len(records),
        "train_sample_count": sum(record.split == "train" for record in records),
        "val_sample_count": sum(record.split == "val" for record in records),
        "train_episodes": train_episodes,
        "val_episodes": val_episodes,
        "action_horizon": action_horizon,
        "action_dim": action_dim,
        "state_dim": state_dim,
        "configuration": dict(source_manifest.get("configuration", {})),
        "records_sha256": records_digest(records),
        "mean_source_inversion_mse": float(np.mean(np.asarray(arrays["inversion_mse"]))),
        "source_cache_dir": str(source_dir.resolve()),
    }
    if extra_manifest:
        manifest.update(extra_manifest)
    atomic_write_json(output_dir / MANIFEST_NAME, manifest)
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
    ) -> Iterator[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
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
                np.asarray(self.arrays["gt_action"][batch_indices], dtype=np.float32),
                np.asarray(self.arrays["state"][batch_indices], dtype=np.float32),
            )


class MultiCachedPairs:
    """Read several independent action caches as one source-aware sample set.

    Global cache indices are only an in-memory training convention.  Each
    underlying cache keeps its original dataset/episode indices, so datasets
    whose local frame numbers overlap cannot corrupt each other.
    """

    def __init__(
        self,
        cache_dirs: Sequence[str | pathlib.Path],
        *,
        source_names: Sequence[str] | None = None,
    ):
        if not cache_dirs:
            raise ValueError("MultiCachedPairs requires at least one cache directory.")
        self.sources = tuple(CachedPairs(directory) for directory in cache_dirs)
        if source_names is None:
            source_names = tuple(source.cache_dir.name for source in self.sources)
        if len(source_names) != len(self.sources):
            raise ValueError(
                f"source_names/cache_dirs length mismatch: {len(source_names)} != {len(self.sources)}"
            )
        self.source_names = tuple(str(name) for name in source_names)
        if len(set(self.source_names)) != len(self.source_names):
            raise ValueError(f"source_names must be unique, got {self.source_names}")

        action_horizon = int(self.sources[0].manifest["action_horizon"])
        action_dim = int(self.sources[0].manifest["action_dim"])
        state_dim = int(self.sources[0].manifest["state_dim"])
        for name, source in zip(self.source_names, self.sources, strict=True):
            shape = (
                int(source.manifest["action_horizon"]),
                int(source.manifest["action_dim"]),
            )
            if shape != (action_horizon, action_dim):
                raise ValueError(
                    f"action shape mismatch for {name}: {shape} != {(action_horizon, action_dim)}"
                )
            if int(source.manifest["state_dim"]) != state_dim:
                raise ValueError(
                    f"state dimension mismatch for {name}: "
                    f"{source.manifest['state_dim']} != {state_dim}"
                )

        counts = np.asarray(
            [int(source.manifest["sample_count"]) for source in self.sources],
            dtype=np.int64,
        )
        self._starts = np.concatenate((np.zeros((1,), dtype=np.int64), np.cumsum(counts)[:-1]))
        self._stops = np.cumsum(counts)
        digest = hashlib.sha256()
        for name, source in zip(self.source_names, self.sources, strict=True):
            digest.update(f"{name}:{source.manifest['records_sha256']}\n".encode())
        self.manifest: dict[str, Any] = {
            "version": CACHE_VERSION,
            "status": "complete",
            "sample_count": int(np.sum(counts)),
            "train_sample_count": int(
                sum(int(source.manifest["train_sample_count"]) for source in self.sources)
            ),
            "val_sample_count": int(
                sum(int(source.manifest["val_sample_count"]) for source in self.sources)
            ),
            "action_horizon": action_horizon,
            "action_dim": action_dim,
            "state_dim": state_dim,
            "records_sha256": digest.hexdigest(),
            "configuration": {
                "sources": [
                    {
                        "name": name,
                        "cache_dir": str(source.cache_dir.resolve()),
                        "configuration": source.manifest.get("configuration", {}),
                    }
                    for name, source in zip(self.source_names, self.sources, strict=True)
                ]
            },
        }

    def __len__(self) -> int:
        return int(self.manifest["sample_count"])

    def source_and_local_indices(self, indices: Sequence[int] | np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        global_indices = np.asarray(indices, dtype=np.int64)
        if global_indices.ndim != 1:
            raise ValueError(f"indices must be one-dimensional, got {global_indices.shape}")
        if np.any(global_indices < 0) or np.any(global_indices >= len(self)):
            raise IndexError("global cache index out of range")
        source_indices = np.searchsorted(self._stops, global_indices, side="right")
        local_indices = global_indices - self._starts[source_indices]
        return source_indices.astype(np.int32), local_indices.astype(np.int64)

    def metadata_values(self, indices: Sequence[int] | np.ndarray, key: str) -> np.ndarray:
        global_indices = np.asarray(indices, dtype=np.int64)
        source_indices, local_indices = self.source_and_local_indices(global_indices)
        output = np.empty(global_indices.shape, dtype=np.int64)
        for source_index in np.unique(source_indices):
            positions = np.flatnonzero(source_indices == source_index)
            source = self.sources[int(source_index)]
            output[positions] = np.asarray(
                source.arrays[key][local_indices[positions]], dtype=np.int64
            )
        return output

    def indices(self, split: Literal["train", "val"]) -> np.ndarray:
        parts = [
            source.indices(split) + self._starts[source_index]
            for source_index, source in enumerate(self.sources)
        ]
        return np.concatenate(parts).astype(np.int64, copy=False)

    def batches(
        self,
        split: Literal["train", "val"],
        *,
        batch_size: int,
        shuffle: bool,
        seed: int,
    ) -> Iterator[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}.")
        indices = self.indices(split)
        if shuffle:
            indices = np.random.default_rng(seed).permutation(indices)
        shape = (
            batch_size,
            int(self.manifest["action_horizon"]),
            int(self.manifest["action_dim"]),
        )
        for start in range(0, len(indices), batch_size):
            batch_indices = indices[start : start + batch_size]
            batch_n = len(batch_indices)
            x_base = np.empty((batch_n,) + shape[1:], dtype=np.float32)
            predicted = np.empty_like(x_base)
            gt_action = np.empty_like(x_base)
            state = np.empty((batch_n, int(self.manifest["state_dim"])), dtype=np.float32)
            source_indices, local_indices = self.source_and_local_indices(batch_indices)
            for source_index in np.unique(source_indices):
                positions = np.flatnonzero(source_indices == source_index)
                local = local_indices[positions]
                arrays = self.sources[int(source_index)].arrays
                x_base[positions] = np.asarray(arrays["x_base"][local], dtype=np.float32)
                predicted[positions] = np.asarray(arrays["target"][local], dtype=np.float32)
                gt_action[positions] = np.asarray(arrays["gt_action"][local], dtype=np.float32)
                state[positions] = np.asarray(arrays["state"][local], dtype=np.float32)
            yield batch_indices, x_base, predicted, gt_action, state
