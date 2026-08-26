"""Minimal direct-Pi0.5 action-cache storage without FRS latent arrays."""

from __future__ import annotations

import dataclasses
import fcntl
import hashlib
import json
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Literal

import numpy as np


CACHE_VERSION = 1
MANIFEST_NAME = "manifest.json"
WRITER_LOCK_NAME = ".writer.lock"
ARRAY_SPECS = {
    "coarse_actions": ("coarse_actions.npy", np.dtype(np.float32), 3),
    "expert_actions": ("expert_actions.npy", np.dtype(np.float32), 3),
    "valid_masks": ("valid_masks.npy", np.dtype(np.bool_), 2),
    "dataset_indices": ("dataset_indices.npy", np.dtype(np.int64), 1),
    "episode_indices": ("episode_indices.npy", np.dtype(np.int64), 1),
    "split_ids": ("split_ids.npy", np.dtype(np.uint8), 1),
}
SPLIT_IDS = {"train": 0, "validation": 1, "test": 2}


@dataclasses.dataclass(frozen=True)
class SampleRecord:
    dataset_index: int
    episode_index: int
    frame_index: int
    split_id: int


def _atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _as_int(value: Any) -> int:
    array = np.asarray(value)
    if array.size != 1:
        raise ValueError(f"expected scalar metadata value, got shape {array.shape}")
    return int(array.reshape(()).item())


def _episode_bounds(metadata: Any, episode_index: int) -> tuple[int, int]:
    episode = metadata.episodes[episode_index]
    start = _as_int(episode["dataset_from_index"])
    end = _as_int(episode["dataset_to_index"])
    if end <= start:
        raise ValueError(f"episode {episode_index} has empty frame range [{start}, {end})")
    return start, end


def _split_episode_ids(episode_count: int, fractions: Sequence[float], seed: int) -> np.ndarray:
    if len(fractions) != 3:
        raise ValueError("fractions must contain train, validation, and test values")
    values = np.asarray(tuple(float(value) for value in fractions), dtype=np.float64)
    if any(value < 0 for value in values) or not np.isclose(sum(values), 1.0, rtol=0.0, atol=1e-9):
        raise ValueError("fractions must be nonnegative and sum to one")
    nonzero_splits = np.flatnonzero(values > 0)
    if episode_count < len(nonzero_splits):
        raise ValueError("not enough episodes to represent every nonzero split")

    counts = np.floor(values * episode_count).astype(np.int64)
    counts[nonzero_splits] = np.maximum(counts[nonzero_splits], 1)
    while int(counts.sum()) > episode_count:
        for index in nonzero_splits[np.argsort(counts[nonzero_splits])[::-1]]:
            if counts[index] > 1:
                counts[index] -= 1
                break
    remainders = values * episode_count - np.floor(values * episode_count)
    while int(counts.sum()) < episode_count:
        for index in nonzero_splits[np.argsort(remainders[nonzero_splits])[::-1]]:
            counts[index] += 1
            if int(counts.sum()) == episode_count:
                break

    shuffled = np.arange(episode_count, dtype=np.int64)
    np.random.default_rng(seed).shuffle(shuffled)
    split_ids = np.empty(episode_count, dtype=np.uint8)
    start = 0
    for split_id, count in enumerate(counts.tolist()):
        split_ids[shuffled[start : start + count]] = split_id
        start += count
    return split_ids


def build_records(
    metadata: Any,
    *,
    split_seed: int,
    fractions: Sequence[float] = (0.8, 0.1, 0.1),
    frame_stride: int,
    max_episodes: int | None = None,
    max_samples: int | None = None,
) -> tuple[SampleRecord, ...]:
    """Build deterministic, episode-disjoint current-frame records.

    Every selected frame remains a record, including episode tails.  Producers
    construct the matching action window and mark padded tail steps invalid.
    """

    if frame_stride <= 0:
        raise ValueError("frame_stride must be positive")
    episode_count = int(metadata.total_episodes)
    if max_episodes is not None:
        if max_episodes <= 0:
            raise ValueError("max_episodes must be positive")
        episode_count = min(episode_count, int(max_episodes))
    split_ids = _split_episode_ids(episode_count, fractions, split_seed)
    records: list[SampleRecord] = []
    for episode_index in range(episode_count):
        start, end = _episode_bounds(metadata, episode_index)
        for dataset_index in range(start, end, frame_stride):
            records.append(
                SampleRecord(
                    dataset_index=dataset_index,
                    episode_index=episode_index,
                    frame_index=dataset_index,
                    split_id=int(split_ids[episode_index]),
                )
            )
    if max_samples is not None:
        if max_samples < 0:
            raise ValueError("max_samples must be nonnegative")
        records = records[: int(max_samples)]
    if not records:
        raise ValueError("record selection produced no samples")
    return tuple(records)


def _records_digest(records: Sequence[SampleRecord]) -> str:
    digest = hashlib.sha256()
    for record in records:
        digest.update(
            f"{record.dataset_index}:{record.episode_index}:{record.frame_index}:{record.split_id}\n".encode()
        )
    return digest.hexdigest()


def _manifest_immutable(manifest: Mapping[str, Any], sample_count: int, horizon: int, action_dim: int) -> dict[str, Any]:
    return {
        "cache_version": CACHE_VERSION,
        "sample_count": sample_count,
        "horizon": horizon,
        "action_dim": action_dim,
        "dataset_identity": manifest.get("dataset_identity"),
        "split": manifest.get("split"),
        "source_checkpoint": manifest.get("source_checkpoint"),
        "norm_stats": manifest.get("norm_stats"),
        "sample_steps": manifest.get("sample_steps"),
        "noise_seed": manifest.get("noise_seed"),
        "source_model_action_width": manifest.get("source_model_action_width"),
        "decoder_action_width": manifest.get("decoder_action_width"),
        "action_space": manifest.get("action_space"),
    }


def _validate_manifest_input(manifest: Mapping[str, Any], action_dim: int) -> None:
    required = (
        "dataset_identity",
        "split",
        "source_checkpoint",
        "norm_stats",
        "sample_steps",
        "noise_seed",
        "source_model_action_width",
        "decoder_action_width",
        "action_space",
    )
    missing = [key for key in required if key not in manifest]
    if missing:
        raise ValueError(f"manifest missing required fields: {', '.join(missing)}")
    if not isinstance(manifest["dataset_identity"], Mapping) or not isinstance(manifest["split"], Mapping):
        raise ValueError("manifest dataset_identity and split must be mappings")
    if int(manifest["decoder_action_width"]) != action_dim:
        raise ValueError("manifest decoder_action_width must match action_dim")
    if int(manifest["source_model_action_width"]) < action_dim:
        raise ValueError("manifest source_model_action_width must be at least action_dim")
    if not str(manifest["action_space"]):
        raise ValueError("manifest action_space must be a nonempty normalized action-space name")


def _array_shapes(sample_count: int, horizon: int, action_dim: int) -> dict[str, tuple[int, ...]]:
    return {
        "coarse_actions": (sample_count, horizon, action_dim),
        "expert_actions": (sample_count, horizon, action_dim),
        "valid_masks": (sample_count, horizon),
        "dataset_indices": (sample_count,),
        "episode_indices": (sample_count,),
        "split_ids": (sample_count,),
    }


def _open_arrays(cache_dir: Path, sample_count: int, horizon: int, action_dim: int, mode: str) -> dict[str, np.ndarray]:
    arrays: dict[str, np.ndarray] = {}
    for name, (filename, dtype, _) in ARRAY_SPECS.items():
        path = cache_dir / filename
        if not path.exists():
            raise FileNotFoundError(f"cache array is missing: {filename}")
        array = np.load(path, mmap_mode=mode, allow_pickle=False)
        if array.dtype != dtype or tuple(array.shape) != _array_shapes(sample_count, horizon, action_dim)[name]:
            raise ValueError(f"cache array {name} has invalid dtype or shape")
        arrays[name] = array
    return arrays


def _acquire_writer_lock(cache_dir: Path) -> int:
    lock_fd = os.open(cache_dir / WRITER_LOCK_NAME, os.O_RDWR | os.O_CREAT, 0o600)
    try:
        fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        os.close(lock_fd)
        raise RuntimeError(f"action cache writer is locked: {cache_dir}") from error
    return lock_fd


class ActionCacheWriter:
    def __init__(self, cache_dir: Path, manifest: dict[str, Any], arrays: dict[str, np.ndarray], lock_fd: int) -> None:
        self.cache_dir = cache_dir
        self.manifest = manifest
        self.arrays = arrays
        self._lock_fd: int | None = lock_fd

    @classmethod
    def create(
        cls,
        path: str | Path,
        *,
        sample_count: int,
        horizon: int,
        action_dim: int,
        manifest: Mapping[str, Any],
    ) -> "ActionCacheWriter":
        if sample_count <= 0 or horizon <= 0 or action_dim <= 0:
            raise ValueError("sample_count, horizon, and action_dim must be positive")
        _validate_manifest_input(manifest, action_dim)
        cache_dir = Path(path)
        cache_dir.mkdir(parents=True, exist_ok=True)
        lock_fd = _acquire_writer_lock(cache_dir)
        try:
            if any(entry.name != WRITER_LOCK_NAME for entry in cache_dir.iterdir()):
                raise FileExistsError(f"cache directory is not empty: {cache_dir}")
            arrays: dict[str, np.ndarray] = {}
            for name, (filename, dtype, _) in ARRAY_SPECS.items():
                arrays[name] = np.lib.format.open_memmap(
                    cache_dir / filename,
                    mode="w+",
                    dtype=dtype,
                    shape=_array_shapes(sample_count, horizon, action_dim)[name],
                )
            for array in arrays.values():
                array.flush()
            immutable = _manifest_immutable(manifest, sample_count, horizon, action_dim)
            cache_manifest = dict(manifest)
            cache_manifest.update(
                {
                    "version": CACHE_VERSION,
                    "status": "incomplete",
                    "sample_count": sample_count,
                    "completed_samples": 0,
                    "horizon": horizon,
                    "action_dim": action_dim,
                    "immutable_manifest": immutable,
                    "records_sha256": None,
                }
            )
            _atomic_write_json(cache_dir / MANIFEST_NAME, cache_manifest)
            return cls(cache_dir, cache_manifest, arrays, lock_fd)
        except Exception:
            fcntl.flock(lock_fd, fcntl.LOCK_UN)
            os.close(lock_fd)
            raise

    @classmethod
    def resume(cls, path: str | Path, manifest: Mapping[str, Any]) -> "ActionCacheWriter":
        cache_dir = Path(path)
        lock_fd = _acquire_writer_lock(cache_dir)
        try:
            cache_manifest = _load_manifest(cache_dir)
            if cache_manifest.get("status") != "incomplete":
                raise ValueError("only incomplete caches can be resumed")
            sample_count = _positive_int(cache_manifest, "sample_count")
            horizon = _positive_int(cache_manifest, "horizon")
            action_dim = _positive_int(cache_manifest, "action_dim")
            _validate_manifest_input(manifest, action_dim)
            if cache_manifest.get("immutable_manifest") != _manifest_immutable(manifest, sample_count, horizon, action_dim):
                raise ValueError("immutable manifest does not match cache")
            completed = _completed_samples(cache_manifest, sample_count)
            arrays = _open_arrays(cache_dir, sample_count, horizon, action_dim, "r+")
            if completed and cache_manifest.get("completed_records_sha256") != _records_digest(_records_from_arrays(arrays, completed)):
                raise ValueError("completed record digest does not match cache arrays")
            return cls(cache_dir, cache_manifest, arrays, lock_fd)
        except Exception:
            fcntl.flock(lock_fd, fcntl.LOCK_UN)
            os.close(lock_fd)
            raise

    def write_batch(
        self,
        start: int,
        *,
        coarse: np.ndarray,
        expert: np.ndarray,
        valid: np.ndarray,
        records: Sequence[SampleRecord],
    ) -> None:
        if self._lock_fd is None:
            raise RuntimeError("action cache writer is closed")
        sample_count = _positive_int(self.manifest, "sample_count")
        horizon = _positive_int(self.manifest, "horizon")
        action_dim = _positive_int(self.manifest, "action_dim")
        completed = _completed_samples(self.manifest, sample_count)
        if start != completed:
            raise ValueError(f"batch start must equal completed_samples ({completed}), got {start}")
        coarse = np.asarray(coarse)
        expert = np.asarray(expert)
        valid = np.asarray(valid)
        batch_size = len(records)
        expected_actions = (batch_size, horizon, action_dim)
        if coarse.shape != expected_actions or expert.shape != expected_actions or valid.shape != (batch_size, horizon):
            raise ValueError("batch shape does not match cache schema")
        if coarse.dtype != np.float32 or expert.dtype != np.float32 or valid.dtype != np.bool_:
            raise ValueError("batch dtypes must be float32, float32, and bool")
        if not np.isfinite(coarse).all() or not np.isfinite(expert).all():
            raise ValueError("action batches must contain only finite values")
        if not records or start + batch_size > sample_count:
            raise ValueError("batch bounds exceed cache sample_count")
        for record in records:
            if not isinstance(record, SampleRecord) or record.split_id not in SPLIT_IDS.values():
                raise ValueError("records must be SampleRecord values with split IDs 0, 1, or 2")
        stop = start + batch_size
        self.arrays["coarse_actions"][start:stop] = coarse
        self.arrays["expert_actions"][start:stop] = expert
        self.arrays["valid_masks"][start:stop] = valid
        self.arrays["dataset_indices"][start:stop] = [record.dataset_index for record in records]
        self.arrays["episode_indices"][start:stop] = [record.episode_index for record in records]
        self.arrays["split_ids"][start:stop] = [record.split_id for record in records]
        for array in self.arrays.values():
            array.flush()
        self.manifest["completed_samples"] = stop
        self.manifest["completed_records_sha256"] = _records_digest(_records_from_arrays(self.arrays, stop))
        _atomic_write_json(self.cache_dir / MANIFEST_NAME, self.manifest)

    def finalize(self) -> None:
        try:
            sample_count = _positive_int(self.manifest, "sample_count")
            if _completed_samples(self.manifest, sample_count) != sample_count:
                raise ValueError("cannot finalize cache before all samples are written")
            for array in self.arrays.values():
                array.flush()
            records = _records_from_arrays(self.arrays, sample_count)
            self.manifest["records_sha256"] = _records_digest(records)
            self.manifest["status"] = "complete"
            _atomic_write_json(self.cache_dir / MANIFEST_NAME, self.manifest)
        finally:
            self.close()

    def close(self) -> None:
        if self._lock_fd is not None:
            fcntl.flock(self._lock_fd, fcntl.LOCK_UN)
            os.close(self._lock_fd)
            self._lock_fd = None

    def __enter__(self) -> "ActionCacheWriter":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


def _positive_int(manifest: Mapping[str, Any], name: str) -> int:
    value = manifest.get(name)
    if type(value) is not int or value <= 0:
        raise ValueError(f"manifest {name} must be a positive integer")
    return value


def _completed_samples(manifest: Mapping[str, Any], sample_count: int) -> int:
    completed = manifest.get("completed_samples")
    if type(completed) is not int or not 0 <= completed <= sample_count:
        raise ValueError("manifest completed_samples is invalid")
    return completed


def _load_manifest(cache_dir: Path) -> dict[str, Any]:
    path = cache_dir / MANIFEST_NAME
    if not path.exists():
        raise FileNotFoundError(f"cache manifest is missing: {path}")
    with path.open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    if manifest.get("version") != CACHE_VERSION:
        raise ValueError("unsupported cache manifest version")
    return manifest


def _records_from_arrays(arrays: Mapping[str, np.ndarray], count: int) -> tuple[SampleRecord, ...]:
    split_ids = np.asarray(arrays["split_ids"][:count])
    if not np.isin(split_ids, tuple(SPLIT_IDS.values())).all():
        raise ValueError("cache split_ids must be 0, 1, or 2")
    return tuple(
        SampleRecord(
            dataset_index=int(arrays["dataset_indices"][index]),
            episode_index=int(arrays["episode_indices"][index]),
            frame_index=int(arrays["dataset_indices"][index]),
            split_id=int(arrays["split_ids"][index]),
        )
        for index in range(count)
    )


class ActionCache:
    def __init__(self, cache_dir: Path, manifest: dict[str, Any], arrays: dict[str, np.ndarray]) -> None:
        self.cache_dir = cache_dir
        self.manifest = manifest
        self.coarse_actions = arrays["coarse_actions"]
        self.expert_actions = arrays["expert_actions"]
        self.valid_masks = arrays["valid_masks"]
        self.dataset_indices = arrays["dataset_indices"]
        self.episode_indices = arrays["episode_indices"]
        self.split_ids = arrays["split_ids"]

    @classmethod
    def open(cls, path: str | Path) -> "ActionCache":
        cache_dir = Path(path)
        expected = {MANIFEST_NAME, *(spec[0] for spec in ARRAY_SPECS.values())}
        actual = {entry.name for entry in cache_dir.iterdir()} if cache_dir.exists() else set()
        if actual not in (expected, expected | {WRITER_LOCK_NAME}):
            raise ValueError("cache contains missing or unexpected array files")
        manifest = _load_manifest(cache_dir)
        if manifest.get("status") != "complete":
            raise ValueError("cache is not complete")
        sample_count = _positive_int(manifest, "sample_count")
        if _completed_samples(manifest, sample_count) != sample_count:
            raise ValueError("complete cache has incomplete progress")
        arrays = _open_arrays(cache_dir, sample_count, _positive_int(manifest, "horizon"), _positive_int(manifest, "action_dim"), "r")
        records = _records_from_arrays(arrays, sample_count)
        if manifest.get("records_sha256") != _records_digest(records):
            raise ValueError("record digest does not match cache arrays")
        return cls(cache_dir, manifest, arrays)

    def indices(self, split: Literal["train", "validation", "test"]) -> np.ndarray:
        if split not in SPLIT_IDS:
            raise ValueError(f"unknown split: {split}")
        return np.flatnonzero(np.asarray(self.split_ids) == SPLIT_IDS[split]).astype(np.int64)
