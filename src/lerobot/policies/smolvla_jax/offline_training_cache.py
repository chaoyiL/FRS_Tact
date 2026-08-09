"""Contract and reader for frozen SmolVLA offline training caches."""

from __future__ import annotations

import json
import operator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import ml_dtypes
import numpy as np
from numpy.typing import ArrayLike

OFFLINE_CACHE_SCHEMA_VERSION = 1
METADATA_NAME = "metadata.json"
PROGRESS_NAME = "progress.json"
VISION_TOKENS_NAME = "vision_tokens.uint16.npy"
STATE_NAME = "state.npy"
ACTIONS_NAME = "actions.npy"
ACTION_IS_PAD_NAME = "action_is_pad.npy"
LANGUAGE_TOKENS_NAME = "language_tokens.npy"
LANGUAGE_MASKS_NAME = "language_masks.npy"
EPISODE_INDEX_NAME = "episode_index.npy"
FRAME_INDEX_NAME = "frame_index.npy"


@dataclass(frozen=True)
class OfflineCacheSpec:
    repo_id: str
    total_frames: int
    camera_keys: tuple[str, ...]
    vision_tokens_per_camera: int
    vision_hidden_size: int
    state_dim: int
    action_dim: int
    chunk_size: int
    tokenizer_max_length: int
    checkpoint_source: str
    vision_mode: str
    connector_mode: str


def offline_cache_dir(root: Path, repo_id: str) -> Path:
    """Return a cache path derived from the namespace and name in ``repo_id``."""

    parts = str(repo_id).split("/")
    if not parts or any(part in ("", ".", "..") for part in parts):
        raise ValueError(f"invalid offline cache repo id: {repo_id!r}")
    return Path(root).expanduser().joinpath(*parts)


def bfloat16_to_uint16(values: ArrayLike) -> np.ndarray:
    """Store BF16 values as their raw uint16 payloads without rounding."""

    array = np.ascontiguousarray(np.asarray(values, dtype=ml_dtypes.bfloat16))
    return array.view(np.uint16)


def uint16_to_bfloat16(values: ArrayLike) -> np.ndarray:
    """View raw uint16 BF16 payloads as logical BF16 values."""

    array = np.asarray(values)
    if array.dtype != np.uint16:
        raise TypeError(f"expected uint16 BF16 storage, got {array.dtype}")
    return array.view(ml_dtypes.bfloat16)


def _spec_metadata(spec: OfflineCacheSpec) -> dict[str, Any]:
    return {
        "repo_id": spec.repo_id,
        "total_frames": spec.total_frames,
        "camera_keys": list(spec.camera_keys),
        "vision_tokens_per_camera": spec.vision_tokens_per_camera,
        "vision_hidden_size": spec.vision_hidden_size,
        "state_dim": spec.state_dim,
        "action_dim": spec.action_dim,
        "chunk_size": spec.chunk_size,
        "tokenizer_max_length": spec.tokenizer_max_length,
        "checkpoint_source": spec.checkpoint_source,
        "vision_mode": spec.vision_mode,
        "connector_mode": spec.connector_mode,
    }


class OfflineTrainingCache:
    """Validated, lazy memmap reader for precomputed frozen vision tokens."""

    def __init__(self, cache_dir: str | Path, spec: OfflineCacheSpec):
        self.cache_dir = Path(cache_dir).expanduser()
        self.spec = spec
        self.metadata = self._load_and_validate_metadata()
        self._arrays = self._load_and_validate_arrays()

    def _load_and_validate_metadata(self) -> Mapping[str, Any]:
        path = self.cache_dir / METADATA_NAME
        if not path.is_file():
            raise FileNotFoundError(f"offline training cache metadata not found: {path}")
        with path.open(encoding="utf-8") as file:
            metadata = json.load(file)
        if int(metadata.get("version", -1)) != OFFLINE_CACHE_SCHEMA_VERSION:
            raise ValueError(
                f"unsupported offline cache version {metadata.get('version')!r}; "
                f"expected {OFFLINE_CACHE_SCHEMA_VERSION}"
            )
        if metadata.get("status") != "complete":
            raise ValueError(
                f"offline training cache is {metadata.get('status')!r}, expected 'complete'"
            )
        for field, expected in _spec_metadata(self.spec).items():
            actual = metadata.get(field)
            if actual != expected:
                raise ValueError(
                    f"offline training cache {field} is incompatible: "
                    f"expected {expected!r}, got {actual!r}"
                )
        return metadata

    def _load_and_validate_arrays(self) -> dict[str, np.ndarray]:
        spec = self.spec
        contracts = {
            "vision_tokens": (
                VISION_TOKENS_NAME,
                np.dtype(np.uint16),
                (
                    spec.total_frames,
                    len(spec.camera_keys),
                    spec.vision_tokens_per_camera,
                    spec.vision_hidden_size,
                ),
            ),
            "state": (STATE_NAME, np.dtype(np.float32), (spec.total_frames, spec.state_dim)),
            "actions": (
                ACTIONS_NAME,
                np.dtype(np.float32),
                (spec.total_frames, spec.chunk_size, spec.action_dim),
            ),
            "action_is_pad": (
                ACTION_IS_PAD_NAME,
                np.dtype(np.bool_),
                (spec.total_frames, spec.chunk_size),
            ),
            "language_tokens": (
                LANGUAGE_TOKENS_NAME,
                np.dtype(np.int32),
                (spec.total_frames, spec.tokenizer_max_length),
            ),
            "language_masks": (
                LANGUAGE_MASKS_NAME,
                np.dtype(np.bool_),
                (spec.total_frames, spec.tokenizer_max_length),
            ),
            "episode_index": (EPISODE_INDEX_NAME, np.dtype(np.int64), (spec.total_frames,)),
            "frame_index": (FRAME_INDEX_NAME, np.dtype(np.int64), (spec.total_frames,)),
        }
        arrays: dict[str, np.ndarray] = {}
        for field, (filename, dtype, shape) in contracts.items():
            path = self.cache_dir / filename
            if not path.is_file():
                raise FileNotFoundError(f"offline training cache {field} file not found: {path}")
            array = np.load(path, mmap_mode="r", allow_pickle=False)
            if array.dtype != dtype:
                raise ValueError(
                    f"offline training cache {field} dtype is {array.dtype}, expected {dtype}"
                )
            if array.shape != shape:
                raise ValueError(
                    f"offline training cache {field} shape is {array.shape}, expected {shape}"
                )
            arrays[field] = array
        return arrays

    def __len__(self) -> int:
        return self.spec.total_frames

    def __getitem__(self, index: int) -> dict[str, np.ndarray]:
        item_index = operator.index(index)
        if not 0 <= item_index < len(self):
            raise IndexError(f"offline cache index {item_index} out of range for {len(self)} frames")
        return {
            "vision_tokens": uint16_to_bfloat16(self._arrays["vision_tokens"][item_index]),
            "state": self._arrays["state"][item_index],
            "actions": self._arrays["actions"][item_index],
            "action_is_pad": self._arrays["action_is_pad"][item_index],
            "language_tokens": self._arrays["language_tokens"][item_index],
            "language_masks": self._arrays["language_masks"][item_index],
            "episode_index": self._arrays["episode_index"][item_index],
            "frame_index": self._arrays["frame_index"][item_index],
        }
