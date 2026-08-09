"""Resumable writer for complete frozen SmolVLA training caches."""

from __future__ import annotations

import json
import os
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from .atomic_checkpoint import _path_exists, _rename_noreplace
from .offline_training_cache import (
    ACTION_IS_PAD_NAME,
    ACTIONS_NAME,
    EPISODE_INDEX_NAME,
    FRAME_INDEX_NAME,
    LANGUAGE_MASKS_NAME,
    LANGUAGE_TOKENS_NAME,
    METADATA_NAME,
    OFFLINE_CACHE_SCHEMA_VERSION,
    PROGRESS_NAME,
    STATE_NAME,
    VISION_TOKENS_NAME,
    OfflineCacheSpec,
    OfflineTrainingCache,
    _spec_metadata,
    bfloat16_to_uint16,
    uint16_to_bfloat16,
)


class InjectedStop(RuntimeError):
    """Test-only interruption raised after durable progress reaches a boundary."""


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as file:
        json.dump(dict(payload), file, indent=2, sort_keys=True)
        file.write("\n")
        file.flush()
        os.fsync(file.fileno())
    temporary.replace(path)


def read_progress(cache_dir: str | Path) -> dict[str, Any]:
    path = Path(cache_dir) / PROGRESS_NAME
    if not path.is_file():
        raise FileNotFoundError(f"offline cache progress not found: {path}")
    with path.open(encoding="utf-8") as file:
        progress = json.load(file)
    if not isinstance(progress, dict):
        raise ValueError(f"offline cache progress must be a mapping: {path}")
    return progress


def _array_contracts(spec: OfflineCacheSpec) -> dict[str, tuple[str, np.dtype, tuple[int, ...]]]:
    return {
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
        "episode_index": (
            EPISODE_INDEX_NAME,
            np.dtype(np.int64),
            (spec.total_frames,),
        ),
        "frame_index": (FRAME_INDEX_NAME, np.dtype(np.int64), (spec.total_frames,)),
    }


def _progress_payload(spec: OfflineCacheSpec, *, next_index: int, status: str) -> dict[str, Any]:
    return {
        "version": OFFLINE_CACHE_SCHEMA_VERSION,
        "status": status,
        "next_index": int(next_index),
        **_spec_metadata(spec),
    }


class OfflineCachePrecomputer:
    """Write each dataset row once and publish only a validated complete cache."""

    def __init__(
        self,
        *,
        spec: OfflineCacheSpec,
        output_dir: str | Path,
        dataset: Sequence[Mapping[str, Any]],
        encode_vision: Callable[[np.ndarray], Any],
        tokenize: Callable[[list[str]], tuple[Any, Any]],
        batch_size: int,
    ) -> None:
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        if len(dataset) != spec.total_frames:
            raise ValueError(
                f"dataset has {len(dataset)} frames, expected spec total_frames={spec.total_frames}"
            )
        self.spec = spec
        self.output_dir = Path(output_dir).expanduser()
        self.dataset = dataset
        self.encode_vision = encode_vision
        self.tokenize = tokenize
        self.batch_size = int(batch_size)

    @property
    def staging_dir(self) -> Path:
        return self.output_dir.with_name(self.output_dir.name + ".incomplete")

    def _create_arrays(self) -> dict[str, np.memmap]:
        self.staging_dir.mkdir(parents=True)
        arrays = {
            field: np.lib.format.open_memmap(
                self.staging_dir / filename,
                mode="w+",
                dtype=dtype,
                shape=shape,
            )
            for field, (filename, dtype, shape) in _array_contracts(self.spec).items()
        }
        for array in arrays.values():
            array.flush()
        _atomic_write_json(
            self.staging_dir / PROGRESS_NAME,
            _progress_payload(self.spec, next_index=0, status="incomplete"),
        )
        return arrays

    def _open_arrays(self) -> dict[str, np.memmap]:
        arrays: dict[str, np.memmap] = {}
        for field, (filename, dtype, shape) in _array_contracts(self.spec).items():
            path = self.staging_dir / filename
            if not path.is_file():
                raise FileNotFoundError(f"offline cache {field} file not found: {path}")
            array = np.lib.format.open_memmap(path, mode="r+")
            if array.dtype != dtype or array.shape != shape:
                raise ValueError(
                    f"offline cache {field} contract mismatch: "
                    f"got dtype={array.dtype} shape={array.shape}, expected dtype={dtype} shape={shape}"
                )
            arrays[field] = array
        return arrays

    def _validate_progress(self, progress: Mapping[str, Any]) -> int:
        expected = _progress_payload(self.spec, next_index=0, status="incomplete")
        for field, value in expected.items():
            if field in ("next_index", "status"):
                continue
            if progress.get(field) != value:
                raise ValueError(
                    f"offline cache progress {field} is incompatible: "
                    f"expected {value!r}, got {progress.get(field)!r}"
                )
        next_index = progress.get("next_index")
        if isinstance(next_index, bool) or not isinstance(next_index, int):
            raise ValueError(f"offline cache progress next_index must be an integer, got {next_index!r}")
        if not 0 <= next_index <= self.spec.total_frames:
            raise ValueError(
                f"offline cache progress next_index={next_index} is outside "
                f"[0, {self.spec.total_frames}]"
            )
        status = progress.get("status")
        if status not in ("incomplete", "complete"):
            raise ValueError(f"offline cache progress has invalid status {status!r}")
        if status == "complete" and next_index != self.spec.total_frames:
            raise ValueError(
                "offline cache progress cannot be complete before next_index reaches "
                f"{self.spec.total_frames}"
            )
        return next_index

    @staticmethod
    def _stack(
        samples: list[Mapping[str, Any]],
        field: str,
        *,
        dtype: np.dtype,
        expected_shape: tuple[int, ...],
    ) -> np.ndarray:
        values = np.stack([np.asarray(sample[field]) for sample in samples], axis=0).astype(
            dtype, copy=False
        )
        if values.shape != expected_shape:
            raise ValueError(
                f"offline cache input {field} has shape {values.shape}, expected {expected_shape}"
            )
        return values

    def _write_batch(
        self,
        arrays: Mapping[str, np.memmap],
        *,
        start: int,
        end: int,
    ) -> None:
        samples = [self.dataset[index] for index in range(start, end)]
        count = end - start
        images = np.stack([np.asarray(sample["images"]) for sample in samples], axis=0)
        vision = np.asarray(self.encode_vision(images))
        expected_vision = (
            count,
            len(self.spec.camera_keys),
            self.spec.vision_tokens_per_camera,
            self.spec.vision_hidden_size,
        )
        if vision.shape != expected_vision:
            raise ValueError(
                f"encoded vision tokens have shape {vision.shape}, expected {expected_vision}"
            )
        language_tokens, language_masks = self.tokenize([str(sample["task"]) for sample in samples])
        language_tokens = np.asarray(language_tokens, dtype=np.int32)
        language_masks = np.asarray(language_masks, dtype=np.bool_)
        expected_language = (count, self.spec.tokenizer_max_length)
        if language_tokens.shape != expected_language or language_masks.shape != expected_language:
            raise ValueError(
                "tokenizer outputs must both have shape "
                f"{expected_language}, got {language_tokens.shape} and {language_masks.shape}"
            )

        arrays["vision_tokens"][start:end] = bfloat16_to_uint16(vision)
        arrays["state"][start:end] = self._stack(
            samples,
            "state",
            dtype=np.dtype(np.float32),
            expected_shape=(count, self.spec.state_dim),
        )
        arrays["actions"][start:end] = self._stack(
            samples,
            "actions",
            dtype=np.dtype(np.float32),
            expected_shape=(count, self.spec.chunk_size, self.spec.action_dim),
        )
        arrays["action_is_pad"][start:end] = self._stack(
            samples,
            "action_is_pad",
            dtype=np.dtype(np.bool_),
            expected_shape=(count, self.spec.chunk_size),
        )
        arrays["language_tokens"][start:end] = language_tokens
        arrays["language_masks"][start:end] = language_masks
        arrays["episode_index"][start:end] = self._stack(
            samples,
            "episode_index",
            dtype=np.dtype(np.int64),
            expected_shape=(count,),
        )
        arrays["frame_index"][start:end] = self._stack(
            samples,
            "frame_index",
            dtype=np.dtype(np.int64),
            expected_shape=(count,),
        )

    def _validate_finite(self, arrays: Mapping[str, np.memmap]) -> None:
        for field, array in arrays.items():
            if array.dtype == np.bool_:
                continue
            for start in range(0, self.spec.total_frames, self.batch_size):
                chunk = np.asarray(array[start : start + self.batch_size])
                if field == "vision_tokens":
                    chunk = uint16_to_bfloat16(chunk).astype(np.float32)
                if not np.isfinite(chunk).all():
                    raise ValueError(f"offline cache {field} contains non-finite values")

    def run(self, *, stop_after: int | None = None) -> Path:
        if _path_exists(self.output_dir):
            completed = OfflineTrainingCache(self.output_dir, self.spec)
            self._validate_finite(completed._arrays)
            return self.output_dir

        if stop_after is not None:
            if isinstance(stop_after, bool) or not isinstance(stop_after, int):
                raise ValueError(f"stop_after must be an integer, got {stop_after!r}")
            if not 0 <= stop_after <= self.spec.total_frames:
                raise ValueError(
                    f"stop_after={stop_after} is outside [0, {self.spec.total_frames}]"
                )

        if _path_exists(self.staging_dir):
            progress = read_progress(self.staging_dir)
            next_index = self._validate_progress(progress)
            arrays = self._open_arrays()
        else:
            arrays = self._create_arrays()
            next_index = 0

        if stop_after is not None and next_index >= stop_after:
            raise InjectedStop(f"injected stop at frame {next_index}")

        target = self.spec.total_frames if stop_after is None else stop_after
        while next_index < target:
            end = min(next_index + self.batch_size, target)
            self._write_batch(arrays, start=next_index, end=end)
            for array in arrays.values():
                array.flush()
            next_index = end
            _atomic_write_json(
                self.staging_dir / PROGRESS_NAME,
                _progress_payload(self.spec, next_index=next_index, status="incomplete"),
            )

        if next_index != self.spec.total_frames:
            raise InjectedStop(f"injected stop at frame {next_index}")

        self._validate_finite(arrays)
        metadata = {
            "version": OFFLINE_CACHE_SCHEMA_VERSION,
            "status": "complete",
            **_spec_metadata(self.spec),
        }
        _atomic_write_json(self.staging_dir / METADATA_NAME, metadata)
        _atomic_write_json(
            self.staging_dir / PROGRESS_NAME,
            _progress_payload(self.spec, next_index=next_index, status="complete"),
        )
        OfflineTrainingCache(self.staging_dir, self.spec)
        if _path_exists(self.output_dir):
            raise FileExistsError(f"offline cache path already exists: {self.output_dir}")
        _rename_noreplace(self.staging_dir, self.output_dir)
        return self.output_dir
