from __future__ import annotations

import hashlib
import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

TACTILE_CACHE_VERSION = 1
TACTILE_EMBEDDINGS_NAME = "embeddings.npy"
TACTILE_METADATA_NAME = "metadata.json"
TACTILE_EMBEDDING_OBSERVATION_KEY = "observation.tactile_embeddings"


def tactile_cache_dir(cache_root: str | Path, repo_id: str) -> Path:
    """Return ``cache_root/namespace/dataset`` for a Hugging Face repo id."""

    parts = [part for part in str(repo_id).split("/") if part not in ("", ".", "..")]
    if not parts:
        raise ValueError(f"invalid dataset repo id: {repo_id!r}")
    return Path(cache_root).expanduser().joinpath(*parts)


def dataset_root_fingerprint(dataset_root: str | Path) -> str | None:
    """Fingerprint dataset file identities without reading large video payloads."""

    root = Path(dataset_root).expanduser().resolve()
    if not root.is_dir():
        return None
    digest = hashlib.sha256()
    files = sorted(path for path in root.rglob("*") if path.is_file())
    for path in files:
        stat = path.stat()
        digest.update(
            f"{path.relative_to(root)}:{stat.st_size}:{stat.st_mtime_ns}\n".encode()
        )
    return digest.hexdigest()


def atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as file:
        json.dump(dict(value), file, indent=2, sort_keys=True)
        file.write("\n")
        file.flush()
        os.fsync(file.fileno())
    temporary.replace(path)


@lru_cache(maxsize=8)
def tactile_encoder_fingerprint(checkpoint_dir: str | Path) -> str:
    """Hash the encoder files that determine frozen ResNet embeddings."""

    directory = Path(checkpoint_dir).expanduser().resolve()
    checkpoint_path = directory / "checkpoint.json"
    params_name = "params.npz"
    if checkpoint_path.is_file():
        with checkpoint_path.open(encoding="utf-8") as file:
            checkpoint_metadata = json.load(file)
        params_name = str(checkpoint_metadata.get("params_file", params_name))
    candidates = [checkpoint_path, directory / params_name]
    missing = [path for path in candidates if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"tactile encoder files are missing: {missing}")
    digest = hashlib.sha256()
    for path in candidates:
        digest.update(path.name.encode())
        with path.open("rb") as file:
            while chunk := file.read(8 * 1024 * 1024):
                digest.update(chunk)
    return digest.hexdigest()


def create_tactile_cache_metadata(
    *,
    repo_id: str,
    revision: str | None,
    dataset_root: str | Path,
    total_frames: int,
    tactile_keys: Sequence[str],
    source_tactile_keys: Sequence[str],
    embedding_dim: int,
    image_size: int,
    dtype: str | np.dtype,
    encoder_path: str | Path,
    completed_frames: int = 0,
    status: str = "incomplete",
) -> dict[str, Any]:
    resolved_dtype = np.dtype(dtype)
    if resolved_dtype not in (np.dtype(np.float16), np.dtype(np.float32)):
        raise ValueError(f"cache dtype must be float16 or float32, got {resolved_dtype}")
    return {
        "version": TACTILE_CACHE_VERSION,
        "status": status,
        "completed_frames": int(completed_frames),
        "repo_id": str(repo_id),
        "revision": None if revision is None else str(revision),
        "dataset_root": str(Path(dataset_root).expanduser().resolve()),
        "dataset_fingerprint": dataset_root_fingerprint(dataset_root),
        "total_frames": int(total_frames),
        "tactile_keys": list(tactile_keys),
        "source_tactile_keys": list(source_tactile_keys),
        "num_tactile_tokens": len(tactile_keys),
        "embedding_dim": int(embedding_dim),
        "image_size": int(image_size),
        "dtype": resolved_dtype.name,
        "encoder_path": str(Path(encoder_path).expanduser().resolve()),
        "encoder_sha256": tactile_encoder_fingerprint(encoder_path),
        "preprocessing": "tactile_encoder.parse_image_to_unit.v1",
    }


def load_tactile_cache_metadata(cache_dir: str | Path) -> dict[str, Any]:
    path = Path(cache_dir) / TACTILE_METADATA_NAME
    if not path.is_file():
        raise FileNotFoundError(f"tactile embedding cache metadata not found: {path}")
    with path.open(encoding="utf-8") as file:
        metadata = json.load(file)
    if int(metadata.get("version", -1)) != TACTILE_CACHE_VERSION:
        raise ValueError(
            f"unsupported tactile cache version {metadata.get('version')}; "
            f"expected {TACTILE_CACHE_VERSION}"
        )
    return metadata


class TactileEmbeddingCache:
    """Lazy, spawn-safe memmap reader for per-frame frozen ResNet embeddings."""

    def __init__(
        self,
        cache_dir: str | Path,
        *,
        repo_id: str,
        revision: str | None,
        total_frames: int,
        tactile_keys: Sequence[str],
        source_tactile_keys: Sequence[str],
        embedding_dim: int,
        image_size: int,
        encoder_path: str | Path,
        dataset_root: str | Path | None = None,
    ):
        self.cache_dir = Path(cache_dir).expanduser()
        self.metadata = load_tactile_cache_metadata(self.cache_dir)
        self._embeddings: np.ndarray | None = None

        expected = {
            "repo_id": str(repo_id),
            "revision": None if revision is None else str(revision),
            "total_frames": int(total_frames),
            "tactile_keys": list(tactile_keys),
            "source_tactile_keys": list(source_tactile_keys),
            "num_tactile_tokens": len(tactile_keys),
            "embedding_dim": int(embedding_dim),
            "image_size": int(image_size),
            "encoder_sha256": tactile_encoder_fingerprint(encoder_path),
        }
        if dataset_root is not None:
            expected["dataset_root"] = str(Path(dataset_root).expanduser().resolve())
            if "dataset_fingerprint" in self.metadata:
                expected["dataset_fingerprint"] = dataset_root_fingerprint(dataset_root)
        mismatches = {
            key: (self.metadata.get(key), value)
            for key, value in expected.items()
            if self.metadata.get(key) != value
        }
        if self.metadata.get("status") != "complete":
            raise ValueError(
                f"tactile embedding cache is incomplete: "
                f"{self.metadata.get('completed_frames', 0)}/{self.metadata.get('total_frames')} "
                f"at {self.cache_dir}"
            )
        if mismatches:
            raise ValueError(f"tactile embedding cache does not match training inputs: {mismatches}")
        embeddings_path = self.cache_dir / TACTILE_EMBEDDINGS_NAME
        if not embeddings_path.is_file():
            raise FileNotFoundError(f"tactile embeddings not found: {embeddings_path}")

    def _array(self) -> np.ndarray:
        if self._embeddings is None:
            embeddings = np.load(
                self.cache_dir / TACTILE_EMBEDDINGS_NAME,
                mmap_mode="r",
                allow_pickle=False,
            )
            expected_shape = (
                int(self.metadata["total_frames"]),
                int(self.metadata["num_tactile_tokens"]),
                int(self.metadata["embedding_dim"]),
            )
            if embeddings.shape != expected_shape:
                raise ValueError(
                    f"tactile embedding array shape mismatch: {embeddings.shape} != {expected_shape}"
                )
            self._embeddings = embeddings
        return self._embeddings

    def __len__(self) -> int:
        return int(self.metadata["total_frames"])

    def __getitem__(self, frame_index: int) -> np.ndarray:
        index = int(frame_index)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        # Copy only 4x512 values so default_collate receives writable storage.
        return np.array(self._array()[index], copy=True)

    def get_many(self, frame_indices: Sequence[int] | np.ndarray) -> np.ndarray:
        """Return copied embeddings for an arbitrary array of absolute frame indices."""

        indices = np.asarray(frame_indices, dtype=np.int64)
        if np.any(indices < 0) or np.any(indices >= len(self)):
            raise IndexError("tactile embedding frame index out of range")
        return np.array(self._array()[indices], copy=True)

    def __getstate__(self) -> dict[str, Any]:
        state = dict(self.__dict__)
        state["_embeddings"] = None
        return state
