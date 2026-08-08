from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .provenance import (
    TACTILE_ENCODER_PROVENANCE_FILENAME,
    local_dataset_content_identity,
    sha256_file,
    validate_tactile_encoder_provenance,
    write_tactile_encoder_provenance,
)

TACTILE_CACHE_VERSION = 2
TACTILE_EMBEDDINGS_NAME = "embeddings.npy"
TACTILE_METADATA_NAME = "metadata.json"
TACTILE_EMBEDDING_OBSERVATION_KEY = "observation.tactile_embeddings"


def _immutable_revision(revision: str | None) -> str | None:
    if revision is None:
        return None
    value = str(revision).lower()
    if len(value) == 40 and all(character in "0123456789abcdef" for character in value):
        return value
    return None


def tactile_cache_dir(cache_root: str | Path, repo_id: str) -> Path:
    """Return ``cache_root/namespace/dataset`` for a Hugging Face repo id."""

    parts = [part for part in str(repo_id).split("/") if part not in ("", ".", "..")]
    if not parts:
        raise ValueError(f"invalid dataset repo id: {repo_id!r}")
    return Path(cache_root).expanduser().joinpath(*parts)


def dataset_root_fingerprint(dataset_root: str | Path) -> str | None:
    """Return the path-independent canonical local dataset identity digest."""

    root = Path(dataset_root).expanduser().resolve()
    if not root.is_dir():
        return None
    return str(local_dataset_content_identity(root)["sha256"])


def atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as file:
        json.dump(dict(value), file, indent=2, sort_keys=True)
        file.write("\n")
        file.flush()
        os.fsync(file.fileno())
    temporary.replace(path)


def tactile_encoder_fingerprint(checkpoint_dir: str | Path) -> str:
    """Validate encoder provenance and return its canonical checkpoint digest."""

    provenance = validate_tactile_encoder_provenance(checkpoint_dir)
    return str(provenance["checkpoint_sha256"])


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
    dataset_identity = local_dataset_content_identity(dataset_root)
    encoder_provenance = validate_tactile_encoder_provenance(encoder_path)
    encoder_provenance_path = (
        Path(encoder_path).expanduser().resolve() / TACTILE_ENCODER_PROVENANCE_FILENAME
    )
    return {
        "version": TACTILE_CACHE_VERSION,
        "status": status,
        "completed_frames": int(completed_frames),
        "repo_id": str(repo_id),
        "revision": _immutable_revision(revision),
        "dataset_content_identity": dataset_identity,
        # Keep the concise field name for diagnostics while version 2 rejects
        # every legacy cache that did not carry the full canonical identity.
        "dataset_fingerprint": dataset_identity["sha256"],
        "total_frames": int(total_frames),
        "tactile_keys": list(tactile_keys),
        "source_tactile_keys": list(source_tactile_keys),
        "num_tactile_tokens": len(tactile_keys),
        "embedding_dim": int(embedding_dim),
        "image_size": int(image_size),
        "dtype": resolved_dtype.name,
        "encoder_repo_id": encoder_provenance["repo_id"],
        "encoder_requested_revision": encoder_provenance["requested_revision"],
        "encoder_revision": encoder_provenance["resolved_revision"],
        "encoder_checkpoint_sha256": encoder_provenance["checkpoint_sha256"],
        "encoder_sha256": encoder_provenance["checkpoint_sha256"],
        "encoder_provenance_sha256": sha256_file(encoder_provenance_path),
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
        if dataset_root is None:
            raise ValueError("dataset_root is required to validate tactile cache content identity")
        encoder_provenance = validate_tactile_encoder_provenance(encoder_path)
        encoder_provenance_path = (
            Path(encoder_path).expanduser().resolve() / TACTILE_ENCODER_PROVENANCE_FILENAME
        )
        self.metadata = load_tactile_cache_metadata(self.cache_dir)
        self._embeddings: np.ndarray | None = None

        expected = {
            "repo_id": str(repo_id),
            "revision": _immutable_revision(revision),
            "total_frames": int(total_frames),
            "tactile_keys": list(tactile_keys),
            "source_tactile_keys": list(source_tactile_keys),
            "num_tactile_tokens": len(tactile_keys),
            "embedding_dim": int(embedding_dim),
            "image_size": int(image_size),
            "encoder_repo_id": encoder_provenance["repo_id"],
            "encoder_requested_revision": encoder_provenance["requested_revision"],
            "encoder_revision": encoder_provenance["resolved_revision"],
            "encoder_checkpoint_sha256": encoder_provenance["checkpoint_sha256"],
            "encoder_sha256": encoder_provenance["checkpoint_sha256"],
            "encoder_provenance_sha256": sha256_file(encoder_provenance_path),
        }
        dataset_identity = local_dataset_content_identity(dataset_root)
        expected["dataset_content_identity"] = dataset_identity
        expected["dataset_fingerprint"] = dataset_identity["sha256"]
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
