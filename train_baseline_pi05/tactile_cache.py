"""Frozen current-frame tactile ResNet18 feature cache."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import uuid
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .config import TACTILE_KEYS

EMBEDDINGS_NAME = "embeddings.npy"
MANIFEST_NAME = "manifest.json"
WRITER_LOCK_NAME = ".writer.lock"
PREPROCESS_VERSION = "resize_with_pad_uint8_to_unit_v1"


def _identity(path: Path) -> str:
    digest = hashlib.sha256(str(path.expanduser().resolve()).encode())
    if path.exists():
        for file in sorted(candidate for candidate in path.rglob("*") if candidate.is_file()):
            digest.update(str(file.relative_to(path)).encode())
            digest.update(file.read_bytes())
    return digest.hexdigest()


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, sort_keys=True, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _acquire_writer_lock(cache_dir: Path) -> int:
    lock_fd = os.open(cache_dir / WRITER_LOCK_NAME, os.O_RDWR | os.O_CREAT, 0o600)
    try:
        fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        os.close(lock_fd)
        raise RuntimeError(f"tactile cache writer is locked: {cache_dir}") from error
    return lock_fd


class TactileEmbeddingCache:
    def __init__(self, cache_dir: Path, metadata: Mapping[str, Any], embeddings: np.memmap) -> None:
        self.cache_dir, self.metadata, self.embeddings = cache_dir, dict(metadata), embeddings

    @classmethod
    def open(cls, path: str | Path, *, tactile_keys: Sequence[str] = TACTILE_KEYS, encoder_path: str | Path) -> "TactileEmbeddingCache":
        cache_dir = Path(path)
        metadata = json.loads((cache_dir / MANIFEST_NAME).read_text(encoding="utf-8"))
        expected_keys = list(tactile_keys)
        if metadata.get("status") != "complete" or metadata.get("tactile_keys") != expected_keys:
            raise ValueError("tactile cache manifest has an invalid status or key order")
        if metadata.get("preprocess_version") != PREPROCESS_VERSION or metadata.get("encoder_identity") != _identity(Path(encoder_path)):
            raise ValueError("tactile cache preprocessing or encoder identity does not match")
        embeddings = np.load(cache_dir / EMBEDDINGS_NAME, mmap_mode="r", allow_pickle=False)
        shape = (int(metadata["total_frames"]), len(expected_keys), int(metadata["embedding_dim"]))
        if embeddings.dtype != np.float32 or tuple(embeddings.shape) != shape or not np.isfinite(embeddings).all():
            raise ValueError("tactile embeddings have invalid dtype, shape, or values")
        return cls(cache_dir, metadata, embeddings)

    def get_many(self, indices: Sequence[int]) -> np.ndarray:
        values = np.asarray(indices, dtype=np.int64)
        if values.ndim != 1 or np.any(values < 0) or np.any(values >= self.embeddings.shape[0]):
            raise IndexError("tactile cache indices are out of range")
        return np.asarray(self.embeddings[values])


def _default_encoder(checkpoint: Path) -> Callable[[np.ndarray], np.ndarray]:
    from .tactile_encoder.encoder_checkpoint import load_tactile_encoder
    from .tactile_encoder.resnet import encode_resnet18
    import jax.numpy as jnp

    variables = load_tactile_encoder(checkpoint).params.get("tactile_resnet")
    if not isinstance(variables, dict):
        raise ValueError("encoder checkpoint must contain the shared ResNet18 subtree")
    def encode(images: np.ndarray) -> np.ndarray:
        result, _ = encode_resnet18(variables, jnp.asarray(images), train=False)
        return np.asarray(result, dtype=np.float32)
    return encode


def _required_tactile_frames(
    metadata: Any,
    *,
    frame_stride: int,
    max_samples: int,
    split_seed: int = 0,
    fractions: Sequence[float] = (0.8, 0.1, 0.1),
) -> int:
    """Cover every frame index selected by the matching capped action records."""
    from .action_cache import build_records

    records = build_records(
        metadata,
        split_seed=split_seed,
        fractions=fractions,
        frame_stride=frame_stride,
        max_samples=max_samples,
    )
    return int(records[-1].dataset_index) + 1


def _selection_contract(dataset_info: Any, max_samples: int | None) -> dict[str, object]:
    return {
        "max_samples": max_samples,
        "frame_stride": int(getattr(dataset_info, "frame_stride", 1)),
        "split_seed": int(getattr(dataset_info, "split_seed", 0)),
        "fractions": [
            float(getattr(dataset_info, "train_fraction", 0.8)),
            float(getattr(dataset_info, "validation_fraction", 0.1)),
            float(getattr(dataset_info, "test_fraction", 0.1)),
        ],
    }


def _manifest_contract(
    *,
    count: int,
    dim: int,
    keys: Sequence[str],
    checkpoint: Path,
    dataset_info: Any,
    selection: Mapping[str, object],
) -> dict[str, object]:
    return {
        "version": 1,
        "status": "complete",
        "total_frames": count,
        "tactile_keys": list(keys),
        "embedding_dim": dim,
        "dtype": "float32",
        "image_size": 224,
        "preprocess_version": PREPROCESS_VERSION,
        "encoder_identity": _identity(checkpoint),
        "dataset_identity": {
            "repo_id": getattr(dataset_info, "repo_id", "injected"),
            "root": str(Path(getattr(dataset_info, "root", "injected")).expanduser().resolve()),
            "revision": getattr(dataset_info, "revision", None),
        },
        "selection": dict(selection),
    }


def prepare_tactile_cache(config: Any, dependencies: Mapping[str, Any] | None = None, *, max_samples: int | None = None) -> Path:
    """Encode four current tactile frames into RMS-normalized float32 tokens."""
    deps = dict(dependencies or {})
    tactile = getattr(config, "tactile", config)
    dataset_info = getattr(config, "dataset", config)
    cache_info = getattr(config, "cache", config)
    output, checkpoint = Path(cache_info.tactile_root), Path(tactile.encoder_checkpoint)
    keys = tuple(getattr(getattr(config, "decoder", config), "tactile_keys", TACTILE_KEYS))
    if keys != TACTILE_KEYS:
        raise ValueError("tactile keys must use the canonical four-current-frame order")
    dataset = deps.get("dataset")
    if dataset is None:
        from .runtime_path import activate_vendored_lerobot
        activate_vendored_lerobot()
        from lerobot.datasets import LeRobotDataset
        dataset = LeRobotDataset(dataset_info.repo_id, root=dataset_info.root, revision=getattr(dataset_info, "revision", None))
    if max_samples is not None and max_samples <= 0:
        raise ValueError("max_samples must be positive when provided")
    selection = _selection_contract(dataset_info, max_samples)
    if max_samples is None:
        count = len(dataset)
    else:
        metadata = deps.get("metadata") or getattr(dataset, "meta", None)
        if metadata is None:
            raise ValueError("max_samples requires dataset metadata for action-cache alignment")
        count = min(
            len(dataset),
            _required_tactile_frames(
                metadata,
                frame_stride=int(selection["frame_stride"]),
                max_samples=max_samples,
                split_seed=int(selection["split_seed"]),
                fractions=tuple(selection["fractions"]),
            ),
        )
    dim = int(tactile.embedding_dim)
    if dim != 512:
        raise ValueError("frozen ResNet18 cache embedding_dim must be 512")
    contract = _manifest_contract(
        count=count,
        dim=dim,
        keys=keys,
        checkpoint=checkpoint,
        dataset_info=dataset_info,
        selection=selection,
    )
    manifest_path = output / MANIFEST_NAME
    embeddings_path = output / EMBEDDINGS_NAME
    output.mkdir(parents=True, exist_ok=True)
    lock_fd = _acquire_writer_lock(output)
    try:
        if manifest_path.exists():
            existing = json.loads(manifest_path.read_text(encoding="utf-8"))
            if existing != contract:
                raise ValueError("complete tactile cache does not match requested encoder, dataset, count, or selection contract")
            TactileEmbeddingCache.open(output, tactile_keys=keys, encoder_path=checkpoint)
            return output
        if any(entry.name != WRITER_LOCK_NAME for entry in output.iterdir()):
            raise ValueError("tactile cache root contains files without a matching complete manifest")
        temporary = output / f".{EMBEDDINGS_NAME}.{uuid.uuid4().hex}.npy"
        published = manifest_published = False
        encoder = deps.get("encoder") or _default_encoder(checkpoint)
        from .tactile_encoder.preprocess import parse_image_to_unit
        embeddings = np.lib.format.open_memmap(temporary, mode="w+", dtype=np.float32, shape=(count, 4, dim))
        for index in range(count):
            sample = dataset[index]
            images = np.stack([parse_image_to_unit(sample[key], image_size=224) for key in keys])
            tokens = np.asarray(encoder(images), dtype=np.float32)
            if tokens.shape != (4, dim) or not np.isfinite(tokens).all():
                raise ValueError("shared ResNet18 encoder must return finite [4, 512] tokens")
            rms = np.sqrt(np.mean(np.square(tokens), axis=-1, keepdims=True))
            if np.any(rms == 0):
                raise ValueError("tactile encoder produced a zero RMS token")
            embeddings[index] = tokens / rms
        embeddings.flush()
        del embeddings
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, embeddings_path)
        published = True
        _atomic_json(manifest_path, contract)
        manifest_published = True
    except BaseException:
        if "temporary" in locals():
            temporary.unlink(missing_ok=True)
        (manifest_path.with_suffix(".tmp")).unlink(missing_ok=True)
        if "published" in locals() and published and not manifest_published:
            embeddings_path.unlink(missing_ok=True)
        raise
    finally:
        fcntl.flock(lock_fd, fcntl.LOCK_UN)
        os.close(lock_fd)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--max-samples", type=int)
    args = parser.parse_args()
    from .config import load_config
    prepare_tactile_cache(load_config(args.config), max_samples=args.max_samples)


if __name__ == "__main__":
    main()
