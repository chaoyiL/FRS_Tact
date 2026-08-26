"""Frozen current-frame tactile ResNet18 feature cache."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .config import TACTILE_KEYS
from .tactile_encoder.preprocess import parse_image_to_unit

EMBEDDINGS_NAME = "embeddings.npy"
MANIFEST_NAME = "manifest.json"
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


def prepare_tactile_cache(config: Any, dependencies: Mapping[str, Any] | None = None) -> Path:
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
        from lerobot.datasets import LeRobotDataset
        dataset = LeRobotDataset(dataset_info.repo_id, root=dataset_info.root, revision=getattr(dataset_info, "revision", None))
    encoder = deps.get("encoder") or _default_encoder(checkpoint)
    count, dim = len(dataset), int(tactile.embedding_dim)
    if dim != 512:
        raise ValueError("frozen ResNet18 cache embedding_dim must be 512")
    output.mkdir(parents=True, exist_ok=True)
    embeddings = np.lib.format.open_memmap(output / EMBEDDINGS_NAME, mode="w+", dtype=np.float32, shape=(count, 4, dim))
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
    _atomic_json(output / MANIFEST_NAME, {
        "version": 1, "status": "complete", "total_frames": count, "tactile_keys": list(keys),
        "embedding_dim": dim, "dtype": "float32", "image_size": 224,
        "preprocess_version": PREPROCESS_VERSION, "encoder_identity": _identity(checkpoint),
        "dataset_identity": {"repo_id": getattr(dataset_info, "repo_id", "injected"), "revision": getattr(dataset_info, "revision", None)},
    })
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()
    from .config import load_config
    prepare_tactile_cache(load_config(args.config))


if __name__ == "__main__":
    main()
