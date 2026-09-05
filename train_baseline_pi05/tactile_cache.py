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
from tqdm import tqdm

from .config import RIGHT_TACTILE_KEYS, TACTILE_KEYS

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
    def open(cls, path: str | Path, *, tactile_keys: Sequence[str] | None = None, encoder_path: str | Path) -> "TactileEmbeddingCache":
        cache_dir = Path(path)
        metadata = json.loads((cache_dir / MANIFEST_NAME).read_text(encoding="utf-8"))
        cached_keys = tuple(metadata.get("tactile_keys", ()))
        if (
            metadata.get("status") != "complete"
            or cached_keys not in (TACTILE_KEYS, RIGHT_TACTILE_KEYS)
            or (tactile_keys is not None and cached_keys != tuple(tactile_keys))
        ):
            raise ValueError("tactile cache manifest has an invalid status or key order")
        if metadata.get("preprocess_version") != PREPROCESS_VERSION or metadata.get("encoder_identity") != _identity(Path(encoder_path)):
            raise ValueError("tactile cache preprocessing or encoder identity does not match")
        embeddings = np.load(cache_dir / EMBEDDINGS_NAME, mmap_mode="r", allow_pickle=False)
        shape = (int(metadata["total_frames"]), len(cached_keys), int(metadata["embedding_dim"]))
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
    import jax
    import jax.numpy as jnp

    variables = load_tactile_encoder(checkpoint).params.get("tactile_resnet")
    if not isinstance(variables, dict):
        raise ValueError("encoder checkpoint must contain the shared ResNet18 subtree")

    @jax.jit
    def forward(variables, images):
        # Compile the entire frozen network once per batch shape; weights remain
        # arguments rather than multi-megabyte constants embedded in the graph.
        return encode_resnet18(variables, images, train=False)[0]

    def encode(images: np.ndarray) -> np.ndarray:
        result = forward(variables, jnp.asarray(images))
        return np.asarray(result, dtype=np.float32)
    return encode


def _raw_image_to_unit(image: Any) -> np.ndarray:
    """Keep legacy float-image resize semantics without constructing a Torch tensor."""
    from .tactile_encoder.preprocess import parse_image_to_unit

    array = np.asarray(image)
    if array.dtype == np.uint8:
        array = array.astype(np.float32) / np.float32(255)
    return parse_image_to_unit(array, image_size=224)


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
    """Encode the selected current tactile frames into RMS-normalized float32 tokens."""
    print("[Tactile cache] Loading dataset and checking cache...", flush=True)
    deps = dict(dependencies or {})
    tactile = getattr(config, "tactile", config)
    dataset_info = getattr(config, "dataset", config)
    cache_info = getattr(config, "cache", config)
    output, checkpoint = Path(cache_info.tactile_root), Path(tactile.encoder_checkpoint)
    batch_size = int(getattr(cache_info, "tactile_batch_size", 32))
    if batch_size <= 0:
        raise ValueError("cache.tactile_batch_size must be positive")
    keys = tuple(getattr(getattr(config, "decoder", config), "tactile_keys", TACTILE_KEYS))
    if keys not in (TACTILE_KEYS, RIGHT_TACTILE_KEYS):
        raise ValueError("tactile keys must use canonical right-two or four-current-frame order")
    dataset = deps.get("dataset")
    if dataset is None:
        from .runtime_path import activate_vendored_lerobot
        activate_vendored_lerobot()
        from lerobot.datasets import LeRobotDataset
        dataset = LeRobotDataset(
            dataset_info.repo_id, root=dataset_info.root,
            revision=getattr(dataset_info, "revision", None), visual_keys=list(keys),
        )
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
            print(f"[Tactile cache] Reusing complete cache: {count:,} frames at {output}", flush=True)
            return output
        if any(entry.name != WRITER_LOCK_NAME for entry in output.iterdir()):
            raise ValueError("tactile cache root contains files without a matching complete manifest")
        temporary = output / f".{EMBEDDINGS_NAME}.{uuid.uuid4().hex}.npy"
        published = manifest_published = False
        print(f"[Tactile cache] Loading encoder: {checkpoint}", flush=True)
        encoder = deps.get("encoder") or _default_encoder(checkpoint)
        from .tactile_encoder.preprocess import parse_image_to_unit
        # Embedded image datasets already contain PIL images in uint8. Avoid the
        # costly PIL -> Torch float CHW -> NumPy -> uint8 roundtrip. Video-backed
        # sensors retain LeRobot's timestamp-aware frame decoding path.
        image_rows = None
        if set(keys).issubset(getattr(getattr(dataset, "meta", None), "image_keys", ())):
            image_rows = dataset.hf_dataset.select_columns(list(keys)).with_format(None)
        embeddings = np.lib.format.open_memmap(temporary, mode="w+", dtype=np.float32, shape=(count, len(keys), dim))
        print(f"[Tactile cache] Encoding {count:,} frames, batch size {batch_size}; the first batch includes JAX compilation.", flush=True)
        with tqdm(total=count, desc="Tactile cache", unit="frame", mininterval=1.0, dynamic_ncols=True, disable=False) as progress:
            for start in range(0, count, batch_size):
                end = min(start + batch_size, count)
                if image_rows is not None:
                    columns = image_rows[start:end]
                    images = np.stack([
                        _raw_image_to_unit(columns[key][index])
                        for index in range(end - start) for key in keys
                    ])
                else:
                    samples = [dataset[index] for index in range(start, end)]
                    images = np.stack([
                        parse_image_to_unit(sample[key], image_size=224)
                        for sample in samples for key in keys
                    ])
                tokens = np.asarray(encoder(images), dtype=np.float32)
                expected_shape = ((end - start) * len(keys), dim)
                if tokens.shape != expected_shape or not np.isfinite(tokens).all():
                    raise ValueError(f"shared ResNet18 encoder must return finite {expected_shape} tokens")
                tokens = tokens.reshape(end - start, len(keys), dim)
                rms = np.sqrt(np.mean(np.square(tokens), axis=-1, keepdims=True))
                if np.any(rms == 0):
                    raise ValueError("tactile encoder produced a zero RMS token")
                embeddings[start:end] = tokens / rms
                progress.update(end - start)
        print("[Tactile cache] Flushing embeddings and publishing cache...", flush=True)
        embeddings.flush()
        del embeddings
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, embeddings_path)
        published = True
        _atomic_json(manifest_path, contract)
        manifest_published = True
        print(f"[Tactile cache] Complete: {output}", flush=True)
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
