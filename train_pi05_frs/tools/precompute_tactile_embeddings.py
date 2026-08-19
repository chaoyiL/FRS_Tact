#!/usr/bin/env python
"""Precompute four frozen ResNet tactile embeddings for every dataset frame."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import os
from pathlib import Path
import time
from typing import Any

import numpy as np

from train_pi05_frs.tools.train_frs import (
    DEFAULT_CONFIG,
    load_config,
    resolve_local_path,
    resolved_dataset_sources,
    validate_config,
)


def _start_cpu_only_workers(loader: Any, num_workers: int):
    if num_workers <= 0:
        return iter(loader)
    saved = {
        "JAX_PLATFORMS": os.environ.get("JAX_PLATFORMS"),
        "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "JAX_SKIP_CUDA_CONSTRAINTS_CHECK": os.environ.get("JAX_SKIP_CUDA_CONSTRAINTS_CHECK"),
    }
    os.environ.update(
        JAX_PLATFORMS="cpu",
        CUDA_VISIBLE_DEVICES="",
        JAX_SKIP_CUDA_CONSTRAINTS_CHECK="1",
    )
    try:
        return iter(loader)
    finally:
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _scalar(value: Any) -> int:
    if hasattr(value, "detach"):
        return int(value.detach().cpu().reshape(()).item())
    return int(np.asarray(value).reshape(()).item())


def _absolute_frame_index(dataset: Any, sample: Mapping[str, Any]) -> int:
    episode_index = _scalar(sample["episode_index"])
    frame_index = _scalar(sample["frame_index"])
    episode = dataset.meta.episodes[episode_index]
    return int(episode["dataset_from_index"]) + frame_index


class _TactileFrameDataset:
    def __init__(self, dataset: Any, *, tactile_keys: Sequence[str], image_size: int):
        self.dataset = dataset
        self.tactile_keys = tuple(tactile_keys)
        self.image_size = int(image_size)

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> tuple[int, np.ndarray]:
        from train_encoder.utils.image_dataset import parse_image_to_uint8

        sample = self.dataset[index]
        images = np.stack(
            [
                parse_image_to_uint8(sample[key], image_size=self.image_size)
                for key in self.tactile_keys
            ],
            axis=0,
        )
        return _absolute_frame_index(self.dataset, sample), images


def _cache_settings(
    config: Mapping[str, Any], args: argparse.Namespace
) -> tuple[Path, np.dtype[Any]]:
    cache_config = config["tactile_embedding_cache"]
    root = args.output_root or cache_config["root"]
    dtype = np.dtype(args.dtype or cache_config.get("dtype", "float16"))
    if dtype not in (np.dtype(np.float16), np.dtype(np.float32)):
        raise ValueError(f"embedding cache only supports float16/float32, got {dtype}")
    return resolve_local_path(str(root)), dtype


def _resolved_source_config(config: Mapping[str, Any]) -> dict[str, Any]:
    return {**config, "datasets": resolved_dataset_sources(config["datasets"])}


def _open_existing_embeddings(
    path: Path,
    *,
    shape: tuple[int, ...],
    dtype: str | np.dtype[Any],
) -> np.memmap:
    if not path.is_file():
        raise FileNotFoundError(f"tactile embeddings array is missing: {path}")
    try:
        embeddings = np.lib.format.open_memmap(path, mode="r+")
    except (OSError, ValueError) as error:
        raise ValueError(f"cannot open tactile embeddings array {path}: {error}") from error
    expected_dtype = np.dtype(dtype)
    if embeddings.shape != shape or embeddings.dtype != expected_dtype:
        raise ValueError(
            "tactile embeddings shape/dtype mismatch: "
            f"expected {shape}/{expected_dtype}, got {embeddings.shape}/{embeddings.dtype}"
        )
    return embeddings


def _validate_cache_file_pair(
    metadata_path: Path,
    embeddings_path: Path,
    *,
    overwrite: bool,
) -> None:
    if overwrite:
        return
    if metadata_path.exists() != embeddings_path.exists():
        raise ValueError(
            "inconsistent tactile cache files: metadata.json and embeddings.npy "
            "must either both exist or both be absent; pass --overwrite to replace them"
        )


def _validate_cache_progress(*, status: object, completed: int, total: int) -> None:
    consistent = (
        status in ("incomplete", "complete")
        and 0 <= completed <= total
        and ((status == "complete") == (completed == total))
    )
    if not consistent:
        raise ValueError(
            f"inconsistent cache progress: status={status!r}, "
            f"completed_frames={completed}, total_frames={total}"
        )


def _precompute_source(
    *,
    source: Any,
    model_config: Mapping[str, Any],
    encoder_bundle: Any,
    cache_root: Path,
    cache_dtype: np.dtype[Any],
    batch_size: int,
    num_workers: int,
    prefetch_factor: int,
    video_backend: str | None,
    flush_every: int,
    overwrite: bool,
) -> None:
    import jax
    import jax.numpy as jnp
    from torch.utils.data import DataLoader, Subset

    from lerobot.datasets import LeRobotDataset, LeRobotDatasetMetadata
    from lerobot.datasets.dataset_sources import resolve_source_visual_keys
    from lerobot.datasets.tactile_cache import (
        TACTILE_EMBEDDINGS_NAME,
        TACTILE_METADATA_NAME,
        atomic_write_json,
        create_tactile_cache_metadata,
        load_tactile_cache_metadata,
        tactile_cache_dir,
    )
    from train_encoder.utils.model import encode_resnet18, tactile_clip_config_from_dict

    metadata = LeRobotDatasetMetadata(
        repo_id=source.repo_id,
        root=source.root,
        revision=source.revision,
    )
    model_tactile_keys = tuple(model_config["tactile_keys"])
    source_tactile_keys = tuple(
        resolve_source_visual_keys(model_tactile_keys, source.rename_map, metadata.camera_keys)
    )
    encoder_config = tactile_clip_config_from_dict(
        encoder_bundle.metadata["tactile_clip_config"]
    )
    embedding_dim = int(model_config.get("tactile_embedding_dim", encoder_config.embedding_dim))
    image_size = int(model_config.get("tactile_image_size", encoder_config.tactile_image_size))
    if embedding_dim != int(encoder_config.embedding_dim):
        raise ValueError(f"{source.repo_id}: embedding dimension does not match encoder")
    if image_size != int(encoder_config.tactile_image_size):
        raise ValueError(f"{source.repo_id}: tactile image size does not match encoder")

    output_dir = tactile_cache_dir(cache_root, source.repo_id)
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = output_dir / TACTILE_METADATA_NAME
    embeddings_path = output_dir / TACTILE_EMBEDDINGS_NAME
    _validate_cache_file_pair(metadata_path, embeddings_path, overwrite=overwrite)
    expected_metadata = create_tactile_cache_metadata(
        repo_id=source.repo_id,
        revision=metadata.revision,
        dataset_root=metadata.root,
        total_frames=metadata.total_frames,
        tactile_keys=model_tactile_keys,
        source_tactile_keys=source_tactile_keys,
        embedding_dim=embedding_dim,
        image_size=image_size,
        dtype=cache_dtype,
        encoder_path=resolve_local_path(str(model_config["tactile_encoder_path"])),
    )

    completed = 0
    if overwrite:
        embeddings_path.unlink(missing_ok=True)
        metadata_path.unlink(missing_ok=True)
    if metadata_path.exists():
        existing = load_tactile_cache_metadata(output_dir)
        comparable = set(expected_metadata) - {"status", "completed_frames"}
        mismatches = {
            key: (existing.get(key), expected_metadata.get(key))
            for key in comparable
            if existing.get(key) != expected_metadata.get(key)
        }
        if mismatches:
            raise ValueError(
                f"{source.repo_id}: cache provenance differs: {mismatches}; "
                "choose another directory or pass --overwrite"
            )
        completed = int(existing.get("completed_frames", 0))
        _validate_cache_progress(
            status=existing.get("status"),
            completed=completed,
            total=metadata.total_frames,
        )
        embeddings = _open_existing_embeddings(
            embeddings_path,
            shape=(metadata.total_frames, len(model_tactile_keys), embedding_dim),
            dtype=cache_dtype,
        )
        if existing.get("status") == "complete" and completed == metadata.total_frames:
            print(f"already complete: {source.repo_id} -> {output_dir}", flush=True)
            return
    else:
        embeddings = np.lib.format.open_memmap(
            embeddings_path,
            mode="w+",
            dtype=cache_dtype,
            shape=(metadata.total_frames, len(model_tactile_keys), embedding_dim),
        )
        atomic_write_json(metadata_path, expected_metadata)

    dataset = LeRobotDataset(
        repo_id=source.repo_id,
        root=metadata.root,
        revision=metadata.revision,
        visual_keys=source_tactile_keys,
        video_backend=video_backend,
        return_uint8=True,
        download_videos=True,
    )
    frames: Any = _TactileFrameDataset(
        dataset, tactile_keys=source_tactile_keys, image_size=image_size
    )
    if completed:
        frames = Subset(frames, range(completed, len(frames)))
    loader_options: dict[str, Any] = {
        "dataset": frames,
        "batch_size": batch_size,
        "shuffle": False,
        "num_workers": num_workers,
        "persistent_workers": num_workers > 0,
    }
    if num_workers > 0:
        loader_options.update(
            prefetch_factor=prefetch_factor,
            multiprocessing_context="spawn",
        )
    iterator = _start_cpu_only_workers(DataLoader(**loader_options), num_workers)
    resnet_params = encoder_bundle.params["tactile_resnet"]

    @jax.jit
    def encode(images: jax.Array) -> jax.Array:
        values, _ = encode_resnet18(
            resnet_params, images, train=False, embedding_dim=embedding_dim
        )
        return values

    started = time.perf_counter()
    last_index = completed
    for batch_number, (frame_indices, image_batch) in enumerate(iterator, start=1):
        images = np.asarray(image_batch.numpy(), dtype=np.uint8)
        batch_count, token_count = images.shape[:2]
        flat = images.reshape((batch_count * token_count,) + images.shape[2:])
        values = encode(jnp.asarray(flat, dtype=jnp.float32) / 255.0)
        values = np.asarray(jax.device_get(values), dtype=np.float32).reshape(
            batch_count, token_count, embedding_dim
        )
        indices = np.asarray(frame_indices, dtype=np.int64)
        embeddings[indices] = values.astype(cache_dtype, copy=False)
        last_index = int(indices[-1]) + 1
        if batch_number % flush_every == 0 or last_index >= metadata.total_frames:
            embeddings.flush()
            progress = dict(expected_metadata)
            progress["completed_frames"] = last_index
            progress["status"] = "complete" if last_index >= metadata.total_frames else "incomplete"
            atomic_write_json(metadata_path, progress)
            elapsed = max(time.perf_counter() - started, 1e-9)
            print(
                f"{source.repo_id}: {last_index}/{metadata.total_frames} "
                f"{(last_index - completed) / elapsed:.1f} frames/s",
                flush=True,
            )
    if last_index != metadata.total_frames:
        raise RuntimeError(
            f"{source.repo_id}: precompute ended early {last_index}/{metadata.total_frames}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--dtype", choices=("float16", "float32"))
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--prefetch-factor", type=int)
    parser.add_argument("--video-backend")
    parser.add_argument("--flush-every", type=int)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    validate_config(config, check_paths=True)
    cache_config = config["tactile_embedding_cache"]
    model_config = config["model"]
    if not cache_config["enabled"]:
        raise ValueError("config.tactile_embedding_cache.enabled must be true")
    if not model_config["use_tactile_encoder"]:
        raise ValueError("config.model.use_tactile_encoder must be true")
    batch_size = args.batch_size or cache_config.get("precompute_batch_size", 128)
    num_workers = (
        args.num_workers
        if args.num_workers is not None
        else cache_config.get("precompute_num_workers", 4)
    )
    prefetch_factor = args.prefetch_factor or cache_config.get("precompute_prefetch_factor", 2)
    flush_every = args.flush_every or cache_config.get("precompute_flush_every", 20)
    if min(batch_size, prefetch_factor, flush_every) <= 0 or num_workers < 0:
        raise ValueError("batch/prefetch/flush must be positive and workers non-negative")
    cache_root, cache_dtype = _cache_settings(config, args)

    # Heavy target-encoder and dataset imports happen only after complete schema validation.
    import jax
    from lerobot.datasets.dataset_sources import parse_dataset_sources
    from train_encoder.utils.checkpoint import load_tactile_encoder

    encoder_bundle = load_tactile_encoder(
        resolve_local_path(str(model_config["tactile_encoder_path"]))
    )
    if "tactile_resnet" not in encoder_bundle.params:
        raise KeyError("tactile encoder checkpoint is missing tactile_resnet")
    print(f"JAX devices={jax.devices()}", flush=True)
    source_config = _resolved_source_config(config)
    for source in parse_dataset_sources(source_config):
        _precompute_source(
            source=source,
            model_config=model_config,
            encoder_bundle=encoder_bundle,
            cache_root=cache_root,
            cache_dtype=cache_dtype,
            batch_size=int(batch_size),
            num_workers=int(num_workers),
            prefetch_factor=int(prefetch_factor),
            video_backend=args.video_backend
            or cache_config.get("precompute_video_backend", "torchcodec"),
            flush_every=int(flush_every),
            overwrite=args.overwrite,
        )


if __name__ == "__main__":
    main()
