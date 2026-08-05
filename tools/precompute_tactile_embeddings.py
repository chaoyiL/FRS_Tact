#!/usr/bin/env python
"""为 VT-SmolVLA/FRS 预计算每一帧的四路 tactile ResNet embedding。"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import jax
import jax.numpy as jnp
import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Dataset, Subset

from lerobot.datasets import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.policies.smolvla_jax.data import parse_dataset_sources, resolve_source_visual_keys
from lerobot.policies.smolvla_jax.tactile_cache import (
    TACTILE_EMBEDDINGS_NAME,
    TACTILE_METADATA_NAME,
    atomic_write_json,
    create_tactile_cache_metadata,
    load_tactile_cache_metadata,
    tactile_cache_dir,
)
from tactile_encoder.utils.checkpoint import load_tactile_encoder
from tactile_encoder.utils.image_dataset import parse_image_to_uint8
from tactile_encoder.utils.model import encode_resnet18, tactile_clip_config_from_dict

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "train_vtsmolvla_jax.yaml"


def _scalar(value: Any) -> int:
    if isinstance(value, torch.Tensor):
        return int(value.detach().cpu().reshape(()).item())
    return int(np.asarray(value).reshape(()).item())


def _absolute_frame_index(dataset: LeRobotDataset, sample: Mapping[str, Any]) -> int:
    episode_index = _scalar(sample["episode_index"])
    frame_index = _scalar(sample["frame_index"])
    episode = dataset.meta.episodes[episode_index]
    return int(episode["dataset_from_index"]) + frame_index


class _TactileFrameDataset(Dataset):
    def __init__(
        self,
        dataset: LeRobotDataset,
        *,
        tactile_keys: Sequence[str],
        image_size: int,
    ):
        self.dataset = dataset
        self.tactile_keys = tuple(tactile_keys)
        self.image_size = int(image_size)

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> tuple[int, np.ndarray]:
        sample = self.dataset[index]
        images = np.stack(
            [
                parse_image_to_uint8(sample[key], image_size=self.image_size)
                for key in self.tactile_keys
            ],
            axis=0,
        )
        return _absolute_frame_index(self.dataset, sample), images


def _load_config(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as file:
        config = yaml.safe_load(file) or {}
    if not isinstance(config, dict):
        raise ValueError(f"配置根节点必须是 mapping：{path}")
    return config


def _cache_settings(config: Mapping[str, Any], args: argparse.Namespace) -> tuple[Path, np.dtype]:
    cache_config = config.get("tactile_embedding_cache") or {}
    if not isinstance(cache_config, Mapping):
        raise ValueError("tactile_embedding_cache 必须是 mapping")
    root = args.output_root or cache_config.get("root")
    if not root:
        raise ValueError("请在 YAML 设置 tactile_embedding_cache.root 或传 --output-root")
    dtype = np.dtype(args.dtype or cache_config.get("dtype", "float16"))
    if dtype not in (np.dtype(np.float16), np.dtype(np.float32)):
        raise ValueError(f"embedding cache 只支持 float16/float32，收到 {dtype}")
    return Path(root), dtype


def _precompute_source(
    *,
    source,
    model_config: Mapping[str, Any],
    encoder_bundle,
    cache_root: Path,
    cache_dtype: np.dtype,
    batch_size: int,
    num_workers: int,
    prefetch_factor: int,
    video_backend: str | None,
    flush_every: int,
    overwrite: bool,
) -> None:
    metadata = LeRobotDatasetMetadata(
        repo_id=source.repo_id,
        root=source.root,
        revision=source.revision,
    )
    model_tactile_keys = tuple(model_config["tactile_keys"])
    source_tactile_keys = tuple(
        resolve_source_visual_keys(
            model_tactile_keys,
            source.rename_map,
            metadata.camera_keys,
        )
    )
    tactile_config = tactile_clip_config_from_dict(
        encoder_bundle.metadata["tactile_clip_config"]
    )
    embedding_dim = int(model_config.get("tactile_embedding_dim", tactile_config.embedding_dim))
    image_size = int(model_config.get("tactile_image_size", tactile_config.tactile_image_size))
    if embedding_dim != int(tactile_config.embedding_dim):
        raise ValueError(
            f"{source.repo_id}: YAML embedding_dim={embedding_dim} 与 encoder="
            f"{tactile_config.embedding_dim} 不一致"
        )
    if image_size != int(tactile_config.tactile_image_size):
        raise ValueError(
            f"{source.repo_id}: YAML tactile_image_size={image_size} 与 encoder="
            f"{tactile_config.tactile_image_size} 不一致"
        )

    output_dir = tactile_cache_dir(cache_root, source.repo_id)
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = output_dir / TACTILE_METADATA_NAME
    embeddings_path = output_dir / TACTILE_EMBEDDINGS_NAME
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
        encoder_path=model_config["tactile_encoder_path"],
    )

    completed = 0
    if overwrite:
        embeddings_path.unlink(missing_ok=True)
        metadata_path.unlink(missing_ok=True)
    if metadata_path.exists():
        existing = load_tactile_cache_metadata(output_dir)
        comparable_keys = set(expected_metadata) - {"status", "completed_frames"}
        mismatches = {
            key: (existing.get(key), expected_metadata.get(key))
            for key in comparable_keys
            if existing.get(key) != expected_metadata.get(key)
        }
        if mismatches:
            raise ValueError(
                f"{source.repo_id}: 现有 cache 与当前输入不一致：{mismatches}。"
                "请换目录或使用 --overwrite。"
            )
        completed = int(existing.get("completed_frames", 0))
        if existing.get("status") == "complete" and completed == metadata.total_frames:
            print(f"已完成，跳过：{source.repo_id} -> {output_dir}", flush=True)
            return
        embeddings = np.lib.format.open_memmap(embeddings_path, mode="r+")
        print(
            f"继续预计算：{source.repo_id} frame={completed}/{metadata.total_frames}",
            flush=True,
        )
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
    frame_dataset: Dataset = _TactileFrameDataset(
        dataset,
        tactile_keys=source_tactile_keys,
        image_size=image_size,
    )
    if completed:
        frame_dataset = Subset(frame_dataset, range(completed, len(frame_dataset)))
    loader_kwargs: dict[str, Any] = {
        "dataset": frame_dataset,
        "batch_size": batch_size,
        "shuffle": False,
        "num_workers": num_workers,
        "persistent_workers": num_workers > 0,
    }
    if num_workers > 0:
        loader_kwargs["prefetch_factor"] = prefetch_factor
        loader_kwargs["multiprocessing_context"] = "spawn"
    loader = DataLoader(**loader_kwargs)

    resnet_params = encoder_bundle.params["tactile_resnet"]

    @jax.jit
    def encode(images: jax.Array) -> jax.Array:
        values, _ = encode_resnet18(
            resnet_params,
            images,
            train=False,
            embedding_dim=embedding_dim,
        )
        return values

    started = time.perf_counter()
    last_index = completed
    for batch_number, (frame_indices, image_batch) in enumerate(loader, start=1):
        images = np.asarray(image_batch.numpy(), dtype=np.uint8)
        batch_count, token_count = images.shape[:2]
        flat = images.reshape((batch_count * token_count,) + images.shape[2:])
        encoded = encode(jnp.asarray(flat, dtype=jnp.float32) * (1.0 / 255.0))
        encoded = np.asarray(jax.device_get(encoded), dtype=np.float32).reshape(
            batch_count, token_count, embedding_dim
        )
        indices = np.asarray(frame_indices, dtype=np.int64)
        embeddings[indices] = encoded.astype(cache_dtype, copy=False)
        last_index = int(indices[-1]) + 1

        should_flush = batch_number % flush_every == 0 or last_index >= metadata.total_frames
        if should_flush:
            embeddings.flush()
            progress = dict(expected_metadata)
            progress["completed_frames"] = last_index
            progress["status"] = "complete" if last_index >= metadata.total_frames else "incomplete"
            atomic_write_json(metadata_path, progress)
            elapsed = max(time.perf_counter() - started, 1e-9)
            processed = last_index - completed
            rate = processed / elapsed
            remaining = metadata.total_frames - last_index
            print(
                f"{source.repo_id}: {last_index}/{metadata.total_frames} "
                f"{rate:.1f} frames/s ETA={remaining / max(rate, 1e-9) / 60:.1f} min",
                flush=True,
            )

    if last_index != metadata.total_frames:
        raise RuntimeError(
            f"{source.repo_id}: 预计算提前结束 {last_index}/{metadata.total_frames}"
        )
    print(f"完成：{source.repo_id} -> {output_dir}", flush=True)


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
    config = _load_config(args.config)
    cache_config = config.get("tactile_embedding_cache") or {}
    if not isinstance(cache_config, Mapping):
        raise ValueError("tactile_embedding_cache 必须是 mapping")
    batch_size = int(args.batch_size or cache_config.get("precompute_batch_size", 128))
    num_workers = int(
        args.num_workers
        if args.num_workers is not None
        else cache_config.get("precompute_num_workers", 4)
    )
    prefetch_factor = int(
        args.prefetch_factor or cache_config.get("precompute_prefetch_factor", 2)
    )
    video_backend = args.video_backend or cache_config.get(
        "precompute_video_backend", "torchcodec"
    )
    flush_every = int(args.flush_every or cache_config.get("precompute_flush_every", 20))
    if min(batch_size, prefetch_factor, flush_every) <= 0 or num_workers < 0:
        raise ValueError("batch/prefetch/flush 必须为正数，num_workers 不能为负数")
    model_config = config.get("model") or {}
    if not bool(model_config.get("use_tactile_encoder", False)):
        raise ValueError("model.use_tactile_encoder 必须为 true")
    for key in ("tactile_encoder_path", "tactile_keys"):
        if not model_config.get(key):
            raise ValueError(f"model.{key} 是必填项")
    cache_root, cache_dtype = _cache_settings(config, args)
    encoder_bundle = load_tactile_encoder(model_config["tactile_encoder_path"])
    if "tactile_resnet" not in encoder_bundle.params:
        raise KeyError("tactile encoder checkpoint 缺少 tactile_resnet")

    print(f"JAX devices={jax.devices()}", flush=True)
    print(
        f"cache_root={cache_root.resolve()} dtype={cache_dtype.name} "
        f"batch={batch_size} workers={num_workers}",
        flush=True,
    )
    for source in parse_dataset_sources(config):
        _precompute_source(
            source=source,
            model_config=model_config,
            encoder_bundle=encoder_bundle,
            cache_root=cache_root,
            cache_dtype=cache_dtype,
            batch_size=batch_size,
            num_workers=num_workers,
            prefetch_factor=prefetch_factor,
            video_backend=video_backend,
            flush_every=flush_every,
            overwrite=args.overwrite,
        )


if __name__ == "__main__":
    main()
