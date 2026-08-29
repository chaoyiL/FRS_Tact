#!/usr/bin/env python3
"""Precompute four 512-D tactile features from pick_tube LeRobot v2.1 Parquet."""

from __future__ import annotations

import argparse
import io
import json
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pyarrow.parquet as pq
import yaml
from PIL import Image

from reactive_diffusion_policy.model.tactile_encoder_jax import load_tactile_encoder
from reactive_diffusion_policy.model.tactile_encoder_jax import encode_resnet18


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("pick_tube_tactile_cache_0809.yaml"))
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument(
        "--datasets",
        nargs="+",
        help=(
            "Dataset names or repo IDs to process. When provided, --dataset-root "
            "must point to the directory containing their local folders. Bare "
            "names are interpreted as KaiyueChen/<name>."
        ),
    )
    parser.add_argument("--cache-root", type=Path)
    parser.add_argument("--encoder-path", type=Path)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--prefetch-factor", type=int)
    parser.add_argument("--flush-every", type=int)
    parser.add_argument("--max-episodes-per-dataset", type=int)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--resume-after-episodes", type=int)
    return parser.parse_args()


def decode_image(value: object) -> np.ndarray:
    payload = value.get("bytes") if isinstance(value, dict) else value
    if not isinstance(payload, (bytes, bytearray, memoryview)):
        raise ValueError(f"unsupported LeRobot image value: {type(value)!r}")
    with Image.open(io.BytesIO(payload)) as image:
        return np.asarray(image.convert("RGB"), dtype=np.uint8)


def load_episodes(dataset_dir: Path) -> list[dict]:
    with (dataset_dir / "meta" / "episodes.jsonl").open(encoding="utf-8") as file:
        episodes = [json.loads(line) for line in file]
    return sorted(episodes, key=lambda item: int(item["episode_index"]))


def iter_decoded_batches(
    columns: list[list[object]],
    *,
    batch_size: int,
    executor: ThreadPoolExecutor,
    prefetch_factor: int,
):
    """Decode bounded batches ahead of GPU execution while preserving order."""
    if not columns:
        return
    frame_count = len(columns[0])
    if any(len(column) != frame_count for column in columns):
        raise ValueError("tactile image columns have different frame counts")
    pending = deque()
    next_start = 0

    def submit_one(start: int):
        end = min(start + batch_size, frame_count)
        values = [
            columns[sensor][frame]
            for frame in range(start, end)
            for sensor in range(len(columns))
        ]
        futures = [executor.submit(decode_image, value) for value in values]
        return start, end, futures

    while next_start < frame_count and len(pending) < prefetch_factor:
        pending.append(submit_one(next_start))
        next_start += batch_size
    while pending:
        start, end, futures = pending.popleft()
        images = np.stack([future.result() for future in futures])
        if next_start < frame_count:
            pending.append(submit_one(next_start))
            next_start += batch_size
        yield start, end, images


def pad_images(images: np.ndarray, target_count: int) -> np.ndarray:
    """Pad the final batch so JAX compiles one batch shape for the full run."""
    if images.shape[0] == target_count:
        return images
    if not 0 < images.shape[0] < target_count:
        raise ValueError(
            f"cannot pad image batch of {images.shape[0]} to {target_count}"
        )
    padded = np.empty((target_count, *images.shape[1:]), dtype=images.dtype)
    padded[: images.shape[0]] = images
    padded[images.shape[0] :] = images[-1]
    return padded


def main() -> None:
    args = parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    cache_config = config["tactile_embedding_cache"]
    model_config = config["model"]
    if args.datasets is not None:
        if args.dataset_root is None:
            raise ValueError("--dataset-root is required when --datasets is provided")
        config["datasets"] = [
            {
                "repo_id": value if "/" in value else f"KaiyueChen/{value}",
                "root": str(args.dataset_root / value.rsplit("/", 1)[-1]),
            }
            for value in args.datasets
        ]
    if args.dataset_root is not None:
        for source in config["datasets"]:
            source["root"] = str(args.dataset_root / str(source["repo_id"]).rsplit("/", 1)[-1])
    if args.cache_root is not None:
        cache_config["root"] = str(args.cache_root)
    if args.encoder_path is not None:
        model_config["tactile_encoder_path"] = str(args.encoder_path)
    cache_root = Path(cache_config["root"])
    tactile_keys = tuple(model_config["tactile_keys"])
    batch_size = args.batch_size or int(cache_config["precompute_batch_size"])
    num_workers = args.num_workers if args.num_workers is not None else int(cache_config["precompute_num_workers"])
    prefetch_factor = (
        args.prefetch_factor
        if args.prefetch_factor is not None
        else int(cache_config.get("precompute_prefetch_factor", 2))
    )
    flush_every = (
        args.flush_every
        if args.flush_every is not None
        else int(cache_config.get("precompute_flush_every", 20))
    )
    if batch_size < 1 or num_workers < 1 or prefetch_factor < 1 or flush_every < 1:
        raise ValueError(
            "batch-size, num-workers, prefetch-factor, and flush-every must be positive"
        )
    embedding_dim = int(model_config["tactile_embedding_dim"])
    image_size = int(model_config["tactile_image_size"])

    bundle = load_tactile_encoder(model_config["tactile_encoder_path"])
    checkpoint_config = bundle.metadata["tactile_clip_config"]
    if embedding_dim != int(checkpoint_config["embedding_dim"]) or image_size != int(checkpoint_config["tactile_image_size"]):
        raise ValueError("config and encoder embedding/image dimensions differ")
    resnet_params = bundle.params["tactile_resnet"]

    @jax.jit
    def encode(images: jax.Array) -> jax.Array:
        values, _ = encode_resnet18(
            resnet_params,
            images,
            train=False,
            embedding_dim=embedding_dim,
        )
        return values

    print(f"JAX devices={jax.devices()}", flush=True)
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        for source in config["datasets"]:
            repo_id = str(source["repo_id"])
            dataset_dir = Path(source["root"])
            episodes = load_episodes(dataset_dir)
            if args.max_episodes_per_dataset is not None:
                episodes = episodes[: args.max_episodes_per_dataset]
            total_frames = sum(int(item["length"]) for item in episodes)
            output_dir = cache_root.joinpath(*repo_id.split("/"))
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / "embeddings.npy"
            metadata_path = output_dir / "metadata.json"
            progress_path = output_dir / "progress.json"
            if output_path.exists() and metadata_path.exists() and not args.overwrite:
                print(f"exists, skipping: {output_path}", flush=True)
                continue

            completed_episodes = 0
            if output_path.exists() and not args.overwrite:
                if progress_path.exists():
                    completed_episodes = int(json.loads(progress_path.read_text())["completed_episodes"])
                elif args.resume_after_episodes is not None:
                    completed_episodes = args.resume_after_episodes
                    args.resume_after_episodes = None
                else:
                    raise ValueError(
                        f"partial cache has no progress marker: {output_path}; "
                        "pass --resume-after-episodes or --overwrite"
                    )
                write_index = sum(int(item["length"]) for item in episodes[:completed_episodes])
                embeddings = np.lib.format.open_memmap(output_path, mode="r+")
                expected_shape = (total_frames, len(tactile_keys), embedding_dim)
                if embeddings.shape != expected_shape or embeddings.dtype != np.float16:
                    raise ValueError(
                        f"{output_path}: expected {expected_shape} float16, "
                        f"got {embeddings.shape} {embeddings.dtype}"
                    )
                print(
                    f"resuming: {repo_id} after episode {completed_episodes}, frame {write_index}",
                    flush=True,
                )
            else:
                embeddings = np.lib.format.open_memmap(
                    output_path,
                    mode="w+",
                    dtype=np.float16,
                    shape=(total_frames, len(tactile_keys), embedding_dim),
                )
                write_index = 0
            started = time.perf_counter()
            initial_write_index = write_index
            timings = {
                "parquet_read_seconds": 0.0,
                "parquet_materialize_seconds": 0.0,
                "decode_wait_seconds": 0.0,
                "gpu_encode_seconds": 0.0,
                "cache_write_seconds": 0.0,
            }
            for episode_number, episode in enumerate(
                episodes[completed_episodes:], start=completed_episodes + 1
            ):
                episode_index = int(episode["episode_index"])
                episode_path = (
                    dataset_dir
                    / "data"
                    / f"chunk-{episode_index // 1000:03d}"
                    / f"episode_{episode_index:06d}.parquet"
                )
                stage_started = time.perf_counter()
                table = pq.read_table(episode_path, columns=list(tactile_keys))
                timings["parquet_read_seconds"] += time.perf_counter() - stage_started
                if table.num_rows != int(episode["length"]):
                    raise ValueError(f"{episode_path}: episode length mismatch")
                stage_started = time.perf_counter()
                columns = [table[key].to_pylist() for key in tactile_keys]
                timings["parquet_materialize_seconds"] += (
                    time.perf_counter() - stage_started
                )
                batches = iter_decoded_batches(
                    columns,
                    batch_size=batch_size,
                    executor=executor,
                    prefetch_factor=prefetch_factor,
                )
                while True:
                    stage_started = time.perf_counter()
                    try:
                        start, end, images = next(batches)
                    except StopIteration:
                        break
                    timings["decode_wait_seconds"] += time.perf_counter() - stage_started
                    if images.shape[1:] != (image_size, image_size, 3):
                        raise ValueError(f"{episode_path}: tactile image shape {images.shape}")
                    count = end - start
                    valid_image_count = count * len(tactile_keys)
                    images = pad_images(images, batch_size * len(tactile_keys))
                    stage_started = time.perf_counter()
                    encoded_device = encode(
                        jnp.asarray(images, dtype=jnp.float32) * (1.0 / 255.0)
                    )
                    encoded_device.block_until_ready()
                    encoded = np.asarray(
                        jax.device_get(encoded_device), dtype=np.float32
                    )[:valid_image_count].reshape(
                        count, len(tactile_keys), embedding_dim
                    )
                    timings["gpu_encode_seconds"] += time.perf_counter() - stage_started
                    stage_started = time.perf_counter()
                    embeddings[write_index : write_index + count] = encoded.astype(np.float16)
                    write_index += count
                    timings["cache_write_seconds"] += time.perf_counter() - stage_started
                progress_path.write_text(
                    json.dumps({"completed_episodes": episode_number, "frames": write_index}) + "\n",
                    encoding="utf-8",
                )
                if episode_number % flush_every == 0 or episode_number == len(episodes):
                    embeddings.flush()
                    elapsed = max(time.perf_counter() - started, 1e-9)
                    processed_frames = write_index - initial_write_index
                    print(
                        f"{repo_id}: episode {episode_number}/{len(episodes)} "
                        f"frames {write_index}/{total_frames} "
                        f"({processed_frames / elapsed:.1f} frames/s)",
                        flush=True,
                    )

            metadata = {
                "repo_id": repo_id,
                "total_frames": total_frames,
                "tactile_keys": list(tactile_keys),
                "shape": [total_frames, len(tactile_keys), embedding_dim],
                "dtype": "float16",
                "encoder_path": str(Path(model_config["tactile_encoder_path"]).resolve()),
                "batch_size": batch_size,
                "num_workers": num_workers,
                "prefetch_factor": prefetch_factor,
                "timings": {
                    **timings,
                    "total_seconds": time.perf_counter() - started,
                },
            }
            embeddings.flush()
            metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
            progress_path.unlink(missing_ok=True)
            print(f"completed: {output_path}", flush=True)


if __name__ == "__main__":
    main()
