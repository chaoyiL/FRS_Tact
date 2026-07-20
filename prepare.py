from __future__ import annotations

import argparse
import hashlib
import pathlib
import sys
import time
from collections.abc import Mapping, Sequence
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lerobot.datasets import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.policies.smolvla_jax.data import action_delta_timestamps

from eval_scripts.utils import EvalObservation
from eval_scripts.utils import SmolVLAEvalModel
from eval_scripts.utils import add_eval_data_arguments
from eval_scripts.utils import load_model
from eval_scripts.utils import parse_rename_map
from utils.cache import CACHE_VERSION
from utils.cache import MANIFEST_NAME
from utils.cache import SampleRecord
from utils.cache import atomic_write_json
from utils.cache import create_cache_arrays
from utils.cache import flush_arrays
from utils.cache import limit_records
from utils.cache import load_manifest
from utils.cache import open_cache_arrays
from utils.cache import records_digest
from utils.cache import split_episodes
from utils.source_model import deterministic_noise
from utils.source_model import inversion_mse
from utils.source_model import sample_and_reverse
from utils.source_model import stack_observations


def _as_scalar(value: Any) -> Any:
    value = np.asarray(value)
    if value.shape == ():
        return value.item()
    if value.size == 1:
        return value.reshape(()).item()
    return value


def _episode_bounds(metadata: LeRobotDatasetMetadata, episode_index: int) -> tuple[int, int]:
    if episode_index < 0 or episode_index >= metadata.total_episodes:
        raise ValueError(
            f"Episode {episode_index} is out of range for this dataset. "
            f"Available episode indices are 0..{metadata.total_episodes - 1}."
        )
    episode = metadata.episodes[episode_index]
    start = int(_as_scalar(episode["dataset_from_index"]))
    end = int(_as_scalar(episode["dataset_to_index"]))
    if end <= start:
        raise ValueError(f"Episode {episode_index} has an empty frame range [{start}, {end}).")
    return start, end


def _indices_for_episode(metadata: LeRobotDatasetMetadata, episode_index: int) -> tuple[int, ...]:
    start, end = _episode_bounds(metadata, episode_index)
    return tuple(range(start, end))


def build_records(
    metadata: LeRobotDatasetMetadata,
    *,
    val_fraction: float,
    split_seed: int,
    frame_stride: int,
    max_episodes: int | None,
    max_samples: int | None,
) -> tuple[list[SampleRecord], tuple[int, ...], tuple[int, ...]]:
    if frame_stride <= 0:
        raise ValueError(f"frame_stride must be positive, got {frame_stride}.")
    episode_count = int(metadata.total_episodes)
    if episode_count < 2:
        raise ValueError("At least two episodes are required for an episode-disjoint train/validation split.")
    episodes = list(range(episode_count))
    if max_episodes is not None:
        if max_episodes < 2:
            raise ValueError("max_episodes must be at least 2 for an episode-disjoint split.")
        episodes = episodes[:max_episodes]
    train_episodes, val_episodes = split_episodes(episodes, val_fraction=val_fraction, seed=split_seed)
    val_set = set(val_episodes)

    records: list[SampleRecord] = []
    for episode_index in episodes:
        split = "val" if episode_index in val_set else "train"
        dataset_indices = _indices_for_episode(metadata, episode_index)[::frame_stride]
        records.extend(SampleRecord(int(index), episode_index, split) for index in dataset_indices)
    records = limit_records(records, max_samples=max_samples, seed=split_seed)
    if not records:
        raise ValueError("Dataset selection produced no samples.")
    return records, train_episodes, val_episodes


def _configuration(
    *,
    checkpoint_dir: pathlib.Path,
    dataset_repo_id: str,
    dataset_root: pathlib.Path | None,
    dataset_revision: str | None,
    action_key: str | None,
    rename_map: Mapping[str, str] | None,
    model_sample_steps: int,
    reverse_steps: int,
    reverse_solver: str,
    inference_seed: int,
    split_seed: int,
    val_fraction: float,
    frame_stride: int,
    max_episodes: int | None,
    max_samples: int | None,
) -> dict[str, Any]:
    return {
        "checkpoint_dir": str(checkpoint_dir.resolve()),
        "checkpoint_fingerprint": _checkpoint_fingerprint(checkpoint_dir),
        "dataset_repo_id": dataset_repo_id,
        "dataset_root": str(dataset_root.resolve()) if dataset_root is not None else None,
        "dataset_revision": dataset_revision,
        "action_key": action_key,
        "rename_map": dict(rename_map) if rename_map is not None else None,
        "model_sample_steps": model_sample_steps,
        "reverse_steps": reverse_steps,
        "reverse_solver": reverse_solver,
        "inference_seed": inference_seed,
        "split_seed": split_seed,
        "val_fraction": val_fraction,
        "frame_stride": frame_stride,
        "max_episodes": max_episodes,
        "max_samples": max_samples,
    }


def _checkpoint_fingerprint(checkpoint_dir: pathlib.Path) -> str:
    checkpoint_dir = checkpoint_dir.expanduser().resolve()
    if checkpoint_dir.name == "params":
        checkpoint_dir = checkpoint_dir.parent

    digest = hashlib.sha256()
    candidates: list[pathlib.Path] = []
    params_dir = checkpoint_dir / "params"
    model_file = checkpoint_dir / "model.safetensors"
    if params_dir.is_dir():
        candidates.extend(sorted(path for path in params_dir.rglob("*") if path.is_file()))
    elif model_file.is_file():
        candidates.append(model_file)
    else:
        raise FileNotFoundError(
            f"Checkpoint params not found under {checkpoint_dir}: expected params/ or model.safetensors"
        )

    for name in ("config.json", "conversion_manifest.json"):
        path = checkpoint_dir / name
        if path.is_file():
            candidates.append(path)

    for path in candidates:
        stat = path.stat()
        digest.update(str(path.relative_to(checkpoint_dir)).encode())
        digest.update(f":{stat.st_size}:{stat.st_mtime_ns}\n".encode())
    return digest.hexdigest()


def _create_dataset(model: SmolVLAEvalModel, metadata: LeRobotDatasetMetadata) -> LeRobotDataset:
    return LeRobotDataset(
        model.dataset_repo_id,
        root=model.dataset_root,
        revision=model.dataset_revision,
        delta_timestamps=action_delta_timestamps(
            model.action_key,
            model.config.chunk_size,
            metadata.fps,
        ),
    )


def _load_observation(model: SmolVLAEvalModel, dataset: LeRobotDataset, dataset_index: int):
    sample = dataset[dataset_index]
    observation, _, _ = model.prepare_sample(sample)
    return observation


def _load_observation_batch(
    model: SmolVLAEvalModel,
    dataset: LeRobotDataset,
    batch_records: Sequence[SampleRecord],
) -> EvalObservation:
    observations = [_load_observation(model, dataset, record.dataset_index) for record in batch_records]
    return stack_observations(observations)


def _pad_observation_batch(observation: EvalObservation, target_batch: int) -> EvalObservation:
    current = int(observation.state.shape[0])
    if current == target_batch:
        return observation
    if current > target_batch:
        raise ValueError(f"Cannot pad observation batch of {current} down to {target_batch}.")
    pad = target_batch - current

    def pad_array(value: Any) -> Any:
        if value is None:
            return None
        array = jnp.asarray(value)
        pad_width = [(0, pad)] + [(0, 0)] * (array.ndim - 1)
        return jnp.pad(array, pad_width)

    return EvalObservation(
        images=pad_array(observation.images),
        image_masks=pad_array(observation.image_masks),
        language_tokens=pad_array(observation.language_tokens),
        language_masks=pad_array(observation.language_masks),
        state=pad_array(observation.state),
        image_keys=observation.image_keys,
    )


def prepare_cache(
    *,
    checkpoint_dir: pathlib.Path,
    cache_dir: pathlib.Path,
    dataset_repo_id: str,
    dataset_root: pathlib.Path | None = None,
    dataset_revision: str | None = None,
    action_key: str | None = None,
    rename_map: Mapping[str, str] | None = None,
    allow_download: bool = False,
    model_sample_steps: int,
    reverse_steps: int,
    reverse_solver: str,
    batch_size: int,
    inference_seed: int,
    split_seed: int,
    val_fraction: float,
    frame_stride: int,
    max_episodes: int | None,
    max_samples: int | None,
    flush_every: int = 8,
) -> dict[str, Any]:
    if model_sample_steps <= 0 or reverse_steps <= 0 or batch_size <= 0:
        raise ValueError("model_sample_steps, reverse_steps, and batch_size must all be positive.")
    if reverse_solver not in ("euler", "fireflow"):
        raise ValueError(f"reverse_solver must be 'euler' or 'fireflow', got {reverse_solver!r}.")
    if flush_every <= 0:
        raise ValueError(f"flush_every must be positive, got {flush_every}.")

    model = load_model(
        checkpoint_dir,
        dataset_repo_id=dataset_repo_id,
        dataset_root=dataset_root,
        dataset_revision=dataset_revision,
        action_key=action_key,
        rename_map=rename_map,
        local_files_only=not allow_download,
    )
    metadata = LeRobotDatasetMetadata(
        model.dataset_repo_id,
        root=model.dataset_root,
        revision=model.dataset_revision,
    )
    records, train_episodes, val_episodes = build_records(
        metadata,
        val_fraction=val_fraction,
        split_seed=split_seed,
        frame_stride=frame_stride,
        max_episodes=max_episodes,
        max_samples=max_samples,
    )
    configuration = _configuration(
        checkpoint_dir=checkpoint_dir,
        dataset_repo_id=model.dataset_repo_id,
        dataset_root=model.dataset_root,
        dataset_revision=model.dataset_revision,
        action_key=model.action_key,
        rename_map=rename_map,
        model_sample_steps=model_sample_steps,
        reverse_steps=reverse_steps,
        reverse_solver=reverse_solver,
        inference_seed=inference_seed,
        split_seed=split_seed,
        val_fraction=val_fraction,
        frame_stride=frame_stride,
        max_episodes=max_episodes,
        max_samples=max_samples,
    )
    digest = records_digest(records)
    manifest_path = cache_dir / MANIFEST_NAME
    action_horizon = model.action_horizon
    action_dim = model.action_dim

    if manifest_path.exists():
        manifest = load_manifest(cache_dir, require_complete=False)
        if manifest.get("configuration") != configuration or manifest.get("records_sha256") != digest:
            raise ValueError(
                f"Existing cache at {cache_dir} was created with different inputs. "
                "Choose a new cache directory instead of mixing runs."
            )
        if manifest.get("status") == "complete":
            print(f"cache already complete: {cache_dir}")
            return manifest
        arrays = open_cache_arrays(cache_dir, mode="r+")
        completed = int(manifest.get("completed_samples", 0))
        print(f"resuming cache at sample {completed}/{len(records)}")
    else:
        cache_dir.mkdir(parents=True, exist_ok=True)
        arrays = create_cache_arrays(
            cache_dir,
            records,
            action_horizon=action_horizon,
            action_dim=action_dim,
        )
        completed = 0
        manifest = {
            "version": CACHE_VERSION,
            "status": "incomplete",
            "completed_samples": 0,
            "sample_count": len(records),
            "train_sample_count": sum(record.split == "train" for record in records),
            "val_sample_count": sum(record.split == "val" for record in records),
            "train_episodes": list(train_episodes),
            "val_episodes": list(val_episodes),
            "action_horizon": action_horizon,
            "action_dim": action_dim,
            "configuration": configuration,
            "records_sha256": digest,
        }
        atomic_write_json(manifest_path, manifest)

    if completed == len(records):
        manifest["status"] = "complete"
        manifest["mean_source_inversion_mse"] = float(np.mean(np.asarray(arrays["inversion_mse"])))
        atomic_write_json(manifest_path, manifest)
        print(f"cache complete: {cache_dir}")
        return manifest

    dataset = _create_dataset(model, metadata)
    action_shape = (action_horizon, action_dim)
    loop_started = time.perf_counter()
    batches_since_flush = 0
    starts = list(range(completed, len(records), batch_size))

    pending: dict[str, Any] | None = None

    def _commit_pending(*, force_flush: bool) -> None:
        nonlocal pending, batches_since_flush
        if pending is None:
            return
        start = pending["start"]
        stop = pending["stop"]
        valid = pending["valid"]
        predicted_actions, x_base = jax.device_get(pending["predicted"]), jax.device_get(pending["x_base"])
        noise = pending["noise"]
        predicted_actions = np.asarray(predicted_actions[:valid], dtype=np.float32)
        x_base = np.asarray(x_base[:valid], dtype=np.float32)
        noise = np.asarray(jax.device_get(noise[:valid]), dtype=np.float32)

        arrays["target"][start:stop] = predicted_actions
        arrays["x_base"][start:stop] = x_base
        arrays["inversion_mse"][start:stop] = inversion_mse(x_base, noise)
        manifest["completed_samples"] = stop
        batches_since_flush += 1
        if force_flush or batches_since_flush >= flush_every or stop >= len(records):
            flush_arrays(arrays)
            atomic_write_json(manifest_path, manifest)
            batches_since_flush = 0
        elapsed = time.perf_counter() - loop_started
        done = stop - completed
        rate = done / elapsed if elapsed > 0 else 0.0
        remaining = len(records) - stop
        eta = remaining / rate if rate > 0 else float("inf")
        print(
            f"prepared {stop}/{len(records)} samples "
            f"({rate:.2f} samples/s, eta {eta / 60.0:.1f} min)"
        )
        pending = None

    for batch_number, start in enumerate(starts):
        stop = min(start + batch_size, len(records))
        batch_records = records[start:stop]
        valid = len(batch_records)
        observation_batch = _load_observation_batch(model, dataset, batch_records)
        if valid < batch_size:
            observation_batch = _pad_observation_batch(observation_batch, batch_size)
            pad_indices = [batch_records[-1].dataset_index] * (batch_size - valid)
        else:
            pad_indices = []
        dataset_indices = [record.dataset_index for record in batch_records] + pad_indices
        noise = deterministic_noise(dataset_indices, action_shape, seed=inference_seed)

        # Overlap: while this batch runs on device, commit the previous batch to host/disk.
        predicted_actions, x_base = sample_and_reverse(
            model,
            observation_batch,
            noise,
            sample_steps=model_sample_steps,
            reverse_steps=reverse_steps,
            solver=reverse_solver,
        )
        _commit_pending(force_flush=False)
        pending = {
            "start": start,
            "stop": stop,
            "valid": valid,
            "predicted": predicted_actions,
            "x_base": x_base,
            "noise": noise,
        }
        if batch_number == 0:
            # Force first-batch compile/sync so ETA is meaningful after warmup.
            jax.block_until_ready((predicted_actions, x_base))

    _commit_pending(force_flush=True)

    manifest["status"] = "complete"
    manifest["mean_source_inversion_mse"] = float(np.mean(np.asarray(arrays["inversion_mse"])))
    atomic_write_json(manifest_path, manifest)
    print(f"cache complete: {cache_dir}")
    print(f"mean_source_inversion_mse={manifest['mean_source_inversion_mse']:.8f}")
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Precompute SmolVLA predicted-action / reverse-integrated x_base pairs."
    )
    add_eval_data_arguments(parser, required=True)
    parser.add_argument("--cache-dir", type=pathlib.Path, required=True)
    parser.add_argument("--model-sample-steps", type=int, default=10)
    parser.add_argument("--reverse-steps", type=int, default=50)
    parser.add_argument(
        "--reverse-solver",
        choices=("euler", "fireflow"),
        default="fireflow",
        help="Numerical integrator for reverse action integration (default: fireflow).",
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument(
        "--flush-every",
        type=int,
        default=8,
        help="Flush memmap + manifest every N batches (default: 8).",
    )
    parser.add_argument("--inference-seed", type=int, default=0)
    parser.add_argument("--split-seed", type=int, default=0)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--max-episodes", type=int)
    parser.add_argument("--max-samples", type=int)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    prepare_cache(
        checkpoint_dir=args.checkpoint_dir,
        cache_dir=args.cache_dir,
        dataset_repo_id=args.dataset_repo_id,
        dataset_root=args.dataset_root,
        dataset_revision=args.dataset_revision,
        action_key=args.action_key,
        rename_map=parse_rename_map(args.rename_map),
        allow_download=args.allow_download,
        model_sample_steps=args.model_sample_steps,
        reverse_steps=args.reverse_steps,
        reverse_solver=args.reverse_solver,
        batch_size=args.batch_size,
        inference_seed=args.inference_seed,
        split_seed=args.split_seed,
        val_fraction=args.val_fraction,
        frame_stride=args.frame_stride,
        max_episodes=args.max_episodes,
        max_samples=args.max_samples,
        flush_every=args.flush_every,
    )


if __name__ == "__main__":
    main()
