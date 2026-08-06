from __future__ import annotations

import argparse
import hashlib
import pathlib
import sys
import threading
import time
from collections.abc import Mapping, Sequence
from contextlib import contextmanager, nullcontext
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lerobot.datasets import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.policies.smolvla_jax.data import action_delta_timestamps
from lerobot.policies.smolvla_jax.data import resolve_source_visual_keys

from modalities_eval.utils import EvalObservation
from modalities_eval.utils import SmolVLAEvalModel
from modalities_eval.utils import add_eval_data_arguments
from modalities_eval.utils import load_model
from modalities_eval.utils import parse_rename_map
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
from utils.cache import trim_episode_tail
from utils.integration import REVERSE_INTEGRATION_VERSION
from utils.source_model import deterministic_noise
from utils.source_model import inversion_mse
from utils.source_model import sample_and_reverse
from utils.source_model import stack_observations


@contextmanager
def _progress_heartbeat(label: str, *, interval_seconds: float = 10.0):
    """Report liveness while a blocking JAX compile/warmup is in progress."""

    if interval_seconds <= 0:
        raise ValueError(f"interval_seconds must be positive, got {interval_seconds}.")
    started = time.perf_counter()
    stopped = threading.Event()

    def report() -> None:
        while not stopped.wait(interval_seconds):
            elapsed = time.perf_counter() - started
            print(f"{label}: still running (elapsed {elapsed:.0f}s)", flush=True)

    thread = threading.Thread(target=report, name="prepare-progress-heartbeat", daemon=True)
    print(f"{label}: started", flush=True)
    thread.start()
    try:
        yield
    finally:
        stopped.set()
        thread.join()
        elapsed = time.perf_counter() - started
        print(f"{label}: finished in {elapsed:.1f}s", flush=True)


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
    action_horizon: int,
    drop_tail_action_chunks: int = 1,
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
        trimmed = trim_episode_tail(
            _indices_for_episode(metadata, episode_index),
            drop_tail_action_chunks=drop_tail_action_chunks,
            action_horizon=action_horizon,
        )
        dataset_indices = trimmed[::frame_stride]
        records.extend(SampleRecord(int(index), episode_index, split) for index in dataset_indices)
    records = limit_records(records, max_samples=max_samples, seed=split_seed)
    if not records:
        raise ValueError(
            "Dataset selection produced no samples. "
            "Try lowering --drop-tail-action-chunks or using longer episodes."
        )
    present = {record.episode_index for record in records}
    train_episodes = tuple(episode for episode in train_episodes if episode in present)
    val_episodes = tuple(episode for episode in val_episodes if episode in present)
    if not train_episodes or not val_episodes:
        raise ValueError(
            "After dropping episode tails, train or val split is empty. "
            "Try lowering --drop-tail-action-chunks."
        )
    return records, train_episodes, val_episodes


def _configuration(
    *,
    checkpoint_dir: pathlib.Path,
    dataset_repo_id: str,
    dataset_root: pathlib.Path | None,
    dataset_revision: str | None,
    action_key: str | None,
    rename_map: Mapping[str, str] | None,
    normalization_source: str,
    model_sample_steps: int,
    reverse_steps: int,
    reverse_solver: str,
    inference_seed: int,
    split_seed: int,
    val_fraction: float,
    frame_stride: int,
    max_episodes: int | None,
    max_samples: int | None,
    drop_tail_action_chunks: int,
) -> dict[str, Any]:
    return {
        "checkpoint_dir": str(checkpoint_dir.resolve()),
        "checkpoint_fingerprint": _checkpoint_fingerprint(checkpoint_dir),
        "dataset_repo_id": dataset_repo_id,
        "dataset_root": str(dataset_root.resolve()) if dataset_root is not None else None,
        "dataset_revision": dataset_revision,
        "action_key": action_key,
        "rename_map": dict(rename_map) if rename_map is not None else None,
        "normalization_source": normalization_source,
        "model_sample_steps": model_sample_steps,
        "reverse_steps": reverse_steps,
        "reverse_solver": reverse_solver,
        "reverse_integration_version": REVERSE_INTEGRATION_VERSION,
        "inference_seed": inference_seed,
        "split_seed": split_seed,
        "val_fraction": val_fraction,
        "frame_stride": frame_stride,
        "max_episodes": max_episodes,
        "max_samples": max_samples,
        "drop_tail_action_chunks": drop_tail_action_chunks,
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
    visual_keys = resolve_source_visual_keys(
        model.config.image_keys,
        model.preprocessor.rename_map,
        metadata.camera_keys,
    )
    print(f"action-cache visual_keys={visual_keys}", flush=True)
    return LeRobotDataset(
        model.dataset_repo_id,
        root=model.dataset_root,
        revision=model.dataset_revision,
        delta_timestamps=action_delta_timestamps(
            model.action_key,
            model.config.chunk_size,
            metadata.fps,
        ),
        visual_keys=visual_keys,
    )


def _load_observation_and_gt(
    model: SmolVLAEvalModel, dataset: LeRobotDataset, dataset_index: int
) -> tuple[EvalObservation, jax.Array]:
    sample = dataset[dataset_index]
    observation, gt_actions, _ = model.prepare_sample(sample)
    return observation, jnp.asarray(gt_actions, dtype=jnp.float32)


def _load_observation_batch(
    model: SmolVLAEvalModel,
    dataset: LeRobotDataset,
    batch_records: Sequence[SampleRecord],
    *,
    report_progress: bool = False,
) -> tuple[EvalObservation, jax.Array]:
    observations: list[EvalObservation] = []
    gt_actions: list[jax.Array] = []
    for offset, record in enumerate(batch_records, start=1):
        if report_progress:
            print(
                f"first batch data load: sample {offset}/{len(batch_records)} "
                f"dataset_index={record.dataset_index}",
                flush=True,
            )
        observation, actions = _load_observation_and_gt(model, dataset, record.dataset_index)
        observations.append(observation)
        gt_actions.append(actions)
    return stack_observations(observations), jnp.stack(gt_actions, axis=0)


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


def _pad_action_batch(actions: jax.Array, target_batch: int) -> jax.Array:
    current = int(actions.shape[0])
    if current == target_batch:
        return actions
    if current > target_batch:
        raise ValueError(f"Cannot pad action batch of {current} down to {target_batch}.")
    pad = target_batch - current
    return jnp.pad(actions, ((0, pad), (0, 0), (0, 0)))


def _require_finite_cache_batch(**values: np.ndarray) -> None:
    """Fail before writing a cache batch containing NaN or infinity."""

    for name, value in values.items():
        array = np.asarray(value)
        invalid = np.argwhere(~np.isfinite(array))
        if invalid.size == 0:
            continue
        location = tuple(int(index) for index in invalid[0])
        raise FloatingPointError(
            f"non-finite value in {name} at batch index {location}: "
            f"{array[location]!r}"
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
    normalization_source: str = "checkpoint",
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
    drop_tail_action_chunks: int = 1,
    flush_every: int = 8,
) -> dict[str, Any]:
    if model_sample_steps <= 0 or reverse_steps <= 0 or batch_size <= 0:
        raise ValueError("model_sample_steps, reverse_steps, and batch_size must all be positive.")
    if reverse_solver not in ("euler", "fireflow", "slerpflow"):
        raise ValueError(
            "reverse_solver must be 'euler', 'fireflow', or 'slerpflow', "
            f"got {reverse_solver!r}."
        )
    if flush_every <= 0:
        raise ValueError(f"flush_every must be positive, got {flush_every}.")
    if drop_tail_action_chunks < 0:
        raise ValueError(f"drop_tail_action_chunks must be >= 0, got {drop_tail_action_chunks}.")

    model = load_model(
        checkpoint_dir,
        dataset_repo_id=dataset_repo_id,
        dataset_root=dataset_root,
        dataset_revision=dataset_revision,
        action_key=action_key,
        rename_map=rename_map,
        normalization_source=normalization_source,
        local_files_only=not allow_download,
    )
    metadata = LeRobotDatasetMetadata(
        model.dataset_repo_id,
        root=model.dataset_root,
        revision=model.dataset_revision,
    )
    action_horizon = int(model.action_horizon)
    records, train_episodes, val_episodes = build_records(
        metadata,
        val_fraction=val_fraction,
        split_seed=split_seed,
        frame_stride=frame_stride,
        max_episodes=max_episodes,
        max_samples=max_samples,
        action_horizon=action_horizon,
        drop_tail_action_chunks=drop_tail_action_chunks,
    )
    configuration = _configuration(
        checkpoint_dir=checkpoint_dir,
        dataset_repo_id=model.dataset_repo_id,
        dataset_root=model.dataset_root,
        dataset_revision=model.dataset_revision,
        action_key=model.action_key,
        rename_map=rename_map,
        normalization_source=normalization_source,
        model_sample_steps=model_sample_steps,
        reverse_steps=reverse_steps,
        reverse_solver=reverse_solver,
        inference_seed=inference_seed,
        split_seed=split_seed,
        val_fraction=val_fraction,
        frame_stride=frame_stride,
        max_episodes=max_episodes,
        max_samples=max_samples,
        drop_tail_action_chunks=drop_tail_action_chunks,
    )
    digest = records_digest(records)
    manifest_path = cache_dir / MANIFEST_NAME
    action_dim = model.action_dim
    drop_frames = int(drop_tail_action_chunks) * action_horizon
    if drop_frames > 0:
        print(
            f"dropping last {drop_tail_action_chunks} action chunk(s) "
            f"({drop_frames} frames) from each episode before sampling"
        )

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
        gt_actions = pending["gt_action"]
        noise = pending["noise"]
        predicted_actions = np.asarray(predicted_actions[:valid], dtype=np.float32)
        x_base = np.asarray(x_base[:valid], dtype=np.float32)
        gt_actions = np.asarray(gt_actions[:valid], dtype=np.float32)
        noise = np.asarray(jax.device_get(noise[:valid]), dtype=np.float32)
        batch_inversion_mse = inversion_mse(x_base, noise)
        _require_finite_cache_batch(
            predicted_actions=predicted_actions,
            x_base=x_base,
            gt_actions=gt_actions,
            noise=noise,
            inversion_mse=batch_inversion_mse,
        )

        arrays["target"][start:stop] = predicted_actions
        arrays["x_base"][start:stop] = x_base
        arrays["gt_action"][start:stop] = gt_actions
        arrays["inversion_mse"][start:stop] = batch_inversion_mse
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
            f"({rate:.2f} samples/s, eta {eta / 60.0:.1f} min)",
            flush=True,
        )
        pending = None

    for batch_number, start in enumerate(starts):
        stop = min(start + batch_size, len(records))
        batch_records = records[start:stop]
        valid = len(batch_records)
        if batch_number == 0:
            print(
                f"first batch data load: started "
                f"(samples={start}:{stop}, batch_size={batch_size})",
                flush=True,
            )
        observation_batch, gt_action_batch = _load_observation_batch(
            model,
            dataset,
            batch_records,
            report_progress=batch_number == 0,
        )
        if batch_number == 0:
            print("first batch data load: finished", flush=True)
        if valid < batch_size:
            observation_batch = _pad_observation_batch(observation_batch, batch_size)
            gt_action_batch = _pad_action_batch(gt_action_batch, batch_size)
            pad_indices = [batch_records[-1].dataset_index] * (batch_size - valid)
        else:
            pad_indices = []
        dataset_indices = [record.dataset_index for record in batch_records] + pad_indices
        noise = deterministic_noise(dataset_indices, action_shape, seed=inference_seed)

        # The first call includes XLA compilation and GPU warmup.  Keep emitting
        # liveness messages because JAX does not expose a numeric compile percentage.
        heartbeat = (
            _progress_heartbeat(
                f"first batch XLA compile/warmup "
                f"(solver={reverse_solver}, reverse_steps={reverse_steps}, batch_size={batch_size})"
            )
            if batch_number == 0
            else nullcontext()
        )
        with heartbeat:
            # Overlap: while this batch runs on device, commit the previous batch to host/disk.
            predicted_actions, x_base = sample_and_reverse(
                model,
                observation_batch,
                noise,
                sample_steps=model_sample_steps,
                reverse_steps=reverse_steps,
                solver=reverse_solver,
            )
            if batch_number == 0:
                # Force first-batch compile/sync so ETA is meaningful after warmup.
                jax.block_until_ready((predicted_actions, x_base))
        _commit_pending(force_flush=False)
        pending = {
            "start": start,
            "stop": stop,
            "valid": valid,
            "predicted": predicted_actions,
            "x_base": x_base,
            "gt_action": gt_action_batch,
            "noise": noise,
        }
    _commit_pending(force_flush=True)

    manifest["status"] = "complete"
    manifest["mean_source_inversion_mse"] = float(np.mean(np.asarray(arrays["inversion_mse"])))
    atomic_write_json(manifest_path, manifest)
    print(f"cache complete: {cache_dir}")
    print(f"mean_source_inversion_mse={manifest['mean_source_inversion_mse']:.8f}")
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Precompute SmolVLA predicted actions, reverse-integrated x_base, "
            "and dataset ground-truth actions."
        ),
    )
    add_eval_data_arguments(parser, required=True)
    # FRS must stay in the base policy's normalized action/state space across
    # every source dataset.  Individual dataset stats would make four
    # incompatible cache spaces.
    parser.set_defaults(normalization_source="checkpoint")
    parser.add_argument("--cache-dir", type=pathlib.Path, required=True)
    parser.add_argument("--model-sample-steps", type=int, default=10)
    parser.add_argument("--reverse-steps", type=int, default=50)
    parser.add_argument(
        "--reverse-solver",
        choices=("euler", "fireflow", "slerpflow"),
        default="slerpflow",
        help="Numerical integrator for reverse action integration (default: slerpflow).",
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
    parser.add_argument(
        "--drop-tail-action-chunks",
        type=int,
        default=1,
        help=(
            "Drop the last K * action_horizon frames from each episode before "
            "frame-stride sampling (default: 1). Set 0 to keep episode tails."
        ),
    )
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
        normalization_source=args.normalization_source,
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
        drop_tail_action_chunks=args.drop_tail_action_chunks,
        flush_every=args.flush_every,
    )


if __name__ == "__main__":
    main()
