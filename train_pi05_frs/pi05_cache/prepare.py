"""pi0.5 analogue of prepare.py: precompute predicted actions, reverse-integrated x_base, and GT
actions for FRS, using lerobot.policies.pi05_jax instead of SmolVLA JAX.

Structurally identical to prepare.py (record selection via utils/cache.py's build_records,
manifest/resume handling, memmap arrays are all shared) -- only the model side (loading,
observation-building, sampling) differs, via modalities_eval.pi05_utils.Pi05EvalModel and
utils.pi05_source_model.sample_and_reverse instead of modalities_eval.utils.SmolVLAEvalModel and
utils.source_model.sample_and_reverse.

UNTESTED, like the rest of this branch's pi0.5 integration -- see
src/lerobot/policies/pi05_jax/README.md and pi05_frs_plan.md.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import hashlib
import pathlib
import time
import urllib.parse
from collections.abc import Mapping, Sequence
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from lerobot.datasets import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.datasets.dataset_sources import resolve_source_visual_keys
from lerobot.datasets.sample_utils import to_numpy

from train_pi05_frs.pi05_cache.policy_inputs import Pi05EvalModel, stack_observations
from lerobot.policies.pi05_jax import Observation, Pi0, load_norm_stats
from lerobot.policies.pi05_jax.normalize import NormStats
from train_pi05_frs.pi05_cache.cache import CACHE_VERSION
from train_pi05_frs.pi05_cache.cache import MANIFEST_NAME
from train_pi05_frs.pi05_cache.cache import SampleRecord
from train_pi05_frs.pi05_cache.cache import atomic_write_json
from train_pi05_frs.pi05_cache.cache import build_records
from train_pi05_frs.pi05_cache.cache import create_cache_arrays
from train_pi05_frs.pi05_cache.cache import flush_arrays
from train_pi05_frs.pi05_cache.cache import load_manifest
from train_pi05_frs.pi05_cache.cache import open_cache_arrays
from train_pi05_frs.pi05_cache.cache import records_digest
from train_pi05_frs.pi05_cache.cache import validate_manifest_progress
from train_pi05_frs.pi05_cache.source_model import deterministic_noise, inversion_mse, sample_and_reverse


def _is_local_path(value: str | pathlib.Path) -> bool:
    """True unless `value` is a URL (`gs://...`, etc.) `download.maybe_download` would fetch.

    Deliberately checks the *string* before ever wrapping it in `pathlib.Path`:
    `Path("gs://bucket/x")` collapses the `//` to a single `/` (POSIX path normalization doesn't
    know about URL schemes), silently corrupting the URL -- see pi05_frs_plan.md's notes on this
    if this comment ever needs re-deriving.
    """
    return urllib.parse.urlparse(str(value)).scheme == ""


def _checkpoint_fingerprint(checkpoint_dir: pathlib.Path) -> str:
    """See prepare.py's version of this. pi0.5 checkpoints are an orbax `params/` directory
    (not a single model.safetensors), so this just hashes every file under the checkpoint root.
    """
    checkpoint_dir = checkpoint_dir.expanduser().resolve()
    digest = hashlib.sha256()
    candidates = sorted(path for path in checkpoint_dir.rglob("*") if path.is_file())
    if not candidates:
        raise FileNotFoundError(f"no checkpoint files found under {checkpoint_dir}")
    for path in candidates:
        stat = path.stat()
        digest.update(str(path.relative_to(checkpoint_dir)).encode())
        digest.update(f":{stat.st_size}:{stat.st_mtime_ns}\n".encode())
    return digest.hexdigest()


def _load_observation_batch(
    model: Pi05EvalModel,
    dataset: LeRobotDataset,
    action_dataset: Any,
    batch_records: Sequence[SampleRecord],
    *,
    action_key: str,
    action_horizon: int,
    episode_end_indices: Mapping[int, int],
    load_workers: int,
) -> tuple[Observation, jax.Array]:
    indices = [record.dataset_index for record in batch_records]
    if load_workers == 1:
        samples = [dataset[index] for index in indices]
        action_windows = _load_action_windows(
            action_dataset,
            batch_records,
            action_key=action_key,
            action_horizon=action_horizon,
            episode_end_indices=episode_end_indices,
        )
    else:
        with ThreadPoolExecutor(max_workers=load_workers + 1) as pool:
            action_future = pool.submit(
                _load_action_windows,
                action_dataset,
                batch_records,
                action_key=action_key,
                action_horizon=action_horizon,
                episode_end_indices=episode_end_indices,
            )
            samples = list(pool.map(dataset.__getitem__, indices))
            action_windows = action_future.result()

    observations: list[Observation] = []
    gt_actions: list[jax.Array] = []
    for sample, action_window in zip(samples, action_windows, strict=True):
        sample = {**sample, action_key: action_window}
        observation, actions, _ = model.prepare_sample(sample)
        observations.append(observation)
        gt_actions.append(actions)
    return stack_observations(observations), jnp.stack(gt_actions, axis=0)


def _load_action_windows(
    action_dataset: Any,
    batch_records: Sequence[SampleRecord],
    *,
    action_key: str,
    action_horizon: int,
    episode_end_indices: Mapping[int, int],
) -> np.ndarray:
    """Read overlapping action windows as merged contiguous Arrow slices."""
    if not batch_records:
        raise ValueError("batch_records must not be empty")

    offsets = np.arange(action_horizon, dtype=np.int64)
    query_indices = np.empty((len(batch_records), action_horizon), dtype=np.int64)
    for row, record in enumerate(batch_records):
        try:
            episode_end = int(episode_end_indices[record.episode_index])
        except KeyError as error:
            raise KeyError(f"missing end index for episode {record.episode_index}") from error
        if record.dataset_index >= episode_end:
            raise ValueError(
                f"dataset index {record.dataset_index} is outside episode {record.episode_index} "
                f"ending at {episode_end}"
            )
        query_indices[row] = np.minimum(record.dataset_index + offsets, episode_end - 1)

    unique_indices = np.unique(query_indices)
    split_points = np.flatnonzero(np.diff(unique_indices) > action_horizon) + 1
    index_groups = np.split(unique_indices, split_points)
    windows: np.ndarray | None = None

    for group in index_groups:
        start = int(group[0])
        stop = int(group[-1]) + 1
        rows = action_dataset[start:stop][action_key]
        if isinstance(rows, np.ndarray):
            contiguous = rows
        else:
            contiguous = np.stack([to_numpy(row) for row in rows], axis=0)
        contiguous = np.asarray(contiguous, dtype=np.float32)
        if contiguous.shape[0] != stop - start:
            raise ValueError(
                f"action slice [{start}:{stop}] returned {contiguous.shape[0]} rows"
            )
        if windows is None:
            windows = np.empty((*query_indices.shape, *contiguous.shape[1:]), dtype=np.float32)
        mask = (query_indices >= start) & (query_indices < stop)
        windows[mask] = contiguous[query_indices[mask] - start]

    assert windows is not None
    return windows


def _pad_observation_batch(observation: Observation, target_batch: int) -> Observation:
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

    return jax.tree.map(pad_array, observation)


def _pad_action_batch(actions: jax.Array, target_batch: int) -> jax.Array:
    current = int(actions.shape[0])
    if current == target_batch:
        return actions
    if current > target_batch:
        raise ValueError(f"Cannot pad action batch of {current} down to {target_batch}.")
    pad = target_batch - current
    return jnp.pad(actions, ((0, pad), (0, 0), (0, 0)))


def load_norm_stats_or_raise(
    norm_stats_dir: str | pathlib.Path, asset_id: str
) -> tuple[NormStats, NormStats]:
    """Load state/action NormStats, failing loudly instead of silently picking a default.

    See modalities_eval/pi05_utils.py's module docstring and pi05_frs_plan.md: pi0.5's released
    checkpoints only ship norm stats for the datasets they were trained on, keyed by `asset_id`.
    There is no asset_id for a brand-new robot/dataset -- the caller must have already decided
    (and computed, if needed) which stats to use before calling this.
    """
    stats = load_norm_stats(norm_stats_dir, asset_id)
    missing = {"state", "actions"} - set(stats)
    if missing:
        raise KeyError(
            f"norm stats at {pathlib.Path(norm_stats_dir) / asset_id} are missing {sorted(missing)}; "
            "expected keys 'state' and 'actions' (openpi convention)."
        )
    return stats["state"], stats["actions"]


def _cache_provenance_mismatches(
    manifest: Mapping[str, Any],
    *,
    configuration: Mapping[str, Any],
    records_sha256: str,
    action_horizon: int,
    action_dim: int,
    state_dim: int,
) -> list[str]:
    mismatches: list[str] = []
    previous_configuration = manifest.get("configuration")
    if not isinstance(previous_configuration, Mapping):
        mismatches.append("configuration")
    else:
        for key in sorted(set(previous_configuration) | set(configuration)):
            if previous_configuration.get(key) != configuration.get(key):
                mismatches.append(f"configuration.{key}")
    if manifest.get("records_sha256") != records_sha256:
        mismatches.append("records_sha256")
    for key, expected in (
        ("action_horizon", action_horizon),
        ("action_dim", action_dim),
        ("state_dim", state_dim),
    ):
        if manifest.get(key) != expected:
            mismatches.append(key)
    return mismatches


def _cache_array_shape_mismatches(
    arrays: Mapping[str, np.ndarray],
    *,
    sample_count: int,
    action_horizon: int,
    action_dim: int,
    state_dim: int,
) -> list[str]:
    mismatches: list[str] = []
    action_shape = (sample_count, action_horizon, action_dim)
    for key in ("x_base", "target", "gt_action"):
        if tuple(arrays[key].shape) != action_shape:
            mismatches.append(f"arrays.{key}.shape")
    if tuple(arrays["state"].shape) != (sample_count, state_dim):
        mismatches.append("arrays.state.shape")
    for key in ("dataset_index", "episode_index", "split", "inversion_mse"):
        if tuple(arrays[key].shape) != (sample_count,):
            mismatches.append(f"arrays.{key}.shape")
    return mismatches


def prepare_cache(
    *,
    checkpoint_dir: str,
    cache_dir: pathlib.Path,
    dataset_repo_id: str,
    dataset_root: pathlib.Path | None,
    dataset_revision: str | None,
    action_key: str | None,
    rename_map: Mapping[str, str] | None,
    camera_map: Mapping[str, str],
    norm_stats_dir: str | pathlib.Path,
    norm_stats_asset_id: str,
    use_quantile_norm: bool,
    action_dim: int,
    action_horizon: int,
    # Must match the TrainConfig the checkpoint was trained with; the defaults describe the
    # official pi05_base. See modalities_eval/pi05_utils.py:Pi05SampleProcessor.
    paligemma_variant: str = "gemma_2b",
    action_expert_variant: str = "gemma_300m",
    model_sample_steps: int,
    reverse_steps: int,
    reverse_solver: str,
    batch_size: int,
    load_workers: int,
    inference_seed: int,
    split_seed: int,
    val_fraction: float,
    frame_stride: int,
    max_episodes: int | None,
    max_samples: int | None,
    drop_tail_action_chunks: int = 1,
    flush_every: int = 8,
    loaded_model: Pi0 | None = None,
) -> pathlib.Path:
    if min(model_sample_steps, reverse_steps, batch_size, load_workers) <= 0:
        raise ValueError(
            "model_sample_steps, reverse_steps, batch_size, and load_workers must all be positive."
        )
    if reverse_solver not in ("euler", "fireflow", "slerpflow"):
        raise ValueError(
            "reverse_solver must be 'euler', 'fireflow', or 'slerpflow', "
            f"got {reverse_solver!r}."
        )
    if flush_every <= 0:
        raise ValueError(f"flush_every must be positive, got {flush_every}.")
    if drop_tail_action_chunks < 0:
        raise ValueError(f"drop_tail_action_chunks must be >= 0, got {drop_tail_action_chunks}.")

    state_stats, action_stats = load_norm_stats_or_raise(norm_stats_dir, norm_stats_asset_id)
    model = Pi05EvalModel(
        checkpoint_dir,
        dataset_repo_id=dataset_repo_id,
        dataset_root=dataset_root,
        dataset_revision=dataset_revision,
        action_key=action_key,
        rename_map=rename_map,
        camera_map=camera_map,
        state_stats=state_stats,
        action_stats=action_stats,
        use_quantile_norm=use_quantile_norm,
        action_dim=action_dim,
        action_horizon=action_horizon,
        paligemma_variant=paligemma_variant,
        action_expert_variant=action_expert_variant,
        loaded_model=loaded_model,
    )
    metadata = LeRobotDatasetMetadata(model.dataset_repo_id, root=model.dataset_root, revision=model.dataset_revision)
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
    checkpoint_is_local = _is_local_path(checkpoint_dir)
    configuration = {
        "checkpoint_dir": str(pathlib.Path(checkpoint_dir).resolve()) if checkpoint_is_local else str(checkpoint_dir),
        "checkpoint_fingerprint": (
            _checkpoint_fingerprint(pathlib.Path(checkpoint_dir)) if checkpoint_is_local else None
        ),
        "dataset_repo_id": model.dataset_repo_id,
        "dataset_root": str(model.dataset_root.resolve()) if model.dataset_root is not None else None,
        "dataset_revision": model.dataset_revision,
        "action_key": model.action_key,
        "rename_map": dict(rename_map) if rename_map is not None else None,
        "camera_map": dict(camera_map),
        "norm_stats_dir": str(norm_stats_dir),
        "norm_stats_asset_id": norm_stats_asset_id,
        "use_quantile_norm": use_quantile_norm,
        "base_model": "pi0.5",
        "action_dim": int(model.action_dim),
        "action_horizon": int(model.action_horizon),
        "paligemma_variant": paligemma_variant,
        "action_expert_variant": action_expert_variant,
        "model_sample_steps": model_sample_steps,
        "reverse_steps": reverse_steps,
        "reverse_solver": reverse_solver,
        "inference_seed": inference_seed,
        "split_seed": split_seed,
        "val_fraction": val_fraction,
        "frame_stride": frame_stride,
        "max_episodes": max_episodes,
        "max_samples": max_samples,
        "drop_tail_action_chunks": drop_tail_action_chunks,
    }
    digest = records_digest(records)
    manifest_path = cache_dir / MANIFEST_NAME
    action_dim = model.action_dim
    # Store the dataset's real normalized state width, not the zero padding added for pi0.5.
    state_dim = int(np.asarray(model.state_stats.mean).shape[-1])
    drop_frames = int(drop_tail_action_chunks) * action_horizon
    if drop_frames > 0:
        print(
            f"dropping last {drop_tail_action_chunks} action chunk(s) "
            f"({drop_frames} frames) from each episode before sampling"
        )

    if manifest_path.exists():
        manifest = load_manifest(cache_dir, require_complete=False)
        mismatches = _cache_provenance_mismatches(
            manifest,
            configuration=configuration,
            records_sha256=digest,
            action_horizon=action_horizon,
            action_dim=action_dim,
            state_dim=state_dim,
        )
        if mismatches:
            raise ValueError(
                f"Existing cache at {cache_dir} was created with different inputs. "
                f"Mismatched fields: {', '.join(mismatches)}. "
                "Choose a new cache directory instead of mixing runs."
            )
        status, completed, _ = validate_manifest_progress(
            manifest, expected_sample_count=len(records)
        )
        arrays = open_cache_arrays(cache_dir, mode="r+")
        shape_mismatches = _cache_array_shape_mismatches(
            arrays,
            sample_count=len(records),
            action_horizon=action_horizon,
            action_dim=action_dim,
            state_dim=state_dim,
        )
        if shape_mismatches:
            raise ValueError(
                f"Existing cache at {cache_dir} has incompatible arrays. "
                f"Mismatched fields: {', '.join(shape_mismatches)}."
            )
        if status == "complete":
            print(f"cache already complete: {cache_dir}")
            return cache_dir
        print(f"resuming cache at sample {completed}/{len(records)}")
    else:
        cache_dir.mkdir(parents=True, exist_ok=True)
        arrays = create_cache_arrays(
            cache_dir,
            records,
            action_horizon=action_horizon,
            action_dim=action_dim,
            state_dim=state_dim,
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
            "state_dim": state_dim,
            "configuration": configuration,
            "records_sha256": digest,
        }
        atomic_write_json(manifest_path, manifest)

    if completed == len(records):
        manifest["status"] = "complete"
        manifest["mean_source_inversion_mse"] = float(np.mean(np.asarray(arrays["inversion_mse"])))
        atomic_write_json(manifest_path, manifest)
        print(f"cache complete: {cache_dir}")
        return cache_dir

    source_camera_keys = resolve_source_visual_keys(
        tuple(camera_map.values()),
        rename_map,
        metadata.camera_keys,
    )
    print(
        f"loading only pi0.5 RGB columns: {source_camera_keys} "
        f"(workers={load_workers}, available_cameras={len(metadata.camera_keys)})",
        flush=True,
    )
    dataset = LeRobotDataset(
        model.dataset_repo_id,
        root=model.dataset_root,
        revision=model.dataset_revision,
        visual_keys=source_camera_keys,
    )
    action_dataset = dataset.select_columns(model.action_key)
    episode_end_indices = {
        episode_index: int(np.asarray(metadata.episodes[episode_index]["dataset_to_index"]).item())
        for episode_index in range(metadata.total_episodes)
    }
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
        states = pending["state"]
        noise = pending["noise"]
        predicted_actions = np.asarray(predicted_actions[:valid], dtype=np.float32)
        x_base = np.asarray(x_base[:valid], dtype=np.float32)
        gt_actions = np.asarray(gt_actions[:valid], dtype=np.float32)
        states = np.asarray(jax.device_get(states[:valid]), dtype=np.float32)
        noise = np.asarray(jax.device_get(noise[:valid]), dtype=np.float32)

        arrays["target"][start:stop] = predicted_actions
        arrays["x_base"][start:stop] = x_base
        arrays["gt_action"][start:stop] = gt_actions
        arrays["state"][start:stop] = states
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
        print(f"prepared {stop}/{len(records)} samples ({rate:.2f} samples/s, eta {eta / 60.0:.1f} min)")
        pending = None

    for batch_number, start in enumerate(starts):
        stop = min(start + batch_size, len(records))
        batch_records = records[start:stop]
        valid = len(batch_records)
        observation_batch, gt_action_batch = _load_observation_batch(
            model,
            dataset,
            action_dataset,
            batch_records,
            action_key=model.action_key,
            action_horizon=action_horizon,
            episode_end_indices=episode_end_indices,
            load_workers=load_workers,
        )
        if valid < batch_size:
            observation_batch = _pad_observation_batch(observation_batch, batch_size)
            gt_action_batch = _pad_action_batch(gt_action_batch, batch_size)
            pad_indices = [batch_records[-1].dataset_index] * (batch_size - valid)
        else:
            pad_indices = []
        dataset_indices = [record.dataset_index for record in batch_records] + pad_indices
        noise = deterministic_noise(dataset_indices, action_shape, seed=inference_seed)

        predicted_actions, x_base = sample_and_reverse(
            model.model,
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
            "gt_action": gt_action_batch,
            "state": observation_batch.state[..., :state_dim],
            "noise": noise,
        }
        if batch_number == 0:
            jax.block_until_ready((predicted_actions, x_base))

    _commit_pending(force_flush=True)

    manifest["status"] = "complete"
    manifest["mean_source_inversion_mse"] = float(np.mean(np.asarray(arrays["inversion_mse"])))
    atomic_write_json(manifest_path, manifest)
    print(f"cache complete: {cache_dir}")
    print(f"mean_source_inversion_mse={manifest['mean_source_inversion_mse']:.8f}")
    return cache_dir
