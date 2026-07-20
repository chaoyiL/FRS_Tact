from __future__ import annotations

import argparse
import hashlib
import pathlib
from collections.abc import Sequence
from typing import Any

import jax
import numpy as np

from flow_decoder.utils.cache import CACHE_VERSION
from flow_decoder.utils.cache import MANIFEST_NAME
from flow_decoder.utils.cache import SampleRecord
from flow_decoder.utils.cache import atomic_write_json
from flow_decoder.utils.cache import create_cache_arrays
from flow_decoder.utils.cache import flush_arrays
from flow_decoder.utils.cache import limit_records
from flow_decoder.utils.cache import load_manifest
from flow_decoder.utils.cache import open_cache_arrays
from flow_decoder.utils.cache import records_digest
from flow_decoder.utils.cache import split_episodes


def _episode_count(raw_dataset: Any) -> int:
    from eval_scripts import utils as eval_utils

    episode_count = eval_utils.episode_count_from_metadata(raw_dataset)
    if episode_count is None:
        raise ValueError(
            "Dataset does not expose LeRobot episode metadata. "
            "Expected episode_data_index (v2.1) or meta.total_episodes (v3.0)."
        )
    return episode_count


def build_records(
    raw_dataset: Any,
    *,
    val_fraction: float,
    split_seed: int,
    frame_stride: int,
    max_episodes: int | None,
    max_samples: int | None,
) -> tuple[list[SampleRecord], tuple[int, ...], tuple[int, ...]]:
    from eval_scripts import utils as eval_utils

    if frame_stride <= 0:
        raise ValueError(f"frame_stride must be positive, got {frame_stride}.")
    episode_count = _episode_count(raw_dataset)
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
        dataset_indices = eval_utils._indices_for_episode(raw_dataset, episode_index)[::frame_stride]
        records.extend(SampleRecord(int(index), episode_index, split) for index in dataset_indices)
    records = limit_records(records, max_samples=max_samples, seed=split_seed)
    if not records:
        raise ValueError("Dataset selection produced no samples.")
    return records, train_episodes, val_episodes


def _configuration(
    *,
    config_name: str,
    checkpoint_dir: pathlib.Path,
    data_config: Any,
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
    repo_id = data_config.repo_id
    if isinstance(repo_id, tuple):
        repo_id = list(repo_id)
    return {
        "config_name": config_name,
        "checkpoint_dir": str(checkpoint_dir.resolve()),
        "checkpoint_fingerprint": _checkpoint_fingerprint(checkpoint_dir),
        "dataset_repo_id": repo_id,
        "asset_id": data_config.asset_id,
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
    params_dir = checkpoint_dir if checkpoint_dir.name == "params" else checkpoint_dir / "params"
    if not params_dir.is_dir():
        raise FileNotFoundError(f"Checkpoint params directory not found: {params_dir}")
    digest = hashlib.sha256()
    files = sorted(path for path in params_dir.rglob("*") if path.is_file())
    for path in files:
        stat = path.stat()
        digest.update(str(path.relative_to(params_dir)).encode())
        digest.update(f":{stat.st_size}:{stat.st_mtime_ns}\n".encode())
    return digest.hexdigest()


def _load_observation(transformed_dataset: Any, dataset_index: int) -> Any:
    from eval_scripts import utils as eval_utils
    from openpi.src.openpi.models import model as model_lib

    transformed = eval_utils._copy_tree(transformed_dataset[dataset_index])
    transformed = eval_utils._normalize_observation_dict(transformed)
    return model_lib.Observation.from_dict(transformed)


def prepare_cache(
    *,
    config_name: str,
    checkpoint_dir: pathlib.Path,
    cache_dir: pathlib.Path,
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
) -> dict[str, Any]:
    try:
        from eval_scripts import utils as eval_utils
        from flow_decoder.utils.source_model import deterministic_noise
        from flow_decoder.utils.source_model import inversion_mse
        from flow_decoder.utils.source_model import reverse_integrate_actions
        from flow_decoder.utils.source_model import stack_observations
        from openpi.src.openpi.shared import nnx_utils
        from openpi.src.openpi.training import config as config_lib
    except (AttributeError, ModuleNotFoundError) as exc:
        raise RuntimeError(
            "OpenPI could not be imported. Verify that eval_scripts/loglike_evaluate.py can start in this "
            "environment and that its LeRobot and jaxtyping dependencies are present."
        ) from exc

    if model_sample_steps <= 0 or reverse_steps <= 0 or batch_size <= 0:
        raise ValueError("model_sample_steps, reverse_steps, and batch_size must all be positive.")
    if reverse_solver not in ("euler", "fireflow"):
        raise ValueError(f"reverse_solver must be 'euler' or 'fireflow', got {reverse_solver!r}.")

    train_config = config_lib.get_config(config_name)
    data_config, raw_dataset, transformed_dataset = eval_utils.create_transformed_dataset(
        train_config, checkpoint_dir
    )
    records, train_episodes, val_episodes = build_records(
        raw_dataset,
        val_fraction=val_fraction,
        split_seed=split_seed,
        frame_stride=frame_stride,
        max_episodes=max_episodes,
        max_samples=max_samples,
    )
    configuration = _configuration(
        config_name=config_name,
        checkpoint_dir=checkpoint_dir,
        data_config=data_config,
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
            action_horizon=train_config.model.action_horizon,
            action_dim=train_config.model.action_dim,
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
            "action_horizon": train_config.model.action_horizon,
            "action_dim": train_config.model.action_dim,
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

    model = eval_utils.load_model(train_config, checkpoint_dir)
    sample_actions = nnx_utils.module_jit(model.sample_actions)
    action_shape = (train_config.model.action_horizon, train_config.model.action_dim)
    base_rng = jax.random.key(inference_seed)

    for start in range(completed, len(records), batch_size):
        stop = min(start + batch_size, len(records))
        batch_records = records[start:stop]
        observations = [
            _load_observation(transformed_dataset, record.dataset_index) for record in batch_records
        ]
        observation_batch = stack_observations(observations)
        dataset_indices = [record.dataset_index for record in batch_records]
        noise = deterministic_noise(dataset_indices, action_shape, seed=inference_seed)
        call_rng = jax.random.fold_in(base_rng, start)
        predicted_actions = sample_actions(
            call_rng,
            observation_batch,
            num_steps=model_sample_steps,
            noise=noise,
        ).astype(np.float32)
        x_base = reverse_integrate_actions(
            model,
            observation_batch,
            predicted_actions,
            num_steps=reverse_steps,
            solver=reverse_solver,
        )

        arrays["target"][start:stop] = np.asarray(jax.device_get(predicted_actions), dtype=np.float32)
        arrays["x_base"][start:stop] = np.asarray(jax.device_get(x_base), dtype=np.float32)
        arrays["inversion_mse"][start:stop] = inversion_mse(x_base, noise)
        flush_arrays(arrays)
        manifest["completed_samples"] = stop
        atomic_write_json(manifest_path, manifest)
        print(f"prepared {stop}/{len(records)} samples")

    manifest["status"] = "complete"
    manifest["mean_source_inversion_mse"] = float(np.mean(np.asarray(arrays["inversion_mse"])))
    atomic_write_json(manifest_path, manifest)
    print(f"cache complete: {cache_dir}")
    print(f"mean_source_inversion_mse={manifest['mean_source_inversion_mse']:.8f}")
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Precompute OpenPI predicted-action/x_base pairs.")
    parser.add_argument("--config-name", required=True)
    parser.add_argument("--checkpoint-dir", type=pathlib.Path, required=True)
    parser.add_argument("--cache-dir", type=pathlib.Path, required=True)
    parser.add_argument("--model-sample-steps", type=int, default=10)
    parser.add_argument("--reverse-steps", type=int, default=50)
    parser.add_argument(
        "--reverse-solver",
        choices=("euler", "fireflow"),
        default="fireflow",
        help="Numerical integrator for reverse action integration (default: euler).",
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--inference-seed", type=int, default=0)
    parser.add_argument("--split-seed", type=int, default=0)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--max-episodes", type=int)
    parser.add_argument("--max-samples", type=int)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    prepare_cache(**vars(args))


if __name__ == "__main__":
    main()
