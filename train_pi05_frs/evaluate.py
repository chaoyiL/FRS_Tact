from __future__ import annotations

import argparse
import csv
import pathlib
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from train_pi05_frs.utils.checkpoint import load_checkpoint
from train_pi05_frs.utils.bimanual_schema import BIMANUAL_LOSS_MODE
from train_pi05_frs.utils.bimanual_schema import validate_bimanual_objective_metadata
from train_pi05_frs.utils.bimanual_schema import validate_bimanual_tactile_keys
from train_pi05_frs.utils.objective_schema import COMPOSITE_GATED_LOSS_MODE
from train_pi05_frs.utils.data import CachedTactileEmbeddingBatches
from train_pi05_frs.utils.data import TactileConditionedBatches
from train_pi05_frs.utils.data import resolve_tactile_window
from train_pi05_frs.utils.metrics import EvalTarget
from train_pi05_frs.utils.metrics import bimanual_source_decode_metrics
from train_pi05_frs.utils.metrics import evaluate_split
from train_pi05_frs.utils.model import FlowSolver
from train_pi05_frs.utils.visualize import write_evaluation_plots
from train_pi05_frs.utils.bimanual_visualize import plot_bimanual_diagnostics
from train_pi05_frs.utils.bimanual_visualize import write_bimanual_evaluation_snapshot
from train_pi05_frs.utils.window_io import TACTILE_KEYS
from train_pi05_frs.pi05_cache.cache import CachedPairs, MultiCachedPairs, atomic_write_json


def _evaluate_decoder_legacy(
    *,
    cache_dir: pathlib.Path,
    tactile_encoder_dir: pathlib.Path,
    checkpoint_dir: pathlib.Path,
    output_dir: pathlib.Path,
    dataset_repo_id: str | None,
    dataset_root: pathlib.Path | None,
    tactile_window_divisor: int | None,
    history_stride: int | None,
    batch_size: int,
    num_steps: int,
    solver: FlowSolver,
    target: EvalTarget | None,
    save_predictions: bool,
    write_plots: bool,
    num_trajectory_samples: int,
    num_episode_strips: int,
    num_workers: int,
    prefetch_batches: int,
    load_threads: int,
    pipeline_prefetch: int,
    image_cache_size: int,
    loaded_checkpoint: tuple[Any, dict[str, Any]] | None = None,
) -> dict[str, float | int | str]:
    pairs = CachedPairs(cache_dir)
    model, checkpoint_metadata = (
        load_checkpoint(checkpoint_dir)
        if loaded_checkpoint is None
        else loaded_checkpoint
    )
    checkpoint_cache_digest = checkpoint_metadata.get("extra_metadata", {}).get("cache_records_sha256")
    if checkpoint_cache_digest is not None and checkpoint_cache_digest != pairs.manifest["records_sha256"]:
        raise ValueError("Checkpoint was trained from a different cache sample set.")
    expected_shape = (int(pairs.manifest["action_horizon"]), int(pairs.manifest["action_dim"]))
    actual_shape = (model.config.action_horizon, model.config.action_dim)
    if actual_shape != expected_shape:
        raise ValueError(f"Checkpoint/cache action shape mismatch: {actual_shape} != {expected_shape}.")
    if model.config.state_conditioning and model.config.state_dim != int(pairs.manifest["state_dim"]):
        raise ValueError(
            "Checkpoint/cache state dimension mismatch: "
            f"{model.config.state_dim} != {pairs.manifest['state_dim']}."
        )

    extra = checkpoint_metadata.get("extra_metadata") or {}
    if tactile_window_divisor is None:
        tactile_window_divisor = int(extra.get("tactile_window_divisor", 1))
    if history_stride is None:
        history_stride = int(extra.get("history_stride", 1))
    if target is None:
        loss_mode = str(extra.get("loss_mode", "gt"))
        target = "predicted" if loss_mode == "predicted" else "gt"
    action_horizon = int(pairs.manifest["action_horizon"])
    tactile_window = resolve_tactile_window(
        action_horizon=action_horizon,
        window_divisor=tactile_window_divisor,
    )
    if tactile_window != model.config.tactile_window:
        raise ValueError(
            f"Resolved tactile_window={tactile_window} does not match "
            f"checkpoint tactile_window={model.config.tactile_window}."
        )

    conditioner = TactileConditionedBatches(
        pairs,
        tactile_encoder_dir=tactile_encoder_dir,
        tactile_window=tactile_window,
        dataset_repo_id=dataset_repo_id,
        dataset_root=dataset_root,
        history_stride=history_stride,
        build_episode_baselines=(
            str(extra.get("loss_mode", "gt"))
            in ("gated", COMPOSITE_GATED_LOSS_MODE)
        ),
        num_workers=num_workers,
        prefetch_batches=prefetch_batches,
        load_threads=load_threads,
        pipeline_prefetch=pipeline_prefetch,
        image_cache_size=image_cache_size,
    )
    try:
        if conditioner.resnet_embedding_dim != model.config.resnet_embedding_dim:
            raise ValueError(
                f"Encoder resnet_embedding_dim={conditioner.resnet_embedding_dim} does not match "
                f"checkpoint resnet_embedding_dim={model.config.resnet_embedding_dim}."
            )

        result = evaluate_split(
            model,
            conditioner,
            split="val",
            batch_size=batch_size,
            num_steps=num_steps,
            solver=solver,
            keep_predictions=save_predictions,
            target=target,
            gate_tau=extra.get("gate_tau"),
            gate_temperature=extra.get("gate_temperature"),
            low_gate_threshold=float(extra.get("low_gate_threshold", 0.3)),
            high_gate_threshold=float(extra.get("high_gate_threshold", 0.7)),
            low_gate_safety_margin=float(extra.get("low_gate_safety_margin", 0.03)),
            low_gate_regression_margin=float(
                extra.get("low_gate_regression_margin", 0.005)
            ),
            rank_margin=float(extra.get("rank_margin", 0.0)),
            repair_margin=float(extra.get("repair_margin", 0.0)),
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        metrics: dict[str, float | int | str] = {
            "checkpoint": str(checkpoint_dir.resolve()),
            "checkpoint_epoch": int(checkpoint_metadata["epoch"]),
            "sample_count": len(result.cache_indices),
            "decoder_steps": num_steps,
            "decoder_solver": solver,
            "tactile_window": tactile_window,
            "tactile_window_divisor": tactile_window_divisor,
            "target": result.target,
            "flow_loss": result.flow_loss,
            "mse": result.mse,
            "rmse": result.rmse,
            "mae": result.mae,
            "flow_loss_gt": result.flow_loss_gt,
            "mse_gt": result.mse_gt,
            "rmse_gt": result.rmse_gt,
            "mae_gt": result.mae_gt,
            "flow_loss_pred": result.flow_loss_pred,
            "mse_pred": result.mse_pred,
            "rmse_pred": result.rmse_pred,
            "mae_pred": result.mae_pred,
        }
        if result.low_gate_unsafe_frac is not None:
            metrics.update(
                {
                    "low_gate_unsafe_frac": result.low_gate_unsafe_frac,
                    "low_gate_regression_frac": result.low_gate_regression_frac,
                    "high_gate_gain": result.high_gate_gain,  # type: ignore[dict-item]
                    "high_gate_harm_p95": result.high_gate_harm_p95,
                    "high_gate_rank_satisfied_frac": result.high_gate_rank_satisfied_frac,  # type: ignore[dict-item]
                    "high_gate_repair_satisfied_frac": result.high_gate_repair_satisfied_frac,  # type: ignore[dict-item]
                }
            )
        atomic_write_json(output_dir / "metrics.json", metrics)

        arrays = pairs.arrays
        with (output_dir / "per_sample.csv").open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(
                file,
                fieldnames=[
                    "cache_index",
                    "dataset_index",
                    "episode_index",
                    "flow_loss",
                    "mse",
                    "rmse",
                    "mae",
                    "mse_gt",
                    "mae_gt",
                    "mse_pred",
                    "mae_pred",
                ],
            )
            writer.writeheader()
            for position, cache_index in enumerate(result.cache_indices):
                writer.writerow(
                    {
                        "cache_index": int(cache_index),
                        "dataset_index": int(arrays["dataset_index"][cache_index]),
                        "episode_index": int(arrays["episode_index"][cache_index]),
                        "flow_loss": float(result.sample_flow_loss[position]),
                        "mse": float(result.sample_mse[position]),
                        "rmse": float(result.sample_rmse[position]),
                        "mae": float(result.sample_mae[position]),
                        "mse_gt": float(result.sample_mse_gt[position]),
                        "mae_gt": float(result.sample_mae_gt[position]),
                        "mse_pred": float(result.sample_mse_pred[position]),
                        "mae_pred": float(result.sample_mae_pred[position]),
                    }
                )
        if result.predictions is not None:
            np.savez(
                output_dir / "predictions.npz",
                cache_indices=result.cache_indices,
                predicted_actions=result.predictions,
            )

        if write_plots:
            try:
                plot_paths = write_evaluation_plots(
                    output_dir=output_dir,
                    result=result,
                    pairs=pairs,
                    model=model,
                    conditioner=conditioner,
                    num_steps=num_steps,
                    solver=solver,
                    num_trajectory_samples=num_trajectory_samples,
                    num_episode_strips=num_episode_strips,
                )
            except Exception as exc:
                print(f"warning: could not render evaluation plots: {exc}", flush=True)
            else:
                for plot_path in plot_paths:
                    print(f"plot={plot_path}")

        print(
            f"validation_samples={len(result.cache_indices)} solver={solver} "
            f"target={result.target} flow_loss={result.flow_loss:.8f} "
            f"mse={result.mse:.8f} mse_gt={result.mse_gt:.8f} mse_pred={result.mse_pred:.8f}"
        )
        print(f"evaluation={output_dir}")
        return metrics
    finally:
        conditioner.close()


def _bimanual_gate_region(
    weight: float, *, low_threshold: float, high_threshold: float
) -> str:
    if weight <= low_threshold:
        return "low"
    if weight >= high_threshold:
        return "high"
    return "mid"


def _bimanual_quadrant(left_region: str, right_region: str) -> str:
    if left_region == "mid" or right_region == "mid":
        return ""
    return f"{left_region}_{right_region}"


def _validate_bimanual_evaluation_contract(
    metadata: Mapping[str, Any],
    *,
    action_dim: int,
    tactile_keys: Sequence[object],
) -> None:
    """Validate action and tactile semantics before standalone evaluation."""

    validate_bimanual_objective_metadata(metadata, action_dim=action_dim)
    validate_bimanual_tactile_keys(
        tactile_keys, field_name="evaluation tactile_keys"
    )


def _bimanual_evaluation_history_path(
    *,
    checkpoint_dir: pathlib.Path,
    output_dir: pathlib.Path,
    result: Any,
    epoch: int,
) -> pathlib.Path:
    """Prefer the run history for best/last aliases, otherwise write a snapshot."""

    checkpoint_dir = pathlib.Path(checkpoint_dir)
    if checkpoint_dir.name in {"best", "last"}:
        run_history = checkpoint_dir.parent / "history.csv"
        if run_history.is_file():
            return run_history
    snapshot = output_dir / "evaluation_snapshot_history.csv"
    try:
        return write_bimanual_evaluation_snapshot(snapshot, result, epoch=epoch)
    except Exception as exc:
        print(
            f"warning: could not write bimanual evaluation snapshot: {exc}",
            flush=True,
        )
        return snapshot


def evaluate_decoder(
    *,
    cache_dir: pathlib.Path | None,
    tactile_encoder_dir: pathlib.Path,
    checkpoint_dir: pathlib.Path,
    output_dir: pathlib.Path,
    dataset_repo_id: str | None,
    dataset_root: pathlib.Path | None,
    tactile_window_divisor: int | None,
    history_stride: int | None,
    batch_size: int,
    num_steps: int,
    solver: FlowSolver,
    target: EvalTarget | None,
    save_predictions: bool,
    write_plots: bool,
    num_trajectory_samples: int,
    num_episode_strips: int,
    num_workers: int,
    prefetch_batches: int,
    load_threads: int,
    pipeline_prefetch: int,
    image_cache_size: int,
    encode_batch_size: int = 256,
    cache_dirs: Sequence[pathlib.Path] | None = None,
    dataset_sources: Sequence[Mapping[str, Any]] | None = None,
    tactile_embedding_cache_root: pathlib.Path | None = None,
    tactile_keys: Sequence[str] | None = None,
    tactile_embedding_dim: int = 512,
    tactile_image_size: int = 224,
) -> dict[str, Any]:
    """Evaluate legacy checkpoints unchanged and emit bimanual/source outputs."""
    model, checkpoint_metadata = load_checkpoint(checkpoint_dir)
    extra = checkpoint_metadata.get("extra_metadata") or {}
    loss_mode = str(extra.get("loss_mode", "gt"))
    if loss_mode != BIMANUAL_LOSS_MODE and cache_dirs is None:
        if cache_dir is None:
            raise ValueError("cache_dir is required for single-dataset evaluation")
        return _evaluate_decoder_legacy(
            cache_dir=cache_dir,
            tactile_encoder_dir=tactile_encoder_dir,
            checkpoint_dir=checkpoint_dir,
            output_dir=output_dir,
            dataset_repo_id=dataset_repo_id,
            dataset_root=dataset_root,
            tactile_window_divisor=tactile_window_divisor,
            history_stride=history_stride,
            batch_size=batch_size,
            num_steps=num_steps,
            solver=solver,
            target=target,
            save_predictions=save_predictions,
            write_plots=write_plots,
            num_trajectory_samples=num_trajectory_samples,
            num_episode_strips=num_episode_strips,
            num_workers=num_workers,
            prefetch_batches=prefetch_batches,
            load_threads=load_threads,
            pipeline_prefetch=pipeline_prefetch,
            image_cache_size=image_cache_size,
            loaded_checkpoint=(model, checkpoint_metadata),
        )
    if loss_mode != BIMANUAL_LOSS_MODE:
        raise ValueError("multi-source evaluation currently requires bimanual_gated")

    if cache_dirs is not None:
        if not cache_dirs:
            raise ValueError("cache_dirs must be non-empty when provided")
        if dataset_sources is None or len(dataset_sources) != len(cache_dirs):
            raise ValueError("dataset_sources must have one entry per cache directory")
        if not tactile_keys:
            raise ValueError(
                "multi-dataset evaluation requires tactile embedding cache root and tactile keys"
            )
        evaluation_tactile_keys: Sequence[object] = tactile_keys
        _validate_bimanual_evaluation_contract(
            extra,
            action_dim=int(model.config.action_dim),
            tactile_keys=evaluation_tactile_keys,
        )
        source_names = [str(source["repo_id"]) for source in dataset_sources]
        pairs: CachedPairs | MultiCachedPairs = MultiCachedPairs(
            cache_dirs, source_names=source_names
        )
    else:
        if cache_dir is None:
            raise ValueError("cache_dir is required for single-dataset evaluation")
        _validate_bimanual_evaluation_contract(
            extra,
            action_dim=int(model.config.action_dim),
            tactile_keys=TACTILE_KEYS,
        )
        pairs = CachedPairs(cache_dir)

    checkpoint_cache_digest = extra.get("cache_records_sha256")
    if (
        checkpoint_cache_digest is not None
        and checkpoint_cache_digest != pairs.manifest["records_sha256"]
    ):
        raise ValueError("Checkpoint was trained from a different cache sample set.")
    expected_shape = (
        int(pairs.manifest["action_horizon"]),
        int(pairs.manifest["action_dim"]),
    )
    actual_shape = (model.config.action_horizon, model.config.action_dim)
    if actual_shape != expected_shape:
        raise ValueError(
            f"Checkpoint/cache action shape mismatch: {actual_shape} != {expected_shape}."
        )
    cache_state_dim = int(pairs.manifest.get("state_dim", 0))
    if model.config.state_conditioning and model.config.state_dim != cache_state_dim:
        raise ValueError(
            "Checkpoint/cache state dimension mismatch: "
            f"{model.config.state_dim} != {cache_state_dim}."
        )
    if tactile_window_divisor is None:
        tactile_window_divisor = int(extra.get("tactile_window_divisor", 1))
    if history_stride is None:
        history_stride = int(extra.get("history_stride", 1))
    if target is None:
        target = "gt"
    low_threshold = float(
        extra.get("rank_low_gate_threshold", extra.get("low_gate_threshold", 0.3))
    )
    high_threshold = float(
        extra.get("rank_high_gate_threshold", extra.get("high_gate_threshold", 0.7))
    )
    low_safety_margin = float(extra.get("low_gate_safety_margin", 0.03))
    rank_margin = float(extra.get("rank_margin", 0.0))
    repair_margin = float(extra.get("repair_margin", 0.0))
    tactile_window = resolve_tactile_window(
        action_horizon=int(pairs.manifest["action_horizon"]),
        window_divisor=tactile_window_divisor,
    )
    if tactile_window != model.config.tactile_window:
        raise ValueError(
            f"Resolved tactile_window={tactile_window} does not match "
            f"checkpoint tactile_window={model.config.tactile_window}."
        )

    if isinstance(pairs, MultiCachedPairs):
        if tactile_embedding_cache_root is None:
            raise ValueError(
                "multi-dataset evaluation requires tactile embedding cache root and tactile keys"
            )
        assert dataset_sources is not None
        conditioner = CachedTactileEmbeddingBatches(
            pairs,
            sources=dataset_sources,
            tactile_cache_root=tactile_embedding_cache_root,
            tactile_encoder_dir=tactile_encoder_dir,
            tactile_keys=tactile_keys,
            tactile_window=tactile_window,
            history_stride=history_stride,
            embedding_dim=tactile_embedding_dim,
            image_size=tactile_image_size,
            build_episode_baselines=True,
        )
    else:
        conditioner = TactileConditionedBatches(
            pairs,
            tactile_encoder_dir=tactile_encoder_dir,
            tactile_window=tactile_window,
            dataset_repo_id=dataset_repo_id,
            dataset_root=dataset_root,
            history_stride=history_stride,
            build_episode_baselines=True,
            num_workers=num_workers,
            prefetch_batches=prefetch_batches,
            load_threads=load_threads,
            pipeline_prefetch=pipeline_prefetch,
            image_cache_size=image_cache_size,
            encode_batch_size=encode_batch_size,
        )
    try:
        if conditioner.resnet_embedding_dim != model.config.resnet_embedding_dim:
            raise ValueError(
                f"Encoder resnet_embedding_dim={conditioner.resnet_embedding_dim} does not match "
                f"checkpoint resnet_embedding_dim={model.config.resnet_embedding_dim}."
            )
        keep_actions = save_predictions or (
            write_plots and loss_mode == BIMANUAL_LOSS_MODE
        )
        result = evaluate_split(
            model,
            conditioner,
            split="val",
            batch_size=batch_size,
            num_steps=num_steps,
            solver=solver,
            keep_predictions=keep_actions,
            target=target,
            loss_mode=loss_mode,
            gate_tau=float(extra["gate_tau"]),
            gate_temperature=float(extra["gate_temperature"]),
            low_gate_safety_margin=low_safety_margin,
            low_gate_regression_margin=float(
                extra.get("low_gate_regression_margin", 0.005)
            ),
            rank_margin=rank_margin,
            repair_margin=repair_margin,
            rank_low_gate_threshold=low_threshold,
            rank_high_gate_threshold=high_threshold,
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        metrics: dict[str, Any] = {
            "checkpoint": str(checkpoint_dir.resolve()),
            "checkpoint_epoch": int(checkpoint_metadata["epoch"]),
            "sample_count": len(result.cache_indices),
            "decoder_steps": num_steps,
            "decoder_solver": solver,
            "tactile_window": tactile_window,
            "tactile_window_divisor": tactile_window_divisor,
            "target": result.target,
            "flow_loss": result.flow_loss,
            "mse": result.mse,
            "rmse": result.rmse,
            "mae": result.mae,
            "flow_loss_gt": result.flow_loss_gt,
            "mse_gt": result.mse_gt,
            "rmse_gt": result.rmse_gt,
            "mae_gt": result.mae_gt,
            "flow_loss_pred": result.flow_loss_pred,
            "mse_pred": result.mse_pred,
            "rmse_pred": result.rmse_pred,
            "mae_pred": result.mae_pred,
            "mse_vla_gt": result.mse_vla_gt,
            "gt_gain": result.gt_gain,
            "relative_gt_error": result.relative_gt_error,
            "composite_fm": result.composite_fm,
            "bimanual_quadrants": result.bimanual_quadrants,
            "bimanual_gate_region_counts": result.bimanual_gate_region_counts.tolist(),
            "rank_low_gate_threshold": low_threshold,
            "rank_high_gate_threshold": high_threshold,
        }
        for wrist in ("left", "right"):
            for metric_name in (
                "gate_w",
                "tactile_change",
                "gate_w_p10",
                "gate_w_p25",
                "gate_w_p50",
                "gate_w_p75",
                "gate_w_p90",
                "tactile_change_p10",
                "tactile_change_p25",
                "tactile_change_p50",
                "tactile_change_p75",
                "tactile_change_p90",
                "mse_gt_high_w",
                "mse_vla_high_w",
                "mse_vla_gt_high_w",
                "gt_gain_high_w",
                "rank_penalty_high_w",
                "rank_satisfied_high_frac",
                "repair_penalty_high_w",
                "repair_satisfied_high_frac",
                "low_nearest_endpoint_mse",
                "low_safety_penalty",
                "low_safe_frac",
                "low_unsafe_frac",
                "n_high_w",
                "n_low_w",
                "n_mid_w",
            ):
                value = getattr(result, f"{metric_name}_{wrist}")
                if value is not None:
                    output_name = {
                        "gate_w": "gate_w_mean",
                        "tactile_change": "tactile_change_mean",
                    }.get(metric_name, metric_name)
                    metrics[f"{output_name}_{wrist}"] = (
                        int(value) if metric_name.startswith("n_") else float(value)
                    )

        if isinstance(pairs, MultiCachedPairs):
            source_indices, local_indices = pairs.source_and_local_indices(
                result.cache_indices
            )
            dataset_indices = pairs.metadata_values(
                result.cache_indices, "dataset_index"
            )
            episode_indices = pairs.metadata_values(
                result.cache_indices, "episode_index"
            )
            source_names = tuple(pairs.source_names)
        else:
            source_indices = np.zeros(len(result.cache_indices), dtype=np.int32)
            local_indices = np.asarray(result.cache_indices, dtype=np.int64)
            dataset_indices = np.asarray(
                pairs.arrays["dataset_index"][result.cache_indices], dtype=np.int64
            )
            episode_indices = np.asarray(
                pairs.arrays["episode_index"][result.cache_indices], dtype=np.int64
            )
            source_names = (str(dataset_repo_id or "single"),)

        required_arrays = (
            result.sample_mse_gt_left,
            result.sample_mse_gt_right,
            result.sample_mse_vla_left,
            result.sample_mse_vla_right,
            result.sample_mse_vla_gt_left,
            result.sample_mse_vla_gt_right,
            result.sample_gate_w_left,
            result.sample_gate_w_right,
        )
        if any(value is None for value in required_arrays):
            raise ValueError("bimanual evaluation result is missing per-wrist arrays")
        per_source, rollups = bimanual_source_decode_metrics(
            sample_mse_gt_left=result.sample_mse_gt_left,
            sample_mse_gt_right=result.sample_mse_gt_right,
            sample_mse_vla_left=result.sample_mse_vla_left,
            sample_mse_vla_right=result.sample_mse_vla_right,
            sample_mse_vla_gt_left=result.sample_mse_vla_gt_left,
            sample_mse_vla_gt_right=result.sample_mse_vla_gt_right,
            sample_gate_w_left=result.sample_gate_w_left,
            sample_gate_w_right=result.sample_gate_w_right,
            source_indices=source_indices,
            num_sources=len(source_names),
            low_w_threshold=low_threshold,
            high_w_threshold=high_threshold,
            ranking_margin=rank_margin,
            repair_margin=repair_margin,
            low_safety_margin=low_safety_margin,
        )
        metrics.update(rollups)
        per_dataset: dict[str, dict[str, float | int]] = {}
        for source_index, source_name in enumerate(source_names):
            source_mask = source_indices == source_index
            source_mse_gt = result.sample_mse_gt[source_mask]
            source_mse_pred = result.sample_mse_pred[source_mask]
            source_mse_vla_gt = result.sample_mse_vla_gt[source_mask]
            per_dataset[source_name] = {
                "sample_count": int(np.count_nonzero(source_mask)),
                "mse_gt": float(np.mean(source_mse_gt)),
                "mse_pred": float(np.mean(source_mse_pred)),
                "mse_vla_gt": float(np.mean(source_mse_vla_gt)),
                "gt_gain": float(np.mean(source_mse_vla_gt - source_mse_gt)),
                "relative_gt_error": float(
                    np.mean(source_mse_gt)
                    / max(float(np.mean(source_mse_vla_gt)), 1e-8)
                ),
                **per_source[source_index],
            }
        metrics["per_dataset"] = per_dataset
        atomic_write_json(output_dir / "metrics.json", metrics)

        fieldnames = [
            "cache_index",
            "source",
            "source_cache_index",
            "dataset_index",
            "episode_index",
            "flow_loss",
            "mse",
            "rmse",
            "mae",
            "mse_gt",
            "mae_gt",
            "mse_pred",
            "mae_pred",
            "mse_vla_gt",
            "gt_gain",
            "relative_gt_error",
            "composite_fm",
            "tactile_change_left",
            "tactile_change_right",
            "gate_w_left",
            "gate_w_right",
            "mse_gt_left",
            "mse_gt_right",
            "mse_vla_left",
            "mse_vla_right",
            "mse_vla_gt_left",
            "mse_vla_gt_right",
            "gate_region_left",
            "gate_region_right",
            "bimanual_quadrant",
        ]
        with (output_dir / "per_sample.csv").open(
            "w", newline="", encoding="utf-8"
        ) as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            for position, cache_index in enumerate(result.cache_indices):
                left_region = _bimanual_gate_region(
                    float(result.sample_gate_w_left[position]),
                    low_threshold=low_threshold,
                    high_threshold=high_threshold,
                )
                right_region = _bimanual_gate_region(
                    float(result.sample_gate_w_right[position]),
                    low_threshold=low_threshold,
                    high_threshold=high_threshold,
                )
                writer.writerow(
                    {
                        "cache_index": int(cache_index),
                        "source": source_names[int(source_indices[position])],
                        "source_cache_index": int(local_indices[position]),
                        "dataset_index": int(dataset_indices[position]),
                        "episode_index": int(episode_indices[position]),
                        "flow_loss": float(result.sample_flow_loss[position]),
                        "mse": float(result.sample_mse[position]),
                        "rmse": float(result.sample_rmse[position]),
                        "mae": float(result.sample_mae[position]),
                        "mse_gt": float(result.sample_mse_gt[position]),
                        "mae_gt": float(result.sample_mae_gt[position]),
                        "mse_pred": float(result.sample_mse_pred[position]),
                        "mae_pred": float(result.sample_mae_pred[position]),
                        "mse_vla_gt": float(result.sample_mse_vla_gt[position]),
                        "gt_gain": float(result.sample_gt_gain[position]),
                        "relative_gt_error": float(
                            result.sample_relative_gt_error[position]
                        ),
                        "composite_fm": float(result.sample_composite_fm[position]),
                        "tactile_change_left": float(
                            result.sample_tactile_change_left[position]
                        ),
                        "tactile_change_right": float(
                            result.sample_tactile_change_right[position]
                        ),
                        "gate_w_left": float(result.sample_gate_w_left[position]),
                        "gate_w_right": float(result.sample_gate_w_right[position]),
                        "mse_gt_left": float(result.sample_mse_gt_left[position]),
                        "mse_gt_right": float(result.sample_mse_gt_right[position]),
                        "mse_vla_left": float(result.sample_mse_vla_left[position]),
                        "mse_vla_right": float(result.sample_mse_vla_right[position]),
                        "mse_vla_gt_left": float(
                            result.sample_mse_vla_gt_left[position]
                        ),
                        "mse_vla_gt_right": float(
                            result.sample_mse_vla_gt_right[position]
                        ),
                        "gate_region_left": left_region,
                        "gate_region_right": right_region,
                        "bimanual_quadrant": _bimanual_quadrant(
                            left_region, right_region
                        ),
                    }
                )
        if save_predictions and result.predictions is not None:
            np.savez(
                output_dir / "predictions.npz",
                cache_indices=result.cache_indices,
                predicted_actions=result.predictions,
            )
        if write_plots:
            history_path = _bimanual_evaluation_history_path(
                checkpoint_dir=checkpoint_dir,
                output_dir=output_dir,
                result=result,
                epoch=int(checkpoint_metadata["epoch"]),
            )
            for plot_path in plot_bimanual_diagnostics(
                history_path,
                result,
                output_dir=output_dir,
                pairs=pairs,
            ):
                print(f"plot={plot_path}")
        print(
            f"validation_samples={len(result.cache_indices)} solver={solver} "
            f"target={result.target} flow_loss={result.flow_loss:.8f} "
            f"mse={result.mse:.8f} mse_gt={result.mse_gt:.8f} "
            f"mse_pred={result.mse_pred:.8f} composite_fm={result.composite_fm:.8f}"
        )
        print(f"evaluation={output_dir}")
        return metrics
    finally:
        conditioner.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate tactile GRU-conditioned flow decoder. "
            "Primary metrics follow --target; mse_gt and mse_pred are always reported."
        )
    )
    parser.add_argument("--cache-dir", type=pathlib.Path, required=True)
    parser.add_argument("--tactile-encoder-dir", type=pathlib.Path, required=True)
    parser.add_argument("--checkpoint-dir", type=pathlib.Path, required=True)
    parser.add_argument("--output-dir", type=pathlib.Path, required=True)
    parser.add_argument("--dataset-repo-id", type=str, default=None)
    parser.add_argument("--dataset-root", type=pathlib.Path, default=None)
    parser.add_argument(
        "--tactile-window-divisor",
        type=int,
        default=None,
        help="Override window divisor (default: value stored in checkpoint metadata).",
    )
    parser.add_argument(
        "--history-stride",
        type=int,
        default=None,
        help="Override history stride (default: value stored in checkpoint metadata).",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-steps", type=int, default=5)
    parser.add_argument(
        "--target",
        choices=("gt", "predicted"),
        default=None,
        help=(
            "Primary eval target for flow_loss/mse. "
            "Default: predicted if checkpoint loss_mode=predicted, else gt. "
            "mse_gt and mse_pred are always written."
        ),
    )
    parser.add_argument(
        "--solver",
        "--decoder-solver",
        choices=("euler", "fireflow"),
        default="fireflow",
    )
    parser.add_argument("--save-predictions", action="store_true")
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--num-trajectory-samples", type=int, default=6)
    parser.add_argument("--num-episode-strips", type=int, default=6)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--prefetch-batches", type=int, default=8)
    parser.add_argument("--load-threads", type=int, default=16)
    parser.add_argument("--pipeline-prefetch", type=int, default=4)
    parser.add_argument("--image-cache-size", type=int, default=8192)
    parser.add_argument("--encode-batch-size", type=int, default=256)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    evaluate_decoder(
        cache_dir=args.cache_dir,
        tactile_encoder_dir=args.tactile_encoder_dir,
        checkpoint_dir=args.checkpoint_dir,
        output_dir=args.output_dir,
        dataset_repo_id=args.dataset_repo_id,
        dataset_root=args.dataset_root,
        tactile_window_divisor=args.tactile_window_divisor,
        history_stride=args.history_stride,
        batch_size=args.batch_size,
        num_steps=args.num_steps,
        solver=args.solver,
        target=args.target,
        save_predictions=args.save_predictions,
        write_plots=not args.no_plots,
        num_trajectory_samples=args.num_trajectory_samples,
        num_episode_strips=args.num_episode_strips,
        num_workers=args.num_workers,
        prefetch_batches=args.prefetch_batches,
        load_threads=args.load_threads,
        pipeline_prefetch=args.pipeline_prefetch,
        image_cache_size=args.image_cache_size,
    )


if __name__ == "__main__":
    main()
