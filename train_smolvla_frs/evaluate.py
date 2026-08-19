from __future__ import annotations

import argparse
import csv
import pathlib
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from train_smolvla_frs.train_frs import resolve_decode_solver
from train_smolvla_frs.utils.checkpoint import load_checkpoint
from train_smolvla_frs.utils.data import (
    CachedTactileEmbeddingBatches,
    TactileConditionedBatches,
    resolve_tactile_window,
)
from train_smolvla_frs.utils.gate_regions import GATE_BIN_SPECS
from train_smolvla_frs.utils.metrics import (
    EvalTarget,
    bimanual_source_decode_metrics,
    evaluate_split,
    gate_binned_decode_metrics,
)
from train_smolvla_frs.utils.model import FlowSolver
from train_smolvla_frs.utils.visualize import write_evaluation_plots
from utils.cache import CachedPairs, MultiCachedPairs, atomic_write_json


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
    multi_source = cache_dirs is not None
    if multi_source:
        if not cache_dirs:
            raise ValueError("cache_dirs must be non-empty when provided")
        if dataset_sources is None or len(dataset_sources) != len(cache_dirs):
            raise ValueError("dataset_sources must have one entry per cache directory")
        source_names = [str(source["repo_id"]) for source in dataset_sources]
        pairs: CachedPairs | MultiCachedPairs = MultiCachedPairs(cache_dirs, source_names=source_names)
    else:
        if cache_dir is None:
            raise ValueError("cache_dir is required for single-dataset evaluation")
        pairs = CachedPairs(cache_dir)
    model, checkpoint_metadata = load_checkpoint(checkpoint_dir)
    checkpoint_cache_digest = checkpoint_metadata.get("extra_metadata", {}).get("cache_records_sha256")
    if checkpoint_cache_digest is not None and checkpoint_cache_digest != pairs.manifest["records_sha256"]:
        raise ValueError("Checkpoint was trained from a different cache sample set.")
    expected_shape = (
        int(pairs.manifest["action_horizon"]),
        int(pairs.manifest["action_dim"]),
    )
    actual_shape = (model.config.action_horizon, model.config.action_dim)
    if actual_shape != expected_shape:
        raise ValueError(f"Checkpoint/cache action shape mismatch: {actual_shape} != {expected_shape}.")
    cache_state_dim = int(pairs.manifest.get("state_dim", 0))
    if model.config.state_conditioning and model.config.state_dim != cache_state_dim:
        raise ValueError(
            f"Checkpoint/cache state shape mismatch: {model.config.state_dim} != {cache_state_dim}."
        )

    extra = checkpoint_metadata.get("extra_metadata") or {}
    if tactile_window_divisor is None:
        tactile_window_divisor = int(extra.get("tactile_window_divisor", 1))
    if history_stride is None:
        history_stride = int(extra.get("history_stride", 1))
    loss_mode = str(extra.get("loss_mode", "gt"))
    if target is None:
        target = "predicted" if loss_mode == "predicted" else "gt"
    track_gate = loss_mode in ("gated", "bimanual_gated")
    gate_tau = float(extra["gate_tau"]) if track_gate else None
    gate_temperature = float(extra["gate_temperature"]) if track_gate else None
    rank_margin = float(extra.get("rank_margin", 0.0)) if track_gate else 0.0
    repair_margin = float(extra.get("repair_margin", 0.0)) if track_gate else 0.0
    low_safety_margin = float(extra.get("low_gate_safety_margin", 0.0)) if track_gate else 0.0
    rank_low_gate_threshold = float(extra.get("rank_low_gate_threshold", 0.3))
    rank_high_gate_threshold = float(extra.get("rank_high_gate_threshold", 0.7))
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

    if multi_source:
        if tactile_embedding_cache_root is None or not tactile_keys:
            raise ValueError("multi-dataset evaluation requires tactile embedding cache root and tactile keys")
        assert isinstance(pairs, MultiCachedPairs)
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
            build_episode_baselines=track_gate,
            return_raw_images=model.config.tactile_encoder_trainable,
            image_cache_size=image_cache_size,
            load_threads=load_threads,
            num_workers=num_workers,
            prefetch_batches=prefetch_batches,
            pipeline_prefetch=pipeline_prefetch,
        )
    else:
        assert isinstance(pairs, CachedPairs)
        conditioner = TactileConditionedBatches(
            pairs,
            tactile_encoder_dir=tactile_encoder_dir,
            tactile_window=tactile_window,
            dataset_repo_id=dataset_repo_id,
            dataset_root=dataset_root,
            history_stride=history_stride,
            build_episode_baselines=track_gate,
            num_workers=num_workers,
            prefetch_batches=prefetch_batches,
            load_threads=load_threads,
            pipeline_prefetch=pipeline_prefetch,
            image_cache_size=image_cache_size,
            encode_batch_size=encode_batch_size,
            return_raw_images=model.config.tactile_encoder_trainable,
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
            loss_mode=loss_mode,
            gate_tau=gate_tau,
            gate_temperature=gate_temperature,
            rank_margin=rank_margin,
            repair_margin=repair_margin,
            low_safety_margin=low_safety_margin,
            rank_low_gate_threshold=rank_low_gate_threshold,
            rank_high_gate_threshold=rank_high_gate_threshold,
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
        }
        if result.composite_fm is not None:
            metrics["composite_fm"] = float(result.composite_fm)
        if result.n_high_w is not None:
            metrics.update(
                {
                    "mse_gt_high_w": float(result.mse_gt_high_w),
                    "mse_gt_low_w": float(result.mse_gt_low_w),
                    "mse_pred_high_w": float(result.mse_pred_high_w),
                    "mse_pred_low_w": float(result.mse_pred_low_w),
                    "mse_vla_gt_high_w": float(result.mse_vla_gt_high_w),
                    "mse_vla_gt_low_w": float(result.mse_vla_gt_low_w),
                    "gt_gain_high_w": float(result.gt_gain_high_w),
                    "gt_gain_low_w": float(result.gt_gain_low_w),
                    "relative_gt_error_high_w": float(result.relative_gt_error_high_w),
                    "relative_gt_error_low_w": float(result.relative_gt_error_low_w),
                    "rank_penalty_high_w": float(result.rank_penalty_high_w),
                    "rank_penalty_low_w": float(result.rank_penalty_low_w),
                    "rank_satisfied_high_frac": float(result.rank_satisfied_high_frac),
                    "rank_satisfied_low_frac": float(result.rank_satisfied_low_frac),
                    "repair_penalty_high_w": float(result.repair_penalty_high_w),
                    "repair_satisfied_high_frac": float(result.repair_satisfied_high_frac),
                    "low_nearest_endpoint_mse": float(result.low_nearest_endpoint_mse),
                    "low_safety_penalty": float(result.low_safety_penalty),
                    "low_safe_frac": float(result.low_safe_frac),
                    "low_unsafe_frac": float(result.low_unsafe_frac),
                    "gate_w_mean": float(result.gate_w),
                    "gate_active_frac": float(result.gate_active_frac),
                    "gate_w_high_mean": float(result.gate_w_high_mean),
                    "gate_w_low_mean": float(result.gate_w_low_mean),
                    "gate_w_p10": float(result.gate_w_p10),
                    "gate_w_p25": float(result.gate_w_p25),
                    "gate_w_p50": float(result.gate_w_p50),
                    "gate_w_p75": float(result.gate_w_p75),
                    "gate_w_p90": float(result.gate_w_p90),
                    "tactile_change_mean": float(result.tactile_change),
                    "tactile_change_p10": float(result.tactile_change_p10),
                    "tactile_change_p25": float(result.tactile_change_p25),
                    "tactile_change_p50": float(result.tactile_change_p50),
                    "tactile_change_p75": float(result.tactile_change_p75),
                    "tactile_change_p90": float(result.tactile_change_p90),
                    "n_high_w": int(result.n_high_w),
                    "n_low_w": int(result.n_low_w),
                    "n_mid_w": int(result.n_mid_w),
                    "rank_low_gate_threshold": rank_low_gate_threshold,
                    "rank_high_gate_threshold": rank_high_gate_threshold,
                }
            )
            metrics["gate_bins"] = result.gate_bin_metrics
        if result.n_high_w_left is not None:
            for wrist in ("left", "right"):
                metrics[f"gate_w_mean_{wrist}"] = float(
                    getattr(result, f"gate_w_{wrist}")
                )
                metrics[f"tactile_change_mean_{wrist}"] = float(
                    getattr(result, f"tactile_change_{wrist}")
                )
                for quantile in ("p10", "p25", "p50", "p75", "p90"):
                    metrics[f"gate_w_{quantile}_{wrist}"] = float(
                        getattr(result, f"gate_w_{quantile}_{wrist}")
                    )
                    metrics[f"tactile_change_{quantile}_{wrist}"] = float(
                        getattr(result, f"tactile_change_{quantile}_{wrist}")
                    )
                for metric_name in (
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
                    metrics[f"{metric_name}_{wrist}"] = (
                        int(value) if metric_name.startswith("n_") else float(value)
                    )
        if isinstance(pairs, MultiCachedPairs):
            source_indices, local_indices = pairs.source_and_local_indices(result.cache_indices)
            dataset_indices = pairs.metadata_values(result.cache_indices, "dataset_index")
            episode_indices = pairs.metadata_values(result.cache_indices, "episode_index")
            bimanual_per_source: dict[int, dict[str, float | int]] | None = None
            if result.sample_gate_w_left is not None:
                bimanual_per_source, bimanual_rollups = bimanual_source_decode_metrics(
                    sample_mse_gt_left=result.sample_mse_gt_left,
                    sample_mse_gt_right=result.sample_mse_gt_right,
                    sample_mse_vla_left=result.sample_mse_vla_left,
                    sample_mse_vla_right=result.sample_mse_vla_right,
                    sample_mse_vla_gt_left=result.sample_mse_vla_gt_left,
                    sample_mse_vla_gt_right=result.sample_mse_vla_gt_right,
                    sample_gate_w_left=result.sample_gate_w_left,
                    sample_gate_w_right=result.sample_gate_w_right,
                    source_indices=source_indices,
                    num_sources=len(pairs.source_names),
                    low_w_threshold=rank_low_gate_threshold,
                    high_w_threshold=rank_high_gate_threshold,
                    ranking_margin=rank_margin,
                    repair_margin=repair_margin,
                    low_safety_margin=low_safety_margin,
                )
                metrics.update(bimanual_rollups)
            per_dataset: dict[str, dict[str, float | int]] = {}
            for source_index, source_name in enumerate(pairs.source_names):
                mask = source_indices == source_index
                if not np.any(mask):
                    continue
                source_mse_gt = result.sample_mse_gt[mask]
                source_mse_pred = result.sample_mse_pred[mask]
                source_vla_gt = result.sample_mse_vla_gt[mask]
                source_metrics: dict[str, Any] = {
                    "sample_count": int(np.sum(mask)),
                    "mse_gt": float(np.mean(source_mse_gt)),
                    "mse_pred": float(np.mean(source_mse_pred)),
                    "mse_vla_gt": float(np.mean(source_vla_gt)),
                    "gt_gain": float(np.mean(source_vla_gt - source_mse_gt)),
                    "relative_gt_error": float(np.mean(source_mse_gt) / max(float(np.mean(source_vla_gt)), 1e-8)),
                }
                if result.sample_gate_w is not None:
                    source_gate = result.sample_gate_w[mask]
                    source_high = source_gate >= rank_high_gate_threshold
                    source_low = source_gate <= rank_low_gate_threshold
                    source_mid = ~(source_high | source_low)
                    source_metrics["n_high_w"] = int(np.sum(source_high))
                    source_metrics["n_low_w"] = int(np.sum(source_low))
                    source_metrics["n_mid_w"] = int(np.sum(source_mid))
                    if np.any(source_high):
                        source_metrics["mse_gt_high_w"] = float(np.mean(source_mse_gt[source_high]))
                        source_metrics["gt_gain_high_w"] = float(np.mean((source_vla_gt - source_mse_gt)[source_high]))
                    if np.any(source_low):
                        source_metrics["mse_pred_low_w"] = float(np.mean(source_mse_pred[source_low]))
                        source_metrics["gt_gain_low_w"] = float(np.mean((source_vla_gt - source_mse_gt)[source_low]))
                        source_nearest = np.minimum(
                            source_mse_gt[source_low],
                            source_mse_pred[source_low],
                        )
                        source_metrics["low_nearest_endpoint_mse"] = float(
                            np.mean(source_nearest)
                        )
                        source_metrics["low_safety_penalty"] = float(
                            np.mean(np.maximum(source_nearest - low_safety_margin, 0.0))
                        )
                        source_metrics["low_unsafe_frac"] = float(
                            np.mean(source_nearest > low_safety_margin)
                        )
                    source_metrics["gate_bins"] = gate_binned_decode_metrics(
                        source_mse_gt,
                        source_mse_pred,
                        source_vla_gt,
                        source_gate,
                        ranking_margin=rank_margin,
                    )
                elif bimanual_per_source is not None:
                    source_metrics.update(bimanual_per_source[source_index])
                per_dataset[source_name] = source_metrics
            metrics["per_dataset"] = per_dataset
        else:
            source_indices = np.zeros(len(result.cache_indices), dtype=np.int32)
            local_indices = result.cache_indices
            dataset_indices = np.asarray(pairs.arrays["dataset_index"][result.cache_indices], dtype=np.int64)
            episode_indices = np.asarray(pairs.arrays["episode_index"][result.cache_indices], dtype=np.int64)
        atomic_write_json(output_dir / "metrics.json", metrics)

        with (output_dir / "per_sample.csv").open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(
                file,
                fieldnames=[
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
                    "tactile_change",
                    "tactile_change_left",
                    "tactile_change_right",
                    "gate_w",
                    "gate_w_left",
                    "gate_w_right",
                    "mse_gt_left",
                    "mse_gt_right",
                    "mse_vla_left",
                    "mse_vla_right",
                    "gate_region",
                    "gate_bin",
                ],
            )
            writer.writeheader()
            for position, cache_index in enumerate(result.cache_indices):
                writer.writerow(
                    {
                        "cache_index": int(cache_index),
                        "source": (
                            pairs.source_names[int(source_indices[position])]
                            if isinstance(pairs, MultiCachedPairs)
                            else str(dataset_repo_id or "single")
                        ),
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
                        "relative_gt_error": float(result.sample_relative_gt_error[position]),
                        "composite_fm": (
                            ""
                            if result.sample_composite_fm is None
                            else float(result.sample_composite_fm[position])
                        ),
                        "tactile_change": (
                            ""
                            if result.sample_tactile_change is None
                            else float(result.sample_tactile_change[position])
                        ),
                        "tactile_change_left": (
                            ""
                            if result.sample_tactile_change_left is None
                            else float(result.sample_tactile_change_left[position])
                        ),
                        "tactile_change_right": (
                            ""
                            if result.sample_tactile_change_right is None
                            else float(result.sample_tactile_change_right[position])
                        ),
                        "gate_w": ("" if result.sample_gate_w is None else float(result.sample_gate_w[position])),
                        "gate_w_left": (
                            ""
                            if result.sample_gate_w_left is None
                            else float(result.sample_gate_w_left[position])
                        ),
                        "gate_w_right": (
                            ""
                            if result.sample_gate_w_right is None
                            else float(result.sample_gate_w_right[position])
                        ),
                        "mse_gt_left": (
                            ""
                            if result.sample_mse_gt_left is None
                            else float(result.sample_mse_gt_left[position])
                        ),
                        "mse_gt_right": (
                            ""
                            if result.sample_mse_gt_right is None
                            else float(result.sample_mse_gt_right[position])
                        ),
                        "mse_vla_left": (
                            ""
                            if result.sample_mse_vla_left is None
                            else float(result.sample_mse_vla_left[position])
                        ),
                        "mse_vla_right": (
                            ""
                            if result.sample_mse_vla_right is None
                            else float(result.sample_mse_vla_right[position])
                        ),
                        "gate_region": (
                            ""
                            if result.sample_gate_w is None
                            else (
                                "low"
                                if result.sample_gate_w[position] <= rank_low_gate_threshold
                                else (
                                    "high"
                                    if result.sample_gate_w[position] >= rank_high_gate_threshold
                                    else "transition"
                                )
                            )
                        ),
                        "gate_bin": (
                            ""
                            if result.sample_gate_w is None
                            else next(
                                bin_id
                                for bin_index, (bin_id, lower, upper) in enumerate(GATE_BIN_SPECS)
                                if result.sample_gate_w[position] >= lower
                                and (
                                    result.sample_gate_w[position] < upper
                                    or (
                                        bin_index == len(GATE_BIN_SPECS) - 1 and result.sample_gate_w[position] <= upper
                                    )
                                )
                            )
                        ),
                    }
                )
        if result.predictions is not None:
            np.savez(
                output_dir / "predictions.npz",
                cache_indices=result.cache_indices,
                predicted_actions=result.predictions,
            )

        if write_plots:
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
            for plot_path in plot_paths:
                print(f"plot={plot_path}")

        print(
            f"validation_samples={len(result.cache_indices)} solver={solver} "
            f"target={result.target} flow_loss={result.flow_loss:.8f} "
            f"mse={result.mse:.8f} mse_gt={result.mse_gt:.8f} "
            f"mse_pred={result.mse_pred:.8f} mse_vla_gt={result.mse_vla_gt:.8f} "
            f"gt_gain={result.gt_gain:.8f} relative_gt_error={result.relative_gt_error:.4f}"
        )
        print(f"evaluation={output_dir}")
        return metrics
    finally:
        conditioner.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate tactile/state-conditioned flow decoder. "
            "Primary metrics follow --target; mse_gt and mse_pred are always reported."
        )
    )
    parser.add_argument(
        "--config",
        type=pathlib.Path,
        help=(
            "FRS training YAML. When set, evaluates all configured datasets with the "
            "same action/embedding caches; checkpoint defaults to <frs_training.output>/best."
        ),
    )
    parser.add_argument("--cache-dir", type=pathlib.Path)
    parser.add_argument("--tactile-encoder-dir", type=pathlib.Path)
    parser.add_argument("--checkpoint-dir", type=pathlib.Path)
    parser.add_argument("--output-dir", type=pathlib.Path)
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
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Defaults to frs_training.batch_size with --config, otherwise 64.",
    )
    parser.add_argument(
        "--num-steps",
        type=int,
        default=None,
        help="Decode steps (config mode defaults to frs_training.validation_steps; otherwise 10).",
    )
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
        default=None,
        help="Decoder solver (default: frs_training.aux_decode_solver, else euler).",
    )
    parser.add_argument("--save-predictions", action="store_true")
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--num-trajectory-samples", type=int, default=6)
    parser.add_argument("--num-episode-strips", type=int, default=6)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--prefetch-batches", type=int, default=None)
    parser.add_argument("--load-threads", type=int, default=None)
    parser.add_argument("--pipeline-prefetch", type=int, default=None)
    parser.add_argument("--image-cache-size", type=int, default=None)
    parser.add_argument("--encode-batch-size", type=int, default=256)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.config is not None:
        from train_smolvla_frs.train_frs import load_config, source_cache_dir

        config = load_config(args.config)
        datasets = config.get("datasets") or []
        action_cache = config.get("action_cache") or {}
        tactile_cache = config.get("tactile_embedding_cache") or {}
        model_config = config.get("model") or {}
        training = config.get("frs_training") or {}
        if not isinstance(datasets, list) or not datasets:
            raise ValueError("config.datasets must be a non-empty list")
        for name, value in (
            ("action_cache", action_cache),
            ("tactile_embedding_cache", tactile_cache),
            ("model", model_config),
            ("frs_training", training),
        ):
            if not isinstance(value, Mapping):
                raise ValueError(f"config.{name} must be a mapping")
        if not action_cache.get("root") or not tactile_cache.get("root"):
            raise ValueError("config action and tactile cache roots are required")
        training_output = pathlib.Path(str(training["output"])).expanduser()
        cache_dirs = [source_cache_dir(action_cache["root"], str(source["repo_id"])) for source in datasets]
        tactile_encoder_dir = (
            args.tactile_encoder_dir or pathlib.Path(str(model_config["tactile_encoder_path"])).expanduser()
        )
        checkpoint_dir = args.checkpoint_dir or training_output / "best"
        output_dir = args.output_dir or training_output / "evaluation"
        tactile_keys = tuple(str(key) for key in model_config["tactile_keys"])
        evaluate_decoder(
            cache_dir=None,
            cache_dirs=cache_dirs,
            dataset_sources=datasets,
            tactile_embedding_cache_root=pathlib.Path(str(tactile_cache["root"])).expanduser(),
            tactile_keys=tactile_keys,
            tactile_embedding_dim=int(model_config.get("tactile_embedding_dim", 512)),
            tactile_image_size=int(model_config.get("tactile_image_size", 224)),
            tactile_encoder_dir=tactile_encoder_dir,
            checkpoint_dir=checkpoint_dir,
            output_dir=output_dir,
            dataset_repo_id=None,
            dataset_root=None,
            tactile_window_divisor=(
                int(training.get("tactile_window_divisor", 1))
                if args.tactile_window_divisor is None
                else args.tactile_window_divisor
            ),
            history_stride=(
                int(training.get("history_stride", 3)) if args.history_stride is None else args.history_stride
            ),
            batch_size=(
                int(training.get("batch_size", 64))
                if args.batch_size is None
                else args.batch_size
            ),
            num_steps=(int(training.get("validation_steps", 10)) if args.num_steps is None else args.num_steps),
            solver=resolve_decode_solver(args.solver or training.get("aux_decode_solver", "euler")),
            target=args.target,
            save_predictions=args.save_predictions,
            write_plots=not args.no_plots,
            num_trajectory_samples=args.num_trajectory_samples,
            num_episode_strips=args.num_episode_strips,
            num_workers=(
                args.num_workers
                if args.num_workers is not None
                else int(training.get("num_workers", 8))
            ),
            prefetch_batches=(
                args.prefetch_batches
                if args.prefetch_batches is not None
                else int(training.get("prefetch_batches", 8))
            ),
            load_threads=(
                args.load_threads
                if args.load_threads is not None
                else int(training.get("load_threads", 8))
            ),
            pipeline_prefetch=(
                args.pipeline_prefetch
                if args.pipeline_prefetch is not None
                else int(training.get("pipeline_prefetch", 4))
            ),
            image_cache_size=(
                args.image_cache_size
                if args.image_cache_size is not None
                else int(training.get("image_cache_size", 8192))
            ),
            encode_batch_size=args.encode_batch_size,
        )
        return

    missing = [
        name
        for name, value in (
            ("--cache-dir", args.cache_dir),
            ("--tactile-encoder-dir", args.tactile_encoder_dir),
            ("--checkpoint-dir", args.checkpoint_dir),
            ("--output-dir", args.output_dir),
        )
        if value is None
    ]
    if missing:
        raise ValueError(f"legacy single-dataset evaluation requires: {', '.join(missing)}")
    evaluate_decoder(
        cache_dir=args.cache_dir,
        tactile_encoder_dir=args.tactile_encoder_dir,
        checkpoint_dir=args.checkpoint_dir,
        output_dir=args.output_dir,
        dataset_repo_id=args.dataset_repo_id,
        dataset_root=args.dataset_root,
        tactile_window_divisor=args.tactile_window_divisor,
        history_stride=args.history_stride,
        batch_size=64 if args.batch_size is None else args.batch_size,
        num_steps=10 if args.num_steps is None else args.num_steps,
        solver=resolve_decode_solver(args.solver or "euler"),
        target=args.target,
        save_predictions=args.save_predictions,
        write_plots=not args.no_plots,
        num_trajectory_samples=args.num_trajectory_samples,
        num_episode_strips=args.num_episode_strips,
        num_workers=8 if args.num_workers is None else args.num_workers,
        prefetch_batches=8 if args.prefetch_batches is None else args.prefetch_batches,
        load_threads=8 if args.load_threads is None else args.load_threads,
        pipeline_prefetch=4 if args.pipeline_prefetch is None else args.pipeline_prefetch,
        image_cache_size=8192 if args.image_cache_size is None else args.image_cache_size,
        encode_batch_size=args.encode_batch_size,
    )


if __name__ == "__main__":
    main()
