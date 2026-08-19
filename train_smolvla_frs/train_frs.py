#!/usr/bin/env python
"""Train multi-dataset tactile FRS from a YAML config.

IMPORTANT: keep module-level imports free of JAX/Flax/data loaders. Mp spawn workers
re-import this file as ``__main__`` under ``CUDA_VISIBLE_DEVICES=""``; eager ``import jax``
there causes ``CUDA_ERROR_NO_DEVICE`` spam and fails the light-import guard.
"""

from __future__ import annotations

import argparse
import math
import pathlib
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from train_smolvla_frs.utils.bimanual_schema import (  # noqa: E402
    BIMANUAL_ACTION_DIM,
    BIMANUAL_LOSS_MODE,
    BIMANUAL_OBJECTIVE_VERSION,
    LEFT_ACTION_SLICE,
    LEFT_WRIST_TOKEN_INDICES,
    RIGHT_ACTION_SLICE,
    RIGHT_WRIST_TOKEN_INDICES,
    validate_bimanual_objective_metadata,
)

LossMode = Literal["gt", "predicted", "gated", "bimanual_gated"]
DecodeSolver = Literal["euler", "fireflow"]


@dataclass(frozen=True)
class LossSettings:
    loss_mode: LossMode
    gate_tau: float
    gate_temperature: float
    gate_lambda: float


def parse_loss_settings(config: Mapping[str, Any]) -> LossSettings:
    """Parse loss settings while keeping legacy scalar-gate semantics explicit."""

    training = config.get("frs_training") or {}
    if not isinstance(training, Mapping):
        raise ValueError("config.frs_training must be a mapping")
    loss_mode = str(training.get("loss_mode", "gated"))
    if loss_mode not in ("gt", "predicted", "gated", "bimanual_gated"):
        raise ValueError(
            "frs_training.loss_mode must be 'gt', 'predicted', 'gated', or "
            f"'bimanual_gated', got {loss_mode!r}"
        )
    if loss_mode == "bimanual_gated" and "gate_lambda" in training:
        raise ValueError("frs_training.gate_lambda is not supported with bimanual_gated")
    return LossSettings(
        loss_mode=loss_mode,  # type: ignore[arg-type]
        gate_tau=float(training.get("gate_tau", 0.5)),
        gate_temperature=float(training.get("gate_temperature", 0.1)),
        gate_lambda=float(training.get("gate_lambda", 1.0)),
    )


def resolve_decode_solver(value: Any, *, default: DecodeSolver = "euler") -> DecodeSolver:
    solver = str(default if value is None else value).strip().lower()
    if solver not in ("euler", "fireflow"):
        raise ValueError(f"decode solver must be 'euler' or 'fireflow', got {value!r}")
    return solver  # type: ignore[return-value]


def resolve_optional_loss_weight(enabled: Any, weight: float) -> float:
    """Return ``weight`` when the term is on, otherwise 0.

    ``enabled is None`` keeps current configs that only set the weight.
    """

    resolved = float(weight)
    if resolved < 0:
        raise ValueError(f"loss weight must be >= 0, got {resolved}.")
    if enabled is None or bool(enabled):
        return resolved
    return 0.0


def update_early_stop_state(
    *,
    improved: bool,
    evaluation_count: int,
    no_improve_count: int,
    patience: int,
    min_evaluations: int,
) -> tuple[int, int, bool]:
    """Advance evaluation-based early stopping and return ``(evals, stale, stop)``."""

    evaluation_count += 1
    no_improve_count = 0 if improved else no_improve_count + 1
    should_stop = (
        patience > 0
        and evaluation_count >= min_evaluations
        and no_improve_count >= patience
    )
    return evaluation_count, no_improve_count, should_stop


def high_gate_rank_statistics(
    mse_gt: Any,
    mse_pred: Any,
    *,
    margin: float,
) -> tuple[float, float, float]:
    """Return mean per-sample hinge, signed gap, and satisfied fraction."""

    import numpy as np

    gt = np.asarray(mse_gt, dtype=np.float64)
    pred = np.asarray(mse_pred, dtype=np.float64)
    if gt.shape != pred.shape or gt.size == 0:
        raise ValueError(f"rank arrays must be non-empty and shape-matched: {gt.shape} != {pred.shape}")
    gap = gt - pred
    return (
        float(np.mean(np.maximum(0.0, gap + float(margin)))),
        float(np.mean(gap)),
        float(np.mean(gap + float(margin) <= 0.0)),
    )


def checkpoint_selection_key(
    metrics: Mapping[str, Any],
    *,
    loss_mode: LossMode,
    max_low_gate_unsafe_frac: float,
    min_high_gate_gain: float,
    min_high_gate_rank_satisfied: float = 0.8,
    high_gate_rank_margin: float = 0.0,
) -> tuple[float, ...]:
    """Return a lower-is-better safety key for best-checkpoint selection.

    A gated checkpoint is feasible only when low-gate outputs remain near at
    least one acceptable endpoint in every dataset, repairs toward GT in every high-gate dataset,
    keeps the mean signed rank gap below the requested margin, and reaches the
    required per-sample rank success fraction. The mean per-sample hinge remains
    a ranking objective, so violating samples cannot cancel each other. For
    infeasible models, high-gate violations are always compared before the weak
    low-gate safety constraint, matching the deployment priority.
    """

    aggregate_mse = float(metrics.get("val_mse", float("inf")))
    if loss_mode == BIMANUAL_LOSS_MODE:
        aggregate_gt_mse = float(metrics.get("val_mse_gt", float("inf")))
        if not math.isfinite(aggregate_gt_mse):
            aggregate_gt_mse = float("inf")
        wrist_values: list[tuple[float, float, float]] = []
        total_violation = 0.0
        for wrist in ("left", "right"):
            low_unsafe = float(
                metrics.get(
                    f"val_worst_dataset_low_unsafe_frac_{wrist}",
                    metrics.get(f"val_low_unsafe_frac_{wrist}", float("nan")),
                )
            )
            high_gain = float(
                metrics.get(
                    f"val_min_dataset_gt_gain_high_w_{wrist}",
                    metrics.get(f"val_gt_gain_high_w_{wrist}", float("nan")),
                )
            )
            rank_satisfied = float(
                metrics.get(
                    f"val_min_dataset_rank_satisfied_high_frac_{wrist}",
                    metrics.get(f"val_rank_satisfied_high_frac_{wrist}", float("nan")),
                )
            )
            if not all(
                math.isfinite(value)
                for value in (low_unsafe, high_gain, rank_satisfied, aggregate_gt_mse)
            ):
                return (float("inf"),) * 5
            total_violation += max(0.0, low_unsafe - float(max_low_gate_unsafe_frac))
            total_violation += max(0.0, float(min_high_gate_gain) - high_gain)
            total_violation += max(
                0.0,
                float(min_high_gate_rank_satisfied) - rank_satisfied,
            )
            wrist_values.append((high_gain, rank_satisfied, low_unsafe))
        return (
            total_violation,
            -min(value[0] for value in wrist_values),
            -min(value[1] for value in wrist_values),
            max(value[2] for value in wrist_values),
            aggregate_gt_mse,
        )
    if loss_mode != "gated":
        return (0.0, aggregate_mse)
    low_unsafe_frac = float(
        metrics.get(
            "val_worst_dataset_low_unsafe_frac",
            metrics.get("val_low_unsafe_frac", float("nan")),
        )
    )
    high_gain = float(
        metrics.get(
            "val_min_dataset_gt_gain_high_w",
            metrics.get("val_gt_gain_high_w", float("nan")),
        )
    )
    rank_hinge = float(
        metrics.get(
            "val_worst_dataset_rank_violation_high_w",
            metrics.get("val_rank_penalty_high_w", float("nan")),
        )
    )
    if not math.isfinite(rank_hinge):
        high_mse_gt = float(metrics.get("val_mse_gt_high_w", float("nan")))
        high_mse_pred = float(metrics.get("val_mse_pred_high_w", float("nan")))
        rank_hinge = (
            max(0.0, high_mse_gt - high_mse_pred + float(high_gate_rank_margin))
            if math.isfinite(high_mse_gt) and math.isfinite(high_mse_pred)
            else float("nan")
        )
    rank_gap = float(metrics.get("val_worst_dataset_rank_gap_high_w", float("nan")))
    if not math.isfinite(rank_gap):
        high_mse_gt = float(metrics.get("val_mse_gt_high_w", float("nan")))
        high_mse_pred = float(metrics.get("val_mse_pred_high_w", float("nan")))
        rank_gap = (
            high_mse_gt - high_mse_pred
            if math.isfinite(high_mse_gt) and math.isfinite(high_mse_pred)
            else float("nan")
        )
    rank_satisfied = float(
        metrics.get(
            "val_min_dataset_rank_satisfied_high_frac",
            metrics.get("val_rank_satisfied_high_frac", float("nan")),
        )
    )
    required = (low_unsafe_frac, high_gain, rank_hinge, rank_gap, rank_satisfied)
    if not all(math.isfinite(value) for value in required):
        return (
            1.0,
            1.0,
            float("inf"),
            3.0,
            float("inf"),
            float("inf"),
            float("inf"),
            float("inf"),
            1.0,
            float("inf"),
            0.0,
            aggregate_mse,
        )
    low_violation = max(0.0, low_unsafe_frac - float(max_low_gate_unsafe_frac))
    gain_violation = max(0.0, float(min_high_gate_gain) - high_gain)
    rank_gap_violation = max(0.0, rank_gap + float(high_gate_rank_margin))
    satisfaction_violation = max(0.0, float(min_high_gate_rank_satisfied) - rank_satisfied)
    high_violations = (gain_violation, rank_gap_violation, satisfaction_violation)
    high_failed_count = float(sum(value > 0.0 for value in high_violations))
    low_scale = max(float(max_low_gate_unsafe_frac), 1.0e-8)
    constraint_scale = max(abs(float(min_high_gate_gain)), float(high_gate_rank_margin), 1.0e-8)
    satisfaction_scale = max(float(min_high_gate_rank_satisfied), 1.0e-8)
    low_normalized = low_violation / low_scale
    gain_normalized = gain_violation / constraint_scale
    rank_gap_normalized = rank_gap_violation / constraint_scale
    rank_hinge_normalized = rank_hinge / constraint_scale
    satisfaction_normalized = satisfaction_violation / satisfaction_scale
    return (
        float(high_failed_count > 0.0 or low_violation > 0.0),
        float(high_failed_count > 0.0),
        max(gain_normalized, rank_gap_normalized, satisfaction_normalized),
        high_failed_count,
        rank_gap_normalized,
        satisfaction_normalized,
        gain_normalized,
        rank_hinge_normalized,
        float(low_violation > 0.0),
        low_normalized,
        -high_gain,
        aggregate_mse,
    )


def checkpoint_selection_sentinel(*, loss_mode: LossMode) -> tuple[float, ...]:
    """Return an all-infinite initial key with the active mode's tuple shape."""

    key_length = 5 if loss_mode == BIMANUAL_LOSS_MODE else 12
    return (float("inf"),) * key_length


def checkpoint_specialist_keys(
    metrics: Mapping[str, Any],
    *,
    high_gate_rank_margin: float = 0.0,
) -> dict[str, tuple[float, ...]]:
    """Return independent lower-is-better keys for specialist checkpoints."""

    aggregate_mse = float(metrics.get("val_mse", float("inf")))
    low_unsafe_frac = float(
        metrics.get(
            "val_worst_dataset_low_unsafe_frac",
            metrics.get("val_low_unsafe_frac", float("nan")),
        )
    )
    low_safety_penalty = float(
        metrics.get(
            "val_worst_dataset_low_safety_penalty",
            metrics.get("val_low_safety_penalty", float("nan")),
        )
    )
    high_gain = float(
        metrics.get(
            "val_min_dataset_gt_gain_high_w",
            metrics.get("val_gt_gain_high_w", float("nan")),
        )
    )
    rank_violation = float(
        metrics.get(
            "val_worst_dataset_rank_violation_high_w",
            metrics.get("val_rank_penalty_high_w", float("nan")),
        )
    )
    if not math.isfinite(rank_violation):
        high_mse_gt = float(metrics.get("val_mse_gt_high_w", float("nan")))
        high_mse_pred = float(metrics.get("val_mse_pred_high_w", float("nan")))
        rank_violation = (
            max(0.0, high_mse_gt - high_mse_pred + float(high_gate_rank_margin))
            if math.isfinite(high_mse_gt) and math.isfinite(high_mse_pred)
            else float("nan")
        )
    rank_gap = float(metrics.get("val_worst_dataset_rank_gap_high_w", float("nan")))
    if not math.isfinite(rank_gap):
        high_mse_gt = float(metrics.get("val_mse_gt_high_w", float("nan")))
        high_mse_pred = float(metrics.get("val_mse_pred_high_w", float("nan")))
        rank_gap = (
            high_mse_gt - high_mse_pred
            if math.isfinite(high_mse_gt) and math.isfinite(high_mse_pred)
            else float("nan")
        )
    rank_satisfied = float(
        metrics.get(
            "val_min_dataset_rank_satisfied_high_frac",
            metrics.get("val_rank_satisfied_high_frac", float("nan")),
        )
    )
    low_key = low_unsafe_frac if math.isfinite(low_unsafe_frac) else float("inf")
    low_penalty_key = (
        low_safety_penalty if math.isfinite(low_safety_penalty) else float("inf")
    )
    gain_key = -high_gain if math.isfinite(high_gain) else float("inf")
    rank_key = rank_violation if math.isfinite(rank_violation) else float("inf")
    rank_gap_key = rank_gap if math.isfinite(rank_gap) else float("inf")
    rank_satisfied_key = -rank_satisfied if math.isfinite(rank_satisfied) else float("inf")
    return {
        "best_rank": (rank_gap_key, rank_key, rank_satisfied_key, gain_key, low_key, aggregate_mse),
        "best_low_preservation": (
            low_key,
            low_penalty_key,
            rank_key,
            rank_gap_key,
            rank_satisfied_key,
            gain_key,
            aggregate_mse,
        ),
        "best_gain": (gain_key, rank_key, rank_gap_key, rank_satisfied_key, low_key, aggregate_mse),
    }


def _existing_run_artifacts(output_dir: pathlib.Path) -> tuple[pathlib.Path, ...]:
    candidates = (
        output_dir / "history.csv",
        output_dir / "best" / "checkpoint.json",
        output_dir / "best_feasible" / "checkpoint.json",
        output_dir / "best_rank" / "checkpoint.json",
        output_dir / "best_low_preservation" / "checkpoint.json",
        output_dir / "best_gain" / "checkpoint.json",
        output_dir / "last" / "checkpoint.json",
    )
    return tuple(path for path in candidates if path.exists())


def _validate_resume_cache(
    resume_metadata: Mapping[str, Any],
    cache_manifest: Mapping[str, Any],
) -> None:
    extra = resume_metadata.get("extra_metadata")
    if not isinstance(extra, Mapping):
        raise ValueError("resume checkpoint is missing cache provenance metadata")
    expected_digest = cache_manifest.get("records_sha256")
    if extra.get("cache_records_sha256") != expected_digest:
        raise ValueError(
            "resume checkpoint/cache record digest mismatch: "
            f"{extra.get('cache_records_sha256')!r} != {expected_digest!r}"
        )
    expected_configuration = cache_manifest.get("configuration")
    if extra.get("cache_configuration") != expected_configuration:
        raise ValueError("resume checkpoint was trained with a different action-cache configuration")


def _validate_resume_loss_objective(
    resume_metadata: Mapping[str, Any],
    *,
    loss_mode: LossMode,
) -> None:
    """Validate only the new objective contract, preserving legacy resume behavior."""

    if loss_mode != BIMANUAL_LOSS_MODE:
        return
    extra = resume_metadata.get("extra_metadata")
    if not isinstance(extra, Mapping):
        raise ValueError("resume checkpoint is missing bimanual loss objective metadata")
    validate_bimanual_objective_metadata(extra)


def _resolve_resume_dir(
    *,
    output_dir: pathlib.Path,
    resume: bool,
    resume_from: pathlib.Path | None,
) -> pathlib.Path | None:
    if resume_from is not None:
        return resume_from
    if resume:
        return output_dir / "last"
    return None


def _history_fieldnames_for_write(
    history_path: pathlib.Path,
    *,
    default_fields: Sequence[str],
    append: bool,
) -> list[str]:
    """Keep an existing CSV header stable when resuming a legacy run."""

    if not append or not history_path.exists():
        return list(default_fields)
    import csv

    with history_path.open(encoding="utf-8", newline="") as file:
        existing = csv.DictReader(file).fieldnames
    return list(existing) if existing else list(default_fields)


def train_decoder(
    *,
    cache_dir: pathlib.Path | None,
    tactile_encoder_dir: pathlib.Path,
    output_dir: pathlib.Path,
    dataset_repo_id: str | None,
    dataset_root: pathlib.Path | None,
    tactile_window_divisor: int,
    history_stride: int,
    loss_mode: LossMode,
    gate_tau: float,
    gate_temperature: float,
    gate_lambda: float,
    aux_decode_weight: float,
    aux_decode_steps: int,
    aux_decode_solver: DecodeSolver,
    low_gate_safety_weight: float,
    low_gate_safety_margin: float,
    rank_weight: float,
    rank_margin: float,
    repair_weight: float,
    repair_margin: float,
    rank_low_gate_threshold: float,
    rank_high_gate_threshold: float,
    state_conditioning: bool,
    state_dropout_rate: float,
    model_dim: int,
    depth: int,
    num_heads: int,
    mlp_ratio: int,
    learning_rate: float,
    weight_decay: float,
    grad_clip_norm: float | None,
    warmup_epochs: int,
    lr_reference_dim: int | None,
    min_learning_rate_ratio: float,
    cosine_decay: bool,
    batch_size: int,
    epochs: int,
    validation_steps: int,
    eval_every: int,
    seed: int,
    write_plots: bool,
    num_workers: int,
    prefetch_batches: int,
    load_threads: int,
    pipeline_prefetch: int,
    image_cache_size: int,
    encode_batch_size: int,
    resume: bool = False,
    resume_from: pathlib.Path | None = None,
    init_from: pathlib.Path | None = None,
    cache_dirs: Sequence[pathlib.Path] | None = None,
    dataset_sources: Sequence[Mapping[str, Any]] | None = None,
    tactile_embedding_cache_root: pathlib.Path | None = None,
    tactile_keys: Sequence[str] | None = None,
    tactile_embedding_dim: int = 512,
    tactile_image_size: int = 224,
    tactile_num_tokens: int = 4,
    train_tactile_encoder: bool = False,
    tactile_encode_microbatch_size: int = 8,
    best_max_low_gate_unsafe_frac: float = 0.1,
    best_min_high_gate_gain: float = 0.0,
    best_min_high_gate_rank_satisfied: float = 0.8,
    dataset_balanced_sampling: bool = False,
    dataset_balanced_loss: bool = False,
    early_stop_patience: int = 0,
    early_stop_min_evals: int = 0,
    high_gate_rank_aggregation: str = "balanced_mean",
    high_gate_rank_hard_fraction: float = 0.3,
    high_gate_rank_worst_beta: float = 20.0,
    high_gate_rank_source_weights: Mapping[str, float] | None = None,
) -> None:
    import csv
    import json

    import jax
    import jax.numpy as jnp
    import numpy as np
    from flax import nnx

    from train_smolvla_frs.utils.checkpoint import (
        CHECKPOINT_NAME,
        load_checkpoint,
        load_optimizer_state,
        restore_optimizer_state,
        save_checkpoint,
    )
    from train_smolvla_frs.utils.data import (
        CachedTactileEmbeddingBatches,
        TactileConditionedBatches,
        gate_weights_from_change,
        resolve_tactile_window,
    )
    from train_smolvla_frs.utils.gate_regions import GATE_BIN_SPECS
    from train_smolvla_frs.utils.history_plot import plot_training_history
    from train_smolvla_frs.utils.metrics import (
        bimanual_source_decode_metrics,
        evaluate_split,
    )
    from train_smolvla_frs.utils.model import (
        DEFAULT_GRU_HIDDEN_DIM,
        DecoderConfig,
        TactileConditionedFlowDecoder,
        make_optimizer,
        resolve_peak_learning_rate,
        train_step,
    )
    from utils.cache import CachedPairs, MultiCachedPairs

    history_fields = [
        "epoch",
        "train_loss_total",
        "train_loss_gt_fm",
        "train_loss_vla_fm",
        "train_loss_composite_fm",
        "train_loss_low_safety",
        "train_loss_decode",
        "train_loss_rank",
        "train_loss_repair",
        "train_gate_w_left",
        "train_gate_w_right",
        # Backward-compatible alias for readers of pre-refactor histories.
        "train_flow_loss",
        "val_flow_loss",
        "val_mse",
        "val_rmse",
        "val_mae",
        "val_flow_loss_gt",
        "val_mse_gt",
        "val_rmse_gt",
        "val_mae_gt",
        "val_flow_loss_pred",
        "val_mse_pred",
        "val_rmse_pred",
        "val_mae_pred",
        "val_mse_vla_gt",
        "val_gt_gain",
        "val_relative_gt_error",
        "eval_target",
        "val_mse_gt_high_w",
        "val_mse_gt_low_w",
        "val_mse_pred_high_w",
        "val_mse_pred_low_w",
        "val_mse_vla_gt_high_w",
        "val_mse_vla_gt_low_w",
        "val_gt_gain_high_w",
        "val_gt_gain_low_w",
        "val_relative_gt_error_high_w",
        "val_relative_gt_error_low_w",
        "val_rank_penalty_high_w",
        "val_rank_penalty_low_w",
        "val_rank_satisfied_high_frac",
        "val_rank_satisfied_low_frac",
        "val_repair_penalty_high_w",
        "val_repair_satisfied_high_frac",
        "val_low_nearest_endpoint_mse",
        "val_low_safety_penalty",
        "val_low_safe_frac",
        "val_low_unsafe_frac",
        "val_gate_w",
        "val_gate_active_frac",
        "val_gate_w_high_mean",
        "val_gate_w_low_mean",
        "val_gate_w_p10",
        "val_gate_w_p25",
        "val_gate_w_p50",
        "val_gate_w_p75",
        "val_gate_w_p90",
        "val_tactile_change",
        "val_tactile_change_high_mean",
        "val_tactile_change_low_mean",
        "val_tactile_change_p10",
        "val_tactile_change_p25",
        "val_tactile_change_p50",
        "val_tactile_change_p75",
        "val_tactile_change_p90",
        "val_n_high_w",
        "val_n_low_w",
        "val_n_mid_w",
        "val_worst_dataset_low_safety_penalty",
        "val_worst_dataset_low_unsafe_frac",
        "val_min_dataset_gt_gain_high_w",
        "val_worst_dataset_rank_violation_high_w",
        "val_worst_dataset_rank_gap_high_w",
        "val_min_dataset_rank_satisfied_high_frac",
        "checkpoint_selection_key",
        "checkpoint_selection_feasible",
        "early_stop_no_improve_evals",
    ]
    bimanual_validation_metric_names = (
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
    )
    for wrist in ("left", "right"):
        history_fields.extend(
            f"val_{metric_name}_{wrist}"
            for metric_name in bimanual_validation_metric_names
        )
    gate_bin_metric_names = (
        "n",
        "mse_gt",
        "mse_pred",
        "mse_vla_gt",
        "gt_gain",
        "relative_gt_error",
        "rank_satisfied_frac",
    )
    for bin_id, _, _ in GATE_BIN_SPECS:
        history_fields.extend(f"val_gate_bin_{bin_id}_{metric_name}" for metric_name in gate_bin_metric_names)
    source_metric_names = (
        "low_nearest_endpoint_mse",
        "low_safety_penalty",
        "low_unsafe_frac",
        "gt_gain_high_w",
        "rank_gap_high_w",
        "rank_hinge_high_w",
        "rank_satisfied_high_frac",
    )
    bimanual_source_metric_names = (
        "low_nearest_endpoint_mse",
        "low_safety_penalty",
        "low_unsafe_frac",
        "gt_gain_high_w",
        "rank_gap_high_w",
        "rank_hinge_high_w",
        "rank_satisfied_high_frac",
        "repair_penalty_high_w",
        "repair_satisfied_high_frac",
        "n_high_w",
        "n_low_w",
        "n_mid_w",
    )
    bimanual_rollup_metric_names = (
        "worst_dataset_low_safety_penalty",
        "worst_dataset_low_unsafe_frac",
        "min_dataset_gt_gain_high_w",
        "worst_dataset_rank_violation_high_w",
        "worst_dataset_rank_gap_high_w",
        "min_dataset_rank_satisfied_high_frac",
        "worst_dataset_repair_penalty_high_w",
        "min_dataset_repair_satisfied_high_frac",
    )
    for wrist in ("left", "right"):
        history_fields.extend(
            f"val_{metric_name}_{wrist}"
            for metric_name in bimanual_rollup_metric_names
        )
    for source_index in range(len(dataset_sources or ())):
        history_fields.extend(
            f"val_source_{source_index}_{metric_name}"
            for metric_name in source_metric_names
        )
        for wrist in ("left", "right"):
            history_fields.extend(
                f"val_source_{source_index}_{metric_name}_{wrist}"
                for metric_name in bimanual_source_metric_names
            )

    def _blank_history_row(epoch: int, **filled: float | int | str) -> dict[str, float | int | str]:
        row: dict[str, float | int | str] = dict.fromkeys(history_fields, "")
        row["epoch"] = epoch
        row.update(filled)
        return row

    if epochs <= 0 or batch_size <= 0:
        raise ValueError("epochs and batch_size must be positive.")
    if warmup_epochs < 0:
        raise ValueError("warmup_epochs must be non-negative.")
    if not 0.0 <= min_learning_rate_ratio <= 1.0:
        raise ValueError("min_learning_rate_ratio must be in [0, 1].")
    if loss_mode not in ("gt", "predicted", "gated", "bimanual_gated"):
        raise ValueError(
            "loss_mode must be 'gt', 'predicted', 'gated', or "
            f"'bimanual_gated', got {loss_mode!r}."
        )
    eval_target = "predicted" if loss_mode == "predicted" else "gt"
    if gate_temperature <= 0:
        raise ValueError(f"gate_temperature must be positive, got {gate_temperature}.")
    if gate_lambda < 0:
        raise ValueError(f"gate_lambda must be non-negative, got {gate_lambda}.")
    if rank_weight < 0:
        raise ValueError(f"rank_weight must be non-negative, got {rank_weight}.")
    if rank_margin < 0:
        raise ValueError(f"rank_margin must be non-negative, got {rank_margin}.")
    if repair_weight < 0:
        raise ValueError(f"repair_weight must be non-negative, got {repair_weight}.")
    if repair_margin < 0:
        raise ValueError(f"repair_margin must be non-negative, got {repair_margin}.")
    if low_gate_safety_weight < 0:
        raise ValueError("low_gate_safety_weight must be non-negative.")
    if low_gate_safety_margin < 0:
        raise ValueError("low_gate_safety_margin must be non-negative.")
    if not 0.0 <= rank_low_gate_threshold < 0.5:
        raise ValueError("rank_low_gate_threshold must be in [0, 0.5), got " f"{rank_low_gate_threshold}.")
    if not 0.5 < rank_high_gate_threshold <= 1.0:
        raise ValueError("rank_high_gate_threshold must be in (0.5, 1], got " f"{rank_high_gate_threshold}.")
    if rank_low_gate_threshold >= rank_high_gate_threshold:
        raise ValueError("rank low-gate threshold must be below the high-gate threshold.")
    if not 0.0 <= best_max_low_gate_unsafe_frac <= 1.0:
        raise ValueError("best_max_low_gate_unsafe_frac must be in [0, 1].")
    if not 0.0 <= best_min_high_gate_rank_satisfied <= 1.0:
        raise ValueError("best_min_high_gate_rank_satisfied must be in [0, 1].")
    if early_stop_patience < 0 or early_stop_min_evals < 0:
        raise ValueError("early-stop patience and minimum evaluations must be non-negative")
    if dataset_balanced_loss and not dataset_balanced_sampling:
        raise ValueError("dataset_balanced_loss requires dataset_balanced_sampling")
    if high_gate_rank_aggregation not in ("balanced_mean", "worst_source_cvar"):
        raise ValueError(
            "high_gate_rank_aggregation must be 'balanced_mean' or 'worst_source_cvar'"
        )
    if not 0.0 < high_gate_rank_hard_fraction <= 1.0:
        raise ValueError("high_gate_rank_hard_fraction must be in (0, 1]")
    if high_gate_rank_worst_beta <= 0.0:
        raise ValueError("high_gate_rank_worst_beta must be positive")
    if loss_mode not in ("gated", "bimanual_gated") and (
        rank_weight != 0 or repair_weight != 0
    ):
        raise ValueError(
            "rank_weight and repair_weight are only supported with "
            "loss_mode='gated' or 'bimanual_gated'."
        )
    if eval_every <= 0:
        raise ValueError(f"eval_every must be positive, got {eval_every}.")
    if tactile_num_tokens <= 0:
        raise ValueError(f"tactile_num_tokens must be positive, got {tactile_num_tokens}.")
    if tactile_encode_microbatch_size <= 0:
        raise ValueError("tactile_encode_microbatch_size must be positive.")
    if not 0.0 <= state_dropout_rate < 1.0:
        raise ValueError("state_dropout_rate must be in [0, 1).")

    resume_dir = _resolve_resume_dir(output_dir=output_dir, resume=resume, resume_from=resume_from)
    if resume_dir is not None and init_from is not None:
        raise ValueError("init_from cannot be combined with resume/resume_from")
    if resume_dir is None:
        existing_artifacts = _existing_run_artifacts(output_dir)
        if existing_artifacts:
            raise FileExistsError(
                "refusing to start a fresh run in an existing FRS output directory; "
                f"found {list(existing_artifacts)}. Choose a new output or enable resume."
            )
    start_epoch = 1
    resume_metadata: dict | None = None
    init_metadata: dict | None = None
    resumed_opt_state = None
    resumed_opt_step: int | None = None
    if resume_dir is not None:
        if not (resume_dir / CHECKPOINT_NAME).exists():
            raise FileNotFoundError(f"Resume checkpoint not found: {resume_dir}")
        model, resume_metadata = load_checkpoint(resume_dir)
        _validate_resume_loss_objective(resume_metadata, loss_mode=loss_mode)
        resumed_opt_state, resumed_opt_step = load_optimizer_state(resume_dir)
        start_epoch = int(resume_metadata["epoch"]) + 1
        print(
            f"resuming from {resume_dir} epoch={resume_metadata['epoch']} "
            f"next_epoch={start_epoch} has_opt_state={resumed_opt_state is not None}",
            flush=True,
        )
        if start_epoch > epochs:
            print(
                f"already finished: last epoch {resume_metadata['epoch']} >= --epochs {epochs}",
                flush=True,
            )
            return
    elif init_from is not None:
        init_from = pathlib.Path(init_from).expanduser()
        if not (init_from / CHECKPOINT_NAME).exists():
            raise FileNotFoundError(f"Initialization checkpoint not found: {init_from}")
        model, init_metadata = load_checkpoint(init_from)
        print(
            f"initializing model parameters from {init_from} "
            f"source_epoch={init_metadata.get('epoch')} optimizer=fresh epoch=1",
            flush=True,
        )

    print(f"jax_devices={jax.devices()}", flush=True)
    if not any(d.platform == "gpu" for d in jax.devices()):
        print(
            "WARNING: no JAX GPU device visible; ResNet encode + training will run on CPU "
            "(very slow). Check nvidia-smi / CUDA_VISIBLE_DEVICES.",
            flush=True,
        )

    use_cached_embeddings = cache_dirs is not None
    if use_cached_embeddings:
        if not cache_dirs:
            raise ValueError("cache_dirs must be non-empty when provided")
        if dataset_sources is None or len(dataset_sources) != len(cache_dirs):
            raise ValueError("dataset_sources must have one entry per cache_dirs entry")
        if tactile_embedding_cache_root is None:
            raise ValueError("tactile_embedding_cache_root is required for multi-source FRS")
        if not tactile_keys:
            raise ValueError("tactile_keys is required for multi-source FRS")
        source_names = [str(source["repo_id"]) for source in dataset_sources]
        pairs = MultiCachedPairs(cache_dirs, source_names=source_names)
    else:
        if cache_dir is None:
            raise ValueError("cache_dir is required when cache_dirs is not provided")
        pairs = CachedPairs(cache_dir)
    if (
        loss_mode == "bimanual_gated"
        and int(pairs.manifest["action_dim"]) != BIMANUAL_ACTION_DIM
    ):
        raise ValueError(
            f"loss_mode='bimanual_gated' requires action_dim={BIMANUAL_ACTION_DIM}, got "
            f"{pairs.manifest['action_dim']}."
        )
    source_rank_weight_values = np.ones(
        (len(pairs.sources) if isinstance(pairs, MultiCachedPairs) else 1,),
        dtype=np.float32,
    )
    if high_gate_rank_source_weights:
        if not isinstance(pairs, MultiCachedPairs):
            raise ValueError("high_gate_rank_source_weights requires multi-dataset training")
        unknown_sources = set(high_gate_rank_source_weights) - set(pairs.source_names)
        if unknown_sources:
            raise ValueError(
                f"high_gate_rank_source_weights contains unknown sources: {sorted(unknown_sources)}"
            )
        source_rank_weight_values = np.asarray(
            [float(high_gate_rank_source_weights.get(name, 1.0)) for name in pairs.source_names],
            dtype=np.float32,
        )
    if np.any(source_rank_weight_values < 0.0) or not np.any(source_rank_weight_values > 0.0):
        raise ValueError("high-gate rank source weights must be non-negative with at least one positive")
    if high_gate_rank_aggregation == "worst_source_cvar":
        print(
            "high_gate_rank="
            f"{high_gate_rank_aggregation} hard_fraction={high_gate_rank_hard_fraction:g} "
            f"worst_beta={high_gate_rank_worst_beta:g} "
            f"source_weights={source_rank_weight_values.tolist()}",
            flush=True,
        )
    if resume_metadata is not None:
        _validate_resume_cache(resume_metadata, pairs.manifest)
        resume_extra = resume_metadata.get("extra_metadata") or {}
        stored_rank_weight = float(resume_extra.get("rank_weight", 0.0))
        stored_rank_margin = float(resume_extra.get("rank_margin", 0.0))
        stored_repair_weight = float(resume_extra.get("repair_weight", 0.0))
        stored_repair_margin = float(resume_extra.get("repair_margin", 0.0))
        stored_low_safety_weight = float(resume_extra.get("low_gate_safety_weight", 0.0))
        stored_low_safety_margin = float(resume_extra.get("low_gate_safety_margin", 0.0))
        stored_rank_low_gate_threshold = float(resume_extra.get("rank_low_gate_threshold", 0.5))
        stored_rank_high_gate_threshold = float(resume_extra.get("rank_high_gate_threshold", 0.5))
        stored_weighting_version = int(resume_extra.get("loss_weighting_version", 1))
        stored_state_dropout_rate = float(resume_extra.get("state_dropout_rate", 0.0))
        stored_low_gate_limit = float(
            resume_extra.get(
                "best_max_low_gate_unsafe_frac",
                best_max_low_gate_unsafe_frac,
            )
        )
        stored_min_high_gain = float(resume_extra.get("best_min_high_gate_gain", best_min_high_gate_gain))
        stored_min_rank_satisfied = float(
            resume_extra.get("best_min_high_gate_rank_satisfied", best_min_high_gate_rank_satisfied)
        )
        stored_balanced_sampling = bool(
            resume_extra.get("dataset_balanced_sampling", False)
        )
        stored_balanced_loss = bool(
            resume_extra.get("dataset_balanced_loss", False)
        )
        stored_rank_aggregation = str(
            resume_extra.get("high_gate_rank_aggregation", "balanced_mean")
        )
        stored_rank_hard_fraction = float(
            resume_extra.get("high_gate_rank_hard_fraction", 0.3)
        )
        stored_rank_worst_beta = float(
            resume_extra.get("high_gate_rank_worst_beta", 20.0)
        )
        stored_source_rank_weights = np.asarray(
            resume_extra.get(
                "high_gate_rank_source_weight_values",
                np.ones_like(source_rank_weight_values).tolist(),
            ),
            dtype=np.float32,
        )
        compatible_weighting_version = stored_weighting_version == 7
        if (
            not compatible_weighting_version
            or stored_rank_weight != rank_weight
            or stored_rank_margin != rank_margin
            or stored_repair_weight != repair_weight
            or stored_repair_margin != repair_margin
            or stored_low_safety_weight != low_gate_safety_weight
            or stored_low_safety_margin != low_gate_safety_margin
            or stored_state_dropout_rate != state_dropout_rate
            or stored_rank_low_gate_threshold != rank_low_gate_threshold
            or stored_rank_high_gate_threshold != rank_high_gate_threshold
            or stored_low_gate_limit != best_max_low_gate_unsafe_frac
            or stored_min_high_gain != best_min_high_gate_gain
            or stored_min_rank_satisfied != best_min_high_gate_rank_satisfied
            or stored_balanced_sampling != dataset_balanced_sampling
            or stored_balanced_loss != dataset_balanced_loss
            or stored_rank_aggregation != high_gate_rank_aggregation
            or stored_rank_hard_fraction != high_gate_rank_hard_fraction
            or stored_rank_worst_beta != high_gate_rank_worst_beta
            or not np.array_equal(stored_source_rank_weights, source_rank_weight_values)
        ):
            raise ValueError(
                "Resume checkpoint constraint objective differs from this run: "
                "checkpoint="
                f"(rank_weight={stored_rank_weight:g}, rank_margin={stored_rank_margin:g}, "
                f"repair_weight={stored_repair_weight:g}, "
                f"repair_margin={stored_repair_margin:g}, weighting_v={stored_weighting_version}, "
                f"low_safety_weight={stored_low_safety_weight:g}, "
                f"low_safety_margin={stored_low_safety_margin:g}, "
                f"state_dropout_rate={stored_state_dropout_rate:g}, "
                f"rank_gate=[{stored_rank_low_gate_threshold:g},"
                f"{stored_rank_high_gate_threshold:g}], "
                f"low_gate_limit={stored_low_gate_limit:g}, min_high_gain={stored_min_high_gain:g}, "
                f"min_rank_satisfied={stored_min_rank_satisfied:g}, "
                f"balanced_sampling={stored_balanced_sampling}, "
                f"balanced_loss={stored_balanced_loss}, "
                f"rank_aggregation={stored_rank_aggregation}, "
                f"rank_hard_fraction={stored_rank_hard_fraction:g}, "
                f"rank_worst_beta={stored_rank_worst_beta:g}, "
                f"source_rank_weights={stored_source_rank_weights.tolist()}) "
                "requested="
                f"(rank_weight={rank_weight:g}, rank_margin={rank_margin:g}, "
                f"repair_weight={repair_weight:g}, repair_margin={repair_margin:g}, weighting_v=7, "
                f"low_safety_weight={low_gate_safety_weight:g}, "
                f"low_safety_margin={low_gate_safety_margin:g}, "
                f"state_dropout_rate={state_dropout_rate:g}, "
                f"rank_gate=[{rank_low_gate_threshold:g},{rank_high_gate_threshold:g}], "
                f"low_unsafe_frac_limit={best_max_low_gate_unsafe_frac:g}, "
                f"min_high_gain={best_min_high_gate_gain:g}, "
                f"min_rank_satisfied={best_min_high_gate_rank_satisfied:g}, "
                f"balanced_sampling={dataset_balanced_sampling}, "
                f"balanced_loss={dataset_balanced_loss}, "
                f"rank_aggregation={high_gate_rank_aggregation}, "
                f"rank_hard_fraction={high_gate_rank_hard_fraction:g}, "
                f"rank_worst_beta={high_gate_rank_worst_beta:g}, "
                f"source_rank_weights={source_rank_weight_values.tolist()}). "
                "Start a fresh run in a new frs_training.output directory."
            )
    if init_metadata is not None:
        _validate_resume_cache(init_metadata, pairs.manifest)
    action_horizon = int(pairs.manifest["action_horizon"])
    tactile_window = resolve_tactile_window(
        action_horizon=action_horizon,
        window_divisor=tactile_window_divisor,
    )
    if use_cached_embeddings:
        assert dataset_sources is not None
        assert tactile_embedding_cache_root is not None
        assert tactile_keys is not None
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
            build_episode_baselines=(loss_mode in ("gated", "bimanual_gated")),
            return_raw_images=train_tactile_encoder,
            image_cache_size=image_cache_size,
            load_threads=load_threads,
            num_workers=num_workers,
            prefetch_batches=prefetch_batches,
            pipeline_prefetch=pipeline_prefetch,
        )
    else:
        conditioner = TactileConditionedBatches(
            pairs,
            tactile_encoder_dir=tactile_encoder_dir,
            tactile_window=tactile_window,
            dataset_repo_id=dataset_repo_id,
            dataset_root=dataset_root,
            history_stride=history_stride,
            build_episode_baselines=(loss_mode in ("gated", "bimanual_gated")),
            num_workers=num_workers,
            prefetch_batches=prefetch_batches,
            load_threads=load_threads,
            pipeline_prefetch=pipeline_prefetch,
            image_cache_size=image_cache_size,
            encode_batch_size=encode_batch_size,
            return_raw_images=train_tactile_encoder,
        )
    decoder_config = DecoderConfig(
        action_dim=int(pairs.manifest["action_dim"]),
        action_horizon=action_horizon,
        tactile_window=tactile_window,
        gru_hidden_dim=DEFAULT_GRU_HIDDEN_DIM,
        resnet_embedding_dim=conditioner.resnet_embedding_dim,
        model_dim=model_dim,
        depth=depth,
        num_heads=num_heads,
        mlp_ratio=mlp_ratio,
        num_tactile_tokens=tactile_num_tokens,
        state_dim=int(pairs.manifest.get("state_dim", 0)),
        state_conditioning=state_conditioning,
        tactile_encoder_trainable=train_tactile_encoder,
        tactile_image_size=tactile_image_size,
        tactile_encode_microbatch_size=tactile_encode_microbatch_size,
    )
    if resume_metadata is None and init_metadata is None:
        tactile_resnet_variables = None
        if train_tactile_encoder:
            from train_encoder.utils.checkpoint import load_tactile_encoder

            encoder_bundle = load_tactile_encoder(tactile_encoder_dir)
            tactile_resnet_variables = encoder_bundle.params.get("tactile_resnet")
            if not isinstance(tactile_resnet_variables, Mapping):
                raise KeyError(
                    "Tactile encoder checkpoint is missing tactile_resnet variables."
                )
        model = TactileConditionedFlowDecoder(
            decoder_config,
            rngs=nnx.Rngs(seed),
            tactile_resnet_variables=tactile_resnet_variables,
        )
    else:
        source_metadata = resume_metadata if resume_metadata is not None else init_metadata
        assert source_metadata is not None
        ckpt_config = DecoderConfig(**source_metadata["decoder_config"])
        if dataclasses_asdict_mismatch := _config_diff(ckpt_config, decoder_config):
            if init_metadata is not None:
                raise ValueError(
                    "init_from decoder config differs from this run: "
                    f"{dataclasses_asdict_mismatch}"
                )
            print(
                "warning: CLI decoder config differs from resume checkpoint; "
                f"keeping checkpoint weights. diffs={dataclasses_asdict_mismatch}",
                flush=True,
            )
        # ``model`` already loaded above.
    if model.config.tactile_encoder_trainable != train_tactile_encoder:
        raise ValueError(
            "Checkpoint/config tactile encoder mode mismatch: "
            f"checkpoint trainable={model.config.tactile_encoder_trainable}, "
            f"requested trainable={train_tactile_encoder}. Start a fresh run "
            "with a matching model.freeze_tactile_encoder value."
        )
    source_count = len(pairs.sources) if isinstance(pairs, MultiCachedPairs) else 1
    if (dataset_balanced_sampling or dataset_balanced_loss) and not isinstance(
        pairs, MultiCachedPairs
    ):
        raise ValueError("dataset-balanced sampling/loss requires multiple action caches")
    train_samples = len(pairs.indices("train"))
    if isinstance(pairs, MultiCachedPairs):
        steps_per_epoch = pairs.batch_count(
            "train",
            batch_size=batch_size,
            source_balanced=dataset_balanced_sampling,
        )
        if dataset_balanced_sampling:
            source_train_counts = [len(source.indices("train")) for source in pairs.sources]
            source_quotas = pairs.source_batch_quotas(batch_size).tolist()
            print(
                f"source_balanced_batches counts={source_train_counts} "
                f"per_batch={source_quotas} steps_per_epoch={steps_per_epoch} "
                f"effective_samples={steps_per_epoch * batch_size}",
                flush=True,
            )
    else:
        steps_per_epoch = max(1, (train_samples + batch_size - 1) // batch_size)
    warmup_steps = min(warmup_epochs, epochs) * steps_per_epoch
    total_steps = epochs * steps_per_epoch
    peak_learning_rate = resolve_peak_learning_rate(
        learning_rate,
        model_dim=int(model.config.model_dim),
        lr_reference_dim=lr_reference_dim,
    )
    optimizer = make_optimizer(
        model,
        learning_rate=peak_learning_rate,
        weight_decay=weight_decay,
        grad_clip_norm=grad_clip_norm,
        warmup_steps=warmup_steps,
        total_steps=total_steps,
        min_learning_rate_ratio=min_learning_rate_ratio,
        cosine_decay=cosine_decay,
    )
    if resumed_opt_state is not None:
        restore_optimizer_state(optimizer, opt_state=resumed_opt_state, step=resumed_opt_step)
    elif resume_dir is not None:
        print(
            "warning: optimizer state missing in checkpoint; reinitialized Adam state.",
            flush=True,
        )
    if lr_reference_dim is not None:
        print(
            f"learning_rate={learning_rate:g} scaled by sqrt({lr_reference_dim}/{model.config.model_dim}) "
            f"-> peak={peak_learning_rate:g}"
        )
    else:
        print(f"learning_rate peak={peak_learning_rate:g}")
    print(
        f"tactile_window={tactile_window} "
        f"(action_horizon={action_horizon} / divisor={tactile_window_divisor}) "
        f"gru_hidden_dim={DEFAULT_GRU_HIDDEN_DIM} resnet_dim={conditioner.resnet_embedding_dim} "
        f"state_dim={model.config.state_dim} state_conditioning={model.config.state_conditioning} "
        f"state_dropout={state_dropout_rate:g} "
        f"(trainable ResNet={model.config.tactile_encoder_trainable}, "
        "shared GRU=True, state MLP=True)"
    )
    if use_cached_embeddings:
        image_loader = (
            f"raw_images workers={num_workers} prefetch_batches={prefetch_batches} "
            f"load_threads={load_threads} pipeline_prefetch={pipeline_prefetch} "
            f"image_cache_size={image_cache_size}"
            if train_tactile_encoder
            else "precomputed_tactile_embeddings"
        )
        print(
            f"dataloader={image_loader} sources={len(dataset_sources or ())} "
            f"cache_root={tactile_embedding_cache_root} "
            f"dataset_balanced_sampling={dataset_balanced_sampling} "
            f"dataset_balanced_loss={dataset_balanced_loss} "
            f"eval_every={eval_every} start_epoch={start_epoch} epochs={epochs}"
        )
    else:
        print(
            f"dataloader=num_workers={num_workers} prefetch_batches={prefetch_batches} "
            f"load_threads={load_threads} pipeline_prefetch={pipeline_prefetch} "
            f"image_cache_size={image_cache_size} encode_batch_size={encode_batch_size} "
            f"eval_every={eval_every} start_epoch={start_epoch} epochs={epochs}"
        )
    if aux_decode_weight < 0:
        raise ValueError(f"aux_decode_weight must be >= 0, got {aux_decode_weight}.")
    if aux_decode_steps <= 0:
        raise ValueError(f"aux_decode_steps must be positive, got {aux_decode_steps}.")
    aux_decode_solver = resolve_decode_solver(aux_decode_solver)
    if loss_mode == "gt":
        print(
            "loss_mode=gt L=FM(gt)+aux*MSE(decode,gt) "
            f"(aux={aux_decode_weight:g}, decode_steps={aux_decode_steps}, "
            f"decode_solver={aux_decode_solver}; "
            "primary eval vs gt; also log vs predicted)"
        )
    elif loss_mode == "predicted":
        print("loss_mode=predicted (train/eval primary target=predicted_actions; also log vs gt; " "no aux decode MSE)")
    elif loss_mode == "gated":
        print(
            "loss_mode=gated w_eff=clip((w-low)/(high-low),0,1) "
            "L=w_eff*FM(gt)+lambda*(1-w_eff)*FM(pred) "
            "+low_safety_weight*low_gate_nearest_endpoint_hinge "
            "+aux*active_high_group_MSE(decode,gt) "
            "+rank_weight*active_high_group_preference_loss "
            "+repair_weight*active_high_group_normalized_repair_loss "
            f"tau={gate_tau:g} T={gate_temperature:g} lambda={gate_lambda:g} "
            f"aux={aux_decode_weight:g} decode_steps={aux_decode_steps} "
            f"decode_solver={aux_decode_solver} "
            f"low_safety_weight={low_gate_safety_weight:g} "
            f"low_safety_margin={low_gate_safety_margin:g} "
            f"rank_weight={rank_weight:g} rank_margin={rank_margin:g} "
            f"repair_weight={repair_weight:g} repair_margin={repair_margin:g} "
            f"rank_regions=low<={rank_low_gate_threshold:g},"
            f"mid,high>={rank_high_gate_threshold:g} "
            f"(primary eval=gt; also log vs predicted)"
        )
    else:
        print(
            "loss_mode=bimanual_gated L=FM(per-wrist composite endpoint)+per-wrist auxiliaries "
            f"tau={gate_tau:g} T={gate_temperature:g} "
            f"aux={aux_decode_weight:g} decode_steps={aux_decode_steps} "
            f"decode_solver={aux_decode_solver} "
            f"rank_regions=low<={rank_low_gate_threshold:g},"
            f"mid,high>={rank_high_gate_threshold:g}"
        )
    if cosine_decay:
        print(
            f"lr_schedule=warmup({warmup_steps} steps)+cosine "
            f"min_ratio={min_learning_rate_ratio:g} total_steps={total_steps}"
        )
    elif warmup_steps > 0:
        print(f"lr_schedule=warmup({warmup_steps} steps)+constant total_steps={total_steps}")

    output_dir.mkdir(parents=True, exist_ok=True)
    history_path = output_dir / "history.csv"
    plot_path = output_dir / "training_curves.png"
    best_key = checkpoint_selection_sentinel(loss_mode=loss_mode)
    best_feasible_key = checkpoint_selection_sentinel(loss_mode=loss_mode)
    specialist_best_keys = {
        "best_rank": (float("inf"),) * 6,
        "best_low_preservation": (float("inf"),) * 7,
        "best_gain": (float("inf"),) * 6,
    }
    resume_extra_for_early_stop = (resume_metadata or {}).get("extra_metadata") or {}
    early_stop_eval_count = int(resume_extra_for_early_stop.get("early_stop_eval_count", 0))
    early_stop_no_improve = int(
        resume_extra_for_early_stop.get("early_stop_no_improve_evals", 0)
    )
    if resume_dir is not None:
        for checkpoint_name in ("best", "best_feasible", *specialist_best_keys):
            checkpoint_path = output_dir / checkpoint_name / CHECKPOINT_NAME
            if not checkpoint_path.exists():
                continue
            with checkpoint_path.open(encoding="utf-8") as file:
                checkpoint_meta = json.load(file)
            _validate_resume_cache(checkpoint_meta, pairs.manifest)
            checkpoint_metrics = checkpoint_meta.get("metrics", {})
            if checkpoint_name in specialist_best_keys:
                specialist_best_keys[checkpoint_name] = checkpoint_specialist_keys(
                    checkpoint_metrics,
                    high_gate_rank_margin=rank_margin,
                )[checkpoint_name]
            else:
                restored_key = checkpoint_selection_key(
                    checkpoint_metrics,
                    loss_mode=loss_mode,
                    max_low_gate_unsafe_frac=best_max_low_gate_unsafe_frac,
                    min_high_gate_gain=best_min_high_gate_gain,
                    min_high_gate_rank_satisfied=best_min_high_gate_rank_satisfied,
                    high_gate_rank_margin=rank_margin,
                )
                if checkpoint_name == "best":
                    best_key = restored_key
                else:
                    best_feasible_key = restored_key
    base_key = jax.random.key(seed)
    history_exists = history_path.exists() and history_path.stat().st_size > 0
    history_mode = "a" if resume_dir is not None and history_exists else "w"
    history_writer_fields = _history_fieldnames_for_write(
        history_path,
        default_fields=history_fields,
        append=history_mode == "a",
    )

    def _refresh_training_plot(*, announce: bool = False) -> None:
        if not write_plots:
            return
        try:
            written = plot_training_history(history_path, output_path=plot_path)
        except (FileNotFoundError, ValueError) as exc:
            print(f"warning: could not refresh training plot: {exc}", flush=True)
            return
        if announce:
            print(f"plot={written}", flush=True)

    try:
        with history_path.open(history_mode, newline="", encoding="utf-8") as history_file:
            writer = csv.DictWriter(
                history_file,
                fieldnames=history_writer_fields,
                extrasaction="ignore",
            )
            if history_mode == "w":
                writer.writeheader()

            for epoch in range(start_epoch, epochs + 1):
                losses: list[float] = []
                component_losses: dict[str, list[float]] = {
                    name: []
                    for name in (
                        "gt_fm",
                        "vla_fm",
                        "composite_fm",
                        "low_safety",
                        "decode",
                        "rank",
                        "repair",
                    )
                }
                gate_w_left_values: list[float] = []
                gate_w_right_values: list[float] = []
                weights: list[int] = []
                if dataset_balanced_sampling:
                    training_batches = conditioner.batches(
                        "train",
                        batch_size=batch_size,
                        shuffle=True,
                        seed=seed + epoch,
                        source_balanced=True,
                    )
                else:
                    training_batches = conditioner.batches(
                        "train",
                        batch_size=batch_size,
                        shuffle=True,
                        seed=seed + epoch,
                    )
                for batch_number, (
                    indices,
                    x_base_np,
                    predicted_np,
                    gt_action_np,
                    state_np,
                    tactile_input,
                ) in enumerate(training_batches):
                    step_key = jax.random.fold_in(base_key, epoch * 1_000_000 + batch_number)
                    batch_n = len(x_base_np)
                    if loss_mode in ("gated", "bimanual_gated"):
                        gate_token_fn = getattr(
                            conditioner,
                            "gate_current_tokens",
                            None,
                        )
                        current_tokens = (
                            gate_token_fn(indices, tactile_input)
                            if gate_token_fn is not None
                            else np.asarray(
                                tactile_input[:, -1, :, :],
                                dtype=np.float32,
                            )
                        )
                    if loss_mode == "gated":
                        change = conditioner.tactile_change_for_cache_indices(
                            indices,
                            current_tokens,
                        )
                        gate_w = gate_weights_from_change(change, tau=gate_tau, temperature=gate_temperature)
                        batch_gate_w = float(np.mean(gate_w))
                        batch_gate_w_left = 0.0
                        batch_gate_w_right = 0.0
                    elif loss_mode == "bimanual_gated":
                        change_per_wrist = conditioner.tactile_change_per_wrist_for_cache_indices(
                            indices,
                            current_tokens,
                        )
                        gate_w = gate_weights_from_change(
                            change_per_wrist,
                            tau=gate_tau,
                            temperature=gate_temperature,
                        )
                        batch_gate_w_left = float(np.mean(gate_w[:, 0]))
                        batch_gate_w_right = float(np.mean(gate_w[:, 1]))
                        batch_gate_w = 0.5 * (batch_gate_w_left + batch_gate_w_right)
                    else:
                        gate_w = np.ones((batch_n,), dtype=np.float32)
                        batch_gate_w = 1.0
                        batch_gate_w_left = 0.0
                        batch_gate_w_right = 0.0
                    if isinstance(pairs, MultiCachedPairs):
                        batch_source_indices, _ = pairs.source_and_local_indices(indices)
                    else:
                        batch_source_indices = np.zeros((batch_n,), dtype=np.int32)
                    loss, loss_components = train_step(
                        model,
                        optimizer,
                        jnp.asarray(x_base_np),
                        jnp.asarray(gt_action_np),
                        jnp.asarray(predicted_np),
                        jnp.asarray(tactile_input),
                        jnp.asarray(gate_w),
                        step_key,
                        jnp.asarray(batch_source_indices),
                        jnp.asarray(source_rank_weight_values),
                        state=jnp.asarray(state_np),
                        state_dropout_rate=state_dropout_rate,
                        loss_mode=loss_mode,
                        gate_lambda=gate_lambda,
                        aux_decode_weight=aux_decode_weight,
                        aux_decode_steps=aux_decode_steps,
                        aux_decode_solver=aux_decode_solver,
                        low_gate_safety_weight=low_gate_safety_weight,
                        low_gate_safety_margin=low_gate_safety_margin,
                        rank_weight=rank_weight,
                        rank_margin=rank_margin,
                        repair_weight=repair_weight,
                        repair_margin=repair_margin,
                        rank_low_gate_threshold=rank_low_gate_threshold,
                        rank_high_gate_threshold=rank_high_gate_threshold,
                        source_balanced_loss=dataset_balanced_loss,
                        num_sources=source_count,
                        high_gate_rank_aggregation=high_gate_rank_aggregation,
                        high_gate_rank_hard_fraction=high_gate_rank_hard_fraction,
                        high_gate_rank_worst_beta=high_gate_rank_worst_beta,
                    )
                    losses.append(float(jax.device_get(loss)))
                    for name in component_losses:
                        component_losses[name].append(float(jax.device_get(loss_components[name])))
                    weights.append(batch_n)
                    gate_w_left_values.append(batch_gate_w_left)
                    gate_w_right_values.append(batch_gate_w_right)
                    if batch_number == 0 or (batch_number + 1) % 20 == 0:
                        if loss_mode == "gated":
                            extra = f" gate_w={batch_gate_w:.4f}"
                        elif loss_mode == "bimanual_gated":
                            extra = (
                                f" gate_w_left={batch_gate_w_left:.4f}"
                                f" gate_w_right={batch_gate_w_right:.4f}"
                            )
                        else:
                            extra = ""
                        print(
                            f"epoch={epoch}/{epochs} batch={batch_number + 1}/{steps_per_epoch} "
                            f"loss_total={losses[-1]:.6f} "
                            f"gt_fm={component_losses['gt_fm'][-1]:.6f} "
                            f"vla_fm={component_losses['vla_fm'][-1]:.6f} "
                            f"composite_fm={component_losses['composite_fm'][-1]:.6f} "
                            f"low_safety={component_losses['low_safety'][-1]:.6f} "
                            f"decode={component_losses['decode'][-1]:.6f} "
                            f"rank={component_losses['rank'][-1]:.6f} "
                            f"repair={component_losses['repair'][-1]:.6f}{extra}",
                            flush=True,
                        )
                train_loss = float(np.average(losses, weights=weights))
                train_components = {
                    name: float(np.average(values, weights=weights)) for name, values in component_losses.items()
                }
                train_metrics: dict[str, float] = {
                    "train_loss_total": train_loss,
                    "train_loss_gt_fm": train_components["gt_fm"],
                    "train_loss_vla_fm": train_components["vla_fm"],
                    "train_loss_composite_fm": train_components["composite_fm"],
                    "train_loss_low_safety": train_components["low_safety"],
                    "train_loss_decode": train_components["decode"],
                    "train_loss_rank": train_components["rank"],
                    "train_loss_repair": train_components["repair"],
                    "train_flow_loss": train_loss,
                }
                if loss_mode == "bimanual_gated":
                    train_metrics.update(
                        {
                            "train_gate_w_left": float(
                                np.average(gate_w_left_values, weights=weights)
                            ),
                            "train_gate_w_right": float(
                                np.average(gate_w_right_values, weights=weights)
                            ),
                        }
                    )
                run_eval = (epoch % eval_every == 0) or (epoch == epochs)
                checkpoint_extra = {
                    "cache_records_sha256": pairs.manifest["records_sha256"],
                    "cache_configuration": pairs.manifest["configuration"],
                    "tactile_encoder_dir": str(tactile_encoder_dir.resolve()),
                    "tactile_window_divisor": tactile_window_divisor,
                    "tactile_window": tactile_window,
                    "gru_hidden_dim": DEFAULT_GRU_HIDDEN_DIM,
                    "history_stride": history_stride,
                    "loss_mode": loss_mode,
                    "eval_target": eval_target,
                    "gate_tau": gate_tau,
                    "gate_temperature": gate_temperature,
                    "gate_lambda": gate_lambda,
                    "decoder_input_version": 2,
                    "state_conditioning": bool(model.config.state_conditioning),
                    "state_dim": int(model.config.state_dim),
                    "state_dropout_rate": state_dropout_rate,
                    "tactile_encoder_trainable": bool(
                        model.config.tactile_encoder_trainable
                    ),
                    "state_encoder_trainable": bool(
                        model.config.state_conditioning
                    ),
                    "aux_decode_weight": aux_decode_weight,
                    "aux_decode_steps": aux_decode_steps,
                    "aux_decode_solver": aux_decode_solver,
                    "low_gate_safety_weight": low_gate_safety_weight,
                    "low_gate_safety_margin": low_gate_safety_margin,
                    "rank_weight": rank_weight,
                    "rank_margin": rank_margin,
                    "repair_weight": repair_weight,
                    "repair_margin": repair_margin,
                    "rank_low_gate_threshold": rank_low_gate_threshold,
                    "rank_high_gate_threshold": rank_high_gate_threshold,
                    "gate_eval_bins": [
                        {"id": bin_id, "lower": lower, "upper": upper} for bin_id, lower, upper in GATE_BIN_SPECS
                    ],
                    "loss_weighting_version": 7,
                    "gate_weight_transform": "saturated_linear_v1",
                    "constraint_reduction": "active_group_weighted_mean_v1",
                    "validation_steps": validation_steps,
                    "validation_solver": aux_decode_solver,
                    "best_max_low_gate_unsafe_frac": best_max_low_gate_unsafe_frac,
                    "best_min_high_gate_gain": best_min_high_gate_gain,
                    "best_min_high_gate_rank_satisfied": best_min_high_gate_rank_satisfied,
                    "checkpoint_selection_version": 3,
                    "dataset_balanced_sampling": dataset_balanced_sampling,
                    "dataset_balanced_loss": dataset_balanced_loss,
                    "high_gate_rank_aggregation": high_gate_rank_aggregation,
                    "high_gate_rank_hard_fraction": high_gate_rank_hard_fraction,
                    "high_gate_rank_worst_beta": high_gate_rank_worst_beta,
                    "high_gate_rank_source_weights": dict(high_gate_rank_source_weights or {}),
                    "high_gate_rank_source_weight_values": source_rank_weight_values.tolist(),
                    "source_count": source_count,
                    "source_names": list(pairs.source_names) if isinstance(pairs, MultiCachedPairs) else [],
                    "early_stop_patience": early_stop_patience,
                    "early_stop_min_evals": early_stop_min_evals,
                    "early_stop_eval_count": early_stop_eval_count,
                    "early_stop_no_improve_evals": early_stop_no_improve,
                    "eval_every": eval_every,
                }
                if loss_mode == "bimanual_gated":
                    checkpoint_extra.pop("gate_lambda")
                    checkpoint_extra.update(
                        {
                            "loss_objective_version": BIMANUAL_OBJECTIVE_VERSION,
                            "action_slices": {
                                "left": [LEFT_ACTION_SLICE.start, LEFT_ACTION_SLICE.stop],
                                "right": [RIGHT_ACTION_SLICE.start, RIGHT_ACTION_SLICE.stop],
                            },
                            "wrist_token_indices": {
                                "left": list(LEFT_WRIST_TOKEN_INDICES),
                                "right": list(RIGHT_WRIST_TOKEN_INDICES),
                            },
                        }
                    )
                if init_metadata is not None and init_from is not None:
                    checkpoint_extra["initialized_from"] = str(init_from.resolve())
                    checkpoint_extra["initialized_from_epoch"] = int(init_metadata["epoch"])
                if run_eval:
                    validation = evaluate_split(
                        model,
                        conditioner,
                        split="val",
                        batch_size=batch_size,
                        num_steps=validation_steps,
                        keep_predictions=False,
                        solver=aux_decode_solver,
                        target=eval_target,
                        loss_mode=loss_mode,
                        gate_tau=(
                            gate_tau if loss_mode in ("gated", BIMANUAL_LOSS_MODE) else None
                        ),
                        gate_temperature=(
                            gate_temperature
                            if loss_mode in ("gated", BIMANUAL_LOSS_MODE)
                            else None
                        ),
                        rank_margin=(
                            rank_margin if loss_mode in ("gated", BIMANUAL_LOSS_MODE) else 0.0
                        ),
                        repair_margin=(
                            repair_margin if loss_mode in ("gated", BIMANUAL_LOSS_MODE) else 0.0
                        ),
                        low_safety_margin=(
                            low_gate_safety_margin
                            if loss_mode in ("gated", BIMANUAL_LOSS_MODE)
                            else 0.0
                        ),
                        rank_low_gate_threshold=rank_low_gate_threshold,
                        rank_high_gate_threshold=rank_high_gate_threshold,
                    )
                    metrics: dict[str, float | str | int] = {
                        **train_metrics,
                        "val_flow_loss": validation.flow_loss,
                        "val_mse": validation.mse,
                        "val_rmse": validation.rmse,
                        "val_mae": validation.mae,
                        "val_flow_loss_gt": validation.flow_loss_gt,
                        "val_mse_gt": validation.mse_gt,
                        "val_rmse_gt": validation.rmse_gt,
                        "val_mae_gt": validation.mae_gt,
                        "val_flow_loss_pred": validation.flow_loss_pred,
                        "val_mse_pred": validation.mse_pred,
                        "val_rmse_pred": validation.rmse_pred,
                        "val_mae_pred": validation.mae_pred,
                        "val_mse_vla_gt": validation.mse_vla_gt,
                        "val_gt_gain": validation.gt_gain,
                        "val_relative_gt_error": validation.relative_gt_error,
                        "eval_target": validation.target,
                    }
                    if validation.n_high_w is not None:
                        metrics.update(
                            {
                                "val_mse_gt_high_w": float(validation.mse_gt_high_w),
                                "val_mse_gt_low_w": float(validation.mse_gt_low_w),
                                "val_mse_pred_high_w": float(validation.mse_pred_high_w),
                                "val_mse_pred_low_w": float(validation.mse_pred_low_w),
                                "val_mse_vla_gt_high_w": float(validation.mse_vla_gt_high_w),
                                "val_mse_vla_gt_low_w": float(validation.mse_vla_gt_low_w),
                                "val_gt_gain_high_w": float(validation.gt_gain_high_w),
                                "val_gt_gain_low_w": float(validation.gt_gain_low_w),
                                "val_relative_gt_error_high_w": float(validation.relative_gt_error_high_w),
                                "val_relative_gt_error_low_w": float(validation.relative_gt_error_low_w),
                                "val_rank_penalty_high_w": float(validation.rank_penalty_high_w),
                                "val_rank_penalty_low_w": float(validation.rank_penalty_low_w),
                                "val_rank_satisfied_high_frac": float(validation.rank_satisfied_high_frac),
                                "val_rank_satisfied_low_frac": float(validation.rank_satisfied_low_frac),
                                "val_repair_penalty_high_w": float(validation.repair_penalty_high_w),
                                "val_repair_satisfied_high_frac": float(validation.repair_satisfied_high_frac),
                                "val_low_nearest_endpoint_mse": float(validation.low_nearest_endpoint_mse),
                                "val_low_safety_penalty": float(validation.low_safety_penalty),
                                "val_low_safe_frac": float(validation.low_safe_frac),
                                "val_low_unsafe_frac": float(validation.low_unsafe_frac),
                                "val_gate_w": float(validation.gate_w),
                                "val_gate_active_frac": float(validation.gate_active_frac),
                                "val_gate_w_high_mean": float(validation.gate_w_high_mean),
                                "val_gate_w_low_mean": float(validation.gate_w_low_mean),
                                "val_gate_w_p10": float(validation.gate_w_p10),
                                "val_gate_w_p25": float(validation.gate_w_p25),
                                "val_gate_w_p50": float(validation.gate_w_p50),
                                "val_gate_w_p75": float(validation.gate_w_p75),
                                "val_gate_w_p90": float(validation.gate_w_p90),
                                "val_tactile_change": float(validation.tactile_change),
                                "val_tactile_change_high_mean": float(validation.tactile_change_high_mean),
                                "val_tactile_change_low_mean": float(validation.tactile_change_low_mean),
                                "val_tactile_change_p10": float(validation.tactile_change_p10),
                                "val_tactile_change_p25": float(validation.tactile_change_p25),
                                "val_tactile_change_p50": float(validation.tactile_change_p50),
                                "val_tactile_change_p75": float(validation.tactile_change_p75),
                                "val_tactile_change_p90": float(validation.tactile_change_p90),
                                "val_n_high_w": int(validation.n_high_w),
                                "val_n_low_w": int(validation.n_low_w),
                                "val_n_mid_w": int(validation.n_mid_w),
                            }
                        )
                    if getattr(validation, "n_high_w_left", None) is not None:
                        for wrist in ("left", "right"):
                            for metric_name in bimanual_validation_metric_names:
                                value = getattr(validation, f"{metric_name}_{wrist}")
                                metrics[f"val_{metric_name}_{wrist}"] = (
                                    int(value) if metric_name.startswith("n_") else float(value)
                                )
                    if validation.gate_bin_metrics is not None:
                        for bin_id, bin_metrics in validation.gate_bin_metrics.items():
                            for metric_name in gate_bin_metric_names:
                                metrics[f"val_gate_bin_{bin_id}_{metric_name}"] = bin_metrics[metric_name]
                    if isinstance(pairs, MultiCachedPairs) and validation.sample_gate_w is not None:
                        source_indices, _ = pairs.source_and_local_indices(validation.cache_indices)
                        low_safety_penalties: list[float] = []
                        low_unsafe_fractions: list[float] = []
                        high_gains: list[float] = []
                        high_rank_violations: list[float] = []
                        high_rank_gaps: list[float] = []
                        high_rank_satisfied: list[float] = []
                        missing_confident_low = False
                        missing_confident_high = False
                        for source_index in range(len(pairs.sources)):
                            source_mask = source_indices == source_index
                            source_gate = validation.sample_gate_w[source_mask]
                            source_low = source_gate <= rank_low_gate_threshold
                            source_high = source_gate >= rank_high_gate_threshold
                            if np.any(source_low):
                                source_low_gt = validation.sample_mse_gt[source_mask][source_low]
                                source_low_pred = validation.sample_mse_pred[source_mask][source_low]
                                source_nearest = np.minimum(source_low_gt, source_low_pred)
                                source_penalty = np.maximum(
                                    source_nearest - low_gate_safety_margin,
                                    0.0,
                                )
                                nearest_mean = float(np.mean(source_nearest))
                                penalty_mean = float(np.mean(source_penalty))
                                unsafe_frac = float(
                                    np.mean(source_nearest > low_gate_safety_margin)
                                )
                                low_safety_penalties.append(penalty_mean)
                                low_unsafe_fractions.append(unsafe_frac)
                                metrics[
                                    f"val_source_{source_index}_low_nearest_endpoint_mse"
                                ] = nearest_mean
                                metrics[
                                    f"val_source_{source_index}_low_safety_penalty"
                                ] = penalty_mean
                                metrics[
                                    f"val_source_{source_index}_low_unsafe_frac"
                                ] = unsafe_frac
                            else:
                                missing_confident_low = True
                                for metric_name in source_metric_names[:3]:
                                    metrics[f"val_source_{source_index}_{metric_name}"] = float("nan")
                            if np.any(source_high):
                                source_gt_gain = validation.sample_gt_gain[source_mask][source_high]
                                source_mse_gt = validation.sample_mse_gt[source_mask][source_high]
                                source_mse_pred = validation.sample_mse_pred[source_mask][source_high]
                                rank_hinge, rank_gap, rank_satisfied = high_gate_rank_statistics(
                                    source_mse_gt,
                                    source_mse_pred,
                                    margin=rank_margin,
                                )
                                high_gains.append(float(np.mean(source_gt_gain)))
                                high_rank_violations.append(rank_hinge)
                                high_rank_gaps.append(rank_gap)
                                high_rank_satisfied.append(rank_satisfied)
                                metrics[f"val_source_{source_index}_gt_gain_high_w"] = float(
                                    np.mean(source_gt_gain)
                                )
                                metrics[f"val_source_{source_index}_rank_gap_high_w"] = rank_gap
                                metrics[f"val_source_{source_index}_rank_hinge_high_w"] = rank_hinge
                                metrics[
                                    f"val_source_{source_index}_rank_satisfied_high_frac"
                                ] = rank_satisfied
                            else:
                                missing_confident_high = True
                                for metric_name in source_metric_names[3:]:
                                    metrics[f"val_source_{source_index}_{metric_name}"] = float("nan")
                        if low_safety_penalties and not missing_confident_low:
                            metrics["val_worst_dataset_low_safety_penalty"] = max(
                                low_safety_penalties
                            )
                            metrics["val_worst_dataset_low_unsafe_frac"] = max(
                                low_unsafe_fractions
                            )
                        elif missing_confident_low:
                            metrics["val_worst_dataset_low_safety_penalty"] = float("nan")
                            metrics["val_worst_dataset_low_unsafe_frac"] = float("nan")
                        if high_gains and not missing_confident_high:
                            metrics["val_min_dataset_gt_gain_high_w"] = min(high_gains)
                        elif missing_confident_high:
                            metrics["val_min_dataset_gt_gain_high_w"] = float("nan")
                        if high_rank_violations and not missing_confident_high:
                            metrics["val_worst_dataset_rank_violation_high_w"] = max(high_rank_violations)
                            metrics["val_worst_dataset_rank_gap_high_w"] = max(high_rank_gaps)
                            metrics["val_min_dataset_rank_satisfied_high_frac"] = min(high_rank_satisfied)
                        elif missing_confident_high:
                            metrics["val_worst_dataset_rank_violation_high_w"] = float("nan")
                            metrics["val_worst_dataset_rank_gap_high_w"] = float("nan")
                            metrics["val_min_dataset_rank_satisfied_high_frac"] = float("nan")
                    elif (
                        isinstance(pairs, MultiCachedPairs)
                        and getattr(validation, "sample_gate_w_left", None) is not None
                    ):
                        source_indices, _ = pairs.source_and_local_indices(
                            validation.cache_indices
                        )
                        per_source_wrist, wrist_rollups = bimanual_source_decode_metrics(
                            sample_mse_gt_left=validation.sample_mse_gt_left,
                            sample_mse_gt_right=validation.sample_mse_gt_right,
                            sample_mse_vla_left=validation.sample_mse_vla_left,
                            sample_mse_vla_right=validation.sample_mse_vla_right,
                            sample_mse_vla_gt_left=validation.sample_mse_vla_gt_left,
                            sample_mse_vla_gt_right=validation.sample_mse_vla_gt_right,
                            sample_gate_w_left=validation.sample_gate_w_left,
                            sample_gate_w_right=validation.sample_gate_w_right,
                            source_indices=source_indices,
                            num_sources=len(pairs.sources),
                            low_w_threshold=rank_low_gate_threshold,
                            high_w_threshold=rank_high_gate_threshold,
                            ranking_margin=rank_margin,
                            repair_margin=repair_margin,
                            low_safety_margin=low_gate_safety_margin,
                        )
                        for source_index, source_metrics in per_source_wrist.items():
                            for metric_name, value in source_metrics.items():
                                metrics[f"val_source_{source_index}_{metric_name}"] = value
                        for metric_name, value in wrist_rollups.items():
                            metrics[f"val_{metric_name}"] = value
                    selection_key = checkpoint_selection_key(
                        metrics,
                        loss_mode=loss_mode,
                        max_low_gate_unsafe_frac=best_max_low_gate_unsafe_frac,
                        min_high_gate_gain=best_min_high_gate_gain,
                        min_high_gate_rank_satisfied=best_min_high_gate_rank_satisfied,
                        high_gate_rank_margin=rank_margin,
                    )
                    metrics["checkpoint_selection_key"] = ",".join(f"{value:.12g}" for value in selection_key)
                    metrics["checkpoint_selection_feasible"] = int(selection_key[0] == 0.0)
                    selection_improved = selection_key < best_key
                    (
                        early_stop_eval_count,
                        early_stop_no_improve,
                        should_early_stop,
                    ) = update_early_stop_state(
                        improved=selection_improved,
                        evaluation_count=early_stop_eval_count,
                        no_improve_count=early_stop_no_improve,
                        patience=early_stop_patience,
                        min_evaluations=early_stop_min_evals,
                    )
                    metrics["early_stop_no_improve_evals"] = early_stop_no_improve
                    checkpoint_extra["early_stop_eval_count"] = early_stop_eval_count
                    checkpoint_extra["early_stop_no_improve_evals"] = early_stop_no_improve
                    specialist_keys = checkpoint_specialist_keys(
                        metrics,
                        high_gate_rank_margin=rank_margin,
                    )
                    writer.writerow(_blank_history_row(epoch, **metrics))
                    history_file.flush()
                    _refresh_training_plot()
                    save_checkpoint(
                        output_dir / "last",
                        model,
                        epoch=epoch,
                        metrics=metrics,
                        extra_metadata=checkpoint_extra,
                        optimizer=optimizer,
                    )
                    saved_checkpoints = ["last"]
                    if selection_improved:
                        best_key = selection_key
                        save_checkpoint(
                            output_dir / "best",
                            model,
                            epoch=epoch,
                            metrics=metrics,
                            extra_metadata=checkpoint_extra,
                            optimizer=optimizer,
                        )
                        saved_checkpoints.append("best")
                    if (
                        loss_mode == "gated"
                        and selection_key[0] == 0.0
                        and selection_key < best_feasible_key
                    ):
                        best_feasible_key = selection_key
                        save_checkpoint(
                            output_dir / "best_feasible",
                            model,
                            epoch=epoch,
                            metrics=metrics,
                            extra_metadata=checkpoint_extra,
                        )
                        saved_checkpoints.append("best_feasible")
                    if loss_mode == "gated":
                        for checkpoint_name, specialist_key in specialist_keys.items():
                            if specialist_key >= specialist_best_keys[checkpoint_name]:
                                continue
                            specialist_best_keys[checkpoint_name] = specialist_key
                            save_checkpoint(
                                output_dir / checkpoint_name,
                                model,
                                epoch=epoch,
                                metrics=metrics,
                                extra_metadata=checkpoint_extra,
                            )
                            saved_checkpoints.append(checkpoint_name)
                    stratified_msg = ""
                    if validation.n_high_w is not None:
                        stratified_msg = (
                            f" mse_gt(w>={rank_high_gate_threshold:g})="
                            f"{validation.mse_gt_high_w:.4f}"
                            f" mse_gt(w<={rank_low_gate_threshold:g})="
                            f"{validation.mse_gt_low_w:.4f}"
                            f" mse_pred(w>={rank_high_gate_threshold:g})="
                            f"{validation.mse_pred_high_w:.4f}"
                            f" mse_pred(w<={rank_low_gate_threshold:g})="
                            f"{validation.mse_pred_low_w:.4f}"
                            f" vla_gt(w>={rank_high_gate_threshold:g})="
                            f"{validation.mse_vla_gt_high_w:.4f}"
                            f" gain(w>={rank_high_gate_threshold:g})="
                            f"{validation.gt_gain_high_w:.4f}"
                            f" rel_gt(w>={rank_high_gate_threshold:g})="
                            f"{validation.relative_gt_error_high_w:.4f}"
                            f" rank_ok_hi={validation.rank_satisfied_high_frac:.3f}"
                            f" repair_ok_hi={validation.repair_satisfied_high_frac:.3f}"
                            f" low_nearest={validation.low_nearest_endpoint_mse:.4f}"
                            f" low_safe={validation.low_safe_frac:.3f}"
                            f" w_mean_hi={validation.gate_w_high_mean:.3f}"
                            f" w_mean_lo={validation.gate_w_low_mean:.3f}"
                            f" n_high={validation.n_high_w} n_mid={validation.n_mid_w}"
                            f" n_low={validation.n_low_w}"
                        )
                    selection_msg = (
                        f" best_feasible={int(selection_key[0] == 0.0)}"
                        f" selection_key={metrics['checkpoint_selection_key']}"
                        f" saved={','.join(saved_checkpoints)}"
                    )
                    print(
                        f"epoch={epoch}/{epochs} train_loss_total={train_loss:.8f} "
                        f"val_flow_loss={validation.flow_loss:.8f} "
                        f"val_mse={validation.mse:.8f} (target={validation.target}) "
                        f"val_mse_gt={validation.mse_gt:.8f} val_mse_pred={validation.mse_pred:.8f} "
                        f"vla_mse_gt={validation.mse_vla_gt:.8f} "
                        f"gt_gain={validation.gt_gain:.8f} "
                        f"relative_gt_error={validation.relative_gt_error:.4f}"
                        f"{stratified_msg}{selection_msg}",
                        flush=True,
                    )
                    if should_early_stop:
                        print(
                            f"early stopping at epoch={epoch}: no checkpoint-selection "
                            f"improvement for {early_stop_no_improve} evaluations "
                            f"(patience={early_stop_patience}, eval_every={eval_every})",
                            flush=True,
                        )
                        break
                else:
                    metrics = dict(train_metrics)
                    writer.writerow(_blank_history_row(epoch, **metrics))
                    history_file.flush()
                    _refresh_training_plot()
                    save_checkpoint(
                        output_dir / "last",
                        model,
                        epoch=epoch,
                        metrics=metrics,
                        extra_metadata=checkpoint_extra,
                        optimizer=optimizer,
                    )
                    print(
                        f"epoch={epoch}/{epochs} train_loss_total={train_loss:.8f} (skip val)",
                        flush=True,
                    )

        print(f"best_checkpoint_selection_key={best_key}")
        print(f"best_feasible_checkpoint_selection_key={best_feasible_key}")
        print(f"specialist_checkpoint_keys={specialist_best_keys}")
        print(f"checkpoints={output_dir}")
        _refresh_training_plot(announce=True)
    finally:
        conditioner.close()


def _config_diff(left: object, right: object) -> dict[str, tuple[object, object]]:
    import dataclasses

    diffs: dict[str, tuple[object, object]] = {}
    left_dict = dataclasses.asdict(left)  # type: ignore[arg-type]
    right_dict = dataclasses.asdict(right)  # type: ignore[arg-type]
    for key, left_value in left_dict.items():
        right_value = right_dict.get(key)
        if left_value != right_value:
            diffs[key] = (left_value, right_value)
    return diffs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Train tactile/state cross-attention flow decoder "
            "(frozen tactile ResNet features; loss-mode gt / predicted / gated / bimanual_gated)."
        )
    )
    parser.add_argument("--cache-dir", type=pathlib.Path, required=True)
    parser.add_argument("--tactile-encoder-dir", type=pathlib.Path, required=True)
    parser.add_argument("--output-dir", type=pathlib.Path, required=True)
    parser.add_argument(
        "--dataset-repo-id",
        type=str,
        default=None,
        help="Override LeRobot dataset repo id (default: cache manifest configuration).",
    )
    parser.add_argument(
        "--dataset-root",
        type=pathlib.Path,
        default=None,
        help="Optional local dataset root hint (currently unused by image loader; reserved).",
    )
    parser.add_argument(
        "--tactile-window-divisor",
        type=int,
        default=1,
        help="tactile_window = action_horizon // divisor (must divide evenly). Default 1.",
    )
    parser.add_argument(
        "--history-stride",
        type=int,
        default=1,
        help="Frame stride when looking back for the tactile window (default 1 = contiguous).",
    )
    parser.add_argument(
        "--loss-mode",
        choices=("gt", "predicted", "gated", "bimanual_gated"),
        default="gt",
        help=(
            "gt: FM(gt)+aux*MSE(decode,gt) (primary eval vs GT). "
            "predicted: FM vs VLA predicted_actions only (no aux; sanity check). "
            "gated: w*FM(gt)+lambda*(1-w)*FM(pred), plus a weak low-gate "
            "nearest-endpoint safety hinge and high-gate GT decode/rank/repair terms. "
            "bimanual_gated: one per-wrist composite-endpoint FM plus per-wrist auxiliaries. "
            "All modes always log both val_mse_gt and val_mse_pred."
        ),
    )
    parser.add_argument(
        "--gate-tau",
        type=float,
        default=0.5,
        help="Soft-gate midpoint tau for w=sigmoid((s-tau)/T). Default 0.5.",
    )
    parser.add_argument(
        "--gate-temperature",
        type=float,
        default=0.1,
        help="Soft-gate temperature T. Default 0.1.",
    )
    parser.add_argument(
        "--gate-lambda",
        type=float,
        default=1.0,
        help="Weight on the low/mid-gate (1-w)*FM(VLA) anchor. Default 1.0.",
    )
    parser.add_argument(
        "--aux-decode-weight",
        type=float,
        default=1.0,
        help=(
            "Weight on MSE(decode(x_base), gt): all samples in gt mode, and "
            "high-gate samples only in gated mode. Set 0 to disable. Default 1.0."
        ),
    )
    parser.add_argument(
        "--aux-decode",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable aux decode MSE. --no-aux-decode turns it off even if the weight is nonzero.",
    )
    parser.add_argument(
        "--aux-decode-steps",
        type=int,
        default=None,
        help=("ODE steps used for aux decode MSE during training " "(default: same as --validation-steps)."),
    )
    parser.add_argument(
        "--aux-decode-solver",
        choices=("euler", "fireflow"),
        default="euler",
        help="ODE solver for aux decode MSE and validation decode (default: euler).",
    )
    parser.add_argument(
        "--low-gate-safety-weight",
        type=float,
        default=0.0,
        help="Weight of the low-gate nearest-endpoint safety hinge.",
    )
    parser.add_argument(
        "--low-gate-safety",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Enable the low-gate safety hinge. "
            "--no-low-gate-safety turns it off even if the weight is nonzero."
        ),
    )
    parser.add_argument(
        "--low-gate-safety-margin",
        type=float,
        default=0.03,
        help="Allowed low-gate MSE to the nearer of GT/VLA before the hinge activates.",
    )
    parser.add_argument(
        "--rank",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable the high-gate rank loss. --no-rank turns it off even if the weight is nonzero.",
    )
    parser.add_argument(
        "--rank-weight",
        type=float,
        default=0.0,
        help="Weight of the high-gate GT-over-VLA preference ranking loss.",
    )
    parser.add_argument(
        "--rank-margin",
        type=float,
        default=0.0,
        help="Required MSE separation between the preferred and other endpoint.",
    )
    parser.add_argument(
        "--repair",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable the high-gate repair loss. --no-repair turns it off even if the weight is nonzero.",
    )
    parser.add_argument(
        "--repair-weight",
        type=float,
        default=0.0,
        help="Weight requiring high-gate GT error to beat the frozen VLA baseline.",
    )
    parser.add_argument(
        "--repair-margin",
        type=float,
        default=0.0,
        help="Required high-gate GT MSE improvement over the VLA baseline.",
    )
    parser.add_argument(
        "--rank-low-gate-threshold",
        type=float,
        default=0.3,
        help="Apply nearest-endpoint safety when w is at or below this value.",
    )
    parser.add_argument(
        "--rank-high-gate-threshold",
        type=float,
        default=0.7,
        help="Apply GT-preference rank/repair losses only when w is at or above this value.",
    )
    parser.add_argument(
        "--state-conditioning",
        action="store_true",
        help="Condition FRS on the normalized current observation.state token.",
    )
    parser.add_argument(
        "--state-dropout-rate",
        type=float,
        default=0.0,
        help="Training-only probability of masking the complete state token.",
    )
    parser.add_argument(
        "--best-max-low-gate-unsafe-frac",
        type=float,
        default=0.1,
        help="Maximum low-gate fraction outside the nearest-endpoint safety margin.",
    )
    parser.add_argument(
        "--best-min-high-gate-gain",
        type=float,
        default=0.0,
        help="Minimum high-gate GT gain for a feasible best checkpoint.",
    )
    parser.add_argument(
        "--best-min-high-gate-rank-satisfied",
        type=float,
        default=0.8,
        help="Minimum per-dataset high-gate fraction satisfying the GT-preference rank margin.",
    )
    parser.add_argument(
        "--dataset-balanced-sampling",
        action="store_true",
        help="Oversample sources so every multi-dataset training batch has near-equal source counts.",
    )
    parser.add_argument(
        "--dataset-balanced-loss",
        action="store_true",
        help="Average each loss inside each source first, then average sources equally.",
    )
    parser.add_argument(
        "--early-stop-patience",
        type=int,
        default=0,
        help="Stop after this many evaluations without checkpoint-selection improvement; 0 disables.",
    )
    parser.add_argument(
        "--early-stop-min-evals",
        type=int,
        default=0,
        help="Minimum number of evaluations before early stopping is allowed.",
    )

    parser.add_argument("--model-dim", type=int, default=256)
    parser.add_argument("--depth", type=int, default=6)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--mlp-ratio", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)

    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--warmup-epochs", type=int, default=10)
    parser.add_argument("--lr-reference-dim", type=int, default=256)
    parser.add_argument("--min-lr-ratio", type=float, default=0.1)
    parser.add_argument("--lr-schedule", choices=("cosine", "constant"), default="cosine")

    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--validation-steps", type=int, default=10)
    parser.add_argument(
        "--eval-every",
        type=int,
        default=5,
        help="Run full validation every N epochs (also always on the final epoch). Default 5.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from output-dir/last (params + optimizer state if present).",
    )
    parser.add_argument(
        "--resume-from",
        type=pathlib.Path,
        help="Resume from an explicit checkpoint directory (overrides --resume).",
    )
    parser.add_argument(
        "--init-from",
        type=pathlib.Path,
        help="Load model parameters only and start a fresh optimizer/run at epoch 1.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=8,
        help="Spawn process workers for video/parquet decode (0/1 = in-process threads only).",
    )
    parser.add_argument(
        "--prefetch-batches",
        type=int,
        default=8,
        help="In-flight mp decode batches queued ahead of the trainer.",
    )
    parser.add_argument(
        "--load-threads",
        type=int,
        default=8,
        help="Per-process threads for unique-frame decode within a batch.",
    )
    parser.add_argument(
        "--pipeline-prefetch",
        type=int,
        default=4,
        help="Decoded image batches buffered while parent runs ResNet/train step.",
    )
    parser.add_argument(
        "--image-cache-size",
        type=int,
        default=8192,
        help="Total LRU decoded-frame budget (split across mp workers).",
    )
    parser.add_argument(
        "--encode-batch-size",
        type=int,
        default=256,
        help="Frozen ResNet microbatch size on the parent process/GPU.",
    )
    return parser


def train_from_cli_args(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    train_decoder(
        cache_dir=args.cache_dir,
        tactile_encoder_dir=args.tactile_encoder_dir,
        output_dir=args.output_dir,
        dataset_repo_id=args.dataset_repo_id,
        dataset_root=args.dataset_root,
        tactile_window_divisor=args.tactile_window_divisor,
        history_stride=args.history_stride,
        loss_mode=args.loss_mode,
        gate_tau=args.gate_tau,
        gate_temperature=args.gate_temperature,
        gate_lambda=args.gate_lambda,
        aux_decode_weight=resolve_optional_loss_weight(
            getattr(args, "aux_decode", None), args.aux_decode_weight
        ),
        aux_decode_steps=(args.validation_steps if args.aux_decode_steps is None else args.aux_decode_steps),
        aux_decode_solver=args.aux_decode_solver,
        low_gate_safety_weight=resolve_optional_loss_weight(
            getattr(args, "low_gate_safety", None), args.low_gate_safety_weight
        ),
        low_gate_safety_margin=args.low_gate_safety_margin,
        rank_weight=resolve_optional_loss_weight(
            getattr(args, "rank", None), args.rank_weight
        ),
        rank_margin=args.rank_margin,
        repair_weight=resolve_optional_loss_weight(
            getattr(args, "repair", None), args.repair_weight
        ),
        repair_margin=args.repair_margin,
        rank_low_gate_threshold=args.rank_low_gate_threshold,
        rank_high_gate_threshold=args.rank_high_gate_threshold,
        state_conditioning=args.state_conditioning,
        state_dropout_rate=args.state_dropout_rate,
        model_dim=args.model_dim,
        depth=args.depth,
        num_heads=args.num_heads,
        mlp_ratio=args.mlp_ratio,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        grad_clip_norm=args.grad_clip_norm if args.grad_clip_norm > 0 else None,
        warmup_epochs=args.warmup_epochs,
        lr_reference_dim=args.lr_reference_dim if args.lr_reference_dim > 0 else None,
        min_learning_rate_ratio=args.min_lr_ratio,
        cosine_decay=args.lr_schedule == "cosine",
        batch_size=args.batch_size,
        epochs=args.epochs,
        validation_steps=args.validation_steps,
        eval_every=args.eval_every,
        seed=args.seed,
        write_plots=not args.no_plots,
        num_workers=args.num_workers,
        prefetch_batches=args.prefetch_batches,
        load_threads=args.load_threads,
        pipeline_prefetch=args.pipeline_prefetch,
        image_cache_size=args.image_cache_size,
        encode_batch_size=args.encode_batch_size,
        resume=args.resume,
        resume_from=args.resume_from,
        init_from=args.init_from,
        best_max_low_gate_unsafe_frac=args.best_max_low_gate_unsafe_frac,
        best_min_high_gate_gain=args.best_min_high_gate_gain,
        best_min_high_gate_rank_satisfied=args.best_min_high_gate_rank_satisfied,
        dataset_balanced_sampling=args.dataset_balanced_sampling,
        dataset_balanced_loss=args.dataset_balanced_loss,
        early_stop_patience=args.early_stop_patience,
        early_stop_min_evals=args.early_stop_min_evals,
    )


DEFAULT_CONFIG = Path(__file__).resolve().parent / "configs" / "train_frs.yaml"
RUN_CONFIG_NAME = "train_config.yaml"


def source_cache_dir(cache_root: str | Path, repo_id: str) -> Path:
    parts = [part for part in str(repo_id).split("/") if part not in ("", ".", "..")]
    if not parts:
        raise ValueError(f"invalid repo id: {repo_id!r}")
    return Path(cache_root).expanduser().joinpath(*parts)


def load_config(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as file:
        value = yaml.safe_load(file) or {}
    if not isinstance(value, dict):
        raise ValueError(f"config root must be a mapping: {path}")
    return value


def _positive_int(config: Mapping[str, Any], key: str, default: int) -> int:
    value = int(config.get(key, default))
    if value <= 0:
        raise ValueError(f"{key} must be positive, got {value}")
    return value


def resolve_resume_mode(value: Any, *, output_dir: Path) -> bool:
    """Resolve false/true/auto without treating non-empty strings as true."""

    if isinstance(value, bool):
        return value
    mode = str(value if value is not None else "false").strip().lower()
    if mode == "auto":
        return (output_dir / "last" / "checkpoint.json").is_file()
    if mode in {"true", "1", "yes", "on"}:
        return True
    if mode in {"false", "0", "no", "off", ""}:
        return False
    raise ValueError("frs_training.resume must be false, true, or auto")


def save_run_config(config: Mapping[str, Any], *, output_dir: Path) -> Path:
    """Persist the effective YAML config once and reject mixed-config runs."""

    output_dir.mkdir(parents=True, exist_ok=True)
    destination = output_dir / RUN_CONFIG_NAME
    config_dict = dict(config)
    if destination.exists():
        existing = load_config(destination)
        if existing != config_dict:
            raise ValueError(
                f"FRS output directory already contains a different {RUN_CONFIG_NAME}: "
                f"{destination}. Use a new frs_training.output directory for the new "
                "parameter set, or restore the original config before resuming."
            )
        print(f"run_config={destination} (existing, matched)", flush=True)
        return destination

    serialized = yaml.safe_dump(
        config_dict,
        allow_unicode=True,
        sort_keys=False,
    )
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(serialized, encoding="utf-8")
    temporary.replace(destination)
    print(f"run_config={destination}", flush=True)
    return destination


def train_from_config(config: Mapping[str, Any]) -> None:
    loss_settings = parse_loss_settings(config)
    datasets = config.get("datasets") or []
    if not isinstance(datasets, list) or not datasets:
        raise ValueError("config.datasets must be a non-empty list")
    action_cache = config.get("action_cache") or {}
    tactile_cache = config.get("tactile_embedding_cache") or {}
    model = config.get("model") or {}
    training = config.get("frs_training") or {}
    for name, value in (
        ("action_cache", action_cache),
        ("tactile_embedding_cache", tactile_cache),
        ("model", model),
        ("frs_training", training),
    ):
        if not isinstance(value, Mapping):
            raise ValueError(f"config.{name} must be a mapping")
    if not action_cache.get("root") or not tactile_cache.get("root"):
        raise ValueError("action_cache.root and tactile_embedding_cache.root are required")
    from train_smolvla_frs.prepare_frs_caches import prepare_tactile_embeddings_from_config

    prepare_tactile_embeddings_from_config(config)
    encoder_dir = Path(str(model["tactile_encoder_path"])).expanduser()
    if not encoder_dir.is_dir():
        raise FileNotFoundError(f"tactile encoder does not exist: {encoder_dir}")
    cache_dirs = [source_cache_dir(action_cache["root"], str(source["repo_id"])) for source in datasets]
    missing = [path for path in cache_dirs if not (path / "manifest.json").is_file()]
    if missing:
        raise FileNotFoundError(
            f"action caches are missing: {missing}. Run python -m train_smolvla_frs.prepare_frs_caches first."
        )

    output_dir = Path(str(training["output"])).expanduser()
    rank_source_weights = training.get("high_gate_rank_source_weights") or {}
    if not isinstance(rank_source_weights, Mapping):
        raise ValueError("frs_training.high_gate_rank_source_weights must be a mapping")
    rank_source_weights = {
        str(repo_id): float(weight) for repo_id, weight in rank_source_weights.items()
    }
    tactile_keys = tuple(str(key) for key in model["tactile_keys"])
    train_tactile_encoder = not bool(model.get("freeze_tactile_encoder", True))
    tactile_num_tokens = _positive_int(model, "tactile_num_tokens", len(tactile_keys))
    if tactile_num_tokens != len(tactile_keys):
        raise ValueError(
            "model.tactile_num_tokens must match model.tactile_keys length: "
            f"{tactile_num_tokens} != {len(tactile_keys)}"
        )
    save_run_config(config, output_dir=output_dir)
    train_decoder(
        cache_dir=None,
        tactile_encoder_dir=encoder_dir,
        output_dir=output_dir,
        dataset_repo_id=None,
        dataset_root=None,
        tactile_window_divisor=_positive_int(training, "tactile_window_divisor", 1),
        history_stride=_positive_int(training, "history_stride", 3),
        loss_mode=loss_settings.loss_mode,
        gate_tau=loss_settings.gate_tau,
        gate_temperature=loss_settings.gate_temperature,
        gate_lambda=loss_settings.gate_lambda,
        aux_decode_weight=resolve_optional_loss_weight(
            training.get("aux_decode"),
            float(training.get("aux_decode_weight", 1.0)),
        ),
        aux_decode_steps=_positive_int(training, "aux_decode_steps", 10),
        aux_decode_solver=resolve_decode_solver(training.get("aux_decode_solver", "euler")),
        low_gate_safety_weight=resolve_optional_loss_weight(
            training.get("low_gate_safety"),
            float(training.get("low_gate_safety_weight", 0.0)),
        ),
        low_gate_safety_margin=float(training.get("low_gate_safety_margin", 0.03)),
        rank_weight=resolve_optional_loss_weight(
            training.get("rank"),
            float(training.get("rank_weight", 0.0)),
        ),
        rank_margin=float(training.get("rank_margin", 0.0)),
        repair_weight=resolve_optional_loss_weight(
            training.get("repair"),
            float(training.get("repair_weight", 0.0)),
        ),
        repair_margin=float(training.get("repair_margin", 0.0)),
        rank_low_gate_threshold=float(training.get("rank_low_gate_threshold", 0.3)),
        rank_high_gate_threshold=float(training.get("rank_high_gate_threshold", 0.7)),
        state_conditioning=bool(model.get("state_conditioning", False)),
        state_dropout_rate=float(model.get("state_dropout_rate", 0.0)),
        model_dim=_positive_int(training, "model_dim", 256),
        depth=_positive_int(training, "depth", 6),
        num_heads=_positive_int(training, "num_heads", 4),
        mlp_ratio=_positive_int(training, "mlp_ratio", 4),
        learning_rate=float(training.get("learning_rate", 3e-4)),
        weight_decay=float(training.get("weight_decay", 1e-4)),
        grad_clip_norm=float(training.get("grad_clip_norm", 1.0)),
        warmup_epochs=int(training.get("warmup_epochs", 5)),
        lr_reference_dim=int(training.get("lr_reference_dim", 256)),
        min_learning_rate_ratio=float(training.get("min_lr_ratio", 0.1)),
        cosine_decay=str(training.get("lr_schedule", "cosine")) == "cosine",
        batch_size=_positive_int(training, "batch_size", 64),
        epochs=_positive_int(training, "epochs", 300),
        validation_steps=_positive_int(training, "validation_steps", 10),
        eval_every=_positive_int(training, "eval_every", 5),
        seed=int(training.get("seed", 42)),
        write_plots=bool(training.get("write_plots", True)),
        num_workers=int(
            training.get("num_workers", 8 if train_tactile_encoder else 0)
        ),
        prefetch_batches=_positive_int(
            training,
            "prefetch_batches",
            8 if train_tactile_encoder else 1,
        ),
        load_threads=_positive_int(training, "load_threads", 8),
        pipeline_prefetch=_positive_int(
            training,
            "pipeline_prefetch",
            4 if train_tactile_encoder else 1,
        ),
        image_cache_size=int(training.get("image_cache_size", 8192)),
        encode_batch_size=_positive_int(training, "encode_batch_size", 64),
        resume=resolve_resume_mode(training.get("resume", False), output_dir=output_dir),
        resume_from=(
            None if training.get("resume_from") in (None, "") else Path(str(training["resume_from"])).expanduser()
        ),
        init_from=(
            None if training.get("init_from") in (None, "") else Path(str(training["init_from"])).expanduser()
        ),
        cache_dirs=cache_dirs,
        dataset_sources=datasets,
        tactile_embedding_cache_root=Path(str(tactile_cache["root"])).expanduser(),
        tactile_keys=tactile_keys,
        tactile_embedding_dim=int(model.get("tactile_embedding_dim", 512)),
        tactile_image_size=int(model.get("tactile_image_size", 224)),
        tactile_num_tokens=tactile_num_tokens,
        train_tactile_encoder=train_tactile_encoder,
        tactile_encode_microbatch_size=_positive_int(
            training,
            "tactile_encode_microbatch_size",
            8,
        ),
        best_max_low_gate_unsafe_frac=float(
            training.get("best_max_low_gate_unsafe_frac", 0.1)
        ),
        best_min_high_gate_gain=float(training.get("best_min_high_gate_gain", 0.0)),
        best_min_high_gate_rank_satisfied=float(
            training.get("best_min_high_gate_rank_satisfied", 0.8)
        ),
        dataset_balanced_sampling=bool(training.get("dataset_balanced_sampling", False)),
        dataset_balanced_loss=bool(training.get("dataset_balanced_loss", False)),
        early_stop_patience=int(training.get("early_stop_patience", 0)),
        early_stop_min_evals=int(training.get("early_stop_min_evals", 0)),
        high_gate_rank_aggregation=str(
            training.get("high_gate_rank_aggregation", "balanced_mean")
        ),
        high_gate_rank_hard_fraction=float(
            training.get("high_gate_rank_hard_fraction", 0.3)
        ),
        high_gate_rank_worst_beta=float(
            training.get("high_gate_rank_worst_beta", 20.0)
        ),
        high_gate_rank_source_weights=rank_source_weights,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train multi-dataset tactile FRS from a YAML config.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    train_from_config(load_config(args.config))


if __name__ == "__main__":
    main()
