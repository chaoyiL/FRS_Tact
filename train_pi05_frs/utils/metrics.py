from __future__ import annotations

import dataclasses
from typing import Literal

import jax
import jax.numpy as jnp
import numpy as np

from train_pi05_frs.utils.bimanual_metrics import bimanual_gate_region_counts
from train_pi05_frs.utils.bimanual_metrics import bimanual_quadrant_metrics
from train_pi05_frs.utils.bimanual_schema import BIMANUAL_LOSS_MODE
from train_pi05_frs.utils.bimanual_schema import STEERED_ACTION_DIM
from train_pi05_frs.utils.data import TactileConditionedBatches
from train_pi05_frs.utils.model import FlowSolver
from train_pi05_frs.utils.model import TactileConditionedFlowDecoder
from train_pi05_frs.utils.model import bimanual_composite_endpoint
from train_pi05_frs.utils.model import bimanual_mse_per_sample
from train_pi05_frs.utils.model import composite_endpoint
from train_pi05_frs.utils.model import decode_actions
from train_pi05_frs.utils.model import decode_bimanual_actions
from train_pi05_frs.utils.model import flow_matching_loss_per_sample
from train_pi05_frs.utils.model import masked_flow_matching_loss_per_sample
from train_pi05_frs.utils.objective_schema import COMPOSITE_GATED_LOSS_MODE

EvalTarget = Literal["gt", "predicted"]


@dataclasses.dataclass(frozen=True)
class EvaluationResult:
    """Primary metrics follow ``target``; both GT and predicted decode errors are always filled."""

    target: EvalTarget
    flow_loss: float
    mse: float
    rmse: float
    mae: float
    flow_loss_gt: float
    mse_gt: float
    rmse_gt: float
    mae_gt: float
    flow_loss_pred: float
    mse_pred: float
    rmse_pred: float
    mae_pred: float
    cache_indices: np.ndarray
    sample_flow_loss: np.ndarray
    sample_mse: np.ndarray
    sample_rmse: np.ndarray
    sample_mae: np.ndarray
    sample_mse_gt: np.ndarray
    sample_mae_gt: np.ndarray
    sample_mse_pred: np.ndarray
    sample_mae_pred: np.ndarray
    predictions: np.ndarray | None
    tactile_change: float | None = None
    tactile_sim: float | None = None
    gate_w: float | None = None
    gate_active_frac: float | None = None
    # Gate-stratified decode metrics (filled when tactile gate is tracked).
    mse_gt_high_w: float | None = None
    mse_gt_low_w: float | None = None
    mse_pred_high_w: float | None = None
    mse_pred_low_w: float | None = None
    n_high_w: int | None = None
    n_low_w: int | None = None
    low_gate_unsafe_frac: float | None = None
    low_gate_regression_frac: float | None = None
    high_gate_gain: float | None = None
    high_gate_harm_p95: float | None = None
    high_gate_rank_satisfied_frac: float | None = None
    high_gate_repair_satisfied_frac: float | None = None
    # Endpoint-baseline metrics are additive and do not alter legacy fields.
    mse_vla_gt: float | None = None
    gt_gain: float | None = None
    relative_gt_error: float | None = None
    sample_mse_vla_gt: np.ndarray | None = None
    sample_gt_gain: np.ndarray | None = None
    sample_relative_gt_error: np.ndarray | None = None
    sample_tactile_change: np.ndarray | None = None
    sample_gate_w: np.ndarray | None = None
    # Bimanual validation fields. Every wrist error is over physical dims 0:20.
    sample_gate_w_left: np.ndarray | None = None
    sample_gate_w_right: np.ndarray | None = None
    composite_fm: float | None = None
    sample_composite_fm: np.ndarray | None = None
    tactile_change_left: float | None = None
    tactile_change_right: float | None = None
    sample_tactile_change_left: np.ndarray | None = None
    sample_tactile_change_right: np.ndarray | None = None
    gate_w_left: float | None = None
    gate_w_right: float | None = None
    gate_w_p10_left: float | None = None
    gate_w_p25_left: float | None = None
    gate_w_p50_left: float | None = None
    gate_w_p75_left: float | None = None
    gate_w_p90_left: float | None = None
    gate_w_p10_right: float | None = None
    gate_w_p25_right: float | None = None
    gate_w_p50_right: float | None = None
    gate_w_p75_right: float | None = None
    gate_w_p90_right: float | None = None
    tactile_change_p10_left: float | None = None
    tactile_change_p25_left: float | None = None
    tactile_change_p50_left: float | None = None
    tactile_change_p75_left: float | None = None
    tactile_change_p90_left: float | None = None
    tactile_change_p10_right: float | None = None
    tactile_change_p25_right: float | None = None
    tactile_change_p50_right: float | None = None
    tactile_change_p75_right: float | None = None
    tactile_change_p90_right: float | None = None
    sample_mse_gt_left: np.ndarray | None = None
    sample_mse_gt_right: np.ndarray | None = None
    sample_mse_vla_left: np.ndarray | None = None
    sample_mse_vla_right: np.ndarray | None = None
    sample_mse_vla_gt_left: np.ndarray | None = None
    sample_mse_vla_gt_right: np.ndarray | None = None
    mse_gt_high_w_left: float | None = None
    mse_gt_high_w_right: float | None = None
    mse_vla_high_w_left: float | None = None
    mse_vla_high_w_right: float | None = None
    mse_vla_gt_high_w_left: float | None = None
    mse_vla_gt_high_w_right: float | None = None
    gt_gain_high_w_left: float | None = None
    gt_gain_high_w_right: float | None = None
    rank_penalty_high_w_left: float | None = None
    rank_penalty_high_w_right: float | None = None
    rank_satisfied_high_frac_left: float | None = None
    rank_satisfied_high_frac_right: float | None = None
    repair_penalty_high_w_left: float | None = None
    repair_penalty_high_w_right: float | None = None
    repair_satisfied_high_frac_left: float | None = None
    repair_satisfied_high_frac_right: float | None = None
    low_nearest_endpoint_mse_left: float | None = None
    low_nearest_endpoint_mse_right: float | None = None
    low_safety_penalty_left: float | None = None
    low_safety_penalty_right: float | None = None
    low_safe_frac_left: float | None = None
    low_safe_frac_right: float | None = None
    low_unsafe_frac_left: float | None = None
    low_unsafe_frac_right: float | None = None
    n_high_w_left: int | None = None
    n_high_w_right: int | None = None
    n_low_w_left: int | None = None
    n_low_w_right: int | None = None
    n_mid_w_left: int | None = None
    n_mid_w_right: int | None = None
    bimanual_quadrants: dict[str, dict[str, object]] | None = None
    bimanual_gate_region_counts: np.ndarray | None = None
    gate_low_threshold: float = 0.3
    gate_high_threshold: float = 0.7
    gt_actions: np.ndarray | None = None
    vla_actions: np.ndarray | None = None


def _per_sample_errors(prediction: jax.Array, reference: jax.Array) -> tuple[jax.Array, jax.Array]:
    difference = prediction - reference
    mse = jnp.mean(jnp.square(difference), axis=(1, 2))
    mae = jnp.mean(jnp.abs(difference), axis=(1, 2))
    return mse, mae


def _mean_or_nan(values: np.ndarray) -> float:
    if values.size == 0:
        return float("nan")
    return float(np.mean(values))


def gate_stratified_decode_metrics(
    sample_mse_gt: np.ndarray,
    sample_mse_pred: np.ndarray,
    gate_weights: np.ndarray,
    *,
    high_w_threshold: float = 0.5,
) -> dict[str, float | int]:
    """Split decode MSE by gate weight ``w > threshold`` vs ``w <= threshold``."""

    mse_gt = np.asarray(sample_mse_gt, dtype=np.float64)
    mse_pred = np.asarray(sample_mse_pred, dtype=np.float64)
    weights = np.asarray(gate_weights, dtype=np.float64)
    if mse_gt.shape != mse_pred.shape or mse_gt.shape != weights.shape:
        raise ValueError(
            f"Shape mismatch: mse_gt={mse_gt.shape}, mse_pred={mse_pred.shape}, "
            f"gate_w={weights.shape}."
        )
    high = weights > float(high_w_threshold)
    low = ~high
    return {
        "mse_gt_high_w": _mean_or_nan(mse_gt[high]),
        "mse_gt_low_w": _mean_or_nan(mse_gt[low]),
        "mse_pred_high_w": _mean_or_nan(mse_pred[high]),
        "mse_pred_low_w": _mean_or_nan(mse_pred[low]),
        "n_high_w": int(np.count_nonzero(high)),
        "n_low_w": int(np.count_nonzero(low)),
    }


def _ratio_of_means(numerator: np.ndarray, denominator: np.ndarray) -> float:
    if numerator.size == 0 or denominator.size == 0:
        return float("nan")
    return float(np.mean(numerator) / max(float(np.mean(denominator)), 1e-8))


def _quantiles(values: np.ndarray) -> tuple[float, float, float, float, float]:
    return tuple(
        float(value)
        for value in np.quantile(
            np.asarray(values, dtype=np.float64), [0.1, 0.25, 0.5, 0.75, 0.9]
        )
    )  # type: ignore[return-value]


def _bimanual_wrist_decode_metrics(
    sample_mse_gt: np.ndarray,
    sample_mse_vla: np.ndarray,
    sample_mse_vla_gt: np.ndarray,
    gate_weights: np.ndarray,
    *,
    low_w_threshold: float,
    high_w_threshold: float,
    ranking_margin: float,
    repair_margin: float,
    low_safety_margin: float,
) -> dict[str, float | int]:
    if ranking_margin < 0 or repair_margin < 0 or low_safety_margin < 0:
        raise ValueError("bimanual validation margins must be non-negative")
    if not 0.0 <= low_w_threshold < high_w_threshold <= 1.0:
        raise ValueError("gate thresholds must satisfy 0 <= low < high <= 1")
    gt = np.asarray(sample_mse_gt, dtype=np.float64)
    vla = np.asarray(sample_mse_vla, dtype=np.float64)
    baseline = np.asarray(sample_mse_vla_gt, dtype=np.float64)
    gates = np.asarray(gate_weights, dtype=np.float64)
    if not (gt.shape == vla.shape == baseline.shape == gates.shape) or gt.ndim != 1:
        raise ValueError("bimanual wrist metric arrays must be shape-matched [N]")
    if any(np.any(~np.isfinite(value)) for value in (gt, vla, baseline, gates)):
        raise ValueError("bimanual wrist metric arrays must be finite")

    high = gates >= float(high_w_threshold)
    low = gates <= float(low_w_threshold)
    middle = ~(high | low)
    nearest = np.minimum(gt, vla)
    high_rank_penalty = np.maximum(gt - vla + float(ranking_margin), 0.0)
    high_repair_penalty = np.maximum(
        gt - baseline + float(repair_margin), 0.0
    )
    low_penalty = np.maximum(nearest - float(low_safety_margin), 0.0)
    return {
        "mse_gt_high_w": _mean_or_nan(gt[high]),
        "mse_vla_high_w": _mean_or_nan(vla[high]),
        "mse_vla_gt_high_w": _mean_or_nan(baseline[high]),
        "gt_gain_high_w": _mean_or_nan((baseline - gt)[high]),
        "rank_penalty_high_w": _mean_or_nan(high_rank_penalty[high]),
        "rank_satisfied_high_frac": _mean_or_nan(
            (gt[high] + float(ranking_margin) <= vla[high]).astype(np.float32)
        ),
        "repair_penalty_high_w": _mean_or_nan(high_repair_penalty[high]),
        "repair_satisfied_high_frac": _mean_or_nan(
            (gt[high] + float(repair_margin) <= baseline[high]).astype(np.float32)
        ),
        "low_nearest_endpoint_mse": _mean_or_nan(nearest[low]),
        "low_safety_penalty": _mean_or_nan(low_penalty[low]),
        "low_safe_frac": _mean_or_nan(
            (nearest[low] <= float(low_safety_margin)).astype(np.float32)
        ),
        "low_unsafe_frac": _mean_or_nan(
            (nearest[low] > float(low_safety_margin)).astype(np.float32)
        ),
        "n_high_w": int(np.count_nonzero(high)),
        "n_low_w": int(np.count_nonzero(low)),
        "n_mid_w": int(np.count_nonzero(middle)),
    }


def bimanual_source_decode_metrics(
    *,
    sample_mse_gt_left: np.ndarray,
    sample_mse_gt_right: np.ndarray,
    sample_mse_vla_left: np.ndarray,
    sample_mse_vla_right: np.ndarray,
    sample_mse_vla_gt_left: np.ndarray,
    sample_mse_vla_gt_right: np.ndarray,
    sample_gate_w_left: np.ndarray,
    sample_gate_w_right: np.ndarray,
    source_indices: np.ndarray,
    num_sources: int,
    low_w_threshold: float,
    high_w_threshold: float,
    ranking_margin: float,
    repair_margin: float,
    low_safety_margin: float,
) -> tuple[dict[int, dict[str, float | int]], dict[str, float]]:
    """Aggregate within each source and wrist before worst/min rollups."""
    if num_sources <= 0:
        raise ValueError(f"num_sources must be positive, got {num_sources}.")
    arrays = {
        "mse_gt_left": sample_mse_gt_left,
        "mse_gt_right": sample_mse_gt_right,
        "mse_vla_left": sample_mse_vla_left,
        "mse_vla_right": sample_mse_vla_right,
        "mse_vla_gt_left": sample_mse_vla_gt_left,
        "mse_vla_gt_right": sample_mse_vla_gt_right,
        "gate_w_left": sample_gate_w_left,
        "gate_w_right": sample_gate_w_right,
        "source_indices": source_indices,
    }
    converted = {name: np.asarray(value) for name, value in arrays.items()}
    expected_shape = converted["source_indices"].shape
    if len(expected_shape) != 1 or any(
        value.shape != expected_shape for value in converted.values()
    ):
        shapes = {name: value.shape for name, value in converted.items()}
        raise ValueError(
            f"bimanual source metric arrays must be shape-matched [N], got {shapes}."
        )
    sources = converted.pop("source_indices").astype(np.int64, copy=False)
    if np.any((sources < 0) | (sources >= num_sources)):
        raise ValueError(
            f"source indices must be in [0, {num_sources}), got {sources}."
        )

    per_source: dict[int, dict[str, float | int]] = {}
    rollups: dict[str, float] = {}
    for wrist in ("left", "right"):
        collected: dict[str, list[float]] = {
            "low_safety_penalty": [],
            "low_unsafe_frac": [],
            "gt_gain_high_w": [],
            "rank_penalty_high_w": [],
            "rank_gap_high_w": [],
            "rank_satisfied_high_frac": [],
            "repair_penalty_high_w": [],
            "repair_satisfied_high_frac": [],
        }
        missing_low = False
        missing_high = False
        for source_index in range(num_sources):
            source_mask = sources == source_index
            gt = converted[f"mse_gt_{wrist}"][source_mask]
            vla = converted[f"mse_vla_{wrist}"][source_mask]
            baseline = converted[f"mse_vla_gt_{wrist}"][source_mask]
            gate = converted[f"gate_w_{wrist}"][source_mask]
            stratified = _bimanual_wrist_decode_metrics(
                gt,
                vla,
                baseline,
                gate,
                low_w_threshold=low_w_threshold,
                high_w_threshold=high_w_threshold,
                ranking_margin=ranking_margin,
                repair_margin=repair_margin,
                low_safety_margin=low_safety_margin,
            )
            source_metrics = per_source.setdefault(source_index, {})
            for metric_name, value in stratified.items():
                source_metrics[f"{metric_name}_{wrist}"] = value
            high = gate >= float(high_w_threshold)
            rank_gap = _mean_or_nan((gt - vla)[high])
            source_metrics[f"rank_gap_high_w_{wrist}"] = rank_gap
            if int(stratified["n_low_w"]) == 0:
                missing_low = True
            else:
                for name in ("low_safety_penalty", "low_unsafe_frac"):
                    collected[name].append(float(stratified[name]))
            if int(stratified["n_high_w"]) == 0:
                missing_high = True
            else:
                for name in (
                    "gt_gain_high_w",
                    "rank_penalty_high_w",
                    "rank_satisfied_high_frac",
                    "repair_penalty_high_w",
                    "repair_satisfied_high_frac",
                ):
                    collected[name].append(float(stratified[name]))
                collected["rank_gap_high_w"].append(rank_gap)

        rollups[f"worst_dataset_low_safety_penalty_{wrist}"] = (
            float("nan")
            if missing_low
            else max(collected["low_safety_penalty"])
        )
        rollups[f"worst_dataset_low_unsafe_frac_{wrist}"] = (
            float("nan") if missing_low else max(collected["low_unsafe_frac"])
        )
        rollups[f"min_dataset_gt_gain_high_w_{wrist}"] = (
            float("nan") if missing_high else min(collected["gt_gain_high_w"])
        )
        rollups[f"worst_dataset_rank_violation_high_w_{wrist}"] = (
            float("nan")
            if missing_high
            else max(collected["rank_penalty_high_w"])
        )
        rollups[f"worst_dataset_rank_gap_high_w_{wrist}"] = (
            float("nan") if missing_high else max(collected["rank_gap_high_w"])
        )
        rollups[f"min_dataset_rank_satisfied_high_frac_{wrist}"] = (
            float("nan")
            if missing_high
            else min(collected["rank_satisfied_high_frac"])
        )
        rollups[f"worst_dataset_repair_penalty_high_w_{wrist}"] = (
            float("nan")
            if missing_high
            else max(collected["repair_penalty_high_w"])
        )
        rollups[f"min_dataset_repair_satisfied_high_frac_{wrist}"] = (
            float("nan")
            if missing_high
            else min(collected["repair_satisfied_high_frac"])
        )
    return per_source, rollups


def _evaluate_split_legacy(
    model: TactileConditionedFlowDecoder,
    conditioner: TactileConditionedBatches,
    *,
    split: str,
    batch_size: int,
    num_steps: int,
    keep_predictions: bool,
    solver: FlowSolver = "euler",
    target: EvalTarget = "gt",
    gate_tau: float | None = None,
    gate_temperature: float | None = None,
    low_gate_threshold: float = 0.3,
    high_gate_threshold: float = 0.7,
    low_gate_safety_margin: float = 0.03,
    low_gate_regression_margin: float = 0.005,
    rank_margin: float = 0.0,
    repair_margin: float = 0.0,
    track_composite: bool = False,
) -> EvaluationResult:
    from train_pi05_frs.utils.data import gate_weights_from_change

    if target not in ("gt", "predicted"):
        raise ValueError(f"target must be 'gt' or 'predicted', got {target!r}.")

    cache_indices: list[np.ndarray] = []
    flow_gt_parts: list[np.ndarray] = []
    flow_pred_parts: list[np.ndarray] = []
    mse_gt_parts: list[np.ndarray] = []
    mae_gt_parts: list[np.ndarray] = []
    mse_pred_parts: list[np.ndarray] = []
    mae_pred_parts: list[np.ndarray] = []
    pi05_gt_mse_parts: list[np.ndarray] = []
    composite_parts: list[np.ndarray] = []
    predictions: list[np.ndarray] = []
    gt_actions: list[np.ndarray] = []
    vla_actions: list[np.ndarray] = []
    tactile_changes: list[np.ndarray] = []
    gate_weights: list[np.ndarray] = []
    track_tactile = (
        gate_tau is not None
        and gate_temperature is not None
        and bool(conditioner.episode_baselines)
    )

    for indices, x_base_np, predicted_np, gt_action_np, state_np, tactile_seq in conditioner.batches(
        split, batch_size=batch_size, shuffle=False, seed=0
    ):
        x_base = jnp.asarray(x_base_np)
        gt_action = jnp.asarray(gt_action_np)
        predicted_action = jnp.asarray(predicted_np)
        state = jnp.asarray(state_np)
        t = jnp.full((len(indices),), 0.5, dtype=jnp.float32)
        flow_gt = flow_matching_loss_per_sample(
            model, x_base, gt_action, t, tactile_seq, state=state
        )
        flow_pred = flow_matching_loss_per_sample(
            model, x_base, predicted_action, t, tactile_seq, state=state
        )
        prediction = decode_actions(
            model, x_base, tactile_seq, num_steps=num_steps, solver=solver, state=state
        )
        endpoint_width = (
            9 if track_composite and prediction.shape[-1] == 10 else prediction.shape[-1]
        )
        endpoint_prediction = prediction[..., :endpoint_width]
        endpoint_gt = gt_action[..., :endpoint_width]
        endpoint_vla = predicted_action[..., :endpoint_width]
        mse_gt, mae_gt = _per_sample_errors(endpoint_prediction, endpoint_gt)
        mse_pred, mae_pred = _per_sample_errors(endpoint_prediction, endpoint_vla)
        pi05_gt_mse, _ = _per_sample_errors(endpoint_vla, endpoint_gt)

        cache_indices.append(indices)
        flow_gt_parts.append(np.asarray(jax.device_get(flow_gt)))
        flow_pred_parts.append(np.asarray(jax.device_get(flow_pred)))
        mse_gt_parts.append(np.asarray(jax.device_get(mse_gt)))
        mae_gt_parts.append(np.asarray(jax.device_get(mae_gt)))
        mse_pred_parts.append(np.asarray(jax.device_get(mse_pred)))
        mae_pred_parts.append(np.asarray(jax.device_get(mae_pred)))
        pi05_gt_mse_parts.append(np.asarray(jax.device_get(pi05_gt_mse)))
        if keep_predictions:
            predictions.append(np.asarray(jax.device_get(prediction), dtype=np.float32))
            gt_actions.append(np.asarray(jax.device_get(gt_action), dtype=np.float32))
            vla_actions.append(
                np.asarray(jax.device_get(predicted_action), dtype=np.float32)
            )
        if track_tactile:
            current_tokens = np.asarray(tactile_seq[:, -1, :, :], dtype=np.float32)
            change = conditioner.tactile_change_for_cache_indices(indices, current_tokens)
            gate_w = gate_weights_from_change(
                change, tau=float(gate_tau), temperature=float(gate_temperature)
            )
            tactile_changes.append(change)
            gate_weights.append(gate_w)
            if track_composite:
                composite_target, _ = composite_endpoint(
                    gt_action,
                    predicted_action,
                    jnp.asarray(gate_w),
                    low_gate_threshold=low_gate_threshold,
                    high_gate_threshold=high_gate_threshold,
                    steered_action_dim=(9 if gt_action.shape[-1] == 10 else None),
                )
                composite_flow = flow_matching_loss_per_sample(
                    model,
                    x_base,
                    composite_target,
                    t,
                    tactile_seq,
                    state=state,
                )
                composite_parts.append(
                    np.asarray(jax.device_get(composite_flow), dtype=np.float32)
                )

    if not cache_indices:
        raise ValueError(f"No samples found for split {split!r}.")

    all_indices = np.concatenate(cache_indices)
    all_flow_gt = np.concatenate(flow_gt_parts)
    all_flow_pred = np.concatenate(flow_pred_parts)
    all_mse_gt = np.concatenate(mse_gt_parts)
    all_mae_gt = np.concatenate(mae_gt_parts)
    all_mse_pred = np.concatenate(mse_pred_parts)
    all_mae_pred = np.concatenate(mae_pred_parts)
    all_pi05_gt_mse = np.concatenate(pi05_gt_mse_parts)
    all_gt_gain = all_pi05_gt_mse - all_mse_gt
    all_relative_gt_error = all_mse_gt / np.maximum(all_pi05_gt_mse, 1e-8)
    all_composite = np.concatenate(composite_parts) if composite_parts else None
    if target == "gt":
        primary_flow, primary_mse, primary_mae = all_flow_gt, all_mse_gt, all_mae_gt
    else:
        primary_flow, primary_mse, primary_mae = all_flow_pred, all_mse_pred, all_mae_pred
    primary_rmse = np.sqrt(primary_mse)

    if tactile_changes:
        all_change = np.concatenate(tactile_changes)
        all_gate = np.concatenate(gate_weights)
        tactile_change = float(np.mean(all_change))
        tactile_sim = float(np.mean(1.0 - all_change))
        gate_w_mean = float(np.mean(all_gate))
        gate_active_frac = float(np.mean(all_gate > 0.5))
        stratified = gate_stratified_decode_metrics(all_mse_gt, all_mse_pred, all_gate)
        low = all_gate <= float(low_gate_threshold)
        high = all_gate >= float(high_gate_threshold)
        low_gate_unsafe_frac = _mean_or_nan(
            (all_mse_pred[low] > low_gate_safety_margin).astype(np.float32)
        )
        low_gate_regression_frac = _mean_or_nan(
            (
                all_mse_gt[low]
                > all_pi05_gt_mse[low] + float(low_gate_regression_margin)
            ).astype(np.float32)
        )
        high_gate_gain = _mean_or_nan(all_pi05_gt_mse[high] - all_mse_gt[high])
        high_gate_harm_p95 = (
            float(
                np.quantile(
                    np.maximum(all_mse_gt[high] - all_pi05_gt_mse[high], 0.0),
                    0.95,
                )
            )
            if np.any(high)
            else float("nan")
        )
        high_gate_rank_satisfied_frac = _mean_or_nan(
            (all_mse_gt[high] + rank_margin <= all_mse_pred[high]).astype(np.float32)
        )
        high_gate_repair_satisfied_frac = _mean_or_nan(
            (all_mse_gt[high] + repair_margin <= all_pi05_gt_mse[high]).astype(np.float32)
        )
    else:
        tactile_change = None
        tactile_sim = None
        gate_w_mean = None
        gate_active_frac = None
        stratified = {
            "mse_gt_high_w": None,
            "mse_gt_low_w": None,
            "mse_pred_high_w": None,
            "mse_pred_low_w": None,
            "n_high_w": None,
            "n_low_w": None,
        }
        low_gate_unsafe_frac = None
        low_gate_regression_frac = None
        high_gate_gain = None
        high_gate_harm_p95 = None
        high_gate_rank_satisfied_frac = None
        high_gate_repair_satisfied_frac = None
        all_change = None
        all_gate = None

    return EvaluationResult(
        target=target,
        flow_loss=float(np.mean(primary_flow)),
        mse=float(np.mean(primary_mse)),
        rmse=float(np.sqrt(np.mean(primary_mse))),
        mae=float(np.mean(primary_mae)),
        flow_loss_gt=float(np.mean(all_flow_gt)),
        mse_gt=float(np.mean(all_mse_gt)),
        rmse_gt=float(np.sqrt(np.mean(all_mse_gt))),
        mae_gt=float(np.mean(all_mae_gt)),
        flow_loss_pred=float(np.mean(all_flow_pred)),
        mse_pred=float(np.mean(all_mse_pred)),
        rmse_pred=float(np.sqrt(np.mean(all_mse_pred))),
        mae_pred=float(np.mean(all_mae_pred)),
        cache_indices=all_indices,
        sample_flow_loss=primary_flow,
        sample_mse=primary_mse,
        sample_rmse=primary_rmse,
        sample_mae=primary_mae,
        sample_mse_gt=all_mse_gt,
        sample_mae_gt=all_mae_gt,
        sample_mse_pred=all_mse_pred,
        sample_mae_pred=all_mae_pred,
        predictions=np.concatenate(predictions) if keep_predictions else None,
        tactile_change=tactile_change,
        tactile_sim=tactile_sim,
        gate_w=gate_w_mean,
        gate_active_frac=gate_active_frac,
        mse_gt_high_w=stratified["mse_gt_high_w"],  # type: ignore[arg-type]
        mse_gt_low_w=stratified["mse_gt_low_w"],  # type: ignore[arg-type]
        mse_pred_high_w=stratified["mse_pred_high_w"],  # type: ignore[arg-type]
        mse_pred_low_w=stratified["mse_pred_low_w"],  # type: ignore[arg-type]
        n_high_w=stratified["n_high_w"],  # type: ignore[arg-type]
        n_low_w=stratified["n_low_w"],  # type: ignore[arg-type]
        low_gate_unsafe_frac=low_gate_unsafe_frac,
        low_gate_regression_frac=low_gate_regression_frac,
        high_gate_gain=high_gate_gain,
        high_gate_harm_p95=high_gate_harm_p95,
        high_gate_rank_satisfied_frac=high_gate_rank_satisfied_frac,
        high_gate_repair_satisfied_frac=high_gate_repair_satisfied_frac,
        mse_vla_gt=float(np.mean(all_pi05_gt_mse)),
        gt_gain=float(np.mean(all_gt_gain)),
        relative_gt_error=_ratio_of_means(all_mse_gt, all_pi05_gt_mse),
        sample_mse_vla_gt=all_pi05_gt_mse,
        sample_gt_gain=all_gt_gain,
        sample_relative_gt_error=all_relative_gt_error,
        sample_tactile_change=all_change,
        sample_gate_w=all_gate,
        composite_fm=(
            None if all_composite is None else float(np.mean(all_composite))
        ),
        sample_composite_fm=all_composite,
        gate_low_threshold=float(low_gate_threshold),
        gate_high_threshold=float(high_gate_threshold),
        gt_actions=np.concatenate(gt_actions) if gt_actions else None,
        vla_actions=np.concatenate(vla_actions) if vla_actions else None,
    )


def evaluate_split(
    model: TactileConditionedFlowDecoder,
    conditioner: TactileConditionedBatches,
    *,
    split: str,
    batch_size: int,
    num_steps: int,
    keep_predictions: bool,
    solver: FlowSolver = "euler",
    target: EvalTarget = "gt",
    gate_tau: float | None = None,
    gate_temperature: float | None = None,
    low_gate_threshold: float = 0.3,
    high_gate_threshold: float = 0.7,
    low_gate_safety_margin: float = 0.03,
    low_gate_regression_margin: float = 0.005,
    rank_margin: float = 0.0,
    repair_margin: float = 0.0,
    loss_mode: str | None = None,
    rank_low_gate_threshold: float | None = None,
    rank_high_gate_threshold: float | None = None,
) -> EvaluationResult:
    """Evaluate legacy/scalar modes or add independent wrist metrics for bimanual.

    Scalar composite evaluation additionally retains its Gate, frozen-VLA
    baseline, composite FM, and optional action arrays for live diagnostics.
    Bimanual endpoint metrics and composite FM are restricted to the first 20
    physical action dimensions; retained action arrays keep model width.
    """
    if loss_mode != BIMANUAL_LOSS_MODE:
        return _evaluate_split_legacy(
            model,
            conditioner,
            split=split,
            batch_size=batch_size,
            num_steps=num_steps,
            keep_predictions=keep_predictions,
            solver=solver,
            target=target,
            gate_tau=gate_tau,
            gate_temperature=gate_temperature,
            low_gate_threshold=low_gate_threshold,
            high_gate_threshold=high_gate_threshold,
            low_gate_safety_margin=low_gate_safety_margin,
            low_gate_regression_margin=low_gate_regression_margin,
            rank_margin=rank_margin,
            repair_margin=repair_margin,
            track_composite=(loss_mode == COMPOSITE_GATED_LOSS_MODE),
        )
    from train_pi05_frs.utils.data import gate_weights_from_change

    if target not in ("gt", "predicted"):
        raise ValueError(f"target must be 'gt' or 'predicted', got {target!r}.")
    low_threshold = (
        float(low_gate_threshold)
        if rank_low_gate_threshold is None
        else float(rank_low_gate_threshold)
    )
    high_threshold = (
        float(high_gate_threshold)
        if rank_high_gate_threshold is None
        else float(rank_high_gate_threshold)
    )
    if gate_tau is None or gate_temperature is None or not bool(
        conditioner.episode_baselines
    ):
        raise ValueError(
            "bimanual validation requires Gate metadata and episode baselines"
        )

    cache_index_parts: list[np.ndarray] = []
    flow_gt_parts: list[np.ndarray] = []
    flow_pred_parts: list[np.ndarray] = []
    composite_parts: list[np.ndarray] = []
    mse_gt_parts: list[np.ndarray] = []
    mae_gt_parts: list[np.ndarray] = []
    mse_pred_parts: list[np.ndarray] = []
    mae_pred_parts: list[np.ndarray] = []
    mse_vla_gt_parts: list[np.ndarray] = []
    wrist_gt_parts: list[np.ndarray] = []
    wrist_vla_parts: list[np.ndarray] = []
    wrist_vla_gt_parts: list[np.ndarray] = []
    change_parts: list[np.ndarray] = []
    gate_parts: list[np.ndarray] = []
    prediction_parts: list[np.ndarray] = []
    gt_action_parts: list[np.ndarray] = []
    vla_action_parts: list[np.ndarray] = []

    for (
        indices,
        x_base_np,
        predicted_np,
        gt_action_np,
        state_np,
        tactile_seq,
    ) in conditioner.batches(
        split, batch_size=batch_size, shuffle=False, seed=0
    ):
        x_base = jnp.asarray(x_base_np)
        gt_action = jnp.asarray(gt_action_np)
        vla_action = jnp.asarray(predicted_np)
        state = jnp.asarray(state_np)
        tactile_seq = jnp.asarray(tactile_seq)
        current_tokens = np.asarray(tactile_seq[:, -1, :, :], dtype=np.float32)
        change = conditioner.tactile_change_per_wrist_for_cache_indices(
            indices, current_tokens
        )
        gates = gate_weights_from_change(
            change, tau=float(gate_tau), temperature=float(gate_temperature)
        )
        if gates.shape != (len(indices), 2) or not np.all(np.isfinite(gates)):
            raise ValueError(
                "bimanual validation requires finite per-wrist gate weights"
            )
        t = jnp.full((len(indices),), 0.5, dtype=jnp.float32)
        flow_gt = masked_flow_matching_loss_per_sample(
            model, x_base, gt_action, t, tactile_seq, state=state
        )
        flow_pred = masked_flow_matching_loss_per_sample(
            model, x_base, vla_action, t, tactile_seq, state=state
        )
        composite_target, _ = bimanual_composite_endpoint(
            gt_action,
            vla_action,
            jnp.asarray(gates),
            low_gate_threshold=low_threshold,
            high_gate_threshold=high_threshold,
        )
        composite_flow = masked_flow_matching_loss_per_sample(
            model, x_base, composite_target, t, tactile_seq, state=state
        )
        prediction = decode_bimanual_actions(
            model,
            x_base,
            tactile_seq,
            frozen_endpoint=vla_action,
            num_steps=num_steps,
            solver=solver,
            state=state,
        )
        mse_gt, mae_gt = _per_sample_errors(
            prediction[..., :STEERED_ACTION_DIM],
            gt_action[..., :STEERED_ACTION_DIM],
        )
        mse_pred, mae_pred = _per_sample_errors(
            prediction[..., :STEERED_ACTION_DIM],
            vla_action[..., :STEERED_ACTION_DIM],
        )
        mse_vla_gt, _ = _per_sample_errors(
            vla_action[..., :STEERED_ACTION_DIM],
            gt_action[..., :STEERED_ACTION_DIM],
        )

        cache_index_parts.append(np.asarray(indices, dtype=np.int64))
        flow_gt_parts.append(np.asarray(jax.device_get(flow_gt)))
        flow_pred_parts.append(np.asarray(jax.device_get(flow_pred)))
        composite_parts.append(np.asarray(jax.device_get(composite_flow)))
        mse_gt_parts.append(np.asarray(jax.device_get(mse_gt)))
        mae_gt_parts.append(np.asarray(jax.device_get(mae_gt)))
        mse_pred_parts.append(np.asarray(jax.device_get(mse_pred)))
        mae_pred_parts.append(np.asarray(jax.device_get(mae_pred)))
        mse_vla_gt_parts.append(np.asarray(jax.device_get(mse_vla_gt)))
        wrist_gt_parts.append(
            np.asarray(jax.device_get(bimanual_mse_per_sample(prediction, gt_action)))
        )
        wrist_vla_parts.append(
            np.asarray(jax.device_get(bimanual_mse_per_sample(prediction, vla_action)))
        )
        wrist_vla_gt_parts.append(
            np.asarray(jax.device_get(bimanual_mse_per_sample(vla_action, gt_action)))
        )
        change_parts.append(np.asarray(change, dtype=np.float32))
        gate_parts.append(np.asarray(gates, dtype=np.float32))
        if keep_predictions:
            prediction_parts.append(
                np.asarray(jax.device_get(prediction), dtype=np.float32)
            )
            gt_action_parts.append(
                np.asarray(jax.device_get(gt_action), dtype=np.float32)
            )
            vla_action_parts.append(
                np.asarray(jax.device_get(vla_action), dtype=np.float32)
            )

    if not cache_index_parts:
        raise ValueError(f"No samples found for split {split!r}.")

    all_indices = np.concatenate(cache_index_parts)
    all_flow_gt = np.concatenate(flow_gt_parts)
    all_flow_pred = np.concatenate(flow_pred_parts)
    all_composite = np.concatenate(composite_parts)
    all_mse_gt = np.concatenate(mse_gt_parts)
    all_mae_gt = np.concatenate(mae_gt_parts)
    all_mse_pred = np.concatenate(mse_pred_parts)
    all_mae_pred = np.concatenate(mae_pred_parts)
    all_mse_vla_gt = np.concatenate(mse_vla_gt_parts)
    all_wrist_gt = np.concatenate(wrist_gt_parts)
    all_wrist_vla = np.concatenate(wrist_vla_parts)
    all_wrist_vla_gt = np.concatenate(wrist_vla_gt_parts)
    all_change = np.concatenate(change_parts)
    all_gate = np.concatenate(gate_parts)
    all_gt_gain = all_mse_vla_gt - all_mse_gt
    all_relative_gt_error = all_mse_gt / np.maximum(all_mse_vla_gt, 1e-8)
    if target == "gt":
        primary_flow, primary_mse, primary_mae = (
            all_flow_gt,
            all_mse_gt,
            all_mae_gt,
        )
    else:
        primary_flow, primary_mse, primary_mae = (
            all_flow_pred,
            all_mse_pred,
            all_mae_pred,
        )

    wrist_metrics = {
        wrist: _bimanual_wrist_decode_metrics(
            all_wrist_gt[:, wrist_index],
            all_wrist_vla[:, wrist_index],
            all_wrist_vla_gt[:, wrist_index],
            all_gate[:, wrist_index],
            low_w_threshold=low_threshold,
            high_w_threshold=high_threshold,
            ranking_margin=rank_margin,
            repair_margin=repair_margin,
            low_safety_margin=low_gate_safety_margin,
        )
        for wrist_index, wrist in enumerate(("left", "right"))
    }
    quadrant_metrics = bimanual_quadrant_metrics(
        mse_gt=all_wrist_gt,
        mse_vla=all_wrist_vla,
        mse_vla_gt=all_wrist_vla_gt,
        gate_weights=all_gate,
        low_threshold=low_threshold,
        high_threshold=high_threshold,
        ranking_margin=rank_margin,
    )
    joint_counts = bimanual_gate_region_counts(
        all_gate, low_threshold=low_threshold, high_threshold=high_threshold
    )
    gate_quantiles = (_quantiles(all_gate[:, 0]), _quantiles(all_gate[:, 1]))
    change_quantiles = (
        _quantiles(all_change[:, 0]),
        _quantiles(all_change[:, 1]),
    )

    result = EvaluationResult(
        target=target,
        flow_loss=float(np.mean(primary_flow)),
        mse=float(np.mean(primary_mse)),
        rmse=float(np.sqrt(np.mean(primary_mse))),
        mae=float(np.mean(primary_mae)),
        flow_loss_gt=float(np.mean(all_flow_gt)),
        mse_gt=float(np.mean(all_mse_gt)),
        rmse_gt=float(np.sqrt(np.mean(all_mse_gt))),
        mae_gt=float(np.mean(all_mae_gt)),
        flow_loss_pred=float(np.mean(all_flow_pred)),
        mse_pred=float(np.mean(all_mse_pred)),
        rmse_pred=float(np.sqrt(np.mean(all_mse_pred))),
        mae_pred=float(np.mean(all_mae_pred)),
        cache_indices=all_indices,
        sample_flow_loss=primary_flow,
        sample_mse=primary_mse,
        sample_rmse=np.sqrt(primary_mse),
        sample_mae=primary_mae,
        sample_mse_gt=all_mse_gt,
        sample_mae_gt=all_mae_gt,
        sample_mse_pred=all_mse_pred,
        sample_mae_pred=all_mae_pred,
        predictions=(
            np.concatenate(prediction_parts) if prediction_parts else None
        ),
        tactile_change=float(np.mean(all_change)),
        tactile_sim=float(np.mean(1.0 - all_change)),
        gate_w=float(np.mean(all_gate)),
        gate_active_frac=float(np.mean(all_gate > 0.5)),
        mse_vla_gt=float(np.mean(all_mse_vla_gt)),
        gt_gain=float(np.mean(all_gt_gain)),
        relative_gt_error=_ratio_of_means(all_mse_gt, all_mse_vla_gt),
        sample_mse_vla_gt=all_mse_vla_gt,
        sample_gt_gain=all_gt_gain,
        sample_relative_gt_error=all_relative_gt_error,
        sample_gate_w_left=all_gate[:, 0],
        sample_gate_w_right=all_gate[:, 1],
        composite_fm=float(np.mean(all_composite)),
        sample_composite_fm=all_composite,
        tactile_change_left=float(np.mean(all_change[:, 0])),
        tactile_change_right=float(np.mean(all_change[:, 1])),
        sample_tactile_change_left=all_change[:, 0],
        sample_tactile_change_right=all_change[:, 1],
        gate_w_left=float(np.mean(all_gate[:, 0])),
        gate_w_right=float(np.mean(all_gate[:, 1])),
        sample_mse_gt_left=all_wrist_gt[:, 0],
        sample_mse_gt_right=all_wrist_gt[:, 1],
        sample_mse_vla_left=all_wrist_vla[:, 0],
        sample_mse_vla_right=all_wrist_vla[:, 1],
        sample_mse_vla_gt_left=all_wrist_vla_gt[:, 0],
        sample_mse_vla_gt_right=all_wrist_vla_gt[:, 1],
        bimanual_quadrants=quadrant_metrics,
        bimanual_gate_region_counts=joint_counts,
        gate_low_threshold=low_threshold,
        gate_high_threshold=high_threshold,
        gt_actions=np.concatenate(gt_action_parts) if gt_action_parts else None,
        vla_actions=np.concatenate(vla_action_parts) if vla_action_parts else None,
    )
    wrist_updates: dict[str, float | int] = {}
    for wrist_index, wrist in enumerate(("left", "right")):
        for quantile_index, quantile_name in enumerate(
            ("p10", "p25", "p50", "p75", "p90")
        ):
            wrist_updates[f"gate_w_{quantile_name}_{wrist}"] = gate_quantiles[
                wrist_index
            ][quantile_index]
            wrist_updates[
                f"tactile_change_{quantile_name}_{wrist}"
            ] = change_quantiles[wrist_index][quantile_index]
        for metric_name, value in wrist_metrics[wrist].items():
            wrist_updates[f"{metric_name}_{wrist}"] = value
    return dataclasses.replace(result, **wrist_updates)
