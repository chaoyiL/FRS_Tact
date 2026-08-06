from __future__ import annotations

import dataclasses
from typing import Literal

import jax
import jax.numpy as jnp
import numpy as np

from tactile_flow_steering.utils.data import TactileConditionedBatches
from tactile_flow_steering.utils.model import FlowSolver
from tactile_flow_steering.utils.model import TactileConditionedFlowDecoder
from tactile_flow_steering.utils.model import decode_actions
from tactile_flow_steering.utils.model import flow_matching_loss_per_sample

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
    # Frozen source-policy baseline and FRS improvement against that baseline.
    mse_vla_gt: float
    gt_gain: float
    relative_gt_error: float
    cache_indices: np.ndarray
    sample_flow_loss: np.ndarray
    sample_mse: np.ndarray
    sample_rmse: np.ndarray
    sample_mae: np.ndarray
    sample_mse_gt: np.ndarray
    sample_mae_gt: np.ndarray
    sample_mse_pred: np.ndarray
    sample_mae_pred: np.ndarray
    sample_mse_vla_gt: np.ndarray
    sample_gt_gain: np.ndarray
    sample_relative_gt_error: np.ndarray
    predictions: np.ndarray | None
    tactile_change: float | None = None
    tactile_sim: float | None = None
    gate_w: float | None = None
    gate_active_frac: float | None = None
    sample_tactile_change: np.ndarray | None = None
    sample_gate_w: np.ndarray | None = None
    # Gate-stratified decode metrics (filled when tactile gate is tracked).
    mse_gt_high_w: float | None = None
    mse_gt_low_w: float | None = None
    mse_pred_high_w: float | None = None
    mse_pred_low_w: float | None = None
    mse_vla_gt_high_w: float | None = None
    mse_vla_gt_low_w: float | None = None
    gt_gain_high_w: float | None = None
    gt_gain_low_w: float | None = None
    relative_gt_error_high_w: float | None = None
    relative_gt_error_low_w: float | None = None
    gate_w_high_mean: float | None = None
    gate_w_low_mean: float | None = None
    tactile_change_high_mean: float | None = None
    tactile_change_low_mean: float | None = None
    n_high_w: int | None = None
    n_low_w: int | None = None
    gate_w_p10: float | None = None
    gate_w_p25: float | None = None
    gate_w_p50: float | None = None
    gate_w_p75: float | None = None
    gate_w_p90: float | None = None
    tactile_change_p10: float | None = None
    tactile_change_p25: float | None = None
    tactile_change_p50: float | None = None
    tactile_change_p75: float | None = None
    tactile_change_p90: float | None = None


def _per_sample_errors(prediction: jax.Array, reference: jax.Array) -> tuple[jax.Array, jax.Array]:
    difference = prediction - reference
    mse = jnp.mean(jnp.square(difference), axis=(1, 2))
    mae = jnp.mean(jnp.abs(difference), axis=(1, 2))
    return mse, mae


def _mean_or_nan(values: np.ndarray) -> float:
    if values.size == 0:
        return float("nan")
    return float(np.mean(values))


def _ratio_of_means(numerator: np.ndarray, denominator: np.ndarray) -> float:
    if numerator.size == 0 or denominator.size == 0:
        return float("nan")
    return float(np.mean(numerator) / max(float(np.mean(denominator)), 1e-8))


def _quantiles(values: np.ndarray) -> tuple[float, float, float, float, float]:
    levels = [0.1, 0.25, 0.5, 0.75, 0.9]
    return tuple(
        float(value)
        for value in np.quantile(np.asarray(values, dtype=np.float64), levels)
    )  # type: ignore[return-value]


def gate_stratified_decode_metrics(
    sample_mse_gt: np.ndarray,
    sample_mse_pred: np.ndarray,
    sample_mse_vla_gt: np.ndarray,
    gate_weights: np.ndarray,
    tactile_changes: np.ndarray | None = None,
    *,
    high_w_threshold: float = 0.5,
) -> dict[str, float | int]:
    """Split decode MSE by gate weight ``w > threshold`` vs ``w <= threshold``."""

    mse_gt = np.asarray(sample_mse_gt, dtype=np.float64)
    mse_pred = np.asarray(sample_mse_pred, dtype=np.float64)
    mse_vla_gt = np.asarray(sample_mse_vla_gt, dtype=np.float64)
    weights = np.asarray(gate_weights, dtype=np.float64)
    if not (mse_gt.shape == mse_pred.shape == mse_vla_gt.shape == weights.shape):
        raise ValueError(
            f"Shape mismatch: mse_gt={mse_gt.shape}, mse_pred={mse_pred.shape}, "
            f"mse_vla_gt={mse_vla_gt.shape}, gate_w={weights.shape}."
        )
    changes = None if tactile_changes is None else np.asarray(tactile_changes, dtype=np.float64)
    if changes is not None and changes.shape != weights.shape:
        raise ValueError(
            f"Shape mismatch: tactile_change={changes.shape}, gate_w={weights.shape}."
        )
    high = weights > float(high_w_threshold)
    low = ~high
    result: dict[str, float | int] = {
        "mse_gt_high_w": _mean_or_nan(mse_gt[high]),
        "mse_gt_low_w": _mean_or_nan(mse_gt[low]),
        "mse_pred_high_w": _mean_or_nan(mse_pred[high]),
        "mse_pred_low_w": _mean_or_nan(mse_pred[low]),
        "mse_vla_gt_high_w": _mean_or_nan(mse_vla_gt[high]),
        "mse_vla_gt_low_w": _mean_or_nan(mse_vla_gt[low]),
        "gt_gain_high_w": _mean_or_nan((mse_vla_gt - mse_gt)[high]),
        "gt_gain_low_w": _mean_or_nan((mse_vla_gt - mse_gt)[low]),
        "relative_gt_error_high_w": _ratio_of_means(mse_gt[high], mse_vla_gt[high]),
        "relative_gt_error_low_w": _ratio_of_means(mse_gt[low], mse_vla_gt[low]),
        "gate_w_high_mean": _mean_or_nan(weights[high]),
        "gate_w_low_mean": _mean_or_nan(weights[low]),
        "n_high_w": int(np.count_nonzero(high)),
        "n_low_w": int(np.count_nonzero(low)),
    }
    if changes is not None:
        result.update(
            {
                "tactile_change_high_mean": _mean_or_nan(changes[high]),
                "tactile_change_low_mean": _mean_or_nan(changes[low]),
            }
        )
    else:
        result.update(
            {
                "tactile_change_high_mean": float("nan"),
                "tactile_change_low_mean": float("nan"),
            }
        )
    return result


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
) -> EvaluationResult:
    from tactile_flow_steering.utils.data import gate_weights_from_change

    if target not in ("gt", "predicted"):
        raise ValueError(f"target must be 'gt' or 'predicted', got {target!r}.")

    cache_indices: list[np.ndarray] = []
    flow_gt_parts: list[np.ndarray] = []
    flow_pred_parts: list[np.ndarray] = []
    mse_gt_parts: list[np.ndarray] = []
    mae_gt_parts: list[np.ndarray] = []
    mse_pred_parts: list[np.ndarray] = []
    mae_pred_parts: list[np.ndarray] = []
    mse_vla_gt_parts: list[np.ndarray] = []
    predictions: list[np.ndarray] = []
    tactile_changes: list[np.ndarray] = []
    gate_weights: list[np.ndarray] = []
    track_tactile = (
        gate_tau is not None
        and gate_temperature is not None
        and bool(conditioner.episode_baselines)
    )
    if model.config.gate_conditioning and not track_tactile:
        raise ValueError(
            "Gate-conditioned checkpoint evaluation requires gate_tau, gate_temperature, "
            "and episode baseline embeddings."
        )

    for indices, x_base_np, predicted_np, gt_action_np, tactile_seq in conditioner.batches(
        split, batch_size=batch_size, shuffle=False, seed=0
    ):
        x_base = jnp.asarray(x_base_np)
        gt_action = jnp.asarray(gt_action_np)
        predicted_action = jnp.asarray(predicted_np)
        t = jnp.full((len(indices),), 0.5, dtype=jnp.float32)
        if track_tactile:
            current_tokens = np.asarray(tactile_seq[:, -1, :, :], dtype=np.float32)
            change = conditioner.tactile_change_for_cache_indices(indices, current_tokens)
            gate_w = gate_weights_from_change(
                change, tau=float(gate_tau), temperature=float(gate_temperature)
            )
            model_gate = jnp.asarray(gate_w)
        else:
            change = None
            gate_w = None
            model_gate = None
        flow_gt = flow_matching_loss_per_sample(
            model, x_base, gt_action, t, tactile_seq, model_gate
        )
        flow_pred = flow_matching_loss_per_sample(
            model, x_base, predicted_action, t, tactile_seq, model_gate
        )
        prediction = decode_actions(
            model,
            x_base,
            tactile_seq,
            model_gate,
            num_steps=num_steps,
            solver=solver,
        )
        mse_gt, mae_gt = _per_sample_errors(prediction, gt_action)
        mse_pred, mae_pred = _per_sample_errors(prediction, predicted_action)
        mse_vla_gt, _ = _per_sample_errors(predicted_action, gt_action)

        cache_indices.append(indices)
        flow_gt_parts.append(np.asarray(jax.device_get(flow_gt)))
        flow_pred_parts.append(np.asarray(jax.device_get(flow_pred)))
        mse_gt_parts.append(np.asarray(jax.device_get(mse_gt)))
        mae_gt_parts.append(np.asarray(jax.device_get(mae_gt)))
        mse_pred_parts.append(np.asarray(jax.device_get(mse_pred)))
        mae_pred_parts.append(np.asarray(jax.device_get(mae_pred)))
        mse_vla_gt_parts.append(np.asarray(jax.device_get(mse_vla_gt)))
        if keep_predictions:
            predictions.append(np.asarray(jax.device_get(prediction), dtype=np.float32))
        if change is not None and gate_w is not None:
            tactile_changes.append(change)
            gate_weights.append(gate_w)

    if not cache_indices:
        raise ValueError(f"No samples found for split {split!r}.")

    all_indices = np.concatenate(cache_indices)
    all_flow_gt = np.concatenate(flow_gt_parts)
    all_flow_pred = np.concatenate(flow_pred_parts)
    all_mse_gt = np.concatenate(mse_gt_parts)
    all_mae_gt = np.concatenate(mae_gt_parts)
    all_mse_pred = np.concatenate(mse_pred_parts)
    all_mae_pred = np.concatenate(mae_pred_parts)
    all_mse_vla_gt = np.concatenate(mse_vla_gt_parts)
    all_gt_gain = all_mse_vla_gt - all_mse_gt
    all_relative_gt_error = all_mse_gt / np.maximum(all_mse_vla_gt, 1e-8)
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
        stratified = gate_stratified_decode_metrics(
            all_mse_gt,
            all_mse_pred,
            all_mse_vla_gt,
            all_gate,
            all_change,
        )
        gate_quantiles = _quantiles(all_gate)
        change_quantiles = _quantiles(all_change)
    else:
        all_change = None
        all_gate = None
        tactile_change = None
        tactile_sim = None
        gate_w_mean = None
        gate_active_frac = None
        stratified = {
            "mse_gt_high_w": None,
            "mse_gt_low_w": None,
            "mse_pred_high_w": None,
            "mse_pred_low_w": None,
            "mse_vla_gt_high_w": None,
            "mse_vla_gt_low_w": None,
            "gt_gain_high_w": None,
            "gt_gain_low_w": None,
            "relative_gt_error_high_w": None,
            "relative_gt_error_low_w": None,
            "gate_w_high_mean": None,
            "gate_w_low_mean": None,
            "tactile_change_high_mean": None,
            "tactile_change_low_mean": None,
            "n_high_w": None,
            "n_low_w": None,
        }
        gate_quantiles = (None, None, None, None, None)
        change_quantiles = (None, None, None, None, None)

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
        mse_vla_gt=float(np.mean(all_mse_vla_gt)),
        gt_gain=float(np.mean(all_gt_gain)),
        relative_gt_error=_ratio_of_means(all_mse_gt, all_mse_vla_gt),
        cache_indices=all_indices,
        sample_flow_loss=primary_flow,
        sample_mse=primary_mse,
        sample_rmse=primary_rmse,
        sample_mae=primary_mae,
        sample_mse_gt=all_mse_gt,
        sample_mae_gt=all_mae_gt,
        sample_mse_pred=all_mse_pred,
        sample_mae_pred=all_mae_pred,
        sample_mse_vla_gt=all_mse_vla_gt,
        sample_gt_gain=all_gt_gain,
        sample_relative_gt_error=all_relative_gt_error,
        predictions=np.concatenate(predictions) if keep_predictions else None,
        tactile_change=tactile_change,
        tactile_sim=tactile_sim,
        gate_w=gate_w_mean,
        gate_active_frac=gate_active_frac,
        sample_tactile_change=all_change,
        sample_gate_w=all_gate,
        mse_gt_high_w=stratified["mse_gt_high_w"],  # type: ignore[arg-type]
        mse_gt_low_w=stratified["mse_gt_low_w"],  # type: ignore[arg-type]
        mse_pred_high_w=stratified["mse_pred_high_w"],  # type: ignore[arg-type]
        mse_pred_low_w=stratified["mse_pred_low_w"],  # type: ignore[arg-type]
        mse_vla_gt_high_w=stratified["mse_vla_gt_high_w"],  # type: ignore[arg-type]
        mse_vla_gt_low_w=stratified["mse_vla_gt_low_w"],  # type: ignore[arg-type]
        gt_gain_high_w=stratified["gt_gain_high_w"],  # type: ignore[arg-type]
        gt_gain_low_w=stratified["gt_gain_low_w"],  # type: ignore[arg-type]
        relative_gt_error_high_w=stratified["relative_gt_error_high_w"],  # type: ignore[arg-type]
        relative_gt_error_low_w=stratified["relative_gt_error_low_w"],  # type: ignore[arg-type]
        gate_w_high_mean=stratified["gate_w_high_mean"],  # type: ignore[arg-type]
        gate_w_low_mean=stratified["gate_w_low_mean"],  # type: ignore[arg-type]
        tactile_change_high_mean=stratified["tactile_change_high_mean"],  # type: ignore[arg-type]
        tactile_change_low_mean=stratified["tactile_change_low_mean"],  # type: ignore[arg-type]
        n_high_w=stratified["n_high_w"],  # type: ignore[arg-type]
        n_low_w=stratified["n_low_w"],  # type: ignore[arg-type]
        gate_w_p10=gate_quantiles[0],
        gate_w_p25=gate_quantiles[1],
        gate_w_p50=gate_quantiles[2],
        gate_w_p75=gate_quantiles[3],
        gate_w_p90=gate_quantiles[4],
        tactile_change_p10=change_quantiles[0],
        tactile_change_p25=change_quantiles[1],
        tactile_change_p50=change_quantiles[2],
        tactile_change_p75=change_quantiles[3],
        tactile_change_p90=change_quantiles[4],
    )
