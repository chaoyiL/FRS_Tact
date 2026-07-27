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
    predictions: list[np.ndarray] = []
    tactile_changes: list[np.ndarray] = []
    gate_weights: list[np.ndarray] = []
    track_tactile = (
        gate_tau is not None
        and gate_temperature is not None
        and bool(conditioner.episode_baselines)
    )

    for indices, x_base_np, predicted_np, gt_action_np, tactile_seq in conditioner.batches(
        split, batch_size=batch_size, shuffle=False, seed=0
    ):
        x_base = jnp.asarray(x_base_np)
        gt_action = jnp.asarray(gt_action_np)
        predicted_action = jnp.asarray(predicted_np)
        t = jnp.full((len(indices),), 0.5, dtype=jnp.float32)
        flow_gt = flow_matching_loss_per_sample(model, x_base, gt_action, t, tactile_seq)
        flow_pred = flow_matching_loss_per_sample(model, x_base, predicted_action, t, tactile_seq)
        prediction = decode_actions(
            model, x_base, tactile_seq, num_steps=num_steps, solver=solver
        )
        mse_gt, mae_gt = _per_sample_errors(prediction, gt_action)
        mse_pred, mae_pred = _per_sample_errors(prediction, predicted_action)

        cache_indices.append(indices)
        flow_gt_parts.append(np.asarray(jax.device_get(flow_gt)))
        flow_pred_parts.append(np.asarray(jax.device_get(flow_pred)))
        mse_gt_parts.append(np.asarray(jax.device_get(mse_gt)))
        mae_gt_parts.append(np.asarray(jax.device_get(mae_gt)))
        mse_pred_parts.append(np.asarray(jax.device_get(mse_pred)))
        mae_pred_parts.append(np.asarray(jax.device_get(mae_pred)))
        if keep_predictions:
            predictions.append(np.asarray(jax.device_get(prediction), dtype=np.float32))
        if track_tactile:
            current_tokens = np.asarray(tactile_seq[:, -1, :, :], dtype=np.float32)
            change = conditioner.tactile_change_for_cache_indices(indices, current_tokens)
            gate_w = gate_weights_from_change(
                change, tau=float(gate_tau), temperature=float(gate_temperature)
            )
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
    )
