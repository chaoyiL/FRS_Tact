from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np

from tactile_flow_steering.utils.data import TactileConditionedBatches
from tactile_flow_steering.utils.model import FlowSolver
from tactile_flow_steering.utils.model import TactileConditionedFlowDecoder
from tactile_flow_steering.utils.model import decode_actions
from tactile_flow_steering.utils.model import flow_matching_loss_per_sample


@dataclasses.dataclass(frozen=True)
class EvaluationResult:
    flow_loss: float
    mse: float
    rmse: float
    mae: float
    cache_indices: np.ndarray
    sample_flow_loss: np.ndarray
    sample_mse: np.ndarray
    sample_rmse: np.ndarray
    sample_mae: np.ndarray
    predictions: np.ndarray | None
    tactile_change: float | None = None
    tactile_sim: float | None = None
    gate_w: float | None = None
    gate_active_frac: float | None = None


def evaluate_split(
    model: TactileConditionedFlowDecoder,
    conditioner: TactileConditionedBatches,
    *,
    split: str,
    batch_size: int,
    num_steps: int,
    keep_predictions: bool,
    solver: FlowSolver = "euler",
    gate_tau: float | None = None,
    gate_temperature: float | None = None,
) -> EvaluationResult:
    from tactile_flow_steering.utils.data import gate_weights_from_change

    cache_indices: list[np.ndarray] = []
    flow_losses: list[np.ndarray] = []
    mses: list[np.ndarray] = []
    maes: list[np.ndarray] = []
    predictions: list[np.ndarray] = []
    tactile_changes: list[np.ndarray] = []
    gate_weights: list[np.ndarray] = []
    track_tactile = (
        gate_tau is not None
        and gate_temperature is not None
        and bool(conditioner.episode_baselines)
    )

    for indices, x_base_np, _predicted_np, gt_action_np, tactile_seq in conditioner.batches(
        split, batch_size=batch_size, shuffle=False, seed=0
    ):
        x_base = jnp.asarray(x_base_np)
        target = jnp.asarray(gt_action_np)
        t = jnp.full((len(indices),), 0.5, dtype=jnp.float32)
        flow_loss = flow_matching_loss_per_sample(model, x_base, target, t, tactile_seq)
        prediction = decode_actions(
            model, x_base, tactile_seq, num_steps=num_steps, solver=solver
        )
        difference = prediction - target
        mse = jnp.mean(jnp.square(difference), axis=(1, 2))
        mae = jnp.mean(jnp.abs(difference), axis=(1, 2))

        cache_indices.append(indices)
        flow_losses.append(np.asarray(jax.device_get(flow_loss)))
        mses.append(np.asarray(jax.device_get(mse)))
        maes.append(np.asarray(jax.device_get(mae)))
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
    all_flow = np.concatenate(flow_losses)
    all_mse = np.concatenate(mses)
    all_mae = np.concatenate(maes)
    all_rmse = np.sqrt(all_mse)
    if tactile_changes:
        all_change = np.concatenate(tactile_changes)
        all_gate = np.concatenate(gate_weights)
        tactile_change = float(np.mean(all_change))
        tactile_sim = float(np.mean(1.0 - all_change))
        gate_w_mean = float(np.mean(all_gate))
        gate_active_frac = float(np.mean(all_gate > 0.5))
    else:
        tactile_change = None
        tactile_sim = None
        gate_w_mean = None
        gate_active_frac = None
    return EvaluationResult(
        flow_loss=float(np.mean(all_flow)),
        mse=float(np.mean(all_mse)),
        rmse=float(np.sqrt(np.mean(all_mse))),
        mae=float(np.mean(all_mae)),
        cache_indices=all_indices,
        sample_flow_loss=all_flow,
        sample_mse=all_mse,
        sample_rmse=all_rmse,
        sample_mae=all_mae,
        predictions=np.concatenate(predictions) if keep_predictions else None,
        tactile_change=tactile_change,
        tactile_sim=tactile_sim,
        gate_w=gate_w_mean,
        gate_active_frac=gate_active_frac,
    )
