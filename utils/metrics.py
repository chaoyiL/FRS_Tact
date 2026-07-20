from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np

from utils.cache import CachedPairs
from utils.model import FlowSolver
from utils.model import SelfAttentionFlowDecoder
from utils.model import decode_actions
from utils.model import flow_matching_loss_per_sample


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


def evaluate_split(
    model: SelfAttentionFlowDecoder,
    pairs: CachedPairs,
    *,
    split: str,
    batch_size: int,
    num_steps: int,
    keep_predictions: bool,
    solver: FlowSolver = "euler",
) -> EvaluationResult:
    cache_indices: list[np.ndarray] = []
    flow_losses: list[np.ndarray] = []
    mses: list[np.ndarray] = []
    maes: list[np.ndarray] = []
    predictions: list[np.ndarray] = []

    for indices, x_base_np, target_np in pairs.batches(
        split, batch_size=batch_size, shuffle=False, seed=0
    ):
        x_base = jnp.asarray(x_base_np)
        target = jnp.asarray(target_np)
        t = jnp.full((len(indices),), 0.5, dtype=jnp.float32)
        flow_loss = flow_matching_loss_per_sample(model, x_base, target, t)
        prediction = decode_actions(model, x_base, num_steps=num_steps, solver=solver)
        difference = prediction - target
        mse = jnp.mean(jnp.square(difference), axis=(1, 2))
        mae = jnp.mean(jnp.abs(difference), axis=(1, 2))

        cache_indices.append(indices)
        flow_losses.append(np.asarray(jax.device_get(flow_loss)))
        mses.append(np.asarray(jax.device_get(mse)))
        maes.append(np.asarray(jax.device_get(mae)))
        if keep_predictions:
            predictions.append(np.asarray(jax.device_get(prediction), dtype=np.float32))

    if not cache_indices:
        raise ValueError(f"No samples found for split {split!r}.")
    all_indices = np.concatenate(cache_indices)
    all_flow = np.concatenate(flow_losses)
    all_mse = np.concatenate(mses)
    all_mae = np.concatenate(maes)
    all_rmse = np.sqrt(all_mse)
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
    )
