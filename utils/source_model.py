from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from modalities_eval.utils import EvalObservation
from modalities_eval.utils import SmolVLAEvalModel
from modalities_eval.utils import _stack_observations
from train_smolvla.modeling import PrefixContext
from utils.source_flow import ReverseSolver
from utils.source_flow import VelocityContext
from utils.source_flow import _jitted_reverse_from_context
from utils.source_flow import build_velocity_context_from_prepared
from utils.source_flow import reverse_integrate_prepared_actions


def stack_observations(observations: Sequence[EvalObservation]) -> EvalObservation:
    if not observations:
        raise ValueError("Cannot stack an empty observation batch.")
    return _stack_observations(*observations)


_SAMPLE_CACHE: dict[tuple[int, int], Any] = {}


def _pad_actions_to_model(model: SmolVLAEvalModel, actions: jax.Array) -> jax.Array:
    actions = jnp.asarray(actions, dtype=jnp.float32)
    pad = model.config.max_action_dim - actions.shape[-1]
    if pad > 0:
        actions = jnp.pad(actions, ((0, 0), (0, 0), (0, pad)))
    return actions


def build_velocity_context(model: SmolVLAEvalModel, observation: EvalObservation) -> VelocityContext:
    """JIT-compiled prefix encode for the prepare hot path."""
    return build_velocity_context_from_prepared(
        model,
        {
            "images": observation.images,
            "image_masks": observation.image_masks,
            "language_tokens": observation.language_tokens,
            "language_masks": observation.language_masks,
            "state": observation.state,
        },
    )


def _jitted_sample_from_context(model: SmolVLAEvalModel, *, num_steps: int):
    cache_key = (id(model), num_steps)
    run = _SAMPLE_CACHE.get(cache_key)
    if run is not None:
        return run

    functional_model = model.model
    action_dim = int(model.config.action_dim)

    @jax.jit
    def run(params, context: VelocityContext, noise: jax.Array) -> jax.Array:
        batch = noise.shape[0]
        dt = -1.0 / num_steps

        def body(step: int, x_t: jax.Array) -> jax.Array:
            time = 1.0 + step * dt
            timestep = jnp.full((batch,), time, dtype=jnp.float32)
            velocity = functional_model.denoise_step(
                params,
                PrefixContext(pad_mask=context.pad_mask, cache=context.cache),
                x_t,
                timestep,
            )
            return x_t + dt * velocity

        actions = jax.lax.fori_loop(0, num_steps, body, noise)
        return actions[..., :action_dim]

    _SAMPLE_CACHE[cache_key] = run
    return run


def sample_and_reverse(
    model: SmolVLAEvalModel,
    observation: EvalObservation,
    noise: jax.Array,
    *,
    sample_steps: int,
    reverse_steps: int,
    solver: ReverseSolver = "slerpflow",
) -> tuple[jax.Array, jax.Array]:
    """One shared prefix encode, then sample t:1→0 and reverse t:0→1."""
    if sample_steps <= 0 or reverse_steps <= 0:
        raise ValueError("sample_steps and reverse_steps must be positive.")
    if solver not in ("euler", "fireflow", "slerpflow"):
        raise ValueError(
            f"solver must be 'euler', 'fireflow', or 'slerpflow', got {solver!r}."
        )

    context = build_velocity_context(model, observation)
    padded_noise = _pad_actions_to_model(model, noise)
    predicted = _jitted_sample_from_context(model, num_steps=sample_steps)(
        model.params, context, padded_noise
    )
    x_base = _jitted_reverse_from_context(model, num_steps=reverse_steps, solver=solver)(
        model.params, context, predicted
    )
    return predicted, x_base


def reverse_integrate_actions(
    model: SmolVLAEvalModel,
    observation: EvalObservation,
    actions: jax.Array,
    *,
    num_steps: int,
    solver: ReverseSolver = "slerpflow",
) -> jax.Array:
    """Integrate model-space actions from data time t=0 to base noise time t=1."""
    batch = {
        "images": observation.images,
        "image_masks": observation.image_masks,
        "language_tokens": observation.language_tokens,
        "language_masks": observation.language_masks,
        "state": observation.state,
    }
    return reverse_integrate_prepared_actions(
        model,
        batch,
        actions,
        num_steps=num_steps,
        solver=solver,
    )


def deterministic_noise(indices: Sequence[int], shape: tuple[int, int], *, seed: int) -> jax.Array:
    base_key = jax.random.key(seed)
    index_arr = jnp.asarray(list(indices), dtype=jnp.int32)

    def one(index: jax.Array) -> jax.Array:
        return jax.random.normal(jax.random.fold_in(base_key, index), shape, dtype=jnp.float32)

    return jax.vmap(one)(index_arr)


def inversion_mse(x_base: jax.Array, initial_noise: jax.Array) -> np.ndarray:
    axes = tuple(range(1, x_base.ndim))
    return np.asarray(jax.device_get(jnp.mean(jnp.square(x_base - initial_noise), axis=axes)), dtype=np.float32)
