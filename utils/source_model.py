from __future__ import annotations

from collections.abc import Sequence
from typing import Any
from typing import Literal

import jax
import jax.numpy as jnp
import numpy as np

from eval_scripts.utils import EvalObservation
from eval_scripts.utils import SmolVLAEvalModel
from eval_scripts.utils import VelocityContext
from eval_scripts.utils import _stack_observations
from lerobot.policies.smolvla_jax.modeling import PrefixContext
from utils.integration import euler_integrate_velocity
from utils.integration import fireflow_integrate_velocity


def stack_observations(observations: Sequence[EvalObservation]) -> EvalObservation:
    if not observations:
        raise ValueError("Cannot stack an empty observation batch.")
    return _stack_observations(*observations)


_PREFIX_CACHE: dict[int, Any] = {}
_SAMPLE_CACHE: dict[tuple[int, int], Any] = {}
_REVERSE_CACHE: dict[tuple[int, int, str], Any] = {}


def _pad_actions_to_model(model: SmolVLAEvalModel, actions: jax.Array) -> jax.Array:
    actions = jnp.asarray(actions, dtype=jnp.float32)
    pad = model.config.max_action_dim - actions.shape[-1]
    if pad > 0:
        actions = jnp.pad(actions, ((0, 0), (0, 0), (0, pad)))
    return actions


def _jitted_prefix_builder(model: SmolVLAEvalModel):
    cache_key = id(model)
    run = _PREFIX_CACHE.get(cache_key)
    if run is not None:
        return run

    functional_model = model.model

    @jax.jit
    def run(
        params,
        images: jax.Array,
        image_masks: jax.Array,
        language_tokens: jax.Array,
        language_masks: jax.Array,
        state: jax.Array,
    ) -> VelocityContext:
        prefix = functional_model.build_prefix_context(
            params,
            images,
            image_masks,
            language_tokens,
            language_masks,
            state,
        )
        return VelocityContext(pad_mask=prefix.pad_mask, cache=prefix.cache)

    _PREFIX_CACHE[cache_key] = run
    return run


def build_velocity_context(model: SmolVLAEvalModel, observation: EvalObservation) -> VelocityContext:
    """JIT-compiled prefix encode for the prepare hot path."""
    return _jitted_prefix_builder(model)(
        model.params,
        observation.images,
        observation.image_masks,
        observation.language_tokens,
        observation.language_masks,
        observation.state,
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


def _jitted_reverse_from_context(
    model: SmolVLAEvalModel,
    *,
    num_steps: int,
    solver: Literal["euler", "fireflow"],
):
    cache_key = (id(model), num_steps, solver)
    run = _REVERSE_CACHE.get(cache_key)
    if run is not None:
        return run

    integrate = euler_integrate_velocity if solver == "euler" else fireflow_integrate_velocity
    functional_model = model.model
    max_action_dim = int(model.config.max_action_dim)

    @jax.jit
    def run(params, context: VelocityContext, actions: jax.Array) -> jax.Array:
        actions = jnp.asarray(actions, dtype=jnp.float32)

        def velocity_fn(x: jax.Array, t: jax.Array) -> jax.Array:
            x_in = x
            pad = max_action_dim - x.shape[-1]
            if pad > 0:
                x_in = jnp.pad(x, ((0, 0), (0, 0), (0, pad)))
            t = jnp.asarray(t, dtype=jnp.float32)
            if t.ndim == 0:
                t = jnp.full((x.shape[0],), t)
            velocity = functional_model.denoise_step(
                params,
                PrefixContext(pad_mask=context.pad_mask, cache=context.cache),
                x_in,
                t,
            )
            return velocity[..., : x.shape[-1]].astype(jnp.float32)

        return integrate(velocity_fn, actions, num_steps=num_steps)

    _REVERSE_CACHE[cache_key] = run
    return run


def sample_and_reverse(
    model: SmolVLAEvalModel,
    observation: EvalObservation,
    noise: jax.Array,
    *,
    sample_steps: int,
    reverse_steps: int,
    solver: Literal["euler", "fireflow"] = "euler",
) -> tuple[jax.Array, jax.Array]:
    """One shared prefix encode, then sample t:1→0 and reverse t:0→1."""
    if sample_steps <= 0 or reverse_steps <= 0:
        raise ValueError("sample_steps and reverse_steps must be positive.")
    if solver not in ("euler", "fireflow"):
        raise ValueError(f"solver must be 'euler' or 'fireflow', got {solver!r}.")

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
    solver: Literal["euler", "fireflow"] = "euler",
) -> jax.Array:
    """Integrate model-space actions from data time t=0 to base noise time t=1."""
    if solver not in ("euler", "fireflow"):
        raise ValueError(f"solver must be 'euler' or 'fireflow', got {solver!r}.")
    context = build_velocity_context(model, observation)
    return _jitted_reverse_from_context(model, num_steps=num_steps, solver=solver)(
        model.params, context, jnp.asarray(actions, dtype=jnp.float32)
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
