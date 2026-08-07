"""pi0.5 analogue of utils/source_model.py: build a prefix cache once per observation batch, then
sample forward (t:1->0, like `Pi0.sample_actions`) and reverse-integrate (t:0->1, for FRS's
action_cache) through the same flow-matching velocity field.

Kept separate from utils/source_model.py (SmolVLA) rather than branching that file, because
importing it would pull in `modalities_eval.utils`/`lerobot.policies.smolvla_jax` -- exactly the
coupling pi0.5's own code avoids (see src/lerobot/policies/pi05_jax/README.md). The two share
only the base-model-agnostic pieces: `utils/integration.py`'s euler/fireflow solvers and
`utils/flow_matching.py`'s deterministic_noise/inversion_mse.

UNTESTED, like the rest of pi05_jax (see its README.md): the caching/jit pattern below follows
`lerobot.policies.pi05_jax.nnx_utils.module_jit`'s documented technique (split state once outside
jit, merge once inside), but has not been run.
"""

from __future__ import annotations

from typing import Any, Literal

import flax.nnx as nnx
import jax
import jax.numpy as jnp

from lerobot.policies.pi05_jax import Observation, Pi0
from lerobot.policies.pi05_jax.pi0 import Pi0PrefixCache
from utils.integration import euler_integrate_velocity, fireflow_integrate_velocity

_PREFIX_CACHE: dict[int, Any] = {}
_SAMPLE_CACHE: dict[tuple[int, int], Any] = {}
_REVERSE_CACHE: dict[tuple[int, int, str], Any] = {}


def build_prefix_cache(model: Pi0, observation: Observation) -> Pi0PrefixCache:
    """JIT-compiled `Pi0.build_prefix_cache`, frozen to `model`'s current params.

    Follows `nnx_utils.module_jit`'s pattern manually (rather than calling it directly) so the
    cache key can be `id(model)` alone, matching `utils/source_model.py`'s
    `_jitted_prefix_builder` convention.
    """
    cache_key = id(model)
    run = _PREFIX_CACHE.get(cache_key)
    if run is None:
        graphdef, state = nnx.split(model)

        @jax.jit
        def jitted(state: nnx.State, observation: Observation) -> Pi0PrefixCache:
            return nnx.merge(graphdef, state).build_prefix_cache(observation)

        run = (jitted, state)
        _PREFIX_CACHE[cache_key] = run
    jitted, state = run
    return jitted(state, observation)


def _jitted_sample_from_cache(model: Pi0, *, num_steps: int):
    cache_key = (id(model), num_steps)
    run = _SAMPLE_CACHE.get(cache_key)
    if run is not None:
        return run
    graphdef, state = nnx.split(model)

    @jax.jit
    def jitted(state: nnx.State, cache: Pi0PrefixCache, noise: jax.Array) -> jax.Array:
        merged = nnx.merge(graphdef, state)
        dt = -1.0 / num_steps
        batch = noise.shape[0]

        def body(step: jax.Array, x_t: jax.Array) -> jax.Array:
            t = jnp.full((batch,), 1.0 + step.astype(jnp.float32) * dt, dtype=jnp.float32)
            return x_t + dt * merged.denoise_step(cache, x_t, t)

        return jax.lax.fori_loop(0, num_steps, body, noise)

    def run(cache: Pi0PrefixCache, noise: jax.Array) -> jax.Array:
        return jitted(state, cache, noise)

    _SAMPLE_CACHE[cache_key] = run
    return run


def _jitted_reverse_from_cache(model: Pi0, *, num_steps: int, solver: Literal["euler", "fireflow"]):
    cache_key = (id(model), num_steps, solver)
    run = _REVERSE_CACHE.get(cache_key)
    if run is not None:
        return run
    integrate = euler_integrate_velocity if solver == "euler" else fireflow_integrate_velocity
    graphdef, state = nnx.split(model)

    @jax.jit
    def jitted(state: nnx.State, cache: Pi0PrefixCache, actions: jax.Array) -> jax.Array:
        merged = nnx.merge(graphdef, state)

        def velocity_fn(x: jax.Array, t: jax.Array) -> jax.Array:
            return merged.denoise_step(cache, x, t)

        return integrate(velocity_fn, jnp.asarray(actions, dtype=jnp.float32), num_steps=num_steps)

    def run(cache: Pi0PrefixCache, actions: jax.Array) -> jax.Array:
        return jitted(state, cache, actions)

    _REVERSE_CACHE[cache_key] = run
    return run


def sample_and_reverse(
    model: Pi0,
    observation: Observation,
    noise: jax.Array,
    *,
    sample_steps: int,
    reverse_steps: int,
    solver: Literal["euler", "fireflow"] = "fireflow",
) -> tuple[jax.Array, jax.Array]:
    """One shared prefix encode, then sample t:1->0 and reverse t:0->1. Mirrors
    utils/source_model.py:sample_and_reverse for SmolVLA."""
    if sample_steps <= 0 or reverse_steps <= 0:
        raise ValueError("sample_steps and reverse_steps must be positive.")
    if solver not in ("euler", "fireflow"):
        raise ValueError(f"solver must be 'euler' or 'fireflow', got {solver!r}.")

    cache = build_prefix_cache(model, observation)
    predicted = _jitted_sample_from_cache(model, num_steps=sample_steps)(cache, noise)
    x_base = _jitted_reverse_from_cache(model, num_steps=reverse_steps, solver=solver)(cache, predicted)
    return predicted, x_base


def reverse_integrate_actions(
    model: Pi0,
    observation: Observation,
    actions: jax.Array,
    *,
    num_steps: int,
    solver: Literal["euler", "fireflow"] = "euler",
) -> jax.Array:
    """Integrate model-space actions from data time t=0 to base noise time t=1."""
    if solver not in ("euler", "fireflow"):
        raise ValueError(f"solver must be 'euler' or 'fireflow', got {solver!r}.")
    cache = build_prefix_cache(model, observation)
    return _jitted_reverse_from_cache(model, num_steps=num_steps, solver=solver)(
        cache, jnp.asarray(actions, dtype=jnp.float32)
    )
