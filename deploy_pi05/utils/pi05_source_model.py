"""pi0.5 analogue of utils/source_model.py: build a prefix cache once per observation batch, then
sample forward (t:1->0, like `Pi0.sample_actions`) and reverse-integrate (t:0->1, for FRS's
action_cache) through the same flow-matching velocity field.

Kept separate from utils/source_model.py (SmolVLA) rather than branching that file, because
importing it would pull in `modalities_eval.utils`/`lerobot.policies.smolvla_jax` -- exactly the
coupling pi0.5's own code avoids (see src/lerobot/policies/pi05_jax/README.md). The two share
only the base-model-agnostic pieces: `utils/integration.py`'s reverse ODE solvers and
`utils/flow_matching.py`'s deterministic_noise/inversion_mse.

UNTESTED, like the rest of pi05_jax (see its README.md): the caching/jit pattern below follows
`lerobot.policies.pi05_jax.nnx_utils.module_jit`'s documented technique (split state outside jit,
merge inside), but has not been run.
"""

from __future__ import annotations

from typing import Any, Literal

import flax.nnx as nnx
import jax
import jax.numpy as jnp

from lerobot.policies.pi05_jax import Observation, Pi0, frs as _frs
from lerobot.policies.pi05_jax.frs import Pi0PrefixCache
from utils.integration import (
    euler_integrate_velocity,
    fireflow_integrate_velocity,
    slerpflow_integrate_velocity,
)

ReverseSolver = Literal["euler", "fireflow", "slerpflow"]

# Keyed by `_cache_key`; values are jitted functions taking `state` as their first argument.
# Deliberately hold no reference to any `Pi0` (nor to its weights) -- see `_cache_key`.
_PREFIX_CACHE: dict[tuple, Any] = {}
_SAMPLE_CACHE: dict[tuple, Any] = {}
_REVERSE_CACHE: dict[tuple, Any] = {}


def _cache_key(model: Pi0, *extra: Any) -> tuple:
    """Cache key for the jitted wrappers below.

    `id(model)` alone would NOT be safe. `tools/prepare_frs_pi05_cache.py` calls
    `prepare_pi05.prepare_cache()` once per dataset, and each call builds its own `Pi0` and drops
    it on return -- so CPython can hand the next model the freed one's address, making `id()`
    collide across distinct models. (`utils/source_model.py` gets away with a bare `id()` because
    its jitted functions take `params` as a runtime argument, so a collision only reuses a
    compiled trace, never stale weights.)

    Two things together make collisions harmless here:
      * the architecture fields below, so two *differently shaped* models can't share an entry;
      * `state` staying a runtime argument (never closed over), so the weights always come from
        the caller. A collision between identically shaped models then just reuses an
        interchangeable `graphdef` plus an already-compiled trace.

    Cache values must therefore never capture `model` or its weights -- that would both resurrect
    the staleness hazard and pin every dataset's model in (GPU) memory for the whole run.
    """
    return (
        id(model),
        model.action_dim,
        model.action_horizon,
        model.max_token_len,
        model.pi05,
        tuple(model.image_keys),
        *extra,
    )


def _jitted_prefix(model: Pi0):
    cache_key = _cache_key(model)
    jitted = _PREFIX_CACHE.get(cache_key)
    if jitted is None:
        graphdef, _ = nnx.split(model)

        @jax.jit
        def jitted(state: nnx.State, observation: Observation) -> Pi0PrefixCache:
            return _frs.build_prefix_cache(nnx.merge(graphdef, state), observation)

        _PREFIX_CACHE[cache_key] = jitted
    return jitted


def _jitted_sample(model: Pi0, *, num_steps: int):
    cache_key = _cache_key(model, num_steps)
    jitted = _SAMPLE_CACHE.get(cache_key)
    if jitted is None:
        graphdef, _ = nnx.split(model)

        @jax.jit
        def jitted(state: nnx.State, cache: Pi0PrefixCache, noise: jax.Array) -> jax.Array:
            merged = nnx.merge(graphdef, state)
            dt = -1.0 / num_steps
            batch = noise.shape[0]

            def body(step: jax.Array, x_t: jax.Array) -> jax.Array:
                t = jnp.full((batch,), 1.0 + step.astype(jnp.float32) * dt, dtype=jnp.float32)
                return x_t + dt * _frs.denoise_step(merged, cache, x_t, t)

            return jax.lax.fori_loop(0, num_steps, body, noise)

        _SAMPLE_CACHE[cache_key] = jitted
    return jitted


def _jitted_reverse(model: Pi0, *, num_steps: int, solver: ReverseSolver):
    cache_key = _cache_key(model, num_steps, solver)
    jitted = _REVERSE_CACHE.get(cache_key)
    if jitted is None:
        if solver == "euler":
            integrate = euler_integrate_velocity
        elif solver == "fireflow":
            integrate = fireflow_integrate_velocity
        else:
            integrate = slerpflow_integrate_velocity
        graphdef, _ = nnx.split(model)

        @jax.jit
        def jitted(state: nnx.State, cache: Pi0PrefixCache, actions: jax.Array) -> jax.Array:
            merged = nnx.merge(graphdef, state)

            def velocity_fn(x: jax.Array, t: jax.Array) -> jax.Array:
                return _frs.denoise_step(merged, cache, x, t)

            return integrate(velocity_fn, jnp.asarray(actions, dtype=jnp.float32), num_steps=num_steps)

        _REVERSE_CACHE[cache_key] = jitted
    return jitted


def build_prefix_cache(model: Pi0, observation: Observation) -> Pi0PrefixCache:
    """JIT-compiled `frs.build_prefix_cache` (one image/language encode per observation batch)."""
    _, state = nnx.split(model)
    return _jitted_prefix(model)(state, observation)


def sample_and_reverse(
    model: Pi0,
    observation: Observation,
    noise: jax.Array,
    *,
    sample_steps: int,
    reverse_steps: int,
    solver: ReverseSolver = "fireflow",
) -> tuple[jax.Array, jax.Array]:
    """One shared prefix encode, then sample t:1->0 and reverse t:0->1. Mirrors
    utils/source_model.py:sample_and_reverse for SmolVLA."""
    if sample_steps <= 0 or reverse_steps <= 0:
        raise ValueError("sample_steps and reverse_steps must be positive.")
    if solver not in ("euler", "fireflow", "slerpflow"):
        raise ValueError(
            f"solver must be 'euler', 'fireflow', or 'slerpflow', got {solver!r}."
        )

    # Split once and reuse for all three calls -- `nnx.split` walks the whole module graph, so
    # doing it per call would repeat that work three times per batch for no benefit.
    _, state = nnx.split(model)
    cache = _jitted_prefix(model)(state, observation)
    predicted = _jitted_sample(model, num_steps=sample_steps)(state, cache, noise)
    x_base = _jitted_reverse(model, num_steps=reverse_steps, solver=solver)(state, cache, predicted)
    return predicted, x_base


def reverse_integrate_actions(
    model: Pi0,
    observation: Observation,
    actions: jax.Array,
    *,
    num_steps: int,
    solver: ReverseSolver = "euler",
) -> jax.Array:
    """Integrate model-space actions from data time t=0 to base noise time t=1."""
    if solver not in ("euler", "fireflow", "slerpflow"):
        raise ValueError(
            f"solver must be 'euler', 'fireflow', or 'slerpflow', got {solver!r}."
        )
    _, state = nnx.split(model)
    cache = _jitted_prefix(model)(state, observation)
    return _jitted_reverse(model, num_steps=num_steps, solver=solver)(
        state, cache, jnp.asarray(actions, dtype=jnp.float32)
    )
