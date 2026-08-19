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

from collections.abc import Callable, Sequence
from typing import Any, Literal

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np

from lerobot.policies.pi05_jax import Observation, Pi0, frs as _frs
from lerobot.policies.pi05_jax.frs import Pi0PrefixCache
ReverseSolver = Literal["euler", "fireflow", "slerpflow"]

def euler_integrate_velocity(
    velocity_fn: Callable[[jax.Array, jax.Array], jax.Array],
    x: jax.Array,
    *,
    num_steps: int,
) -> jax.Array:
    """Integrate dx/dt=velocity_fn(x,t) from t=0 to t=1 with forward Euler."""
    if num_steps <= 0:
        raise ValueError(f"num_steps must be positive, got {num_steps}.")
    x = jnp.asarray(x, dtype=jnp.float32)
    batch_size = x.shape[0]
    dt = jnp.asarray(1.0 / num_steps, dtype=jnp.float32)

    def body(carry: jax.Array, step: jax.Array):
        t = jnp.full((batch_size,), step.astype(jnp.float32) * dt, dtype=jnp.float32)
        return carry + dt * velocity_fn(carry, t), None

    result, _ = jax.lax.scan(body, x, jnp.arange(num_steps, dtype=jnp.int32))
    return result


def _broadcast_time(t_scalar: jax.Array, batch_size: int) -> jax.Array:
    return jnp.full((batch_size,), t_scalar, dtype=jnp.float32)


def fireflow_integrate_velocity(
    velocity_fn: Callable[[jax.Array, jax.Array], jax.Array],
    x: jax.Array,
    *,
    num_steps: int,
    return_nfe: bool = False,
) -> jax.Array | tuple[jax.Array, jax.Array]:
    """Integrate dx/dt=velocity_fn(x,t) from t=0 to t=1 with FireFlow modified midpoint."""
    if num_steps <= 0:
        raise ValueError(f"num_steps must be positive, got {num_steps}.")
    x = jnp.asarray(x, dtype=jnp.float32)
    batch_size = x.shape[0]
    timesteps = jnp.linspace(0.0, 1.0, num_steps + 1, dtype=jnp.float32)
    dts = jnp.diff(timesteps)

    t0 = timesteps[0]
    dt0 = dts[0]
    t_mid0 = t0 + 0.5 * dt0

    v0 = velocity_fn(x, _broadcast_time(t0, batch_size))
    x_mid = x + 0.5 * dt0 * v0
    v_mid_prev = velocity_fn(x_mid, _broadcast_time(t_mid0, batch_size))
    x = x + dt0 * v_mid_prev
    nfe = jnp.asarray(2, dtype=jnp.int32)

    def body(carry: tuple[jax.Array, jax.Array, jax.Array], step: jax.Array):
        x_carry, v_mid_prev_carry, nfe_carry = carry
        t = timesteps[step]
        dt = dts[step]
        t_mid = t + 0.5 * dt
        x_mid_carry = x_carry + 0.5 * dt * v_mid_prev_carry
        v_mid = velocity_fn(x_mid_carry, _broadcast_time(t_mid, batch_size))
        x_carry = x_carry + dt * v_mid
        return (x_carry, v_mid, nfe_carry + 1), None

    if num_steps > 1:
        (x, _, nfe), _ = jax.lax.scan(
            body,
            (x, v_mid_prev, nfe),
            jnp.arange(1, num_steps, dtype=jnp.int32),
        )
    if return_nfe:
        return x, nfe
    return x


def slerpflow_integrate_velocity(
    velocity_fn: Callable[[jax.Array, jax.Array], jax.Array],
    x: jax.Array,
    *,
    num_steps: int,
    return_nfe: bool = False,
) -> jax.Array | tuple[jax.Array, jax.Array]:
    """Integrate ``dx/dt=velocity_fn(x,t)`` with SlerpFlow trajectory correction."""
    if num_steps <= 0:
        raise ValueError(f"num_steps must be positive, got {num_steps}.")

    x = jnp.asarray(x, dtype=jnp.float32)
    batch_size = x.shape[0]
    timesteps = jnp.linspace(0.0, 1.0, num_steps + 1, dtype=jnp.float32)
    dts = jnp.diff(timesteps)
    reduction_axes = tuple(range(1, x.ndim))

    def batch_dot(left: jax.Array, right: jax.Array) -> jax.Array:
        return jnp.sum(left * right, axis=reduction_axes, keepdims=True)

    def batch_norm(value: jax.Array) -> jax.Array:
        return jnp.sqrt(jnp.sum(jnp.square(value), axis=reduction_axes, keepdims=True) + 1e-8)

    def slerp(left: jax.Array, right: jax.Array, alpha: float = 0.5) -> jax.Array:
        dot = jnp.clip(batch_dot(left, right), -1.0 + 1e-7, 1.0 - 1e-7)
        omega = jnp.arccos(dot)
        lerp_result = (1.0 - alpha) * left + alpha * right

        sin_omega = jnp.sin(omega)
        sin_omega_safe = jnp.where(omega < 1e-6, 1e-6, sin_omega)
        slerp_result = (
            jnp.sin((1.0 - alpha) * omega) / sin_omega_safe
        ) * left + (jnp.sin(alpha * omega) / sin_omega_safe) * right
        direction = jnp.where(omega < 1e-6, lerp_result, slerp_result)
        return direction / batch_norm(direction)

    velocity = velocity_fn(x, _broadcast_time(timesteps[0], batch_size))
    nfe = jnp.asarray(1, dtype=jnp.int32)

    def body(carry: tuple[jax.Array, jax.Array, jax.Array], step: jax.Array):
        state, current_velocity, current_nfe = carry
        dt = dts[step]
        next_time = timesteps[step + 1]

        euler_state = state + dt * current_velocity
        next_velocity = velocity_fn(euler_state, _broadcast_time(next_time, batch_size))
        average_velocity = 0.5 * (current_velocity + next_velocity)

        radius = batch_norm(state)
        direction = state / radius
        next_radius = radius + dt * batch_dot(direction, average_velocity)

        average_state = state + dt * average_velocity
        euler_direction = average_state / batch_norm(average_state)
        next_direction = slerp(direction, euler_direction, alpha=0.5)
        corrected_velocity = (next_radius * next_direction - state) / dt
        next_state = state + dt * corrected_velocity

        return (next_state, next_velocity, current_nfe + 1), None

    (result, _, nfe), _ = jax.lax.scan(
        body,
        (x, velocity, nfe),
        jnp.arange(num_steps, dtype=jnp.int32),
    )
    if return_nfe:
        return result, nfe
    return result

def deterministic_noise(indices: Sequence[int], shape: tuple[int, int], *, seed: int) -> jax.Array:
    base_key = jax.random.key(seed)
    index_arr = jnp.asarray(list(indices), dtype=jnp.int32)

    def one(index: jax.Array) -> jax.Array:
        return jax.random.normal(jax.random.fold_in(base_key, index), shape, dtype=jnp.float32)

    return jax.vmap(one)(index_arr)


def inversion_mse(x_base: jax.Array, initial_noise: jax.Array) -> np.ndarray:
    axes = tuple(range(1, x_base.ndim))
    return np.asarray(jax.device_get(jnp.mean(jnp.square(x_base - initial_noise), axis=axes)), dtype=np.float32)

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
