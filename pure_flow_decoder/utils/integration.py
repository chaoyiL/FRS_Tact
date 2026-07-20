from __future__ import annotations

from collections.abc import Callable

import jax
import jax.numpy as jnp


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
