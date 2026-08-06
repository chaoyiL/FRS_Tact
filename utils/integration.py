from __future__ import annotations

from collections.abc import Callable

import jax
import jax.numpy as jnp

REVERSE_INTEGRATION_VERSION = 1


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
        return jnp.sqrt(
            jnp.sum(jnp.square(value), axis=reduction_axes, keepdims=True) + 1e-8
        )

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
        next_velocity = velocity_fn(
            euler_state,
            _broadcast_time(next_time, batch_size),
        )
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
