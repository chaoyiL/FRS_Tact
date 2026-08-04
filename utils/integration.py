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

def slerpflow_integrate_velocity(
    velocity_fn: Callable[[jax.Array, jax.Array], jax.Array],
    x: jax.Array,
    *,
    num_steps: int,
    return_nfe: bool = False,
) -> jax.Array | tuple[jax.Array, jax.Array]:
    """
    Integrate dx/dt=velocity_fn(x,t) from t=0 to t=1 using the SlerpFlow algorithm.
    Reference: SlerpFlow: Spherical Trajectory Correction for Rectified Flow Inversion.
    """
    if num_steps <= 0:
        raise ValueError(f"num_steps must be positive, got {num_steps}.")

    x = jnp.asarray(x, dtype=jnp.float32)
    batch_size = x.shape[0]
    timesteps = jnp.linspace(0.0, 1.0, num_steps + 1, dtype=jnp.float32)
    dts = jnp.diff(timesteps)

    axes = tuple(range(1, x.ndim))

    def batch_dot(a: jax.Array, b: jax.Array) -> jax.Array:
        return jnp.sum(a * b, axis=axes, keepdims=True)

    def batch_norm(a: jax.Array) -> jax.Array:
        return jnp.sqrt(jnp.sum(jnp.square(a), axis=axes, keepdims=True) + 1e-8)

    def slerp(u: jax.Array, v: jax.Array, alpha: float = 0.5) -> jax.Array:
        dot = batch_dot(u, v)
        dot = jnp.clip(dot, -1.0 + 1e-7, 1.0 - 1e-7)
        omega = jnp.arccos(dot)
        
        lerp_res = (1.0 - alpha) * u + alpha * v
        
        sin_omega = jnp.sin(omega)
        sin_omega_safe = jnp.where(omega < 1e-6, 1e-6, sin_omega)
        slerp_res = (jnp.sin((1.0 - alpha) * omega) / sin_omega_safe) * u + \
                    (jnp.sin(alpha * omega) / sin_omega_safe) * v
        
        d_next = jnp.where(omega < 1e-6, lerp_res, slerp_res)
        return d_next / batch_norm(d_next)

    t0 = timesteps[0]
    dt0 = dts[0]
    v_t = velocity_fn(x, _broadcast_time(t0, batch_size))
    nfe = jnp.asarray(1, dtype=jnp.int32)

    def body(carry: tuple[jax.Array, jax.Array, jax.Array], step: jax.Array):
        Z_t, v_curr, nfe_curr = carry
        
        t = timesteps[step]
        dt = dts[step]
        t_next = timesteps[step + 1]

        Z_euler = Z_t + dt * v_curr
        
        v_next = velocity_fn(Z_euler, _broadcast_time(t_next, batch_size))
        v_bar = (v_curr + v_next) / 2.0
        
        rho_t = batch_norm(Z_t)
        d_t = Z_t / rho_t
        
        # Target Radius
        rho_next = rho_t + dt * batch_dot(d_t, v_bar)
        
        # Target Direction
        Z_bar = Z_t + dt * v_bar
        d_euler = Z_bar / batch_norm(Z_bar)
        d_next = slerp(d_t, d_euler, alpha=0.5)
        
        v_slerp = (rho_next * d_next - Z_t) / dt
        Z_next = Z_t + dt * v_slerp
        
        return (Z_next, v_next, nfe_curr + 1), None

    (x_final, _, nfe_final), _ = jax.lax.scan(
        body,
        (x, v_t, nfe),
        jnp.arange(0, num_steps, dtype=jnp.int32)
    )

    if return_nfe:
        return x_final, nfe_final
    return x_final
