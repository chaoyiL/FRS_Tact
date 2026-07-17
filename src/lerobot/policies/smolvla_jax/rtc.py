from __future__ import annotations

import math
from dataclasses import dataclass

import jax
import jax.numpy as jnp

Array = jax.Array


@dataclass(frozen=True)
class JaxRTCConfig:
    enabled: bool = True
    prefix_attention_schedule: str = "LINEAR"
    max_guidance_weight: float = 10.0
    execution_horizon: int = 10


def prefix_weights(schedule: str, start: int, end: int, total: int) -> Array:
    start = min(start, end)
    if schedule == "ZEROS":
        return (jnp.arange(total) < start).astype(jnp.float32)
    if schedule == "ONES":
        return (jnp.arange(total) < end).astype(jnp.float32)
    middle_length = max(end - start, 0)
    if middle_length:
        middle = jnp.linspace(1.0, 0.0, middle_length + 2, dtype=jnp.float32)[1:-1]
        if schedule == "EXP":
            middle = middle * jnp.expm1(middle) / (math.e - 1.0)
    else:
        middle = jnp.zeros((0,), dtype=jnp.float32)
    return jnp.concatenate(
        (
            jnp.ones((min(start, total),), dtype=jnp.float32),
            middle,
            jnp.zeros((max(total - end, 0),), dtype=jnp.float32),
        )
    )[:total]


def rtc_guided_velocity(
    velocity_fn,
    x_t: Array,
    previous_chunk: Array | None,
    *,
    time: Array,
    inference_delay: int,
    execution_horizon: int,
    config: JaxRTCConfig,
) -> Array:
    velocity = velocity_fn(x_t)
    if previous_chunk is None:
        return velocity
    if previous_chunk.ndim == 2:
        previous_chunk = previous_chunk[None]
    if previous_chunk.shape[0] == 1 and x_t.shape[0] > 1:
        previous_chunk = jnp.broadcast_to(
            previous_chunk,
            (x_t.shape[0], *previous_chunk.shape[1:]),
        )
    if previous_chunk.shape[0] != x_t.shape[0]:
        raise ValueError(
            f"previous chunk batch {previous_chunk.shape[0]} does not match input batch {x_t.shape[0]}"
        )
    if previous_chunk.shape[1] > x_t.shape[1] or previous_chunk.shape[2] > x_t.shape[2]:
        raise ValueError(
            f"previous chunk shape {previous_chunk.shape} cannot be padded to input shape {x_t.shape}"
        )
    previous_chunk = jnp.pad(
        previous_chunk,
        (
            (0, x_t.shape[0] - previous_chunk.shape[0]),
            (0, x_t.shape[1] - previous_chunk.shape[1]),
            (0, x_t.shape[2] - previous_chunk.shape[2]),
        ),
    )
    horizon = min(execution_horizon, previous_chunk.shape[1])
    weights = prefix_weights(
        config.prefix_attention_schedule,
        inference_delay,
        horizon,
        x_t.shape[1],
    )[None, :, None]
    x1_t = x_t - time * velocity
    error = (previous_chunk - x1_t) * weights

    # This mirrors the current LeRobot implementation: x_t.requires_grad_()
    # is set after velocity is evaluated, so the correction is d(x_t)/dx_t.
    correction = error
    tau = 1.0 - time
    one_minus_tau_squared = jnp.square(1.0 - tau)
    inv_r2 = (one_minus_tau_squared + jnp.square(tau)) / one_minus_tau_squared
    coefficient = jnp.nan_to_num((1.0 - tau) / tau, posinf=config.max_guidance_weight)
    guidance = jnp.nan_to_num(coefficient * inv_r2, posinf=config.max_guidance_weight)
    guidance = jnp.minimum(guidance, config.max_guidance_weight)
    return velocity - guidance * correction
