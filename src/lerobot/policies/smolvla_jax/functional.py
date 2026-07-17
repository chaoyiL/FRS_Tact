from __future__ import annotations

import math

import jax
import jax.numpy as jnp

Array = jax.Array


def linear(x: Array, weight: Array, bias: Array | None = None) -> Array:
    """PyTorch-layout linear layer: weight is [out_features, in_features]."""

    output = jnp.einsum("...d,od->...o", x, weight, preferred_element_type=jnp.float32)
    if bias is not None:
        output = output + bias
    return output.astype(jnp.result_type(x.dtype, weight.dtype))


def layer_norm(x: Array, weight: Array, bias: Array, eps: float) -> Array:
    input_dtype = x.dtype
    x32 = x.astype(jnp.float32)
    mean = jnp.mean(x32, axis=-1, keepdims=True)
    variance = jnp.mean(jnp.square(x32 - mean), axis=-1, keepdims=True)
    normalized = (x32 - mean) * jax.lax.rsqrt(variance + eps)
    return (normalized.astype(input_dtype) * weight + bias).astype(input_dtype)


def rms_norm(x: Array, weight: Array, eps: float) -> Array:
    input_dtype = x.dtype
    x32 = x.astype(jnp.float32)
    variance = jnp.mean(jnp.square(x32), axis=-1, keepdims=True)
    normalized = x32 * jax.lax.rsqrt(variance + eps)
    return (weight * normalized.astype(input_dtype)).astype(jnp.result_type(weight.dtype, input_dtype))


def gelu_pytorch_tanh(x: Array) -> Array:
    input_dtype = x.dtype
    x = x.astype(jnp.float32)
    coefficient = math.sqrt(2.0 / math.pi)
    output = 0.5 * x * (1.0 + jnp.tanh(coefficient * (x + 0.044715 * jnp.power(x, 3))))
    return output.astype(input_dtype)


def silu_pytorch(x: Array) -> Array:
    """Match torch.nn.functional.silu for BF16 inputs (FP32 compute, cast back)."""

    input_dtype = x.dtype
    x = x.astype(jnp.float32)
    return (x * jax.nn.sigmoid(x)).astype(input_dtype)


def apply_rope(x: Array, positions: Array, max_wavelength: float = 10_000.0) -> Array:
    """Apply the exact non-interleaved RoPE variant used by SmolVLA."""

    input_dtype = x.dtype
    x32 = x.astype(jnp.float32)
    half = x.shape[-1] // 2
    exponents = (2.0 / x.shape[-1]) * jnp.arange(half, dtype=jnp.float32)
    timescale = max_wavelength**exponents
    radians = positions[..., None].astype(jnp.float32) / timescale[None, None, :]
    sin = jnp.sin(radians)[..., None, :]
    cos = jnp.cos(radians)[..., None, :]
    x1, x2 = x32[..., :half], x32[..., half:]
    return jnp.concatenate((x1 * cos - x2 * sin, x2 * cos + x1 * sin), axis=-1).astype(input_dtype)


def repeat_kv(x: Array, repeats: int) -> Array:
    if repeats == 1:
        return x
    batch, length, kv_heads, head_dim = x.shape
    x = jnp.broadcast_to(x[:, :, :, None, :], (batch, length, kv_heads, repeats, head_dim))
    return x.reshape(batch, length, kv_heads * repeats, head_dim)


def eager_attention(query: Array, key: Array, value: Array, mask: Array) -> Array:
    """GQA attention with [batch, length, heads, head_dim] inputs."""

    repeats = query.shape[2] // key.shape[2]
    key = repeat_kv(key, repeats)
    value = repeat_kv(value, repeats)
    scores = jnp.einsum("bqhd,bkhd->bhqk", query.astype(jnp.float32), key.astype(jnp.float32)) * (
        query.shape[-1] ** -0.5
    )
    scores = jnp.where(mask[:, None, :, :], scores, jnp.finfo(jnp.float32).min)
    probs = jax.nn.softmax(scores, axis=-1).astype(value.dtype)
    output = jnp.einsum("bhqk,bkhd->bqhd", probs, value)
    return output.reshape(output.shape[0], output.shape[1], -1)


def make_att_2d_masks(pad_masks: Array, att_masks: Array) -> Array:
    cumulative = jnp.cumsum(att_masks, axis=1)
    attention = cumulative[:, None, :] <= cumulative[:, :, None]
    padding = pad_masks[:, None, :] & pad_masks[:, :, None]
    return attention & padding


def sinusoidal_time_embedding(time: Array, dimension: int, min_period: float, max_period: float) -> Array:
    if dimension % 2:
        raise ValueError(f"dimension must be even, got {dimension}")
    fraction = jnp.linspace(0.0, 1.0, dimension // 2, dtype=jnp.float32)
    period = min_period * jnp.power(max_period / min_period, fraction)
    phase = (2.0 * math.pi / period)[None, :] * time[:, None]
    return jnp.concatenate((jnp.sin(phase), jnp.cos(phase)), axis=-1)


def pad_last_dim(x: Array, target: int) -> Array:
    if x.shape[-1] > target:
        raise ValueError(f"cannot pad dimension {x.shape[-1]} to smaller target {target}")
    return jnp.pad(x, [(0, 0)] * (x.ndim - 1) + [(0, target - x.shape[-1])])
