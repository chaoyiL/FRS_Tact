"""Tactile-conditioned flow matching decoder with shared trainable GRU + cross-attention."""

from __future__ import annotations

import dataclasses
import math
from functools import partial
from typing import Literal

import jax
import jax.numpy as jnp
import optax
from flax import nnx

from tactile_flow_steering.utils.integration import fireflow_integrate_velocity

Array = jax.Array
FlowSolver = Literal["euler", "fireflow"]
DEFAULT_GRU_HIDDEN_DIM = 256
DEFAULT_RESNET_EMBEDDING_DIM = 512


@dataclasses.dataclass(frozen=True)
class DecoderConfig:
    action_dim: int
    action_horizon: int
    tactile_window: int
    gru_hidden_dim: int = DEFAULT_GRU_HIDDEN_DIM
    resnet_embedding_dim: int = DEFAULT_RESNET_EMBEDDING_DIM
    model_dim: int = 128
    depth: int = 4
    num_heads: int = 4
    mlp_ratio: int = 4
    num_tactile_tokens: int = 4

    def __post_init__(self) -> None:
        if min(
            self.action_dim,
            self.action_horizon,
            self.tactile_window,
            self.gru_hidden_dim,
            self.resnet_embedding_dim,
            self.model_dim,
            self.depth,
            self.num_heads,
            self.mlp_ratio,
            self.num_tactile_tokens,
        ) <= 0:
            raise ValueError("All decoder dimensions must be positive.")
        if self.model_dim % self.num_heads:
            raise ValueError(
                f"model_dim ({self.model_dim}) must be divisible by num_heads ({self.num_heads})."
            )

    @property
    def tactile_token_dim(self) -> int:
        """Cross-attn token feature size (== GRU hidden dim)."""

        return self.gru_hidden_dim


def sinusoidal_embedding(x: Array, dim: int, max_period: float = 10_000.0) -> Array:
    half = dim // 2
    frequencies = jnp.exp(-math.log(max_period) * jnp.arange(half) / max(half - 1, 1))
    arguments = x[..., None] * frequencies
    embedding = jnp.concatenate([jnp.sin(arguments), jnp.cos(arguments)], axis=-1)
    if dim % 2:
        embedding = jnp.pad(embedding, [(0, 0)] * x.ndim + [(0, 1)])
    return embedding


def sequence_position_embedding(length: int, dim: int) -> Array:
    return sinusoidal_embedding(jnp.arange(length, dtype=jnp.float32), dim)


class TimeMLP(nnx.Module):
    def __init__(self, dim: int, *, rngs: nnx.Rngs):
        self.dim = dim
        self.fc1 = nnx.Linear(dim, 4 * dim, rngs=rngs)
        self.fc2 = nnx.Linear(4 * dim, dim, rngs=rngs)

    def __call__(self, t: Array) -> Array:
        hidden = nnx.silu(self.fc1(sinusoidal_embedding(t, self.dim)))
        return self.fc2(hidden)


class SharedTactileGRU(nnx.Module):
    """Shared single-layer GRU: ``[B, T, D] → [B, H]`` final hidden."""

    def __init__(self, input_dim: int, hidden_dim: int, *, rngs: nnx.Rngs):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.cell = nnx.GRUCell(input_dim, hidden_dim, rngs=rngs)

    def __call__(self, xs: Array) -> Array:
        if xs.ndim != 3:
            raise ValueError(f"Expected GRU inputs [B, T, D], got {xs.shape}.")
        if xs.shape[-1] != self.input_dim:
            raise ValueError(
                f"Expected input_dim={self.input_dim}, got {xs.shape[-1]}."
            )
        batch_size = xs.shape[0]
        carry = jnp.zeros((batch_size, self.hidden_dim), dtype=xs.dtype)
        xs_time_major = jnp.swapaxes(xs, 0, 1)  # [T, B, D]

        def step(carry_t: Array, x_t: Array) -> tuple[Array, Array]:
            new_carry, output = self.cell(carry_t, x_t)
            return new_carry, output

        final_carry, _ = jax.lax.scan(step, carry, xs_time_major)
        return final_carry


class ConditionedTransformerBlock(nnx.Module):
    """Self-attn on action tokens, then cross-attn to tactile tokens, then MLP."""

    def __init__(self, dim: int, num_heads: int, mlp_ratio: int, *, rngs: nnx.Rngs):
        self.norm_self = nnx.LayerNorm(dim, rngs=rngs)
        self.self_attention = nnx.MultiHeadAttention(
            num_heads=num_heads,
            in_features=dim,
            qkv_features=dim,
            out_features=dim,
            dropout_rate=0.0,
            decode=False,
            rngs=rngs,
        )
        self.norm_cross_q = nnx.LayerNorm(dim, rngs=rngs)
        self.norm_cross_kv = nnx.LayerNorm(dim, rngs=rngs)
        self.cross_attention = nnx.MultiHeadAttention(
            num_heads=num_heads,
            in_features=dim,
            qkv_features=dim,
            out_features=dim,
            dropout_rate=0.0,
            decode=False,
            rngs=rngs,
        )
        self.norm_mlp = nnx.LayerNorm(dim, rngs=rngs)
        hidden_dim = mlp_ratio * dim
        self.fc1 = nnx.Linear(dim, hidden_dim, rngs=rngs)
        self.fc2 = nnx.Linear(hidden_dim, dim, rngs=rngs)

    def __call__(self, x: Array, tactile_tokens: Array) -> Array:
        self_normalized = self.norm_self(x)
        x = x + self.self_attention(self_normalized, deterministic=True)
        q = self.norm_cross_q(x)
        kv = self.norm_cross_kv(tactile_tokens)
        x = x + self.cross_attention(q, kv, kv, deterministic=True)
        mlp_normalized = self.norm_mlp(x)
        return x + self.fc2(nnx.gelu(self.fc1(mlp_normalized)))


class TactileConditionedFlowDecoder(nnx.Module):
    """v_theta(x_t, t, tactile_seq) with shared GRU + per-block tactile cross-attention."""

    def __init__(self, config: DecoderConfig, *, rngs: nnx.Rngs):
        self.config = config
        self.action_in = nnx.Linear(config.action_dim, config.model_dim, rngs=rngs)
        self.time_mlp = TimeMLP(config.model_dim, rngs=rngs)
        self.tactile_gru = SharedTactileGRU(
            config.resnet_embedding_dim,
            config.gru_hidden_dim,
            rngs=rngs,
        )
        self.tactile_proj = nnx.Linear(config.gru_hidden_dim, config.model_dim, rngs=rngs)
        self.blocks = nnx.List(
            [
                ConditionedTransformerBlock(
                    config.model_dim, config.num_heads, config.mlp_ratio, rngs=rngs
                )
                for _ in range(config.depth)
            ]
        )
        self.out_norm = nnx.LayerNorm(config.model_dim, rngs=rngs)
        self.action_out = nnx.Linear(config.model_dim, config.action_dim, rngs=rngs)

    def encode_tactile_tokens(self, tactile_seq: Array) -> Array:
        """``[B, T, N, D] → [B, N, H]`` via shared GRU over each sensor stream."""

        if tactile_seq.ndim != 4:
            raise ValueError(
                f"Expected tactile_seq with shape [B, T, N, D], got {tactile_seq.shape}."
            )
        batch_size, time_steps, num_streams, embedding_dim = tactile_seq.shape
        if time_steps != self.config.tactile_window:
            raise ValueError(
                f"Expected tactile_window={self.config.tactile_window}, got T={time_steps}."
            )
        if num_streams != self.config.num_tactile_tokens:
            raise ValueError(
                f"Expected {self.config.num_tactile_tokens} tactile streams, got {num_streams}."
            )
        if embedding_dim != self.config.resnet_embedding_dim:
            raise ValueError(
                f"Expected resnet_embedding_dim={self.config.resnet_embedding_dim}, "
                f"got {embedding_dim}."
            )
        # [B, T, N, D] -> [B, N, T, D] -> [B * N, T, D]
        sequences = jnp.transpose(tactile_seq, (0, 2, 1, 3)).reshape(
            batch_size * num_streams, time_steps, embedding_dim
        )
        hidden = self.tactile_gru(sequences)
        return hidden.reshape(batch_size, num_streams, self.config.gru_hidden_dim)

    def __call__(self, x_t: Array, t: Array, tactile_seq: Array) -> Array:
        if x_t.ndim != 3:
            raise ValueError(f"Expected x_t with shape [B, T, A], got {x_t.shape}.")
        tactile_tokens = self.encode_tactile_tokens(tactile_seq)
        x = self.action_in(x_t)
        x = x + sequence_position_embedding(x.shape[1], self.config.model_dim)[None, :, :]
        x = x + self.time_mlp(t)[:, None, :]
        condition = self.tactile_proj(tactile_tokens)
        for block in self.blocks:
            x = block(x, condition)
        return self.action_out(self.out_norm(x))


def flow_matching_loss_per_sample(
    model: TactileConditionedFlowDecoder,
    x_base: Array,
    target: Array,
    t: Array,
    tactile_seq: Array,
) -> Array:
    t_view = t[:, None, None]
    x_t = (1.0 - t_view) * x_base + t_view * target
    target_velocity = target - x_base
    predicted_velocity = model(x_t, t, tactile_seq)
    return jnp.mean(jnp.square(predicted_velocity - target_velocity), axis=(1, 2))


def gated_flow_matching_loss_per_sample(
    model: TactileConditionedFlowDecoder,
    x_base: Array,
    gt_action: Array,
    predicted_action: Array,
    t: Array,
    tactile_seq: Array,
    gate_weights: Array,
    *,
    gate_lambda: float,
) -> Array:
    """Per-sample ``w L* + λ (1-w) L_stop`` with shared noise time ``t``."""

    loss_star = flow_matching_loss_per_sample(model, x_base, gt_action, t, tactile_seq)
    loss_stop = flow_matching_loss_per_sample(model, x_base, predicted_action, t, tactile_seq)
    weights = jax.lax.stop_gradient(gate_weights)
    return weights * loss_star + float(gate_lambda) * (1.0 - weights) * loss_stop


@partial(nnx.jit, static_argnames=("loss_mode", "gate_lambda"))
def train_step(
    model: TactileConditionedFlowDecoder,
    optimizer: nnx.Optimizer,
    x_base: Array,
    gt_action: Array,
    predicted_action: Array,
    tactile_seq: Array,
    gate_weights: Array,
    key: Array,
    *,
    loss_mode: str = "gt",
    gate_lambda: float = 1.0,
) -> Array:
    t = jax.random.uniform(key, (x_base.shape[0],), minval=0.0, maxval=1.0)

    def loss_fn(candidate: TactileConditionedFlowDecoder) -> Array:
        if loss_mode == "gt":
            return jnp.mean(
                flow_matching_loss_per_sample(candidate, x_base, gt_action, t, tactile_seq)
            )
        if loss_mode == "gated":
            return jnp.mean(
                gated_flow_matching_loss_per_sample(
                    candidate,
                    x_base,
                    gt_action,
                    predicted_action,
                    t,
                    tactile_seq,
                    gate_weights,
                    gate_lambda=gate_lambda,
                )
            )
        raise ValueError(f"loss_mode must be 'gt' or 'gated', got {loss_mode!r}.")

    loss, gradients = nnx.value_and_grad(loss_fn)(model)
    optimizer.update(model, gradients)
    return loss


@partial(nnx.jit, static_argnames=("num_steps",))
def decode_euler(
    model: TactileConditionedFlowDecoder,
    x_base: Array,
    tactile_seq: Array,
    *,
    num_steps: int,
) -> Array:
    if num_steps <= 0:
        raise ValueError(f"num_steps must be positive, got {num_steps}.")
    batch_size = x_base.shape[0]
    dt = jnp.asarray(1.0 / num_steps, dtype=jnp.float32)

    def body(step: int, x_t: Array) -> Array:
        t = jnp.full((batch_size,), step * dt, dtype=jnp.float32)
        return x_t + dt * model(x_t, t, tactile_seq)

    return jax.lax.fori_loop(0, num_steps, body, jnp.asarray(x_base, dtype=jnp.float32))


@partial(nnx.jit, static_argnames=("num_steps",))
def decode_fireflow(
    model: TactileConditionedFlowDecoder,
    x_base: Array,
    tactile_seq: Array,
    *,
    num_steps: int,
) -> Array:
    return fireflow_integrate_velocity(
        lambda x, t: model(x, t, tactile_seq),
        x_base,
        num_steps=num_steps,
    )


def decode_actions(
    model: TactileConditionedFlowDecoder,
    x_base: Array,
    tactile_seq: Array,
    *,
    num_steps: int,
    solver: FlowSolver = "euler",
) -> Array:
    if solver == "euler":
        return decode_euler(model, x_base, tactile_seq, num_steps=num_steps)
    if solver == "fireflow":
        return decode_fireflow(model, x_base, tactile_seq, num_steps=num_steps)
    raise ValueError(f"solver must be 'euler' or 'fireflow', got {solver!r}.")


def resolve_peak_learning_rate(
    learning_rate: float,
    *,
    model_dim: int,
    lr_reference_dim: int | None,
) -> float:
    if lr_reference_dim is None:
        return learning_rate
    if lr_reference_dim <= 0:
        raise ValueError(f"lr_reference_dim must be positive when set, got {lr_reference_dim}.")
    return learning_rate * math.sqrt(lr_reference_dim / model_dim)


def make_learning_rate_schedule(
    *,
    learning_rate: float,
    warmup_steps: int,
    total_steps: int,
    min_learning_rate_ratio: float = 0.1,
    cosine_decay: bool = True,
) -> optax.Schedule | float:
    if total_steps <= 0:
        raise ValueError(f"total_steps must be positive, got {total_steps}.")
    if warmup_steps < 0:
        raise ValueError(f"warmup_steps must be non-negative, got {warmup_steps}.")
    if not 0.0 <= min_learning_rate_ratio <= 1.0:
        raise ValueError(
            f"min_learning_rate_ratio must be in [0, 1], got {min_learning_rate_ratio}."
        )

    end_value = learning_rate * min_learning_rate_ratio
    if not cosine_decay:
        if warmup_steps <= 0:
            return learning_rate
        return optax.warmup_constant_schedule(
            init_value=0.0,
            peak_value=learning_rate,
            warmup_steps=warmup_steps,
        )

    if warmup_steps > 0:
        return optax.warmup_cosine_decay_schedule(
            init_value=0.0,
            peak_value=learning_rate,
            warmup_steps=warmup_steps,
            decay_steps=total_steps,
            end_value=end_value,
        )
    if min_learning_rate_ratio == 1.0:
        return learning_rate
    return optax.cosine_decay_schedule(
        init_value=learning_rate,
        decay_steps=total_steps,
        alpha=min_learning_rate_ratio,
    )


def make_optimizer(
    model: TactileConditionedFlowDecoder,
    *,
    learning_rate: float,
    weight_decay: float,
    grad_clip_norm: float | None = 1.0,
    warmup_steps: int = 0,
    total_steps: int = 1,
    min_learning_rate_ratio: float = 0.1,
    cosine_decay: bool = True,
) -> nnx.Optimizer:
    if grad_clip_norm is not None and grad_clip_norm <= 0:
        raise ValueError(f"grad_clip_norm must be positive when set, got {grad_clip_norm}.")
    lr = make_learning_rate_schedule(
        learning_rate=learning_rate,
        warmup_steps=warmup_steps,
        total_steps=total_steps,
        min_learning_rate_ratio=min_learning_rate_ratio,
        cosine_decay=cosine_decay,
    )
    adamw = optax.adamw(lr, weight_decay=weight_decay)
    transform = (
        optax.chain(optax.clip_by_global_norm(grad_clip_norm), adamw)
        if grad_clip_norm is not None
        else adamw
    )
    return nnx.Optimizer(model, transform, wrt=nnx.Param)
