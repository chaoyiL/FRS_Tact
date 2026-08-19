"""Tactile-conditioned flow matching decoder with shared trainable GRU + cross-attention."""

from __future__ import annotations

import dataclasses
import math
from functools import partial
from typing import Literal

import jax
import jax.numpy as jnp
from flax import nnx

from .integration import fireflow_integrate_velocity

Array = jax.Array
FlowSolver = Literal["euler", "fireflow"]
DEFAULT_GRU_HIDDEN_DIM = 256
DEFAULT_RESNET_EMBEDDING_DIM = 512
DECODER_INPUT_VERSION = 2


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
    state_conditioning: bool = False
    state_dim: int = 0
    decoder_input_version: int = DECODER_INPUT_VERSION

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
        if self.state_conditioning and self.state_dim <= 0:
            raise ValueError("state_dim must be positive when state_conditioning is enabled.")
        if self.decoder_input_version != DECODER_INPUT_VERSION:
            raise ValueError(
                f"decoder_input_version must be {DECODER_INPUT_VERSION}, "
                f"got {self.decoder_input_version}."
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
        if config.state_conditioning:
            self.state_norm = nnx.LayerNorm(config.state_dim, rngs=rngs)
            self.state_fc1 = nnx.Linear(config.state_dim, config.model_dim, rngs=rngs)
            self.state_fc2 = nnx.Linear(config.model_dim, config.model_dim, rngs=rngs)
        self.blocks = [
            ConditionedTransformerBlock(
                config.model_dim, config.num_heads, config.mlp_ratio, rngs=rngs
            )
            for _ in range(config.depth)
        ]
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

    def encode_condition(
        self,
        tactile_seq: Array,
        state: Array | None = None,
        state_keep_mask: Array | None = None,
    ) -> Array:
        tactile = self.tactile_proj(self.encode_tactile_tokens(tactile_seq))
        if not self.config.state_conditioning:
            return tactile
        if state is None:
            raise ValueError("state is required by this state-conditioned decoder.")
        state = jnp.asarray(state, dtype=jnp.float32)
        expected = (tactile.shape[0], self.config.state_dim)
        if state.shape != expected:
            raise ValueError(f"Expected state shape {expected}, got {state.shape}.")
        state_token = self.state_fc2(nnx.silu(self.state_fc1(self.state_norm(state))))
        if state_keep_mask is not None:
            state_keep_mask = jnp.asarray(state_keep_mask, dtype=state_token.dtype)
            if state_keep_mask.shape != (tactile.shape[0],):
                raise ValueError(
                    f"Expected state_keep_mask shape {(tactile.shape[0],)}, "
                    f"got {state_keep_mask.shape}."
                )
            state_token = state_token * state_keep_mask[:, None]
        return jnp.concatenate((state_token[:, None, :], tactile), axis=1)

    def __call__(
        self,
        x_t: Array,
        t: Array,
        tactile_seq: Array,
        *,
        state: Array | None = None,
        state_keep_mask: Array | None = None,
    ) -> Array:
        condition = self.encode_condition(tactile_seq, state, state_keep_mask)
        return self.velocity_from_condition(x_t, t, condition)

    def velocity_from_condition(
        self,
        x_t: Array,
        t: Array,
        condition: Array,
    ) -> Array:
        """Predict velocity while reusing already encoded tactile/state tokens."""
        if x_t.ndim != 3:
            raise ValueError(f"Expected x_t with shape [B, T, A], got {x_t.shape}.")
        x = self.action_in(x_t)
        x = x + sequence_position_embedding(x.shape[1], self.config.model_dim)[None, :, :]
        x = x + self.time_mlp(t)[:, None, :]
        for block in self.blocks:
            x = block(x, condition)
        return self.action_out(self.out_norm(x))


@partial(nnx.jit, static_argnames=("num_steps",))
def decode_euler(
    model: TactileConditionedFlowDecoder,
    x_base: Array,
    tactile_seq: Array,
    *,
    num_steps: int,
    state: Array | None = None,
    state_keep_mask: Array | None = None,
) -> Array:
    if num_steps <= 0:
        raise ValueError(f"num_steps must be positive, got {num_steps}.")
    batch_size = x_base.shape[0]
    dt = jnp.asarray(1.0 / num_steps, dtype=jnp.float32)
    condition = model.encode_condition(tactile_seq, state, state_keep_mask)

    def body(step: int, x_t: Array) -> Array:
        t = jnp.full((batch_size,), step * dt, dtype=jnp.float32)
        return x_t + dt * model.velocity_from_condition(x_t, t, condition)

    return jax.lax.fori_loop(0, num_steps, body, jnp.asarray(x_base, dtype=jnp.float32))


@partial(nnx.jit, static_argnames=("num_steps",))
def decode_fireflow(
    model: TactileConditionedFlowDecoder,
    x_base: Array,
    tactile_seq: Array,
    *,
    num_steps: int,
    state: Array | None = None,
    state_keep_mask: Array | None = None,
) -> Array:
    condition = model.encode_condition(tactile_seq, state, state_keep_mask)
    return fireflow_integrate_velocity(
        lambda x, t: model.velocity_from_condition(x, t, condition),
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
    state: Array | None = None,
    state_keep_mask: Array | None = None,
) -> Array:
    if solver == "euler":
        return decode_euler(
            model,
            x_base,
            tactile_seq,
            num_steps=num_steps,
            state=state,
            state_keep_mask=state_keep_mask,
        )
    if solver == "fireflow":
        return decode_fireflow(
            model,
            x_base,
            tactile_seq,
            num_steps=num_steps,
            state=state,
            state_keep_mask=state_keep_mask,
        )
    raise ValueError(f"solver must be 'euler' or 'fireflow', got {solver!r}.")
