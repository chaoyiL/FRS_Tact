"""Tactile-conditioned flow matching decoder with shared trainable GRU + cross-attention."""

from __future__ import annotations

import dataclasses
import math
from functools import partial
from typing import Literal

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import nnx

from train_pi05_frs.utils.bimanual_schema import (
    LEFT_ACTION_SLICE,
    RIGHT_ACTION_SLICE,
    STEERED_ACTION_DIM,
    validate_bimanual_action_dim,
)
from train_pi05_frs.utils.integration import fireflow_integrate_velocity

Array = jax.Array
FlowSolver = Literal["euler", "fireflow"]
DEFAULT_GRU_HIDDEN_DIM = 256
DEFAULT_RESNET_EMBEDDING_DIM = 512
DECODER_INPUT_VERSION = 2
LOSS_COMPONENT_NAMES = ("gt_fm", "vla_fm", "low_safety", "decode", "rank", "repair")
TRAIN_LOSS_COMPONENT_NAMES = (
    "gt_fm",
    "vla_fm",
    "composite_fm",
    "low_safety",
    "decode",
    "rank",
    "repair",
)


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


def flow_matching_loss_per_sample(
    model: TactileConditionedFlowDecoder,
    x_base: Array,
    target: Array,
    t: Array,
    tactile_seq: Array,
    *,
    state: Array | None = None,
    state_keep_mask: Array | None = None,
) -> Array:
    t_view = t[:, None, None]
    x_t = (1.0 - t_view) * x_base + t_view * target
    target_velocity = target - x_base
    predicted_velocity = model(
        x_t,
        t,
        tactile_seq,
        state=state,
        state_keep_mask=state_keep_mask,
    )
    return jnp.mean(jnp.square(predicted_velocity - target_velocity), axis=(1, 2))


def masked_flow_matching_loss_per_sample(
    model: TactileConditionedFlowDecoder,
    x_base: Array,
    target: Array,
    t: Array,
    tactile_seq: Array,
    *,
    state: Array | None = None,
    state_keep_mask: Array | None = None,
) -> Array:
    """Flow-matching loss normalized over only the 20 physical action dimensions."""

    if x_base.shape != target.shape:
        raise ValueError("masked flow matching requires matching actions")
    validate_bimanual_action_dim(
        x_base.shape[-1], field_name="masked flow matching action_dim"
    )
    t_view = t[:, None, None]
    x_t = (1.0 - t_view) * x_base + t_view * target
    target_velocity = target - x_base
    predicted_velocity = model(
        x_t,
        t,
        tactile_seq,
        state=state,
        state_keep_mask=state_keep_mask,
    )
    residual = predicted_velocity - target_velocity
    return jnp.mean(jnp.square(residual[..., :STEERED_ACTION_DIM]), axis=(1, 2))


def decode_mse_per_sample(
    model: TactileConditionedFlowDecoder,
    x_base: Array,
    target: Array,
    tactile_seq: Array,
    *,
    num_steps: int,
    solver: FlowSolver = "euler",
    state: Array | None = None,
    state_keep_mask: Array | None = None,
) -> Array:
    """Per-sample MSE between integrated decode(x_base) and ``target``."""

    decoded = decode_actions(
        model,
        x_base,
        tactile_seq,
        num_steps=num_steps,
        solver=solver,
        state=state,
        state_keep_mask=state_keep_mask,
    )
    return jnp.mean(jnp.square(decoded - target), axis=(1, 2))


def gt_supervised_loss_per_sample(
    model: TactileConditionedFlowDecoder,
    x_base: Array,
    gt_action: Array,
    t: Array,
    tactile_seq: Array,
    *,
    aux_decode_weight: float,
    aux_decode_steps: int,
    aux_decode_solver: FlowSolver = "euler",
    state: Array | None = None,
    state_keep_mask: Array | None = None,
) -> Array:
    """Per-sample ``FM(gt) + λ_aux MSE(decode, gt)``."""

    flow = flow_matching_loss_per_sample(
        model,
        x_base,
        gt_action,
        t,
        tactile_seq,
        state=state,
        state_keep_mask=state_keep_mask,
    )
    if aux_decode_weight == 0.0:
        return flow
    decode_mse = decode_mse_per_sample(
        model,
        x_base,
        gt_action,
        tactile_seq,
        num_steps=aux_decode_steps,
        solver=aux_decode_solver,
        state=state,
        state_keep_mask=state_keep_mask,
    )
    return flow + float(aux_decode_weight) * decode_mse


def three_region_effective_gate_weights(
    gate_weights: Array,
    *,
    low_gate_threshold: float = 0.3,
    high_gate_threshold: float = 0.7,
) -> Array:
    """Saturate confident low/high Gate regions and interpolate only between them."""
    if not 0.0 <= low_gate_threshold < high_gate_threshold <= 1.0:
        raise ValueError(
            "gate thresholds must satisfy 0 <= low < high <= 1, got "
            f"{low_gate_threshold}, {high_gate_threshold}."
        )
    weights = jnp.clip(jax.lax.stop_gradient(gate_weights), 0.0, 1.0)
    return jnp.clip(
        (weights - float(low_gate_threshold))
        / float(high_gate_threshold - low_gate_threshold),
        0.0,
        1.0,
    )


def bimanual_composite_endpoint(
    gt_action: Array,
    predicted_action: Array,
    gate_weights: Array,
    *,
    low_gate_threshold: float = 0.3,
    high_gate_threshold: float = 0.7,
) -> tuple[Array, Array]:
    """Steer the two physical wrists independently and preserve the VLA padding tail."""

    if gt_action.ndim != 3 or gt_action.shape != predicted_action.shape:
        raise ValueError("bimanual composite endpoint requires matching actions")
    validate_bimanual_action_dim(
        gt_action.shape[-1], field_name="bimanual composite endpoint action_dim"
    )
    if gate_weights.shape != (gt_action.shape[0], 2):
        raise ValueError("bimanual gate_weights must have shape [B, 2]")
    if not isinstance(gate_weights, jax.core.Tracer) and not bool(
        jnp.all(jnp.isfinite(gate_weights))
    ):
        raise ValueError("bimanual gate_weights must be finite")
    effective = three_region_effective_gate_weights(
        gate_weights,
        low_gate_threshold=low_gate_threshold,
        high_gate_threshold=high_gate_threshold,
    )
    physical_weights = jnp.concatenate(
        [
            jnp.repeat(effective[:, :1], LEFT_ACTION_SLICE.stop, axis=1),
            jnp.repeat(
                effective[:, 1:],
                RIGHT_ACTION_SLICE.stop - RIGHT_ACTION_SLICE.start,
                axis=1,
            ),
        ],
        axis=1,
    )[:, None, :]
    physical = physical_weights * gt_action[..., :STEERED_ACTION_DIM] + (
        1.0 - physical_weights
    ) * predicted_action[..., :STEERED_ACTION_DIM]
    target = jnp.concatenate(
        [physical, predicted_action[..., STEERED_ACTION_DIM:]], axis=-1
    )
    return target, effective


def bimanual_mse_per_sample(left: Array, right: Array) -> Array:
    """Return physical endpoint MSE independently for the fixed left and right wrists."""

    if left.ndim != 3 or left.shape != right.shape:
        raise ValueError("bimanual MSE requires matching actions")
    validate_bimanual_action_dim(
        left.shape[-1], field_name="bimanual MSE action_dim"
    )
    squared = jnp.square(
        left[..., :STEERED_ACTION_DIM] - right[..., :STEERED_ACTION_DIM]
    )
    return jnp.stack(
        [
            jnp.mean(squared[..., LEFT_ACTION_SLICE], axis=(1, 2)),
            jnp.mean(squared[..., RIGHT_ACTION_SLICE], axis=(1, 2)),
        ],
        axis=1,
    )


def _average_active_wrist_terms(
    left_term: Array,
    left_active: Array,
    right_term: Array,
    right_active: Array,
) -> Array:
    """Average wrist terms without counting a globally inactive wrist."""

    left_active = jnp.asarray(left_active)
    right_active = jnp.asarray(right_active)
    active_count = left_active.astype(left_term.dtype) + right_active.astype(
        left_term.dtype
    )
    total = jnp.where(left_active, left_term, 0.0) + jnp.where(
        right_active, right_term, 0.0
    )
    return total / jnp.maximum(active_count, 1.0)


def _bimanual_active_group_normalized_per_sample(
    penalty: Array, strength: Array
) -> tuple[Array, Array]:
    """Return a vector whose batch mean is the active group's weighted mean."""

    total_strength = jnp.sum(strength)
    active = total_strength > 0.0
    batch_size = jnp.asarray(penalty.shape[0], dtype=penalty.dtype)
    scale = jnp.where(
        active,
        batch_size / jnp.maximum(total_strength, jnp.finfo(penalty.dtype).tiny),
        0.0,
    )
    return strength * penalty * scale, active


def _bimanual_source_group_normalized_per_sample(
    penalty: Array,
    strength: Array,
    source_indices: Array,
) -> tuple[Array, Array]:
    """Normalize an active wrist group independently inside each present source."""

    source_indices = jnp.asarray(source_indices, dtype=jnp.int32)
    if source_indices.shape != (penalty.shape[0],):
        raise ValueError("source_indices must have shape [B]")
    same_source = source_indices[:, None] == source_indices[None, :]
    totals = jnp.sum(same_source * strength[None, :], axis=1)
    active_for_sample = totals > 0.0

    positions = jnp.arange(penalty.shape[0])
    first_position = jnp.min(
        jnp.where(same_source, positions[None, :], penalty.shape[0]), axis=1
    )
    first_in_source = positions == first_position
    active_sources = jnp.sum(
        (first_in_source & active_for_sample).astype(penalty.dtype)
    )
    batch_size = jnp.asarray(penalty.shape[0], dtype=penalty.dtype)
    normalized = (
        strength
        * penalty
        * batch_size
        / jnp.maximum(totals, jnp.finfo(penalty.dtype).tiny)
        / jnp.maximum(active_sources, 1.0)
    )
    return jnp.where(active_for_sample, normalized, 0.0), active_sources > 0.0


def _active_group_normalized_per_sample(penalty: Array, strength: Array) -> Array:
    """Scale an active Gate group so its batch mean equals its weighted group mean."""
    total_strength = jnp.sum(strength)
    batch_size = jnp.asarray(penalty.shape[0], dtype=penalty.dtype)
    scale = jnp.where(
        total_strength > 0.0,
        batch_size / jnp.maximum(total_strength, jnp.finfo(penalty.dtype).tiny),
        0.0,
    )
    return strength * penalty * scale


def gate_preference_ranking_loss_per_sample(
    decoded_action: Array,
    gt_action: Array,
    predicted_action: Array,
    gate_weights: Array,
    *,
    margin: float,
    high_gate_threshold: float = 0.7,
) -> Array:
    """Require confident high-Gate decodes to be closer to GT than pi0.5."""
    if margin < 0:
        raise ValueError(f"ranking margin must be non-negative, got {margin}.")
    mse_gt = jnp.mean(jnp.square(decoded_action - gt_action), axis=(1, 2))
    mse_pred = jnp.mean(jnp.square(decoded_action - predicted_action), axis=(1, 2))
    weights = jnp.clip(jax.lax.stop_gradient(gate_weights), 0.0, 1.0)
    penalty = jax.nn.relu(mse_gt - mse_pred + float(margin))
    strength = weights * (weights >= float(high_gate_threshold))
    return _active_group_normalized_per_sample(penalty, strength)


def low_gate_safety_loss_per_sample(
    decoded_action: Array,
    gt_action: Array,
    predicted_action: Array,
    gate_weights: Array,
    *,
    tolerance: float,
    low_gate_threshold: float = 0.3,
) -> Array:
    """Penalize low-Gate decodes only when far from both acceptable endpoints."""
    if tolerance < 0:
        raise ValueError(f"low-gate safety tolerance must be non-negative, got {tolerance}.")
    mse_gt = jnp.mean(jnp.square(decoded_action - gt_action), axis=(1, 2))
    mse_pred = jnp.mean(jnp.square(decoded_action - predicted_action), axis=(1, 2))
    nearest = jnp.minimum(mse_gt, mse_pred)
    penalty = jax.nn.relu(nearest - float(tolerance))
    weights = jnp.clip(jax.lax.stop_gradient(gate_weights), 0.0, 1.0)
    strength = (1.0 - weights) * (weights <= float(low_gate_threshold))
    return _active_group_normalized_per_sample(penalty, strength)


def high_gate_repair_loss_per_sample(
    decoded_action: Array,
    gt_action: Array,
    predicted_action: Array,
    gate_weights: Array,
    *,
    margin: float,
    high_gate_threshold: float = 0.7,
) -> Array:
    """Require confident high-Gate decodes to beat the frozen pi0.5 baseline."""
    if margin < 0:
        raise ValueError(f"repair margin must be non-negative, got {margin}.")
    mse_gt = jnp.mean(jnp.square(decoded_action - gt_action), axis=(1, 2))
    mse_pi05_gt = jnp.mean(jnp.square(predicted_action - gt_action), axis=(1, 2))
    weights = jnp.clip(jax.lax.stop_gradient(gate_weights), 0.0, 1.0)
    penalty = jax.nn.relu(mse_gt - mse_pi05_gt + float(margin))
    strength = weights * (weights >= float(high_gate_threshold))
    return _active_group_normalized_per_sample(penalty, strength)


def gated_loss_components_per_sample(
    model: TactileConditionedFlowDecoder,
    x_base: Array,
    gt_action: Array,
    predicted_action: Array,
    t: Array,
    tactile_seq: Array,
    gate_weights: Array,
    *,
    state: Array | None = None,
    state_keep_mask: Array | None = None,
    gate_lambda: float,
    aux_decode_weight: float = 1.0,
    aux_decode_steps: int = 10,
    aux_decode_solver: FlowSolver = "euler",
    low_gate_safety_weight: float = 0.0,
    low_gate_safety_margin: float = 0.0,
    rank_weight: float = 0.0,
    rank_margin: float = 0.0,
    repair_weight: float = 0.0,
    repair_margin: float = 0.0,
    low_gate_threshold: float = 0.3,
    high_gate_threshold: float = 0.7,
) -> dict[str, Array]:
    if low_gate_safety_weight < 0:
        raise ValueError(
            f"low-gate safety weight must be non-negative, got {low_gate_safety_weight}."
        )
    if low_gate_safety_margin < 0:
        raise ValueError(
            f"low-gate safety margin must be non-negative, got {low_gate_safety_margin}."
        )
    if rank_weight < 0:
        raise ValueError(f"ranking weight must be non-negative, got {rank_weight}.")
    if repair_weight < 0:
        raise ValueError(f"repair weight must be non-negative, got {repair_weight}.")

    flow_gt = flow_matching_loss_per_sample(
        model,
        x_base,
        gt_action,
        t,
        tactile_seq,
        state=state,
        state_keep_mask=state_keep_mask,
    )
    flow_pi05 = flow_matching_loss_per_sample(
        model,
        x_base,
        predicted_action,
        t,
        tactile_seq,
        state=state,
        state_keep_mask=state_keep_mask,
    )
    effective_weights = three_region_effective_gate_weights(
        gate_weights,
        low_gate_threshold=low_gate_threshold,
        high_gate_threshold=high_gate_threshold,
    )
    zeros = jnp.zeros_like(flow_gt)
    components = {
        "gt_fm": effective_weights * flow_gt,
        "vla_fm": float(gate_lambda) * (1.0 - effective_weights) * flow_pi05,
        "low_safety": zeros,
        "decode": zeros,
        "rank": zeros,
        "repair": zeros,
    }

    decoded = None
    if any(
        weight != 0.0
        for weight in (
            aux_decode_weight,
            low_gate_safety_weight,
            rank_weight,
            repair_weight,
        )
    ):
        decoded = decode_actions(
            model,
            x_base,
            tactile_seq,
            num_steps=aux_decode_steps,
            solver=aux_decode_solver,
            state=state,
            state_keep_mask=state_keep_mask,
        )
    raw_weights = jnp.clip(jax.lax.stop_gradient(gate_weights), 0.0, 1.0)
    high_strength = raw_weights * (raw_weights >= float(high_gate_threshold))
    if aux_decode_weight != 0.0:
        assert decoded is not None
        mse_gt = jnp.mean(jnp.square(decoded - gt_action), axis=(1, 2))
        components["decode"] = float(aux_decode_weight) * _active_group_normalized_per_sample(
            mse_gt, high_strength
        )
    if low_gate_safety_weight != 0.0:
        assert decoded is not None
        components["low_safety"] = float(low_gate_safety_weight) * low_gate_safety_loss_per_sample(
            decoded,
            gt_action,
            predicted_action,
            gate_weights,
            tolerance=low_gate_safety_margin,
            low_gate_threshold=low_gate_threshold,
        )
    if rank_weight != 0.0:
        assert decoded is not None
        components["rank"] = float(rank_weight) * gate_preference_ranking_loss_per_sample(
            decoded,
            gt_action,
            predicted_action,
            gate_weights,
            margin=rank_margin,
            high_gate_threshold=high_gate_threshold,
        )
    if repair_weight != 0.0:
        assert decoded is not None
        components["repair"] = float(repair_weight) * high_gate_repair_loss_per_sample(
            decoded,
            gt_action,
            predicted_action,
            gate_weights,
            margin=repair_margin,
            high_gate_threshold=high_gate_threshold,
        )
    return components


def bimanual_loss_components_per_sample(
    model: TactileConditionedFlowDecoder,
    x_base: Array,
    gt_action: Array,
    predicted_action: Array,
    t: Array,
    tactile_seq: Array,
    gate_weights: Array,
    *,
    state: Array | None = None,
    state_keep_mask: Array | None = None,
    source_indices: Array | None = None,
    gate_lambda: float = 1.0,
    aux_decode_weight: float = 1.0,
    aux_decode_steps: int = 10,
    aux_decode_solver: FlowSolver = "euler",
    low_gate_safety_weight: float = 0.0,
    low_gate_safety_margin: float = 0.0,
    rank_weight: float = 0.0,
    rank_margin: float = 0.0,
    repair_weight: float = 0.0,
    repair_margin: float = 0.0,
    low_gate_threshold: float = 0.3,
    high_gate_threshold: float = 0.7,
) -> dict[str, Array]:
    """Return one masked composite FM call plus independently gated wrist auxiliaries."""

    del gate_lambda
    if low_gate_safety_weight < 0:
        raise ValueError(
            f"low-gate safety weight must be non-negative, got {low_gate_safety_weight}."
        )
    if low_gate_safety_margin < 0:
        raise ValueError(
            f"low-gate safety margin must be non-negative, got {low_gate_safety_margin}."
        )
    if rank_weight < 0:
        raise ValueError(f"ranking weight must be non-negative, got {rank_weight}.")
    if repair_weight < 0:
        raise ValueError(f"repair weight must be non-negative, got {repair_weight}.")

    target, effective = bimanual_composite_endpoint(
        gt_action,
        predicted_action,
        gate_weights,
        low_gate_threshold=low_gate_threshold,
        high_gate_threshold=high_gate_threshold,
    )
    flow = masked_flow_matching_loss_per_sample(
        model,
        x_base,
        target,
        t,
        tactile_seq,
        state=state,
        state_keep_mask=state_keep_mask,
    )
    zeros = jnp.zeros_like(flow)
    low_safety_term = zeros
    decode_term = zeros
    rank_term = zeros
    repair_term = zeros

    needs_decode = any(
        weight != 0.0
        for weight in (
            aux_decode_weight,
            low_gate_safety_weight,
            rank_weight,
            repair_weight,
        )
    )
    if needs_decode:
        decoded = decode_actions(
            model,
            x_base,
            tactile_seq,
            num_steps=aux_decode_steps,
            solver=aux_decode_solver,
            state=state,
            state_keep_mask=state_keep_mask,
        )
        mse_gt = bimanual_mse_per_sample(decoded, gt_action)
        mse_vla = bimanual_mse_per_sample(decoded, predicted_action)
        raw_gates = jnp.clip(jax.lax.stop_gradient(gate_weights), 0.0, 1.0)

        def normalize_wrist(
            penalty: Array, strength: Array
        ) -> tuple[Array, Array]:
            if source_indices is None:
                return _bimanual_active_group_normalized_per_sample(
                    penalty, strength
                )
            return _bimanual_source_group_normalized_per_sample(
                penalty, strength, source_indices
            )

        if aux_decode_weight != 0.0:
            decode_term = float(aux_decode_weight) * jnp.mean(
                effective * mse_gt + (1.0 - effective) * mse_vla,
                axis=1,
            )

        low_strength = (1.0 - raw_gates) * (
            raw_gates <= float(low_gate_threshold)
        )
        high_strength = raw_gates * (raw_gates >= float(high_gate_threshold))

        if low_gate_safety_weight != 0.0:
            low_penalty = jax.nn.relu(
                jnp.minimum(mse_gt, mse_vla) - float(low_gate_safety_margin)
            )
            left_term, left_active = normalize_wrist(
                low_penalty[:, 0], low_strength[:, 0]
            )
            right_term, right_active = normalize_wrist(
                low_penalty[:, 1], low_strength[:, 1]
            )
            low_safety_term = float(
                low_gate_safety_weight
            ) * _average_active_wrist_terms(
                left_term, left_active, right_term, right_active
            )

        if rank_weight != 0.0:
            rank_penalty = jax.nn.relu(
                mse_gt - mse_vla + float(rank_margin)
            )
            left_term, left_active = normalize_wrist(
                rank_penalty[:, 0], high_strength[:, 0]
            )
            right_term, right_active = normalize_wrist(
                rank_penalty[:, 1], high_strength[:, 1]
            )
            rank_term = float(rank_weight) * _average_active_wrist_terms(
                left_term, left_active, right_term, right_active
            )

        if repair_weight != 0.0:
            baseline = bimanual_mse_per_sample(predicted_action, gt_action)
            repair_penalty = jax.nn.relu(
                mse_gt - baseline + float(repair_margin)
            )
            left_term, left_active = normalize_wrist(
                repair_penalty[:, 0], high_strength[:, 0]
            )
            right_term, right_active = normalize_wrist(
                repair_penalty[:, 1], high_strength[:, 1]
            )
            repair_term = float(repair_weight) * _average_active_wrist_terms(
                left_term, left_active, right_term, right_active
            )

    return {
        "gt_fm": zeros,
        "vla_fm": zeros,
        "composite_fm": flow,
        "low_safety": low_safety_term,
        "decode": decode_term,
        "rank": rank_term,
        "repair": repair_term,
    }


def gated_flow_matching_loss_per_sample(
    model: TactileConditionedFlowDecoder,
    x_base: Array,
    gt_action: Array,
    predicted_action: Array,
    t: Array,
    tactile_seq: Array,
    gate_weights: Array,
    **kwargs,
) -> Array:
    """Return the per-sample sum of the six FRS_Tact gated-v7 loss terms."""
    return sum(
        gated_loss_components_per_sample(
            model,
            x_base,
            gt_action,
            predicted_action,
            t,
            tactile_seq,
            gate_weights,
            **kwargs,
        ).values()
    )


@partial(
    nnx.jit,
    static_argnames=(
        "loss_mode",
        "gate_lambda",
        "aux_decode_weight",
        "aux_decode_steps",
        "aux_decode_solver",
        "low_gate_safety_weight",
        "low_gate_safety_margin",
        "rank_weight",
        "rank_margin",
        "repair_weight",
        "repair_margin",
        "low_gate_threshold",
        "high_gate_threshold",
        "state_dropout_rate",
    ),
)
def _train_step_jit(
    model: TactileConditionedFlowDecoder,
    optimizer: nnx.Optimizer,
    x_base: Array,
    gt_action: Array,
    predicted_action: Array,
    tactile_seq: Array,
    gate_weights: Array,
    key: Array,
    source_indices: Array | None = None,
    *,
    state: Array | None = None,
    state_dropout_rate: float = 0.0,
    loss_mode: str = "gt",
    gate_lambda: float = 1.0,
    aux_decode_weight: float = 1.0,
    aux_decode_steps: int = 10,
    aux_decode_solver: FlowSolver = "euler",
    low_gate_safety_weight: float = 0.0,
    low_gate_safety_margin: float = 0.0,
    rank_weight: float = 0.0,
    rank_margin: float = 0.0,
    repair_weight: float = 0.0,
    repair_margin: float = 0.0,
    low_gate_threshold: float = 0.3,
    high_gate_threshold: float = 0.7,
) -> tuple[Array, dict[str, Array]]:
    if not 0.0 <= state_dropout_rate < 1.0:
        raise ValueError(f"state_dropout_rate must be in [0, 1), got {state_dropout_rate}.")
    time_key, state_key = jax.random.split(key)
    t = jax.random.uniform(time_key, (x_base.shape[0],), minval=0.0, maxval=1.0)
    state_keep_mask = None
    if model.config.state_conditioning:
        if state is None:
            raise ValueError("state is required when state_conditioning is enabled.")
        state_keep_mask = jax.random.bernoulli(
            state_key,
            p=1.0 - float(state_dropout_rate),
            shape=(x_base.shape[0],),
        )

    def loss_fn(candidate: TactileConditionedFlowDecoder) -> tuple[Array, dict[str, Array]]:
        if loss_mode == "gt":
            flow = flow_matching_loss_per_sample(
                candidate,
                x_base,
                gt_action,
                t,
                tactile_seq,
                state=state,
                state_keep_mask=state_keep_mask,
            )
            decode = jnp.zeros_like(flow)
            if aux_decode_weight != 0.0:
                decode = float(aux_decode_weight) * decode_mse_per_sample(
                    candidate,
                    x_base,
                    gt_action,
                    tactile_seq,
                    num_steps=aux_decode_steps,
                    solver=aux_decode_solver,
                    state=state,
                    state_keep_mask=state_keep_mask,
                )
            components = {
                name: jnp.asarray(0.0, dtype=flow.dtype) for name in LOSS_COMPONENT_NAMES
            }
            components["gt_fm"] = jnp.mean(flow)
            components["decode"] = jnp.mean(decode)
        elif loss_mode == "predicted":
            total = flow_matching_loss_per_sample(
                candidate,
                x_base,
                predicted_action,
                t,
                tactile_seq,
                state=state,
                state_keep_mask=state_keep_mask,
            )
            components = {name: jnp.asarray(0.0, dtype=total.dtype) for name in LOSS_COMPONENT_NAMES}
            components["vla_fm"] = jnp.mean(total)
        elif loss_mode == "gated":
            per_sample = gated_loss_components_per_sample(
                candidate,
                x_base,
                gt_action,
                predicted_action,
                t,
                tactile_seq,
                gate_weights,
                state=state,
                state_keep_mask=state_keep_mask,
                gate_lambda=gate_lambda,
                aux_decode_weight=aux_decode_weight,
                aux_decode_steps=aux_decode_steps,
                aux_decode_solver=aux_decode_solver,
                low_gate_safety_weight=low_gate_safety_weight,
                low_gate_safety_margin=low_gate_safety_margin,
                rank_weight=rank_weight,
                rank_margin=rank_margin,
                repair_weight=repair_weight,
                repair_margin=repair_margin,
                low_gate_threshold=low_gate_threshold,
                high_gate_threshold=high_gate_threshold,
            )
            components = {name: jnp.mean(per_sample[name]) for name in LOSS_COMPONENT_NAMES}
        elif loss_mode == "bimanual_gated":
            per_sample = bimanual_loss_components_per_sample(
                candidate,
                x_base,
                gt_action,
                predicted_action,
                t,
                tactile_seq,
                gate_weights,
                state=state,
                state_keep_mask=state_keep_mask,
                source_indices=source_indices,
                aux_decode_weight=aux_decode_weight,
                aux_decode_steps=aux_decode_steps,
                aux_decode_solver=aux_decode_solver,
                low_gate_safety_weight=low_gate_safety_weight,
                low_gate_safety_margin=low_gate_safety_margin,
                rank_weight=rank_weight,
                rank_margin=rank_margin,
                repair_weight=repair_weight,
                repair_margin=repair_margin,
                low_gate_threshold=low_gate_threshold,
                high_gate_threshold=high_gate_threshold,
            )
            components = {
                name: jnp.mean(per_sample[name])
                for name in TRAIN_LOSS_COMPONENT_NAMES
            }
        else:
            raise ValueError(
                "loss_mode must be 'gt', 'predicted', 'gated', or 'bimanual_gated', "
                f"got {loss_mode!r}."
            )
        loss = sum(components.values())
        return loss, components

    (loss, components), gradients = nnx.value_and_grad(loss_fn, has_aux=True)(model)
    optimizer.update(gradients)
    return loss, components


def train_step(
    model: TactileConditionedFlowDecoder,
    optimizer: nnx.Optimizer,
    x_base: Array,
    gt_action: Array,
    predicted_action: Array,
    tactile_seq: Array,
    gate_weights: Array,
    key: Array,
    source_indices: Array | None = None,
    *,
    state: Array | None = None,
    state_dropout_rate: float = 0.0,
    loss_mode: str = "gt",
    gate_lambda: float = 1.0,
    aux_decode_weight: float = 1.0,
    aux_decode_steps: int = 10,
    aux_decode_solver: FlowSolver = "euler",
    low_gate_safety_weight: float = 0.0,
    low_gate_safety_margin: float = 0.0,
    rank_weight: float = 0.0,
    rank_margin: float = 0.0,
    repair_weight: float = 0.0,
    repair_margin: float = 0.0,
    low_gate_threshold: float = 0.3,
    high_gate_threshold: float = 0.7,
) -> tuple[Array, dict[str, Array]]:
    """Validate the small bimanual label before the compiled optimizer update."""

    if loss_mode == "bimanual_gated":
        host_gates = np.asarray(jax.device_get(gate_weights))
        if np.any(~np.isfinite(host_gates)):
            raise ValueError(
                "bimanual gate_weights must be finite before optimizer update"
            )
        if source_indices is not None and source_indices.shape != (x_base.shape[0],):
            raise ValueError("source_indices must have shape [B]")
    return _train_step_jit(
        model,
        optimizer,
        x_base,
        gt_action,
        predicted_action,
        tactile_seq,
        gate_weights,
        key,
        source_indices,
        state=state,
        state_dropout_rate=state_dropout_rate,
        loss_mode=loss_mode,
        gate_lambda=gate_lambda,
        aux_decode_weight=aux_decode_weight,
        aux_decode_steps=aux_decode_steps,
        aux_decode_solver=aux_decode_solver,
        low_gate_safety_weight=low_gate_safety_weight,
        low_gate_safety_margin=low_gate_safety_margin,
        rank_weight=rank_weight,
        rank_margin=rank_margin,
        repair_weight=repair_weight,
        repair_margin=repair_margin,
        low_gate_threshold=low_gate_threshold,
        high_gate_threshold=high_gate_threshold,
    )


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
