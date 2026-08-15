"""Tactile/state-conditioned flow decoder with GRU and cross-attention."""

from __future__ import annotations

import dataclasses
import math
from functools import partial
from typing import Literal

import jax
import jax.numpy as jnp
import optax
from flax import nnx
from flax.core import FrozenDict

from train_encoder.utils.resnet import encode_resnet18, init_resnet18_params
from train_frs.utils.integration import fireflow_integrate_velocity

Array = jax.Array
FlowSolver = Literal["euler", "fireflow"]
HighGateRankAggregation = Literal["balanced_mean", "worst_source_cvar"]
LOSS_COMPONENT_NAMES = (
    "gt_fm",
    "vla_fm",
    "low_safety",
    "decode",
    "rank",
    "repair",
)
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
    state_dim: int = 0
    state_conditioning: bool = False
    tactile_encoder_trainable: bool = False
    tactile_image_size: int = 224
    tactile_encode_microbatch_size: int = 8
    use_gru: bool = True
    zero_tactile_tokens: bool = False

    def __post_init__(self) -> None:
        if (
            min(
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
            )
            <= 0
        ):
            raise ValueError("All decoder dimensions must be positive.")
        if self.model_dim % self.num_heads:
            raise ValueError(f"model_dim ({self.model_dim}) must be divisible by num_heads ({self.num_heads}).")
        if self.state_dim < 0:
            raise ValueError("state_dim must be non-negative.")
        if self.state_conditioning and self.state_dim <= 0:
            raise ValueError("state_dim must be positive when state_conditioning is enabled.")
        if self.tactile_encoder_trainable and self.tactile_image_size <= 0:
            raise ValueError("tactile_image_size must be positive for a trainable tactile encoder.")
        if self.tactile_encode_microbatch_size <= 0:
            raise ValueError("tactile_encode_microbatch_size must be positive.")

    @property
    def tactile_token_dim(self) -> int:
        """Token feature size before the model-dim projection."""

        return self.gru_hidden_dim if self.use_gru else self.resnet_embedding_dim


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
            raise ValueError(f"Expected input_dim={self.input_dim}, got {xs.shape[-1]}.")
        batch_size = xs.shape[0]
        carry = jnp.zeros((batch_size, self.hidden_dim), dtype=xs.dtype)
        xs_time_major = jnp.swapaxes(xs, 0, 1)  # [T, B, D]

        def step(carry_t: Array, x_t: Array) -> tuple[Array, Array]:
            new_carry, output = self.cell(carry_t, x_t)
            return new_carry, output

        final_carry, _ = jax.lax.scan(step, carry, xs_time_major)
        return final_carry


class ConditionedTransformerBlock(nnx.Module):
    """Self-attn on action tokens, then cross-attn to condition tokens, then MLP."""

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


def _as_nnx_variable_tree(
    tree: dict | FrozenDict,
    variable_type: type[nnx.Variable],
) -> nnx.Dict:
    """Recursively make a Linen variable tree visible to NNX transforms."""

    return nnx.Dict(
        {
            str(name): (
                _as_nnx_variable_tree(value, variable_type)
                if isinstance(value, (dict, FrozenDict))
                else variable_type(value)
            )
            for name, value in tree.items()
        }
    )


class TactileConditionedFlowDecoder(nnx.Module):
    """Flow decoder conditioned on tactile history and optional state."""

    def __init__(
        self,
        config: DecoderConfig,
        *,
        rngs: nnx.Rngs,
        tactile_resnet_variables: dict | None = None,
    ):
        self.config = config
        if config.tactile_encoder_trainable:
            if tactile_resnet_variables is None:
                tactile_resnet_variables = init_resnet18_params(
                    jax.random.key(0),
                    image_size=config.tactile_image_size,
                    embedding_dim=config.resnet_embedding_dim,
                )
            missing = {"params", "batch_stats"} - set(tactile_resnet_variables)
            if missing:
                raise KeyError(f"Tactile ResNet variables are missing: {sorted(missing)}")
            self.tactile_resnet_params = _as_nnx_variable_tree(
                tactile_resnet_variables["params"],
                nnx.Param,
            )
            # BatchNorm running statistics are checkpointed model state but are
            # intentionally excluded from AdamW. Fine-tuning uses them in inference
            # mode so small FRS batches do not corrupt the pretrained statistics.
            self.tactile_resnet_batch_stats = _as_nnx_variable_tree(
                tactile_resnet_variables["batch_stats"],
                nnx.BatchStat,
            )
        self.action_in = nnx.Linear(config.action_dim, config.model_dim, rngs=rngs)
        self.time_mlp = TimeMLP(config.model_dim, rngs=rngs)
        if config.use_gru:
            self.tactile_gru = SharedTactileGRU(
                config.resnet_embedding_dim,
                config.gru_hidden_dim,
                rngs=rngs,
            )
            tactile_proj_in = config.gru_hidden_dim
        else:
            tactile_proj_in = config.resnet_embedding_dim
        self.tactile_proj = nnx.Linear(tactile_proj_in, config.model_dim, rngs=rngs)
        if config.state_conditioning:
            self.state_norm = nnx.LayerNorm(config.state_dim, rngs=rngs)
            self.state_fc1 = nnx.Linear(config.state_dim, config.model_dim, rngs=rngs)
            self.state_fc2 = nnx.Linear(config.model_dim, config.model_dim, rngs=rngs)
        self.blocks = nnx.List(
            [
                ConditionedTransformerBlock(config.model_dim, config.num_heads, config.mlp_ratio, rngs=rngs)
                for _ in range(config.depth)
            ]
        )
        self.out_norm = nnx.LayerNorm(config.model_dim, rngs=rngs)
        self.action_out = nnx.Linear(config.model_dim, config.action_dim, rngs=rngs)

    def encode_tactile_images(self, tactile_images: Array) -> Array:
        """``[B,T,N,H,W,C] → [B,T,N,D]`` through the trainable ResNet."""

        if not self.config.tactile_encoder_trainable:
            raise ValueError("This checkpoint does not contain a trainable tactile encoder.")
        if tactile_images.ndim != 6:
            raise ValueError(
                "Expected tactile images [B,T,N,H,W,C], got "
                f"{tactile_images.shape}."
            )
        batch_size, time_steps, num_streams, height, width, channels = (
            tactile_images.shape
        )
        if num_streams != self.config.num_tactile_tokens:
            raise ValueError(
                f"Expected {self.config.num_tactile_tokens} tactile streams, "
                f"got {num_streams}."
            )
        expected_image_shape = (
            self.config.tactile_image_size,
            self.config.tactile_image_size,
            3,
        )
        if (height, width, channels) != expected_image_shape:
            raise ValueError(
                f"Expected tactile image shape {expected_image_shape}, "
                f"got {(height, width, channels)}."
            )

        flat = jnp.asarray(tactile_images).reshape(
            batch_size * time_steps * num_streams,
            height,
            width,
            channels,
        )
        microbatch_size = min(
            self.config.tactile_encode_microbatch_size,
            flat.shape[0],
        )
        padding = (-flat.shape[0]) % microbatch_size
        if padding:
            flat = jnp.pad(flat, ((0, padding), (0, 0), (0, 0), (0, 0)))
        chunks = flat.reshape(
            -1,
            microbatch_size,
            height,
            width,
            channels,
        )
        variables = {
            "params": nnx.to_pure_dict(nnx.state(self.tactile_resnet_params)),
            "batch_stats": nnx.to_pure_dict(
                nnx.state(self.tactile_resnet_batch_stats)
            ),
        }

        @jax.checkpoint
        def encode_chunk(images: Array) -> Array:
            integer_input = jnp.issubdtype(images.dtype, jnp.integer)
            images = jnp.asarray(images, dtype=jnp.float32)
            if integer_input:
                images = images / 255.0
            embeddings, _ = encode_resnet18(
                variables,
                images,
                train=False,
                embedding_dim=self.config.resnet_embedding_dim,
            )
            return embeddings

        encoded = jax.lax.map(encode_chunk, chunks).reshape(
            -1,
            self.config.resnet_embedding_dim,
        )
        encoded = encoded[: batch_size * time_steps * num_streams]
        return encoded.reshape(
            batch_size,
            time_steps,
            num_streams,
            self.config.resnet_embedding_dim,
        )

    def encode_tactile_input(self, tactile_input: Array) -> Array:
        """Accept cached embeddings or raw images and return embedding sequences."""

        if tactile_input.ndim == 4:
            return jnp.asarray(tactile_input, dtype=jnp.float32)
        if tactile_input.ndim == 6:
            return self.encode_tactile_images(tactile_input)
        raise ValueError(
            "Expected tactile embeddings [B,T,N,D] or images [B,T,N,H,W,C], "
            f"got {tactile_input.shape}."
        )

    def encode_tactile_tokens(self, tactile_seq: Array) -> Array:
        """``[B, T, N, D] → [B, N, token_dim]``.

        With GRU, each sensor stream is reduced over time. Without GRU, the
        current (last) frame's ResNet embeddings are returned unchanged.
        """

        if tactile_seq.ndim != 4:
            raise ValueError(f"Expected tactile_seq with shape [B, T, N, D], got {tactile_seq.shape}.")
        batch_size, time_steps, num_streams, embedding_dim = tactile_seq.shape
        if time_steps < 1:
            raise ValueError("tactile_seq must contain at least one time step")
        if num_streams != self.config.num_tactile_tokens:
            raise ValueError(f"Expected {self.config.num_tactile_tokens} tactile streams, got {num_streams}.")
        if embedding_dim != self.config.resnet_embedding_dim:
            raise ValueError(
                f"Expected resnet_embedding_dim={self.config.resnet_embedding_dim}, " f"got {embedding_dim}."
            )
        if not self.config.use_gru:
            return tactile_seq[:, -1, :, :]
        # [B, T, N, D] -> [B, N, T, D] -> [B * N, T, D]
        sequences = jnp.transpose(tactile_seq, (0, 2, 1, 3)).reshape(
            batch_size * num_streams, time_steps, embedding_dim
        )
        hidden = self.tactile_gru(sequences)
        return hidden.reshape(batch_size, num_streams, self.config.gru_hidden_dim)

    def encode_tactile_condition(self, tactile_seq: Array) -> Array:
        """Encode tactile history into cross-attention conditioning tokens."""

        tactile_seq = self.encode_tactile_input(tactile_seq)
        return self.tactile_proj(self.encode_tactile_tokens(tactile_seq))

    def encode_condition(
        self,
        tactile_seq: Array,
        state: Array | None = None,
        state_keep_mask: Array | None = None,
    ) -> Array:
        """Return tactile tokens with an optional normalized-current-state token."""

        tactile = self.encode_tactile_condition(tactile_seq)
        if self.config.zero_tactile_tokens:
            tactile = jnp.zeros_like(tactile)
        if not self.config.state_conditioning:
            return tactile
        if state is None:
            raise ValueError("state is required by this state-conditioned checkpoint.")
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

    def _decode_velocity(
        self,
        x_t: Array,
        t: Array,
        tactile_condition: Array,
    ) -> Array:
        if x_t.ndim != 3:
            raise ValueError(f"Expected x_t with shape [B, T, A], got {x_t.shape}.")
        x = self.action_in(x_t)
        x = x + sequence_position_embedding(x.shape[1], self.config.model_dim)[None, :, :]
        x = x + self.time_mlp(t)[:, None, :]
        for block in self.blocks:
            x = block(x, tactile_condition)
        return self.action_out(self.out_norm(x))

    def velocity_from_condition(
        self,
        x_t: Array,
        t: Array,
        tactile_condition: Array,
    ) -> Array:
        return self._decode_velocity(x_t, t, tactile_condition)

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
    """Map raw gate confidence to a saturated three-region objective weight.

    Only loss targets use this remapping: the confident-low region is fully VLA, the
    confident-high region is fully GT, and the transition region interpolates.
    """

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


def _active_group_normalized_per_sample(penalty: Array, strength: Array) -> tuple[Array, Array]:
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


def _source_group_normalized_per_sample(
    penalty: Array,
    strength: Array,
    source_indices: Array,
    *,
    num_sources: int,
) -> tuple[Array, Array]:
    """Normalize an active group independently inside every dataset source."""

    source_indices = jnp.asarray(source_indices, dtype=jnp.int32)
    ones = jnp.ones_like(strength)
    counts = jnp.zeros((num_sources,), dtype=strength.dtype).at[source_indices].add(ones)
    totals = jnp.zeros((num_sources,), dtype=strength.dtype).at[source_indices].add(strength)
    active = totals > 0.0
    scales = jnp.where(
        active,
        counts / jnp.maximum(totals, jnp.finfo(strength.dtype).tiny),
        0.0,
    )
    return strength * penalty * scales[source_indices], active


def source_balanced_mean(values: Array, source_indices: Array, *, num_sources: int) -> Array:
    """Average samples within each present source, then average sources equally."""

    source_indices = jnp.asarray(source_indices, dtype=jnp.int32)
    values = jnp.asarray(values)
    totals = jnp.zeros((num_sources,), dtype=values.dtype).at[source_indices].add(values)
    counts = jnp.zeros((num_sources,), dtype=values.dtype).at[source_indices].add(
        jnp.ones_like(values)
    )
    active = counts > 0.0
    means = jnp.where(active, totals / jnp.maximum(counts, 1.0), 0.0)
    return jnp.sum(means) / jnp.maximum(jnp.sum(active.astype(values.dtype)), 1.0)


def _source_group_mean(
    penalty: Array,
    strength: Array,
    source_indices: Array,
    *,
    num_sources: int,
) -> tuple[Array, Array]:
    """Return the equally source-balanced mean and whether any source is active."""

    source_indices = jnp.asarray(source_indices, dtype=jnp.int32)
    totals = jnp.zeros((num_sources,), dtype=penalty.dtype).at[source_indices].add(
        penalty * strength
    )
    strengths = jnp.zeros((num_sources,), dtype=penalty.dtype).at[source_indices].add(
        strength
    )
    active = strengths > 0.0
    means = jnp.where(
        active,
        totals / jnp.maximum(strengths, jnp.finfo(penalty.dtype).tiny),
        0.0,
    )
    return (
        jnp.sum(means) / jnp.maximum(jnp.sum(active.astype(penalty.dtype)), 1.0),
        jnp.any(active),
    )


def high_gate_worst_source_cvar_loss(
    penalty: Array,
    strength: Array,
    source_indices: Array,
    source_weights: Array,
    *,
    num_sources: int,
    hard_fraction: float,
    worst_beta: float,
) -> tuple[Array, Array]:
    """Smooth worst-source loss over each source's hardest high-gate samples.

    Balanced batches contain either floor(B/D) or ceil(B/D) samples per source.
    We take a static top-k based on the larger quota so this remains JIT-friendly.
    Missing high-gate samples are masked and never enter the CVaR mean.  Positive
    ``source_weights`` act as source priors in the smooth maximum; lowering the
    prior for an already-easy source keeps it from consuming rank gradients while
    still allowing it to dominate if it genuinely becomes the worst source.
    """

    if not 0.0 < hard_fraction <= 1.0:
        raise ValueError(f"hard_fraction must be in (0, 1], got {hard_fraction}.")
    if worst_beta <= 0.0:
        raise ValueError(f"worst_beta must be positive, got {worst_beta}.")
    if num_sources <= 0:
        raise ValueError(f"num_sources must be positive, got {num_sources}.")

    source_indices = jnp.asarray(source_indices, dtype=jnp.int32)
    source_weights = jnp.asarray(source_weights, dtype=penalty.dtype)
    expected_shape = (num_sources,)
    if source_weights.shape != expected_shape:
        raise ValueError(
            f"source_weights must have shape {expected_shape}, got {source_weights.shape}."
        )
    per_source_quota = math.ceil(penalty.shape[0] / num_sources)
    hard_k = max(1, math.ceil(per_source_quota * hard_fraction))
    source_ids = jnp.arange(num_sources, dtype=jnp.int32)[:, None]
    active_mask = (source_indices[None, :] == source_ids) & (strength[None, :] > 0.0)
    weighted_penalty = penalty[None, :] * strength[None, :]
    masked = jnp.where(active_mask, weighted_penalty, -jnp.inf)
    top_values, _ = jax.lax.top_k(masked, hard_k)
    valid = jnp.isfinite(top_values)
    source_losses = jnp.sum(jnp.where(valid, top_values, 0.0), axis=1) / jnp.maximum(
        jnp.sum(valid, axis=1),
        1.0,
    )
    active_sources = jnp.any(valid, axis=1) & (source_weights > 0.0)
    active_weight = jnp.where(active_sources, source_weights, 0.0)
    log_prior = jnp.where(active_sources, jnp.log(jnp.maximum(active_weight, 1.0e-12)), -jnp.inf)
    normalizer = jnp.log(jnp.maximum(jnp.sum(active_weight), 1.0e-12))
    smooth_worst = (
        jax.scipy.special.logsumexp(float(worst_beta) * source_losses + log_prior)
        - normalizer
    ) / float(worst_beta)
    any_active = jnp.any(active_sources)
    return jnp.where(any_active, smooth_worst, 0.0), any_active


def gate_preference_ranking_loss_per_sample(
    decoded_action: Array,
    gt_action: Array,
    predicted_action: Array,
    gate_weights: Array,
    *,
    margin: float,
    low_gate_threshold: float = 0.3,
    high_gate_threshold: float = 0.7,
    source_indices: Array | None = None,
    num_sources: int = 1,
) -> Array:
    """Require confident high-gate decodes to be closer to GT than VLA.

    Low-gate endpoint preference is intentionally absent: either GT-like or
    VLA-like output is acceptable there.  Low-gate safety is handled separately
    by :func:`low_gate_safety_loss_per_sample`.
    """

    if margin < 0:
        raise ValueError(f"ranking margin must be non-negative, got {margin}.")
    if not 0.0 <= low_gate_threshold < high_gate_threshold <= 1.0:
        raise ValueError(
            "gate thresholds must satisfy 0 <= low < high <= 1, got " f"{low_gate_threshold}, {high_gate_threshold}."
        )
    mse_gt = jnp.mean(jnp.square(decoded_action - gt_action), axis=(1, 2))
    mse_pred = jnp.mean(jnp.square(decoded_action - predicted_action), axis=(1, 2))
    weights = jnp.clip(jax.lax.stop_gradient(gate_weights), 0.0, 1.0)
    high_penalty = jax.nn.relu(mse_gt - mse_pred + float(margin))
    high_strength = weights * (weights >= float(high_gate_threshold))
    if source_indices is None:
        high_term, _ = _active_group_normalized_per_sample(high_penalty, high_strength)
        return high_term
    high_term, active_sources = _source_group_normalized_per_sample(
        high_penalty,
        high_strength,
        source_indices,
        num_sources=num_sources,
    )
    return high_term * (
        float(num_sources)
        / jnp.maximum(jnp.sum(active_sources.astype(high_term.dtype)), 1.0)
    )


def low_gate_safety_loss_per_sample(
    decoded_action: Array,
    gt_action: Array,
    predicted_action: Array,
    gate_weights: Array,
    *,
    tolerance: float,
    low_gate_threshold: float = 0.3,
    source_indices: Array | None = None,
    num_sources: int = 1,
) -> Array:
    """Penalize low-gate actions only when they are far from both endpoints.

    ``min(MSE(FRS, GT), MSE(FRS, VLA))`` represents distance to the nearer
    acceptable endpoint.  The hinge stops producing gradients once that distance
    is within ``tolerance``; unlike the old low-gate rank term it does not force a
    preference between GT and VLA.
    """

    if tolerance < 0:
        raise ValueError(f"low-gate safety tolerance must be non-negative, got {tolerance}.")
    if not 0.0 <= low_gate_threshold <= 1.0:
        raise ValueError(f"low_gate_threshold must be in [0, 1], got {low_gate_threshold}.")
    mse_gt = jnp.mean(jnp.square(decoded_action - gt_action), axis=(1, 2))
    mse_pred = jnp.mean(jnp.square(decoded_action - predicted_action), axis=(1, 2))
    nearest = jnp.minimum(mse_gt, mse_pred)
    penalty = jax.nn.relu(nearest - float(tolerance))
    weights = jnp.clip(jax.lax.stop_gradient(gate_weights), 0.0, 1.0)
    low_strength = (1.0 - weights) * (weights <= float(low_gate_threshold))
    if source_indices is None:
        normalized, _ = _active_group_normalized_per_sample(penalty, low_strength)
    else:
        normalized, active_sources = _source_group_normalized_per_sample(
            penalty,
            low_strength,
            source_indices,
            num_sources=num_sources,
        )
        normalized = normalized * (
            float(num_sources)
            / jnp.maximum(jnp.sum(active_sources.astype(normalized.dtype)), 1.0)
        )
    return normalized


def high_gate_repair_loss_per_sample(
    decoded_action: Array,
    gt_action: Array,
    predicted_action: Array,
    gate_weights: Array,
    *,
    margin: float,
    high_gate_threshold: float = 0.7,
    source_indices: Array | None = None,
    num_sources: int = 1,
) -> Array:
    """Require confident high-gate decodes to beat the frozen VLA baseline."""

    if margin < 0:
        raise ValueError(f"repair margin must be non-negative, got {margin}.")
    if not 0.0 <= high_gate_threshold <= 1.0:
        raise ValueError(f"high_gate_threshold must be in [0, 1], got {high_gate_threshold}.")
    mse_gt = jnp.mean(jnp.square(decoded_action - gt_action), axis=(1, 2))
    mse_vla_gt = jnp.mean(jnp.square(predicted_action - gt_action), axis=(1, 2))
    weights = jnp.clip(jax.lax.stop_gradient(gate_weights), 0.0, 1.0)
    penalty = jax.nn.relu(mse_gt - mse_vla_gt + float(margin))
    high_strength = weights * (weights >= float(high_gate_threshold))
    if source_indices is None:
        normalized, _ = _active_group_normalized_per_sample(penalty, high_strength)
    else:
        normalized, active_sources = _source_group_normalized_per_sample(
            penalty,
            high_strength,
            source_indices,
            num_sources=num_sources,
        )
        normalized = normalized * (
            float(num_sources)
            / jnp.maximum(jnp.sum(active_sources.astype(normalized.dtype)), 1.0)
        )
    return normalized


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
    rank_low_gate_threshold: float = 0.3,
    rank_high_gate_threshold: float = 0.7,
    source_indices: Array | None = None,
    num_sources: int = 1,
    high_gate_rank_aggregation: HighGateRankAggregation = "balanced_mean",
    high_gate_rank_hard_fraction: float = 0.3,
    high_gate_rank_worst_beta: float = 20.0,
    source_rank_weights: Array | None = None,
) -> dict[str, Array]:
    """Return asymmetric gated-loss terms whose per-sample sum is the total."""

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
    if high_gate_rank_aggregation not in ("balanced_mean", "worst_source_cvar"):
        raise ValueError(
            "high_gate_rank_aggregation must be 'balanced_mean' or "
            f"'worst_source_cvar', got {high_gate_rank_aggregation!r}."
        )

    flow_gt = flow_matching_loss_per_sample(
        model,
        x_base,
        gt_action,
        t,
        tactile_seq,
        state=state,
        state_keep_mask=state_keep_mask,
    )
    flow_vla = flow_matching_loss_per_sample(
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
        low_gate_threshold=rank_low_gate_threshold,
        high_gate_threshold=rank_high_gate_threshold,
    )
    zeros = jnp.zeros_like(flow_gt)
    low_safety_term = zeros
    decode_term = zeros
    rank_term = zeros
    repair_term = zeros

    decoded = None
    if (
        aux_decode_weight != 0.0
        or low_gate_safety_weight != 0.0
        or rank_weight != 0.0
        or repair_weight != 0.0
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
    if aux_decode_weight != 0.0:
        assert decoded is not None
        decode_mse_gt = jnp.mean(jnp.square(decoded - gt_action), axis=(1, 2))
        raw_weights = jnp.clip(jax.lax.stop_gradient(gate_weights), 0.0, 1.0)
        high_strength = raw_weights * (
            raw_weights >= float(rank_high_gate_threshold)
        )
        if source_indices is None:
            normalized_gt, _ = _active_group_normalized_per_sample(
                decode_mse_gt,
                high_strength,
            )
        else:
            normalized_gt, active_sources = _source_group_normalized_per_sample(
                decode_mse_gt,
                high_strength,
                source_indices,
                num_sources=num_sources,
            )
            normalized_gt = normalized_gt * (
                float(num_sources)
                / jnp.maximum(jnp.sum(active_sources.astype(normalized_gt.dtype)), 1.0)
            )
        decode_term = float(aux_decode_weight) * normalized_gt
    if low_gate_safety_weight != 0.0:
        assert decoded is not None
        low_safety_term = float(low_gate_safety_weight) * low_gate_safety_loss_per_sample(
            decoded,
            gt_action,
            predicted_action,
            gate_weights,
            tolerance=low_gate_safety_margin,
            low_gate_threshold=rank_low_gate_threshold,
            source_indices=source_indices,
            num_sources=num_sources,
        )
    if rank_weight != 0.0:
        assert decoded is not None
        if high_gate_rank_aggregation == "balanced_mean":
            rank_term = float(rank_weight) * gate_preference_ranking_loss_per_sample(
                decoded,
                gt_action,
                predicted_action,
                gate_weights,
                margin=rank_margin,
                low_gate_threshold=rank_low_gate_threshold,
                high_gate_threshold=rank_high_gate_threshold,
                source_indices=source_indices,
                num_sources=num_sources,
            )
        else:
            if source_indices is None:
                raise ValueError("worst_source_cvar rank aggregation requires source_indices")
            mse_gt = jnp.mean(jnp.square(decoded - gt_action), axis=(1, 2))
            mse_pred = jnp.mean(jnp.square(decoded - predicted_action), axis=(1, 2))
            raw_weights = jnp.clip(jax.lax.stop_gradient(gate_weights), 0.0, 1.0)
            high_penalty = jax.nn.relu(mse_gt - mse_pred + float(rank_margin))
            high_strength = raw_weights * (
                raw_weights >= float(rank_high_gate_threshold)
            )
            if source_rank_weights is None:
                source_rank_weights = jnp.ones((num_sources,), dtype=high_penalty.dtype)
            high_loss, high_active = high_gate_worst_source_cvar_loss(
                high_penalty,
                high_strength,
                source_indices,
                source_rank_weights,
                num_sources=num_sources,
                hard_fraction=high_gate_rank_hard_fraction,
                worst_beta=high_gate_rank_worst_beta,
            )
            rank_scalar = float(rank_weight) * jnp.where(high_active, high_loss, 0.0)
            rank_term = jnp.full_like(flow_gt, rank_scalar)
    if repair_weight != 0.0:
        assert decoded is not None
        repair_term = float(repair_weight) * high_gate_repair_loss_per_sample(
            decoded,
            gt_action,
            predicted_action,
            gate_weights,
            margin=repair_margin,
            high_gate_threshold=rank_high_gate_threshold,
            source_indices=source_indices,
            num_sources=num_sources,
        )

    return {
        "gt_fm": effective_weights * flow_gt,
        "vla_fm": float(gate_lambda) * (1.0 - effective_weights) * flow_vla,
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
    rank_low_gate_threshold: float = 0.3,
    rank_high_gate_threshold: float = 0.7,
    high_gate_rank_aggregation: HighGateRankAggregation = "balanced_mean",
    high_gate_rank_hard_fraction: float = 0.3,
    high_gate_rank_worst_beta: float = 20.0,
    source_indices: Array | None = None,
    source_rank_weights: Array | None = None,
    num_sources: int = 1,
) -> Array:
    """Gated endpoint loss plus preference and absolute-repair constraints."""

    components = gated_loss_components_per_sample(
        model,
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
        rank_low_gate_threshold=rank_low_gate_threshold,
        rank_high_gate_threshold=rank_high_gate_threshold,
        source_indices=source_indices,
        num_sources=num_sources,
        high_gate_rank_aggregation=high_gate_rank_aggregation,
        high_gate_rank_hard_fraction=high_gate_rank_hard_fraction,
        high_gate_rank_worst_beta=high_gate_rank_worst_beta,
        source_rank_weights=source_rank_weights,
    )
    return sum(components.values())


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
        "rank_low_gate_threshold",
        "rank_high_gate_threshold",
        "source_balanced_loss",
        "num_sources",
        "high_gate_rank_aggregation",
        "high_gate_rank_hard_fraction",
        "high_gate_rank_worst_beta",
        "state_dropout_rate",
    ),
)
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
    source_rank_weights: Array | None = None,
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
    rank_low_gate_threshold: float = 0.3,
    rank_high_gate_threshold: float = 0.7,
    source_balanced_loss: bool = False,
    num_sources: int = 1,
    high_gate_rank_aggregation: HighGateRankAggregation = "balanced_mean",
    high_gate_rank_hard_fraction: float = 0.3,
    high_gate_rank_worst_beta: float = 20.0,
) -> tuple[Array, dict[str, Array]]:
    if not 0.0 <= state_dropout_rate < 1.0:
        raise ValueError(f"state_dropout_rate must be in [0, 1), got {state_dropout_rate}.")
    time_key, state_dropout_key = jax.random.split(key)
    t = jax.random.uniform(time_key, (x_base.shape[0],), minval=0.0, maxval=1.0)
    state_keep_mask = None
    if model.config.state_conditioning:
        if state is None:
            raise ValueError("state is required when state_conditioning is enabled")
        state_keep_mask = jax.random.bernoulli(
            state_dropout_key,
            p=1.0 - float(state_dropout_rate),
            shape=(x_base.shape[0],),
        )

    def loss_fn(
        candidate: TactileConditionedFlowDecoder,
    ) -> tuple[Array, dict[str, Array]]:
        rank_requires_sources = high_gate_rank_aggregation == "worst_source_cvar"
        if (source_balanced_loss or rank_requires_sources) and source_indices is None:
            raise ValueError(
                "source_indices are required for source-balanced loss or "
                "worst_source_cvar rank aggregation"
            )

        def reduce_component(values: Array) -> Array:
            if source_balanced_loss:
                assert source_indices is not None
                return source_balanced_mean(values, source_indices, num_sources=num_sources)
            return jnp.mean(values)

        # Raw-image mode is expensive: encode the ResNet exactly once, then reuse
        # its differentiable embeddings across both FM endpoints and ODE decode.
        tactile_embeddings = candidate.encode_tactile_input(tactile_seq)
        if loss_mode == "gt":
            flow = flow_matching_loss_per_sample(
                candidate,
                x_base,
                gt_action,
                t,
                tactile_embeddings,
                state=state,
                state_keep_mask=state_keep_mask,
            )
            decode = jnp.zeros_like(flow)
            if aux_decode_weight != 0.0:
                decode = float(aux_decode_weight) * decode_mse_per_sample(
                    candidate,
                    x_base,
                    gt_action,
                    tactile_embeddings,
                    num_steps=aux_decode_steps,
                    solver=aux_decode_solver,
                    state=state,
                    state_keep_mask=state_keep_mask,
                )
            components = {
                "gt_fm": reduce_component(flow),
                "vla_fm": jnp.asarray(0.0, dtype=flow.dtype),
                "low_safety": jnp.asarray(0.0, dtype=flow.dtype),
                "decode": reduce_component(decode),
                "rank": jnp.asarray(0.0, dtype=flow.dtype),
                "repair": jnp.asarray(0.0, dtype=flow.dtype),
            }
        elif loss_mode == "predicted":
            flow = flow_matching_loss_per_sample(
                candidate,
                x_base,
                predicted_action,
                t,
                tactile_embeddings,
                state=state,
                state_keep_mask=state_keep_mask,
            )
            components = {
                "gt_fm": jnp.asarray(0.0, dtype=flow.dtype),
                "vla_fm": reduce_component(flow),
                "low_safety": jnp.asarray(0.0, dtype=flow.dtype),
                "decode": jnp.asarray(0.0, dtype=flow.dtype),
                "rank": jnp.asarray(0.0, dtype=flow.dtype),
                "repair": jnp.asarray(0.0, dtype=flow.dtype),
            }
        elif loss_mode == "gated":
            per_sample = gated_loss_components_per_sample(
                candidate,
                x_base,
                gt_action,
                predicted_action,
                t,
                tactile_embeddings,
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
                rank_low_gate_threshold=rank_low_gate_threshold,
                rank_high_gate_threshold=rank_high_gate_threshold,
                source_indices=(
                    source_indices
                    if source_balanced_loss or rank_requires_sources
                    else None
                ),
                num_sources=num_sources,
                high_gate_rank_aggregation=high_gate_rank_aggregation,
                high_gate_rank_hard_fraction=high_gate_rank_hard_fraction,
                high_gate_rank_worst_beta=high_gate_rank_worst_beta,
                source_rank_weights=source_rank_weights,
            )
            components = {
                name: reduce_component(per_sample[name]) for name in LOSS_COMPONENT_NAMES
            }
        else:
            raise ValueError(f"loss_mode must be 'gt', 'predicted', or 'gated', got {loss_mode!r}.")
        total = sum(components.values())
        return total, components

    (loss, components), gradients = nnx.value_and_grad(loss_fn, has_aux=True)(model)
    optimizer.update(model, gradients)
    return loss, components


@partial(nnx.jit, static_argnames=("num_steps",))
def decode_euler(
    model: TactileConditionedFlowDecoder,
    x_base: Array,
    tactile_condition: Array,
    *,
    num_steps: int,
) -> Array:
    if num_steps <= 0:
        raise ValueError(f"num_steps must be positive, got {num_steps}.")
    batch_size = x_base.shape[0]
    dt = jnp.asarray(1.0 / num_steps, dtype=jnp.float32)

    def body(step: int, x_t: Array) -> Array:
        t = jnp.full((batch_size,), step * dt, dtype=jnp.float32)
        return x_t + dt * model.velocity_from_condition(x_t, t, tactile_condition)

    return jax.lax.fori_loop(0, num_steps, body, jnp.asarray(x_base, dtype=jnp.float32))


@partial(nnx.jit, static_argnames=("num_steps",))
def decode_fireflow(
    model: TactileConditionedFlowDecoder,
    x_base: Array,
    tactile_condition: Array,
    *,
    num_steps: int,
) -> Array:
    return fireflow_integrate_velocity(
        lambda x, t: model.velocity_from_condition(x, t, tactile_condition),
        x_base,
        num_steps=num_steps,
    )


@partial(nnx.jit, static_argnames=("num_steps", "solver"))
def _decode_actions_jitted(
    model: TactileConditionedFlowDecoder,
    x_base: Array,
    tactile_seq: Array,
    state: Array | None,
    state_keep_mask: Array | None,
    *,
    num_steps: int,
    solver: FlowSolver,
) -> Array:
    """Single compiled unit: tactile conditioning + ODE integration."""

    tactile_condition = model.encode_condition(tactile_seq, state, state_keep_mask)
    if solver == "fireflow":
        return fireflow_integrate_velocity(
            lambda x, t: model.velocity_from_condition(x, t, tactile_condition),
            x_base,
            num_steps=num_steps,
        )

    batch_size = x_base.shape[0]
    dt = jnp.asarray(1.0 / num_steps, dtype=jnp.float32)

    def body(step: int, x_t: Array) -> Array:
        t = jnp.full((batch_size,), step * dt, dtype=jnp.float32)
        return x_t + dt * model.velocity_from_condition(x_t, t, tactile_condition)

    return jax.lax.fori_loop(0, num_steps, body, jnp.asarray(x_base, dtype=jnp.float32))


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
    if num_steps <= 0:
        raise ValueError(f"num_steps must be positive, got {num_steps}.")
    if solver not in ("euler", "fireflow"):
        raise ValueError(f"solver must be 'euler' or 'fireflow', got {solver!r}.")
    return _decode_actions_jitted(
        model,
        x_base,
        tactile_seq,
        state,
        state_keep_mask,
        num_steps=num_steps,
        solver=solver,
    )


@nnx.jit
def encode_tactile_embeddings(
    model: TactileConditionedFlowDecoder,
    tactile_input: Array,
) -> Array:
    """Encode raw tactile images once for reuse by evaluation objectives."""

    return model.encode_tactile_input(tactile_input)


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
        raise ValueError(f"min_learning_rate_ratio must be in [0, 1], got {min_learning_rate_ratio}.")

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
    transform = optax.chain(optax.clip_by_global_norm(grad_clip_norm), adamw) if grad_clip_norm is not None else adamw
    return nnx.Optimizer(model, transform, wrt=nnx.Param)
