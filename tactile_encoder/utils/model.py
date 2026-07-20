from __future__ import annotations

import dataclasses
import math
from typing import Any

import jax
import jax.numpy as jnp
import optax
from flax import linen as nn

from tactile_encoder.utils.clip_backend import CLIP_IMAGE_SIZE
from tactile_encoder.utils.clip_backend import DEFAULT_CLIP_MICROBATCH
from tactile_encoder.utils.clip_backend import encode_clip_images
from tactile_encoder.utils.masking import random_patch_zero
from tactile_encoder.utils.metrics import l2_normalize
from tactile_encoder.utils.resnet import encode_resnet18
from tactile_encoder.utils.resnet import init_resnet18_params

Array = jax.Array

DEFAULT_GRU_HIDDEN_DIM = 256


DEFAULT_CONTRASTIVE_TEMPERATURE = 0.07


@dataclasses.dataclass(frozen=True)
class TactileClipConfig:
    embedding_dim: int = 512
    tactile_image_count: int = 2
    tactile_history: int = 0
    gru_hidden_dim: int = DEFAULT_GRU_HIDDEN_DIM
    # Fixed InfoNCE temperature (not learnable).
    temperature: float = DEFAULT_CONTRASTIVE_TEMPERATURE
    tactile_image_size: int = CLIP_IMAGE_SIZE

    def __post_init__(self) -> None:
        if self.temperature <= 0.0:
            raise ValueError(f"temperature must be positive, got {self.temperature}.")

    @property
    def temporal_length(self) -> int:
        """Number of tactile timesteps including the current frame."""

        if self.tactile_history < 0:
            raise ValueError(f"tactile_history must be non-negative, got {self.tactile_history}.")
        return 1 + self.tactile_history

    @property
    def uses_gru(self) -> bool:
        return self.tactile_history > 0

    @property
    def projection_in_dim(self) -> int:
        """Vision embedding plus tactile features for the future projection."""

        if self.uses_gru:
            return self.embedding_dim + self.tactile_image_count * self.gru_hidden_dim
        return self.embedding_dim * (1 + self.tactile_image_count)

    @property
    def logit_scale(self) -> float:
        """Constant multiplicative scale ``1 / temperature`` for contrastive logits."""

        return 1.0 / self.temperature


def tactile_clip_config_from_dict(data: dict[str, Any]) -> TactileClipConfig:
    """Build config from checkpoint metadata, ignoring obsolete fields."""

    known = {field.name for field in dataclasses.fields(TactileClipConfig)}
    filtered = {key: value for key, value in data.items() if key in known}
    if "temperature" not in filtered and "logit_scale_init" in data:
        # Legacy learnable-temperature init stored ``log(1 / T)``.
        filtered["temperature"] = float(math.exp(-float(data["logit_scale_init"])))
    return TactileClipConfig(**filtered)


def _linear_init(key: Array, in_dim: int, out_dim: int) -> dict[str, Array]:
    limit = math.sqrt(6.0 / (in_dim + out_dim))
    kernel_key, _ = jax.random.split(key)
    return {
        "kernel": jax.random.uniform(kernel_key, (in_dim, out_dim), minval=-limit, maxval=limit),
        "bias": jnp.zeros((out_dim,), dtype=jnp.float32),
    }


def _linear(params: dict[str, Array], x: Array) -> Array:
    return x @ params["kernel"] + params["bias"]


class SharedTactileGRU(nn.Module):
    """Shared single-layer GRU over tactile embedding sequences."""

    hidden_dim: int = DEFAULT_GRU_HIDDEN_DIM

    @nn.compact
    def __call__(self, xs: Array) -> Array:
        """Return final hidden states for sequences shaped ``[B, T, D]``."""

        if xs.ndim != 3:
            raise ValueError(f"Expected GRU inputs [B, T, D], got {xs.shape}.")
        cell = nn.GRUCell(features=self.hidden_dim, name="cell")
        carry = jnp.zeros((xs.shape[0], self.hidden_dim), dtype=xs.dtype)

        def step(cell_module, carry_t, x_t):
            return cell_module(carry_t, x_t)

        scan = nn.scan(
            step,
            variable_broadcast="params",
            split_rngs={"params": False},
            in_axes=1,
            out_axes=1,
        )
        carry, _ = scan(cell, carry, xs)
        return carry


def init_gru_params(
    key: Array,
    *,
    input_dim: int,
    hidden_dim: int,
) -> dict[str, Any]:
    module = SharedTactileGRU(hidden_dim=hidden_dim)
    dummy = jnp.zeros((1, 1, input_dim), dtype=jnp.float32)
    return module.init(key, dummy)["params"]


def apply_gru(
    params: dict[str, Any],
    xs: Array,
    *,
    hidden_dim: int,
) -> Array:
    module = SharedTactileGRU(hidden_dim=hidden_dim)
    return module.apply({"params": params}, xs)


def init_trainable_params(
    key: Array,
    config: TactileClipConfig = TactileClipConfig(),
) -> dict[str, Any]:
    resnet_key, proj_key, gru_key = jax.random.split(key, 3)
    params: dict[str, Any] = {
        "tactile_resnet": init_resnet18_params(
            resnet_key,
            image_size=config.tactile_image_size,
            embedding_dim=config.embedding_dim,
        ),
        "future_projection": _linear_init(
            proj_key,
            config.projection_in_dim,
            config.embedding_dim,
        ),
    }
    if config.uses_gru:
        params["tactile_gru"] = init_gru_params(
            gru_key,
            input_dim=config.embedding_dim,
            hidden_dim=config.gru_hidden_dim,
        )
    return params


def encode_tactile_stack(
    tactile_resnet: dict[str, Any],
    tactile_images: Array,
    *,
    train: bool,
    embedding_dim: int,
) -> tuple[Array, dict[str, Any] | None]:
    """Encode tactile images shaped [B, N, H, W, C] with a shared ResNet18."""

    if tactile_images.ndim != 5:
        raise ValueError(f"Expected tactile_images [B, N, H, W, C], got {tactile_images.shape}.")
    batch_size, count = tactile_images.shape[:2]
    flattened = tactile_images.reshape((batch_size * count,) + tactile_images.shape[2:])
    embeddings, new_batch_stats = encode_resnet18(
        tactile_resnet,
        flattened,
        train=train,
        embedding_dim=embedding_dim,
    )
    return embeddings.reshape(batch_size, count, embeddings.shape[-1]), new_batch_stats


def encode_tactile_embedding(
    params: dict[str, Any],
    tactile_images: Array,
    *,
    train: bool,
    config: TactileClipConfig = TactileClipConfig(),
) -> tuple[Array, dict[str, Any] | None]:
    """Encode tactile inputs for the future-projection head.

    - ``tactile_history == 0``: ``[B, N, H, W, C]`` → flatten N ResNet embeddings
    - ``tactile_history  > 0``: ``[B, T, N, H, W, C]`` → shared ResNet + shared GRU
      over each sensor stream, returning ``[B, N * gru_hidden]``
    """

    images = jnp.asarray(tactile_images, dtype=jnp.float32)
    if not config.uses_gru:
        if images.ndim != 5:
            raise ValueError(f"Expected tactile [B, N, H, W, C] without history, got {images.shape}.")
        if images.shape[1] != config.tactile_image_count:
            raise ValueError(
                f"Expected {config.tactile_image_count} tactile sensors, got {images.shape[1]}."
            )
        tactile_tokens, new_batch_stats = encode_tactile_stack(
            params["tactile_resnet"],
            images,
            train=train,
            embedding_dim=config.embedding_dim,
        )
        return tactile_tokens.reshape(tactile_tokens.shape[0], -1), new_batch_stats

    if images.ndim != 6:
        raise ValueError(
            f"Expected tactile [B, T, N, H, W, C] with history, got {images.shape}."
        )
    batch_size, time_steps, sensor_count = images.shape[:3]
    if time_steps != config.temporal_length:
        raise ValueError(
            f"Expected temporal length {config.temporal_length}, got {time_steps}."
        )
    if sensor_count != config.tactile_image_count:
        raise ValueError(
            f"Expected {config.tactile_image_count} tactile sensors, got {sensor_count}."
        )
    if "tactile_gru" not in params:
        raise KeyError("Missing tactile_gru params for history encoding.")

    # Encode all (time, sensor) frames with the shared ResNet.
    flat_images = images.reshape((batch_size * time_steps * sensor_count,) + images.shape[3:])
    frame_embeddings, new_batch_stats = encode_resnet18(
        params["tactile_resnet"],
        flat_images,
        train=train,
        embedding_dim=config.embedding_dim,
    )
    # [B, T, N, D] -> [B, N, T, D] -> [B * N, T, D]
    frame_embeddings = frame_embeddings.reshape(
        batch_size, time_steps, sensor_count, config.embedding_dim
    )
    sequences = jnp.transpose(frame_embeddings, (0, 2, 1, 3)).reshape(
        batch_size * sensor_count, time_steps, config.embedding_dim
    )
    hidden = apply_gru(
        params["tactile_gru"],
        sequences,
        hidden_dim=config.gru_hidden_dim,
    )
    # [B * N, H] -> [B, N * H]
    hidden = hidden.reshape(batch_size, sensor_count * config.gru_hidden_dim)
    return hidden, new_batch_stats


def encode_frozen_rgb(
    clip_model: Any,
    frozen_clip_params: Any,
    batch: dict[str, Array],
    *,
    mask_key: Array | None,
    rgb_mask_patch_size: int,
    rgb_mask_ratio: float,
    microbatch_size: int = DEFAULT_CLIP_MICROBATCH,
) -> tuple[Array, Array]:
    """Encode current/future RGB with the frozen CLIP tower (no gradients)."""

    current_rgb = jnp.asarray(batch["current_rgb"], dtype=jnp.float32)
    if mask_key is not None:
        current_rgb = random_patch_zero(
            current_rgb,
            mask_key,
            patch_size=rgb_mask_patch_size,
            mask_ratio=rgb_mask_ratio,
        )
    v_current = encode_clip_images(
        clip_model,
        frozen_clip_params,
        current_rgb,
        train=False,
        microbatch_size=microbatch_size,
        remat=False,
    )
    v_future = encode_clip_images(
        clip_model,
        frozen_clip_params,
        batch["future_rgb"],
        train=False,
        microbatch_size=microbatch_size,
        remat=False,
    )
    return jax.lax.stop_gradient(v_current), jax.lax.stop_gradient(v_future)


def forward_embeddings(
    clip_model: Any,
    params: dict[str, Any],
    frozen_clip_params: Any,
    batch: dict[str, Array],
    *,
    train: bool,
    mask_key: Array | None = None,
    rgb_mask_patch_size: int = 16,
    rgb_mask_ratio: float = 0.5,
    config: TactileClipConfig = TactileClipConfig(),
    microbatch_size: int = DEFAULT_CLIP_MICROBATCH,
    v_current: Array | None = None,
    v_future: Array | None = None,
) -> tuple[dict[str, Array], dict[str, Any] | None]:
    if v_current is None or v_future is None:
        v_current, v_future = encode_frozen_rgb(
            clip_model,
            frozen_clip_params,
            batch,
            mask_key=mask_key,
            rgb_mask_patch_size=rgb_mask_patch_size,
            rgb_mask_ratio=rgb_mask_ratio,
            microbatch_size=microbatch_size,
        )

    tactile_embedding, new_batch_stats = encode_tactile_embedding(
        params,
        batch["tactile"],
        train=train,
        config=config,
    )
    query = l2_normalize(
        _linear(
            params["future_projection"],
            jnp.concatenate([v_current, tactile_embedding], axis=-1),
        )
    )
    return {
        "v_current": v_current,
        "tactile": tactile_embedding,
        "query": query,
        "future": v_future,
    }, new_batch_stats


def build_temporal_positive_mask(
    future_dataset_index: Array,
    episode_index: Array,
    *,
    side_id: Array | None = None,
    window: int,
) -> Array:
    """Mark gallery items within a temporal window as additional positives.

    Positives share the episode and have ``future_dataset_index`` within
    ``[target - window, target + window]``. When ``side_id`` is provided,
    positives must also come from the same wrist side. The identity diagonal is
    always marked positive. When ``window <= 0``, returns the identity only.
    """

    if window < 0:
        raise ValueError(f"window must be non-negative, got {window}.")
    future = jnp.asarray(future_dataset_index, dtype=jnp.int32)
    episode = jnp.asarray(episode_index, dtype=jnp.int32)
    batch_size = future.shape[0]
    identity = jnp.eye(batch_size, dtype=bool)
    if window == 0:
        return identity
    same_episode = episode[:, None] == episode[None, :]
    within_window = jnp.abs(future[:, None] - future[None, :]) <= window
    mask = same_episode & within_window
    if side_id is not None:
        side = jnp.asarray(side_id, dtype=jnp.int32)
        mask = mask & (side[:, None] == side[None, :])
    return mask | identity


def _multi_positive_cross_entropy(logits: Array, positive_mask: Array) -> Array:
    """Softmax cross-entropy with one or more positive labels per row."""

    neg_inf = jnp.array(-jnp.inf, dtype=logits.dtype)
    positive_logits = jnp.where(positive_mask, logits, neg_inf)
    log_normalizer = jax.nn.logsumexp(logits, axis=1)
    log_positive = jax.nn.logsumexp(positive_logits, axis=1)
    return -(log_positive - log_normalizer)


def init_memory_bank(
    size: int,
    embedding_dim: int,
    *,
    dtype: jnp.dtype = jnp.float32,
) -> dict[str, Array]:
    """Create an empty circular queue of target (future RGB) embeddings."""

    if size < 0:
        raise ValueError(f"memory bank size must be non-negative, got {size}.")
    if embedding_dim <= 0:
        raise ValueError(f"embedding_dim must be positive, got {embedding_dim}.")
    if size == 0:
        return {
            "keys": jnp.zeros((0, embedding_dim), dtype=dtype),
            "future_dataset_index": jnp.zeros((0,), dtype=jnp.int32),
            "episode_index": jnp.zeros((0,), dtype=jnp.int32),
            "side_id": jnp.zeros((0,), dtype=jnp.int32),
            "valid": jnp.zeros((0,), dtype=bool),
            "ptr": jnp.zeros((), dtype=jnp.int32),
        }
    return {
        "keys": jnp.zeros((size, embedding_dim), dtype=dtype),
        "future_dataset_index": jnp.full((size,), -1, dtype=jnp.int32),
        "episode_index": jnp.full((size,), -1, dtype=jnp.int32),
        "side_id": jnp.full((size,), -1, dtype=jnp.int32),
        "valid": jnp.zeros((size,), dtype=bool),
        "ptr": jnp.zeros((), dtype=jnp.int32),
    }


def enqueue_memory_bank(
    bank: dict[str, Array],
    keys: Array,
    *,
    future_dataset_index: Array,
    episode_index: Array,
    side_id: Array,
) -> dict[str, Array]:
    """Push stop-grad keys into the circular memory bank."""

    queue_size = int(bank["keys"].shape[0])
    if queue_size == 0:
        return bank

    keys = l2_normalize(jax.lax.stop_gradient(jnp.asarray(keys, dtype=bank["keys"].dtype)))
    future_dataset_index = jnp.asarray(future_dataset_index, dtype=jnp.int32)
    episode_index = jnp.asarray(episode_index, dtype=jnp.int32)
    side_id = jnp.asarray(side_id, dtype=jnp.int32)
    batch_size = keys.shape[0]

    def _write_one(i: Array, state: dict[str, Array]) -> dict[str, Array]:
        ptr = state["ptr"]
        idx = ptr % queue_size
        return {
            "keys": state["keys"].at[idx].set(keys[i]),
            "future_dataset_index": state["future_dataset_index"].at[idx].set(future_dataset_index[i]),
            "episode_index": state["episode_index"].at[idx].set(episode_index[i]),
            "side_id": state["side_id"].at[idx].set(side_id[i]),
            "valid": state["valid"].at[idx].set(True),
            "ptr": (ptr + 1) % queue_size,
        }

    return jax.lax.fori_loop(0, batch_size, _write_one, bank)


def build_bank_positive_mask(
    future_dataset_index: Array,
    episode_index: Array,
    side_id: Array,
    bank: dict[str, Array],
    *,
    window: int,
) -> Array:
    """Positives in the bank: exact key matches, plus temporal neighbors when window > 0."""

    if int(bank["keys"].shape[0]) == 0:
        batch_size = jnp.asarray(future_dataset_index).shape[0]
        return jnp.zeros((batch_size, 0), dtype=bool)
    if window < 0:
        raise ValueError(f"window must be non-negative, got {window}.")

    future = jnp.asarray(future_dataset_index, dtype=jnp.int32)
    episode = jnp.asarray(episode_index, dtype=jnp.int32)
    side = jnp.asarray(side_id, dtype=jnp.int32)
    same_episode = episode[:, None] == bank["episode_index"][None, :]
    same_side = side[:, None] == bank["side_id"][None, :]
    exact = future[:, None] == bank["future_dataset_index"][None, :]
    if window > 0:
        within_window = jnp.abs(future[:, None] - bank["future_dataset_index"][None, :]) <= window
        temporal = within_window
    else:
        temporal = exact
    return same_episode & same_side & temporal & bank["valid"][None, :]


def _filter_bank_logits_hard_negatives(
    logits_bank: Array,
    *,
    bank_positive_mask: Array,
    bank_valid: Array,
    candidate_mask: Array,
    hard_negatives_k: int,
) -> tuple[Array, Array]:
    """Keep bank positives plus top-K hardest eligible bank negatives.

    Invalid slots are always ``-inf``. When ``hard_negatives_k <= 0``,
    all valid bank logits remain in the denominator.
    """

    neg_inf = jnp.asarray(-jnp.inf, dtype=logits_bank.dtype)
    batch_size, queue_size = logits_bank.shape
    base_keep = bank_valid[None, :]
    if hard_negatives_k <= 0 or queue_size == 0:
        hard_logits = jnp.full((batch_size, 0), neg_inf, dtype=logits_bank.dtype)
        filtered = jnp.where(base_keep, logits_bank, neg_inf)
        return filtered, hard_logits

    k = min(int(hard_negatives_k), int(queue_size))
    candidate = jnp.where(candidate_mask & base_keep & (~bank_positive_mask), logits_bank, neg_inf)
    hard_logits, hard_idx = jax.lax.top_k(candidate, k)
    hard_mask = (
        jnp.zeros((batch_size, queue_size), dtype=bool)
        .at[jnp.arange(batch_size)[:, None], hard_idx]
        .set(True)
    )
    hard_mask = hard_mask & jnp.isfinite(logits_bank) & candidate_mask & base_keep
    keep = (bank_positive_mask | hard_mask) & base_keep
    filtered = jnp.where(keep, logits_bank, neg_inf)
    return filtered, hard_logits


def symmetric_contrastive_loss(
    query: Array,
    target: Array,
    *,
    config: TactileClipConfig = TactileClipConfig(),
    positive_mask: Array | None = None,
    memory_bank: dict[str, Array] | None = None,
    bank_positive_mask: Array | None = None,
    hard_negatives_k: int = 0,
    query_episode_index: Array | None = None,
) -> tuple[Array, dict[str, Array]]:
    if hard_negatives_k < 0:
        raise ValueError(f"hard_negatives_k must be non-negative, got {hard_negatives_k}.")

    query = l2_normalize(query)
    target = l2_normalize(target)
    scale = jnp.asarray(config.logit_scale, dtype=jnp.float32)
    neg_inf = jnp.asarray(-jnp.inf, dtype=query.dtype)
    logits_batch = scale * (query @ target.T)
    if positive_mask is None:
        positive_mask = jnp.eye(logits_batch.shape[0], dtype=bool)
    else:
        positive_mask = jnp.asarray(positive_mask, dtype=bool)

    use_bank = memory_bank is not None and int(memory_bank["keys"].shape[0]) > 0
    bank_filled_frac = jnp.asarray(0.0, dtype=logits_batch.dtype)
    hard_neg_logit_mean = jnp.asarray(0.0, dtype=logits_batch.dtype)
    batch_vs_positive_gap = jnp.asarray(0.0, dtype=logits_batch.dtype)

    if use_bank:
        assert memory_bank is not None
        if query_episode_index is None:
            raise ValueError("query_episode_index is required when using a memory bank.")
        logits_bank = scale * (query @ memory_bank["keys"].T)
        logits_bank = jnp.where(memory_bank["valid"][None, :], logits_bank, neg_inf)
        if bank_positive_mask is None:
            bank_positive_mask = jnp.zeros(
                (logits_batch.shape[0], memory_bank["keys"].shape[0]),
                dtype=bool,
            )
        else:
            bank_positive_mask = jnp.asarray(bank_positive_mask, dtype=bool)

        # Bank negatives are always mined from other episodes only.
        query_episode = jnp.asarray(query_episode_index, dtype=jnp.int32)
        candidate_mask = query_episode[:, None] != memory_bank["episode_index"][None, :]

        logits_bank_for_loss, hard_logits = _filter_bank_logits_hard_negatives(
            logits_bank,
            bank_positive_mask=bank_positive_mask,
            bank_valid=memory_bank["valid"],
            candidate_mask=candidate_mask,
            hard_negatives_k=hard_negatives_k,
        )
        logits_query = jnp.concatenate([logits_batch, logits_bank_for_loss], axis=1)
        positive_query = jnp.concatenate([positive_mask, bank_positive_mask], axis=1)
        loss_query = _multi_positive_cross_entropy(logits_query, positive_query)
        loss_target = _multi_positive_cross_entropy(logits_batch.T, positive_mask.T)

        bank_filled_frac = jnp.mean(memory_bank["valid"].astype(logits_batch.dtype))
        if hard_negatives_k > 0:
            hard_finite = jnp.isfinite(hard_logits)
            hard_neg_logit_mean = jnp.where(
                jnp.any(hard_finite),
                jnp.sum(jnp.where(hard_finite, hard_logits, 0.0))
                / jnp.maximum(jnp.sum(hard_finite.astype(logits_batch.dtype)), 1.0),
                jnp.asarray(0.0, dtype=logits_batch.dtype),
            )
        else:
            neg_mask = (
                candidate_mask
                & (~bank_positive_mask)
                & memory_bank["valid"][None, :]
                & jnp.isfinite(logits_bank)
            )
            hard_neg_logit_mean = jnp.where(
                jnp.any(neg_mask),
                jnp.sum(jnp.where(neg_mask, logits_bank, 0.0))
                / jnp.maximum(jnp.sum(neg_mask.astype(logits_batch.dtype)), 1.0),
                jnp.asarray(0.0, dtype=logits_batch.dtype),
            )
        positive_logit = jnp.diag(logits_batch)
        positive_logit = jnp.where(jnp.isfinite(positive_logit), positive_logit, 0.0)
        batch_vs_positive_gap = jnp.mean(positive_logit - hard_neg_logit_mean)
    else:
        loss_query = _multi_positive_cross_entropy(logits_batch, positive_mask)
        loss_target = _multi_positive_cross_entropy(logits_batch.T, positive_mask.T)

    loss = 0.5 * (jnp.mean(loss_query) + jnp.mean(loss_target))
    diag = jnp.diag(logits_batch)[:, None]
    ranks = jnp.sum(logits_batch > diag, axis=1)
    metrics = {
        "loss": loss,
        "temperature": jnp.asarray(config.temperature, dtype=logits_batch.dtype),
        "logit_scale": scale,
        "batch_recall_at_1": jnp.mean(ranks < 1),
        "batch_recall_at_5": jnp.mean(ranks < 5),
        "batch_mean_rank": jnp.mean(ranks + 1),
        "bank_filled_frac": bank_filled_frac,
        "bank_hard_neg_logit_mean": hard_neg_logit_mean,
        "batch_vs_positive_gap": batch_vs_positive_gap,
    }
    return loss, metrics


def _batch_side_id(batch: dict[str, Array]) -> Array:
    side_id = batch.get("side_id")
    if side_id is None:
        return jnp.zeros((batch["future_dataset_index"].shape[0],), dtype=jnp.int32)
    return jnp.asarray(side_id, dtype=jnp.int32)


def _contrastive_masks_from_batch(
    batch: dict[str, Array],
    memory_bank: dict[str, Array] | None,
    *,
    positive_temporal_window: int,
) -> tuple[Array, Array | None]:
    side_id = _batch_side_id(batch)
    positive_mask = build_temporal_positive_mask(
        batch["future_dataset_index"],
        batch["episode_index"],
        side_id=side_id,
        window=positive_temporal_window,
    )
    bank_positive = None
    if memory_bank is not None and int(memory_bank["keys"].shape[0]) > 0:
        bank_positive = build_bank_positive_mask(
            batch["future_dataset_index"],
            batch["episode_index"],
            side_id,
            memory_bank,
            window=positive_temporal_window,
        )
    return positive_mask, bank_positive


def loss_fn(
    clip_model: Any,
    params: dict[str, Any],
    frozen_clip_params: Any,
    batch: dict[str, Array],
    *,
    train: bool,
    mask_key: Array | None,
    rgb_mask_patch_size: int,
    rgb_mask_ratio: float,
    config: TactileClipConfig = TactileClipConfig(),
    microbatch_size: int = DEFAULT_CLIP_MICROBATCH,
    positive_temporal_window: int = 0,
    memory_bank: dict[str, Array] | None = None,
    hard_negatives_k: int = 0,
) -> tuple[Array, dict[str, Array]]:
    embeddings, _ = forward_embeddings(
        clip_model,
        params,
        frozen_clip_params,
        batch,
        train=train,
        mask_key=mask_key,
        rgb_mask_patch_size=rgb_mask_patch_size,
        rgb_mask_ratio=rgb_mask_ratio,
        config=config,
        microbatch_size=microbatch_size,
    )
    positive_mask, bank_positive = _contrastive_masks_from_batch(
        batch,
        memory_bank,
        positive_temporal_window=positive_temporal_window,
    )
    return symmetric_contrastive_loss(
        embeddings["query"],
        embeddings["future"],
        config=config,
        positive_mask=positive_mask,
        memory_bank=memory_bank,
        bank_positive_mask=bank_positive,
        hard_negatives_k=hard_negatives_k,
        query_episode_index=batch["episode_index"],
    )


def _zero_batch_stats_grads(gradients: dict[str, Any]) -> dict[str, Any]:
    """Drop optimizer updates for BatchNorm running statistics."""

    tactile = gradients.get("tactile_resnet")
    if tactile is None or "batch_stats" not in tactile:
        return gradients
    return {
        **gradients,
        "tactile_resnet": {
            **tactile,
            "batch_stats": jax.tree_util.tree_map(jnp.zeros_like, tactile["batch_stats"]),
        },
    }


def make_train_step(
    clip_model: Any,
    optimizer: optax.GradientTransformation,
    *,
    rgb_mask_patch_size: int,
    rgb_mask_ratio: float,
    config: TactileClipConfig = TactileClipConfig(),
    microbatch_size: int = DEFAULT_CLIP_MICROBATCH,
    positive_temporal_window: int = 0,
    memory_bank_size: int = 0,
    hard_negatives_k: int = 0,
):
    if hard_negatives_k < 0:
        raise ValueError(f"hard_negatives_k must be non-negative, got {hard_negatives_k}.")
    if positive_temporal_window < 0:
        raise ValueError(
            f"positive_temporal_window must be non-negative, got {positive_temporal_window}."
        )
    # Capture a concrete K for XLA; never exceed the queue length.
    effective_hard_k = 0 if memory_bank_size <= 0 else min(int(hard_negatives_k), int(memory_bank_size))

    def step(
        params: dict[str, Any],
        opt_state: optax.OptState,
        frozen_clip_params: Any,
        batch: dict[str, Array],
        key: Array,
        memory_bank: dict[str, Array],
    ):
        # Frozen RGB tower runs outside the trainable value_and_grad tape so its
        # activations are never stored for backprop through the tactile ResNet.
        v_current, v_future = encode_frozen_rgb(
            clip_model,
            frozen_clip_params,
            batch,
            mask_key=key,
            rgb_mask_patch_size=rgb_mask_patch_size,
            rgb_mask_ratio=rgb_mask_ratio,
            microbatch_size=microbatch_size,
        )
        active_bank = memory_bank if memory_bank_size > 0 else None
        positive_mask, bank_positive = _contrastive_masks_from_batch(
            batch,
            active_bank,
            positive_temporal_window=positive_temporal_window,
        )

        def wrapped(candidate_params: dict[str, Any]) -> tuple[Array, tuple[dict[str, Array], dict[str, Any]]]:
            embeddings, new_batch_stats = forward_embeddings(
                clip_model,
                candidate_params,
                frozen_clip_params,
                batch,
                train=True,
                mask_key=None,
                rgb_mask_patch_size=rgb_mask_patch_size,
                rgb_mask_ratio=rgb_mask_ratio,
                config=config,
                microbatch_size=microbatch_size,
                v_current=v_current,
                v_future=v_future,
            )
            loss, metrics = symmetric_contrastive_loss(
                embeddings["query"],
                embeddings["future"],
                config=config,
                positive_mask=positive_mask,
                memory_bank=active_bank,
                bank_positive_mask=bank_positive,
                hard_negatives_k=effective_hard_k,
                query_episode_index=batch["episode_index"],
            )
            if new_batch_stats is None:
                raise RuntimeError("Expected BatchNorm batch_stats updates during training.")
            return loss, (metrics, new_batch_stats)

        (loss, (metrics, new_batch_stats)), gradients = jax.value_and_grad(wrapped, has_aux=True)(params)
        gradients = _zero_batch_stats_grads(gradients)
        updates, opt_state_next = optimizer.update(gradients, opt_state, params)
        params_next = optax.apply_updates(params, updates)
        params_next = {
            **params_next,
            "tactile_resnet": {
                **params_next["tactile_resnet"],
                "batch_stats": new_batch_stats,
            },
        }
        if memory_bank_size > 0:
            memory_bank_next = enqueue_memory_bank(
                memory_bank,
                v_future,
                future_dataset_index=batch["future_dataset_index"],
                episode_index=batch["episode_index"],
                side_id=_batch_side_id(batch),
            )
        else:
            memory_bank_next = memory_bank
        return params_next, opt_state_next, memory_bank_next, loss, metrics


    # Donate params/opt_state/memory_bank buffers so the next step can reuse their memory.
    return jax.jit(step, donate_argnums=(0, 1, 5))
