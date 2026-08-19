"""Flax ResNet18 backbone for tactile image encoding."""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
from flax import linen as nn

Array = jax.Array

RESNET18_FEATURE_DIM = 512


class ResNetBlock(nn.Module):
    """Basic residual block (no bottleneck)."""

    filters: int
    strides: tuple[int, int] = (1, 1)

    @nn.compact
    def __call__(self, x: Array, *, train: bool) -> Array:
        residual = x
        y = nn.Conv(self.filters, (3, 3), self.strides, padding="SAME", use_bias=False, name="conv1")(x)
        y = nn.BatchNorm(use_running_average=not train, name="bn1")(y)
        y = nn.relu(y)
        y = nn.Conv(self.filters, (3, 3), padding="SAME", use_bias=False, name="conv2")(y)
        y = nn.BatchNorm(use_running_average=not train, name="bn2")(y)
        if residual.shape != y.shape:
            residual = nn.Conv(
                self.filters,
                (1, 1),
                self.strides,
                padding="SAME",
                use_bias=False,
                name="proj_conv",
            )(residual)
            residual = nn.BatchNorm(use_running_average=not train, name="proj_bn")(residual)
        return nn.relu(y + residual)


class ResNet18(nn.Module):
    """Standard ResNet18 producing a fixed-size embedding (NHWC input)."""

    embedding_dim: int = RESNET18_FEATURE_DIM

    @nn.compact
    def __call__(self, x: Array, *, train: bool) -> Array:
        if x.ndim != 4 or x.shape[-1] != 3:
            raise ValueError(f"Expected images [B, H, W, 3], got {x.shape}.")
        x = jnp.asarray(x, dtype=jnp.float32)
        x = nn.Conv(64, (7, 7), (2, 2), padding="SAME", use_bias=False, name="conv1")(x)
        x = nn.BatchNorm(use_running_average=not train, name="bn1")(x)
        x = nn.relu(x)
        x = nn.max_pool(x, (3, 3), strides=(2, 2), padding="SAME")

        for block_id, (filters, blocks, stride) in enumerate(
            (
                (64, 2, 1),
                (128, 2, 2),
                (256, 2, 2),
                (512, 2, 2),
            )
        ):
            for i in range(blocks):
                x = ResNetBlock(
                    filters,
                    strides=(stride, stride) if i == 0 else (1, 1),
                    name=f"block{block_id + 1}_{i}",
                )(x, train=train)

        # Global average pool over spatial dims -> [B, 512].
        x = jnp.mean(x, axis=(1, 2))
        if self.embedding_dim != RESNET18_FEATURE_DIM:
            x = nn.Dense(self.embedding_dim, name="embedding")(x)
        return x


def _module(embedding_dim: int = RESNET18_FEATURE_DIM) -> ResNet18:
    return ResNet18(embedding_dim=embedding_dim)


def init_resnet18_params(
    key: Array,
    *,
    image_size: int = 224,
    embedding_dim: int = RESNET18_FEATURE_DIM,
) -> dict[str, Any]:
    """Initialize ResNet18 variables as ``{"params": ..., "batch_stats": ...}``."""

    model = _module(embedding_dim)
    dummy = jnp.zeros((1, image_size, image_size, 3), dtype=jnp.float32)
    variables = model.init(key, dummy, train=False)
    return {
        "params": variables["params"],
        "batch_stats": variables.get("batch_stats", {}),
    }


def encode_resnet18(
    variables: dict[str, Any],
    images: Array,
    *,
    train: bool,
    embedding_dim: int = RESNET18_FEATURE_DIM,
) -> tuple[Array, dict[str, Any] | None]:
    """Encode NHWC images.

    Returns ``(embeddings, new_batch_stats)``. When ``train`` is False,
    ``new_batch_stats`` is None and running averages are used.
    """

    model = _module(embedding_dim)
    apply_vars = {
        "params": variables["params"],
        "batch_stats": variables["batch_stats"],
    }
    if train:
        embeddings, updates = model.apply(
            apply_vars,
            images,
            train=True,
            mutable=["batch_stats"],
        )
        return jnp.asarray(embeddings, dtype=jnp.float32), updates["batch_stats"]
    embeddings = model.apply(apply_vars, images, train=False, mutable=False)
    return jnp.asarray(embeddings, dtype=jnp.float32), None
