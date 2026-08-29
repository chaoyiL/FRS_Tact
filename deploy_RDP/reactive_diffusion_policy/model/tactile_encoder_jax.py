"""Minimal JAX tactile ResNet loader used by the pick-tube cache converter."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn
from flax import traverse_util


@dataclasses.dataclass(frozen=True)
class TactileEncoderBundle:
    params: dict[str, Any]
    metadata: dict[str, Any]


class ResNetBlock(nn.Module):
    filters: int
    strides: tuple[int, int] = (1, 1)

    @nn.compact
    def __call__(self, x: jax.Array, *, train: bool) -> jax.Array:
        residual = x
        x = nn.Conv(self.filters, (3, 3), self.strides, padding="SAME", use_bias=False, name="conv1")(x)
        x = nn.BatchNorm(use_running_average=not train, name="bn1")(x)
        x = nn.relu(x)
        x = nn.Conv(self.filters, (3, 3), padding="SAME", use_bias=False, name="conv2")(x)
        x = nn.BatchNorm(use_running_average=not train, name="bn2")(x)
        if residual.shape != x.shape:
            residual = nn.Conv(
                self.filters, (1, 1), self.strides, padding="SAME", use_bias=False, name="proj_conv"
            )(residual)
            residual = nn.BatchNorm(use_running_average=not train, name="proj_bn")(residual)
        return nn.relu(x + residual)


class ResNet18(nn.Module):
    embedding_dim: int = 512

    @nn.compact
    def __call__(self, x: jax.Array, *, train: bool) -> jax.Array:
        x = jnp.asarray(x, dtype=jnp.float32)
        x = nn.Conv(64, (7, 7), (2, 2), padding="SAME", use_bias=False, name="conv1")(x)
        x = nn.BatchNorm(use_running_average=not train, name="bn1")(x)
        x = nn.relu(x)
        x = nn.max_pool(x, (3, 3), strides=(2, 2), padding="SAME")
        for block_id, (filters, blocks, stride) in enumerate(
            ((64, 2, 1), (128, 2, 2), (256, 2, 2), (512, 2, 2))
        ):
            for block_index in range(blocks):
                x = ResNetBlock(
                    filters,
                    strides=(stride, stride) if block_index == 0 else (1, 1),
                    name=f"block{block_id + 1}_{block_index}",
                )(x, train=train)
        x = jnp.mean(x, axis=(1, 2))
        if self.embedding_dim != 512:
            x = nn.Dense(self.embedding_dim, name="embedding")(x)
        return x


def encode_resnet18(
    variables: dict[str, Any],
    images: jax.Array,
    *,
    train: bool,
    embedding_dim: int = 512,
) -> tuple[jax.Array, dict[str, Any] | None]:
    model = ResNet18(embedding_dim=embedding_dim)
    apply_variables = {
        "params": variables["params"],
        "batch_stats": variables["batch_stats"],
    }
    if train:
        embeddings, updates = model.apply(
            apply_variables, images, train=True, mutable=["batch_stats"]
        )
        return jnp.asarray(embeddings, dtype=jnp.float32), updates["batch_stats"]
    embeddings = model.apply(apply_variables, images, train=False, mutable=False)
    return jnp.asarray(embeddings, dtype=jnp.float32), None


def load_tactile_encoder(checkpoint_dir: str | Path) -> TactileEncoderBundle:
    checkpoint_dir = Path(checkpoint_dir)
    with (checkpoint_dir / "checkpoint.json").open(encoding="utf-8") as file:
        metadata = json.load(file)
    params_path = checkpoint_dir / str(metadata["params_file"])
    restored = {}
    with np.load(params_path) as archive:
        for index, path_name in enumerate(metadata["parameter_paths"]):
            restored[tuple(path_name.split("/"))] = jnp.asarray(archive[f"p{index:05d}"])
    params = traverse_util.unflatten_dict(restored)
    return TactileEncoderBundle(params=params, metadata=metadata)
