from __future__ import annotations

import dataclasses
from typing import Any

import jax
import jax.numpy as jnp

from tactile_encoder.utils.metrics import l2_normalize

Array = jax.Array

DEFAULT_CLIP_MODEL_ID = "openai/clip-vit-base-patch16"
CLIP_IMAGE_SIZE = 224
CLIP_MEAN = jnp.asarray([0.48145466, 0.4578275, 0.40821073], dtype=jnp.float32)
CLIP_STD = jnp.asarray([0.26862954, 0.26130258, 0.27577711], dtype=jnp.float32)

# Encode this many frozen RGB images at a time through CLIP to bound peak memory.
DEFAULT_CLIP_MICROBATCH = 512


@dataclasses.dataclass(frozen=True)
class ClipBackend:
    """Thin wrapper around transformers' Flax CLIP model (frozen RGB tower only)."""

    model: Any
    params: Any
    model_id: str = DEFAULT_CLIP_MODEL_ID

    @classmethod
    def from_pretrained(cls, model_id: str = DEFAULT_CLIP_MODEL_ID) -> "ClipBackend":
        try:
            from transformers import FlaxCLIPModel
        except ImportError as exc:
            raise RuntimeError(
                "transformers with Flax CLIP support is required for frozen RGB encoding."
            ) from exc
        model = FlaxCLIPModel.from_pretrained(model_id)
        return cls(model=model, params=model.params, model_id=model_id)


def clip_pixel_values(unit_images: Array) -> Array:
    """Convert NHWC unit RGB images to CLIP NCHW normalized pixel values."""

    if unit_images.ndim != 4:
        raise ValueError(f"Expected unit_images [B, H, W, C], got {unit_images.shape}.")
    if unit_images.shape[-1] != 3:
        raise ValueError(f"Expected RGB images with 3 channels, got {unit_images.shape[-1]}.")
    images = jnp.asarray(unit_images, dtype=jnp.float32)
    images = (images - CLIP_MEAN[None, None, None, :]) / CLIP_STD[None, None, None, :]
    return jnp.transpose(images, (0, 3, 1, 2))


def encode_clip_images(
    clip_model: Any,
    params: Any,
    unit_images: Array,
    *,
    train: bool = False,
    microbatch_size: int = DEFAULT_CLIP_MICROBATCH,
    remat: bool = True,
) -> Array:
    """Encode NHWC unit RGB images into normalized CLIP image embeddings.

    Used for the frozen RGB tower. Images are processed in micro-batches. When
    ``remat`` is True, activations are recomputed in the backward pass instead of
    being stored (unused when ``train=False``).
    """

    images = jnp.asarray(unit_images, dtype=jnp.float32)
    if images.ndim != 4:
        raise ValueError(f"Expected unit_images [B, H, W, C], got {images.shape}.")
    count = images.shape[0]
    if count == 0:
        return jnp.zeros((0, 512), dtype=jnp.float32)

    def encode_chunk(chunk_params: Any, chunk_images: Array) -> Array:
        pixel_values = clip_pixel_values(chunk_images)
        features = clip_model.get_image_features(
            pixel_values=pixel_values,
            params=chunk_params,
            train=train,
        )
        return l2_normalize(jnp.asarray(features, dtype=jnp.float32))

    if remat:
        encode_chunk = jax.checkpoint(encode_chunk, prevent_cse=False)

    chunk = max(1, int(microbatch_size))
    # Concrete batch size is known at trace time for each JIT specialization.
    concrete_count = int(count)
    if concrete_count <= chunk:
        return encode_chunk(params, images)

    padded = ((concrete_count + chunk - 1) // chunk) * chunk
    if padded != concrete_count:
        pad = jnp.zeros((padded - concrete_count,) + images.shape[1:], dtype=images.dtype)
        images = jnp.concatenate([images, pad], axis=0)

    chunks = images.reshape((-1, chunk) + images.shape[1:])

    def body(chunk_images: Array) -> Array:
        return encode_chunk(params, chunk_images)

    encoded = jax.lax.map(body, chunks)
    encoded = encoded.reshape((padded, encoded.shape[-1]))
    return encoded[:concrete_count]
