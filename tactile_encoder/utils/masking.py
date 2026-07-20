from __future__ import annotations

import jax
import jax.numpy as jnp

Array = jax.Array


def validate_patch_mask_config(*, patch_size: int, mask_ratio: float) -> None:
    if patch_size <= 0:
        raise ValueError(f"patch_size must be positive, got {patch_size}.")
    if not 0.0 <= mask_ratio <= 1.0:
        raise ValueError(f"mask_ratio must be in [0, 1], got {mask_ratio}.")


def random_patch_zero(
    images: Array,
    key: Array,
    *,
    patch_size: int = 16,
    mask_ratio: float = 0.5,
) -> Array:
    """Randomly zero image patches in NHWC images.

    Images are expected to be in unit RGB space, before CLIP normalization.
    """

    validate_patch_mask_config(patch_size=patch_size, mask_ratio=mask_ratio)
    if images.ndim != 4:
        raise ValueError(f"Expected images with shape [B, H, W, C], got {images.shape}.")
    batch_size, height, width, channels = images.shape
    if height % patch_size or width % patch_size:
        raise ValueError(
            f"Image shape {(height, width)} must be divisible by patch_size={patch_size}."
        )

    grid_h = height // patch_size
    grid_w = width // patch_size
    keep = jax.random.uniform(key, (batch_size, grid_h, grid_w, 1)) >= mask_ratio
    mask = jnp.repeat(jnp.repeat(keep, patch_size, axis=1), patch_size, axis=2)
    mask = jnp.broadcast_to(mask, (batch_size, height, width, channels))
    return jnp.where(mask, images, jnp.zeros_like(images))

