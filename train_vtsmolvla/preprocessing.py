from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import numpy as np
from tactile_encoder.utils.image_dataset import parse_image_to_unit

from train_smolvla.preprocessing import JaxSmolVLAPreprocessor

from .configuration import VTSmolVLAConfig
from .tactile_cache import TACTILE_EMBEDDING_OBSERVATION_KEY


def prepare_tactile_batch(image: Any, image_size: int) -> np.ndarray:
    """Apply the tactile encoder's exact per-frame preprocessing to a batch."""

    image = np.asarray(image)
    if image.ndim == 3:
        image = image[None, ...]
    if image.ndim != 4:
        raise ValueError(f"expected a tactile image or batch, got {image.shape}")
    return np.stack(
        [parse_image_to_unit(frame, image_size=image_size) for frame in image],
        axis=0,
    ).astype(np.float32, copy=False)


class VTJaxSmolVLAPreprocessor(JaxSmolVLAPreprocessor):
    """Visual preprocessor extended with live or cached tactile tokens."""

    config: VTSmolVLAConfig

    def __init__(
        self,
        checkpoint: str | Path,
        config: VTSmolVLAConfig | None = None,
        **kwargs: Any,
    ):
        super().__init__(
            checkpoint,
            config or VTSmolVLAConfig.from_pretrained(checkpoint),
            **kwargs,
        )

    def prepare(
        self,
        observation: Mapping[str, Any],
        task: str | Sequence[str],
    ) -> dict[str, Any]:
        prepared = super().prepare(observation, task)
        if not self.config.use_tactile_encoder:
            return prepared

        renamed = {self.rename_map.get(key, key): value for key, value in observation.items()}
        cached = renamed.get(TACTILE_EMBEDDING_OBSERVATION_KEY)
        if cached is not None:
            tactile_embeddings = jnp.asarray(cached)
            if tactile_embeddings.ndim == 2:
                tactile_embeddings = tactile_embeddings[None, ...]
            expected_tail = (
                self.config.tactile_num_tokens,
                self.config.tactile_embedding_dim,
            )
            if tactile_embeddings.ndim != 3 or tactile_embeddings.shape[1:] != expected_tail:
                raise ValueError(
                    "cached tactile embeddings must have shape "
                    f"[B,{expected_tail[0]},{expected_tail[1]}], got "
                    f"{tactile_embeddings.shape}"
                )
            prepared["tactile_embeddings"] = tactile_embeddings
            prepared["tactile_masks"] = jnp.ones(
                tactile_embeddings.shape[:2], dtype=jnp.bool_
            )
            return prepared

        missing = [key for key in self.config.tactile_keys if key not in renamed]
        if missing:
            raise KeyError(f"missing tactile image keys: {missing}")
        tactile_images = [
            jnp.asarray(
                prepare_tactile_batch(renamed[key], self.config.tactile_image_size),
                dtype=jnp.float32,
            )
            for key in self.config.tactile_keys
        ]
        prepared["tactile_images"] = jnp.stack(tactile_images, axis=1)
        prepared["tactile_masks"] = jnp.stack(
            [jnp.ones((image.shape[0],), dtype=jnp.bool_) for image in tactile_images],
            axis=1,
        )
        return prepared
