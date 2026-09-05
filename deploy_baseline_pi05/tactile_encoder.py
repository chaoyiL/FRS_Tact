"""Frozen 0824 tactile ResNet encoder for current tactile observations."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from .deployment import RIGHT_TACTILE_KEYS, TACTILE_KEYS
from .tactile_runtime.encoder_checkpoint import load_tactile_encoder
from .tactile_runtime.encoder_config import tactile_clip_config_from_dict
from .tactile_runtime.preprocess import parse_image_to_unit
from .tactile_runtime.resnet import encode_resnet18


def _rms_normalize_tokens(embeddings: np.ndarray) -> np.ndarray:
    """Match training-cache per-token RMS normalization and reject zero tokens."""
    rms = np.sqrt(np.mean(np.square(embeddings), axis=-1, keepdims=True))
    if not np.isfinite(rms).all() or np.any(rms == 0.0):
        raise ValueError("tactile encoder produced a zero-RMS embedding token")
    return embeddings / rms


class FrozenTactileEncoder:
    """Encode the canonical current HWC tactile images with frozen inference BN."""

    def __init__(
        self, checkpoint: str | Path, *, tactile_keys: tuple[str, ...] = TACTILE_KEYS,
        key_map: Mapping[str, str] | None = None,
    ) -> None:
        if tuple(tactile_keys) not in (TACTILE_KEYS, RIGHT_TACTILE_KEYS):
            raise ValueError("tactile encoder requires canonical right or bimanual tactile keys")
        mapping = dict(key_map) if key_map is not None else {key: key for key in tactile_keys}
        if set(mapping) != set(tactile_keys) or any(not isinstance(key, str) or not key for key in mapping.values()):
            raise ValueError("tactile key_map must map every model tactile key to a wire key")
        if len(set(mapping.values())) != len(mapping):
            raise ValueError("tactile key_map wire keys must be distinct")
        self.key_map = mapping
        bundle = load_tactile_encoder(Path(checkpoint))
        if "tactile_resnet" not in bundle.params:
            raise ValueError("0824 tactile encoder checkpoint is missing tactile_resnet")
        raw_config = bundle.metadata.get("tactile_clip_config", {})
        if not isinstance(raw_config, dict):
            raise ValueError("tactile encoder checkpoint tactile_clip_config must be a mapping")
        config = tactile_clip_config_from_dict(raw_config)
        if config.embedding_dim != 512 or config.tactile_image_size != 224:
            raise ValueError("0824 tactile encoder must use 512D embeddings and 224px images")
        self.tactile_keys = tuple(tactile_keys)
        self.image_size = config.tactile_image_size
        self._variables = bundle.params["tactile_resnet"]

        def encode(variables: Any, images: jax.Array) -> jax.Array:
            embeddings, _ = encode_resnet18(variables, images, train=False, embedding_dim=512)
            return embeddings

        self._encode = jax.jit(encode)

    def _prepare_image(self, value: Any) -> np.ndarray:
        raw = np.asarray(value)
        if raw.ndim != 3 or raw.shape[-1] != 3:
            raise ValueError(f"tactile image must be a current HWC RGB image, got {raw.shape}")
        # Training converts uint8 to unit floats before interpolation.
        if raw.dtype == np.uint8:
            raw = raw.astype(np.float32) / np.float32(255)
        prepared = parse_image_to_unit(raw, image_size=self.image_size)
        if prepared.shape != (224, 224, 3) or not np.isfinite(prepared).all():
            raise ValueError("preprocessed tactile image must be finite HWC [224,224,3]")
        return np.ascontiguousarray(prepared, dtype=np.float32)

    def encode(self, observation: Mapping[str, Any]) -> np.ndarray:
        wire_keys = tuple(self.key_map[key] for key in self.tactile_keys)
        missing = [key for key in wire_keys if key not in observation]
        if missing:
            raise ValueError(f"observation is missing tactile keys: {missing}")
        images = np.stack([self._prepare_image(observation[key]) for key in wire_keys], axis=0)
        embeddings = np.asarray(jax.device_get(self._encode(self._variables, jnp.asarray(images))), dtype=np.float32)
        sensor_count = len(self.tactile_keys)
        if embeddings.shape != (sensor_count, 512) or not np.isfinite(embeddings).all():
            raise ValueError(f"tactile ResNet must return finite [{sensor_count},512] embeddings")
        tokens = _rms_normalize_tokens(embeddings)
        tokens = np.ascontiguousarray(tokens[None, ...], dtype=np.float32)
        if tokens.shape != (1, sensor_count, 512) or not np.isfinite(tokens).all():
            raise ValueError(f"tactile encoder tokens must be finite [1,{sensor_count},512]")
        return tokens
