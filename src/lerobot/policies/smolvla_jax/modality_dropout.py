"""Training-time modality dropout via existing attention / pad masks."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

Array = jax.Array


@dataclass(frozen=True)
class ModalityDropoutConfig:
    """Randomly mask out one modality every ``every_n_steps`` during training.

    Dropping is implemented by zeroing the corresponding pad / attention masks:
    - cameras → ``image_masks[:, camera_index] = False``
    - language → ``language_masks = False``
    - state → ``state_mask = False`` (consumed by ``embed_prefix``)
    """

    enable: bool = False
    every_n_steps: int = 1
    # When the every_n trigger fires, apply dropout with this probability.
    prob: float = 1.0
    drop_language: bool = True
    drop_state: bool = False
    # Which camera indices may be dropped; None = all cameras present in the batch.
    camera_indices: tuple[int, ...] | None = None

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | None) -> ModalityDropoutConfig:
        raw = dict(data or {})
        camera_indices = raw.get("camera_indices")
        if camera_indices is not None:
            camera_indices = tuple(int(index) for index in camera_indices)
        every_n = int(raw.get("every_n_steps", 1))
        if every_n <= 0:
            raise ValueError(f"modality_dropout.every_n_steps must be positive, got {every_n}")
        prob = float(raw.get("prob", 1.0))
        if not 0.0 <= prob <= 1.0:
            raise ValueError(f"modality_dropout.prob must be in [0, 1], got {prob}")
        return cls(
            enable=bool(raw.get("enable", False)),
            every_n_steps=every_n,
            prob=prob,
            drop_language=bool(raw.get("drop_language", True)),
            drop_state=bool(raw.get("drop_state", False)),
            camera_indices=camera_indices,
        )


def _candidate_modalities(
    config: ModalityDropoutConfig,
    *,
    num_cameras: int,
) -> list[tuple[str, int | None]]:
    """Return droppable modalities as ``(kind, camera_index_or_None)``."""

    candidates: list[tuple[str, int | None]] = []
    if num_cameras > 0:
        indices = range(num_cameras) if config.camera_indices is None else config.camera_indices
        for index in indices:
            if index < 0 or index >= num_cameras:
                raise ValueError(
                    f"modality_dropout.camera_indices entry {index} is out of range "
                    f"for num_cameras={num_cameras}"
                )
            candidates.append(("camera", int(index)))
    if config.drop_language:
        candidates.append(("language", None))
    if config.drop_state:
        candidates.append(("state", None))
    return candidates


def apply_modality_dropout(
    batch: Mapping[str, Array],
    *,
    step: int,
    rng: np.random.Generator | int,
    config: ModalityDropoutConfig,
) -> tuple[dict[str, Array], dict[str, Any]]:
    """Maybe drop one modality from ``batch``; return updated batch and info.

    Validation / inference should skip this call (or keep ``enable=False``).
    """

    info: dict[str, Any] = {
        "applied": False,
        "modality": "none",
        "camera_index": -1,
    }
    out = dict(batch)
    if not config.enable or step < 0 or (step % config.every_n_steps) != 0:
        return out, info

    generator = rng if isinstance(rng, np.random.Generator) else np.random.default_rng(rng)
    if float(generator.random()) >= config.prob:
        return out, info

    image_masks = jnp.asarray(out["image_masks"])
    if image_masks.ndim != 2:
        raise ValueError(f"image_masks must be [B, N], got {image_masks.shape}")

    candidates = _candidate_modalities(config, num_cameras=int(image_masks.shape[1]))
    # Avoid dropping the only remaining live camera when there is just one camera modality.
    if image_masks.shape[1] == 1:
        candidates = [item for item in candidates if item[0] != "camera"]
    if not candidates:
        return out, info

    kind, camera_index = candidates[int(generator.integers(0, len(candidates)))]
    batch_size = int(image_masks.shape[0])

    if kind == "camera":
        assert camera_index is not None
        updated = np.asarray(image_masks).copy()
        updated[:, camera_index] = False
        if not updated.any():
            # Should not happen with >1 cameras; refuse to blank all visual tokens.
            return out, info
        out["image_masks"] = jnp.asarray(updated, dtype=image_masks.dtype)
        info.update(applied=True, modality=f"camera_{camera_index}", camera_index=int(camera_index))
        return out, info

    if kind == "language":
        language_masks = jnp.asarray(out["language_masks"])
        out["language_masks"] = jnp.zeros_like(language_masks)
        info.update(applied=True, modality="language", camera_index=-1)
        return out, info

    out["state_mask"] = jnp.zeros((batch_size,), dtype=jnp.bool_)
    info.update(applied=True, modality="state", camera_index=-1)
    return out, info
