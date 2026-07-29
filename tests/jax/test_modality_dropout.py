from __future__ import annotations

import numpy as np
import jax.numpy as jnp

from lerobot.policies.smolvla_jax.modality_dropout import (
    ModalityDropoutConfig,
    apply_modality_dropout,
)


def _batch(num_cameras: int = 2):
    return {
        "images": jnp.zeros((4, num_cameras, 3, 8, 8), dtype=jnp.float32),
        "image_masks": jnp.ones((4, num_cameras), dtype=jnp.bool_),
        "language_tokens": jnp.ones((4, 6), dtype=jnp.int32),
        "language_masks": jnp.ones((4, 6), dtype=jnp.bool_),
        "state": jnp.zeros((4, 8), dtype=jnp.float32),
        "actions": jnp.zeros((4, 2, 8), dtype=jnp.float32),
    }


def test_modality_dropout_disabled_noop() -> None:
    batch, info = apply_modality_dropout(
        _batch(),
        step=0,
        rng=0,
        config=ModalityDropoutConfig(enable=False),
    )
    assert info["applied"] is False
    assert bool(jnp.all(batch["image_masks"]))


def test_modality_dropout_every_n_steps() -> None:
    cfg = ModalityDropoutConfig(
        enable=True,
        every_n_steps=4,
        prob=1.0,
        drop_language=True,
        drop_state=False,
    )
    # Non-trigger steps stay intact.
    batch, info = apply_modality_dropout(_batch(), step=1, rng=0, config=cfg)
    assert info["applied"] is False
    assert bool(jnp.all(batch["image_masks"]))
    assert bool(jnp.all(batch["language_masks"]))

    # Trigger step always drops exactly one modality with fixed RNG stream.
    seen = set()
    for seed in range(20):
        batch, info = apply_modality_dropout(_batch(), step=4, rng=seed, config=cfg)
        assert info["applied"] is True
        seen.add(info["modality"])
        if info["modality"].startswith("camera_"):
            index = info["camera_index"]
            assert not bool(batch["image_masks"][:, index].any())
            # Other camera remains.
            other = 1 - index
            assert bool(batch["image_masks"][:, other].all())
            assert bool(jnp.all(batch["language_masks"]))
        elif info["modality"] == "language":
            assert not bool(batch["language_masks"].any())
            assert bool(jnp.all(batch["image_masks"]))
        else:
            raise AssertionError(info["modality"])
    assert seen & {"camera_0", "camera_1", "language"}


def test_modality_dropout_can_drop_state() -> None:
    cfg = ModalityDropoutConfig(
        enable=True,
        every_n_steps=1,
        prob=1.0,
        drop_language=False,
        drop_state=True,
        camera_indices=(),  # no cameras droppable
    )
    batch, info = apply_modality_dropout(_batch(), step=0, rng=0, config=cfg)
    assert info == {"applied": True, "modality": "state", "camera_index": -1}
    assert "state_mask" in batch
    assert not bool(batch["state_mask"].any())


def test_modality_dropout_config_from_dict() -> None:
    cfg = ModalityDropoutConfig.from_dict(
        {
            "enable": True,
            "every_n_steps": 8,
            "prob": 0.5,
            "drop_language": False,
            "camera_indices": [0],
        }
    )
    assert cfg.every_n_steps == 8
    assert cfg.prob == 0.5
    assert cfg.drop_language is False
    assert cfg.camera_indices == (0,)
