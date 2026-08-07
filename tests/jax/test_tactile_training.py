from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

from lerobot.policies.smolvla_jax.configuration import JaxSmolVLAConfig
from lerobot.policies.smolvla_jax.functional import linear
from lerobot.policies.smolvla_jax.training import JaxSmolVLATrainer


class TinyModel:
    def __init__(self, config: JaxSmolVLAConfig):
        self.config = config

    def loss(self, params, batch, rng):
        del rng
        prediction = linear(batch["x"], params["model.action_in_proj.weight"])
        return jnp.mean(jnp.square(prediction - batch["target"]))


def tiny_setup():
    config = dataclasses.replace(
        JaxSmolVLAConfig(),
        optimizer_lr=1e-2,
        scheduler_decay_lr=1e-3,
        scheduler_warmup_steps=2,
        scheduler_decay_steps=10,
    )
    params = {
        "model.action_in_proj.weight": jnp.asarray([[0.5, -0.25]], dtype=jnp.float32),
        "model.vlm_with_expert.vlm.model.text_model.norm.weight": jnp.ones(2),
    }
    batch = {
        "x": jnp.asarray([[1.0, 2.0], [-1.0, 0.5]], dtype=jnp.float32),
        "target": jnp.asarray([[0.75], [-0.5]], dtype=jnp.float32),
    }
    return config, params, batch


def _remove_repeat_factor_from_resume_metadata(checkpoint: Path) -> None:
    path = checkpoint / "resume_metadata.json"
    metadata = json.loads(path.read_text())
    del metadata["resume_signature"]["model"]["tactile_token_repeat_factor"]
    path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")


def _make_checkpoint_legacy_without_repeat_factor(checkpoint: Path) -> None:
    (checkpoint / "resume_metadata.json").unlink()
    path = checkpoint / "config.json"
    config = json.loads(path.read_text())
    del config["tactile_token_repeat_factor"]
    path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")


def test_resume_treats_legacy_missing_tactile_repeat_factor_as_one(tmp_path: Path) -> None:
    config, params, batch = tiny_setup()
    trainer = JaxSmolVLATrainer(TinyModel(config), params, seed=4, total_steps=10)
    trainer.step(batch)
    checkpoint = trainer.save(tmp_path / "checkpoint")
    _remove_repeat_factor_from_resume_metadata(checkpoint)

    resumed = JaxSmolVLATrainer(TinyModel(config), params, seed=4, total_steps=10)
    resumed.restore(checkpoint)
    assert resumed.step_count == 1


def test_resume_rejects_changed_tactile_repeat_factor(tmp_path: Path) -> None:
    config, params, batch = tiny_setup()
    trainer = JaxSmolVLATrainer(TinyModel(config), params, seed=4, total_steps=10)
    trainer.step(batch)
    checkpoint = trainer.save(tmp_path / "checkpoint")

    changed = dataclasses.replace(config, tactile_token_repeat_factor=8)
    resumed = JaxSmolVLATrainer(TinyModel(changed), params, seed=4, total_steps=10)
    with np.testing.assert_raises_regex(ValueError, "tactile_token_repeat_factor"):
        resumed.restore(checkpoint)


def test_legacy_resume_without_metadata_defaults_tactile_repeat_factor_to_one(
    tmp_path: Path,
) -> None:
    config, params, batch = tiny_setup()
    trainer = JaxSmolVLATrainer(TinyModel(config), params, seed=4, total_steps=10)
    trainer.step(batch)
    checkpoint = trainer.save(tmp_path / "checkpoint")
    _make_checkpoint_legacy_without_repeat_factor(checkpoint)

    resumed = JaxSmolVLATrainer(TinyModel(config), params, seed=4, total_steps=10)
    resumed.restore(checkpoint)
    assert resumed.step_count == 1


@pytest.mark.parametrize("repeat_factor", [8, 21])
def test_legacy_resume_without_metadata_rejects_changed_tactile_repeat_factor(
    tmp_path: Path,
    repeat_factor: int,
) -> None:
    config, params, batch = tiny_setup()
    trainer = JaxSmolVLATrainer(TinyModel(config), params, seed=4, total_steps=10)
    trainer.step(batch)
    checkpoint = trainer.save(tmp_path / "checkpoint")
    _make_checkpoint_legacy_without_repeat_factor(checkpoint)

    changed = dataclasses.replace(config, tactile_token_repeat_factor=repeat_factor)
    resumed = JaxSmolVLATrainer(TinyModel(changed), params, seed=4, total_steps=10)
    with np.testing.assert_raises_regex(ValueError, "tactile_token_repeat_factor"):
        resumed.restore(checkpoint)
