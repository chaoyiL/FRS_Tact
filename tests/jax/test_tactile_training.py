from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from safetensors import safe_open

from lerobot.policies.smolvla_jax import policy as policy_module
from lerobot.policies.smolvla_jax import training as training_module
from lerobot.policies.smolvla_jax.checkpoint import load_params
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


def _remove_compute_dtype_from_resume_metadata(checkpoint: Path) -> None:
    path = checkpoint / "resume_metadata.json"
    metadata = json.loads(path.read_text())
    del metadata["resume_signature"]["model"]["trainable_compute_dtype"]
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


def test_resume_treats_legacy_missing_compute_dtype_as_bfloat16(tmp_path: Path) -> None:
    config, params, batch = tiny_setup()
    trainer = JaxSmolVLATrainer(TinyModel(config), params, seed=4, total_steps=10)
    trainer.step(batch)
    checkpoint = trainer.save(tmp_path / "checkpoint")
    _remove_compute_dtype_from_resume_metadata(checkpoint)

    resumed = JaxSmolVLATrainer(TinyModel(config), params, seed=4, total_steps=10)
    resumed.restore(checkpoint)

    assert resumed.step_count == 1


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


def test_prepare_params_for_compute_casts_only_trainable_floating_leaves() -> None:
    config = JaxSmolVLAConfig()
    trainable_float = "model.action_in_proj.weight"
    frozen_float = "model.vlm_with_expert.vlm.model.text_model.norm.weight"
    trainable_integer = "model.action_in_proj.bias"
    params = {
        trainable_float: jnp.ones((2, 2), dtype=jnp.float32),
        frozen_float: jnp.ones((2,), dtype=jnp.float32),
        trainable_integer: jnp.ones((2,), dtype=jnp.int32),
    }

    compute = training_module.prepare_params_for_compute(params, config)

    assert compute[trainable_float].dtype == jnp.bfloat16
    assert compute[frozen_float].dtype == jnp.float32
    assert compute[trainable_integer].dtype == jnp.int32
    assert params[trainable_float].dtype == jnp.float32


def test_training_and_evaluation_use_the_same_compute_dtypes() -> None:
    class DtypeCheckingModel(TinyModel):
        def loss(self, params, batch, rng):
            assert params["model.action_in_proj.weight"].dtype == jnp.bfloat16
            assert (
                params["model.vlm_with_expert.vlm.model.text_model.norm.weight"].dtype
                == jnp.float32
            )
            return super().loss(params, batch, rng)

    config, params, batch = tiny_setup()
    trainer = JaxSmolVLATrainer(DtypeCheckingModel(config), params, seed=4, total_steps=10)

    trainer.step(batch)
    metrics = trainer.evaluate(
        ({**batch, "actions": jnp.zeros((2, 1, 1), dtype=jnp.float32)},),
        rollout=False,
    )

    assert np.isfinite(metrics["loss"])


def test_save_load_prepare_matches_trainer_compute_and_preserves_fp32_master(
    tmp_path: Path,
) -> None:
    config, params, batch = tiny_setup()
    trainer = JaxSmolVLATrainer(TinyModel(config), params, seed=4, total_steps=10)
    trainer.step(batch)
    assert trainer.state.params["model.action_in_proj.weight"].dtype == jnp.float32
    rng = jax.random.key(17)
    noise = jax.random.normal(jax.random.key(23), batch["x"].shape, dtype=jnp.float32)

    def fixed_forward(compute_params):
        scale = jax.random.uniform(rng, (batch["x"].shape[0], 1), minval=0.9, maxval=1.1)
        return linear(
            batch["x"] + noise,
            compute_params["model.action_in_proj.weight"],
        ) * scale

    expected_prediction = fixed_forward(trainer.compute_params)
    checkpoint = trainer.save(tmp_path / "checkpoint")

    with safe_open(checkpoint / "model.safetensors", framework="numpy") as file:
        assert file.get_tensor("model.action_in_proj.weight").dtype == np.float32

    loaded_compute = training_module.prepare_params_for_compute(load_params(checkpoint), config)
    actual_prediction = fixed_forward(loaded_compute)

    assert loaded_compute["model.action_in_proj.weight"].dtype == jnp.bfloat16
    np.testing.assert_array_equal(actual_prediction, expected_prediction)


def test_resume_rejects_changed_trainable_compute_dtype_before_state_restore(
    tmp_path: Path,
) -> None:
    config, params, batch = tiny_setup()
    trainer = JaxSmolVLATrainer(TinyModel(config), params, seed=4, total_steps=10)
    trainer.step(batch)
    checkpoint = trainer.save(tmp_path / "checkpoint")
    metadata_path = checkpoint / "resume_metadata.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["resume_signature"]["model"]["trainable_compute_dtype"] = "float32"
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")

    resumed = JaxSmolVLATrainer(TinyModel(config), params, seed=4, total_steps=10)
    with pytest.raises(ValueError, match="trainable_compute_dtype"):
        resumed.restore(checkpoint)


def test_policy_prepares_loaded_master_params_for_compute(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = JaxSmolVLAConfig()
    master_params = {"model.action_in_proj.weight": jnp.ones((1, 1), dtype=jnp.float32)}
    prepared_params = {"prepared": jnp.ones((1,), dtype=jnp.bfloat16)}
    calls: list[tuple[object, object]] = []

    monkeypatch.setattr(policy_module, "resolve_checkpoint", lambda *args, **kwargs: tmp_path)
    monkeypatch.setattr(policy_module, "load_config", lambda checkpoint: config)
    monkeypatch.setattr(policy_module, "load_params", lambda checkpoint: master_params)
    monkeypatch.setattr(policy_module, "JaxSmolVLA", lambda cfg: SimpleNamespace(config=cfg))
    monkeypatch.setattr(
        policy_module,
        "JaxSmolVLAPreprocessor",
        lambda *args, **kwargs: SimpleNamespace(),
    )

    def record_prepare(params: object, cfg: object) -> object:
        calls.append((params, cfg))
        return prepared_params

    monkeypatch.setattr(policy_module, "prepare_params_for_compute", record_prepare)

    policy = policy_module.JaxSmolVLAPolicy(tmp_path)

    assert calls == [(master_params, config)]
    assert policy.params is prepared_params
