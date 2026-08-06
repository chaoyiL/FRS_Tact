from __future__ import annotations

import dataclasses
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from lerobot.policies.smolvla_jax.configuration import JaxSmolVLAConfig
from lerobot.policies.smolvla_jax.functional import linear
from lerobot.policies.smolvla_jax.sharding import create_data_parallel_mesh, shard_batch
from lerobot.policies.smolvla_jax.training import (
    JaxSmolVLATrainer,
    cast_trainable_params_for_compute,
    cosine_warmup_schedule,
    create_optimizer,
    partition_params,
    promote_trainable_params_to_fp32,
)


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


def test_partition_and_schedule() -> None:
    config, params, _ = tiny_setup()
    trainable, frozen = partition_params(params, config)
    assert set(trainable) == {"model.action_in_proj.weight"}
    assert set(frozen) == {"model.vlm_with_expert.vlm.model.text_model.norm.weight"}
    schedule = cosine_warmup_schedule(config)
    assert float(schedule(0)) < float(schedule(2))
    assert np.isclose(float(schedule(10)), config.scheduler_decay_lr)


def test_trainable_bfloat16_params_are_promoted_but_frozen_params_are_not() -> None:
    config = JaxSmolVLAConfig()
    expert_key = "model.vlm_with_expert.lm_expert.layers.0.self_attn.q_proj.weight"
    frozen_key = "model.vlm_with_expert.vlm.model.text_model.norm.weight"
    params = {
        expert_key: jnp.ones((2, 2), dtype=jnp.bfloat16),
        frozen_key: jnp.ones((2,), dtype=jnp.bfloat16),
    }

    promoted = promote_trainable_params_to_fp32(params, config)

    assert promoted[expert_key].dtype == jnp.float32
    assert promoted[frozen_key].dtype == jnp.bfloat16

    trainable, _ = partition_params(promoted, config)
    compute_params = cast_trainable_params_for_compute(trainable)
    assert compute_params[expert_key].dtype == jnp.bfloat16

    optimizer, _ = create_optimizer(config)
    opt_state = optimizer.init(trainable)
    floating_state = [
        leaf
        for leaf in jax.tree.leaves(opt_state)
        if hasattr(leaf, "dtype") and jnp.issubdtype(leaf.dtype, jnp.inexact)
    ]
    assert floating_state
    assert all(leaf.dtype == jnp.float32 for leaf in floating_state)


def test_partition_freezes_both_unused_tail_layers_for_a_shallow_expert() -> None:
    config = dataclasses.replace(
        JaxSmolVLAConfig(),
        train_expert_only=False,
        num_vlm_layers=16,
        num_expert_layers=8,
    )
    params = {
        f"model.vlm_with_expert.vlm.model.text_model.layers.{index}.self_attn.q_proj.weight": (
            jnp.ones((1, 1))
        )
        for index in (13, 14, 15)
    }
    trainable, frozen = partition_params(params, config)
    assert set(trainable) == {"model.vlm_with_expert.vlm.model.text_model.layers.13.self_attn.q_proj.weight"}
    assert set(frozen) == {
        "model.vlm_with_expert.vlm.model.text_model.layers.14.self_attn.q_proj.weight",
        "model.vlm_with_expert.vlm.model.text_model.layers.15.self_attn.q_proj.weight",
    }


def test_train_step_and_exact_resume(tmp_path: Path) -> None:
    config, params, batch = tiny_setup()
    trainer = JaxSmolVLATrainer(TinyModel(config), params, seed=4, total_steps=10)
    first_metrics = jax.device_get(trainer.step(batch))
    assert np.isfinite(first_metrics["loss"])
    assert int(trainer.state.step) == 1
    assert trainer.step_count == 1
    checkpoint = trainer.save(tmp_path / "checkpoint")

    resumed = JaxSmolVLATrainer(TinyModel(config), params, seed=999, total_steps=10)
    resumed.restore(checkpoint)
    assert int(resumed.state.step) == 1
    assert resumed.step_count == 1
    reference_metrics = jax.device_get(trainer.step(batch))
    resumed_metrics = jax.device_get(resumed.step(batch))
    np.testing.assert_allclose(resumed_metrics["loss"], reference_metrics["loss"], rtol=0, atol=0)
    np.testing.assert_allclose(
        resumed.state.params["model.action_in_proj.weight"],
        trainer.state.params["model.action_in_proj.weight"],
        rtol=0,
        atol=0,
    )


def test_resume_restores_modality_dropout_rng(tmp_path: Path) -> None:
    config, params, batch = tiny_setup()
    dropout = {
        "enable": True,
        "every_n_steps": 1,
        "prob": 1.0,
        "drop_language": True,
        "camera_indices": [0, 1],
    }
    batch = {
        **batch,
        "image_masks": jnp.ones((2, 2), dtype=jnp.bool_),
        "language_masks": jnp.ones((2, 3), dtype=jnp.bool_),
    }
    trainer = JaxSmolVLATrainer(
        TinyModel(config), params, seed=4, total_steps=10, modality_dropout=dropout
    )
    for _ in range(3):
        trainer.step(batch)
    checkpoint = trainer.save(tmp_path / "checkpoint")

    resumed = JaxSmolVLATrainer(
        TinyModel(config), params, seed=999, total_steps=10, modality_dropout=dropout
    )
    resumed.restore(checkpoint)

    assert resumed._modality_dropout_rng.bit_generator.state == trainer._modality_dropout_rng.bit_generator.state


def test_resume_rejects_changed_lora_rank(tmp_path: Path) -> None:
    config, params, batch = tiny_setup()
    modes = {
        "vision": "frozen",
        "connector": "frozen",
        "vlm_text": "lora",
        "expert": "frozen",
        "action": "full",
        "state_proj": "frozen",
        "tactile_proj": "frozen",
    }
    config = dataclasses.replace(config, module_modes=modes, lora_rank=2, lora_alpha=2.0)
    params = {
        **params,
        "model.vlm_with_expert.vlm.model.text_model.layers.0.self_attn.q_proj.weight": (
            jnp.ones((4, 4), dtype=jnp.float32)
        ),
    }
    trainer = JaxSmolVLATrainer(TinyModel(config), params, seed=4, total_steps=10)
    trainer.step(batch)
    checkpoint = trainer.save(tmp_path / "checkpoint")

    changed = dataclasses.replace(config, lora_rank=4, lora_alpha=4.0)
    resumed = JaxSmolVLATrainer(TinyModel(changed), params, seed=4, total_steps=10)
    with np.testing.assert_raises_regex(ValueError, "lora_(rank|alpha)"):
        resumed.restore(checkpoint)


def test_data_parallel_batch_sharding_on_visible_devices() -> None:
    mesh = create_data_parallel_mesh()
    count = mesh.size
    batch = {"x": jnp.ones((count * 2, 3), dtype=jnp.float32)}
    sharded = shard_batch(batch, mesh)
    assert sharded["x"].shape == (count * 2, 3)
    assert sharded["x"].sharding.spec[0] == "data"
