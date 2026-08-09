from __future__ import annotations

from dataclasses import replace

import jax.numpy as jnp
import numpy as np
import pytest

from train_smolvla.configuration import JaxSmolVLAConfig
from train_smolvla.lora import (
    initialize_lora_params,
    is_trainable_parameter,
    resolve_module_modes,
)
from train_smolvla.modeling import JaxSmolVLA
from train_smolvla.training import partition_params


def all_modes(**overrides: str) -> dict[str, str]:
    modes = {
        "vision": "frozen",
        "connector": "frozen",
        "vlm_text": "frozen",
        "expert": "frozen",
        "action": "frozen",
        "state_proj": "frozen",
    }
    modes.update(overrides)
    return modes


def test_module_modes_partition_full_frozen_and_lora() -> None:
    config = replace(
        JaxSmolVLAConfig(),
        module_modes=all_modes(action="lora", state_proj="full"),
        lora_rank=2,
        lora_alpha=4.0,
    )
    params = {
        "model.action_in_proj.weight": jnp.ones((3, 4)),
        "model.action_in_proj.bias": jnp.ones(3),
        "model.state_proj.weight": jnp.ones((5, 4)),
        "model.state_proj.bias": jnp.ones(5),
        "model.vlm_with_expert.vlm.model.text_model.embed_tokens.weight": jnp.ones((8, 4)),
    }

    adapted = initialize_lora_params(params, config, seed=7)
    assert adapted["model.action_in_proj.lora_a"].shape == (2, 4)
    assert adapted["model.action_in_proj.lora_b"].shape == (3, 2)
    assert float(adapted["model.action_in_proj.lora_scale"]) == 2.0
    np.testing.assert_array_equal(adapted["model.action_in_proj.lora_b"], 0)

    trainable, frozen = partition_params(adapted, config)
    assert set(trainable) == {
        "model.action_in_proj.lora_a",
        "model.action_in_proj.lora_b",
        "model.state_proj.weight",
        "model.state_proj.bias",
    }
    assert "model.action_in_proj.weight" in frozen
    assert "model.action_in_proj.bias" in frozen
    assert "model.action_in_proj.lora_scale" in frozen
    assert "model.vlm_with_expert.vlm.model.text_model.embed_tokens.weight" in frozen


def test_optional_lora_linear_has_zero_impact_then_updates_output() -> None:
    config = replace(
        JaxSmolVLAConfig(),
        module_modes=all_modes(action="lora"),
        lora_rank=1,
        lora_alpha=1.0,
    )
    model = JaxSmolVLA(config)
    base_params = {
        "model.action_in_proj.weight": jnp.asarray([[1.0, 2.0], [3.0, 4.0]]),
        "model.action_in_proj.bias": jnp.asarray([0.5, -0.5]),
    }
    params = initialize_lora_params(base_params, config, seed=0)
    x = jnp.asarray([[2.0, -1.0]])

    base_output = model._linear(base_params, "model.action_in_proj", x, bias=True)
    zero_adapter_output = model._linear(params, "model.action_in_proj", x, bias=True)
    np.testing.assert_allclose(zero_adapter_output, base_output, rtol=0, atol=0)

    params["model.action_in_proj.lora_a"] = jnp.asarray([[1.0, 0.0]])
    params["model.action_in_proj.lora_b"] = jnp.asarray([[2.0], [-1.0]])
    adapted_output = model._linear(params, "model.action_in_proj", x, bias=True)
    np.testing.assert_allclose(adapted_output, base_output + jnp.asarray([[4.0, -2.0]]))


def test_every_module_accepts_each_train_mode() -> None:
    for module in all_modes():
        for mode in ("frozen", "full", "lora"):
            config = replace(JaxSmolVLAConfig(), module_modes=all_modes(**{module: mode}))
            assert resolve_module_modes(config)[module] == mode


def test_tactile_projection_module_is_optional_and_trainable() -> None:
    from train_vtsmolvla.configuration import VTSmolVLAConfig as JaxVTSmolVLAConfig
    from train_vtsmolvla.lora import (
        is_trainable_parameter as is_vt_trainable_parameter,
        resolve_module_modes as resolve_vt_module_modes,
    )

    base_modes = all_modes()
    config = replace(
        JaxVTSmolVLAConfig(),
        use_tactile_encoder=True,
        tactile_encoder_path="checkpoints/encoder/best",
        tactile_keys=("t0", "t1"),
        tactile_num_tokens=2,
        module_modes=base_modes,
    )
    assert resolve_vt_module_modes(config)["tactile_proj"] == "full"
    assert is_vt_trainable_parameter("model.tactile_proj.weight", config)
    assert is_vt_trainable_parameter("model.tactile_proj.bias", config)
    assert not is_vt_trainable_parameter("model.tactile_encoder.params/conv1/kernel", config)


def test_legacy_tactile_projection_is_trainable_when_enabled() -> None:
    from train_vtsmolvla.configuration import VTSmolVLAConfig as JaxVTSmolVLAConfig
    from train_vtsmolvla.lora import is_trainable_parameter as is_vt_trainable_parameter

    config = replace(
        JaxVTSmolVLAConfig(),
        use_tactile_encoder=True,
        tactile_encoder_path="checkpoints/encoder/best",
        tactile_keys=("t0",),
        tactile_num_tokens=1,
        module_modes=None,
    )

    assert is_vt_trainable_parameter("model.tactile_proj.weight", config)
    assert is_vt_trainable_parameter("model.tactile_proj.bias", config)


def test_vt_module_modes_reject_unknown_extension_module() -> None:
    from train_vtsmolvla.configuration import VTSmolVLAConfig
    from train_vtsmolvla.lora import resolve_module_modes as resolve_vt_module_modes

    config = replace(
        VTSmolVLAConfig(),
        module_modes={**all_modes(), "tactile_projection": "full"},
    )

    with pytest.raises(ValueError, match="unknown module_modes keys"):
        resolve_vt_module_modes(config)


def test_vt_module_modes_require_mapping() -> None:
    from train_vtsmolvla.configuration import VTSmolVLAConfig
    from train_vtsmolvla.lora import resolve_module_modes as resolve_vt_module_modes

    config = replace(VTSmolVLAConfig(), module_modes="invalid")

    with pytest.raises(ValueError, match="module_modes must be a mapping"):
        resolve_vt_module_modes(config)


def test_tactile_projection_lora_initializes_and_partitions_adapter() -> None:
    from train_vtsmolvla.configuration import VTSmolVLAConfig
    from train_vtsmolvla.lora import (
        initialize_lora_params as initialize_vt_lora_params,
        is_trainable_parameter as is_vt_trainable_parameter,
    )

    config = replace(
        VTSmolVLAConfig(),
        use_tactile_encoder=True,
        module_modes={**all_modes(), "tactile_proj": "lora"},
        lora_rank=2,
        lora_alpha=4.0,
    )
    params = {"model.tactile_proj.weight": jnp.ones((4, 3), dtype=jnp.float32)}

    adapted = initialize_vt_lora_params(params, config, seed=0)

    assert adapted["model.tactile_proj.lora_a"].shape == (2, 3)
    assert adapted["model.tactile_proj.lora_b"].shape == (4, 2)
    assert float(adapted["model.tactile_proj.lora_scale"]) == 2.0
    assert not is_vt_trainable_parameter("model.tactile_proj.weight", config)
    assert is_vt_trainable_parameter("model.tactile_proj.lora_a", config)
    assert is_vt_trainable_parameter("model.tactile_proj.lora_b", config)
    assert not is_vt_trainable_parameter("model.tactile_proj.lora_scale", config)


def test_vt_lora_initialization_keeps_the_established_parameter_order() -> None:
    from train_vtsmolvla.configuration import VTSmolVLAConfig
    from train_vtsmolvla.lora import initialize_lora_params as initialize_vt_lora_params

    config = replace(
        VTSmolVLAConfig(),
        use_tactile_encoder=True,
        module_modes={**all_modes(vision="lora"), "tactile_proj": "lora"},
        lora_rank=2,
        lora_alpha=4.0,
    )
    visual_q_proj = (
        "model.vlm_with_expert.vlm.model.vision_model.encoder.layers.0."
        "self_attn.q_proj"
    )
    params = {
        "model.tactile_proj.weight": jnp.ones((4, 3), dtype=jnp.float32),
        f"{visual_q_proj}.weight": jnp.ones((5, 4), dtype=jnp.float32),
    }

    actual = initialize_vt_lora_params(params, config, seed=17)

    expected_tactile = np.asarray(
        [
            [0.6358142, 0.19539338, -0.3117527],
            [-0.727601, -1.0938601, 0.01076082],
        ],
        dtype=np.float32,
    )
    expected_visual = np.asarray(
        [
            [-0.40528354, -0.43607798, -0.11098475, -0.02592301],
            [-1.1383914, 0.4625733, -1.0134228, 0.9298121],
        ],
        dtype=np.float32,
    )

    np.testing.assert_allclose(
        actual["model.tactile_proj.lora_a"],
        expected_tactile,
        rtol=1e-6,
        atol=1e-7,
    )
    np.testing.assert_allclose(
        actual[f"{visual_q_proj}.lora_a"],
        expected_visual,
        rtol=1e-6,
        atol=1e-7,
    )


def test_vlm_lora_targets_can_match_vb3_qv_only() -> None:
    config = replace(
        JaxSmolVLAConfig(),
        module_modes=all_modes(vlm_text="lora"),
        lora_rank=2,
        vlm_lora_target_modules=("q_proj", "v_proj"),
    )
    prefix = "model.vlm_with_expert.vlm.model.text_model.layers.0"
    params = {
        f"{prefix}.self_attn.{name}.weight": jnp.ones((4, 4))
        for name in ("q_proj", "k_proj", "v_proj", "o_proj")
    }
    params[f"{prefix}.mlp.gate_proj.weight"] = jnp.ones((8, 4))

    adapted = initialize_lora_params(params, config, seed=0)
    assert f"{prefix}.self_attn.q_proj.lora_a" in adapted
    assert f"{prefix}.self_attn.v_proj.lora_a" in adapted
    assert f"{prefix}.self_attn.k_proj.lora_a" not in adapted
    assert f"{prefix}.self_attn.o_proj.lora_a" not in adapted
    assert f"{prefix}.mlp.gate_proj.lora_a" not in adapted
    assert is_trainable_parameter(f"{prefix}.self_attn.q_proj.lora_a", config)
    assert not is_trainable_parameter(f"{prefix}.self_attn.k_proj.lora_a", config)


def test_adapter_scale_is_never_trainable() -> None:
    config = replace(JaxSmolVLAConfig(), module_modes=all_modes(expert="lora"))
    assert is_trainable_parameter(
        "model.vlm_with_expert.lm_expert.layers.0.self_attn.q_proj.lora_a", config
    )
    assert not is_trainable_parameter(
        "model.vlm_with_expert.lm_expert.layers.0.self_attn.q_proj.lora_scale", config
    )
