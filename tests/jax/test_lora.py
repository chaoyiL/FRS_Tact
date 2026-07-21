from __future__ import annotations

from dataclasses import replace

import jax.numpy as jnp
import numpy as np

from lerobot.policies.smolvla_jax.configuration import JaxSmolVLAConfig
from lerobot.policies.smolvla_jax.lora import (
    initialize_lora_params,
    is_trainable_parameter,
    resolve_module_modes,
)
from lerobot.policies.smolvla_jax.modeling import JaxSmolVLA
from lerobot.policies.smolvla_jax.training import partition_params


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


def test_adapter_scale_is_never_trainable() -> None:
    config = replace(JaxSmolVLAConfig(), module_modes=all_modes(expert="lora"))
    assert is_trainable_parameter(
        "model.vlm_with_expert.lm_expert.layers.0.self_attn.q_proj.lora_a", config
    )
    assert not is_trainable_parameter(
        "model.vlm_with_expert.lm_expert.layers.0.self_attn.q_proj.lora_scale", config
    )
