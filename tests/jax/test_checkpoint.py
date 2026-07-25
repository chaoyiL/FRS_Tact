from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("jax")

from lerobot.policies.smolvla_jax.checkpoint import (  # noqa: E402
    count_expert_layers,
    count_vlm_layers,
    extend_vlm_layers,
    load_safetensors_params,
    parameter_summary,
    save_portable_params,
    write_effective_config,
)
from lerobot.policies.smolvla_jax.configuration import JaxSmolVLAConfig  # noqa: E402


def test_extend_vlm_layers_from_full_checkpoint(tmp_path: Path) -> None:
    from safetensors.flax import save_file as save_safetensors_file

    target_prefix = "model.vlm_with_expert.vlm.model.text_model.layers"
    expert_prefix = "model.vlm_with_expert.lm_expert.layers"
    params = {
        f"{target_prefix}.{layer}.input_layernorm.weight": np.full(2, layer, dtype=np.float16)
        for layer in range(2)
    }
    params.update(
        {
            f"{expert_prefix}.{layer}.input_layernorm.weight": np.full(2, layer, dtype=np.float16)
            for layer in range(2)
        }
    )
    source = tmp_path / "full-vlm"
    source.mkdir()
    save_safetensors_file(
        {
            f"model.text_model.layers.{layer}.input_layernorm.weight": np.full(
                2, layer, dtype=np.float32
            )
            for layer in range(4)
        },
        source / "model.safetensors",
    )

    assert count_vlm_layers(params) == 2
    assert count_expert_layers(params) == 2
    extended = extend_vlm_layers(params, 4, source=source)
    assert count_vlm_layers(extended) == 4
    assert extended[f"{target_prefix}.3.input_layernorm.weight"].dtype == np.float16
    np.testing.assert_array_equal(
        extended[f"{target_prefix}.3.input_layernorm.weight"],
        np.full(2, 3, dtype=np.float16),
    )


def test_vlm_override_auto_expert_layers() -> None:
    config = JaxSmolVLAConfig(num_vlm_layers=16, num_expert_layers=16)
    assert config.with_overrides({"num_vlm_layers": 8, "num_expert_layers": -1}).num_expert_layers == 8
    full = config.with_overrides({"num_vlm_layers": 32, "num_expert_layers": -1})
    assert full.num_expert_layers == 16
    assert config.with_overrides({"num_vlm_layers": 24, "num_expert_layers": -1}).num_expert_layers == 12
    with pytest.raises(ValueError, match="must divide"):
        config.with_overrides({"num_vlm_layers": 24, "num_expert_layers": 16})


def test_portable_round_trip_and_manifest(tmp_path: Path) -> None:
    params = {
        "float.weight": np.arange(12, dtype=np.float32).reshape(3, 4),
        "int.value": np.arange(3, dtype=np.int32),
    }
    output = save_portable_params(params, tmp_path / "checkpoint")
    restored = load_safetensors_params(output)
    np.testing.assert_array_equal(restored["float.weight"], params["float.weight"])
    np.testing.assert_array_equal(restored["int.value"], params["int.value"])
    manifest = json.loads((output / "conversion_manifest.json").read_text())
    assert manifest["tensor_count"] == 2
    assert manifest["parameter_count"] == 15
    assert parameter_summary(restored)["layout"] == "pytorch_source_layout"


def test_effective_config_persists_dimensions_and_lora_settings(tmp_path: Path) -> None:
    config = replace(
        JaxSmolVLAConfig(),
        action_dim=20,
        state_dim=20,
        image_keys=("observation.images.camera1", "observation.images.camera2"),
        empty_cameras=0,
        resize_height=512,
        resize_width=512,
        module_modes={
            "vision": "lora",
            "connector": "frozen",
            "vlm_text": "frozen",
            "expert": "full",
            "action": "full",
            "state_proj": "full",
        },
        lora_rank=4,
        lora_alpha=8.0,
        optimizer_beta1=0.91,
        optimizer_beta2=0.96,
        freeze_vision_encoder=True,
        train_expert_only=True,
    )
    # Pretend we copied a base config with an extra camera that should be dropped.
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "input_features": {
                    "observation.state": {"type": "STATE", "shape": [6]},
                    "observation.images.camera1": {"type": "VISUAL", "shape": [3, 256, 256]},
                    "observation.images.camera2": {"type": "VISUAL", "shape": [3, 256, 256]},
                    "observation.images.camera3": {"type": "VISUAL", "shape": [3, 256, 256]},
                },
                "output_features": {"action": {"type": "ACTION", "shape": [6]}},
            }
        )
    )
    write_effective_config(tmp_path, config)
    raw = json.loads((tmp_path / "config.json").read_text())
    assert raw["input_features"]["observation.state"]["shape"] == [20]
    assert raw["output_features"]["action"]["shape"] == [20]
    assert set(k for k, v in raw["input_features"].items() if v.get("type") == "VISUAL") == {
        "observation.images.camera1",
        "observation.images.camera2",
    }
    assert "observation.images.camera3" not in raw["input_features"]
    assert raw["resize_imgs_with_padding"] == [512, 512]
    assert raw["optimizer_betas"] == [0.91, 0.96]
    assert raw["module_modes"]["vision"] == "lora"
    assert raw["lora_rank"] == 4
    assert raw["lora_alpha"] == 8.0
    # module_modes is the source of truth for the legacy boolean switches.
    assert raw["freeze_vision_encoder"] is False
    assert raw["train_expert_only"] is False
    assert raw["train_state_proj"] is True

    reloaded = JaxSmolVLAConfig.from_pretrained(tmp_path)
    assert reloaded.image_keys == config.image_keys
    assert reloaded.state_dim == 20
    assert reloaded.action_dim == 20
    assert reloaded.module_modes["vision"] == "lora"


def test_processor_configs_sync_rename_map_and_feature_shapes(tmp_path: Path) -> None:
    from safetensors.flax import save_file as save_safetensors_file

    from lerobot.policies.smolvla_jax.preprocessing import JaxSmolVLAPreprocessor

    source = tmp_path / "source"
    destination = tmp_path / "destination"
    source.mkdir()
    destination.mkdir()
    (source / "policy_preprocessor.json").write_text(
        json.dumps(
            {
                "name": "policy_preprocessor",
                "steps": [
                    {"registry_name": "rename_observations_processor", "config": {"rename_map": {}}},
                    {
                        "registry_name": "normalizer_processor",
                        "config": {
                            "features": {
                                "observation.state": {"type": "STATE", "shape": [6]},
                                "action": {"type": "ACTION", "shape": [6]},
                            }
                        },
                        "state_file": "policy_preprocessor_step_5_normalizer_processor.safetensors",
                    },
                ],
            }
        )
    )
    (source / "policy_postprocessor.json").write_text(
        json.dumps(
            {
                "name": "policy_postprocessor",
                "steps": [
                    {
                        "registry_name": "unnormalizer_processor",
                        "config": {"features": {"action": {"type": "ACTION", "shape": [6]}}},
                        "state_file": "policy_postprocessor_step_0_unnormalizer_processor.safetensors",
                    }
                ],
            }
        )
    )
    save_safetensors_file(
        {"observation.state.mean": np.zeros(6, dtype=np.float32)},
        source / "policy_preprocessor_step_5_normalizer_processor.safetensors",
    )
    save_safetensors_file(
        {"action.mean": np.zeros(6, dtype=np.float32)},
        source / "policy_postprocessor_step_0_unnormalizer_processor.safetensors",
    )

    preprocessor = object.__new__(JaxSmolVLAPreprocessor)
    preprocessor.checkpoint = source
    preprocessor.rename_map = {"observation.images.camera0": "observation.images.camera1"}
    preprocessor.config = replace(
        JaxSmolVLAConfig(),
        state_dim=20,
        action_dim=20,
        image_keys=("observation.images.camera1", "observation.images.camera2"),
        resize_height=512,
        resize_width=512,
        tokenizer_max_length=48,
        pad_language_to="max_length",
        tokenizer_name="HuggingFaceTB/SmolVLM2-500M-Video-Instruct",
    )
    preprocessor.stats = {"observation.state.mean": np.zeros(20, dtype=np.float32)}
    preprocessor.post_stats = {"action.mean": np.zeros(20, dtype=np.float32)}
    preprocessor.save_normalization_assets(destination)

    pre = json.loads((destination / "policy_preprocessor.json").read_text())
    post = json.loads((destination / "policy_postprocessor.json").read_text())
    rename_step = next(s for s in pre["steps"] if s["registry_name"] == "rename_observations_processor")
    normalizer = next(s for s in pre["steps"] if s["registry_name"] == "normalizer_processor")
    unnormalizer = next(s for s in post["steps"] if s["registry_name"] == "unnormalizer_processor")
    assert rename_step["config"]["rename_map"] == {
        "observation.images.camera0": "observation.images.camera1"
    }
    assert normalizer["config"]["features"]["observation.state"]["shape"] == [20]
    assert normalizer["config"]["features"]["action"]["shape"] == [20]
    assert set(normalizer["config"]["features"]) == {
        "observation.state",
        "action",
        "observation.images.camera1",
        "observation.images.camera2",
    }
    assert unnormalizer["config"]["features"]["action"]["shape"] == [20]
