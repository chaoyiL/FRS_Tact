from __future__ import annotations

import json
import shutil
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from safetensors import safe_open
from safetensors.numpy import save_file as save_safetensors_file

pytest.importorskip("jax")

from lerobot.policies.smolvla_jax.validation import (  # noqa: E402
    CheckpointContract,
    validate_checkpoint,
)

VT_CONTRACT = CheckpointContract(
    state_dim=20,
    action_dim=20,
    chunk_size=20,
    image_keys=("observation.images.camera1", "observation.images.camera2"),
    tactile_keys=(
        "observation.images.tactile_left_0",
        "observation.images.tactile_right_0",
        "observation.images.tactile_left_1",
        "observation.images.tactile_right_1",
    ),
    tactile_embedding_dim=512,
    tactile_num_tokens=4,
    lora_rank=16,
    vlm_lora_target_modules=("q_proj", "v_proj"),
)

SIDECAR_FILENAMES = (
    "config.json",
    "policy_preprocessor.json",
    "policy_postprocessor.json",
    "policy_preprocessor_step_5_normalizer_processor.safetensors",
    "policy_postprocessor_step_0_unnormalizer_processor.safetensors",
)
RUNTIME_TEXT_HIDDEN_SIZE = 960


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def _target_prefix(layer: int, target: str) -> str:
    block = "self_attn" if target in {"q_proj", "k_proj", "v_proj", "o_proj"} else "mlp"
    return f"model.vlm_with_expert.vlm.model.text_model.layers.{layer}.{block}.{target}"


def _target_weight_shape(target: str, hidden_size: int) -> tuple[int, int]:
    if target in {"q_proj", "o_proj"}:
        return hidden_size, hidden_size
    if target in {"k_proj", "v_proj"}:
        return hidden_size // 2, hidden_size
    if target in {"gate_proj", "up_proj"}:
        return hidden_size * 2, hidden_size
    if target == "down_proj":
        return hidden_size, hidden_size * 2
    raise ValueError(f"unsupported target: {target}")


def _model_tensors(
    *,
    contract: CheckpointContract = VT_CONTRACT,
    tactile: bool = True,
    num_layers: int = 1,
    hidden_size: int = RUNTIME_TEXT_HIDDEN_SIZE,
) -> dict[str, np.ndarray]:
    tensors: dict[str, np.ndarray] = {
        "model.state_proj.weight": np.zeros((2, 2), dtype=np.float32),
        "model.vlm_with_expert.vlm.model.text_model.embed_tokens.weight": np.zeros(
            (32, hidden_size), dtype=np.float16
        ),
    }
    if tactile:
        tensors.update(
            {
                "model.tactile_encoder.params/conv_init/kernel": np.zeros((1,), dtype=np.float32),
                "model.tactile_proj.weight": np.zeros(
                    (hidden_size, contract.tactile_embedding_dim), dtype=np.float16
                ),
                "model.tactile_proj.bias": np.zeros((hidden_size,), dtype=np.float16),
            }
        )
    for layer in range(num_layers):
        for target in contract.vlm_lora_target_modules:
            prefix = _target_prefix(layer, target)
            out_features, in_features = _target_weight_shape(target, hidden_size)
            tensors[f"{prefix}.weight"] = np.zeros((out_features, in_features), dtype=np.float16)
            tensors[f"{prefix}.lora_a"] = np.zeros((contract.lora_rank, in_features), dtype=np.float16)
            tensors[f"{prefix}.lora_b"] = np.zeros((out_features, contract.lora_rank), dtype=np.float16)
            tensors[f"{prefix}.lora_scale"] = np.asarray(1.0, dtype=np.float32)
    return tensors


def _write_bundle(path: Path, contract: CheckpointContract, *, include_weight: bool = True) -> Path:
    path.mkdir()
    use_tactile = bool(contract.tactile_keys)
    input_features = {
        "observation.state": {"type": "STATE", "shape": [contract.state_dim]},
        **{key: {"type": "VISUAL", "shape": [3, 512, 512]} for key in contract.image_keys},
    }
    _write_json(
        path / "config.json",
        {
            "chunk_size": contract.chunk_size,
            "num_vlm_layers": 1,
            "input_features": input_features,
            "output_features": {"action": {"type": "ACTION", "shape": [contract.action_dim]}},
            "use_tactile_encoder": use_tactile,
            "freeze_tactile_encoder": True,
            "tactile_keys": list(contract.tactile_keys),
            "tactile_embedding_dim": contract.tactile_embedding_dim,
            "tactile_num_tokens": contract.tactile_num_tokens,
            "lora_rank": contract.lora_rank,
            "vlm_lora_target_modules": list(contract.vlm_lora_target_modules),
            "module_modes": {
                "vision": "frozen",
                "connector": "frozen",
                "vlm_text": "lora" if contract.vlm_lora_target_modules else "frozen",
                "expert": "full",
                "action": "full",
                "state_proj": "full",
                "tactile_proj": "full" if use_tactile else "frozen",
            },
        },
    )
    normalizer_features = {
        "observation.state": {"type": "STATE", "shape": [contract.state_dim]},
        "action": {"type": "ACTION", "shape": [contract.action_dim]},
        **{key: {"type": "VISUAL", "shape": [3, 512, 512]} for key in contract.image_keys},
    }
    _write_json(
        path / "policy_preprocessor.json",
        {
            "name": "policy_preprocessor",
            "steps": [
                {
                    "registry_name": "normalizer_processor",
                    "config": {"features": normalizer_features},
                    "state_file": "policy_preprocessor_step_5_normalizer_processor.safetensors",
                }
            ],
        },
    )
    _write_json(
        path / "policy_postprocessor.json",
        {
            "name": "policy_postprocessor",
            "steps": [
                {
                    "registry_name": "unnormalizer_processor",
                    "config": {"features": {"action": {"type": "ACTION", "shape": [contract.action_dim]}}},
                    "state_file": "policy_postprocessor_step_0_unnormalizer_processor.safetensors",
                }
            ],
        },
    )
    save_safetensors_file(
        {
            "observation.state.mean": np.zeros(contract.state_dim, dtype=np.float32),
            "observation.state.std": np.ones(contract.state_dim, dtype=np.float32),
            "action.mean": np.zeros(contract.action_dim, dtype=np.float32),
            "action.std": np.ones(contract.action_dim, dtype=np.float32),
        },
        path / "policy_preprocessor_step_5_normalizer_processor.safetensors",
    )
    save_safetensors_file(
        {
            "action.mean": np.zeros(contract.action_dim, dtype=np.float32),
            "action.std": np.ones(contract.action_dim, dtype=np.float32),
        },
        path / "policy_postprocessor_step_0_unnormalizer_processor.safetensors",
    )
    if include_weight:
        save_safetensors_file(
            _model_tensors(contract=contract, tactile=use_tactile),
            path / "model.safetensors",
        )
    return path


@pytest.fixture
def vt_bundle(tmp_path: Path) -> Path:
    return _write_bundle(tmp_path / "vt-bundle", VT_CONTRACT)


@pytest.fixture
def base_sidecars(tmp_path: Path) -> Path:
    base_contract = CheckpointContract(
        state_dim=6,
        action_dim=6,
        chunk_size=50,
        image_keys=(
            "observation.images.camera1",
            "observation.images.camera2",
            "observation.images.camera3",
        ),
    )
    return _write_bundle(tmp_path / "base-sidecars", base_contract)


def test_valid_vt_bundle_passes(vt_bundle: Path) -> None:
    report = validate_checkpoint(vt_bundle, expected=VT_CONTRACT)

    assert report.ok, report.format_errors()
    report.require_valid()


def test_mixed_base_sidecars_are_rejected(vt_bundle: Path, base_sidecars: Path) -> None:
    for filename in SIDECAR_FILENAMES:
        shutil.copy2(base_sidecars / filename, vt_bundle / filename)

    report = validate_checkpoint(vt_bundle, expected=VT_CONTRACT, base_sidecars=base_sidecars)

    assert not report.ok
    assert any("byte-identical to base" in issue for issue in report.issues)


def test_wrong_config_dimensions_are_reported_together(vt_bundle: Path) -> None:
    config_path = vt_bundle / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["input_features"]["observation.state"]["shape"] = [6]
    config["output_features"]["action"]["shape"] = [7]
    config["chunk_size"] = 50
    _write_json(config_path, config)

    report = validate_checkpoint(vt_bundle, expected=VT_CONTRACT)
    errors = report.format_errors()

    assert "state_dim" in errors and "expected 20, got 6" in errors
    assert "action_dim" in errors and "expected 20, got 7" in errors
    assert "chunk_size" in errors and "expected 20, got 50" in errors


def test_missing_chunk_size_is_rejected(vt_bundle: Path) -> None:
    config_path = vt_bundle / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config.pop("chunk_size")
    _write_json(config_path, config)

    report = validate_checkpoint(vt_bundle, expected=VT_CONTRACT)

    assert not report.ok
    assert report.issues.count("config chunk_size is missing") == 1


@pytest.mark.parametrize(
    ("feature_group", "feature_key", "expected_message"),
    (
        (
            "input_features",
            "observation.state",
            "config input_features.observation.state is missing or is not an object",
        ),
        (
            "output_features",
            "action",
            "config output_features.action is missing or is not an object",
        ),
    ),
)
def test_missing_state_and_action_dimensions_have_one_parse_diagnostic(
    vt_bundle: Path,
    feature_group: str,
    feature_key: str,
    expected_message: str,
) -> None:
    config_path = vt_bundle / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config[feature_group].pop(feature_key)
    _write_json(config_path, config)

    report = validate_checkpoint(vt_bundle, expected=VT_CONTRACT)

    assert report.issues.count(expected_message) == 1


def test_processor_feature_specs_must_agree_with_config(vt_bundle: Path) -> None:
    pre_path = vt_bundle / "policy_preprocessor.json"
    pre = json.loads(pre_path.read_text(encoding="utf-8"))
    pre["steps"][0]["config"]["features"]["observation.state"]["shape"] = [6]
    _write_json(pre_path, pre)
    post_path = vt_bundle / "policy_postprocessor.json"
    post = json.loads(post_path.read_text(encoding="utf-8"))
    post["steps"][0]["config"]["features"]["action"]["shape"] = [6]
    _write_json(post_path, post)

    report = validate_checkpoint(vt_bundle, expected=VT_CONTRACT)
    errors = report.format_errors()

    assert "normalizer observation.state shape" in errors
    assert "unnormalizer action shape" in errors


def test_processor_state_files_must_point_to_canonical_assets(vt_bundle: Path) -> None:
    pre_path = vt_bundle / "policy_preprocessor.json"
    pre = json.loads(pre_path.read_text(encoding="utf-8"))
    pre["steps"][0]["state_file"] = "wrong-normalizer.safetensors"
    _write_json(pre_path, pre)

    report = validate_checkpoint(vt_bundle, expected=VT_CONTRACT)

    assert any("normalizer_processor state_file" in issue for issue in report.issues)
    assert any("wrong-normalizer.safetensors" in issue for issue in report.issues)


def test_missing_and_wrong_dimensional_stats_are_rejected(vt_bundle: Path) -> None:
    save_safetensors_file(
        {
            "observation.state.mean": np.zeros(20, dtype=np.float32),
            "action.mean": np.zeros(6, dtype=np.float32),
            "action.std": np.ones(6, dtype=np.float32),
        },
        vt_bundle / "policy_preprocessor_step_5_normalizer_processor.safetensors",
    )
    save_safetensors_file(
        {"action.mean": np.zeros(20, dtype=np.float32)},
        vt_bundle / "policy_postprocessor_step_0_unnormalizer_processor.safetensors",
    )

    report = validate_checkpoint(vt_bundle, expected=VT_CONTRACT)
    errors = report.format_errors()

    assert "missing tensor 'observation.state.std'" in errors
    assert "tensor 'action.mean' expected shape [20], got [6]" in errors
    assert "missing tensor 'action.std'" in errors


def test_missing_tactile_tensors_are_rejected(vt_bundle: Path) -> None:
    save_safetensors_file(
        _model_tensors(tactile=False),
        vt_bundle / "model.safetensors",
    )

    report = validate_checkpoint(vt_bundle, expected=VT_CONTRACT)
    errors = report.format_errors()

    assert "tactile encoder tensors" in errors
    assert "model.tactile_proj.weight" in errors
    assert "model.tactile_proj.bias" in errors


def test_fake_tactile_encoder_prefix_is_rejected(vt_bundle: Path) -> None:
    tensors = _model_tensors()
    tensors.pop("model.tactile_encoder.params/conv_init/kernel")
    tensors["model.tactile_encoder.conv.weight"] = np.zeros((1,), dtype=np.float32)
    save_safetensors_file(tensors, vt_bundle / "model.safetensors")

    report = validate_checkpoint(vt_bundle, expected=VT_CONTRACT)

    assert any("tactile encoder tensors" in issue for issue in report.issues)


def test_truncated_vlm_layers_are_rejected(vt_bundle: Path) -> None:
    config_path = vt_bundle / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["num_vlm_layers"] = 2
    _write_json(config_path, config)

    report = validate_checkpoint(vt_bundle, expected=VT_CONTRACT)

    assert any("missing VLM text layers: [1]" in issue for issue in report.issues)


def test_self_consistent_1024_hidden_checkpoint_is_rejected(vt_bundle: Path) -> None:
    save_safetensors_file(
        _model_tensors(hidden_size=1024),
        vt_bundle / "model.safetensors",
    )

    report = validate_checkpoint(vt_bundle, expected=VT_CONTRACT)

    assert any("runtime text_hidden_size expected 960, got 1024" in issue for issue in report.issues)


@pytest.mark.parametrize(
    ("target", "block"),
    (
        ("q_proj", "self_attn"),
        ("k_proj", "self_attn"),
        ("v_proj", "self_attn"),
        ("o_proj", "self_attn"),
        ("gate_proj", "mlp"),
        ("up_proj", "mlp"),
        ("down_proj", "mlp"),
    ),
)
def test_all_vlm_lora_targets_use_real_module_paths(
    tmp_path: Path,
    target: str,
    block: str,
) -> None:
    contract = replace(VT_CONTRACT, vlm_lora_target_modules=(target,))
    bundle = _write_bundle(tmp_path / target, contract)
    expected_key = f"model.vlm_with_expert.vlm.model.text_model.layers.0.{block}.{target}.weight"
    with safe_open(bundle / "model.safetensors", framework="numpy") as tensors:
        assert expected_key in set(tensors.keys())

    report = validate_checkpoint(bundle, expected=contract)

    assert report.ok, report.format_errors()


def test_missing_target_base_weight_is_rejected(vt_bundle: Path) -> None:
    tensors = _model_tensors()
    tensors.pop(f"{_target_prefix(0, 'q_proj')}.weight")
    save_safetensors_file(tensors, vt_bundle / "model.safetensors")

    report = validate_checkpoint(vt_bundle, expected=VT_CONTRACT)

    assert any("missing q_proj base weight for VLM layer 0" in issue for issue in report.issues)


def test_target_base_weight_shape_must_match_model_hidden_size(vt_bundle: Path) -> None:
    tensors = _model_tensors()
    tensors[f"{_target_prefix(0, 'q_proj')}.weight"] = np.zeros((960, 959), dtype=np.float16)
    save_safetensors_file(tensors, vt_bundle / "model.safetensors")

    report = validate_checkpoint(vt_bundle, expected=VT_CONTRACT)

    assert any("q_proj base weight input dimension expected 960, got 959" in issue for issue in report.issues)


@pytest.mark.parametrize(
    ("component", "wrong_shape", "expected_message"),
    (
        ("lora_a", (16, 961), "lora_a expected shape [16, 960], got [16, 961]"),
        ("lora_b", (961, 16), "lora_b expected shape [960, 16], got [961, 16]"),
        ("lora_scale", (1,), "lora_scale expected scalar shape [], got [1]"),
    ),
)
def test_lora_tensor_shapes_are_checked_against_base_weight(
    vt_bundle: Path,
    component: str,
    wrong_shape: tuple[int, ...],
    expected_message: str,
) -> None:
    tensors = _model_tensors()
    tensors[f"{_target_prefix(0, 'q_proj')}.{component}"] = np.zeros(wrong_shape, dtype=np.float32)
    save_safetensors_file(tensors, vt_bundle / "model.safetensors")

    report = validate_checkpoint(vt_bundle, expected=VT_CONTRACT)

    assert any(expected_message in issue for issue in report.issues)


def test_tactile_projection_shape_uses_model_hidden_size(vt_bundle: Path) -> None:
    tensors = _model_tensors()
    tensors["model.tactile_proj.weight"] = np.zeros((959, 512), dtype=np.float16)
    tensors["model.tactile_proj.bias"] = np.zeros((959,), dtype=np.float16)
    save_safetensors_file(tensors, vt_bundle / "model.safetensors")

    report = validate_checkpoint(vt_bundle, expected=VT_CONTRACT)

    assert any(
        "model.tactile_proj.weight' expected shape [960, 512], got [959, 512]" in issue
        for issue in report.issues
    )


def test_diagnostics_are_aggregated_and_require_valid_raises(vt_bundle: Path) -> None:
    config_path = vt_bundle / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["chunk_size"] = 50
    config["tactile_num_tokens"] = 2
    _write_json(config_path, config)
    (vt_bundle / "policy_preprocessor_step_5_normalizer_processor.safetensors").unlink()
    save_safetensors_file(
        {"model.state_proj.weight": np.zeros((2, 2), dtype=np.float32)},
        vt_bundle / "model.safetensors",
    )

    report = validate_checkpoint(vt_bundle, expected=VT_CONTRACT)

    assert len(report.issues) >= 6
    with pytest.raises(ValueError) as exc_info:
        report.require_valid()
    message = str(exc_info.value)
    assert "chunk_size" in message
    assert "normalizer_processor.safetensors" in message
    assert "tactile encoder tensors" in message
    assert "q_proj LoRA tensors" in message


def test_model_weight_requirement_can_be_disabled(tmp_path: Path) -> None:
    bundle = _write_bundle(tmp_path / "sidecars-only", VT_CONTRACT, include_weight=False)

    required = validate_checkpoint(bundle, expected=VT_CONTRACT)
    sidecars_only = validate_checkpoint(bundle, expected=VT_CONTRACT, require_weight=False)

    assert any("model.safetensors" in issue for issue in required.issues)
    assert sidecars_only.ok, sidecars_only.format_errors()


def test_corrupt_safetensors_are_reported_instead_of_raised(vt_bundle: Path) -> None:
    (vt_bundle / "model.safetensors").write_bytes(b"not a safetensors file")
    (vt_bundle / "policy_preprocessor_step_5_normalizer_processor.safetensors").write_bytes(b"also invalid")

    report = validate_checkpoint(vt_bundle, expected=VT_CONTRACT)

    assert not report.ok
    assert any("could not inspect model.safetensors" in issue for issue in report.issues)
    assert any("could not inspect policy_preprocessor" in issue for issue in report.issues)
