from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from safetensors.numpy import save_file as save_safetensors_file

from train_smolvla import validation
from train_smolvla.validation import CheckpointContract


@pytest.mark.parametrize(
    ("target", "base_shape", "expected_message"),
    [
        ("q_proj", (959, 959), "q_proj base weight input dimension expected 960, got 959"),
        ("o_proj", (959, 959), "o_proj base weight output dimension expected 960, got 959"),
    ],
)
def test_lora_validator_rejects_self_consistent_base_matrix_with_wrong_hidden_dimension(
    tmp_path: Path,
    target: str,
    base_shape: tuple[int, int],
    expected_message: str,
) -> None:
    block = "self_attn" if target in {"q_proj", "k_proj", "v_proj", "o_proj"} else "mlp"
    prefix = f"{validation._VLM_TEXT_PREFIX}.layers.0.{block}.{target}"
    model_path = tmp_path / "model.safetensors"
    save_safetensors_file(
        {
            validation._VLM_EMBED_TOKENS: np.zeros((1, 960), dtype=np.float32),
            f"{prefix}.weight": np.zeros(base_shape, dtype=np.float32),
            f"{prefix}.lora_a": np.zeros((4, base_shape[1]), dtype=np.float32),
            f"{prefix}.lora_b": np.zeros((base_shape[0], 4), dtype=np.float32),
            f"{prefix}.lora_scale": np.asarray(1.0, dtype=np.float32),
        },
        model_path,
    )
    contract = CheckpointContract(
        state_dim=1,
        action_dim=1,
        chunk_size=1,
        image_keys=(),
        lora_rank=4,
        vlm_lora_target_modules=(target,),
    )
    config = validation._ConfigView(
        state_dim=1,
        action_dim=1,
        chunk_size=1,
        image_keys=(),
        lora_rank=4,
        vlm_lora_target_modules=(target,),
        num_vlm_layers=1,
    )
    issues: list[str] = []

    validation._validate_model(model_path, contract, config, issues)

    assert any(expected_message in issue for issue in issues)


def test_sidecar_fingerprint_oserror_is_reported_as_validation_issue(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    base = tmp_path / "base"
    checkpoint = tmp_path / "checkpoint"
    base.mkdir()
    checkpoint.mkdir()
    base_config = {
        "chunk_size": 1,
        "input_features": {"observation.state": {"shape": [1]}},
        "output_features": {"action": {"shape": [1]}},
        "lora_rank": 0,
        "vlm_lora_target_modules": [],
    }
    for directory in (base, checkpoint):
        (directory / "config.json").write_text(json.dumps(base_config), encoding="utf-8")
    issues: list[str] = []
    effective = CheckpointContract(
        state_dim=2,
        action_dim=1,
        chunk_size=1,
        image_keys=(),
    )
    monkeypatch.setattr(validation, "_sha256", lambda path: (_ for _ in ()).throw(OSError("unreadable")))

    validation._check_base_sidecars(checkpoint, base, effective, issues)

    assert any("could not fingerprint sidecar 'config.json'" in issue for issue in issues)
