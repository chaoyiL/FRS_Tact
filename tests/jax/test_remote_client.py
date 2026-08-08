from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from deploy_smolvla.remote_client import DEFAULT_CONFIG, _checkpoint_contract


def _config_contract() -> tuple[dict[str, object], dict[str, int]]:
    raw = {
        "state_dim": 20,
        "action_dim": 20,
        "chunk_size": 10,
        "image_keys": ["camera1", "camera2"],
        "tactile_keys": [],
        "tactile_embedding_dim": 512,
        "tactile_num_tokens": 0,
        "tactile_token_repeat_factor": 1,
        "lora_rank": 0,
        "vlm_lora_target_modules": [],
    }
    return {"checkpoint_contract": raw}, {"action_horizon": 10}


def test_legacy_deploy_contract_missing_compute_dtype_defaults_bfloat16() -> None:
    config, control = _config_contract()

    contract = _checkpoint_contract(config, control)

    assert contract.trainable_compute_dtype == "bfloat16"


@pytest.mark.parametrize("invalid", ["float32", "float16", "BF16", "", 16, None])
def test_deploy_contract_rejects_invalid_compute_dtype(invalid: object) -> None:
    config, control = _config_contract()
    config["checkpoint_contract"]["trainable_compute_dtype"] = invalid  # type: ignore[index]

    with pytest.raises(ValueError, match="checkpoint_contract.trainable_compute_dtype"):
        _checkpoint_contract(config, control)


def test_default_deployment_config_pins_bfloat16_compute_contract() -> None:
    raw = yaml.safe_load(Path(DEFAULT_CONFIG).read_text(encoding="utf-8"))

    assert raw["checkpoint_contract"]["trainable_compute_dtype"] == "bfloat16"
