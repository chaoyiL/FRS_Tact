from pathlib import Path

import pytest

from deploy_deco.artifact import load_sidecar
from deploy_deco.config import (
    load_config,
    make_server_config,
    validate_artifact_contract,
)

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "deploy_deco" / "configs" / "deploy_deco.yaml"


def test_checked_in_config_matches_checked_in_external_artifact():
    config = load_config(CONFIG)
    metadata = load_sidecar(config["checkpoint"])
    validate_artifact_contract(config, metadata)
    server = make_server_config(config)
    assert server["observation_profile"] == "deco_vision_224"
    assert server["action_horizon"] == 32
    assert server["steps_per_inference"] == 8
    assert "execution_protocol" not in server


def test_training_frequency_mismatch_is_rejected():
    config = load_config(CONFIG)
    config["control"]["control_frequency"] = 20.0
    metadata = load_sidecar(config["checkpoint"])
    with pytest.raises(ValueError, match="training frequency"):
        validate_artifact_contract(config, metadata)
