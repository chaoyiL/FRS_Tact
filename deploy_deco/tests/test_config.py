import os
import shutil
import subprocess
from pathlib import Path

import pytest

from deploy_deco.artifact import load_sidecar
from deploy_deco.config import (
    deployment_profile,
    load_config,
    make_server_config,
    validate_config,
    validate_artifact_contract,
)

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "deploy_deco" / "configs" / "deploy_deco.yaml"
RIGHT_CONFIG = ROOT / "deploy_deco" / "configs" / "deploy_deco_right.yaml"


def test_checked_in_config_matches_checked_in_external_artifact():
    config = load_config(CONFIG)
    metadata = load_sidecar(config["checkpoint"])
    validate_artifact_contract(config, metadata)
    server = make_server_config(config)
    assert server["observation_profile"] == "deco_vision_224"
    assert server["action_horizon"] == 32
    assert server["steps_per_inference"] == config["control"]["steps_per_inference"]
    assert "execution_protocol" not in server


def test_server_config_hardcodes_task_zero():
    config = load_config(CONFIG)

    assert make_server_config(config)["task"] == 0


def test_checked_in_right_config_matches_insert_artifact():
    config = load_config(RIGHT_CONFIG)
    metadata = load_sidecar(config["checkpoint"])
    validate_artifact_contract(config, metadata)
    assert deployment_profile(config) == "single-right-arm-7x10"
    server = make_server_config(config)
    assert server["single_arm_mode"] is False
    assert server["steps_per_inference"] == 24


def test_right_config_rejects_bimanual_artifact():
    right = load_config(RIGHT_CONFIG)
    bimanual = load_sidecar(load_config(CONFIG)["checkpoint"])
    with pytest.raises(ValueError, match="profile"):
        validate_artifact_contract(right, bimanual)


def test_training_frequency_mismatch_is_rejected():
    config = load_config(CONFIG)
    config["control"]["control_frequency"] = 20.0
    metadata = load_sidecar(config["checkpoint"])
    with pytest.raises(ValueError, match="training frequency"):
        validate_artifact_contract(config, metadata)


def test_legacy_config_does_not_require_action_ack_timeout():
    config = load_config(CONFIG)
    config["connection"].pop("action_ack_timeout_s", None)
    validate_config(config)


def test_right_launcher_selects_right_config_and_forwards_arguments(tmp_path):
    scripts = tmp_path / "deploy_deco" / "scripts"
    scripts.mkdir(parents=True)
    right_launcher = scripts / "start_deco_right.sh"
    shutil.copy(ROOT / "deploy_deco" / "scripts" / "start_deco_right.sh", right_launcher)

    output = tmp_path / "delegated.txt"
    delegated_launcher = scripts / "start_deco.sh"
    delegated_launcher.write_text('printf "%s\\n" "$CONFIG" "$@" > "$OUTPUT"\n')
    env = {**os.environ, "OUTPUT": str(output)}
    env.pop("CONFIG", None)

    subprocess.run(
        ["bash", str(right_launcher), "--max-iterations", "1"],
        check=True,
        env=env,
    )

    assert output.read_text().splitlines() == [
        str(tmp_path / "deploy_deco" / "configs" / "deploy_deco_right.yaml"),
        "--max-iterations",
        "1",
    ]
