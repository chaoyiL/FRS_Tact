from pathlib import Path
import sys
import types

import numpy as np
import pytest
import yaml

import deploy_pi05_frs.deployment as deployment
from deploy_pi05_frs.deployment import (
    load_deployment_config,
    make_policy_config,
    make_server_config,
    prepare_observation,
)


@pytest.fixture(autouse=True)
def _isolate_frs_validation(monkeypatch):
    """Keep config tests runnable without importing the JAX/FRS runtime on Python 3.11."""

    def validate(config):
        required = {
            "checkpoint",
            "tactile_encoder_checkpoint",
            "tactile_keys",
            "tactile_window_divisor",
            "reverse_steps",
            "reverse_solver",
            "decode_steps",
            "decode_solver",
        }
        missing = sorted(required - set(config["frs"]))
        if missing:
            raise ValueError(f"missing FRS config values: {missing}")

    monkeypatch.setattr(deployment, "_validate_frs_config_section", validate)
    policy_module = types.ModuleType("deploy_pi05_frs.policy")

    class FakePi05DeploymentConfig:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    policy_module.Pi05DeploymentConfig = FakePi05DeploymentConfig
    monkeypatch.setitem(sys.modules, "deploy_pi05_frs.policy", policy_module)


def _frs_config() -> dict:
    return {
        "enabled": True,
        "checkpoint": "/models/frs/best",
        "tactile_encoder_checkpoint": "/models/tactile_encoder",
        "tactile_keys": ["observation.images.tactile_left_0"],
        "tactile_window_divisor": 5,
        "history_stride": 3,
        "reverse_steps": 50,
        "reverse_solver": "slerpflow",
        "decode_steps": 10,
        "decode_solver": "fireflow",
        "steering_protection_interval_s": None,
        "temporal_ensemble_coeff": 0.1,
        "gripper_gain": {"threshold": 0.1, "gain": 0.05},
        "verify_source_checkpoint_fingerprint": False,
        "max_normalized_action_abs": 8.0,
        "max_normalized_delta_rms": 4.0,
    }


def _config(*, include_frs: bool = True) -> dict:
    config = {
        "checkpoint": "/models/pi05/10000",
        "seed": 0,
        "num_steps": 10,
        "model": {
            "action_dim": 32,
            "action_horizon": 50,
            "state_dim": 20,
            "robot_action_dim": 20,
            "camera_map": {
                "left_wrist_0_rgb": "observation.images.camera0",
                "right_wrist_0_rgb": "observation.images.camera1",
            },
            "empty_cameras": ["base_0_rgb"],
        },
        "norm_stats": {
            "dir": "/models/pi05/10000/assets",
            "asset_id": "pick_tube",
            "use_quantile_norm": True,
        },
        "profiles": {
            "pi05": {"data_type": "vision", "observation_output_dir": "outputs/pi05"},
            "frs": {"data_type": "vitac", "observation_output_dir": "outputs/frs"},
        },
        "connection": {"address": "127.0.0.1", "port": 26421, "action_ack_timeout_s": 30.0},
        "observation": {
            "language_prompt": "pick",
            "single_arm_mode": False,
            "no_state_obs_mode": False,
        },
        "control": {
            "control_frequency": 20.0,
            "controller_frequency": 80.0,
            "action_horizon": 50,
            "steps_per_inference": 50,
        },
        "runtime": {"warmup_runs": 1},
        "logging": {"save_observations": False, "save_every": 1, "queue_size": 2},
    }
    if include_frs:
        config["frs"] = _frs_config()
    return config


def _write(tmp_path: Path, config: dict) -> Path:
    path = tmp_path / "deploy.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return path


@pytest.mark.parametrize(
    ("mode", "data_type", "output"),
    [("pi05", "vision", "outputs/pi05"), ("frs", "vitac", "outputs/frs")],
)
def test_load_deployment_config_expands_profile(tmp_path, mode, data_type, output):
    loaded = load_deployment_config(_write(tmp_path, _config()), mode)

    assert loaded["observation"]["data_type"] == data_type
    assert loaded["logging"]["output_dir"] == output
    assert loaded["checkpoint"] == "/models/pi05/10000"


def test_plain_profile_does_not_require_frs_section(tmp_path):
    loaded = load_deployment_config(_write(tmp_path, _config(include_frs=False)), "pi05")

    assert loaded["observation"]["data_type"] == "vision"


def test_plain_server_config_uses_legacy_protocol(tmp_path):
    loaded = load_deployment_config(_write(tmp_path, _config()), "pi05")

    server = make_server_config(loaded, mode="pi05")

    assert server["data_type"] == "vision"
    assert server["action_horizon"] == 50
    assert "execution_protocol" not in server


def test_frs_server_config_requires_runtime(tmp_path):
    loaded = load_deployment_config(_write(tmp_path, _config()), "frs")

    with pytest.raises(ValueError, match="frs_runtime"):
        make_server_config(loaded, mode="frs")


def test_rejects_horizon_drift_before_model_load(tmp_path):
    config = _config()
    config["control"]["action_horizon"] = 49

    with pytest.raises(ValueError, match="action_horizon"):
        load_deployment_config(_write(tmp_path, config), "pi05")


@pytest.mark.parametrize("mode", ["", "pi0", "unknown"])
def test_rejects_unknown_mode(tmp_path, mode):
    with pytest.raises(ValueError, match="unsupported deployment mode"):
        load_deployment_config(_write(tmp_path, _config()), mode)


def test_rejects_missing_selected_profile(tmp_path):
    config = _config()
    del config["profiles"]["pi05"]

    with pytest.raises(ValueError, match="pi05"):
        load_deployment_config(_write(tmp_path, config), "pi05")


@pytest.mark.parametrize(("mode", "data_type"), [("pi05", "vitac"), ("frs", "vision")])
def test_rejects_profile_with_wrong_data_type(tmp_path, mode, data_type):
    config = _config()
    config["profiles"][mode]["data_type"] = data_type

    with pytest.raises(ValueError, match=f"profiles.{mode}.data_type"):
        load_deployment_config(_write(tmp_path, config), mode)


@pytest.mark.parametrize("steps", [0, 51])
def test_rejects_steps_per_inference_outside_action_horizon(tmp_path, steps):
    config = _config()
    config["control"]["steps_per_inference"] = steps

    with pytest.raises(ValueError, match="steps_per_inference"):
        load_deployment_config(_write(tmp_path, config), "pi05")


def test_frs_mode_keeps_strict_frs_config_validation(tmp_path):
    config = _config()
    del config["frs"]["checkpoint"]

    with pytest.raises(ValueError, match="missing FRS config values"):
        load_deployment_config(_write(tmp_path, config), "frs")


@pytest.mark.parametrize(
    "observation",
    [
        {"observation.state": np.zeros(20, dtype=np.float32)},
        {
            "observation.state": np.zeros(19, dtype=np.float32),
            "observation.images.camera0": np.zeros((2, 3, 3), dtype=np.uint8),
        },
        {
            "observation.state": np.full(20, np.nan, dtype=np.float32),
            "observation.images.camera0": np.zeros((2, 3, 3), dtype=np.uint8),
        },
        {
            "observation.state": np.zeros(20, dtype=np.float32),
            "observation.images.camera0": np.zeros((2, 3), dtype=np.uint8),
        },
    ],
)
def test_prepare_observation_rejects_invalid_shapes(observation):
    with pytest.raises(ValueError):
        prepare_observation(
            observation,
            state_dim=20,
            image_keys=("observation.images.camera0",),
        )


def test_make_policy_config_resolves_relative_asset_paths(tmp_path):
    config = _config()
    config["checkpoint"] = "checkpoint"
    config["norm_stats"]["dir"] = "checkpoint/assets"
    config_path = _write(tmp_path, config)
    (tmp_path / "checkpoint").mkdir()
    (tmp_path / "checkpoint" / "assets").mkdir()

    policy = make_policy_config(load_deployment_config(config_path, "pi05"), config_path)

    assert policy.checkpoint == str((tmp_path / "checkpoint").resolve())
    assert policy.assets_dir == str((tmp_path / "checkpoint" / "assets").resolve())
