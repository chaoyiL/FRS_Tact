from pathlib import Path
from types import SimpleNamespace
import sys
import types

import numpy as np
import pytest
import yaml

from deploy_pi05.deployment import (
    load_deployment_config,
    make_policy_config,
    make_server_config,
    optional_bool,
    prepare_observation,
)
from deploy_pi05 import deployment


@pytest.fixture(autouse=True)
def _isolate_policy_import(monkeypatch):
    """Keep policy-config tests independent from the optional model stack."""

    policy_module = types.ModuleType("deploy_pi05.policy")

    class FakePi05DeploymentConfig:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    policy_module.Pi05DeploymentConfig = FakePi05DeploymentConfig
    monkeypatch.setitem(sys.modules, "deploy_pi05.policy", policy_module)


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


def _config(*, data_type: str = "vision", include_frs: bool = True) -> dict:
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
            "empty_cameras": [],
        },
        "norm_stats": {
            "dir": "/models/pi05/10000/assets",
            "asset_id": "pick_tube",
            "use_quantile_norm": True,
        },
        "connection": {"address": "127.0.0.1", "port": 26421, "action_ack_timeout_s": 30.0},
        "observation": {
            "data_type": data_type,
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
        "logging": {
            "save_observations": False,
            "output_dir": f"outputs/{data_type}",
            "save_every": 1,
            "queue_size": 2,
        },
    }
    if include_frs:
        config["frs"] = _frs_config()
    return config


def _write(tmp_path: Path, config: dict) -> Path:
    path = tmp_path / "deploy.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return path


def test_configure_deployment_logging_suppresses_library_info_but_keeps_warnings(
    monkeypatch,
):
    basic_config_calls = []
    library_loggers = {
        name: deployment.logging.getLogger(name)
        for name in ("jax", "orbax", "tensorstore")
    }
    previous_levels = {name: logger.level for name, logger in library_loggers.items()}

    monkeypatch.setattr(
        deployment.logging,
        "basicConfig",
        lambda **kwargs: basic_config_calls.append(kwargs),
    )
    try:
        deployment.configure_deployment_logging()

        assert basic_config_calls == [
            {
                "level": deployment.logging.WARNING,
                "format": "%(levelname)s %(message)s",
                "force": True,
            }
        ]
        assert all(
            logger.level == deployment.logging.WARNING
            for logger in library_loggers.values()
        )
    finally:
        for name, logger in library_loggers.items():
            logger.setLevel(previous_levels[name])


def test_startup_summary_shows_deployment_contract_without_token(capsys):
    config = _config()
    config["connection"].update(
        token="super-secret-token",
        token_env="VB_ROBOT_TOKEN",
    )
    policy_config = SimpleNamespace(
        checkpoint="/models/pi05/10000",
        assets_dir="/models/pi05/10000/assets",
        asset_id="pick_tube",
        camera_map={
            "left_wrist_0_rgb": "observation.images.camera0",
            "right_wrist_0_rgb": "observation.images.camera1",
        },
        state_dim=20,
        action_dim=32,
        robot_action_dim=20,
        action_horizon=50,
    )
    devices = [SimpleNamespace(platform="gpu", id=0, device_kind="NVIDIA RTX 4090")]

    deployment.print_startup_summary(
        config,
        policy_config,
        mode="frs",
        backend="gpu",
        devices=devices,
    )

    output = capsys.readouterr().out
    assert "[startup] mode=frs server=127.0.0.1:26421" in output
    assert "checkpoint=/models/pi05/10000" in output
    assert "norm_stats=/models/pi05/10000/assets/pick_tube" in output
    assert "state=20 model_action=32 robot_action=20 horizon=50" in output
    assert "seed=0 sample_steps=10 control_hz=20 controller_hz=80" in output
    assert "left_wrist_0_rgb<-observation.images.camera0" in output
    assert "jax_backend=gpu devices=[gpu:0 NVIDIA RTX 4090]" in output
    assert "frs_checkpoint=/models/frs/best" in output
    assert "tactile_encoder=/models/tactile_encoder" in output
    assert "super-secret-token" not in output
    assert "VB_ROBOT_TOKEN" not in output


@pytest.mark.parametrize(("mode", "data_type"), [("pi05", "vision"), ("frs", "vitac")])
def test_load_deployment_config_accepts_matching_standalone_mapping(tmp_path, mode, data_type):
    config = _config(data_type=data_type)
    loaded = load_deployment_config(_write(tmp_path, config), mode)

    assert loaded["observation"]["data_type"] == data_type
    assert loaded["logging"]["output_dir"] == f"outputs/{data_type}"
    assert loaded["checkpoint"] == "/models/pi05/10000"


def test_plain_standalone_config_does_not_require_frs_section(tmp_path):
    loaded = load_deployment_config(_write(tmp_path, _config(include_frs=False)), "pi05")

    assert loaded["observation"]["data_type"] == "vision"


def test_plain_server_config_uses_legacy_protocol(tmp_path):
    loaded = load_deployment_config(_write(tmp_path, _config()), "pi05")

    server = make_server_config(loaded, mode="pi05")

    assert server["data_type"] == "vision"
    assert server["action_horizon"] == 50
    assert "execution_protocol" not in server


def test_frs_server_config_requires_runtime(tmp_path):
    loaded = load_deployment_config(_write(tmp_path, _config(data_type="vitac")), "frs")

    with pytest.raises(ValueError, match="frs_runtime"):
        make_server_config(loaded, mode="frs")


def test_default_frs_config_builds_frs_server_config():
    path = Path("configs/deploy_pi05_frs.yaml")
    config = load_deployment_config(path, "frs")
    runtime = SimpleNamespace(
        config=SimpleNamespace(steering_protection_interval_s=None),
        tactile_keys=("t0", "t1", "t2", "t3"),
    )

    server = make_server_config(config, mode="frs", frs_runtime=runtime)

    assert server["data_type"] == "vitac"
    assert server["execution_protocol"] == "frs_steering_v1"
    assert server["frs_tactile_keys"] == ["t0", "t1", "t2", "t3"]


def test_rejects_horizon_drift_before_model_load(tmp_path):
    config = _config()
    config["control"]["action_horizon"] = 49

    with pytest.raises(ValueError, match="action_horizon"):
        load_deployment_config(_write(tmp_path, config), "pi05")


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("state_dim", 19),
        ("robot_action_dim", 19),
        ("action_horizon", 49),
    ],
)
def test_rejects_model_dimension_contract_drift_before_model_load(tmp_path, key, value):
    config = _config()
    config["model"][key] = value

    with pytest.raises(ValueError, match=f"model.{key}"):
        load_deployment_config(_write(tmp_path, config), "pi05")


@pytest.mark.parametrize("action_dim", [20, 32])
def test_accepts_supported_model_action_dimensions(tmp_path, action_dim):
    config = _config()
    config["model"]["action_dim"] = action_dim
    config_path = _write(tmp_path, config)

    loaded = load_deployment_config(config_path, "pi05")
    policy = make_policy_config(loaded, config_path)

    assert policy.action_dim == action_dim


@pytest.mark.parametrize("action_dim", [19, 21, 31, 33])
def test_rejects_unsupported_model_action_dimensions(tmp_path, action_dim):
    config = _config()
    config["model"]["action_dim"] = action_dim
    config_path = _write(tmp_path, config)

    with pytest.raises(ValueError, match="model.action_dim"):
        load_deployment_config(config_path, "pi05")


def test_rejects_camera_map_contract_drift_before_model_load(tmp_path):
    config = _config()
    config["model"]["camera_map"]["left_wrist_0_rgb"] = "observation.images.camera1"

    with pytest.raises(ValueError, match="model.camera_map"):
        load_deployment_config(_write(tmp_path, config), "pi05")


def test_rejects_reversed_camera_order_before_model_load(tmp_path):
    config = _config()
    config["model"]["camera_map"] = {
        "right_wrist_0_rgb": "observation.images.camera1",
        "left_wrist_0_rgb": "observation.images.camera0",
    }
    path = tmp_path / "reversed.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    with pytest.raises(ValueError, match="model.camera_map"):
        load_deployment_config(path, "pi05")


def test_rejects_empty_camera_contract_drift_before_model_load(tmp_path):
    config = _config()
    config["model"]["empty_cameras"] = ["base_0_rgb"]

    with pytest.raises(ValueError, match="model.empty_cameras"):
        load_deployment_config(_write(tmp_path, config), "pi05")


def test_default_deployment_uses_two_images_and_one_warmup():
    path = Path("configs/deploy_pi05.yaml")
    config = load_deployment_config(path, "pi05")

    assert tuple(config["model"]["camera_map"]) == (
        "left_wrist_0_rgb",
        "right_wrist_0_rgb",
    )
    assert config["model"]["empty_cameras"] == []
    assert config["runtime"]["warmup_runs"] == 1


def test_rejects_more_than_one_warmup_for_maniskill_rng_parity(tmp_path):
    config = _config()
    config["runtime"]["warmup_runs"] = 2

    with pytest.raises(ValueError, match="runtime.warmup_runs"):
        load_deployment_config(_write(tmp_path, config), "pi05")


@pytest.mark.parametrize(
    ("section_name", "key", "value"),
    [
        ("model", "action_horizon", 50.9),
        ("control", "steps_per_inference", True),
    ],
)
def test_rejects_non_integral_or_boolean_control_values(tmp_path, section_name, key, value):
    config = _config()
    config[section_name][key] = value

    with pytest.raises(ValueError, match=f"{section_name}.{key} must be an integer"):
        load_deployment_config(_write(tmp_path, config), "pi05")


@pytest.mark.parametrize(
    ("section_name", "key", "value"),
    [
        ("runtime", "auto_start", "false"),
        ("runtime", "auto_start", 1),
        ("norm_stats", "use_quantile_norm", "true"),
        ("norm_stats", "use_quantile_norm", 1),
        ("observation", "single_arm_mode", "false"),
        ("observation", "no_state_obs_mode", 0),
        ("connection", "add_port", "false"),
        ("connection", "add_port", 1),
        ("connection", "require_token", "true"),
        ("logging", "save_observations", "false"),
    ],
)
def test_rejects_pseudo_boolean_config_values(
    tmp_path, section_name, key, value
):
    config = _config()
    config[section_name][key] = value

    with pytest.raises(ValueError, match=rf"{section_name}\.{key} must be a boolean"):
        load_deployment_config(_write(tmp_path, config), "pi05")


def test_optional_add_port_preserves_none(tmp_path):
    config = _config()
    config["connection"]["add_port"] = None

    loaded = load_deployment_config(_write(tmp_path, config), "pi05")

    assert optional_bool(loaded["connection"]["add_port"]) is None


def test_rejects_boolean_frequency(tmp_path):
    config = _config()
    config["control"]["control_frequency"] = True

    with pytest.raises(ValueError, match=r"control\.control_frequency must be a number"):
        load_deployment_config(_write(tmp_path, config), "pi05")


@pytest.mark.parametrize("mode", ["", "pi0", "unknown"])
def test_rejects_unknown_mode(tmp_path, mode):
    with pytest.raises(ValueError, match="unsupported deployment mode"):
        load_deployment_config(_write(tmp_path, _config()), mode)


def test_rejects_missing_standalone_observation_data_type(tmp_path):
    config = _config()
    del config["observation"]["data_type"]

    with pytest.raises(ValueError, match="observation.data_type"):
        load_deployment_config(_write(tmp_path, config), "pi05")


@pytest.mark.parametrize(("mode", "data_type"), [("pi05", "vitac"), ("frs", "vision")])
def test_rejects_standalone_config_with_wrong_data_type(tmp_path, mode, data_type):
    config = _config(data_type=data_type)

    with pytest.raises(ValueError, match="observation.data_type"):
        load_deployment_config(_write(tmp_path, config), mode)


@pytest.mark.parametrize("steps", [0, 51])
def test_rejects_steps_per_inference_outside_action_horizon(tmp_path, steps):
    config = _config()
    config["control"]["steps_per_inference"] = steps

    with pytest.raises(ValueError, match="steps_per_inference"):
        load_deployment_config(_write(tmp_path, config), "pi05")


def test_frs_mode_keeps_strict_frs_config_validation(tmp_path):
    config = _config(data_type="vitac")
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
