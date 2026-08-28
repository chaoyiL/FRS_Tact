"""Configuration contract for standalone DECO deployment."""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

from deploy_deco.artifact import (
    DUAL_ARM_PROFILE,
    SINGLE_RIGHT_ARM_PROFILE,
    artifact_profile,
)

DECO_OBSERVATION_PROFILE = "deco_vision_224"


def deployment_profile(config: Mapping[str, Any]) -> str:
    model = config.get("model", {}) or {}
    if not isinstance(model, Mapping):
        raise ValueError("model must be a mapping")
    return str(model.get("state_action_profile", DUAL_ARM_PROFILE))


def section(config: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    value = config.get(name)
    if not isinstance(value, Mapping):
        raise ValueError(f"missing YAML section: {name}")
    return value


def _integer(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    return value


def _positive_float(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a number")
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a number") from error
    if result <= 0:
        raise ValueError(f"{name} must be positive")
    return result


def _boolean(value: Any, name: str) -> bool:
    if type(value) is not bool:
        raise ValueError(f"{name} must be a boolean")
    return value


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path).expanduser().resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"DECO config not found: {config_path}")
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError("DECO config root must be a mapping")
    validate_config(payload)
    payload["_config_path"] = str(config_path)
    return payload


def validate_config(config: Mapping[str, Any]) -> None:
    if not isinstance(config.get("checkpoint"), str) or not config["checkpoint"].strip():
        raise ValueError("checkpoint must be a non-empty path")
    if not isinstance(config.get("device"), str) or not config["device"].strip():
        raise ValueError("device must be a non-empty string")
    if config["device"] != "cuda:0":
        raise ValueError("the current traced DECO artifact requires device cuda:0")
    if _integer(config.get("seed", 0), "seed") < 0:
        raise ValueError("seed must be nonnegative")
    connection = section(config, "connection")
    observation = section(config, "observation")
    control = section(config, "control")
    runtime = section(config, "runtime")
    for key in ("address", "port"):
        if key not in connection:
            raise ValueError(f"missing connection.{key}")
    _integer(connection["port"], "connection.port")
    if "add_port" in connection and connection["add_port"] is not None:
        _boolean(connection["add_port"], "connection.add_port")
    _boolean(connection.get("require_token", True), "connection.require_token")
    if observation.get("data_type") != "vision":
        raise ValueError("observation.data_type must be 'vision'")
    if observation.get("observation_profile", DECO_OBSERVATION_PROFILE) != DECO_OBSERVATION_PROFILE:
        raise ValueError(f"observation_profile must be {DECO_OBSERVATION_PROFILE!r}")
    profile = deployment_profile(config)
    single_arm_mode = _boolean(
        observation.get("single_arm_mode", False), "observation.single_arm_mode"
    )
    controlled_arm = observation.get("controlled_arm")
    black_camera0 = _boolean(
        observation.get("black_camera0", False), "observation.black_camera0"
    )
    if profile == DUAL_ARM_PROFILE:
        if single_arm_mode or controlled_arm is not None:
            raise ValueError("dual-arm DECO requires bimanual observations")
        if black_camera0:
            raise ValueError("black_camera0 is only supported for single-right-arm DECO")
    elif profile == SINGLE_RIGHT_ARM_PROFILE:
        if not single_arm_mode or controlled_arm != "right":
            raise ValueError("single-right-arm DECO requires controlled_arm='right'")
    else:
        raise ValueError(f"unsupported DECO deployment profile: {profile!r}")
    if _boolean(observation.get("no_state_obs_mode", False), "observation.no_state_obs_mode"):
        raise ValueError("DECO requires the 20D state observation")
    horizon = _integer(control.get("action_horizon"), "control.action_horizon")
    steps = _integer(control.get("steps_per_inference"), "control.steps_per_inference")
    if horizon <= 0 or not 1 <= steps <= horizon:
        raise ValueError("steps_per_inference must be within [1, action_horizon]")
    control_hz = _positive_float(control.get("control_frequency"), "control.control_frequency")
    controller_hz = _positive_float(
        control.get("controller_frequency"), "control.controller_frequency"
    )
    if controller_hz < control_hz:
        raise ValueError("controller_frequency must not be lower than control_frequency")
    if _integer(runtime.get("warmup_runs", 1), "runtime.warmup_runs") < 0:
        raise ValueError("runtime.warmup_runs must be nonnegative")
    if _integer(runtime.get("max_iterations", 0), "runtime.max_iterations") < 0:
        raise ValueError("runtime.max_iterations must be nonnegative")
    _boolean(runtime.get("auto_start", False), "runtime.auto_start")


def resolve_checkpoint(config: Mapping[str, Any]) -> Path:
    checkpoint = Path(str(config["checkpoint"])).expanduser()
    if checkpoint.is_absolute():
        return checkpoint.resolve()
    config_path = Path(str(config["_config_path"]))
    return (config_path.parent / checkpoint).resolve()


def validate_artifact_contract(config: Mapping[str, Any], metadata: Mapping[str, Any]) -> None:
    artifact = artifact_profile(metadata)
    configured = deployment_profile(config)
    if artifact != configured:
        raise ValueError(
            f"DECO config profile {configured!r} does not match artifact profile {artifact!r}"
        )
    control = section(config, "control")
    horizon = int(metadata["output"]["action"][1])
    if control["action_horizon"] != horizon:
        raise ValueError(
            f"control.action_horizon={control['action_horizon']} does not match artifact {horizon}"
        )
    expected_hz = float(metadata["expected_sample_hz"])
    actual_hz = float(control["control_frequency"])
    if abs(actual_hz - expected_hz) > 1e-6:
        raise ValueError(
            f"control_frequency={actual_hz} does not match training frequency {expected_hz}"
        )


def resolve_token(connection: Mapping[str, Any]) -> str | None:
    env_name = str(connection.get("token_env", "VB_ROBOT_TOKEN")).strip()
    token = (os.environ.get(env_name) if env_name else None) or connection.get("token")
    result = str(token).strip() if token else None
    if connection.get("require_token", True) and not result:
        raise ValueError(f"robot token is missing; set {env_name} or connection.token")
    return result


def make_server_config(config: Mapping[str, Any]) -> dict[str, Any]:
    observation = section(config, "observation")
    control = section(config, "control")
    return {
        "task": 0,
        "data_type": "vision",
        "language_prompt": str(observation.get("language_prompt", "")),
        "control_frequency": float(control["control_frequency"]),
        "controller_frequency": float(control["controller_frequency"]),
        "single_arm_mode": False,
        "no_state_obs_mode": False,
        "steps_per_inference": int(control["steps_per_inference"]),
        "action_horizon": int(control["action_horizon"]),
        "observation_profile": DECO_OBSERVATION_PROFILE,
    }
