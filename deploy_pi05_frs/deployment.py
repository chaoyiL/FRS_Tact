"""Shared configuration and runtime helpers for pi0.5 deployments."""

from __future__ import annotations

import copy
import json
import logging
import os
import queue
import threading
from collections.abc import Mapping, Sequence
from datetime import datetime
from numbers import Integral
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import yaml

if TYPE_CHECKING:
    from .policy import Pi05DeploymentConfig


DeploymentMode = Literal["pi05", "frs"]
LOGGER = logging.getLogger(__name__)
_MODEL_CONTRACT = {
    "state_dim": 20,
    "robot_action_dim": 20,
    "action_dim": 32,
    "action_horizon": 50,
}
_CAMERA_MAP_CONTRACT = {
    "left_wrist_0_rgb": "observation.images.camera0",
    "right_wrist_0_rgb": "observation.images.camera1",
}
_EMPTY_CAMERAS_CONTRACT = ["base_0_rgb"]


def section(config: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    """Return a required mapping-valued configuration section."""
    value = config.get(name)
    if not isinstance(value, Mapping):
        raise ValueError(f"Missing YAML section: {name}")
    return value


def required(mapping: Mapping[str, Any], key: str, where: str) -> Any:
    """Return a required value, with a configuration-oriented error message."""
    if key not in mapping:
        raise ValueError(f"Missing config value {where}.{key}")
    return mapping[key]


def _mode_data_type(mode: str) -> str:
    if mode == "pi05":
        return "vision"
    if mode == "frs":
        return "vitac"
    raise ValueError(f"unsupported deployment mode: {mode!r}")


def _validate_frs_config_section(config: Mapping[str, Any]) -> None:
    """Import the FRS-only validator only when an FRS profile is selected."""
    from .frs_runtime import validate_frs_config_section

    validate_frs_config_section(config)


def _as_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be an integer")
    return int(value)


def _as_float(value: Any, name: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a number") from error
    if not np.isfinite(parsed):
        raise ValueError(f"{name} must be finite")
    return parsed


def validate_common_config(config: Mapping[str, Any]) -> None:
    """Validate fields shared by plain and FRS pi0.5 deployment profiles."""
    model = section(config, "model")
    norm_stats = section(config, "norm_stats")
    connection = section(config, "connection")
    observation = section(config, "observation")
    control = section(config, "control")
    runtime = section(config, "runtime")
    required(config, "checkpoint", "root")

    for key in (
        "action_dim",
        "action_horizon",
        "state_dim",
        "robot_action_dim",
        "camera_map",
        "empty_cameras",
    ):
        required(model, key, "model")
    for key in ("dir", "asset_id", "use_quantile_norm"):
        required(norm_stats, key, "norm_stats")
    for key in ("address", "port", "action_ack_timeout_s"):
        required(connection, key, "connection")
    for key in ("data_type", "language_prompt", "single_arm_mode", "no_state_obs_mode"):
        required(observation, key, "observation")
    for key in ("control_frequency", "controller_frequency", "action_horizon", "steps_per_inference"):
        required(control, key, "control")

    if observation["single_arm_mode"] or observation["no_state_obs_mode"]:
        raise ValueError("pi0.5 pick_tube requires bimanual state observations")

    for key, expected in _MODEL_CONTRACT.items():
        if _as_int(model[key], f"model.{key}") != expected:
            raise ValueError(f"model.{key} must be {expected} for this pi0.5 deployment")
    if model["camera_map"] != _CAMERA_MAP_CONTRACT:
        raise ValueError("model.camera_map must match the pi0.5 deployment camera contract")
    if model["empty_cameras"] != _EMPTY_CAMERAS_CONTRACT:
        raise ValueError("model.empty_cameras must be ['base_0_rgb'] for this pi0.5 deployment")

    horizon = _MODEL_CONTRACT["action_horizon"]
    if _as_int(control["action_horizon"], "control.action_horizon") != horizon:
        raise ValueError("model/control action_horizon values must match")
    steps = _as_int(control["steps_per_inference"], "control.steps_per_inference")
    if not 1 <= steps <= horizon:
        raise ValueError("control.steps_per_inference must be within [1, action_horizon]")
    if min(
        _as_float(control["control_frequency"], "control.control_frequency"),
        _as_float(control["controller_frequency"], "control.controller_frequency"),
    ) <= 0:
        raise ValueError("control frequencies must be positive")
    if _as_int(runtime.get("warmup_runs", 1), "runtime.warmup_runs") < 1:
        raise ValueError("runtime.warmup_runs must be at least 1")
    if _as_float(connection["action_ack_timeout_s"], "connection.action_ack_timeout_s") <= 0:
        raise ValueError("connection.action_ack_timeout_s must be positive")


def load_deployment_config(path: Path, mode: DeploymentMode) -> dict[str, Any]:
    """Load one profile from the shared pi0.5 deployment YAML."""
    expected_data_type = _mode_data_type(mode)
    if not path.is_file():
        raise FileNotFoundError(f"config not found: {path}")
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError("config root must be a mapping")
    config = copy.deepcopy(payload)

    profiles = section(config, "profiles")
    profile = section(profiles, mode)
    if profile.get("data_type") != expected_data_type:
        raise ValueError(f"profiles.{mode}.data_type must be {expected_data_type!r}")

    observation = dict(section(config, "observation"))
    observation["data_type"] = expected_data_type
    config["observation"] = observation
    logging_config = dict(config.get("logging", {}) or {})
    logging_config["output_dir"] = required(profile, "observation_output_dir", f"profiles.{mode}")
    config["logging"] = logging_config

    validate_common_config(config)
    if mode == "frs":
        section(config, "frs")
        _validate_frs_config_section(config)
    return config


def _resolve_local(value: str, config_path: Path) -> str:
    if "://" in value:
        return value
    path = Path(value).expanduser()
    if path.is_absolute():
        return str(path)
    candidate = (config_path.parent / path).resolve()
    return str(candidate) if candidate.exists() else value


def make_policy_config(config: Mapping[str, Any], config_path: Path) -> Pi05DeploymentConfig:
    """Build the single pi0.5 policy contract from a loaded deployment config."""
    from .policy import Pi05DeploymentConfig

    model = section(config, "model")
    stats = section(config, "norm_stats")
    camera_map = model["camera_map"]
    if not isinstance(camera_map, Mapping):
        raise ValueError("model.camera_map must be a mapping")
    empty = model.get("empty_cameras", []) or []
    if not isinstance(empty, list):
        raise ValueError("model.empty_cameras must be a list")
    return Pi05DeploymentConfig(
        checkpoint=_resolve_local(str(config["checkpoint"]), config_path),
        assets_dir=_resolve_local(str(stats["dir"]), config_path),
        asset_id=str(stats["asset_id"]),
        camera_map={str(key): str(value) for key, value in camera_map.items()},
        empty_cameras=tuple(str(value) for value in empty),
        state_dim=_as_int(model["state_dim"], "model.state_dim"),
        robot_action_dim=_as_int(model["robot_action_dim"], "model.robot_action_dim"),
        action_dim=_as_int(model["action_dim"], "model.action_dim"),
        action_horizon=_as_int(model["action_horizon"], "model.action_horizon"),
        paligemma_variant=str(model.get("paligemma_variant", "gemma_2b_lora")),
        action_expert_variant=str(model.get("action_expert_variant", "gemma_300m_lora")),
        use_quantile_norm=bool(stats["use_quantile_norm"]),
    )


def resolve_token(connection: Mapping[str, Any]) -> str | None:
    """Resolve a bridge token, preferring the configured environment variable."""
    env_name = str(connection.get("token_env", "")).strip()
    env_token = os.environ.get(env_name) if env_name else None
    config_token = str(connection.get("token") or "").strip() or None
    token = env_token or config_token
    if bool(connection.get("require_token", False)) and not token:
        raise ValueError(f"authentication token is missing; set env {env_name} or connection.token")
    return token


def optional_bool(value: Any) -> bool | None:
    """Preserve an omitted optional boolean while normalizing present values."""
    return None if value is None else bool(value)


def prepare_observation(
    observation: Mapping[str, Any], *, state_dim: int, image_keys: Sequence[str]
) -> dict[str, Any]:
    """Copy and validate the state and RGB images needed for inference."""
    missing = [key for key in (*image_keys, "observation.state") if key not in observation]
    if missing:
        raise ValueError(f"robot observation is missing keys: {missing}")
    state = np.asarray(observation["observation.state"], dtype=np.float32)
    if state.shape != (state_dim,) or not np.isfinite(state).all():
        raise ValueError(f"robot state must be finite with shape ({state_dim},), got {state.shape}")
    prepared: dict[str, Any] = {"observation.state": state.copy()}
    for key in image_keys:
        image = np.asarray(observation[key])
        if image.ndim != 3 or image.shape[-1] != 3:
            raise ValueError(f"{key} must be HWC RGB, got {image.shape}")
        prepared[key] = image.copy()
    return prepared


def make_server_config(
    config: Mapping[str, Any], *, mode: DeploymentMode, frs_runtime: Any | None = None
) -> dict[str, Any]:
    """Build the robot-server config, adding wire fields only for FRS mode."""
    expected_data_type = _mode_data_type(mode)
    observation = section(config, "observation")
    control = section(config, "control")
    if observation.get("data_type") != expected_data_type:
        raise ValueError(f"observation.data_type must be {expected_data_type!r} for {mode}")
    result = {
        "data_type": observation["data_type"],
        "language_prompt": observation["language_prompt"],
        "control_frequency": float(control["control_frequency"]),
        "controller_frequency": float(control["controller_frequency"]),
        "single_arm_mode": bool(observation["single_arm_mode"]),
        "no_state_obs_mode": bool(observation["no_state_obs_mode"]),
        "steps_per_inference": int(control["steps_per_inference"]),
        "action_horizon": int(control["action_horizon"]),
    }
    if mode == "frs":
        if frs_runtime is None:
            raise ValueError("frs_runtime is required for FRS server config")
        result.update(
            execution_protocol="frs_steering_v1",
            steering_protection_interval_s=frs_runtime.config.steering_protection_interval_s,
            frs_tactile_keys=list(frs_runtime.tactile_keys),
        )
    return result


class ObservationSaver:
    """Save robot observations on a bounded background queue."""

    def __init__(self, config: Mapping[str, Any], image_keys: Sequence[str]) -> None:
        self.enabled = bool(config.get("save_observations", False))
        self.save_every = int(config.get("save_every", 1))
        queue_size = int(config.get("queue_size", 32))
        if self.save_every < 1 or queue_size < 1:
            raise ValueError("logging.save_every and logging.queue_size must be positive")
        self.image_keys = tuple(image_keys)
        self.output_dir: Path | None = None
        if self.enabled:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            root = Path(str(config.get("output_dir", "outputs/pi05_observations")))
            self.output_dir = root.expanduser().resolve() / timestamp
            self.output_dir.mkdir(parents=True, exist_ok=False)
            print(f"[client] Saving observations to {self.output_dir}")
        self._queue: queue.Queue[tuple[int, int, dict[str, Any]]] = queue.Queue(queue_size)
        self._thread: threading.Thread | None = None
        self._running = False
        self._dropped = 0

    def start(self) -> None:
        if not self.enabled:
            return
        self._running = True
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    def submit(self, iteration: int, obs_seq: int, observation: Mapping[str, Any]) -> None:
        if not self.enabled or iteration % self.save_every:
            return
        payload = {
            key: np.asarray(observation[key]).copy()
            for key in (*self.image_keys, "observation.state")
            if key in observation
        }
        try:
            self._queue.put_nowait((iteration, obs_seq, payload))
        except queue.Full:
            self._dropped += 1

    def _worker(self) -> None:
        while self._running or not self._queue.empty():
            try:
                item = self._queue.get(timeout=0.2)
            except queue.Empty:
                continue
            try:
                self._save(*item)
            except Exception as error:  # Saving must not stop the robot loop.
                LOGGER.warning("Could not save observation: %s", error)
            finally:
                self._queue.task_done()

    def _save(self, iteration: int, obs_seq: int, observation: Mapping[str, Any]) -> None:
        import cv2

        assert self.output_dir is not None
        step_dir = self.output_dir / f"step_{iteration:06d}"
        step_dir.mkdir()
        for key in self.image_keys:
            if key not in observation:
                continue
            image = np.asarray(observation[key])
            if image.dtype != np.uint8:
                if np.issubdtype(image.dtype, np.floating) and float(image.max()) <= 1.0:
                    image = image * 255.0
                image = np.clip(image, 0, 255).astype(np.uint8)
            name = key.replace("/", "_").replace(".", "_")
            cv2.imwrite(str(step_dir / f"{name}.jpg"), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        np.save(step_dir / "observation_state.npy", observation["observation.state"])
        (step_dir / "metadata.json").write_text(
            json.dumps({"iteration": iteration, "obs_seq": obs_seq}), encoding="utf-8"
        )

    def close(self) -> None:
        if not self.enabled:
            return
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=10.0)
        print(f"[client] Observation saver stopped; dropped={self._dropped}")
