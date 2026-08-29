"""Shared configuration and runtime helpers for pi0.5 deployments."""

from __future__ import annotations

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

from .frs_config import validate_frs_config_section

if TYPE_CHECKING:
    from .policy import Pi05DeploymentConfig


DeploymentMode = Literal["pi05", "frs"]
LOGGER = logging.getLogger(__name__)
DUAL_ARM_PROFILE = "dual-arm-20x20"
SINGLE_RIGHT_ARM_PROFILE = "single-right-arm-7x10"
_STATE_ACTION_PROFILES = {
    DUAL_ARM_PROFILE: {
        "state_dim": 20,
        "robot_action_dim": 20,
        "single_arm_mode": False,
        "controlled_arm": None,
    },
    SINGLE_RIGHT_ARM_PROFILE: {
        "state_dim": 7,
        "robot_action_dim": 10,
        "single_arm_mode": True,
        "controlled_arm": "right",
    },
}
_ACTION_HORIZON = 50
_SUPPORTED_MODEL_ACTION_DIMS = frozenset({10, 20, 32})
_CAMERA_MAP_CONTRACTS = {
    DUAL_ARM_PROFILE: {
        "left_wrist_0_rgb": "observation.images.camera0",
        "right_wrist_0_rgb": "observation.images.camera1",
    },
    SINGLE_RIGHT_ARM_PROFILE: {
        "right_wrist_0_rgb": "observation.images.camera1",
    },
}
_EMPTY_CAMERAS_CONTRACT: list[str] = []
PI05_OBSERVATION_PROFILE = "pi05_vision_224"
PI05_VITAC_OBSERVATION_PROFILE = "pi05_vitac_224"


def configure_deployment_logging() -> None:
    """Hide third-party INFO chatter while preserving warnings and failures."""
    logging.basicConfig(
        level=logging.WARNING,
        format="%(levelname)s %(message)s",
        force=True,
    )
    for name in ("jax", "orbax", "tensorstore"):
        logging.getLogger(name).setLevel(logging.WARNING)


def _device_label(device: Any) -> str:
    platform = str(getattr(device, "platform", "unknown"))
    device_id = getattr(device, "id", "?")
    kind = str(getattr(device, "device_kind", "")).strip()
    return f"{platform}:{device_id}{f' {kind}' if kind else ''}"


def print_startup_summary(
    config: Mapping[str, Any],
    policy_config: Pi05DeploymentConfig,
    *,
    mode: DeploymentMode,
    backend: str,
    devices: Sequence[Any],
) -> None:
    """Print selected deployment-critical fields without exposing credentials."""
    connection = section(config, "connection")
    control = section(config, "control")
    stats_path = f"{policy_config.assets_dir.rstrip('/')}/{policy_config.asset_id}"
    cameras = ", ".join(
        f"{model_key}<-{robot_key}"
        for model_key, robot_key in policy_config.camera_map.items()
    )
    device_text = ", ".join(_device_label(device) for device in devices) or "none"
    print(
        f"[startup] mode={mode} server={connection['address']}:{connection['port']}"
    )
    print(f"[startup] checkpoint={policy_config.checkpoint}")
    print(f"[startup] norm_stats={stats_path}")
    print(
        "[startup] model "
        f"profile={policy_config.state_action_profile} "
        f"state={policy_config.state_dim} model_action={policy_config.action_dim} "
        f"robot_action={policy_config.robot_action_dim} horizon={policy_config.action_horizon}"
    )
    print(
        "[startup] inference "
        f"seed={int(config.get('seed', 0))} sample_steps={int(config.get('num_steps', 10))} "
        f"control_hz={float(control['control_frequency']):g} "
        f"controller_hz={float(control['controller_frequency']):g}"
    )
    print(f"[startup] cameras={cameras}")
    print(f"[startup] jax_backend={backend} devices=[{device_text}]")
    if mode == "frs":
        frs = section(config, "frs")
        if config.get("task", 0) == 1:
            task1 = section(config, "task1")
            print(
                "[startup] task1 "
                f"dispatch_lead_s={float(control['dispatch_lead_time_s']):g} "
                f"approach_gain={float(task1['approach_translation_gain']):g} "
                f"right_approach_gain={float(task1['right_approach_translation_gain']):g} "
                f"translation_gain={float(task1['translation_gain']):g} "
                f"left_min_lift_m={float(task1['left_min_lift_height_m']):g} "
                f"right_preclose_forward_m={float(task1['right_preclose_forward_m']):g}"
            )
        print(f"[startup] frs_checkpoint={frs['checkpoint']}")
        print(f"[startup] tactile_encoder={frs['tactile_encoder_checkpoint']}")
        print(f"[startup] tactile_inputs={list(frs['tactile_keys'])}")


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
    """Compatibility hook around the dependency-light FRS validator."""
    validate_frs_config_section(config)


def _as_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be an integer")
    return int(value)


def _as_float(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a number")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a number") from error
    if not np.isfinite(parsed):
        raise ValueError(f"{name} must be finite")
    return parsed


def _as_bool(value: Any, name: str) -> bool:
    if type(value) is not bool:
        raise ValueError(f"{name} must be a boolean")
    return value


def validate_common_config(config: Mapping[str, Any]) -> None:
    """Validate fields shared by plain and FRS pi0.5 deployment profiles."""
    model = section(config, "model")
    norm_stats = section(config, "norm_stats")
    connection = section(config, "connection")
    observation = section(config, "observation")
    control = section(config, "control")
    runtime = section(config, "runtime")
    logging_config = config.get("logging", {}) or {}
    if not isinstance(logging_config, Mapping):
        raise ValueError("Missing YAML section: logging")
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

    _as_bool(norm_stats["use_quantile_norm"], "norm_stats.use_quantile_norm")
    single_arm_mode = _as_bool(
        observation["single_arm_mode"], "observation.single_arm_mode"
    )
    no_state_obs_mode = _as_bool(
        observation["no_state_obs_mode"], "observation.no_state_obs_mode"
    )
    if no_state_obs_mode:
        raise ValueError("pi0.5 deployment requires state observations")
    _as_bool(runtime.get("auto_start", False), "runtime.auto_start")
    if "add_port" in connection and connection["add_port"] is not None:
        _as_bool(connection["add_port"], "connection.add_port")
    if "require_token" in connection:
        _as_bool(connection["require_token"], "connection.require_token")
    if "save_observations" in logging_config:
        _as_bool(logging_config["save_observations"], "logging.save_observations")

    profile_name = str(model.get("state_action_profile", DUAL_ARM_PROFILE))
    try:
        profile = _STATE_ACTION_PROFILES[profile_name]
    except KeyError as error:
        raise ValueError(
            "model.state_action_profile must be one of "
            f"{sorted(_STATE_ACTION_PROFILES)}"
        ) from error
    for key in ("state_dim", "robot_action_dim"):
        expected = profile[key]
        if _as_int(model[key], f"model.{key}") != expected:
            raise ValueError(
                f"model.{key} must be {expected} for state_action_profile={profile_name!r}"
            )
    if single_arm_mode is not profile["single_arm_mode"]:
        raise ValueError(
            "observation.single_arm_mode does not match "
            f"model.state_action_profile={profile_name!r}"
        )
    controlled_arm = observation.get("controlled_arm")
    if controlled_arm != profile["controlled_arm"]:
        raise ValueError(
            f"observation.controlled_arm must be {profile['controlled_arm']!r} for "
            f"state_action_profile={profile_name!r}"
        )
    if "black_camera0" in observation:
        raise ValueError(
            "observation.black_camera0 is not supported; single-right-arm Pi0.5 "
            "must expose only the real right-wrist camera"
        )
    action_dim = _as_int(model["action_dim"], "model.action_dim")
    if action_dim not in _SUPPORTED_MODEL_ACTION_DIMS:
        supported = ", ".join(
            str(value) for value in sorted(_SUPPORTED_MODEL_ACTION_DIMS)
        )
        raise ValueError(
            f"model.action_dim must be one of {{{supported}}} for this pi0.5 deployment"
        )
    camera_map = model["camera_map"]
    if not isinstance(camera_map, Mapping):
        raise ValueError("model.camera_map must be a mapping")
    expected_camera_map = _CAMERA_MAP_CONTRACTS[profile_name]
    if dict(camera_map) != expected_camera_map:
        if profile_name == SINGLE_RIGHT_ARM_PROFILE:
            raise ValueError(
                "single-right-arm Pi0.5 camera_map must contain only "
                "right_wrist_0_rgb -> observation.images.camera1"
            )
        raise ValueError("model.camera_map must match the dual-arm Pi0.5 camera contract")
    if model["empty_cameras"] != _EMPTY_CAMERAS_CONTRACT:
        raise ValueError("model.empty_cameras must be empty for this Pi0.5 deployment")

    horizon = _ACTION_HORIZON
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
    if _as_int(runtime.get("warmup_runs", 1), "runtime.warmup_runs") != 1:
        raise ValueError("runtime.warmup_runs must be 1 for ManiSkill RNG parity")
    if _as_int(runtime.get("max_iterations", 0), "runtime.max_iterations") < 0:
        raise ValueError("runtime.max_iterations must be non-negative")
    if _as_float(connection["action_ack_timeout_s"], "connection.action_ack_timeout_s") <= 0:
        raise ValueError("connection.action_ack_timeout_s must be positive")
    _as_int(connection["port"], "connection.port")
    for key, default in (
        ("retry_interval_s", 1.0),
        ("ping_interval_s", 20.0),
        ("ping_timeout_s", 20.0),
        ("observation_timeout_s", 30.0),
    ):
        _as_float(connection.get(key, default), f"connection.{key}")
    _as_int(config.get("seed", 0), "root.seed")
    if _as_int(config.get("num_steps", 10), "root.num_steps") <= 0:
        raise ValueError("root.num_steps must be positive")
    if _as_int(logging_config.get("save_every", 1), "logging.save_every") < 1:
        raise ValueError("logging.save_every must be positive")
    if _as_int(logging_config.get("queue_size", 32), "logging.queue_size") < 1:
        raise ValueError("logging.queue_size must be positive")


def load_deployment_config(path: Path, mode: DeploymentMode) -> dict[str, Any]:
    """Load a standalone pi0.5 deployment YAML for the requested mode."""
    if not path.is_file():
        raise FileNotFoundError(f"config not found: {path}")
    return load_deployment_config_bytes(path.read_bytes(), mode)


def load_deployment_config_bytes(payload: bytes, mode: DeploymentMode) -> dict[str, Any]:
    """Load a standalone pi0.5 deployment YAML from its original bytes."""
    expected_data_type = _mode_data_type(mode)
    payload = yaml.safe_load(payload) or {}
    if not isinstance(payload, dict):
        raise ValueError("config root must be a mapping")
    config = payload
    observation = section(config, "observation")
    if observation.get("data_type") != expected_data_type:
        raise ValueError(
            f"observation.data_type must be {expected_data_type!r} for mode {mode!r}"
        )

    validate_common_config(config)
    if mode == "frs":
        if section(config, "runtime").get("auto_start", False):
            raise ValueError(
                "FRS runtime.auto_start must be false so the operator can verify "
                "the shared YAML SHA256 before START"
            )
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
        state_action_profile=str(model.get("state_action_profile", DUAL_ARM_PROFILE)),
        state_dim=_as_int(model["state_dim"], "model.state_dim"),
        robot_action_dim=_as_int(model["robot_action_dim"], "model.robot_action_dim"),
        action_dim=_as_int(model["action_dim"], "model.action_dim"),
        action_horizon=_as_int(model["action_horizon"], "model.action_horizon"),
        paligemma_variant=str(model.get("paligemma_variant", "gemma_2b_lora")),
        action_expert_variant=str(model.get("action_expert_variant", "gemma_300m_lora")),
        use_quantile_norm=_as_bool(stats["use_quantile_norm"], "norm_stats.use_quantile_norm"),
    )


def resolve_token(connection: Mapping[str, Any]) -> str | None:
    """Resolve a bridge token, preferring the configured environment variable."""
    env_name = str(connection.get("token_env", "")).strip()
    env_token = os.environ.get(env_name) if env_name else None
    config_token = str(connection.get("token") or "").strip() or None
    token = env_token or config_token
    if _as_bool(connection.get("require_token", False), "connection.require_token") and not token:
        raise ValueError(f"authentication token is missing; set env {env_name} or connection.token")
    return token


def optional_bool(value: Any, name: str = "value") -> bool | None:
    """Preserve an omitted optional boolean while normalizing present values."""
    return None if value is None else _as_bool(value, name)


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
    """Build the robot-server config for the selected pi0.5 deployment mode."""
    expected_data_type = _mode_data_type(mode)
    observation = section(config, "observation")
    control = section(config, "control")
    if observation.get("data_type") != expected_data_type:
        raise ValueError(f"observation.data_type must be {expected_data_type!r} for {mode}")
    model_value = config.get("model")
    profile_name = (
        DUAL_ARM_PROFILE
        if model_value is None
        else str(section(config, "model").get("state_action_profile", DUAL_ARM_PROFILE))
    )
    configured_single_arm_mode = _as_bool(
        observation["single_arm_mode"], "observation.single_arm_mode"
    )
    wire_single_arm_mode = (
        False
        if profile_name == SINGLE_RIGHT_ARM_PROFILE
        else configured_single_arm_mode
    )
    result = {
        "data_type": observation["data_type"],
        "language_prompt": observation["language_prompt"],
        "control_frequency": _as_float(
            control["control_frequency"], "control.control_frequency"
        ),
        "controller_frequency": _as_float(
            control["controller_frequency"], "control.controller_frequency"
        ),
        "single_arm_mode": wire_single_arm_mode,
        "no_state_obs_mode": _as_bool(
            observation["no_state_obs_mode"], "observation.no_state_obs_mode"
        ),
        "steps_per_inference": _as_int(
            control["steps_per_inference"], "control.steps_per_inference"
        ),
        "action_horizon": _as_int(control["action_horizon"], "control.action_horizon"),
        "observation_profile": (
            PI05_OBSERVATION_PROFILE
            if mode == "pi05"
            else PI05_VITAC_OBSERVATION_PROFILE
        ),
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
        self.enabled = _as_bool(
            config.get("save_observations", False), "logging.save_observations"
        )
        self.save_every = _as_int(config.get("save_every", 1), "logging.save_every")
        queue_size = _as_int(config.get("queue_size", 32), "logging.queue_size")
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


def start_observation_saver(
    config: Mapping[str, Any],
    image_keys: Sequence[str],
    *,
    saver_factory: Any = ObservationSaver,
    logger: logging.Logger = LOGGER,
) -> ObservationSaver | None:
    """Construct and start optional observation logging without blocking control."""
    saver: ObservationSaver | None = None
    try:
        saver = saver_factory(config, image_keys)
        saver.start()
        return saver
    except Exception as error:
        logger.warning("Could not start observation saver: %s", error)
        if saver is not None:
            try:
                saver.close()
            except Exception as close_error:
                logger.warning("Could not close failed observation saver: %s", close_error)
        return None


def submit_observation(
    saver: ObservationSaver | None,
    iteration: int,
    obs_seq: int,
    observation: Mapping[str, Any],
    *,
    logger: logging.Logger = LOGGER,
) -> None:
    """Submit one observation without allowing logging failures into control."""
    if saver is None:
        return
    try:
        saver.submit(iteration, obs_seq, observation)
    except Exception as error:
        logger.warning("Could not queue observation for saving: %s", error)


def cleanup_deployment_resources(
    bridge: Any | None,
    saver: ObservationSaver | None,
    *,
    logger: logging.Logger = LOGGER,
) -> None:
    """Best-effort STOP, saver drain, and socket close in safety order."""
    if bridge is not None:
        try:
            bridge.send_state("stop")
        except Exception as error:
            logger.warning("Could not send STOP: %s", error)
    if saver is not None:
        try:
            saver.close()
        except Exception as error:
            logger.warning("Could not close observation saver: %s", error)
    if bridge is not None:
        try:
            bridge.close()
        except Exception as error:
            logger.warning("Could not close robot bridge: %s", error)
