"""Pure-vision SmolVLA deployment using the official PyTorch LeRobot policy."""

from __future__ import annotations

import os
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

# FRS_Tact contains a deliberately small local ``lerobot`` package for JAX.
# Pure-vision deployment must import the official PyTorch LeRobot installation
# from the selected Python environment (normally the same environment as VB3).
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_removed_sys_paths: list[tuple[int, str]] = []
for _index in range(len(sys.path) - 1, -1, -1):
    _entry = sys.path[_index]
    try:
        _resolved = Path(_entry or os.getcwd()).resolve()
    except OSError:
        continue
    if _resolved == _PROJECT_ROOT:
        _removed_sys_paths.append((_index, sys.path.pop(_index)))
try:
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies import make_pre_post_processors
    from lerobot.policies.smolvla import SmolVLAPolicy
    from lerobot.policies.utils import prepare_observation_for_inference
    from peft import PeftConfig, PeftModel
except ImportError as error:
    raise ImportError(
        "PyTorch SmolVLA requires the official LeRobot+PEFT environment used by VB3; "
        "set FRS_PYTHON to that environment's Python executable"
    ) from error
finally:
    for _index, _entry in sorted(_removed_sys_paths):
        sys.path.insert(_index, _entry)

from .bridge_client import RobotBridgeClient

SMOLVLA_OBSERVATION_PROFILE = "smolvla_vision_256"
SINGLE_RIGHT_ARM_PROFILE = "single-right-arm-7x10"


def _section(config: Mapping[str, Any], name: str) -> dict[str, Any]:
    value = config.get(name)
    if not isinstance(value, dict):
        raise ValueError(f"Missing YAML section: {name}")
    return value


def _load_config(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as file:
        config = yaml.safe_load(file) or {}
    if not isinstance(config, dict):
        raise ValueError(f"config root must be a mapping: {path}")
    if config.get("backend") != "pytorch_smolvla":
        raise ValueError("PyTorch runtime requires backend: pytorch_smolvla")
    for name in ("connection", "observation", "control", "runtime"):
        _section(config, name)
    if "checkpoint" not in config:
        raise ValueError("Missing config value: checkpoint")
    if str(_section(config, "observation").get("data_type")) != "vision":
        raise ValueError("PyTorch SmolVLA deployment supports vision mode only")
    return config


def _resolve_checkpoint(value: str, config_path: Path) -> str:
    checkpoint = Path(value).expanduser()
    if checkpoint.is_absolute():
        return str(checkpoint)
    relative = (config_path.parent / checkpoint).resolve()
    return str(relative) if relative.exists() else value


def _load_policy(checkpoint: str, *, revision: str | None, allow_download: bool):
    load_kwargs: dict[str, Any] = {}
    if revision is not None:
        load_kwargs["revision"] = revision
    load_kwargs["local_files_only"] = not allow_download
    checkpoint_path = Path(checkpoint)
    if not checkpoint_path.is_dir():
        return SmolVLAPolicy.from_pretrained(checkpoint, **load_kwargs)

    adapter_config = checkpoint_path / "adapter_config.json"
    adapter_weights = checkpoint_path / "adapter_model.safetensors"
    if adapter_config.is_file() != adapter_weights.is_file():
        raise FileNotFoundError(f"Incomplete PEFT adapter checkpoint: {checkpoint}")
    if not adapter_config.is_file():
        return SmolVLAPolicy.from_pretrained(checkpoint, **load_kwargs)

    policy_config = PreTrainedConfig.from_pretrained(checkpoint)
    peft_config = PeftConfig.from_pretrained(checkpoint)
    base_model = str(peft_config.base_model_name_or_path or "").strip()
    if not base_model:
        raise ValueError(f"PEFT adapter has no base_model_name_or_path: {checkpoint}")
    policy = SmolVLAPolicy.from_pretrained(base_model, config=policy_config, **load_kwargs)
    return PeftModel.from_pretrained(
        policy,
        checkpoint,
        config=peft_config,
        is_trainable=False,
    )


def _feature_shape(feature: Any) -> tuple[int, ...]:
    shape = getattr(feature, "shape", None)
    if shape is None and isinstance(feature, Mapping):
        shape = feature.get("shape")
    return tuple(int(value) for value in (shape or ()))


def _policy_contract(policy: Any) -> tuple[int, int, tuple[str, ...]]:
    config = policy.config
    input_features = getattr(config, "input_features", {}) or {}
    output_features = getattr(config, "output_features", {}) or {}
    state_feature = getattr(config, "robot_state_feature", None)
    if state_feature is None and isinstance(input_features, Mapping):
        state_feature = input_features.get("observation.state")
    action_feature = getattr(config, "action_feature", None)
    if action_feature is None and isinstance(output_features, Mapping):
        action_feature = output_features.get("action")
    state_shape = _feature_shape(state_feature)
    action_shape = _feature_shape(action_feature)
    if len(state_shape) != 1 or len(action_shape) != 1:
        raise ValueError(
            f"checkpoint must declare 1D state/action features, got {state_shape}/{action_shape}"
        )
    image_keys = tuple(
        str(key)
        for key in input_features
        if str(key).startswith("observation.images.")
    )
    if not image_keys:
        raise ValueError("checkpoint does not declare any visual observation keys")
    return state_shape[0], action_shape[0], image_keys


def _resolve_token(connection: Mapping[str, Any]) -> str | None:
    token_env = str(connection.get("token_env", "")).strip()
    token = os.environ.get(token_env) if token_env else None
    token = token or (str(connection.get("token", "")).strip() or None)
    if bool(connection.get("require_token", False)) and not token:
        raise ValueError(f"Required authentication token is missing: {token_env}")
    return token


def _prepare_frame(
    observation: Mapping[str, Any],
    *,
    task: str,
    device: torch.device,
    state_dim: int,
    model_image_keys: Sequence[str],
    rename_map: Mapping[str, str],
) -> dict[str, Any]:
    state = np.asarray(observation.get("observation.state"))
    if state.shape != (state_dim,) or not np.isfinite(state).all():
        raise ValueError(f"expected finite {state_dim}D observation.state, got {state.shape}")
    reverse = {model: robot for robot, model in rename_map.items()}
    frame: dict[str, Any] = {"observation.state": state.copy()}
    for model_key in model_image_keys:
        robot_key = reverse.get(model_key, model_key)
        if robot_key not in observation:
            raise ValueError(f"robot observation is missing image key: {robot_key}")
        image = np.asarray(observation[robot_key])
        if image.ndim != 3 or image.shape[-1] != 3:
            raise ValueError(f"{robot_key} must be HWC RGB, got {image.shape}")
        frame[model_key] = image.copy()
    return prepare_observation_for_inference(
        frame,
        device=device,
        task=task,
        robot_type="vbvla_smolvla",
    )


@torch.inference_mode()
def _predict_chunk(policy: Any, preprocess: Any, postprocess: Any, frame: dict[str, Any]) -> np.ndarray:
    action = postprocess(policy.predict_action_chunk(preprocess(frame)))
    action = action.detach().cpu().numpy()
    expected = (1, int(policy.config.chunk_size), _feature_shape(policy.config.action_feature)[0])
    if action.shape != expected:
        raise ValueError(f"Expected PyTorch SmolVLA action shaped {expected}, got {action.shape}")
    result = action[0].astype(np.float32, copy=False)
    if not np.isfinite(result).all():
        raise ValueError("SmolVLA action contains NaN or Inf")
    return result


def run(config_path: Path, max_iterations_override: int | None = None) -> None:
    config_path = config_path.expanduser().resolve()
    config = _load_config(config_path)
    connection = _section(config, "connection")
    observation_config = _section(config, "observation")
    control = _section(config, "control")
    runtime = _section(config, "runtime")
    checkpoint = _resolve_checkpoint(str(config["checkpoint"]), config_path)
    device = torch.device(str(config.get("device", "cuda")))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("config requests CUDA but torch.cuda.is_available() is false")

    print(f"[client] Loading PyTorch SmolVLA checkpoint: {checkpoint}")
    policy = _load_policy(
        checkpoint,
        revision=None if config.get("revision") is None else str(config["revision"]),
        allow_download=bool(config.get("allow_download", False)),
    )
    policy.config.device = str(device)
    policy.to(device).eval()
    policy.reset()
    preprocess, postprocess = make_pre_post_processors(
        policy.config,
        checkpoint,
        preprocessor_overrides={"device_processor": {"device": str(device)}},
    )
    state_dim, action_dim, model_image_keys = _policy_contract(policy)
    profile = str(observation_config.get("state_action_profile", "dual-arm-20x20"))
    if profile == SINGLE_RIGHT_ARM_PROFILE and (state_dim, action_dim) != (7, 10):
        raise ValueError(
            f"right-hand checkpoint must use 7D state/10D action, got {state_dim}/{action_dim}"
        )
    horizon = int(control["action_horizon"])
    if int(policy.config.chunk_size) != horizon:
        raise ValueError(
            f"checkpoint chunk_size={policy.config.chunk_size} does not match action_horizon={horizon}"
        )
    rename_map = {str(k): str(v) for k, v in (config.get("rename_map") or {}).items()}
    robot_image_keys = tuple(
        {model: robot for robot, model in rename_map.items()}.get(key, key)
        for key in model_image_keys
    )
    print(
        f"[client] Contract: backend=pytorch profile={profile} state_dim={state_dim} "
        f"action_dim={action_dim} images={list(robot_image_keys)}"
    )

    bridge = RobotBridgeClient(
        address=str(connection["address"]),
        port=int(connection["port"]),
        token=_resolve_token(connection),
        add_port=None if connection.get("add_port") is None else bool(connection["add_port"]),
        retry_interval_s=float(connection.get("retry_interval_s", 1.0)),
        ping_interval_s=float(connection.get("ping_interval_s", 20.0)),
        ping_timeout_s=float(connection.get("ping_timeout_s", 20.0)),
    )
    bridge.send_config(
        {
            "data_type": "vision",
            "observation_profile": SMOLVLA_OBSERVATION_PROFILE,
            "language_prompt": observation_config["language_prompt"],
            "control_frequency": float(control["control_frequency"]),
            "controller_frequency": float(control["controller_frequency"]),
            "single_arm_mode": bool(observation_config["single_arm_mode"]),
            "no_state_obs_mode": bool(observation_config["no_state_obs_mode"]),
            "steps_per_inference": int(control["steps_per_inference"]),
            "action_horizon": horizon,
        }
    )
    task = str(observation_config["language_prompt"])
    timeout = float(connection.get("observation_timeout_s", 30.0))
    ack_timeout = float(connection["action_ack_timeout_s"])
    max_iterations = (
        int(runtime.get("max_iterations", 0))
        if max_iterations_override is None
        else int(max_iterations_override)
    )

    def prepare(observation: Mapping[str, Any]) -> dict[str, Any]:
        return _prepare_frame(
            observation,
            task=task,
            device=device,
            state_dim=state_dim,
            model_image_keys=model_image_keys,
            rename_map=rename_map,
        )

    try:
        warmup_seq, warmup_observation = bridge.receive_observation(timeout=timeout)
        warmup_frame = prepare(warmup_observation)
        for index in range(int(runtime.get("warmup_runs", 1))):
            started = time.perf_counter()
            _predict_chunk(policy, preprocess, postprocess, warmup_frame)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            print(
                f"[client] Warmup {index + 1}: {(time.perf_counter() - started) * 1000:.1f}ms"
            )
        print(f"[client] Warmup observation sequence: {warmup_seq}")
        if not bool(runtime.get("auto_start", False)):
            input("[client] Ready. Press Enter to send START to the robot server... ")
        bridge.send_state("start")

        iteration = 0
        while max_iterations <= 0 or iteration < max_iterations:
            obs_seq, observation = bridge.receive_observation(timeout=timeout)
            started = time.perf_counter()
            action = _predict_chunk(policy, preprocess, postprocess, prepare(observation))
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            bridge.send_action(action, obs_seq)
            bridge.receive_action_ack(obs_seq, timeout=ack_timeout)
            iteration += 1
            if iteration == 1 or iteration % 10 == 0:
                print(
                    f"[client] iter={iteration} obs_seq={obs_seq} "
                    f"inference_ms={(time.perf_counter() - started) * 1000:.1f}"
                )
    except KeyboardInterrupt:
        print("[client] Interrupted")
    finally:
        try:
            bridge.send_state("stop")
        except Exception as error:
            print(f"[client] Could not send STOP: {error}")
        bridge.close()
