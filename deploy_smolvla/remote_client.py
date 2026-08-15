"""Run a JAX SmolVLA checkpoint as the remote policy client for ``vb3_robot_server``."""

from __future__ import annotations

import argparse
import json
import logging
import os
import queue
import threading
import time
from collections.abc import Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import jax
import numpy as np
import yaml

from train_smolvla import JaxSmolVLAPolicy
from train_smolvla.checkpoint import resolve_checkpoint
from train_vtsmolvla import VTJaxSmolVLAPolicy
from train_vtsmolvla.validation import (
    CheckpointContract,
    contract_from_checkpoint,
    validate_checkpoint,
)

from .bridge_client import RobotBridgeClient
from .frs_protocol import FRSChunkEnd, FRSChunkStart, FRSSteerAck, FRSSteerRequest
from .frs_runtime import (
    FRSChunkReady,
    FRSRuntime,
    FRSSteerResult,
    validate_frs_config_section,
)

DEFAULT_CONFIG = Path(__file__).resolve().parent / "configs" / "deploy_smolvla_jax.yaml"
SUPPORTED_DATA_TYPES = frozenset({"vision", "vitac"})
Policy = JaxSmolVLAPolicy | VTJaxSmolVLAPolicy
LOGGER = logging.getLogger(__name__)


class ObservationSaver:
    """Save received observations asynchronously without blocking inference."""

    def __init__(self, config: dict[str, Any], image_keys: Sequence[str]) -> None:
        self.enabled = bool(config.get("save_observations", False))
        self.save_every = int(config.get("save_every", 1))
        queue_size = int(config.get("queue_size", 32))
        if self.save_every < 1 or queue_size < 1:
            raise ValueError("logging.save_every and logging.queue_size must be positive")
        self.image_keys = tuple(image_keys)

        self.output_dir: Path | None = None
        if self.enabled:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.output_dir = Path(
                str(config.get("output_dir", "outputs/vb3_remote_observations"))
            )
            self.output_dir = self.output_dir.expanduser().resolve() / timestamp
            self.output_dir.mkdir(parents=True, exist_ok=False)
            print(f"[client] Saving observations to {self.output_dir}")

        self._queue: queue.Queue[tuple[int, int, dict[str, Any]]] = queue.Queue(
            maxsize=queue_size
        )
        self._thread: threading.Thread | None = None
        self._running = False
        self._dropped = 0

    def start(self) -> None:
        if not self.enabled:
            return
        self._running = True
        self._thread = threading.Thread(
            target=self._worker, name="VBVLAObservationSaver", daemon=True
        )
        self._thread.start()

    def submit(self, iteration: int, obs_seq: int, observation: dict[str, Any]) -> None:
        if not self.enabled or iteration % self.save_every != 0:
            return
        payload = {
            key: np.asarray(observation[key]).copy()
            for key in (*self.image_keys, "observation.state")
            if key in observation
        }
        payload["task"] = str(observation.get("task", ""))
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
            except Exception as error:
                print(f"[client] Failed to save observation: {error}")
            finally:
                self._queue.task_done()

    def _save(self, iteration: int, obs_seq: int, observation: dict[str, Any]) -> None:
        if self.output_dir is None:
            return
        step_dir = self.output_dir / f"step_{iteration:06d}"
        step_dir.mkdir()
        for key in self.image_keys:
            if key not in observation:
                continue
            image = np.asarray(observation[key])
            if image.dtype != np.uint8:
                if (
                    np.issubdtype(image.dtype, np.floating)
                    and float(image.max(initial=0.0)) <= 1.0
                ):
                    image = image * 255.0
                image = np.clip(image, 0, 255).astype(np.uint8)
            safe_name = key.replace("/", "_")
            cv2.imwrite(
                str(step_dir / f"{safe_name}.jpg"), cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            )
        np.save(
            step_dir / "observation.state.npy", np.asarray(observation["observation.state"])
        )
        with (step_dir / "metadata.json").open("w", encoding="utf-8") as file:
            json.dump(
                {"iteration": iteration, "obs_seq": obs_seq, "task": observation["task"]},
                file,
            )

    def close(self) -> None:
        if not self.enabled:
            return
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=10.0)
        print(f"[client] Observation saver stopped; dropped={self._dropped}")


def _section(config: dict[str, Any], name: str) -> dict[str, Any]:
    value = config.get(name)
    if not isinstance(value, dict):
        raise ValueError(f"Missing YAML section: {name}")
    return value


def _required(mapping: Mapping[str, Any], key: str, where: str) -> Any:
    if key not in mapping:
        raise ValueError(f"Missing config value {where}.{key}")
    return mapping[key]


def load_config(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"config not found: {path}")
    with path.open(encoding="utf-8") as file:
        config = yaml.safe_load(file) or {}
    if not isinstance(config, dict):
        raise ValueError(f"config root must be a mapping: {path}")

    connection = _section(config, "connection")
    observation = _section(config, "observation")
    control = _section(config, "control")
    runtime = _section(config, "runtime")
    logging_config = config.get("logging", {}) or {}

    _required(config, "checkpoint", "root")
    for key in ("address", "port", "action_ack_timeout_s"):
        _required(connection, key, "connection")
    for key in (
        "data_type",
        "language_prompt",
        "single_arm_mode",
        "no_state_obs_mode",
    ):
        _required(observation, key, "observation")
    for key in (
        "control_frequency",
        "controller_frequency",
        "steps_per_inference",
        "action_horizon",
    ):
        _required(control, key, "control")

    if observation["data_type"] not in SUPPORTED_DATA_TYPES:
        raise ValueError("observation.data_type must be 'vision' or 'vitac'")
    if observation["single_arm_mode"] or observation["no_state_obs_mode"]:
        raise ValueError("The current checkpoint contract requires bimanual state mode")
    if int(control["action_horizon"]) <= 0:
        raise ValueError("action_horizon must be positive")
    if isinstance(control["steps_per_inference"], bool) or not isinstance(
        control["steps_per_inference"], int
    ):
        raise ValueError("steps_per_inference must be an integer")
    if not 1 <= int(control["steps_per_inference"]) <= int(control["action_horizon"]):
        raise ValueError("steps_per_inference must be between 1 and action_horizon")
    if float(control["control_frequency"]) <= 0 or float(control["controller_frequency"]) <= 0:
        raise ValueError("Control frequencies must be positive")
    if float(connection["action_ack_timeout_s"]) <= 0:
        raise ValueError("action_ack_timeout_s must be positive")
    for key in ("observation_timeout_s", "ping_interval_s", "ping_timeout_s"):
        if key in connection and float(connection[key]) <= 0:
            raise ValueError(f"{key} must be positive")
    if int(runtime.get("warmup_runs", 1)) < 1:
        raise ValueError("warmup_runs must be at least 1")
    if not isinstance(logging_config, dict):
        raise ValueError("logging must be a mapping")
    rename_map = config.get("rename_map", {}) or {}
    if not isinstance(rename_map, dict):
        raise ValueError("rename_map must be a mapping of string to string")
    validate_frs_config_section(config)
    return config


def _resolve_checkpoint(value: str, config_path: Path) -> str:
    checkpoint = Path(value).expanduser()
    if checkpoint.is_absolute():
        return str(checkpoint)
    relative = (config_path.parent / checkpoint).resolve()
    return str(relative) if relative.exists() else value


def _checkpoint_contract(
    config: Mapping[str, Any],
    control: Mapping[str, Any],
    *,
    inferred: CheckpointContract | None = None,
) -> CheckpointContract:
    """Build the deployment contract from checkpoint metadata with optional YAML overrides.

    When ``inferred`` is provided (normal deploy path), missing/null YAML fields keep the
    checkpoint values. When ``inferred`` is omitted, the YAML must still supply a full
    contract (kept for unit tests and fixtures).
    """

    raw = config.get("checkpoint_contract")
    if raw is None:
        raw = {}
    if not isinstance(raw, Mapping):
        raise ValueError("checkpoint_contract must be a mapping when provided")

    def has_override(key: str) -> bool:
        return key in raw and raw[key] is not None

    def integer(key: str, *, allow_zero: bool = False) -> int:
        if has_override(key):
            value = int(raw[key])
        elif inferred is not None:
            value = int(getattr(inferred, key))
        else:
            value = int(_required(raw, key, "checkpoint_contract"))
        if value < 0 or (value == 0 and not allow_zero):
            qualifier = "non-negative" if allow_zero else "positive"
            raise ValueError(f"checkpoint_contract.{key} must be {qualifier}")
        return value

    def string_tuple(key: str, *, allow_empty: bool = False) -> tuple[str, ...]:
        if has_override(key):
            value = raw[key]
        elif inferred is not None:
            return tuple(getattr(inferred, key))
        else:
            value = _required(raw, key, "checkpoint_contract")
        if (
            not isinstance(value, list | tuple)
            or (not value and not allow_empty)
            or any(not isinstance(item, str) or not item for item in value)
        ):
            qualifier = "a list" if allow_empty else "a non-empty list"
            raise ValueError(f"checkpoint_contract.{key} must be {qualifier} of strings")
        return tuple(value)

    tactile_num_tokens = integer("tactile_num_tokens", allow_zero=True)
    if has_override("tactile_proj_mode"):
        tactile_proj_mode = raw["tactile_proj_mode"]
    elif inferred is not None and inferred.tactile_proj_mode is not None:
        tactile_proj_mode = inferred.tactile_proj_mode
    else:
        tactile_proj_mode = "full" if tactile_num_tokens else "frozen"
    if not isinstance(tactile_proj_mode, str) or tactile_proj_mode not in {
        "frozen",
        "full",
        "lora",
    }:
        raise ValueError(
            "checkpoint_contract.tactile_proj_mode must be one of "
            "'frozen', 'full', or 'lora'"
        )
    contract = CheckpointContract(
        state_dim=integer("state_dim"),
        action_dim=integer("action_dim"),
        chunk_size=integer("chunk_size"),
        image_keys=string_tuple("image_keys"),
        tactile_keys=string_tuple("tactile_keys", allow_empty=True),
        tactile_embedding_dim=integer("tactile_embedding_dim"),
        tactile_num_tokens=tactile_num_tokens,
        tactile_proj_mode=tactile_proj_mode,
        lora_rank=integer("lora_rank", allow_zero=True),
        vlm_lora_target_modules=string_tuple("vlm_lora_target_modules", allow_empty=True),
    )
    if contract.chunk_size != int(control["action_horizon"]):
        raise ValueError(
            f"checkpoint chunk_size/contract.chunk_size={contract.chunk_size} does not match "
            f"control.action_horizon={control['action_horizon']}"
        )
    if contract.tactile_num_tokens != len(contract.tactile_keys):
        raise ValueError("checkpoint_contract.tactile_num_tokens must equal the number of tactile_keys")
    overlap = sorted(set(contract.image_keys) & set(contract.tactile_keys))
    if overlap:
        raise ValueError(f"checkpoint_contract RGB and tactile keys overlap: {overlap}")
    return contract


def _load_validated_policy(
    checkpoint: str,
    *,
    revision: str | None,
    allow_download: bool,
    rename_map: Mapping[str, str] | None,
    config: Mapping[str, Any] | None = None,
    control: Mapping[str, Any] | None = None,
    base_sidecars: str | Path | None = None,
    expected: CheckpointContract | None = None,
) -> Policy:
    """Resolve and validate a snapshot before materializing any model tensors."""

    snapshot = resolve_checkpoint(
        checkpoint,
        revision=revision,
        local_files_only=not allow_download,
    )
    if expected is None:
        if config is None or control is None:
            raise ValueError("config and control are required when expected is omitted")
        inferred = None
        if (snapshot / "config.json").is_file():
            inferred = contract_from_checkpoint(snapshot)
        expected = _checkpoint_contract(config, control, inferred=inferred)
        print(
            "[client] Checkpoint contract: "
            f"state_dim={expected.state_dim} "
            f"action_dim={expected.action_dim} "
            f"chunk_size={expected.chunk_size} "
            f"images={list(expected.image_keys)} "
            f"lora_rank={expected.lora_rank} "
            f"vlm_lora={list(expected.vlm_lora_target_modules)}"
        )
    validate_checkpoint(
        snapshot,
        expected=expected,
        base_sidecars=base_sidecars,
        require_weight=True,
    ).require_valid()
    policy_type = VTJaxSmolVLAPolicy if expected.tactile_num_tokens else JaxSmolVLAPolicy
    return policy_type.from_pretrained(
        snapshot,
        rename_map=rename_map,
        revision=None,
        local_files_only=True,
    )


def _resolve_token(connection: dict[str, Any]) -> str | None:
    """Resolve auth token: env var overrides config ``token`` when both are set."""
    token_env = str(connection.get("token_env", "")).strip()
    env_token = os.environ.get(token_env) if token_env else None
    config_token = connection.get("token")
    if config_token is not None:
        config_token = str(config_token).strip() or None
    token = env_token or config_token
    if bool(connection.get("require_token", False)) and not token:
        sources = []
        if token_env:
            sources.append(f"env {token_env}")
        sources.append("connection.token")
        raise ValueError(
            "Required authentication token is missing; set " + " or ".join(sources)
        )
    return token or None


def _optional_bool(value: Any) -> bool | None:
    return None if value is None else bool(value)


def _parse_rename_map(config: Mapping[str, Any]) -> dict[str, str] | None:
    rename_map = config.get("rename_map") or {}
    if not isinstance(rename_map, dict):
        raise ValueError("rename_map must be a mapping of string to string")
    if not rename_map:
        return None
    return {str(key): str(value) for key, value in rename_map.items()}


def _validate_observation_mode(data_type: str, *, use_tactile_encoder: bool) -> None:
    expected = "vitac" if use_tactile_encoder else "vision"
    if data_type != expected:
        raise ValueError(
            f"Checkpoint requires observation.data_type={expected!r}, got {data_type!r}"
        )


def _robot_image_keys(policy: Policy, rename_map: Mapping[str, str] | None) -> tuple[str, ...]:
    """Map checkpoint image keys back to keys expected on the robot observation dict."""
    reverse = {value: key for key, value in (rename_map or {}).items()}
    model_keys = list(policy.config.image_keys)
    use_tactile = bool(getattr(policy.config, "use_tactile_encoder", False))
    tactile_keys = tuple(getattr(policy.config, "tactile_keys", ())) if use_tactile else ()
    model_keys.extend(tactile_keys)
    return tuple(reverse.get(key, key) for key in model_keys)


def _robot_tactile_keys(
    policy: Policy,
    rename_map: Mapping[str, str] | None,
) -> tuple[str, ...]:
    use_tactile = bool(getattr(policy.config, "use_tactile_encoder", False))
    tactile_keys = tuple(getattr(policy.config, "tactile_keys", ())) if use_tactile else ()
    reverse = {value: key for key, value in (rename_map or {}).items()}
    return tuple(reverse.get(key, key) for key in tactile_keys)


def _validate_observation(
    observation: dict[str, Any],
    *,
    state_dim: int,
    image_keys: Sequence[str],
    empty_cameras: int,
    required_image_keys: Sequence[str] = (),
) -> None:
    if "observation.state" not in observation:
        raise ValueError("Robot observation is missing keys: ['observation.state']")
    missing_required = [key for key in required_image_keys if key not in observation]
    if missing_required:
        raise ValueError(f"Robot observation is missing required tactile keys: {missing_required}")
    required = set(required_image_keys)
    optional_image_keys = [key for key in image_keys if key not in required]
    present = [key for key in image_keys if key in observation]
    missing = [key for key in optional_image_keys if key not in observation]
    if not present:
        raise ValueError(f"Robot observation is missing all image keys: {list(image_keys)}")
    if len(missing) > max(empty_cameras, 0):
        raise ValueError(
            f"Robot observation is missing too many image keys: {missing} "
            f"(empty_cameras={empty_cameras})"
        )
    state = np.asarray(observation["observation.state"])
    if state.shape != (state_dim,):
        raise ValueError(f"Expected {state_dim}D state, got {state.shape}")
    if not np.isfinite(state).all():
        raise ValueError("Robot observation state contains NaN or Inf")
    for key in present:
        image = np.asarray(observation[key])
        if image.ndim != 3 or image.shape[-1] != 3:
            raise ValueError(f"{key} must be HWC RGB, got {image.shape}")


def _prepare_observation(
    observation: dict[str, Any],
    *,
    state_dim: int,
    image_keys: Sequence[str],
    empty_cameras: int,
    required_image_keys: Sequence[str] = (),
) -> dict[str, Any]:
    _validate_observation(
        observation,
        state_dim=state_dim,
        image_keys=image_keys,
        empty_cameras=empty_cameras,
        required_image_keys=required_image_keys,
    )
    prepared = {
        key: np.asarray(observation[key]).copy()
        for key in image_keys
        if key in observation
    }
    prepared["observation.state"] = np.asarray(observation["observation.state"]).copy()
    return prepared

# problem!

def _predict_chunk(
    policy: Policy,
    observation: Mapping[str, Any],
    task: str,
    *,
    seed: int,
    jit: bool,
    num_steps: int | None,
    previous_chunk: np.ndarray | None,
    inference_delay: int | None,
    execution_horizon: int | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(robot_action, model_space_action)`` each shaped ``[horizon, action_dim]``."""

    actions_norm = policy.predict_action_chunk(
        observation,
        task,
        seed=seed,
        jit=jit,
        normalized=True,
        num_steps=num_steps,
        previous_chunk=None if previous_chunk is None else np.asarray(previous_chunk),
        inference_delay=inference_delay,
        execution_horizon=execution_horizon,
    )

    jax.block_until_ready(actions_norm)

    actions = policy.preprocessor.unnormalize_actions(actions_norm)
    expected_shape = (1, policy.config.chunk_size, policy.config.action_dim)
    action = np.asarray(actions)
    action_norm = np.asarray(actions_norm)

    if action.shape != expected_shape:
        raise ValueError(f"Expected JAX SmolVLA action shaped {expected_shape}, got {action.shape}")
    action = action[0].astype(np.float32, copy=False)
    action_norm = action_norm[0].astype(np.float32, copy=False)
    if not np.isfinite(action).all():
        raise ValueError("SmolVLA action contains NaN or Inf")
    return action, action_norm


def _trace_action_chunk(value: Any) -> np.ndarray:
    chunk = np.asarray(value, dtype=np.float32)
    if chunk.ndim == 3 and chunk.shape[0] == 1:
        chunk = chunk[0]
    if chunk.ndim != 2:
        raise ValueError(f"trace action chunk must be rank 2, got {chunk.shape}")
    return np.array(chunk, copy=True)


def _immutable_trace_array(value: Any) -> np.ndarray:
    source = np.asarray(value, dtype=np.float32)
    return np.frombuffer(source.tobytes(order="C"), dtype=np.float32).reshape(source.shape)


def _build_frs_chunk_trace(ready: FRSChunkReady) -> dict[str, Any]:
    return {
        "version": 2,
        "kind": "frs_chunk",
        "chunk_id": int(ready.chunk_id),
        "action_vla_normalized": _immutable_trace_array(ready.action_vla_normalized),
        "action_vla": _immutable_trace_array(ready.action_vla),
        "x_base": _immutable_trace_array(ready.x_base),
        "prediction_started_at": float(ready.prediction_started_at),
        "prediction_finished_at": float(ready.prediction_finished_at),
    }


def _build_frs_steer_trace(
    result: FRSSteerResult,
    request: FRSSteerRequest,
) -> dict[str, Any]:
    if (result.chunk_id, result.request_id, result.action_index) != (
        request.chunk_id,
        request.request_id,
        request.action_index,
    ):
        raise ValueError("FRS steer trace result does not match its request")
    diagnostics = result.diagnostics
    return {
        "version": 2,
        "kind": "frs_steer",
        "chunk_id": int(result.chunk_id),
        "request_id": int(result.request_id),
        "action_index": int(result.action_index),
        "target_timestamp": request.target_timestamp,
        "protection_applied": request.protection_applied,
        "decoded_normalized": _immutable_trace_array(result.decoded_normalized),
        "selected_normalized": _immutable_trace_array(result.selected_normalized),
        "selected_action": _immutable_trace_array(result.selected_action),
        "tactile_sequence_length": int(result.tactile_sequence_length),
        "encode_started_at": float(result.encode_started_at),
        "encode_finished_at": float(result.encode_finished_at),
        "decode_started_at": float(result.decode_started_at),
        "decode_finished_at": float(result.decode_finished_at),
        "frs_diagnostics": {
            "tactile_change": float(diagnostics.tactile_change),
            "delta_rms": float(diagnostics.delta_rms),
            "max_normalized_action_abs": float(diagnostics.max_normalized_action_abs),
        },
    }


def _build_trace_or_none(builder: Any, *args: Any) -> dict[str, Any] | None:
    try:
        return builder(*args)
    except Exception as exc:
        LOGGER.warning("Omitting FRS trace after serialization failure: %s", exc)
        return None


def _run_frs_protocol(
    bridge: RobotBridgeClient,
    steering_policy: FRSRuntime,
    *,
    task: str,
    state_dim: int,
    image_keys: Sequence[str],
    empty_cameras: int,
    observation_timeout_s: float,
    action_ack_timeout_s: float,
    seed: int,
    jit: bool,
    num_steps: int | None,
    max_chunks: int,
    observation_saver: ObservationSaver,
) -> None:
    """Run the strictly ordered, server-directed FRS steering protocol."""

    print("[client] Running FRS steering protocol.")

    completed_chunks = 0
    previous_chunk_id: int | None = None
    while max_chunks <= 0 or completed_chunks < max_chunks:

        chunk_start = bridge.receive_frs_message(observation_timeout_s)
        if not isinstance(chunk_start, FRSChunkStart):
            raise RuntimeError(
                "expected FRS chunk start, received "
                f"{type(chunk_start).__name__}"
            )
        if previous_chunk_id is not None and chunk_start.chunk_id <= previous_chunk_id:
            raise RuntimeError(
                "FRS chunk ids must be strictly increasing: "
                f"{chunk_start.chunk_id} <= {previous_chunk_id}"
            )
        observation_saver.submit(
            completed_chunks + 1,
            chunk_start.obs_seq,
            chunk_start.observation,
        )
        initial_observation = _prepare_observation(
            dict(chunk_start.observation),
            state_dim=state_dim,
            image_keys=image_keys,
            empty_cameras=empty_cameras,
            required_image_keys=steering_policy.tactile_keys,
        )

        # 在一个 Chunk 的 Steering 之前做准备
        ready = steering_policy.begin_chunk(
            chunk_start.chunk_id,
            initial_observation,
            task,
            seed=seed,
            jit=jit,
            num_steps=num_steps,
        )
        if ready.chunk_id != chunk_start.chunk_id:
            raise RuntimeError(
                "FRS chunk ready id does not match chunk start: "
                f"{ready.chunk_id} != {chunk_start.chunk_id}"
            )
        print("[client] FRS chunk is ready.")

        bridge.send_frs_chunk_ready(
            chunk_start.obs_seq,
            chunk_start.chunk_id,
            _build_trace_or_none(_build_frs_chunk_trace, ready),
        )

        while True:
            # 接收服务端发送的 FRS 消息，包括：
            # 当前 Action Chunk 编号；Chunk 是否结束；观测量 等信息
            message = bridge.receive_frs_message(observation_timeout_s)
            print('[client] message type: ', type(message).__name__)
            if isinstance(message, FRSChunkEnd):
                if message.chunk_id != chunk_start.chunk_id:
                    raise RuntimeError(
                        "FRS chunk end chunk id does not match active chunk: "
                        f"{message.chunk_id} != {chunk_start.chunk_id}"
                    )
                print("[client] Chunk end.")
                steering_policy.end_chunk(chunk_start.chunk_id)
                previous_chunk_id = chunk_start.chunk_id
                completed_chunks += 1
                break
            print("[client] Received message: ", message.chunk_id, message.request_id, message.action_index)

            if not isinstance(message, FRSSteerRequest):
                raise RuntimeError(
                    "expected FRS steer request or chunk end, received "
                    f"{type(message).__name__}"
                )
            if message.chunk_id != chunk_start.chunk_id:
                raise RuntimeError(
                    "FRS steer request chunk id does not match active chunk: "
                    f"{message.chunk_id} != {chunk_start.chunk_id}"
                )

            # 准备观测量
            request_observation = _prepare_observation(
                dict(message.observation),
                state_dim=state_dim,
                image_keys=image_keys,
                empty_cameras=empty_cameras,
                required_image_keys=steering_policy.tactile_keys,
            )
            # 开始推理
            print("[client] Steering action.")
            time_start = time.time()
            result = steering_policy.steer_action(
                message.chunk_id,
                message.request_id,
                request_observation,
                message.action_index,
            )
            request_ids = (
                message.chunk_id,
                message.request_id,
                message.action_index,
            )
            result_ids = (result.chunk_id, result.request_id, result.action_index)
            if result_ids != request_ids:
                raise RuntimeError(
                    "FRS steer result does not match its request: "
                    f"{result_ids} != {request_ids}"
                )
            selected_action = np.asarray(result.selected_action)

            expected_shape = (int(steering_policy.policy.config.action_dim),)
            if selected_action.shape != expected_shape:
                raise RuntimeError(
                    "FRS selected action must have shape "
                    f"{expected_shape}, got {selected_action.shape}"
                )
            
            # 发送推理结果
            bridge.send_frs_steer_action(
                message.chunk_id,
                message.request_id,
                message.action_index,
                selected_action,
                trace=_build_trace_or_none(_build_frs_steer_trace, result, message),
            )
            time_end = time.time()
            print("[client] Steering action finished in ", time_end - time_start, " seconds")
            # 结束推理

            acknowledgement = bridge.receive_frs_message(action_ack_timeout_s) # 机器人端回执
            print("[client] Received acknowledgement: ", acknowledgement) 
            print()

            if not isinstance(acknowledgement, FRSSteerAck):
                raise RuntimeError(
                    "expected FRS steer acknowledgement, received "
                    f"{type(acknowledgement).__name__}"
                )
            acknowledgement_ids = (
                acknowledgement.chunk_id,
                acknowledgement.request_id,
                acknowledgement.action_index,
            )
            if acknowledgement_ids != request_ids:
                raise RuntimeError(
                    "FRS steer acknowledgement does not match its request: "
                    f"{acknowledgement_ids} != {request_ids}"
                )
            if acknowledgement.status == "rejected":
                raise RuntimeError(
                    "FRS steer action was rejected for "
                    f"chunk={message.chunk_id} request={message.request_id} "
                    f"action_index={message.action_index}"
                )


def _build_action_trace(
    policy: Policy,
    frs_runtime: FRSRuntime | None,
    *,
    inference_wall_start_s: float,
    inference_wall_end_s: float,
) -> dict[str, Any] | None:
    """Build a diagnostic-only v1 trace for a completed FRS inference."""

    if frs_runtime is None:
        return None
    vla_normalized = frs_runtime.last_vla_normalized
    frs_normalized = frs_runtime.last_frs_normalized
    diagnostics = frs_runtime.last_diagnostics
    if vla_normalized is None or frs_normalized is None or diagnostics is None:
        return None

    vla_chunk = _trace_action_chunk(vla_normalized)
    frs_chunk = _trace_action_chunk(frs_normalized)
    vla_action = _trace_action_chunk(policy.preprocessor.unnormalize_actions(vla_normalized))
    frs_action = _trace_action_chunk(policy.preprocessor.unnormalize_actions(frs_normalized))
    return {
        "version": 1,
        "vla_normalized": vla_chunk,
        "frs_normalized": frs_chunk,
        "vla_action": vla_action,
        "frs_action": frs_action,
        "inference_started_at": float(inference_wall_start_s),
        "inference_finished_at": float(inference_wall_end_s),
        "frs_diagnostics": {
            "tactile_change": float(diagnostics.tactile_change),
            "delta_rms": float(diagnostics.delta_rms),
            "max_normalized_action_abs": float(diagnostics.max_normalized_action_abs),
        },
    }


def _build_action_trace_or_none(
    policy: Policy,
    frs_runtime: FRSRuntime | None,
    *,
    inference_wall_start_s: float,
    inference_wall_end_s: float,
) -> dict[str, Any] | None:
    """Keep trace-only failures from affecting the robot action path."""

    try:
        return _build_action_trace(
            policy,
            frs_runtime,
            inference_wall_start_s=inference_wall_start_s,
            inference_wall_end_s=inference_wall_end_s,
        )
    except Exception as error:
        print(f"[client] Action trace omitted: {error}")
        return None


def _rtc_enabled(policy: Policy) -> bool:
    rtc = policy.config.rtc_config
    return rtc is not None and bool(rtc.enabled)


def _remaining_action_chunk(action_chunk: np.ndarray, executed_steps: int) -> np.ndarray:
    """Align the unexecuted tail of a chunk with the next observation."""

    chunk = np.asarray(action_chunk)
    if chunk.ndim != 2:
        raise ValueError(f"action chunk must be rank 2, got {chunk.shape}")
    if not 0 <= int(executed_steps) <= chunk.shape[0]:
        raise ValueError(
            f"executed_steps must be in [0, {chunk.shape[0]}], got {executed_steps}"
        )
    return np.array(chunk[int(executed_steps) :], copy=True)


def _build_server_config(
    observation_config: Mapping[str, Any],
    control: Mapping[str, Any],
    *,
    frs_policy: FRSRuntime | None,
) -> dict[str, Any]:
    server_config = {
        "data_type": observation_config["data_type"],
        "language_prompt": observation_config["language_prompt"],
        "control_frequency": float(control["control_frequency"]),
        "controller_frequency": float(control["controller_frequency"]),
        "single_arm_mode": bool(observation_config["single_arm_mode"]),
        "no_state_obs_mode": bool(observation_config["no_state_obs_mode"]),
        "steps_per_inference": int(control["steps_per_inference"]),
        "action_horizon": int(control["action_horizon"]),
    }
    if frs_policy is not None:
        server_config.update(
            execution_protocol="frs_steering_v1",
            steering_protection_interval_s=(
                frs_policy.config.steering_protection_interval_s
            ),
            frs_tactile_keys=list(frs_policy.tactile_keys),
        )
    return server_config


def run(
    config_path: Path,
    max_iterations_override: int | None = None,
) -> None:
    config_path = config_path.expanduser().resolve()
    config = load_config(config_path)
    connection = _section(config, "connection")
    observation_config = _section(config, "observation")
    control = _section(config, "control")
    runtime = _section(config, "runtime")
    logging_config = config.get("logging", {}) or {}

    checkpoint = _resolve_checkpoint(str(config["checkpoint"]), config_path)
    rename_map = _parse_rename_map(config)
    allow_download = bool(config.get("allow_download", False))
    revision = config.get("revision")
    revision = None if revision is None else str(revision)
    seed = int(config.get("seed", 0))
    jit = bool(config.get("jit", True))
    num_steps = config.get("num_steps")
    if num_steps is not None:
        num_steps = int(num_steps)

    print(f"[client] Loading JAX SmolVLA checkpoint: {checkpoint}")
    policy = _load_validated_policy(
        checkpoint,
        revision=revision,
        allow_download=allow_download,
        config=config,
        control=control,
        rename_map=rename_map,
    )
    print(f"[client] JAX backend: {jax.default_backend()}")
    policy.reset()
    use_tactile = bool(getattr(policy.config, "use_tactile_encoder", False))
    frs_config = config.get("frs")
    frs_enabled = isinstance(frs_config, Mapping) and bool(frs_config.get("enabled", True))
    if frs_enabled:
        source_sample_steps = int(policy.config.num_steps if num_steps is None else num_steps)
        frs_runtime = FRSRuntime(
            frs_config,
            config_path=config_path,
            policy=policy,
            source_sample_steps=source_sample_steps,
        )
    else:
        frs_runtime = None
    _validate_observation_mode(
        str(observation_config["data_type"]),
        use_tactile_encoder=use_tactile or frs_runtime is not None,
    )

    configured_horizon = int(control["action_horizon"])
    if policy.config.chunk_size != configured_horizon:
        raise ValueError(
            f"Checkpoint chunk_size={policy.config.chunk_size} does not match "
            f"action_horizon={configured_horizon}"
        )
    if policy.config.action_dim <= 0:
        raise ValueError(f"Checkpoint action_dim must be positive, got {policy.config.action_dim}")
    configured_steps = int(control["steps_per_inference"])
    if not policy.config.image_keys:
        raise ValueError("Checkpoint does not declare any visual observation keys")

    robot_image_keys = _robot_image_keys(policy, rename_map)
    robot_tactile_keys = _robot_tactile_keys(policy, rename_map)
    if frs_runtime is not None:
        robot_tactile_keys = frs_runtime.tactile_keys
        robot_image_keys = tuple(dict.fromkeys((*robot_image_keys, *robot_tactile_keys)))
    state_dim = int(policy.config.state_dim)
    empty_cameras = int(policy.config.empty_cameras)
    print(
        f"[client] Contract: state_dim={state_dim} action_dim={policy.config.action_dim} "
        f"images={list(robot_image_keys)} tactile={list(robot_tactile_keys)} "
        f"empty_cameras={empty_cameras}"
    )

    steps_per_inference = configured_steps
    rtc_on = _rtc_enabled(policy)
    if frs_runtime is not None and rtc_on:
        raise ValueError("FRS deployment does not support RTC action stitching")
    configured_inference_delay = control.get("inference_delay")
    if rtc_on:
        inference_delay = (
            steps_per_inference
            if configured_inference_delay is None
            else int(configured_inference_delay)
        )
    else:
        inference_delay = None
    execution_horizon = control.get("execution_horizon")
    if execution_horizon is not None:
        execution_horizon = int(execution_horizon)
    elif rtc_on and policy.config.rtc_config is not None:
        execution_horizon = int(policy.config.rtc_config.execution_horizon)
    if rtc_on:
        print(
            f"[client] RTC enabled: inference_delay={inference_delay} "
            f"execution_horizon={execution_horizon}"
        )

    server_config = _build_server_config(
        observation_config,
        control,
        frs_policy=frs_runtime,
    )
    bridge = RobotBridgeClient(
        address=str(connection["address"]),
        port=int(connection["port"]),
        token=_resolve_token(connection),
        add_port=_optional_bool(connection.get("add_port")),
        retry_interval_s=float(connection.get("retry_interval_s", 1.0)),
        ping_interval_s=float(connection.get("ping_interval_s", 20.0)),
        ping_timeout_s=float(connection.get("ping_timeout_s", 20.0)),
    )
    bridge.send_config(server_config)
    observation_saver = ObservationSaver(logging_config, robot_image_keys)
    observation_saver.start()

    status_interval_s = float(runtime.get("status_interval_s", 2.0))
    warmup_runs = int(runtime.get("warmup_runs", 1))
    max_iterations = (
        int(runtime.get("max_iterations", 0))
        if max_iterations_override is None
        else int(max_iterations_override)
    )
    if max_iterations < 0:
        raise ValueError("max_iterations must be non-negative")
    action_ack_timeout_s = float(connection["action_ack_timeout_s"])
    observation_timeout_s = float(connection.get("observation_timeout_s", 30.0))
    task = str(observation_config["language_prompt"])
    previous_chunk: np.ndarray | None = None

    try:
        print("[client] Waiting for robot warmup observation")
        warmup_obs_seq, warmup_observation = bridge.receive_observation(
            timeout=observation_timeout_s
        )
        warmup_frame = _prepare_observation(
            warmup_observation,
            state_dim=state_dim,
            image_keys=robot_image_keys,
            empty_cameras=empty_cameras,
            required_image_keys=robot_tactile_keys,
        )

        if frs_runtime is not None:
            frs_runtime.reset_episode(warmup_frame)
            frs_runtime.warmup_all_tactile_lengths()
            print(
                "[client] FRS enabled: "
                f"checkpoint={frs_runtime.config.checkpoint} "
                f"window={frs_runtime.resolved_tactile_window()} "
                f"divisor={frs_runtime.config.tactile_window_divisor}"
            )
            
        for warmup_index in range(warmup_runs):
            start = time.perf_counter()
            _predict_chunk(
                policy,
                warmup_frame,
                task,
                seed=seed,
                jit=jit,
                num_steps=num_steps,
                previous_chunk=None,
                inference_delay=inference_delay if rtc_on else None,
                execution_horizon=execution_horizon if rtc_on else None,
            )
            warmup_ms = (time.perf_counter() - start) * 1000.0
            print(f"[client] Warmup {warmup_index + 1}/{warmup_runs}: {warmup_ms:.1f}ms")
        print(f"[client] Warmup observation sequence: {warmup_obs_seq}")

        if not bool(runtime.get("auto_start", False)):
            input("[client] Ready. Press Enter to send START to the robot server... ")
        bridge.send_state("start")

        if frs_runtime is not None:
            _run_frs_protocol(
                bridge,
                frs_runtime,
                task=task,
                state_dim=state_dim,
                image_keys=robot_image_keys,
                empty_cameras=empty_cameras,
                observation_timeout_s=observation_timeout_s,
                action_ack_timeout_s=action_ack_timeout_s,
                seed=seed,
                jit=jit,
                num_steps=num_steps,
                max_chunks=max_iterations,
                observation_saver=observation_saver,
            )
            return

        iteration = 0
        last_status_time = time.monotonic()
        while max_iterations <= 0 or iteration < max_iterations:
            obs_seq, observation = bridge.receive_observation(timeout=observation_timeout_s)
            observation_saver.submit(iteration + 1, obs_seq, observation)
            frame = _prepare_observation(
                observation,
                state_dim=state_dim,
                image_keys=robot_image_keys,
                empty_cameras=empty_cameras,
                required_image_keys=robot_tactile_keys,
            )
            inference_started_at = time.time()
            start = time.perf_counter()

            # key entrypoint
            action, action_norm = _predict_chunk(
                policy,
                frame,
                task,
                seed=seed + iteration,
                jit=jit,
                num_steps=num_steps,
                previous_chunk=previous_chunk if rtc_on else None,
                inference_delay=inference_delay if rtc_on else None,
                execution_horizon=execution_horizon if rtc_on else None,
            )

            inference_ms = (time.perf_counter() - start) * 1000.0
            trace = _build_action_trace_or_none(
                policy,
                frs_runtime,
                inference_wall_start_s=inference_started_at,
                inference_wall_end_s=time.time(),
            )
            bridge.send_action(action, obs_seq, trace=trace)
            bridge.receive_action_ack(obs_seq, timeout=action_ack_timeout_s)

            if rtc_on:
                previous_chunk = _remaining_action_chunk(action_norm, steps_per_inference)
            iteration += 1

            now = time.monotonic()
            if now - last_status_time >= status_interval_s:
                frs_status = ""
                if frs_runtime is not None and frs_runtime.last_diagnostics is not None:
                    diag = frs_runtime.last_diagnostics
                    frs_status = (
                        f" tactile_change={diag.tactile_change:.4f}"
                        f" frs_delta_rms={diag.delta_rms:.4f}"
                    )
                print(
                    f"[client] iter={iteration} obs_seq={obs_seq} "
                    f"inference_ms={inference_ms:.1f} action_shape={action.shape}"
                    f"{frs_status}"
                )
                last_status_time = now
    except KeyboardInterrupt:
        print("[client] Interrupted")
    finally:
        observation_saver.close()
        try:
            try:
                bridge.send_state("stop")
            except Exception as exc:
                print(f"[client] Could not send STOP because the bridge is closed: {exc}")
        finally:
            bridge.close()
        print("[client] Stopped")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help=f"YAML config path (default: {DEFAULT_CONFIG})",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        help="override runtime.max_iterations for this run",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    run(args.config, max_iterations_override=args.max_iterations)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
