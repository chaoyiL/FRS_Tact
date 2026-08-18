"""Deploy a JAX pi0.5 + FRS policy through the existing VB robot bridge."""

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

from .bridge_client import RobotBridgeClient
from .deployment import load_deployment_config
from .frs_protocol import FRSChunkEnd, FRSChunkStart, FRSSteerAck, FRSSteerRequest
from .frs_runtime import FRSChunkReady, FRSRuntime, FRSSteerResult
from .policy import Pi05DeploymentConfig, Pi05RemotePolicy

DEFAULT_CONFIG = Path(__file__).resolve().parent / "configs" / "deploy_pi05_frs.yaml"
LOGGER = logging.getLogger(__name__)


class ObservationSaver:
    """Save robot observations on a background thread."""

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
            root = Path(str(config.get("output_dir", "outputs/pi05_frs_observations")))
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
            except Exception as error:  # Observation logging must not stop the robot loop.
                LOGGER.warning("Could not save observation: %s", error)
            finally:
                self._queue.task_done()

    def _save(self, iteration: int, obs_seq: int, observation: Mapping[str, Any]) -> None:
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


def _section(config: Mapping[str, Any], name: str) -> dict[str, Any]:
    value = config.get(name)
    if not isinstance(value, dict):
        raise ValueError(f"Missing YAML section: {name}")
    return value


def _required(mapping: Mapping[str, Any], key: str, where: str) -> Any:
    if key not in mapping:
        raise ValueError(f"Missing config value {where}.{key}")
    return mapping[key]


def load_config(path: Path) -> dict[str, Any]:
    """Load the FRS profile from the shared deployment configuration."""
    return load_deployment_config(path, "frs")


def _resolve_local(value: str, config_path: Path) -> str:
    if "://" in value:
        return value
    path = Path(value).expanduser()
    if path.is_absolute():
        return str(path)
    candidate = (config_path.parent / path).resolve()
    return str(candidate) if candidate.exists() else value


def _policy_config(config: Mapping[str, Any], config_path: Path) -> Pi05DeploymentConfig:
    model = _section(config, "model")
    stats = _section(config, "norm_stats")
    camera_map = model["camera_map"]
    if not isinstance(camera_map, dict):
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
        state_dim=int(model["state_dim"]),
        robot_action_dim=int(model["robot_action_dim"]),
        action_dim=int(model["action_dim"]),
        action_horizon=int(model["action_horizon"]),
        paligemma_variant=str(model.get("paligemma_variant", "gemma_2b_lora")),
        action_expert_variant=str(model.get("action_expert_variant", "gemma_300m_lora")),
        use_quantile_norm=bool(stats["use_quantile_norm"]),
    )


def _resolve_token(connection: Mapping[str, Any]) -> str | None:
    env_name = str(connection.get("token_env", "")).strip()
    env_token = os.environ.get(env_name) if env_name else None
    config_token = str(connection.get("token") or "").strip() or None
    token = env_token or config_token
    if bool(connection.get("require_token", False)) and not token:
        raise ValueError(f"authentication token is missing; set env {env_name} or connection.token")
    return token


def _optional_bool(value: Any) -> bool | None:
    return None if value is None else bool(value)


def _prepare_observation(
    observation: Mapping[str, Any], *, state_dim: int, image_keys: Sequence[str]
) -> dict[str, Any]:
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


def _immutable(value: Any) -> np.ndarray:
    source = np.asarray(value, dtype=np.float32)
    return np.frombuffer(source.tobytes(order="C"), dtype=np.float32).reshape(source.shape)


def _chunk_trace(ready: FRSChunkReady) -> dict[str, Any]:
    return {
        "version": 2,
        "kind": "frs_chunk",
        "chunk_id": ready.chunk_id,
        "action_vla_normalized": _immutable(ready.action_vla_normalized),
        "action_vla": _immutable(ready.action_vla),
        "x_base": _immutable(ready.x_base),
        "prediction_started_at": ready.prediction_started_at,
        "prediction_finished_at": ready.prediction_finished_at,
    }


def _steer_trace(result: FRSSteerResult, request: FRSSteerRequest) -> dict[str, Any]:
    diagnostics = result.diagnostics
    return {
        "version": 2,
        "kind": "frs_steer",
        "chunk_id": result.chunk_id,
        "request_id": result.request_id,
        "action_index": result.action_index,
        "target_timestamp": request.target_timestamp,
        "protection_applied": request.protection_applied,
        "decoded_normalized": _immutable(result.decoded_normalized),
        "selected_normalized": _immutable(result.selected_normalized),
        "selected_action": _immutable(result.selected_action),
        "tactile_sequence_length": result.tactile_sequence_length,
        "encode_started_at": result.encode_started_at,
        "encode_finished_at": result.encode_finished_at,
        "decode_started_at": result.decode_started_at,
        "decode_finished_at": result.decode_finished_at,
        "frs_diagnostics": {
            "tactile_change": diagnostics.tactile_change,
            "delta_rms": diagnostics.delta_rms,
            "max_normalized_action_abs": diagnostics.max_normalized_action_abs,
        },
    }


def _safe_trace(builder: Any, *args: Any) -> dict[str, Any] | None:
    try:
        return builder(*args)
    except Exception as error:
        LOGGER.warning("Omitting diagnostic trace: %s", error)
        return None


def _run_frs(
    bridge: RobotBridgeClient,
    frs: FRSRuntime,
    *,
    task: str,
    image_keys: Sequence[str],
    observation_timeout_s: float,
    action_ack_timeout_s: float,
    seed: int,
    sample_steps: int,
    max_chunks: int,
    saver: ObservationSaver,
) -> None:
    completed = 0
    previous_chunk_id: int | None = None
    while max_chunks <= 0 or completed < max_chunks:
        start = bridge.receive_frs_message(observation_timeout_s)
        if not isinstance(start, FRSChunkStart):
            raise RuntimeError(f"expected FRSChunkStart, got {type(start).__name__}")
        if start.execution_mode != "block":
            raise RuntimeError("pi0.5 FRS deployment requires block execution mode")
        if start.action_horizon != frs.policy.config.action_horizon:
            raise RuntimeError("server and pi0.5 action horizons do not match")
        if previous_chunk_id is not None and start.chunk_id <= previous_chunk_id:
            raise RuntimeError("FRS chunk ids must be strictly increasing")
        saver.submit(completed + 1, start.obs_seq, start.observation)
        observation = _prepare_observation(
            start.observation, state_dim=frs.policy.config.state_dim, image_keys=image_keys
        )
        ready = frs.begin_chunk(
            start.chunk_id, observation, task, seed=seed, num_steps=sample_steps
        )
        bridge.send_frs_chunk_ready(start.obs_seq, start.chunk_id, _safe_trace(_chunk_trace, ready))
        print(f"[client] FRS chunk {start.chunk_id} ready")
        while True:
            message = bridge.receive_frs_message(observation_timeout_s)
            if isinstance(message, FRSChunkEnd):
                if message.chunk_id != start.chunk_id:
                    raise RuntimeError("FRSChunkEnd does not match the active chunk")
                frs.end_chunk(start.chunk_id)
                previous_chunk_id = start.chunk_id
                completed += 1
                break
            if not isinstance(message, FRSSteerRequest):
                raise RuntimeError(f"expected FRSSteerRequest, got {type(message).__name__}")
            if message.chunk_id != start.chunk_id:
                raise RuntimeError("FRSSteerRequest does not match the active chunk")
            frame = _prepare_observation(
                message.observation,
                state_dim=frs.policy.config.state_dim,
                image_keys=image_keys,
            )
            result = frs.steer_action(message.chunk_id, message.request_id, frame, message.action_index)
            action = np.asarray(result.selected_action, dtype=np.float32)
            expected = (frs.policy.config.robot_action_dim,)
            if action.shape != expected:
                raise RuntimeError(f"FRS selected action must have shape {expected}, got {action.shape}")
            bridge.send_frs_steer_action(
                message.chunk_id,
                message.request_id,
                message.action_index,
                action,
                trace=_safe_trace(_steer_trace, result, message),
            )
            ack = bridge.receive_frs_message(action_ack_timeout_s)
            if not isinstance(ack, FRSSteerAck):
                raise RuntimeError(f"expected FRSSteerAck, got {type(ack).__name__}")
            request_ids = (message.chunk_id, message.request_id, message.action_index)
            if (ack.chunk_id, ack.request_id, ack.action_index) != request_ids:
                raise RuntimeError("FRSSteerAck does not match its request")
            if ack.status == "rejected":
                raise RuntimeError(f"robot rejected FRS action {request_ids}")


def run(config_path: Path, max_iterations_override: int | None = None) -> None:
    config_path = config_path.expanduser().resolve()
    config = load_config(config_path)
    policy_config = _policy_config(config, config_path)
    connection = _section(config, "connection")
    observation_config = _section(config, "observation")
    control = _section(config, "control")
    runtime = _section(config, "runtime")
    seed = int(config.get("seed", 0))
    sample_steps = int(config.get("num_steps", 10))
    if sample_steps <= 0:
        raise ValueError("num_steps must be positive")

    print(f"[client] Loading pi0.5 checkpoint: {policy_config.checkpoint}")
    policy = Pi05RemotePolicy(policy_config)
    print(f"[client] JAX backend: {jax.default_backend()}")
    frs = FRSRuntime(
        _section(config, "frs"),
        config_path=config_path,
        policy=policy,
        source_sample_steps=sample_steps,
    )
    image_keys = tuple(dict.fromkeys((*policy.robot_image_keys, *frs.tactile_keys)))
    print(
        "[client] Contract: "
        f"state={policy_config.state_dim}, model_action={policy_config.action_dim}, "
        f"robot_action={policy_config.robot_action_dim}, horizon={policy_config.action_horizon}, "
        f"RGB={list(policy.robot_image_keys)}, empty={list(policy_config.empty_cameras)}, "
        f"tactile={list(frs.tactile_keys)}"
    )
    server_config = {
        "data_type": observation_config["data_type"],
        "language_prompt": observation_config["language_prompt"],
        "control_frequency": float(control["control_frequency"]),
        "controller_frequency": float(control["controller_frequency"]),
        "single_arm_mode": bool(observation_config["single_arm_mode"]),
        "no_state_obs_mode": bool(observation_config["no_state_obs_mode"]),
        "steps_per_inference": int(control["steps_per_inference"]),
        "action_horizon": int(control["action_horizon"]),
        "execution_protocol": "frs_steering_v1",
        "steering_protection_interval_s": frs.config.steering_protection_interval_s,
        "frs_tactile_keys": list(frs.tactile_keys),
    }
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
    saver = ObservationSaver(config.get("logging", {}) or {}, image_keys)
    saver.start()
    timeout = float(connection.get("observation_timeout_s", 30.0))
    ack_timeout = float(connection["action_ack_timeout_s"])
    task = str(observation_config["language_prompt"])
    warmup_runs = int(runtime.get("warmup_runs", 1))
    max_chunks = (
        int(runtime.get("max_iterations", 0))
        if max_iterations_override is None
        else int(max_iterations_override)
    )
    if max_chunks < 0:
        raise ValueError("max_iterations must be non-negative")
    try:
        print("[client] Waiting for robot warmup observation")
        obs_seq, raw = bridge.receive_observation(timeout=timeout)
        warmup = _prepare_observation(raw, state_dim=policy_config.state_dim, image_keys=image_keys)
        frs.reset_episode(warmup)
        for index in range(warmup_runs):
            started = time.perf_counter()
            frs.warmup(warmup, task, seed=seed + index, sample_steps=sample_steps)
            print(
                f"[client] Warmup {index + 1}/{warmup_runs}: {(time.perf_counter() - started) * 1000:.1f}ms"
            )
        print(f"[client] Warmup observation sequence: {obs_seq}")
        if not bool(runtime.get("auto_start", False)):
            input("[client] Ready. Press Enter to send START to the robot server... ")
        bridge.send_state("start")
        _run_frs(
            bridge,
            frs,
            task=task,
            image_keys=image_keys,
            observation_timeout_s=timeout,
            action_ack_timeout_s=ack_timeout,
            seed=seed,
            sample_steps=sample_steps,
            max_chunks=max_chunks,
            saver=saver,
        )
    except KeyboardInterrupt:
        print("[client] Interrupted")
    finally:
        saver.close()
        try:
            bridge.send_state("stop")
        except Exception as error:
            LOGGER.warning("Could not send STOP: %s", error)
        finally:
            bridge.close()
        print("[client] Stopped")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--max-iterations", type=int)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    run(args.config, args.max_iterations)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
