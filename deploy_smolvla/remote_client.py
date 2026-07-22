"""Run a JAX SmolVLA checkpoint as the remote policy client for ``vb3_robot_server``."""

from __future__ import annotations

import argparse
import json
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

from lerobot.policies.smolvla_jax import JaxSmolVLAPolicy

from .bridge_client import RobotBridgeClient

DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "deploy_smolvla_jax.yaml"


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

    if observation["data_type"] != "vision":
        raise ValueError("The current SmolVLA deployment supports data_type='vision' only")
    if observation["single_arm_mode"] or observation["no_state_obs_mode"]:
        raise ValueError("The current checkpoint contract requires bimanual state mode")
    if int(control["action_horizon"]) <= 0:
        raise ValueError("action_horizon must be positive")
    if not 1 <= int(control["steps_per_inference"]) <= int(control["action_horizon"]):
        raise ValueError("steps_per_inference must be between 1 and action_horizon")
    if float(control["control_frequency"]) <= 0 or float(control["controller_frequency"]) <= 0:
        raise ValueError("Control frequencies must be positive")
    if float(connection["action_ack_timeout_s"]) <= 0:
        raise ValueError("action_ack_timeout_s must be positive")
    if int(runtime.get("warmup_runs", 1)) < 1:
        raise ValueError("warmup_runs must be at least 1")
    if not isinstance(logging_config, dict):
        raise ValueError("logging must be a mapping")
    rename_map = config.get("rename_map", {}) or {}
    if not isinstance(rename_map, dict):
        raise ValueError("rename_map must be a mapping of string to string")
    return config


def _resolve_checkpoint(value: str, config_path: Path) -> str:
    checkpoint = Path(value).expanduser()
    if checkpoint.is_absolute():
        return str(checkpoint)
    relative = (config_path.parent / checkpoint).resolve()
    return str(relative) if relative.exists() else value


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


def _robot_image_keys(policy: JaxSmolVLAPolicy, rename_map: Mapping[str, str] | None) -> tuple[str, ...]:
    """Map checkpoint image keys back to keys expected on the robot observation dict."""
    reverse = {value: key for key, value in (rename_map or {}).items()}
    return tuple(reverse.get(key, key) for key in policy.config.image_keys)


def _validate_observation(
    observation: dict[str, Any],
    *,
    state_dim: int,
    image_keys: Sequence[str],
    empty_cameras: int,
) -> None:
    if "observation.state" not in observation:
        raise ValueError("Robot observation is missing keys: ['observation.state']")
    present = [key for key in image_keys if key in observation]
    missing = [key for key in image_keys if key not in observation]
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
) -> dict[str, Any]:
    _validate_observation(
        observation,
        state_dim=state_dim,
        image_keys=image_keys,
        empty_cameras=empty_cameras,
    )
    prepared = {
        key: np.asarray(observation[key]).copy()
        for key in image_keys
        if key in observation
    }
    prepared["observation.state"] = np.asarray(observation["observation.state"]).copy()
    return prepared


def _predict_chunk(
    policy: JaxSmolVLAPolicy,
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


def _rtc_enabled(policy: JaxSmolVLAPolicy) -> bool:
    rtc = policy.config.rtc_config
    return rtc is not None and bool(rtc.enabled)


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
    seed = int(config.get("seed", 0))
    jit = bool(config.get("jit", True))
    num_steps = config.get("num_steps")
    if num_steps is not None:
        num_steps = int(num_steps)

    print(f"[client] Loading JAX SmolVLA checkpoint: {checkpoint}")
    print(f"[client] JAX backend: {jax.default_backend()}")
    policy = JaxSmolVLAPolicy.from_pretrained(
        checkpoint,
        rename_map=rename_map,
        revision=None if revision is None else str(revision),
        local_files_only=not allow_download,
    )
    policy.reset()

    configured_horizon = int(control["action_horizon"])
    if policy.config.chunk_size != configured_horizon:
        raise ValueError(
            f"Checkpoint chunk_size={policy.config.chunk_size} does not match "
            f"action_horizon={configured_horizon}"
        )
    if policy.config.action_dim <= 0:
        raise ValueError(f"Checkpoint action_dim must be positive, got {policy.config.action_dim}")
    if not policy.config.image_keys:
        raise ValueError("Checkpoint does not declare any visual observation keys")

    robot_image_keys = _robot_image_keys(policy, rename_map)
    state_dim = int(policy.config.state_dim)
    empty_cameras = int(policy.config.empty_cameras)
    print(
        f"[client] Contract: state_dim={state_dim} action_dim={policy.config.action_dim} "
        f"images={list(robot_image_keys)} empty_cameras={empty_cameras}"
    )

    steps_per_inference = int(control["steps_per_inference"])
    rtc_on = _rtc_enabled(policy)
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

    server_config = {
        "data_type": observation_config["data_type"],
        "language_prompt": observation_config["language_prompt"],
        "control_frequency": float(control["control_frequency"]),
        "controller_frequency": float(control["controller_frequency"]),
        "single_arm_mode": bool(observation_config["single_arm_mode"]),
        "no_state_obs_mode": bool(observation_config["no_state_obs_mode"]),
        "steps_per_inference": steps_per_inference,
        "action_horizon": configured_horizon,
    }
    bridge = RobotBridgeClient(
        address=str(connection["address"]),
        port=int(connection["port"]),
        token=_resolve_token(connection),
        add_port=_optional_bool(connection.get("add_port")),
        retry_interval_s=float(connection.get("retry_interval_s", 1.0)),
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
    task = str(observation_config["language_prompt"])
    previous_chunk: np.ndarray | None = None

    try:
        print("[client] Waiting for robot warmup observation")
        warmup_obs_seq, warmup_observation = bridge.receive_observation()
        warmup_frame = _prepare_observation(
            warmup_observation,
            state_dim=state_dim,
            image_keys=robot_image_keys,
            empty_cameras=empty_cameras,
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

        iteration = 0
        last_status_time = time.monotonic()
        while max_iterations <= 0 or iteration < max_iterations:
            obs_seq, observation = bridge.receive_observation()
            observation_saver.submit(iteration + 1, obs_seq, observation)
            frame = _prepare_observation(
                observation,
                state_dim=state_dim,
                image_keys=robot_image_keys,
                empty_cameras=empty_cameras,
            )
            start = time.perf_counter()
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
            bridge.send_action(action, obs_seq)
            bridge.receive_action_ack(obs_seq, timeout=action_ack_timeout_s)
            if rtc_on:
                previous_chunk = action_norm
            iteration += 1

            now = time.monotonic()
            if now - last_status_time >= status_interval_s:
                print(
                    f"[client] iter={iteration} obs_seq={obs_seq} "
                    f"inference_ms={inference_ms:.1f} action_shape={action.shape}"
                )
                last_status_time = now
    except KeyboardInterrupt:
        print("[client] Interrupted")
    finally:
        observation_saver.close()
        try:
            bridge.send_state("stop")
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
