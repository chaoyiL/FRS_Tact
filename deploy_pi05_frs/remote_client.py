"""Deploy a JAX pi0.5 + FRS policy through the existing VB robot bridge."""

from __future__ import annotations

import argparse
import logging
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import jax
import numpy as np

from .bridge_client import RobotBridgeClient
from .deployment import (
    ObservationSaver,
    cleanup_deployment_resources,
    load_deployment_config,
    make_policy_config,
    make_server_config,
    optional_bool,
    prepare_observation,
    resolve_token,
    section,
    start_observation_saver,
    submit_observation,
)
from .frs_protocol import FRSChunkEnd, FRSChunkStart, FRSSteerAck, FRSSteerRequest
from .frs_runtime import FRSChunkReady, FRSRuntime, FRSSteerResult
from .policy import Pi05RemotePolicy

DEFAULT_CONFIG = Path(__file__).resolve().parent / "configs" / "deploy_pi05.yaml"
LOGGER = logging.getLogger(__name__)


def load_config(path: Path) -> dict[str, Any]:
    """Load the FRS profile from the shared deployment configuration."""
    return load_deployment_config(path, "frs")


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
    saver: ObservationSaver | None,
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
        submit_observation(
            saver,
            completed + 1,
            start.obs_seq,
            start.observation,
            logger=LOGGER,
        )
        observation = prepare_observation(
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
            frame = prepare_observation(
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
    connection = section(config, "connection")
    observation_config = section(config, "observation")
    runtime = section(config, "runtime")
    seed = int(config.get("seed", 0))
    sample_steps = int(config.get("num_steps", 10))
    if sample_steps <= 0:
        raise ValueError("num_steps must be positive")
    max_chunks = (
        int(runtime.get("max_iterations", 0))
        if max_iterations_override is None
        else int(max_iterations_override)
    )
    if max_chunks < 0:
        raise ValueError("max_iterations must be non-negative")
    timeout = float(connection.get("observation_timeout_s", 30.0))
    ack_timeout = float(connection["action_ack_timeout_s"])
    task = str(observation_config["language_prompt"])
    warmup_runs = int(runtime.get("warmup_runs", 1))

    policy_config = make_policy_config(config, config_path)
    print(f"[client] Loading pi0.5 checkpoint: {policy_config.checkpoint}")
    policy = Pi05RemotePolicy(policy_config)
    print(f"[client] JAX backend: {jax.default_backend()}")
    frs = FRSRuntime(
        section(config, "frs"),
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
    server_config = make_server_config(config, mode="frs", frs_runtime=frs)
    bridge: RobotBridgeClient | None = None
    saver: ObservationSaver | None = None
    try:
        bridge = RobotBridgeClient(
            address=str(connection["address"]),
            port=int(connection["port"]),
            token=resolve_token(connection),
            add_port=optional_bool(connection.get("add_port"), "connection.add_port"),
            retry_interval_s=float(connection.get("retry_interval_s", 1.0)),
            ping_interval_s=float(connection.get("ping_interval_s", 20.0)),
            ping_timeout_s=float(connection.get("ping_timeout_s", 20.0)),
        )
        bridge.send_config(server_config)
        saver = start_observation_saver(
            config.get("logging", {}) or {},
            image_keys,
            saver_factory=ObservationSaver,
            logger=LOGGER,
        )
        print("[client] Waiting for robot warmup observation")
        obs_seq, raw = bridge.receive_observation(timeout=timeout)
        warmup = prepare_observation(
            raw, state_dim=policy_config.state_dim, image_keys=image_keys
        )
        frs.reset_episode(warmup)
        for index in range(warmup_runs):
            started = time.perf_counter()
            frs.warmup(warmup, task, seed=seed + index, sample_steps=sample_steps)
            print(
                f"[client] Warmup {index + 1}/{warmup_runs}: {(time.perf_counter() - started) * 1000:.1f}ms"
            )
        print(f"[client] Warmup observation sequence: {obs_seq}")
        if runtime.get("auto_start", False) is False:
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
        cleanup_deployment_resources(bridge, saver, logger=LOGGER)
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
