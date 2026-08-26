"""Run direct Pi0.5 tactile decoding through the existing scheduling bridge."""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from .deployment import DeploymentConfig, load_deployment_config
from .protocol import ScheduleChunkEnd, ScheduleChunkStart, ScheduleSteerAck, ScheduleSteerRequest


DEFAULT_CONFIG = Path(__file__).resolve().parent / "configs" / "deploy_baseline_pi05.yaml"


def check(config_path: Path) -> DeploymentConfig:
    """Parse the configuration and print its immutable input digest without connecting."""
    path = Path(config_path).expanduser().resolve()
    contents = path.read_bytes()
    digest = hashlib.sha256(contents).hexdigest()
    print(f"[check] deploy config path: {path}")
    print(f"[check] deploy config sha256: {digest}")
    return load_deployment_config(path)


def _make_runtime(config: DeploymentConfig) -> Any:
    """Import heavyweight JAX/Torch deployment code only for a real run."""
    from .checkpoint import load_decoder
    from .deployment import expected_source_contract
    from .policy import Pi05VisualPolicy
    from .runtime import DirectDecoderRuntime
    from .tactile_encoder import FrozenTactileEncoder

    policy = Pi05VisualPolicy(config)
    encoder = FrozenTactileEncoder(config.tactile_encoder.checkpoint, tactile_keys=config.tactile_encoder.tactile_keys)
    decoder = load_decoder(config.direct_decoder.checkpoint, device=config.direct_decoder.device, expected_source=expected_source_contract(config))
    return DirectDecoderRuntime(policy=policy, tactile_encoder=encoder, decoder=decoder, max_normalized_action_abs=config.control.max_normalized_action_abs, max_normalized_delta_rms=config.control.max_normalized_delta_rms, device=config.direct_decoder.device)


def _warmup_observation() -> dict[str, np.ndarray]:
    return {
        "observation.state": np.zeros(20, dtype=np.float32),
        "observation.images.camera0": np.zeros((224, 224, 3), dtype=np.uint8),
        "observation.images.camera1": np.zeros((224, 224, 3), dtype=np.uint8),
    }


def warmup(runtime: Any, config: DeploymentConfig) -> None:
    """Exercise visual chunk creation only; deliberately never emits a robot action."""
    for index in range(config.runtime.warmup_runs):
        warmup_id = -(index + 1)
        runtime.begin_chunk(warmup_id, _warmup_observation(), config.observation.language_prompt, seed=config.source.seed, num_steps=config.source.sample_steps)
        runtime.end_chunk(warmup_id)


def run_schedule(bridge: Any, runtime: Any, *, task: str, observation_timeout_s: float, action_ack_timeout_s: float, seed: int, sample_steps: int, max_iterations: int) -> None:
    """Process only start/steer/ack/end scheduling messages with fail-stop ordering."""
    if max_iterations < 0:
        raise ValueError("max_iterations must be non-negative")
    completed = 0
    previous_chunk_id: int | None = None
    while max_iterations == 0 or completed < max_iterations:
        start = bridge.receive_schedule_message(observation_timeout_s)
        if not isinstance(start, ScheduleChunkStart):
            raise RuntimeError(f"expected ScheduleChunkStart, got {type(start).__name__}")
        if start.execution_mode != "block":
            raise RuntimeError("direct decoder scheduling requires block execution mode")
        if start.action_horizon != 50:
            raise RuntimeError("direct decoder scheduling requires a 50-step action horizon")
        if previous_chunk_id is not None and start.chunk_id <= previous_chunk_id:
            raise RuntimeError("scheduling chunk ids must be strictly increasing")
        active = False
        try:
            runtime.begin_chunk(start.chunk_id, start.observation, task, seed=seed, num_steps=sample_steps)
            active = True
            bridge.send_frs_chunk_ready(start.obs_seq, start.chunk_id)
            while True:
                message = bridge.receive_schedule_message(observation_timeout_s)
                if isinstance(message, ScheduleChunkEnd):
                    if message.chunk_id != start.chunk_id:
                        raise RuntimeError("ScheduleChunkEnd does not match the active chunk")
                    runtime.end_chunk(start.chunk_id)
                    active = False
                    previous_chunk_id = start.chunk_id
                    completed += 1
                    break
                if not isinstance(message, ScheduleSteerRequest):
                    raise RuntimeError(f"expected ScheduleSteerRequest, got {type(message).__name__}")
                if message.chunk_id != start.chunk_id:
                    raise RuntimeError("ScheduleSteerRequest does not match the active chunk")
                result = runtime.steer_action(message.chunk_id, message.request_id, message.observation, message.action_index)
                action = np.asarray(result.selected_action)
                if action.shape != (20,) or action.dtype.kind != "f" or not np.isfinite(action).all():
                    raise RuntimeError("direct decoder must return one finite full physical 20D action")
                bridge.send_frs_steer_action(message.chunk_id, message.request_id, message.action_index, action)
                acknowledgement = bridge.receive_schedule_message(action_ack_timeout_s)
                if not isinstance(acknowledgement, ScheduleSteerAck):
                    raise RuntimeError(f"expected ScheduleSteerAck, got {type(acknowledgement).__name__}")
                if (acknowledgement.chunk_id, acknowledgement.request_id, acknowledgement.action_index) != (message.chunk_id, message.request_id, message.action_index):
                    raise RuntimeError("ScheduleSteerAck does not match its request")
        finally:
            if active:
                try:
                    runtime.end_chunk(start.chunk_id)
                except Exception:
                    pass


def _token(config: DeploymentConfig) -> str | None:
    token = config.connection.token or os.environ.get(config.connection.token_env)
    if config.connection.require_token and not token:
        raise ValueError(f"{config.connection.token_env} must be set because connection.require_token is true")
    return token


def run(config_path: Path, max_iterations_override: int | None = None, *, bridge_factory: Callable[..., Any] | None = None, runtime_factory: Callable[[DeploymentConfig], Any] | None = None) -> None:
    """Run a real bridge session and clean up/stop best-effort after every exit path."""
    config = check(config_path)
    max_iterations = config.runtime.max_iterations if max_iterations_override is None else int(max_iterations_override)
    if max_iterations < 0:
        raise ValueError("max_iterations must be non-negative")
    if bridge_factory is None:
        from .bridge_client import RobotBridgeClient

        bridge_factory = RobotBridgeClient
    runtime = (runtime_factory or _make_runtime)(config)
    bridge: Any | None = None
    try:
        warmup(runtime, config)
        bridge = bridge_factory(config.connection.address, config.connection.port, _token(config), retry_interval_s=config.connection.retry_interval_s, ping_interval_s=config.connection.ping_interval_s, ping_timeout_s=config.connection.ping_timeout_s)
        if not config.runtime.auto_start:
            input("[client] Ready. Press Enter to send START to the robot server... ")
        bridge.send_state("start")
        run_schedule(bridge, runtime, task=config.observation.language_prompt, observation_timeout_s=config.connection.observation_timeout_s, action_ack_timeout_s=config.connection.action_ack_timeout_s, seed=config.source.seed, sample_steps=config.source.sample_steps, max_iterations=max_iterations)
    finally:
        if bridge is not None:
            try:
                bridge.send_state("stop")
            except Exception:
                pass
            try:
                bridge.close()
            except Exception:
                pass


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--max-iterations", type=int)
    parser.add_argument("--check", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        if args.check:
            check(args.config)
        else:
            run(args.config, args.max_iterations)
    except KeyboardInterrupt:
        return 130
    except Exception as error:
        print(f"[client] fail-stop: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
