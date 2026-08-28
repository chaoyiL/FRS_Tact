"""Run a DECO TorchScript policy through the existing VB robot bridge."""

from __future__ import annotations

import argparse
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from .artifact import load_sidecar
from .config import (
    SINGLE_RIGHT_ARM_PROFILE,
    deployment_profile,
    load_config,
    make_server_config,
    resolve_checkpoint,
    resolve_token,
    section,
    validate_artifact_contract,
)
from .right_arm_adapter import expand_right_action, project_right_observation

DEFAULT_CONFIG = Path(__file__).resolve().parent / "configs" / "deploy_deco.yaml"


def check(config_path: Path) -> dict[str, Any]:
    config = load_config(config_path)
    checkpoint = resolve_checkpoint(config)
    metadata = load_sidecar(checkpoint, verify_hash=True)
    validate_artifact_contract(config, metadata)
    print(
        "[check] DECO artifact valid: "
        f"checkpoint={checkpoint} epoch={metadata.get('epoch')} "
        f"horizon={metadata['output']['action'][1]} "
        f"sample_hz={metadata['expected_sample_hz']}"
    )
    return config


def run(config_path: Path, max_iterations_override: int | None = None) -> None:
    from .bridge_client import RobotBridgeClient
    from .policy import DECOPolicy
    config = check(config_path)
    profile = deployment_profile(config)
    checkpoint = resolve_checkpoint(config)
    connection = section(config, "connection")
    runtime = section(config, "runtime")
    seed = int(config.get("seed", 0))
    warmup_runs = int(runtime.get("warmup_runs", 1))
    max_iterations = (
        int(runtime.get("max_iterations", 0))
        if max_iterations_override is None
        else int(max_iterations_override)
    )
    if max_iterations < 0:
        raise ValueError("max_iterations must be nonnegative")
    observation_timeout = float(connection.get("observation_timeout_s", 30.0))

    def policy_observation(observation: dict[str, Any]) -> dict[str, Any]:
        return (
            project_right_observation(observation)
            if profile == SINGLE_RIGHT_ARM_PROFILE
            else observation
        )

    def wire_action(action: Any, observation: dict[str, Any]) -> Any:
        return (
            expand_right_action(action, observation)
            if profile == SINGLE_RIGHT_ARM_PROFILE
            else action
        )

    print(f"[startup] Loading DECO TorchScript on {config['device']}...")
    started = time.perf_counter()
    policy = DECOPolicy(
        checkpoint,
        device=str(config["device"]),
        verify_hash=False,
    )
    print(f"[startup] DECO loaded in {time.perf_counter() - started:.1f}s")
    print(
        "[startup] contract "
        f"state={policy.state_dim} action={policy.action_dim} "
        f"horizon={policy.action_horizon} sample_hz={policy.expected_sample_hz:g}"
    )

    bridge: RobotBridgeClient | None = None
    try:
        bridge = RobotBridgeClient(
            address=str(connection["address"]),
            port=int(connection["port"]),
            token=resolve_token(connection),
            add_port=connection.get("add_port"),
            retry_interval_s=float(connection.get("retry_interval_s", 1.0)),
            ping_interval_s=float(connection.get("ping_interval_s", 20.0)),
            ping_timeout_s=float(connection.get("ping_timeout_s", 20.0)),
        )
        bridge.send_config(make_server_config(config))
        print("[client] Waiting for robot warmup observation")
        warmup_seq, warmup = bridge.receive_observation(timeout=observation_timeout)
        for index in range(warmup_runs):
            warmup_started = time.perf_counter()
            warmup_policy_observation = policy_observation(warmup)
            policy.predict(warmup_policy_observation, seed=seed + index)
            print(
                f"[client] Warmup {index + 1}/{warmup_runs}: "
                f"{(time.perf_counter() - warmup_started) * 1000:.1f}ms"
            )
        print(f"[client] Warmup observation sequence: {warmup_seq}")
        if runtime.get("auto_start", False) is False:
            input("[client] Ready. Press Enter to send START to the robot server... ")
        bridge.send_state("start")

        iteration = 0
        while max_iterations <= 0 or iteration < max_iterations:
            obs_seq, observation = bridge.receive_observation(timeout=observation_timeout)
            inference_started = time.perf_counter()
            policy_input = policy_observation(observation)
            policy_action = policy.predict(
                policy_input, seed=seed + warmup_runs + iteration
            )
            action = wire_action(policy_action, observation)
            inference_ms = (time.perf_counter() - inference_started) * 1000.0
            bridge.send_action(action, obs_seq)
            iteration += 1
            print(
                f"[client] iteration={iteration} obs_seq={obs_seq} "
                f"inference_ms={inference_ms:.1f} action_shape={action.shape}"
            )
    except KeyboardInterrupt:
        print("[client] Interrupted")
    finally:
        if bridge is not None:
            try:
                bridge.send_state("stop")
            except Exception:
                pass
            bridge.close()
        print("[client] Stopped")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--max-iterations", type=int)
    parser.add_argument(
        "--check",
        action="store_true",
        help="validate config, metadata, and SHA256 without loading PyTorch or connecting",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.check:
        check(args.config)
    else:
        run(args.config, args.max_iterations)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
