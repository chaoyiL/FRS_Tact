"""Deploy a plain JAX pi0.5 policy through legacy VB robot action chunks."""

from __future__ import annotations

import argparse
import hashlib
import logging
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from .bridge_client import RobotBridgeClient
from .deployment import (
    ObservationSaver,
    cleanup_deployment_resources,
    configure_deployment_logging,
    load_deployment_config_bytes,
    make_policy_config,
    optional_bool,
    prepare_observation,
    print_startup_summary,
    resolve_token,
    section,
    start_observation_saver,
    submit_observation,
)

DEFAULT_CONFIG = Path(__file__).resolve().parent / "configs" / "deploy_pi05.yaml"
LOGGER = logging.getLogger(__name__)


def _make_policy(policy_config: Any) -> Any:
    """Construct the JAX policy only for a real deployment run."""
    from .policy import Pi05RemotePolicy

    return Pi05RemotePolicy(policy_config)


def _jax_runtime() -> tuple[str, tuple[Any, ...]]:
    """Initialize JAX lazily after deployment logging has been configured."""
    import jax

    return jax.default_backend(), tuple(jax.devices())


def predict_robot_action_chunk(
    policy: Any,
    observation: Mapping[str, Any],
    task: str,
    *,
    seed: int,
    num_steps: int,
) -> np.ndarray:
    """Predict and validate one complete, robot-space pi0.5 action chunk."""
    normalized = policy.predict_action_chunk(observation, task, seed=seed, num_steps=num_steps)
    expected_model = (1, policy.config.action_horizon, policy.config.action_dim)
    if normalized.shape != expected_model:
        raise ValueError(f"pi0.5 action must have shape {expected_model}, got {normalized.shape}")
    action = np.asarray(policy.unnormalize_actions(normalized[0]), dtype=np.float32)
    expected_robot = (policy.config.action_horizon, policy.config.robot_action_dim)
    if action.shape != expected_robot or not np.isfinite(action).all():
        raise ValueError(f"robot action must be finite with shape {expected_robot}, got {action.shape}")
    return np.ascontiguousarray(action)


def run_legacy_loop(
    bridge: RobotBridgeClient,
    policy: Any,
    *,
    task: str,
    image_keys: Sequence[str],
    observation_timeout_s: float,
    seed: int,
    sample_steps: int,
    max_iterations: int,
    saver: ObservationSaver | None,
) -> None:
    """Process legacy chunks serially: receive observation, then send its full action."""
    completed = 0
    while max_iterations <= 0 or completed < max_iterations:
        obs_seq, raw_observation = bridge.receive_observation(timeout=observation_timeout_s)
        observation = prepare_observation(
            raw_observation,
            state_dim=policy.config.state_dim,
            image_keys=image_keys,
        )
        submit_observation(saver, completed + 1, obs_seq, raw_observation, logger=LOGGER)
        action = predict_robot_action_chunk(
            policy, observation, task, seed=seed, num_steps=sample_steps
        )
        bridge.send_action(action, obs_seq)
        completed += 1


def run(config_path: Path, max_iterations_override: int | None = None) -> None:
    """Run the plain pi0.5 legacy-chunk client."""
    config_path = config_path.expanduser().resolve()
    config_bytes = config_path.read_bytes()
    config_sha256 = hashlib.sha256(config_bytes).hexdigest()
    print(f"[startup] deploy config path: {config_path}")
    print(f"[startup] deploy config sha256: {config_sha256}")
    config = load_deployment_config_bytes(config_bytes, mode="pi05")
    policy_config = make_policy_config(config, config_path)
    connection = section(config, "connection")
    observation_config = section(config, "observation")
    runtime = section(config, "runtime")
    seed = int(config.get("seed", 0))
    sample_steps = int(config.get("num_steps", 10))
    if sample_steps <= 0:
        raise ValueError("num_steps must be positive")
    max_iterations = (
        int(runtime.get("max_iterations", 0))
        if max_iterations_override is None
        else int(max_iterations_override)
    )
    if max_iterations < 0:
        raise ValueError("max_iterations must be non-negative")

    backend, devices = _jax_runtime()
    print_startup_summary(
        config,
        policy_config,
        mode="pi05",
        backend=backend,
        devices=devices,
    )
    print("[startup] Loading pi0.5 model...")
    load_started = time.perf_counter()
    policy = _make_policy(policy_config)
    print(f"[startup] pi0.5 model loaded in {time.perf_counter() - load_started:.1f}s")
    image_keys = tuple(policy.robot_image_keys)
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
        saver = start_observation_saver(
            config.get("logging", {}) or {},
            image_keys,
            saver_factory=ObservationSaver,
            logger=LOGGER,
        )
        timeout = float(connection.get("observation_timeout_s", 30.0))
        task = str(observation_config["language_prompt"])
        warmup_runs = int(runtime.get("warmup_runs", 1))

        print("[client] Waiting for robot warmup observation")
        obs_seq, raw_observation = bridge.receive_observation(timeout=timeout)
        warmup_observation = prepare_observation(
            raw_observation,
            state_dim=policy.config.state_dim,
            image_keys=image_keys,
        )
        for _ in range(warmup_runs):
            predict_robot_action_chunk(
                policy,
                warmup_observation,
                task,
                seed=seed,
                num_steps=sample_steps,
            )
        print(f"[client] Warmup observation sequence: {obs_seq}")
        if runtime.get("auto_start", False) is False:
            input("[client] Ready. Press Enter to send START to the robot server... ")
        bridge.send_state("start")
        run_legacy_loop(
            bridge,
            policy,
            task=task,
            image_keys=image_keys,
            observation_timeout_s=timeout,
            seed=seed,
            sample_steps=sample_steps,
            max_iterations=max_iterations,
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
    configure_deployment_logging()
    run(args.config, args.max_iterations)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
