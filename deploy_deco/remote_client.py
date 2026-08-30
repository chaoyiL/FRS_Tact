"""Run a DECO TorchScript policy through the existing VB robot bridge."""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from .artifact import TACTILE_FIELD_ORDER, load_sidecar
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
DEFAULT_OBSERVE_ONLY_OUTPUT_ROOT = Path(__file__).resolve().parent / "outputs"


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


def _rgb_uint8(value: Any, key: str) -> np.ndarray:
    image = np.asarray(value)
    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError(f"{key} must be HWC RGB, got {image.shape}")
    if image.dtype == np.uint8:
        return image
    if not np.issubdtype(image.dtype, np.floating):
        raise ValueError(f"{key} must be uint8 or float RGB, got {image.dtype}")
    if not np.isfinite(image).all() or image.min() < 0.0 or image.max() > 1.0:
        raise ValueError(f"{key} float RGB values must be finite in [0,1]")
    return np.rint(image * 255.0).astype(np.uint8)


def _array_range(value: Any, name: str) -> dict[str, Any]:
    array = np.asarray(value)
    if not array.size or not np.isfinite(array).all():
        raise ValueError(f"{name} must be non-empty and finite")
    return {
        "shape": list(array.shape),
        "min": float(array.min()),
        "max": float(array.max()),
    }


def save_observe_only_bundle(
    output_root: Path, observation: dict[str, Any], policy: Any, action: Any
) -> Path:
    """Save the policy-bound observation and its predicted action for inspection."""
    from PIL import Image

    keys = (*policy.image_keys, *policy.tactile_keys)
    if len(keys) != 6:
        raise ValueError("observe-only bundles require exactly six image streams")
    output_root.mkdir(parents=True, exist_ok=True)
    bundle = output_root / datetime.now().strftime("observe_only_%Y%m%d_%H%M%S")
    bundle.mkdir()
    image_summaries: list[dict[str, Any]] = []
    for key in keys:
        if key not in observation:
            raise ValueError(f"observe-only observation is missing {key}")
        source = np.asarray(observation[key])
        rgb = _rgb_uint8(source, key)
        Image.fromarray(rgb, mode="RGB").save(bundle / f"{key.replace('.', '_')}.png")
        image_summaries.append(
            {
                "key": key,
                "shape": list(source.shape),
                "dtype": str(source.dtype),
                "min": float(source.min()),
                "max": float(source.max()),
            }
        )
    summary = {
        "images": image_summaries,
        "state": _array_range(observation["observation.state"], "state"),
        "action": _array_range(action, "action"),
    }
    (bundle / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return bundle


def run(
    config_path: Path,
    max_iterations_override: int | None = None,
    *,
    server_dry_run: bool = False,
    observe_only: bool = False,
) -> None:
    from .bridge_client import RobotBridgeClient
    from .policy import DECOPolicy

    if server_dry_run and observe_only:
        raise ValueError("--server-dry-run and --observe-only cannot be used together")
    config = check(config_path)
    runtime = section(config, "runtime")
    max_iterations = (
        int(runtime.get("max_iterations", 0))
        if max_iterations_override is None
        else int(max_iterations_override)
    )
    if max_iterations < 0:
        raise ValueError("max_iterations must be nonnegative")
    if server_dry_run and max_iterations <= 0:
        raise ValueError("--server-dry-run requires a positive --max-iterations")
    profile = deployment_profile(config)
    checkpoint = resolve_checkpoint(config)
    connection = section(config, "connection")
    observation_config = section(config, "observation")
    seed = int(config.get("seed", 0))
    warmup_runs = int(runtime.get("warmup_runs", 1))
    observation_timeout = float(connection.get("observation_timeout_s", 30.0))
    black_camera0 = bool(observation_config.get("black_camera0", False))

    def policy_observation(observation: dict[str, Any]) -> dict[str, Any]:
        return (
            project_right_observation(observation, black_camera0=black_camera0)
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
    if black_camera0:
        print("[startup] camera0 input replaced with training-matched black frames")
    tactile_keys = getattr(policy, "tactile_keys", ())
    artifact_mode = "stage2_tactile" if tactile_keys else "stage1_vision"
    print(f"[startup] artifact_mode={artifact_mode}")
    if tactile_keys:
        print(f"[startup] Stage 2 tactile key order: {tactile_keys}")
    if observe_only and tuple(tactile_keys) != TACTILE_FIELD_ORDER:
        raise ValueError("--observe-only requires a Stage 2 six-stream tactile artifact")

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
        warmup_policy_observation = policy_observation(warmup)
        if observe_only:
            action = None
            for index in range(max(1, warmup_runs)):
                action = policy.predict(warmup_policy_observation, seed=seed + index)
            assert action is not None
            bundle = save_observe_only_bundle(
                DEFAULT_OBSERVE_ONLY_OUTPUT_ROOT,
                warmup_policy_observation,
                policy,
                action,
            )
            print(f"[client] Observe-only bundle saved to {bundle}")
            return
        for index in range(warmup_runs):
            warmup_started = time.perf_counter()
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
        if max_iterations > 0 and not server_dry_run:
            bridge.receive_observation(timeout=observation_timeout)
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
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument(
        "--server-dry-run",
        action="store_true",
        help="run a bounded action loop without waiting for the final server observation",
    )
    modes.add_argument(
        "--observe-only",
        action="store_true",
        help="run inference on the warmup observation and save a local inspection bundle",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.check:
        check(args.config)
    else:
        run(
            args.config,
            args.max_iterations,
            server_dry_run=args.server_dry_run,
            observe_only=args.observe_only,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
