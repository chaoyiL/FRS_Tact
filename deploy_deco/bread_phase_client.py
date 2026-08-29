"""Deploy the single-weight, phase-conditioned Bread DECO policy."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from pathlib import Path
import time
from typing import Any

import numpy as np

from .bread_phase_controller import BreadPhaseController, BreadPhaseTimeout
from .config import load_config, make_server_config, resolve_checkpoint, resolve_token, section
from .remote_client import check


DEFAULT_CONFIG = Path(__file__).resolve().parent / "configs" / "deploy_deco_bread_phase.yaml"


def prepare_phase_action(
    controller: BreadPhaseController,
    observation: Mapping[str, Any],
    action: np.ndarray,
    *,
    now_s: float | None = None,
) -> tuple[int, np.ndarray]:
    state = controller.observe(observation["observation.state"], now_s=now_s)
    return controller.phase, controller.mask_chunk(state, action)


def run(config_path: Path, max_iterations_override: int | None = None) -> None:
    from .bridge_client import RobotBridgeClient
    from .policy import DECOPolicy

    config = check(config_path)
    checkpoint = resolve_checkpoint(config)
    connection = section(config, "connection")
    runtime = section(config, "runtime")
    timeout_s = float(runtime.get("right_phase_timeout_s", 15.0))
    max_iterations = (
        int(runtime.get("max_iterations", 0))
        if max_iterations_override is None
        else int(max_iterations_override)
    )
    observation_timeout = float(connection.get("observation_timeout_s", 30.0))
    seed = int(config.get("seed", 0))
    controller = BreadPhaseController(timeout_s=timeout_s)

    print(f"[startup] Loading Bread phase DECO on {config['device']}...")
    started = time.perf_counter()
    policy = DECOPolicy(checkpoint, device=str(config["device"]), verify_hash=False)
    if policy.phase_count != 2:
        raise ValueError("Bread deployment requires a two-phase TorchScript artifact")
    print(f"[startup] Bread phase DECO loaded in {time.perf_counter() - started:.1f}s")

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
        _, warmup = bridge.receive_observation(timeout=observation_timeout)
        for index in range(int(runtime.get("warmup_runs", 1))):
            policy.predict(warmup, seed=seed + index, phase_id=0)
        if runtime.get("auto_start", False) is False:
            input("[client] Ready. Press Enter to start Bread phase deployment... ")
        bridge.send_state("start")

        iteration = 0
        while max_iterations <= 0 or iteration < max_iterations:
            obs_seq, observation = bridge.receive_observation(timeout=observation_timeout)
            state = controller.observe(observation["observation.state"])
            phase_id = controller.phase
            action = policy.predict(
                observation,
                seed=seed + int(runtime.get("warmup_runs", 1)) + iteration,
                phase_id=phase_id,
            )
            bridge.send_action(controller.mask_chunk(state, action), obs_seq)
            iteration += 1
            print(f"[client] iteration={iteration} obs_seq={obs_seq} phase={phase_id}")
        if max_iterations > 0:
            bridge.receive_observation(timeout=observation_timeout)
    except BreadPhaseTimeout as error:
        print(f"[client] STOP: {error}")
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
    parser.add_argument("--check", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.check:
        load_config(args.config)
        check(args.config)
    else:
        run(args.config, args.max_iterations)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

