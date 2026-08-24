#!/usr/bin/env python3
"""Offline A/B comparison of direct PI0.5 output and FRS ``action_vla``.

This tool never creates a robot bridge client. Run ``direct`` and ``frs`` in
separate processes, then use ``compare`` on the two NPZ artifacts.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
for _import_root in (ROOT, ROOT / "deploy_pi05/src"):
    if str(_import_root) not in sys.path:
        sys.path.insert(0, str(_import_root))
DEFAULT_CONFIG = ROOT / "deploy_pi05/configs/deploy_pi05_frs.yaml"
DEFAULT_OBSERVATION = (
    ROOT
    / "deploy_pi05/outputs/pi05_frs_observations/20260822_001132/step_000001"
)
GRIPPER_INDICES = (9, 19)
IMAGE_FILE_BY_KEY = {
    "observation.images.camera0": "observation_images_camera0.jpg",
    "observation.images.camera1": "observation_images_camera1.jpg",
    "observation.images.tactile_left_0": "observation_images_tactile_left_0.jpg",
    "observation.images.tactile_right_0": "observation_images_tactile_right_0.jpg",
    "observation.images.tactile_left_1": "observation_images_tactile_left_1.jpg",
    "observation.images.tactile_right_1": "observation_images_tactile_right_1.jpg",
}
MATCH_METADATA_KEYS = (
    "config_path",
    "observation_dir",
    "checkpoint",
    "assets_dir",
    "asset_id",
    "prompt",
    "seed",
    "num_steps",
    "warmup_runs",
)


def load_saved_observation(step_dir: Path) -> dict[str, np.ndarray]:
    """Reconstruct one saved deployment observation with RGB channel order."""

    directory = Path(step_dir).expanduser().resolve()
    state_path = directory / "observation_state.npy"
    if not state_path.is_file():
        raise FileNotFoundError(f"saved observation state not found: {state_path}")
    state = np.asarray(np.load(state_path, allow_pickle=False), dtype=np.float32)
    if state.shape != (20,) or not np.isfinite(state).all():
        raise ValueError(f"saved observation state must be finite with shape (20,), got {state.shape}")

    observation = {"observation.state": np.array(state, copy=True)}
    for key, filename in IMAGE_FILE_BY_KEY.items():
        path = directory / filename
        bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise FileNotFoundError(f"saved observation image not found or unreadable: {path}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        if rgb.ndim != 3 or rgb.shape[-1] != 3 or rgb.dtype != np.uint8:
            raise ValueError(f"saved image must be uint8 HWC RGB, got {rgb.shape} {rgb.dtype}: {path}")
        observation[key] = rgb
    return observation


def write_artifact(
    path: Path,
    *,
    mode: str,
    normalized: Any,
    robot_action: Any,
    metadata: Mapping[str, object],
) -> None:
    """Write one self-describing, pickle-free A/B artifact."""

    if mode not in {"direct", "frs"}:
        raise ValueError(f"artifact mode must be 'direct' or 'frs', got {mode!r}")
    normalized_array = np.asarray(normalized, dtype=np.float32)
    robot_array = np.asarray(robot_action, dtype=np.float32)
    for name, array in (("normalized", normalized_array), ("robot_action", robot_array)):
        if array.ndim != 3 or array.shape[0] != 1 or array.shape[-1] < 20:
            raise ValueError(f"{name} must have shape [1, horizon, >=20], got {array.shape}")
        if not np.isfinite(array).all():
            raise ValueError(f"{name} contains NaN or Inf")
    if normalized_array.shape[1] != robot_array.shape[1]:
        raise ValueError("normalized and robot action horizons must match")
    missing = [key for key in MATCH_METADATA_KEYS if key not in metadata]
    if missing:
        raise ValueError(f"artifact metadata is missing keys: {missing}")

    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        destination,
        mode=np.asarray(mode),
        normalized=normalized_array,
        robot_action=robot_array,
        metadata_json=np.asarray(json.dumps(dict(metadata), sort_keys=True)),
    )


def _load_artifact(path: Path, expected_mode: str) -> dict[str, Any]:
    source = Path(path).expanduser().resolve()
    with np.load(source, allow_pickle=False) as artifact:
        required = {"mode", "normalized", "robot_action", "metadata_json"}
        missing = required - set(artifact.files)
        if missing:
            raise ValueError(f"artifact {source} is missing arrays: {sorted(missing)}")
        mode = str(artifact["mode"].item())
        if mode != expected_mode:
            raise ValueError(f"expected {expected_mode!r} artifact, got {mode!r}: {source}")
        normalized = np.asarray(artifact["normalized"], dtype=np.float32)
        robot_action = np.asarray(artifact["robot_action"], dtype=np.float32)
        metadata = json.loads(str(artifact["metadata_json"].item()))
    if not isinstance(metadata, dict):
        raise ValueError(f"artifact metadata must decode to an object: {source}")
    return {
        "normalized": normalized,
        "robot_action": robot_action,
        "metadata": metadata,
    }


def _max_abs(left: np.ndarray, right: np.ndarray, name: str) -> float:
    if left.shape != right.shape:
        raise ValueError(f"{name} shapes differ: direct={left.shape}, frs={right.shape}")
    if not np.isfinite(left).all() or not np.isfinite(right).all():
        raise ValueError(f"{name} contains NaN or Inf")
    return float(np.max(np.abs(left - right)))


def _first_close_indices(
    robot_action: np.ndarray,
    *,
    width_slope: float,
    width_offset: float,
    close_threshold: float,
) -> dict[str, int | None]:
    if robot_action.ndim != 3 or robot_action.shape[0] != 1 or robot_action.shape[-1] < 20:
        raise ValueError(f"robot_action must have shape [1, horizon, >=20], got {robot_action.shape}")
    grippers = robot_action[0][:, list(GRIPPER_INDICES)]
    commanded = np.clip((grippers - width_offset) / width_slope, 0.01, 0.04)
    result: dict[str, int | None] = {}
    for side_index, side in enumerate(("left", "right")):
        indices = np.flatnonzero(commanded[:, side_index] < close_threshold)
        result[side] = None if not len(indices) else int(indices[0])
    return result


def compare_artifacts(
    direct_path: Path,
    frs_path: Path,
    *,
    tolerance: float = 1e-6,
    width_slope: float = 1.77,
    width_offset: float = 0.05,
    close_threshold: float = 0.02,
) -> dict[str, Any]:
    """Validate provenance and compare direct PI0.5 with FRS ``action_vla``."""

    if not np.isfinite(tolerance) or tolerance < 0:
        raise ValueError("tolerance must be finite and non-negative")
    if not np.isfinite(width_slope) or width_slope <= 0:
        raise ValueError("width_slope must be finite and positive")
    if not np.isfinite(width_offset) or not np.isfinite(close_threshold):
        raise ValueError("width_offset and close_threshold must be finite")
    direct = _load_artifact(direct_path, "direct")
    frs = _load_artifact(frs_path, "frs")
    for key in MATCH_METADATA_KEYS:
        left = direct["metadata"].get(key)
        right = frs["metadata"].get(key)
        if left != right:
            raise ValueError(f"artifact metadata differs for {key}: direct={left!r}, frs={right!r}")

    direct_normalized = direct["normalized"]
    frs_normalized = frs["normalized"]
    direct_robot = direct["robot_action"]
    frs_robot = frs["robot_action"]
    diffs = {
        "normalized_all": _max_abs(direct_normalized, frs_normalized, "normalized actions"),
        "normalized_grippers": _max_abs(
            direct_normalized[..., list(GRIPPER_INDICES)],
            frs_normalized[..., list(GRIPPER_INDICES)],
            "normalized grippers",
        ),
        "robot_action_all": _max_abs(direct_robot, frs_robot, "robot actions"),
        "robot_grippers": _max_abs(
            direct_robot[..., list(GRIPPER_INDICES)],
            frs_robot[..., list(GRIPPER_INDICES)],
            "robot grippers",
        ),
    }
    return {
        "passed": bool(max(diffs.values()) <= tolerance),
        "tolerance": float(tolerance),
        "max_abs_diff": diffs,
        "first_close_index": {
            "direct": _first_close_indices(
                direct_robot,
                width_slope=width_slope,
                width_offset=width_offset,
                close_threshold=close_threshold,
            ),
            "frs": _first_close_indices(
                frs_robot,
                width_slope=width_slope,
                width_offset=width_offset,
                close_threshold=close_threshold,
            ),
        },
        "metadata": direct["metadata"],
    }


def _load_runtime_inputs(config_path: Path, observation_dir: Path):
    from deploy_pi05.deployment import load_deployment_config, make_policy_config, section

    resolved_config = Path(config_path).expanduser().resolve()
    config = load_deployment_config(resolved_config, "frs")
    policy_config = make_policy_config(config, resolved_config)
    observation = load_saved_observation(observation_dir)
    prompt = str(section(config, "observation")["language_prompt"])
    seed = int(config.get("seed", 0))
    num_steps = int(config.get("num_steps", 10))
    warmup_runs = int(section(config, "runtime").get("warmup_runs", 1))
    metadata = {
        "config_path": str(resolved_config),
        "observation_dir": str(Path(observation_dir).expanduser().resolve()),
        "checkpoint": policy_config.checkpoint,
        "assets_dir": policy_config.assets_dir,
        "asset_id": policy_config.asset_id,
        "prompt": prompt,
        "seed": seed,
        "num_steps": num_steps,
        "warmup_runs": warmup_runs,
    }
    return config, policy_config, observation, metadata


def run_direct(config_path: Path, observation_dir: Path, output: Path) -> None:
    """Run only the source PI0.5 path and write an A artifact."""

    import jax

    from deploy_pi05.policy import Pi05RemotePolicy

    _, policy_config, observation, metadata = _load_runtime_inputs(config_path, observation_dir)
    policy = Pi05RemotePolicy(policy_config)
    for _ in range(int(metadata["warmup_runs"])):
        warmup = policy.predict_action_chunk(
            observation,
            str(metadata["prompt"]),
            seed=int(metadata["seed"]),
            num_steps=int(metadata["num_steps"]),
        )
        jax.block_until_ready(warmup)
    normalized = policy.predict_action_chunk(
        observation,
        str(metadata["prompt"]),
        seed=int(metadata["seed"]),
        num_steps=int(metadata["num_steps"]),
    )
    normalized_host = np.asarray(jax.device_get(normalized), dtype=np.float32)
    robot_action = policy.unnormalize_actions(normalized_host)
    write_artifact(
        output,
        mode="direct",
        normalized=normalized_host,
        robot_action=robot_action,
        metadata=metadata,
    )


def run_frs(config_path: Path, observation_dir: Path, output: Path) -> None:
    """Run the FRS chunk entrypoint and save only its source ``action_vla``."""

    from deploy_pi05.deployment import section
    from deploy_pi05.frs_runtime import FRSRuntime
    from deploy_pi05.policy import Pi05RemotePolicy

    config, policy_config, observation, metadata = _load_runtime_inputs(config_path, observation_dir)
    policy = Pi05RemotePolicy(policy_config)
    runtime = FRSRuntime(
        section(config, "frs"),
        config_path=Path(config_path).expanduser().resolve(),
        policy=policy,
        source_sample_steps=int(metadata["num_steps"]),
    )
    runtime.reset_episode(observation)
    for _ in range(int(metadata["warmup_runs"])):
        runtime.warmup(
            observation,
            str(metadata["prompt"]),
            seed=int(metadata["seed"]),
            sample_steps=int(metadata["num_steps"]),
        )
    ready = runtime.begin_chunk(
        0,
        observation,
        str(metadata["prompt"]),
        seed=int(metadata["seed"]),
        num_steps=int(metadata["num_steps"]),
    )
    write_artifact(
        output,
        mode="frs",
        normalized=ready.action_vla_normalized,
        robot_action=ready.action_vla,
        metadata=metadata,
    )


def _paired_predictions(
    policy: Any,
    runtime: Any,
    observation: Mapping[str, np.ndarray],
    *,
    prompt: str,
    seed: int,
    num_steps: int,
    warmup_runs: int,
) -> dict[str, np.ndarray]:
    """Replay one policy RNG state through direct and FRS source paths."""

    import jax

    runtime.reset_episode(observation)
    for _ in range(warmup_runs):
        runtime.warmup(
            observation,
            prompt,
            seed=seed,
            sample_steps=num_steps,
        )
    if policy._rng is None:
        raise RuntimeError("PI0.5 RNG was not initialized by warmup")
    rng_before_target = policy._rng
    rng_seed_before_target = policy._rng_seed

    direct = policy.predict_action_chunk(
        observation,
        prompt,
        seed=seed,
        num_steps=num_steps,
    )
    direct_host = np.asarray(jax.device_get(direct), dtype=np.float32)
    direct_robot = np.asarray(policy.unnormalize_actions(direct_host), dtype=np.float32)

    policy._rng = rng_before_target
    policy._rng_seed = rng_seed_before_target
    ready = runtime.begin_chunk(
        0,
        observation,
        prompt,
        seed=seed,
        num_steps=num_steps,
    )
    return {
        "direct_normalized": direct_host,
        "direct_robot": direct_robot,
        "frs_normalized": np.asarray(ready.action_vla_normalized, dtype=np.float32),
        "frs_robot": np.asarray(ready.action_vla, dtype=np.float32),
    }


def run_paired(
    config_path: Path,
    observation_dir: Path,
    direct_output: Path,
    frs_output: Path,
) -> None:
    """Run the strongest same-process A/B check with an identical RNG state."""

    from deploy_pi05.deployment import section
    from deploy_pi05.frs_runtime import FRSRuntime
    from deploy_pi05.policy import Pi05RemotePolicy

    config, policy_config, observation, metadata = _load_runtime_inputs(config_path, observation_dir)
    policy = Pi05RemotePolicy(policy_config)
    runtime = FRSRuntime(
        section(config, "frs"),
        config_path=Path(config_path).expanduser().resolve(),
        policy=policy,
        source_sample_steps=int(metadata["num_steps"]),
    )
    result = _paired_predictions(
        policy,
        runtime,
        observation,
        prompt=str(metadata["prompt"]),
        seed=int(metadata["seed"]),
        num_steps=int(metadata["num_steps"]),
        warmup_runs=int(metadata["warmup_runs"]),
    )
    write_artifact(
        direct_output,
        mode="direct",
        normalized=result["direct_normalized"],
        robot_action=result["direct_robot"],
        metadata=metadata,
    )
    write_artifact(
        frs_output,
        mode="frs",
        normalized=result["frs_normalized"],
        robot_action=result["frs_robot"],
        metadata=metadata,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for mode in ("direct", "frs"):
        command = subparsers.add_parser(mode, help=f"write the {mode} A/B artifact")
        command.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
        command.add_argument("--observation-dir", type=Path, default=DEFAULT_OBSERVATION)
        command.add_argument("--output", type=Path, required=True)
    paired = subparsers.add_parser(
        "paired",
        help="write both artifacts in one process with an identical replayed PI0.5 RNG state",
    )
    paired.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    paired.add_argument("--observation-dir", type=Path, default=DEFAULT_OBSERVATION)
    paired.add_argument("--direct-output", type=Path, required=True)
    paired.add_argument("--frs-output", type=Path, required=True)
    compare = subparsers.add_parser("compare", help="compare direct and FRS artifacts")
    compare.add_argument("--direct", type=Path, required=True)
    compare.add_argument("--frs", type=Path, required=True)
    compare.add_argument("--tolerance", type=float, default=1e-6)
    compare.add_argument("--width-slope", type=float, default=1.77)
    compare.add_argument("--width-offset", type=float, default=0.05)
    compare.add_argument("--close-threshold", type=float, default=0.02)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "direct":
        run_direct(args.config, args.observation_dir, args.output)
        print(f"wrote direct artifact: {Path(args.output).expanduser().resolve()}")
        return 0
    if args.command == "frs":
        run_frs(args.config, args.observation_dir, args.output)
        print(f"wrote FRS artifact: {Path(args.output).expanduser().resolve()}")
        return 0
    if args.command == "paired":
        run_paired(
            args.config,
            args.observation_dir,
            args.direct_output,
            args.frs_output,
        )
        print(f"wrote direct artifact: {Path(args.direct_output).expanduser().resolve()}")
        print(f"wrote FRS artifact: {Path(args.frs_output).expanduser().resolve()}")
        return 0
    report = compare_artifacts(
        args.direct,
        args.frs,
        tolerance=args.tolerance,
        width_slope=args.width_slope,
        width_offset=args.width_offset,
        close_threshold=args.close_threshold,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
