#!/usr/bin/env python3
"""Run frozen-initial-observation counterfactual RDP rollouts offline."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
from pathlib import Path
import random
import sys
import time
from typing import Any, Iterator

import cv2
import numpy as np
import torch
import yaml


IMAGE_KEYS = (
    "observation.images.camera0",
    "observation.images.camera1",
    "observation.images.tactile_left_0",
    "observation.images.tactile_right_0",
    "observation.images.tactile_left_1",
    "observation.images.tactile_right_1",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--failure-trial", required=True, type=Path)
    parser.add_argument("--success-trial", required=True, type=Path)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--warmup-runs", type=int, default=2)
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--repeat-seed", type=int, default=7)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--raw-output", required=True, type=Path)
    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        value = yaml.safe_load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a YAML mapping")
    return value


def resolve(repo: Path, value: Any) -> Path:
    path = Path(str(value)).expanduser()
    return (path if path.is_absolute() else repo / path).resolve()


def load_initial_observation(trial: Path) -> dict[str, np.ndarray]:
    with (trial / "steps.jsonl").open(encoding="utf-8") as stream:
        row = json.loads(next(stream))
    if int(row["iter_idx"]) != 0:
        raise ValueError(f"{trial}: first row is not iter 0")
    state = np.asarray(row["state"], dtype=np.float32)
    if state.shape != (20,) or not np.isfinite(state).all():
        raise ValueError(f"{trial}: invalid initial state {state.shape}")
    observation = {"observation.state": state}
    image_dir = trial / "images" / "initial"
    for key in IMAGE_KEYS:
        image = cv2.imread(str(image_dir / f"{key}.png"), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(image_dir / f"{key}.png")
        observation[key] = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return observation


@contextmanager
def seeded(seed: int) -> Iterator[None]:
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.random.get_rng_state()
    cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    try:
        random.seed(seed)
        np.random.seed(seed % (2**32))
        torch.manual_seed(seed)
        yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.random.set_rng_state(torch_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)


def rot6d_to_matrix(values: np.ndarray) -> np.ndarray:
    first = np.asarray(values[:3], dtype=np.float64)
    second = np.asarray(values[3:6], dtype=np.float64)
    first = first / max(float(np.linalg.norm(first)), 1e-12)
    second = second - first * float(np.dot(first, second))
    second = second / max(float(np.linalg.norm(second)), 1e-12)
    return np.stack((first, second, np.cross(first, second)), axis=-1)


def rotation_angle_degrees(matrix: np.ndarray) -> float:
    cosine = np.clip((float(np.trace(matrix)) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


def summarize_rollout(actions: np.ndarray, slow_flags: np.ndarray) -> dict[str, Any]:
    right = np.asarray(actions[:, 10:20], dtype=np.float64)
    xyz_mm = right[:, :3] * 1000.0
    norms_mm = np.linalg.norm(xyz_mm, axis=1)
    rotations = np.stack([rot6d_to_matrix(row[3:9]) for row in right])
    rotation_deg = np.asarray([rotation_angle_degrees(matrix) for matrix in rotations])
    cumulative = np.eye(4, dtype=np.float64)
    for xyz, rotation in zip(right[:, :3], rotations, strict=True):
        delta = np.eye(4, dtype=np.float64)
        delta[:3, :3] = rotation
        delta[:3, 3] = xyz
        cumulative = cumulative @ delta
    return {
        "right_xyz_mean_mm": np.mean(xyz_mm, axis=0).tolist(),
        "right_xyz_std_mm": np.std(xyz_mm, axis=0).tolist(),
        "right_step_translation_norm_mm": {
            "mean": float(np.mean(norms_mm)),
            "p95": float(np.percentile(norms_mm, 95)),
            "max": float(np.max(norms_mm)),
        },
        "right_step_rotation_deg": {
            "mean": float(np.mean(rotation_deg)),
            "p95": float(np.percentile(rotation_deg, 95)),
            "max": float(np.max(rotation_deg)),
        },
        "right_cumulative_xyz_mm": (cumulative[:3, 3] * 1000.0).tolist(),
        "right_cumulative_translation_norm_mm": float(
            np.linalg.norm(cumulative[:3, 3]) * 1000.0
        ),
        "right_cumulative_rotation_deg": rotation_angle_degrees(cumulative[:3, :3]),
        "right_gripper_m": {
            "mean": float(np.mean(right[:, 9])),
            "min": float(np.min(right[:, 9])),
            "max": float(np.max(right[:, 9])),
            "final": float(right[-1, 9]),
        },
        "slow_update_steps": np.flatnonzero(slow_flags).astype(int).tolist(),
    }


def scalar_summary(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "min": float(np.min(values)),
        "p05": float(np.percentile(values, 5)),
        "median": float(np.median(values)),
        "p95": float(np.percentile(values, 95)),
        "max": float(np.max(values)),
    }


def run_rollout(
    runtime: Any,
    observation: dict[str, np.ndarray],
    seed: int,
    steps: int,
    warmup_runs: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, float]:
    with seeded(seed):
        for _ in range(warmup_runs):
            runtime.reset()
            runtime.predict(observation)
        runtime.reset()
        actions = []
        slow_flags = []
        started = time.perf_counter()
        for _ in range(steps):
            action, slow_update = runtime.predict(observation)
            actions.append(np.asarray(action, dtype=np.float32).reshape(20))
            slow_flags.append(bool(slow_update))
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - started
    return np.stack(actions), np.asarray(slow_flags), elapsed


def aggregate_trial(actions: np.ndarray, summaries: list[dict[str, Any]]) -> dict[str, Any]:
    cumulative_translation = np.asarray(
        [item["right_cumulative_translation_norm_mm"] for item in summaries]
    )
    cumulative_rotation = np.asarray(
        [item["right_cumulative_rotation_deg"] for item in summaries]
    )
    step_norm = np.asarray(
        [item["right_step_translation_norm_mm"]["mean"] for item in summaries]
    )
    seed_std = np.std(actions[:, :, 10:20], axis=0)
    return {
        "right_cumulative_translation_norm_mm": scalar_summary(cumulative_translation),
        "right_cumulative_rotation_deg": scalar_summary(cumulative_rotation),
        "right_mean_step_translation_norm_mm": scalar_summary(step_norm),
        "right_action_seed_std_rms": float(np.sqrt(np.mean(seed_std**2))),
        "per_seed": summaries,
    }


def main() -> None:
    args = parse_args()
    if min(args.steps, args.warmup_runs + 1, args.seeds, args.repeats) < 1:
        raise ValueError("steps, warmup-runs, seeds, and repeats must be positive")
    repo = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo))
    from deploy_pick_tube_rdp import PickTubeRDPRuntime, load_policy
    from reactive_diffusion_policy.deploy.tactile_encoder_torch import load_tactile_resnet18
    from reactive_diffusion_policy.model.tactile_pca import BimanualTactilePCA

    config_path = args.config.expanduser().resolve()
    config = load_config(config_path)
    model = config["model"]
    control = config["control"]
    slow_update_interval = control["slow_update_interval"]
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    ldp_path = resolve(repo, model["ldp_checkpoint"])
    at_path = resolve(repo, model["at_checkpoint"])
    pca_path = resolve(repo, model["tactile_pca_path"])
    encoder_path = resolve(repo, model["tactile_encoder_dir"])
    tactile_pca = BimanualTactilePCA.from_npz(pca_path, device=device)
    policy, checkpoint_cfg = load_policy(
        ldp_path,
        at_path,
        device,
        int(model.get("num_inference_steps", 8)),
        tactile_pca.output_dim,
        slow_update_interval=slow_update_interval,
        artifact_verification=str(model.get("artifact_verification", "strict")),
        tactile_pca_path=pca_path,
    )
    tactile_encoder = load_tactile_resnet18(encoder_path, device=device)
    runtime = PickTubeRDPRuntime(
        policy,
        tactile_encoder,
        device,
        tactile_pca,
        slow_update_interval=slow_update_interval,
        dataset_obs_temporal_downsample_ratio=int(
            checkpoint_cfg.dataset_obs_temporal_downsample_ratio
        ),
        n_obs_steps=int(checkpoint_cfg.n_obs_steps),
    )

    observations = {
        "failure": load_initial_observation(args.failure_trial.expanduser().resolve()),
        "success": load_initial_observation(args.success_trial.expanduser().resolve()),
    }
    seeds = list(range(args.seeds))
    all_actions: dict[str, np.ndarray] = {}
    all_summaries: dict[str, list[dict[str, Any]]] = {}
    timings: dict[str, list[float]] = {}
    for label, observation in observations.items():
        action_rows = []
        summary_rows = []
        elapsed_rows = []
        for seed in seeds:
            actions, slow_flags, elapsed = run_rollout(
                runtime, observation, seed, args.steps, args.warmup_runs, device
            )
            action_rows.append(actions)
            summary = summarize_rollout(actions, slow_flags)
            summary["seed"] = seed
            summary_rows.append(summary)
            elapsed_rows.append(elapsed)
            print(f"[{label}] seed={seed:02d} {elapsed:.3f}s", flush=True)
        all_actions[label] = np.stack(action_rows)
        all_summaries[label] = summary_rows
        timings[label] = elapsed_rows

    repeated = []
    for repeat in range(args.repeats):
        actions, slow_flags, elapsed = run_rollout(
            runtime,
            observations["failure"],
            args.repeat_seed,
            args.steps,
            args.warmup_runs,
            device,
        )
        repeated.append(actions)
        print(f"[repeat] run={repeat} seed={args.repeat_seed} {elapsed:.3f}s", flush=True)
    repeated_array = np.stack(repeated)
    reference = repeated_array[0]
    repeat_max_abs = [float(np.max(np.abs(row - reference))) for row in repeated_array]
    repeat_rms = [float(np.sqrt(np.mean((row - reference) ** 2))) for row in repeated_array]

    paired_action_rms = np.sqrt(
        np.mean((all_actions["failure"][:, :, 10:20] - all_actions["success"][:, :, 10:20]) ** 2, axis=(1, 2))
    )
    failure_translation = np.asarray(
        [item["right_cumulative_translation_norm_mm"] for item in all_summaries["failure"]]
    )
    success_translation = np.asarray(
        [item["right_cumulative_translation_norm_mm"] for item in all_summaries["success"]]
    )
    failure_rotation = np.asarray(
        [item["right_cumulative_rotation_deg"] for item in all_summaries["failure"]]
    )
    success_rotation = np.asarray(
        [item["right_cumulative_rotation_deg"] for item in all_summaries["success"]]
    )
    report = {
        "scope": "frozen initial observation, command-only open-loop counterfactual",
        "config": str(config_path),
        "model": {
            "ldp_checkpoint": str(ldp_path),
            "at_checkpoint": str(at_path),
            "tactile_pca_path": str(pca_path),
            "device": str(device),
            "num_inference_steps": int(model.get("num_inference_steps", 8)),
            "slow_update_interval": int(control.get("slow_update_interval", 5)),
        },
        "experiment": {
            "steps": args.steps,
            "warmup_runs": args.warmup_runs,
            "seeds": seeds,
            "repeat_seed": args.repeat_seed,
            "repeat_count": args.repeats,
        },
        "group1_fixed_input_fixed_seed_repeat": {
            "trial": str(args.failure_trial.expanduser().resolve()),
            "seed": args.repeat_seed,
            "max_abs_action_difference_vs_first": repeat_max_abs,
            "rms_action_difference_vs_first": repeat_rms,
            "array_equal_vs_first": [bool(np.array_equal(row, reference)) for row in repeated_array],
        },
        "group2_fixed_failure_input_20_seeds": aggregate_trial(
            all_actions["failure"], all_summaries["failure"]
        ),
        "group3_paired_inputs_same_seeds": {
            "failure_trial": str(args.failure_trial.expanduser().resolve()),
            "success_trial": str(args.success_trial.expanduser().resolve()),
            "right_action_paired_rms": scalar_summary(paired_action_rms),
            "failure_minus_success_cumulative_translation_norm_mm": scalar_summary(
                failure_translation - success_translation
            ),
            "failure_greater_translation_fraction": float(
                np.mean(failure_translation > success_translation)
            ),
            "failure_minus_success_cumulative_rotation_deg": scalar_summary(
                failure_rotation - success_rotation
            ),
            "failure_greater_rotation_fraction": float(
                np.mean(failure_rotation > success_rotation)
            ),
            "failure_input": aggregate_trial(all_actions["failure"], all_summaries["failure"]),
            "success_input": aggregate_trial(all_actions["success"], all_summaries["success"]),
        },
        "timing_seconds": {
            label: scalar_summary(np.asarray(values)) for label, values in timings.items()
        },
        "limitations": [
            "The trial does not record the online RNG state or checkpoint hashes.",
            "Only every fifth image is recorded, so a 30 Hz closed-loop replay is impossible.",
            "The initial observation is held frozen for all steps; absolute accumulated motion is diagnostic, not a reconstructed robot trajectory.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    np.savez_compressed(
        args.raw_output,
        failure_actions=all_actions["failure"],
        success_actions=all_actions["success"],
        repeated_actions=repeated_array,
        seeds=np.asarray(seeds, dtype=np.int64),
    )
    print(f"report={args.output.resolve()}")
    print(f"raw={args.raw_output.resolve()}")


if __name__ == "__main__":
    main()
