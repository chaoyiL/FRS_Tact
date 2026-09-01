#!/usr/bin/env python3
"""Evaluate independent saved Pick Tube RDP observations without a robot bridge."""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from deploy_pick_tube_rdp import (
    IMAGE_SIZE,
    PickTubeRDPRuntime,
    load_config,
    load_policy,
    wire_action_for_profile,
)
from reactive_diffusion_policy.common.pick_tube_action_contract import (
    SINGLE_RIGHT_ARM_7X10,
    StateActionProfile,
    resolve_state_action_profile,
)


IMAGE_FILES = {
    "observation.images.camera0": "camera0_rgb.jpg",
    "observation.images.camera1": "camera1_rgb.jpg",
    "observation.images.tactile_left_0": "camera0_left_tactile.jpg",
    "observation.images.tactile_right_0": "camera0_right_tactile.jpg",
    "observation.images.tactile_left_1": "camera1_left_tactile.jpg",
    "observation.images.tactile_right_1": "camera1_right_tactile.jpg",
}
_STEP_DIRECTORY = re.compile(r"step_(\d+)$")


@dataclass(frozen=True)
class Snapshot:
    step: int
    timestamp: float
    left_pose: np.ndarray
    right_pose: np.ndarray
    left_gripper: float
    right_gripper: float
    images: dict[str, np.ndarray]


def _finite_vector(path: Path, *, size: int, name: str) -> np.ndarray:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"could not read {name}: {path}") from exc
    vector = np.asarray(value, dtype=np.float64).reshape(-1)
    if vector.shape != (size,) or not np.isfinite(vector).all():
        raise ValueError(f"{name} must contain {size} finite values")
    return vector


def _read_rgb(path: Path, key: str) -> np.ndarray:
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError(f"could not decode image for {key}: {path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    if rgb.ndim != 3 or rgb.shape[-1] != 3 or rgb.size == 0:
        raise ValueError(f"{key} must decode as a nonempty HWC RGB image")
    if rgb.shape[:2] != (IMAGE_SIZE, IMAGE_SIZE):
        rgb = cv2.resize(rgb, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_AREA)
    return np.ascontiguousarray(rgb)


def load_snapshots(path: Path | str) -> list[Snapshot]:
    """Load saved bridge observations in numeric step order."""

    root = Path(path)
    if not root.is_dir():
        raise ValueError(f"snapshot directory does not exist: {root}")
    step_directories: list[tuple[int, Path]] = []
    for child in root.iterdir():
        match = _STEP_DIRECTORY.fullmatch(child.name)
        if child.is_dir() and match is not None:
            step_directories.append((int(match.group(1)), child))
    if not step_directories:
        raise ValueError(f"no step_###### directories found in {root}")

    snapshots: list[Snapshot] = []
    for step, directory in sorted(step_directories):
        snapshots.append(
            Snapshot(
                step=step,
                timestamp=float(_finite_vector(directory / "timestamp.json", size=1, name="timestamp")[0]),
                left_pose=np.concatenate(
                    (
                        _finite_vector(directory / "robot0_eef_pos.json", size=3, name="robot0 position"),
                        _finite_vector(
                            directory / "robot0_eef_rot_axis_angle.json",
                            size=3,
                            name="robot0 rotation",
                        ),
                    )
                ).astype(np.float32),
                right_pose=np.concatenate(
                    (
                        _finite_vector(directory / "robot1_eef_pos.json", size=3, name="robot1 position"),
                        _finite_vector(
                            directory / "robot1_eef_rot_axis_angle.json",
                            size=3,
                            name="robot1 rotation",
                        ),
                    )
                ).astype(np.float32),
                left_gripper=float(
                    _finite_vector(
                        directory / "robot0_gripper_width.json", size=1, name="robot0 gripper"
                    )[0]
                ),
                right_gripper=float(
                    _finite_vector(
                        directory / "robot1_gripper_width.json", size=1, name="robot1 gripper"
                    )[0]
                ),
                images={key: _read_rgb(directory / filename, key) for key, filename in IMAGE_FILES.items()},
            )
        )
    return snapshots


def pose_matrix(pose: Any) -> np.ndarray:
    values = np.asarray(pose, dtype=np.float64)
    if values.shape != (6,) or not np.isfinite(values).all():
        raise ValueError(f"pose must be finite with shape (6,), got {values.shape}")
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = Rotation.from_rotvec(values[3:]).as_matrix()
    matrix[:3, 3] = values[:3]
    return matrix


def matrix_pose(matrix: Any) -> np.ndarray:
    value = np.asarray(matrix, dtype=np.float64)
    if value.shape != (4, 4) or not np.isfinite(value).all():
        raise ValueError(f"pose matrix must be finite with shape (4,4), got {value.shape}")
    return np.concatenate((value[:3, 3], Rotation.from_matrix(value[:3, :3]).as_rotvec()))


def _start_pose_pair(start_poses: Sequence[Any] | Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    if isinstance(start_poses, Mapping):
        try:
            return np.asarray(start_poses["left"]), np.asarray(start_poses["right"])
        except KeyError as exc:
            raise ValueError("start poses mapping must contain left and right") from exc
    if len(start_poses) != 2:
        raise ValueError("start poses must contain left and right poses")
    return np.asarray(start_poses[0]), np.asarray(start_poses[1])


def build_server_state(
    start_poses: Sequence[Any] | Mapping[str, Any], snapshot: Snapshot
) -> np.ndarray:
    """Reconstruct the deployment bridge's 20D relative bimanual state."""

    left_start, right_start = _start_pose_pair(start_poses)
    grippers = np.asarray((snapshot.left_gripper, snapshot.right_gripper), dtype=np.float64)
    if not np.isfinite(grippers).all():
        raise ValueError("snapshot gripper widths must be finite")
    left_relative = np.linalg.inv(pose_matrix(left_start)) @ pose_matrix(snapshot.left_pose)
    right_relative = np.linalg.inv(pose_matrix(right_start)) @ pose_matrix(snapshot.right_pose)
    left_from_right = np.linalg.inv(pose_matrix(snapshot.right_pose)) @ pose_matrix(snapshot.left_pose)
    state = np.concatenate(
        (
            matrix_pose(left_relative),
            [snapshot.left_gripper],
            matrix_pose(right_relative),
            [snapshot.right_gripper],
            matrix_pose(left_from_right),
        )
    ).astype(np.float32)
    if state.shape != (20,) or not np.isfinite(state).all():
        raise ValueError("reconstructed server state must be finite with shape (20,)")
    return state



def _prediction_arrays(results: Mapping[str, Any]) -> dict[str, np.ndarray]:
    expected = {
        "states": ((20,), np.float32),
        "policy_actions": ((10,), np.float32),
        "wire_actions": ((20,), np.float32),
        "right_poses": ((6,), np.float32),
        "step_ids": ((), np.int64),
        "timestamps": ((), np.float64),
        "latency_ms": ((), np.float64),
    }
    missing = [name for name in expected if name not in results]
    if missing:
        raise ValueError(f"results are missing required arrays: {missing}")
    arrays = {name: np.asarray(results[name]) for name in expected}
    count = len(arrays["states"])
    if count == 0:
        raise ValueError("results must contain at least one snapshot")
    for name, (trailing_shape, dtype) in expected.items():
        array = arrays[name]
        if array.shape != (count, *trailing_shape):
            raise ValueError(f"{name} must have shape {(count, *trailing_shape)}, got {array.shape}")
        if array.dtype != dtype or not np.isfinite(array).all():
            raise ValueError(f"{name} must be finite with dtype {dtype}")
    return arrays


def predict_independent_snapshots(
    runtime: PickTubeRDPRuntime,
    profile: StateActionProfile,
    snapshots: Sequence[Snapshot],
    seed: int,
) -> dict[str, np.ndarray]:
    """Predict one action per saved observation with no temporal carry-over."""

    if profile != SINGLE_RIGHT_ARM_7X10:
        raise ValueError("offline evaluator requires the single-right-arm-7x10 profile")
    if not snapshots:
        raise ValueError("at least one snapshot is required")

    import time
    import torch

    start_poses = (snapshots[0].left_pose, snapshots[0].right_pose)
    collected: dict[str, list[np.ndarray | float | int]] = {
        "states": [],
        "policy_actions": [],
        "wire_actions": [],
        "right_poses": [],
        "step_ids": [],
        "timestamps": [],
        "latency_ms": [],
    }
    for snapshot in snapshots:
        state = build_server_state(start_poses, snapshot)
        observation: dict[str, Any] = dict(snapshot.images)
        observation["observation.state"] = state
        runtime.reset()
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        started = time.perf_counter()
        prediction = runtime.predict(observation)
        if getattr(getattr(runtime, "device", None), "type", None) == "cuda":
            torch.cuda.synchronize(runtime.device)
        latency_ms = (time.perf_counter() - started) * 1000.0
        policy_action = np.asarray(prediction[0], dtype=np.float32)
        if policy_action.shape != (1, profile.action_dim) or not np.isfinite(policy_action).all():
            raise ValueError(
                f"runtime must return a finite (1,{profile.action_dim}) action, got {policy_action.shape}"
            )
        wire_action = np.asarray(
            wire_action_for_profile(policy_action, observation, profile), dtype=np.float32
        )
        if wire_action.shape != (1, 20) or not np.isfinite(wire_action).all():
            raise ValueError(f"wire action must be finite with shape (1,20), got {wire_action.shape}")
        collected["states"].append(state)
        collected["policy_actions"].append(policy_action[0])
        collected["wire_actions"].append(wire_action[0])
        collected["right_poses"].append(snapshot.right_pose)
        collected["step_ids"].append(snapshot.step)
        collected["timestamps"].append(snapshot.timestamp)
        collected["latency_ms"].append(latency_ms)
    results = {
        "states": np.asarray(collected["states"], dtype=np.float32),
        "policy_actions": np.asarray(collected["policy_actions"], dtype=np.float32),
        "wire_actions": np.asarray(collected["wire_actions"], dtype=np.float32),
        "right_poses": np.asarray(collected["right_poses"], dtype=np.float32),
        "step_ids": np.asarray(collected["step_ids"], dtype=np.int64),
        "timestamps": np.asarray(collected["timestamps"], dtype=np.float64),
        "latency_ms": np.asarray(collected["latency_ms"], dtype=np.float64),
    }
    return _prediction_arrays(results)


def _write_action_overview(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    """Render an OpenCV overview of independent right-arm action responses."""

    height, width = 620, 1200
    image = np.full((height, width, 3), 250, dtype=np.uint8)
    cv2.putText(
        image,
        "Independent RDP snapshot actions (right arm)",
        (28, 42),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (25, 25, 25),
        2,
        cv2.LINE_AA,
    )
    panel_height = 160
    actions = arrays["policy_actions"]
    names = ("translation xyz", "rotation 6D", "gripper")
    groups = (actions[:, :3], actions[:, 3:9], actions[:, 9:])
    colors = ((40, 80, 220), (60, 170, 60), (220, 120, 30), (160, 70, 180), (30, 150, 180), (80, 80, 80))
    for panel, (name, values) in enumerate(zip(names, groups, strict=True)):
        top = 72 + panel * panel_height
        bottom = top + panel_height - 34
        cv2.rectangle(image, (28, top), (width - 28, bottom), (215, 215, 215), 1)
        cv2.putText(image, name, (36, top + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (40, 40, 40), 1, cv2.LINE_AA)
        maximum = float(np.max(np.abs(values)))
        scale = 1.0 if maximum == 0.0 else maximum
        x_values = np.linspace(44, width - 44, len(values)).round().astype(np.int32)
        midline = (top + 32 + bottom) // 2
        cv2.line(image, (44, midline), (width - 44, midline), (190, 190, 190), 1)
        for column in range(values.shape[1]):
            ys = (midline - values[:, column] / scale * (bottom - top - 42) / 2).round().astype(np.int32)
            points = np.column_stack((x_values, ys)).reshape(-1, 1, 2)
            if len(points) == 1:
                cv2.circle(image, tuple(points[0, 0]), 2, colors[column], -1)
            else:
                cv2.polylines(image, [points], False, colors[column], 2, cv2.LINE_AA)
        cv2.putText(image, f"range +/- {scale:.4g}", (width - 210, top + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (70, 70, 70), 1, cv2.LINE_AA)
    if not cv2.imwrite(str(path), image):
        raise OSError(f"could not write action overview: {path}")


def write_reports(output_dir: Path | str, results: Mapping[str, Any]) -> dict[str, Path]:
    """Write numeric predictions, per-snapshot records, CSV, JSON, and an OpenCV plot."""

    import csv

    arrays = _prediction_arrays(results)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    paths = {
        "predictions": output / "predictions.npz",
        "trajectory_csv": output / "trajectory.csv",
        "summary": output / "summary.json",
        "action_overview": output / "action_overview.png",
        "snapshot_responses": output / "snapshot_responses.jsonl",
    }
    np.savez_compressed(paths["predictions"], **arrays)
    with paths["trajectory_csv"].open("w", newline="", encoding="utf-8") as file:
        columns = (
            "step_id", "timestamp", "latency_ms", "right_x", "right_y", "right_z",
            "right_rx", "right_ry", "right_rz",
            *[f"policy_action_{index}" for index in range(10)],
            *[f"wire_action_{index}" for index in range(20)],
        )
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        for index in range(len(arrays["states"])):
            row = {
                "step_id": int(arrays["step_ids"][index]),
                "timestamp": float(arrays["timestamps"][index]),
                "latency_ms": float(arrays["latency_ms"][index]),
            }
            row.update({f"right_{name}": float(value) for name, value in zip(("x", "y", "z", "rx", "ry", "rz"), arrays["right_poses"][index], strict=True)})
            row.update({f"policy_action_{column}": float(value) for column, value in enumerate(arrays["policy_actions"][index])})
            row.update({f"wire_action_{column}": float(value) for column, value in enumerate(arrays["wire_actions"][index])})
            writer.writerow(row)
    with paths["snapshot_responses"].open("w", encoding="utf-8") as file:
        for index in range(len(arrays["states"])):
            response = {
                "step_id": int(arrays["step_ids"][index]),
                "timestamp": float(arrays["timestamps"][index]),
                "latency_ms": float(arrays["latency_ms"][index]),
                "state": arrays["states"][index].tolist(),
                "policy_action": arrays["policy_actions"][index].tolist(),
                "wire_action": arrays["wire_actions"][index].tolist(),
            }
            file.write(json.dumps(response, allow_nan=False, sort_keys=True) + "\\n")
    latency = arrays["latency_ms"]
    summary = {
        "snapshots": int(len(arrays["states"])),
        "step_range": [int(arrays["step_ids"][0]), int(arrays["step_ids"][-1])],
        "latency_ms": {
            "mean": float(np.mean(latency)),
            "p95": float(np.percentile(latency, 95)),
            "max": float(np.max(latency)),
        },
        "right_translation_norm": {
            "mean": float(np.mean(np.linalg.norm(arrays["policy_actions"][:, :3], axis=1))),
            "max": float(np.max(np.linalg.norm(arrays["policy_actions"][:, :3], axis=1))),
        },
        "profile": SINGLE_RIGHT_ARM_7X10.name,
        "evaluation_mode": "independent_snapshot_reset",
    }
    with paths["summary"].open("w", encoding="utf-8") as file:
        json.dump(summary, file, allow_nan=False, indent=2, sort_keys=True)
    _write_action_overview(paths["action_overview"], arrays)
    return paths


def run_evaluation(
    config_path: Path | str,
    obs_dir: Path | str,
    output_dir: Path | str,
    device_name: str,
    seed: int,
) -> dict[str, Path]:
    """Construct the RDP runtime and evaluate saved snapshots without bridge access."""

    import torch

    from reactive_diffusion_policy.deploy.tactile_encoder_torch import load_tactile_resnet18
    from reactive_diffusion_policy.model.tactile_pca import BimanualTactilePCA

    config_file = Path(config_path).expanduser().resolve()
    config = load_config(config_file)
    model_config = config["model"]
    control = config["control"]
    profile = resolve_state_action_profile(str(model_config.get("state_action_profile")))
    if profile != SINGLE_RIGHT_ARM_7X10:
        raise ValueError("offline evaluator requires state_action_profile single-right-arm-7x10")
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    ldp_checkpoint = Path(str(model_config["ldp_checkpoint"])).expanduser().resolve()
    at_checkpoint = Path(str(model_config["at_checkpoint"])).expanduser().resolve()
    encoder_dir = Path(str(model_config["tactile_encoder_dir"])).expanduser().resolve()
    tactile_pca_path = Path(str(model_config["tactile_pca_path"])).expanduser().resolve()
    missing = [path for path in (ldp_checkpoint, at_checkpoint, tactile_pca_path) if not path.is_file()]
    if not encoder_dir.is_dir():
        missing.append(encoder_dir)
    if missing:
        formatted = "\\n".join(f"  - {path}" for path in missing)
        raise FileNotFoundError(f"Missing RDP deployment files:\\n{formatted}")
    tactile_pca = BimanualTactilePCA.from_npz(tactile_pca_path, device=device)
    policy, checkpoint_cfg = load_policy(
        ldp_checkpoint,
        at_checkpoint,
        device,
        int(model_config.get("num_inference_steps", 8)),
        tactile_pca.output_dim,
        profile=profile,
        artifact_verification=str(model_config.get("artifact_verification", "strict")),
        tactile_pca_path=tactile_pca_path,
    )
    runtime = PickTubeRDPRuntime(
        policy,
        load_tactile_resnet18(encoder_dir, device=device),
        device,
        tactile_pca,
        slow_update_interval=int(control.get("slow_update_interval", 5)),
        dataset_obs_temporal_downsample_ratio=int(checkpoint_cfg.dataset_obs_temporal_downsample_ratio),
        n_obs_steps=int(checkpoint_cfg.n_obs_steps),
        profile=profile,
    )
    snapshots = load_snapshots(obs_dir)
    return write_reports(
        output_dir,
        predict_independent_snapshots(runtime, profile, snapshots, seed=int(seed)),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--obs-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    report_paths = run_evaluation(
        arguments.config,
        arguments.obs_dir,
        arguments.output_dir,
        arguments.device,
        arguments.seed,
    )
    for label, path in report_paths.items():
        print(f"[rdp-offline] {label}: {path}")
