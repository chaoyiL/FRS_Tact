#!/usr/bin/env python3
"""Offline-only parser and layer-by-layer diagnostics for SmolVLA runs.

This module reads saved artifacts; it never imports deployment clients or robot
server code.  Action fields are selected from their content, not the logger's
``prediction_source`` label, because legacy records used that label as a
fallback after the original prediction trace was unavailable.
"""


import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from scipy.spatial.transform import Rotation


ACTION_SHAPE = (20, 20)
ACTION_DIMENSION = 20
EXECUTED_ACTIONS = 10
WAYPOINT_DIMENSION = 14
STEP_PATTERN = re.compile(r"step_(\d+)$")


@dataclass(frozen=True)
class ChunkRecord:
    """A validated prediction trace record, independent of its logger label."""

    obs_seq: int
    timestamp: float
    prediction_source: str | None
    prediction_field: str
    raw_actions: np.ndarray
    selected_actions: np.ndarray
    absolute_waypoints: np.ndarray
    action_timestamps: np.ndarray


@dataclass(frozen=True)
class ControllerSample:
    """One controller feedback sample.

    Actual feedback is Quest/world frame.  Target fields are intentionally
    retained only as robot-frame values: the server incorrectly labelled the
    combined trace as ``pose_frame=quest``.
    """

    wall_time: float
    actual_left_quest_z: float
    target_left_robot_z: float | None
    pose_frame: str | None


@dataclass(frozen=True)
class SavedObservation:
    """One saved absolute Quest pose observation."""

    step: int
    timestamp: float
    left_pose: np.ndarray
    left_gripper: float
    right_pose: np.ndarray
    right_gripper: float


def _numeric_payload(value: Any) -> bool:
    if isinstance(value, np.ndarray):
        return value.dtype.kind in "iuf"
    if isinstance(value, (list, tuple)):
        return all(_numeric_payload(item) for item in value)
    return isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(
        value, (bool, np.bool_)
    )


def _finite_array(value: Any, shape: tuple[int, ...], label: str) -> np.ndarray:
    if not _numeric_payload(value):
        raise ValueError(f"{label} must be finite numeric data with shape {shape}")
    try:
        raw_array = np.asarray(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be finite numeric data with shape {shape}") from exc
    if raw_array.dtype.kind not in "iuf":
        raise ValueError(f"{label} must be finite numeric data with shape {shape}")
    array = raw_array.astype(np.float64, copy=False)
    if array.shape != shape or not np.isfinite(array).all():
        raise ValueError(f"{label} must be finite numeric data with shape {shape}, got {array.shape}")
    return np.array(array, dtype=np.float64, copy=True)


def _finite_scalar(value: Any, label: str) -> float:
    if not _numeric_payload(value) or isinstance(value, (list, tuple, np.ndarray)):
        raise ValueError(f"{label} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be a finite number")
    return result


def _positive_integer(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{label} must be an integer")
    result = int(value)
    if result < 0:
        raise ValueError(f"{label} must be non-negative")
    return result


def _json_lines(path: Path) -> Iterable[tuple[int, dict[str, Any]]]:
    path = Path(path)
    if not path.is_file():
        raise ValueError(f"trace file does not exist: {path}")
    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSON in {path} line {line_number}") from exc
        if not isinstance(value, dict):
            raise ValueError(f"{path} line {line_number} must contain a JSON object")
        yield line_number, value


def _exactly_one_prediction(row: dict[str, Any], label: str) -> tuple[str, Any]:
    supplied = [(field, row.get(field)) for field in ("vla_action", "frs_action") if row.get(field) is not None]
    if len(supplied) != 1:
        raise ValueError(f"{label} must contain exactly one non-null vla_action or frs_action")
    return supplied[0]


def load_chunk_trace(path: Path) -> list[ChunkRecord]:
    """Load action chunks, validating their full and executed action shapes."""
    chunks: list[ChunkRecord] = []
    for line_number, row in _json_lines(Path(path)):
        label = f"chunk trace line {line_number}"
        field, full_action = _exactly_one_prediction(row, label)
        raw_actions = _finite_array(full_action, ACTION_SHAPE, f"{label} {field}")
        selected_value = row.get("selected_raw_actions")
        if selected_value is None:
            raise ValueError(f"{label} selected_raw_actions is required")
        selected_actions = _finite_array(
            selected_value,
            (EXECUTED_ACTIONS, ACTION_DIMENSION),
            f"{label} selected_raw_actions",
        )
        if not np.array_equal(selected_actions, raw_actions[:EXECUTED_ACTIONS]):
            raise ValueError(f"{label} selected_raw_actions must equal the first 10 raw actions")
        absolute_waypoints = _finite_array(
            row.get("absolute_waypoints"),
            (EXECUTED_ACTIONS, WAYPOINT_DIMENSION),
            f"{label} absolute_waypoints",
        )
        action_timestamps = _finite_array(
            row.get("action_timestamps"),
            (ACTION_SHAPE[0],),
            f"{label} action_timestamps",
        )
        chunks.append(
            ChunkRecord(
                obs_seq=_positive_integer(row.get("obs_seq"), f"{label} obs_seq"),
                timestamp=_finite_scalar(row.get("time"), f"{label} time"),
                prediction_source=(
                    row["prediction_source"] if isinstance(row.get("prediction_source"), str) else None
                ),
                prediction_field=field,
                raw_actions=raw_actions,
                selected_actions=selected_actions,
                absolute_waypoints=absolute_waypoints,
                action_timestamps=action_timestamps,
            )
        )
    if not chunks:
        raise ValueError(f"chunk trace contains no records: {path}")
    return chunks


def load_controller_trace(path: Path) -> list[ControllerSample]:
    """Load controller feedback without treating robot-frame targets as Quest."""
    samples: list[ControllerSample] = []
    for line_number, row in _json_lines(Path(path)):
        pose_frame = row.get("pose_frame") if isinstance(row.get("pose_frame"), str) else None
        raw_samples = row.get("samples", [row])
        if not isinstance(raw_samples, list):
            raise ValueError(f"controller trace line {line_number} samples must be a list")
        for sample_index, raw in enumerate(raw_samples):
            label = f"controller trace line {line_number} sample {sample_index}"
            if not isinstance(raw, dict):
                raise ValueError(f"{label} must be a JSON object")
            target_value = raw.get("target_pose_left_z")
            samples.append(
                ControllerSample(
                    wall_time=_finite_scalar(raw.get("wall_time"), f"{label} wall_time"),
                    actual_left_quest_z=_finite_scalar(raw.get("ee_pose_left_z"), f"{label} ee_pose_left_z"),
                    target_left_robot_z=(
                        None if target_value is None else _finite_scalar(target_value, f"{label} target_pose_left_z")
                    ),
                    pose_frame=pose_frame,
                )
            )
    if not samples:
        raise ValueError(f"controller trace contains no samples: {path}")
    return samples


def _saved_vector(step_dir: Path, filename: str, dimension: int) -> np.ndarray:
    path = step_dir / filename
    if not path.is_file():
        raise ValueError(f"missing saved observation file: {path}")
    try:
        value = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON in saved observation file: {path}") from exc
    if not _numeric_payload(value):
        raise ValueError(f"{path} must contain finite numeric data")
    try:
        raw_array = np.asarray(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{path} must contain finite numeric data") from exc
    if raw_array.dtype.kind not in "iuf":
        raise ValueError(f"{path} must contain finite numeric data")
    flattened = raw_array.astype(np.float64, copy=False).reshape(-1)
    if flattened.shape != (dimension,) or not np.isfinite(flattened).all():
        raise ValueError(f"{path} must contain finite data with flattened shape ({dimension},)")
    return np.array(flattened, dtype=np.float64, copy=True)


def load_saved_observations(path: Path) -> list[SavedObservation]:
    """Load per-step saved absolute Quest poses in numeric step order."""
    root = Path(path)
    if not root.is_dir():
        raise ValueError(f"saved observation directory does not exist: {root}")
    steps: list[tuple[int, Path]] = []
    for child in root.iterdir():
        match = STEP_PATTERN.fullmatch(child.name)
        if child.is_dir() and match:
            steps.append((int(match.group(1)), child))
    if not steps:
        raise ValueError(f"saved observation directory has no step_ directories: {root}")
    observations: list[SavedObservation] = []
    for step, step_dir in sorted(steps):
        left_pose = np.concatenate(
            (_saved_vector(step_dir, "robot0_eef_pos.json", 3), _saved_vector(step_dir, "robot0_eef_rot_axis_angle.json", 3))
        )
        right_pose = np.concatenate(
            (_saved_vector(step_dir, "robot1_eef_pos.json", 3), _saved_vector(step_dir, "robot1_eef_rot_axis_angle.json", 3))
        )
        observations.append(
            SavedObservation(
                step=step,
                timestamp=float(_saved_vector(step_dir, "timestamp.json", 1)[0]),
                left_pose=left_pose,
                left_gripper=float(_saved_vector(step_dir, "robot0_gripper_width.json", 1)[0]),
                right_pose=right_pose,
                right_gripper=float(_saved_vector(step_dir, "robot1_gripper_width.json", 1)[0]),
            )
        )
    return observations


def _rotation_matrix(pose: np.ndarray) -> np.ndarray:
    return Rotation.from_rotvec(pose[3:]).as_matrix()


def reconstruct_state(observation: SavedObservation, reference: SavedObservation) -> np.ndarray:
    """Reconstruct the exact server 20D relative-start state from Quest poses.

    ``reference`` is mandatory.  Historical saved steps begin after the server
    warmup observation, so silently using step zero would only be approximate.
    """
    if not isinstance(observation, SavedObservation) or not isinstance(reference, SavedObservation):
        raise ValueError("observation and reference must be SavedObservation records")
    left_rotation = _rotation_matrix(observation.left_pose)
    right_rotation = _rotation_matrix(observation.right_pose)
    left_reference_rotation = _rotation_matrix(reference.left_pose)
    right_reference_rotation = _rotation_matrix(reference.right_pose)
    left_relative_xyz = left_reference_rotation.T @ (observation.left_pose[:3] - reference.left_pose[:3])
    right_relative_xyz = right_reference_rotation.T @ (observation.right_pose[:3] - reference.right_pose[:3])
    left_relative_rot = Rotation.from_matrix(left_reference_rotation.T @ left_rotation).as_rotvec()
    right_relative_rot = Rotation.from_matrix(right_reference_rotation.T @ right_rotation).as_rotvec()
    left_relative_right_xyz = right_rotation.T @ (observation.left_pose[:3] - observation.right_pose[:3])
    left_relative_right_rot = Rotation.from_matrix(right_rotation.T @ left_rotation).as_rotvec()
    state = np.concatenate(
        (
            left_relative_xyz,
            left_relative_rot,
            [observation.left_gripper],
            right_relative_xyz,
            right_relative_rot,
            [observation.right_gripper],
            left_relative_right_xyz,
            left_relative_right_rot,
        )
    )
    return _finite_array(state, (20,), "reconstructed state").astype(np.float32)


def _sign_runs(values: np.ndarray) -> list[dict[str, int | str]]:
    signs = np.sign(values).astype(np.int8)
    names = {-1: "negative", 0: "zero", 1: "positive"}
    runs: list[dict[str, int | str]] = []
    start = 0
    for index in range(1, len(signs) + 1):
        if index == len(signs) or signs[index] != signs[start]:
            runs.append({"sign": names[int(signs[start])], "start": start, "end": index - 1, "length": index - start})
            start = index
    return runs


def analyze_action_chain(
    chunks: list[ChunkRecord],
    controller: list[ControllerSample],
    observations: list[SavedObservation] | None = None,
) -> dict[str, Any]:
    """Summarize raw, selected, absolute, and controller layers per chunk.

    Raw action dimension 2 is a local TCP-frame translation.  It is therefore
    never interpreted as Quest/world Z; absolute waypoints supply Quest Z.
    """
    if not chunks:
        raise ValueError("chunks must be nonempty")
    if not controller:
        raise ValueError("controller must be nonempty")
    chunk_sequences = [chunk.obs_seq for chunk in chunks]
    if len(set(chunk_sequences)) != len(chunk_sequences):
        raise ValueError("duplicate chunk obs_seq values are not allowed")
    observation_by_sequence: dict[int, SavedObservation] = {}
    if observations is not None:
        if not observations:
            raise ValueError("observations must be nonempty when supplied")
        for observation in observations:
            if observation.step % EXECUTED_ACTIONS != 0:
                raise ValueError(f"saved observation step {observation.step} is not a multiple of 10")
            sequence = observation.step // EXECUTED_ACTIONS + 1
            if sequence in observation_by_sequence:
                raise ValueError(f"duplicate saved observation mapping for obs_seq {sequence}")
            observation_by_sequence[sequence] = observation
        if set(observation_by_sequence) != set(chunk_sequences):
            raise ValueError("saved observations must map one-to-one to chunk obs_seq values")
    result_chunks: list[dict[str, Any]] = []
    controller_frame_mismatch = any(sample.target_left_robot_z is not None for sample in controller)
    for chunk in chunks:
        observation = observation_by_sequence.get(chunk.obs_seq)
        selected_timestamps = chunk.action_timestamps[:EXECUTED_ACTIONS]
        earliest = float(selected_timestamps.min())
        latest = float(selected_timestamps.max())
        aligned = [sample for sample in controller if earliest <= sample.wall_time <= latest]
        raw_left_z = chunk.raw_actions[:, 2]
        raw_left_gripper = chunk.raw_actions[:, 9]
        raw_right_gripper = chunk.raw_actions[:, 19]
        result_chunks.append(
            {
                "obs_seq": chunk.obs_seq,
                "saved_step": None if observation is None else observation.step,
                "saved_timestamp": None if observation is None else observation.timestamp,
                "chunk_timestamp": chunk.timestamp,
                "prediction_field": chunk.prediction_field,
                "prediction_source": chunk.prediction_source,
                "action_timestamps": chunk.action_timestamps.tolist(),
                "raw_left_gripper": raw_left_gripper.tolist(),
                "raw_right_gripper": raw_right_gripper.tolist(),
                "raw_left_local_z": raw_left_z.tolist(),
                "selected_actions": chunk.selected_actions.tolist(),
                "absolute_left_quest_z": chunk.absolute_waypoints[:, 2].tolist(),
                "controller_actual_left_quest_z": [sample.actual_left_quest_z for sample in aligned],
                "controller_target_left_robot_z": [sample.target_left_robot_z for sample in aligned if sample.target_left_robot_z is not None],
                "left_close_count": int(np.count_nonzero(raw_left_gripper <= 0.09)),
                "right_close_count": int(np.count_nonzero(raw_right_gripper <= 0.09)),
                "raw_left_local_z_sign_runs": _sign_runs(raw_left_z),
                "cumulative_raw_left_local_z": np.cumsum(raw_left_z).tolist(),
            }
        )
    return {
        "controller_actual_pose_frame": "quest",
        "controller_target_frame_mismatch": controller_frame_mismatch,
        "controller_target_frame_note": (
            "target_pose fields are robot-frame despite pose_frame=quest; targets and actuals are not subtracted"
        ),
        "chunks": result_chunks,
    }


def write_action_chain_csv(report: dict[str, Any], path: Path) -> None:
    """Write one offline-diagnostic CSV row per chunk without dropping provenance."""
    import csv

    rows = report.get("chunks")
    if not isinstance(rows, list):
        raise ValueError("report must contain a chunks list")
    fieldnames = (
        "obs_seq", "saved_step", "saved_timestamp", "chunk_timestamp", "prediction_field", "prediction_source",
        "action_timestamps", "raw_left_gripper", "raw_right_gripper", "raw_left_local_z",
        "selected_actions", "absolute_left_quest_z", "controller_actual_left_quest_z",
        "controller_target_left_robot_z", "left_close_count", "right_close_count",
        "raw_left_local_z_sign_runs", "cumulative_raw_left_local_z",
    )
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    field: json.dumps(row[field]) if isinstance(row[field], (list, dict)) else row[field]
                    for field in fieldnames
                }
            )
