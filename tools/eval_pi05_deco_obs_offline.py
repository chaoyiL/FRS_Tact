"""Pure primitives for offline Pi0.5 evaluation against saved DECO observations."""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np
from scipy.spatial.transform import Rotation


class DecoObservation(NamedTuple):
    """One saved DECO observation with only the right-wrist RGB image retained."""

    step: int
    timestamp: float
    left_pose: np.ndarray
    right_pose: np.ndarray
    left_gripper: float
    right_gripper: float
    camera1_rgb: np.ndarray


_STEP_DIRECTORY = re.compile(r"step_(\d+)$")
_ROTATION_EPSILON = 1e-6
_SAFETY_LIMIT_NAMES = ("max_pos_delta", "max_rot_delta", "min_gripper", "max_gripper")
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
for _import_root in (_PROJECT_ROOT, _PROJECT_ROOT / "deploy_pi05" / "src"):
    if str(_import_root) not in sys.path:
        sys.path.insert(0, str(_import_root))
DEFAULT_CONFIG = _PROJECT_ROOT / "deploy_pi05" / "configs" / "deploy_pi05_right.yaml"
DEFAULT_OBS_DIR = Path("/home/typhon/vb3_robot_server/eval_obs_data/eval_obs_20260901_143909")
DEFAULT_OUTPUT_DIR = _PROJECT_ROOT / "outputs" / "pi05_right_offline" / "eval_obs_20260901_143909"
_ARCHIVE_ARRAY_NAMES = (
    "states",
    "normalized_actions",
    "right_actions",
    "wire_actions",
    "absolute_waypoints",
    "step_ids",
    "timestamps",
)
_SAFETY_DEFAULTS = {
    "max_pos_delta": 0.03,
    "max_rot_delta": 0.5,
    "min_gripper": -0.05,
    "max_gripper": 1.05,
}


def _finite_array(value: Any, *, shape: tuple[int, ...], name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != shape or not np.isfinite(array).all():
        raise ValueError(f"{name} must be finite with shape {shape}, got {array.shape}")
    return array


def _read_json_vector(path: Path, *, size: int, name: str) -> np.ndarray:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"could not read {name}: {path}") from error
    array = np.asarray(value, dtype=np.float64).reshape(-1)
    if array.shape != (size,) or not np.isfinite(array).all():
        raise ValueError(f"{name} must contain {size} finite values")
    return array


def _read_camera1_rgb(path: Path) -> np.ndarray:
    import cv2

    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError(f"could not decode camera1 RGB image: {path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.size == 0:
        raise ValueError(f"camera1 RGB image must be nonempty HWC RGB: {path}")
    return np.ascontiguousarray(rgb)


def load_deco_observations(path: Path | str) -> list[DecoObservation]:
    """Load numbered DECO snapshots in numeric order with camera1 decoded as RGB."""

    root = Path(path)
    if not root.is_dir():
        raise ValueError(f"DECO observation directory does not exist: {root}")
    steps: list[tuple[int, Path]] = []
    for child in root.iterdir():
        match = _STEP_DIRECTORY.fullmatch(child.name)
        if child.is_dir() and match is not None:
            steps.append((int(match.group(1)), child))
    if not steps:
        raise ValueError(f"no step_###### directories found in {root}")

    observations: list[DecoObservation] = []
    for step, step_dir in sorted(steps):
        left_pose = np.concatenate(
            (
                _read_json_vector(step_dir / "robot0_eef_pos.json", size=3, name="robot0 position"),
                _read_json_vector(
                    step_dir / "robot0_eef_rot_axis_angle.json", size=3, name="robot0 rotation"
                ),
            )
        )
        right_pose = np.concatenate(
            (
                _read_json_vector(step_dir / "robot1_eef_pos.json", size=3, name="robot1 position"),
                _read_json_vector(
                    step_dir / "robot1_eef_rot_axis_angle.json", size=3, name="robot1 rotation"
                ),
            )
        )
        observations.append(
            DecoObservation(
                step=step,
                timestamp=float(_read_json_vector(step_dir / "timestamp.json", size=1, name="timestamp")[0]),
                left_pose=left_pose.astype(np.float32),
                right_pose=right_pose.astype(np.float32),
                left_gripper=float(
                    _read_json_vector(
                        step_dir / "robot0_gripper_width.json", size=1, name="robot0 gripper"
                    )[0]
                ),
                right_gripper=float(
                    _read_json_vector(
                        step_dir / "robot1_gripper_width.json", size=1, name="robot1 gripper"
                    )[0]
                ),
                camera1_rgb=_read_camera1_rgb(step_dir / "camera1_rgb.jpg"),
            )
        )
    return observations


def _pose_matrix(pose: Any, *, name: str) -> np.ndarray:
    pose_array = _finite_array(pose, shape=(6,), name=name)
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = Rotation.from_rotvec(pose_array[3:]).as_matrix()
    matrix[:3, 3] = pose_array[:3]
    return matrix


def _matrix_pose(matrix: np.ndarray) -> np.ndarray:
    return np.concatenate((matrix[:3, 3], Rotation.from_matrix(matrix[:3, :3]).as_rotvec()))


def _start_pose_pair(start_poses: Sequence[Any] | Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    if isinstance(start_poses, Mapping):
        try:
            return (
                _finite_array(start_poses["left"], shape=(6,), name="left start pose"),
                _finite_array(start_poses["right"], shape=(6,), name="right start pose"),
            )
        except KeyError as error:
            raise ValueError("start poses mapping must contain left and right") from error
    if len(start_poses) != 2:
        raise ValueError("start poses must contain left and right 6D poses")
    return (
        _finite_array(start_poses[0], shape=(6,), name="left start pose"),
        _finite_array(start_poses[1], shape=(6,), name="right start pose"),
    )


def build_server_state(
    start_poses: Sequence[Any] | Mapping[str, Any], observation: DecoObservation
) -> np.ndarray:
    """Reconstruct the production 20D bimanual relative-start server state."""

    required_fields = ("left_pose", "right_pose", "left_gripper", "right_gripper")
    if any(not hasattr(observation, field) for field in required_fields):
        raise ValueError("observation must provide bimanual poses and gripper widths")
    left_start, right_start = _start_pose_pair(start_poses)
    left = _pose_matrix(observation.left_pose, name="left pose")
    right = _pose_matrix(observation.right_pose, name="right pose")
    grippers = np.asarray((observation.left_gripper, observation.right_gripper), dtype=np.float64)
    if not np.isfinite(grippers).all():
        raise ValueError("observation gripper widths must be finite")
    state = np.concatenate(
        (
            _matrix_pose(np.linalg.inv(_pose_matrix(left_start, name="left start pose")) @ left),
            grippers[:1],
            _matrix_pose(np.linalg.inv(_pose_matrix(right_start, name="right start pose")) @ right),
            grippers[1:],
            _matrix_pose(np.linalg.inv(right) @ left),
        )
    ).astype(np.float32)
    if state.shape != (20,) or not np.isfinite(state).all():
        raise ValueError("reconstructed server state must be finite with shape (20,)")
    return state


def _actions_array(actions: Any, *, widths: tuple[int, ...]) -> np.ndarray:
    array = np.asarray(actions)
    expected = " or ".join(f"(H,{width})" for width in widths)
    if array.ndim == 2 and array.shape[0] == 0:
        raise ValueError(f"actions must be nonempty with shape {expected}")
    if array.ndim != 2 or array.shape[1] not in widths:
        raise ValueError(f"actions must have shape {expected}, got {array.shape}")
    if not np.issubdtype(array.dtype, np.floating):
        raise ValueError(f"actions dtype {array.dtype} is not floating")
    if not np.isfinite(array).all():
        raise ValueError(f"actions must be finite with shape {expected}, got {array.shape}")
    if np.any(np.abs(array) > np.finfo(np.float32).max):
        raise ValueError("actions contain values outside the finite float32 range")
    return array


def _rotation_matrix_columns(rotation_6d: Any) -> np.ndarray:
    values = _finite_array(rotation_6d, shape=(6,), name="rotation 6D")
    first = values[:3]
    second = values[3:]
    first_norm = float(np.linalg.norm(first))
    second_norm = float(np.linalg.norm(second))
    if first_norm <= _ROTATION_EPSILON or second_norm <= _ROTATION_EPSILON:
        raise ValueError("rotation 6D columns must be nonzero")
    first = first / first_norm
    second = second / second_norm
    second = second - np.dot(first, second) * first
    orthogonal_norm = float(np.linalg.norm(second))
    if orthogonal_norm <= _ROTATION_EPSILON:
        raise ValueError("rotation 6D columns must not be collinear")
    second = second / orthogonal_norm
    return np.stack((first, second, np.cross(first, second)), axis=-1)


def _server_rotation_angle(rotation_6d: Any) -> float:
    """Match the robot server's normalized-column trace/arccos rotation check."""

    matrix = _rotation_matrix_columns(rotation_6d)
    trace = float(np.trace(matrix))
    return float(np.arccos(np.clip((trace - 1.0) / 2.0, -1.0, 1.0)))


def actions_to_absolute_waypoints(actions: Any, base_pose: Any) -> np.ndarray:
    """Compose right-arm local 10D actions into absolute xyz+axis-angle+gripper waypoints."""

    action_array = _actions_array(actions, widths=(10,))
    current_to_base = _pose_matrix(base_pose, name="base pose")
    waypoints = np.empty((len(action_array), 7), dtype=np.float32)
    for index, action in enumerate(action_array):
        next_to_current = np.eye(4, dtype=np.float64)
        next_to_current[:3, :3] = _rotation_matrix_columns(action[3:9])
        next_to_current[:3, 3] = action[:3]
        current_to_base = current_to_base @ next_to_current
        waypoints[index, :6] = _matrix_pose(current_to_base)
        waypoints[index, 6] = action[9]
    return waypoints


def _safety_limits(limits: Mapping[str, Any]) -> tuple[float, float, float, float]:
    if not isinstance(limits, Mapping):
        raise ValueError("safety limits must be a mapping")
    try:
        parsed = tuple(float(limits[name]) for name in _SAFETY_LIMIT_NAMES)
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"safety limits must contain {_SAFETY_LIMIT_NAMES}") from error
    if not np.isfinite(parsed).all():
        raise ValueError("safety limits must be finite")
    max_position, max_rotation, min_gripper, max_gripper = parsed
    if max_position <= 0.0 or max_rotation <= 0.0 or min_gripper > max_gripper:
        raise ValueError("safety limits must be positive with ordered gripper bounds")
    return parsed


def action_safety_metrics(actions: Any, limits: Mapping[str, Any]) -> dict[str, object]:
    """Return server-equivalent safety maxima and every raw-action limit violation."""

    action_array = _actions_array(actions, widths=(10, 20))
    max_position, max_rotation, min_gripper, max_gripper = _safety_limits(limits)
    raw_per_robot = action_array.reshape(len(action_array), -1, 10)
    with np.errstate(over="ignore", invalid="ignore"):
        server_actions = action_array.astype(np.float32).astype(np.float64)
    per_robot = server_actions.reshape(len(server_actions), -1, 10)
    translation = np.linalg.norm(per_robot[..., :3], axis=-1)
    rotation = np.empty_like(translation)
    violations: list[dict[str, object]] = []
    for step, robot in np.ndindex(translation.shape):
        rotation[step, robot] = _server_rotation_angle(per_robot[step, robot, 3:9])
        if translation[step, robot] > max_position:
            violations.append(
                {
                    "kind": "translation_delta",
                    "step": int(step),
                    "robot_index": int(robot),
                    "value": float(translation[step, robot]),
                    "limit": max_position,
                }
            )
        if rotation[step, robot] > max_rotation:
            violations.append(
                {
                    "kind": "rotation_delta",
                    "step": int(step),
                    "robot_index": int(robot),
                    "value": float(rotation[step, robot]),
                    "limit": max_rotation,
                }
            )
    grippers = raw_per_robot[..., 9]
    for step, robot in np.ndindex(grippers.shape):
        value = float(grippers[step, robot])
        if value < min_gripper or value > max_gripper:
            violations.append(
                {
                    "kind": "gripper",
                    "step": int(step),
                    "robot_index": int(robot),
                    "value": value,
                    "min": min_gripper,
                    "max": max_gripper,
                }
            )
    return {
        "safe": not violations,
        "max_translation_delta": float(np.max(translation)),
        "max_rotation_delta": float(np.max(rotation)),
        "min_gripper": float(np.min(grippers)),
        "max_gripper": float(np.max(grippers)),
        "violations": violations,
    }


def json_safe(value: Any) -> Any:
    """Convert NumPy-rich metadata into JSON values without allowing NaN or Inf."""

    if isinstance(value, Mapping):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not np.isfinite(value):
        raise ValueError("JSON output cannot contain NaN or Inf")
    return value


def _validate_prediction_arrays(arrays: Mapping[str, Any]) -> dict[str, np.ndarray]:
    missing = [name for name in _ARCHIVE_ARRAY_NAMES if name not in arrays]
    if missing:
        raise ValueError(f"prediction archive missing required arrays: {missing}")
    result = {name: np.asarray(arrays[name]) for name in _ARCHIVE_ARRAY_NAMES}
    states = result["states"]
    if states.ndim != 2 or states.shape[1] != 20 or len(states) == 0:
        raise ValueError("states must have nonempty shape (N,20)")
    chunks = len(states)
    horizon: int | None = None
    for name, width in (
        ("normalized_actions", 10),
        ("right_actions", 10),
        ("wire_actions", 20),
        ("absolute_waypoints", 7),
    ):
        array = result[name]
        if array.ndim != 3 or array.shape[0] != chunks or array.shape[2] != width:
            raise ValueError(f"{name} must have shape (N,H,{width}) matching states")
        if horizon is None:
            horizon = int(array.shape[1])
        elif array.shape[1] != horizon:
            raise ValueError("prediction action arrays must share one horizon")
    if horizon is None or horizon <= 0:
        raise ValueError("prediction action horizon must be positive")
    for name in ("step_ids", "timestamps"):
        array = result[name]
        if array.shape != (chunks,):
            raise ValueError(f"{name} must have shape (N,)")
    if not all(np.issubdtype(array.dtype, np.number) and np.isfinite(array).all() for array in result.values()):
        raise ValueError("prediction arrays must be numeric and finite")
    return result


def write_prediction_archive(path: Path | str, arrays: Mapping[str, Any], metadata: Mapping[str, Any]) -> Path:
    """Write a pickle-free partial or final prediction archive."""

    checked = _validate_prediction_arrays(arrays)
    document = json_safe(dict(metadata))
    if not isinstance(document, dict):
        raise ValueError("archive metadata must be a mapping")
    completed = document.get("completed_chunks")
    if isinstance(completed, bool) or not isinstance(completed, int) or not 0 <= completed <= len(checked["states"]):
        raise ValueError("metadata.completed_chunks must be an in-range integer")
    if type(document.get("complete")) is not bool:
        raise ValueError("metadata.complete must be a boolean")
    if int(document.get("action_horizon", -1)) != checked["right_actions"].shape[1]:
        raise ValueError("metadata.action_horizon must match prediction arrays")
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        **checked,
        metadata_json=np.asarray(json.dumps(document, allow_nan=False, sort_keys=True)),
    )
    return output


def load_prediction_archive(path: Path | str) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Load and validate a pickle-free offline-evaluation archive."""

    source = Path(path)
    try:
        with np.load(source, allow_pickle=False) as archive:
            arrays = {name: archive[name].copy() for name in _ARCHIVE_ARRAY_NAMES if name in archive}
            if "metadata_json" not in archive:
                raise ValueError("prediction archive is missing metadata_json")
            raw_metadata = archive["metadata_json"]
            if raw_metadata.shape != () or raw_metadata.dtype.kind not in {"U", "S"}:
                raise ValueError("metadata_json must be a scalar string")
            metadata = json.loads(str(raw_metadata.item()))
    except (OSError, json.JSONDecodeError, ValueError) as error:
        if isinstance(error, ValueError) and str(error).startswith("prediction archive"):
            raise
        raise ValueError(f"could not read prediction archive: {source}") from error
    checked = _validate_prediction_arrays(arrays)
    if not isinstance(metadata, dict):
        raise ValueError("archive metadata must be a mapping")
    completed = metadata.get("completed_chunks")
    if isinstance(completed, bool) or not isinstance(completed, int) or not 0 <= completed <= len(checked["states"]):
        raise ValueError("metadata.completed_chunks must be an in-range integer")
    if type(metadata.get("complete")) is not bool:
        raise ValueError("metadata.complete must be a boolean")
    if int(metadata.get("action_horizon", -1)) != checked["right_actions"].shape[1]:
        raise ValueError("metadata.action_horizon must match prediction arrays")
    return checked, metadata


def interpolate_on_common_seconds(
    source_time: Any, source_values: Any, target_time: Any
) -> tuple[np.ndarray, np.ndarray]:
    """Linearly interpolate source values at target seconds within their shared interval."""

    source = _finite_array(source_time, shape=(np.asarray(source_time).size,), name="source time")
    target = _finite_array(target_time, shape=(np.asarray(target_time).size,), name="target time")
    values = np.asarray(source_values, dtype=np.float64)
    if source.ndim != 1 or target.ndim != 1 or values.ndim < 1 or values.shape[0] != len(source):
        raise ValueError("time interpolation inputs must align on a nonempty first dimension")
    if len(source) < 2 or len(target) == 0 or not np.isfinite(values).all():
        raise ValueError("time interpolation inputs must be finite and nonempty")
    if np.any(np.diff(source) <= 0.0) or np.any(np.diff(target) <= 0.0):
        raise ValueError("time interpolation requires strictly increasing seconds")
    mask = (target >= source[0]) & (target <= source[-1])
    common = target[mask]
    if len(common) == 0:
        raise ValueError("time series have no common duration")
    flat = values.reshape(len(source), -1)
    interpolated = np.stack(
        [np.interp(common, source, flat[:, column]) for column in range(flat.shape[1])], axis=1
    )
    return common, interpolated.reshape((len(common), *values.shape[1:]))


def _load_deco_trace(path: Path | str) -> tuple[np.ndarray, np.ndarray]:
    times: list[float] = []
    right_waypoints: list[np.ndarray] = []
    with Path(path).open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                record_times = np.asarray(record["action_timestamps"], dtype=np.float64)
                waypoints = np.asarray(record["absolute_waypoints"], dtype=np.float64)
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
                raise ValueError(f"invalid DECO trace row {line_number}") from error
            if waypoints.ndim != 2 or waypoints.shape[0] != len(record_times) or waypoints.shape[1] not in {7, 14}:
                raise ValueError(f"invalid DECO trace waypoints at row {line_number}")
            if not np.isfinite(record_times).all() or not np.isfinite(waypoints).all():
                raise ValueError(f"non-finite DECO trace row {line_number}")
            times.extend(record_times.tolist())
            right_waypoints.extend((waypoints[:, -7:]).copy())
    result_times = np.asarray(times, dtype=np.float64)
    result_waypoints = np.asarray(right_waypoints, dtype=np.float64)
    if len(result_times) < 2 or np.any(np.diff(result_times) <= 0.0):
        raise ValueError("DECO trace timestamps must be strictly increasing")
    return result_times, result_waypoints


def _deco_comparison(
    arrays: Mapping[str, np.ndarray], metadata: Mapping[str, Any], trace: Path | str
) -> dict[str, Any]:
    reference_time, reference_waypoints = _load_deco_trace(trace)
    control_hz = float(metadata.get("control_hz", 10.0))
    if not np.isfinite(control_hz) or control_hz <= 0.0:
        raise ValueError("metadata.control_hz must be positive and finite")
    horizon = arrays["absolute_waypoints"].shape[1]
    differences: list[np.ndarray] = []
    endpoint_displacements: list[float] = []
    direction_cosines: list[float] = []
    per_chunk: list[dict[str, Any]] = []
    for chunk_index, (timestamp, waypoints) in enumerate(
        zip(arrays["timestamps"], arrays["absolute_waypoints"], strict=True)
    ):
        predicted_time = float(timestamp) + np.arange(horizon) / control_hz
        try:
            common_time, reference_xyz = interpolate_on_common_seconds(
                reference_time, reference_waypoints[:, :3], predicted_time
            )
        except ValueError as error:
            if str(error) == "time series have no common duration":
                continue
            raise
        mask = np.isin(predicted_time, common_time)
        predicted_xyz = waypoints[mask, :3].astype(np.float64)
        if len(predicted_xyz) != len(reference_xyz) or len(predicted_xyz) < 2:
            continue
        difference = predicted_xyz - reference_xyz
        predicted_direction = predicted_xyz[-1] - predicted_xyz[0]
        reference_direction = reference_xyz[-1] - reference_xyz[0]
        denominator = float(np.linalg.norm(predicted_direction) * np.linalg.norm(reference_direction))
        cosine = float(np.dot(predicted_direction, reference_direction) / denominator) if denominator else 0.0
        differences.append(difference)
        endpoint_displacements.append(float(np.linalg.norm(difference[-1])))
        direction_cosines.append(cosine)
        per_chunk.append({"chunk_index": chunk_index, "common_samples": int(len(predicted_xyz))})
    if not differences:
        raise ValueError("DECO comparison needs at least two common predicted seconds")
    difference = np.concatenate(differences, axis=0)
    return {
        "position_rmse_m": float(np.sqrt(np.mean(np.square(difference)))),
        "endpoint_displacement_m": float(np.mean(endpoint_displacements)),
        "endpoint_direction_cosine": float(np.mean(direction_cosines)),
        "common_samples": int(len(difference)),
        "chunks_compared": int(len(per_chunk)),
        "per_chunk": per_chunk,
    }


def _write_trajectory_csv(output: Path, arrays: Mapping[str, np.ndarray]) -> None:
    columns = (
        "chunk_index", "step_id", "timestamp", "action_index", "relative_x", "relative_y", "relative_z",
        "rotation_6d_0", "rotation_6d_1", "rotation_6d_2", "rotation_6d_3", "rotation_6d_4", "rotation_6d_5",
        "absolute_x", "absolute_y", "absolute_z", "absolute_rx", "absolute_ry", "absolute_rz", "gripper", "safe",
    )
    with output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        for chunk_index, (right, wire, waypoint) in enumerate(zip(arrays["right_actions"], arrays["wire_actions"], arrays["absolute_waypoints"], strict=True)):
            safety = action_safety_metrics(wire, _SAFETY_DEFAULTS)
            for action_index, (action, absolute) in enumerate(zip(right, waypoint, strict=True)):
                row = {
                    "chunk_index": chunk_index,
                    "step_id": int(arrays["step_ids"][chunk_index]),
                    "timestamp": float(arrays["timestamps"][chunk_index]),
                    "action_index": action_index,
                    "relative_x": float(action[0]), "relative_y": float(action[1]), "relative_z": float(action[2]),
                    "absolute_x": float(absolute[0]), "absolute_y": float(absolute[1]), "absolute_z": float(absolute[2]),
                    "absolute_rx": float(absolute[3]), "absolute_ry": float(absolute[4]), "absolute_rz": float(absolute[5]),
                    "gripper": float(absolute[6]), "safe": safety["safe"],
                }
                row.update({f"rotation_6d_{index}": float(action[3 + index]) for index in range(6)})
                writer.writerow(row)


def _write_plots(output: Path, arrays: Mapping[str, np.ndarray], reference: tuple[np.ndarray, np.ndarray] | None) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    start = arrays["absolute_waypoints"][0]
    figure = plt.figure(figsize=(12, 8), constrained_layout=True)
    axis_3d = figure.add_subplot(2, 2, 1, projection="3d")
    axis_3d.plot(start[:, 0], start[:, 1], start[:, 2], label="Pi0.5")
    if reference is not None:
        axis_3d.plot(reference[1][:, 0], reference[1][:, 1], reference[1][:, 2], label="DECO")
    axis_3d.legend()
    axis_3d.set_title("right start chunk")
    axis_xyz = figure.add_subplot(2, 2, 2)
    axis_xyz.plot(start[:, :3])
    axis_xyz.set_title("XYZ")
    axis_rotation = figure.add_subplot(2, 2, 3)
    axis_rotation.plot(start[:, 3:6])
    axis_rotation.set_title("axis-angle")
    axis_gripper = figure.add_subplot(2, 2, 4)
    axis_gripper.plot(start[:, 6])
    axis_gripper.set_title("gripper")
    figure.savefig(output / "right_start_chunk.png", dpi=140)
    plt.close(figure)

    figure = plt.figure(figsize=(8, 6), constrained_layout=True)
    axis = figure.add_subplot(1, 1, 1, projection="3d")
    for chunk_index, waypoints in enumerate(arrays["absolute_waypoints"]):
        axis.plot(waypoints[:, 0], waypoints[:, 1], waypoints[:, 2], label=f"chunk {chunk_index}")
    axis.set_title("independent Pi0.5 right-hand XYZ chunks")
    figure.savefig(output / "right_all_chunks.png", dpi=140)
    plt.close(figure)


def run_report(artifact: Path | str, output_dir: Path | str, *, deco_trace: Path | str | None = None) -> dict[str, Any]:
    """Render CSV, JSON, and figures from a complete offline prediction archive."""

    arrays, metadata = load_prediction_archive(artifact)
    if metadata["complete"] is not True or metadata["completed_chunks"] != len(arrays["states"]):
        raise ValueError("report requires a complete prediction archive")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    safety = [action_safety_metrics(chunk, _SAFETY_DEFAULTS) for chunk in arrays["wire_actions"]]
    summary: dict[str, Any] = {
        "chunks": int(len(arrays["states"])),
        "horizon": int(arrays["right_actions"].shape[1]),
        "safety": {"safe": all(item["safe"] for item in safety), "per_chunk": safety},
        "provenance": metadata,
        "assumption": "first saved observation pose is the episode_start_pose proxy",
    }
    reference: tuple[np.ndarray, np.ndarray] | None = None
    if deco_trace is not None:
        reference = _load_deco_trace(deco_trace)
        summary["deco_comparison"] = _deco_comparison(arrays, metadata, deco_trace)
    _write_trajectory_csv(output / "trajectory.csv", arrays)
    _write_plots(output, arrays, reference)
    with (output / "summary.json").open("w", encoding="utf-8") as file:
        json.dump(json_safe(summary), file, allow_nan=False, indent=2, sort_keys=True)
    return json_safe(summary)


def _require_cuda() -> tuple[Any, str, tuple[Any, ...]]:
    import jax

    backend = str(jax.default_backend())
    devices = tuple(jax.devices())
    gpu_labels = {"gpu", "cuda"}
    if backend not in gpu_labels or not any(
        str(getattr(device, "platform", "")) in gpu_labels for device in devices
    ):
        raise RuntimeError(f"CUDA JAX backend is required, got backend={backend!r}, devices={devices!r}")
    return jax, backend, devices


def infer_deco_observations(
    config_path: Path | str,
    obs_dir: Path | str,
    output_dir: Path | str,
    *,
    require_cuda: bool = True,
) -> Path:
    """Run sequential right-hand Pi0.5 inference and checkpoint its archive per snapshot."""

    from deploy_pi05.deployment import load_deployment_config, make_policy_config, section

    config_file = Path(config_path).expanduser().resolve()
    config = load_deployment_config(config_file, "pi05")
    if require_cuda:
        _, backend, devices = _require_cuda()
    else:
        import jax

        backend, devices = str(jax.default_backend()), tuple(jax.devices())
    model = section(config, "model")
    if str(model.get("state_action_profile")) != "single-right-arm-7x10":
        raise ValueError("offline DECO evaluator requires the single-right-arm-7x10 profile")
    policy_config = make_policy_config(config, config_file)
    from deploy_pi05.policy import Pi05RemotePolicy
    from deploy_pi05.right_arm_adapter import expand_right_action, project_right_observation

    observations = load_deco_observations(obs_dir)
    start_poses = (observations[0].left_pose, observations[0].right_pose)
    policy = Pi05RemotePolicy(policy_config)
    prompt = str(section(config, "observation")["language_prompt"])
    seed = int(config.get("seed", 0))
    num_steps = int(config.get("num_steps", 10))
    warmup_runs = int(section(config, "runtime").get("warmup_runs", 1))
    if warmup_runs < 0:
        raise ValueError("runtime.warmup_runs must be nonnegative")
    first_state = build_server_state(start_poses, observations[0])
    first_raw = {"observation.state": first_state, "observation.images.camera1": observations[0].camera1_rgb}
    first_model = project_right_observation(first_raw)
    for _ in range(warmup_runs):
        policy.predict_action_chunk(first_model, prompt, seed=seed, num_steps=num_steps)

    archive = Path(output_dir) / "predictions.npz"
    collected: dict[str, list[np.ndarray]] = {name: [] for name in _ARCHIVE_ARRAY_NAMES}
    metadata: dict[str, Any] = {
        "complete": False,
        "completed_chunks": 0,
        "action_horizon": int(policy_config.action_horizon),
        "control_hz": float(section(config, "control")["control_frequency"]),
        "config_path": str(config_file),
        "obs_dir": str(Path(obs_dir).resolve()),
        "seed": seed,
        "num_steps": num_steps,
        "warmup_runs": warmup_runs,
        "jax_backend": backend,
        "jax_devices": [str(device) for device in devices],
        "episode_start_pose": "first saved snapshot proxy",
    }
    for observation in observations:
        state = build_server_state(start_poses, observation)
        raw = {"observation.state": state, "observation.images.camera1": observation.camera1_rgb}
        normalized = np.asarray(
            policy.predict_action_chunk(project_right_observation(raw), prompt, seed=seed, num_steps=num_steps)[0],
            dtype=np.float32,
        )
        right = np.asarray(policy.unnormalize_actions(normalized), dtype=np.float32)
        wire = np.asarray(expand_right_action(right, raw), dtype=np.float32)
        waypoints = actions_to_absolute_waypoints(right, observation.right_pose)
        if right.shape != (policy_config.action_horizon, 10) or normalized.shape != right.shape:
            raise ValueError("Pi0.5 right-arm output must have shape (H,10)")
        if wire.shape != (policy_config.action_horizon, 20):
            raise ValueError("expanded wire action must have shape (H,20)")
        action_safety_metrics(wire, _SAFETY_DEFAULTS)
        values = {
            "states": state,
            "normalized_actions": normalized,
            "right_actions": right,
            "wire_actions": wire,
            "absolute_waypoints": waypoints,
            "step_ids": np.asarray(observation.step, dtype=np.int64),
            "timestamps": np.asarray(observation.timestamp, dtype=np.float64),
        }
        for name, value in values.items():
            collected[name].append(value)
        metadata["completed_chunks"] = len(collected["states"])
        partial = {name: np.stack(values_list) for name, values_list in collected.items()}
        write_prediction_archive(archive, partial, metadata)
    metadata["complete"] = True
    final = {name: np.stack(values_list) for name, values_list in collected.items()}
    write_prediction_archive(archive, final, metadata)
    return archive


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    infer = commands.add_parser("infer", help="run offline Pi0.5 inference without a robot connection")
    infer.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    infer.add_argument("--obs-dir", type=Path, default=DEFAULT_OBS_DIR)
    infer.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    infer.add_argument("--require-cuda", action=argparse.BooleanOptionalAction, default=True)
    report = commands.add_parser("report", help="render artifacts from a saved prediction archive")
    report.add_argument("--artifact", type=Path, required=True)
    report.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    report.add_argument("--deco-trace", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    if args.command == "infer":
        infer_deco_observations(args.config, args.obs_dir, args.output_dir, require_cuda=args.require_cuda)
    else:
        run_report(args.artifact, args.output_dir, deco_trace=args.deco_trace)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
