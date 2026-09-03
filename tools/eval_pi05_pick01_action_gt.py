"""Robot-free Pi0.5 action evaluation for pick_01 episodes and saved observations.

The command deliberately only consumes saved data.  It never imports a robot
bridge or contacts a service; the Pi0.5 policy is imported lazily by ``infer``.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
import os
import sys
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np
from scipy.spatial.transform import Rotation


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "deploy_pi05" / "configs" / "deploy_pi05.yaml"
DEFAULT_OUTPUT_DIR = ROOT / "outputs" / "pi05_pick01_action_eval_20260902_145255"
DEFAULT_REAL_OBS_DIR = Path("/home/typhon/vb3_robot_server/eval_obs_data/eval_obs_20260902_145255_364433")
DEFAULT_NORM_STATS = ROOT / "checkpoints" / "model" / "pi05_task1_0902_6k" / "assets" / "pick_0102" / "norm_stats.json"
ACTION_DIM = 20
HORIZON = 50
WINDOWS = (1, 10, 20, 50)
_ROT_EPS = 1e-12


class RealObservation(NamedTuple):
    step: int
    timestamp: float
    left_pose: np.ndarray
    right_pose: np.ndarray
    left_gripper: float
    right_gripper: float
    camera0_rgb: np.ndarray
    camera1_rgb: np.ndarray


def _finite(value: Any, *, shape: tuple[int, ...] | None = None, name: str = "value") -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if shape is not None and array.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must be finite")
    return array


def _pose_matrix(pose: Any) -> np.ndarray:
    """Return the xyz + rotation-vector homogeneous transform without server imports."""
    pose = _finite(pose, shape=(6,), name="pose")
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = Rotation.from_rotvec(pose[3:]).as_matrix()
    matrix[:3, 3] = pose[:3]
    return matrix


def _matrix_pose(matrix: Any) -> np.ndarray:
    matrix = _finite(matrix, shape=(4, 4), name="pose matrix")
    return np.concatenate((matrix[:3, 3], Rotation.from_matrix(matrix[:3, :3]).as_rotvec()))


def build_real_state_proxy(start_poses: Sequence[Any] | Mapping[str, Any], observation: Any) -> np.ndarray:
    """Build the deployed 20D state using the first saved pose as a start proxy."""
    if isinstance(start_poses, Mapping):
        left_start, right_start = start_poses["left"], start_poses["right"]
    else:
        if len(start_poses) != 2:
            raise ValueError("episode_start_pose proxy must contain left and right 6D poses")
        left_start, right_start = start_poses
    def field(name: str) -> Any:
        return observation[name] if isinstance(observation, Mapping) else getattr(observation, name)
    left = _pose_matrix(field("left_pose"))
    right = _pose_matrix(field("right_pose"))
    left_gripper, right_gripper = float(field("left_gripper")), float(field("right_gripper"))
    if not math.isfinite(left_gripper) or not math.isfinite(right_gripper):
        raise ValueError("gripper widths must be finite")
    state = np.concatenate((
        _matrix_pose(np.linalg.inv(_pose_matrix(left_start)) @ left), [left_gripper],
        _matrix_pose(np.linalg.inv(_pose_matrix(right_start)) @ right), [right_gripper],
        _matrix_pose(np.linalg.inv(right) @ left),
    )).astype(np.float32)
    if state.shape != (20,) or not np.isfinite(state).all():
        raise ValueError("real-state proxy must be finite 20D")
    return state


def read_saved_rgb(path: Path | str) -> np.ndarray:
    """Decode a new saver image as Pillow RGB; no BGR/RGB channel swap is applied."""
    from PIL import Image
    try:
        with Image.open(path) as image:
            rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    except OSError as error:
        raise ValueError(f"could not decode RGB image: {path}") from error
    if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.size == 0:
        raise ValueError(f"saved image must be nonempty HWC RGB: {path}")
    return np.ascontiguousarray(rgb)


def _json_vector(path: Path, size: int) -> np.ndarray:
    try:
        values = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"could not read {path}") from error
    return _finite(np.asarray(values).reshape(-1), shape=(size,), name=str(path))


def load_real_observations(root: Path | str) -> list[RealObservation]:
    root = Path(root)
    steps = sorted((child for child in root.glob("step_*") if child.is_dir()), key=lambda item: int(item.name.split("_")[-1]))
    if not steps:
        raise ValueError(f"no step_###### directories found in {root}")
    observations = []
    for directory in steps:
        pose0 = np.concatenate((_json_vector(directory / "robot0_eef_pos.json", 3), _json_vector(directory / "robot0_eef_rot_axis_angle.json", 3)))
        pose1 = np.concatenate((_json_vector(directory / "robot1_eef_pos.json", 3), _json_vector(directory / "robot1_eef_rot_axis_angle.json", 3)))
        observations.append(RealObservation(
            step=int(directory.name.split("_")[-1]), timestamp=float(_json_vector(directory / "timestamp.json", 1)[0]),
            left_pose=pose0.astype(np.float32), right_pose=pose1.astype(np.float32),
            left_gripper=float(_json_vector(directory / "robot0_gripper_width.json", 1)[0]),
            right_gripper=float(_json_vector(directory / "robot1_gripper_width.json", 1)[0]),
            camera0_rgb=read_saved_rgb(directory / "camera0_rgb.jpg"), camera1_rgb=read_saved_rgb(directory / "camera1_rgb.jpg"),
        ))
    return observations


def build_gt_chunk(actions: Any, horizon: int = HORIZON, anchor: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Return ``actions[anchor:anchor+horizon]`` and its terminal/padding mask."""
    action_array = _finite(actions, name="actions")
    if action_array.ndim != 2 or action_array.shape[1] != ACTION_DIM:
        raise ValueError(f"actions must have shape (T,{ACTION_DIM}), got {action_array.shape}")
    if horizon <= 0 or anchor < 0:
        raise ValueError("horizon must be positive and anchor must be nonnegative")
    gt = np.zeros((horizon, ACTION_DIM), dtype=np.float32)
    valid = np.zeros(horizon, dtype=bool)
    terminal_seen = False
    for lead in range(horizon):
        index = anchor + lead
        if index >= len(action_array):
            break
        gt[lead] = action_array[index]
        terminal = bool(np.all(action_array[index] == 0.0))
        if not terminal_seen and not terminal:
            valid[lead] = True
        terminal_seen |= terminal
    return gt, valid


def rotation6d_to_matrix(rotation6d: Any) -> np.ndarray:
    rotation = _finite(rotation6d, name="rotation6d")
    if rotation.shape[-1] != 6:
        raise ValueError("rotation6d last dimension must be 6")
    first, second = rotation[..., :3], rotation[..., 3:]
    first_norm = np.linalg.norm(first, axis=-1, keepdims=True)
    basis_x = first / np.clip(first_norm, _ROT_EPS, None)
    residual = second - np.sum(second * basis_x, axis=-1, keepdims=True) * basis_x
    residual_norm = np.linalg.norm(residual, axis=-1, keepdims=True)
    if np.any(first_norm < _ROT_EPS) or np.any(residual_norm < _ROT_EPS):
        raise ValueError("rotation6d axes must be nonzero and non-collinear")
    basis_y = residual / residual_norm
    return np.stack((basis_x, basis_y, np.cross(basis_x, basis_y)), axis=-1)


def rotation_geodesic_deg(prediction: Any, target: Any) -> float:
    pred, gt = rotation6d_to_matrix(prediction), rotation6d_to_matrix(target)
    trace = np.trace(pred @ np.swapaxes(gt, -1, -2), axis1=-2, axis2=-1)
    return float(np.rad2deg(np.arccos(np.clip((trace - 1.0) / 2.0, -1.0, 1.0))).mean())


def _window_metrics(pred: np.ndarray, gt: np.ndarray, valid: np.ndarray) -> dict[str, Any]:
    count = int(valid.sum())
    if count == 0:
        return {key: None for key in ("overall_mae", "overall_rmse", "left_translation_l2_mm", "right_translation_l2_mm", "left_rotation_geodesic_deg", "right_rotation_geodesic_deg", "left_gripper_mae_mm", "right_gripper_mae_mm")} | {"valid_steps": 0}
    masked_pred, masked_gt = pred[valid], gt[valid]
    diff = masked_pred - masked_gt
    return {
        "valid_steps": count,
        "overall_mae": float(np.abs(diff).mean()), "overall_rmse": float(np.sqrt(np.square(diff).mean())),
        "left_translation_l2_mm": float(np.linalg.norm(diff[:, :3], axis=-1).mean() * 1000.0),
        "right_translation_l2_mm": float(np.linalg.norm(diff[:, 10:13], axis=-1).mean() * 1000.0),
        "left_rotation_geodesic_deg": rotation_geodesic_deg(masked_pred[:, 3:9], masked_gt[:, 3:9]),
        "right_rotation_geodesic_deg": rotation_geodesic_deg(masked_pred[:, 13:19], masked_gt[:, 13:19]),
        "left_gripper_mae_mm": float(np.abs(diff[:, 9]).mean() * 1000.0),
        "right_gripper_mae_mm": float(np.abs(diff[:, 19]).mean() * 1000.0),
    }


def compute_native_metrics(pred: Any, gt: Any, valid: Any) -> dict[str, Any]:
    pred, gt = _finite(pred, name="pred"), _finite(gt, name="gt")
    valid = np.asarray(valid, dtype=bool)
    if pred.ndim != 3 or pred.shape != gt.shape or pred.shape[2] != ACTION_DIM or valid.shape != pred.shape[:2]:
        raise ValueError("pred, gt, and valid must have shapes (N,H,20), (N,H,20), (N,H)")
    return {"windows": {str(window): _window_metrics(pred[:, :window], gt[:, :window], valid[:, :window]) for window in WINDOWS}}


def build_noop_chunks(dataset_states: Any, horizon: int = HORIZON) -> np.ndarray:
    """Build a fixed-pose baseline from each anchor's observed gripper widths."""
    states = _finite(dataset_states, name="dataset states")
    if states.ndim != 2 or states.shape[1] != ACTION_DIM or horizon <= 0:
        raise ValueError("dataset states must have shape (N,20) and horizon must be positive")
    chunks = np.zeros((len(states), horizon, ACTION_DIM), dtype=np.float32)
    identity = np.asarray((1.0, 0.0, 0.0, 0.0, 1.0, 0.0), dtype=np.float32)
    chunks[..., 3:9] = identity
    chunks[..., 13:19] = identity
    chunks[..., 9] = states[:, None, 6]
    chunks[..., 19] = states[:, None, 13]
    return chunks


def _quantile_normalize_state(states: Any, q01: Any, q99: Any) -> np.ndarray:
    states = _finite(states, name="states")
    q01 = _finite(q01, shape=(ACTION_DIM,), name="state q01")
    q99 = _finite(q99, shape=(ACTION_DIM,), name="state q99")
    if states.shape[-1] != ACTION_DIM or np.any(q99 <= q01):
        raise ValueError("state quantiles must be ordered 20D values")
    return (states - q01) / (q99 - q01 + 1e-6) * 2.0 - 1.0


def nearest_reference_chunk(
    real_state: Any,
    reference_states: Any,
    reference_actions: Any,
    reference_episode_indices: Any,
    reference_frame_indices: Any,
    state_q01: Any,
    state_q99: Any,
) -> dict[str, Any]:
    """Find a quantile-normalized nearest frame and its same-episode GT chunk."""
    real = _finite(real_state, shape=(ACTION_DIM,), name="real state")
    states = _finite(reference_states, name="reference states")
    actions = _finite(reference_actions, name="reference actions")
    episodes = np.asarray(reference_episode_indices, dtype=np.int64)
    frames = np.asarray(reference_frame_indices, dtype=np.int64)
    if states.ndim != 2 or states.shape[1] != ACTION_DIM or actions.shape != states.shape:
        raise ValueError("reference states/actions must have matching shape (N,20)")
    if episodes.shape != (len(states),) or frames.shape != (len(states),) or len(states) == 0:
        raise ValueError("reference episode/frame indices must have shape (N,) and be nonempty")
    normalized = _quantile_normalize_state(np.vstack((real, states)), state_q01, state_q99)
    nearest_index = int(np.argmin(np.linalg.norm(normalized[1:] - normalized[0], axis=1)))
    episode = int(episodes[nearest_index])
    episode_indices = np.flatnonzero(episodes == episode)
    ordering = episode_indices[np.argsort(frames[episode_indices], kind="stable")]
    local_anchor = int(np.flatnonzero(ordering == nearest_index)[0])
    gt, valid = build_gt_chunk(actions[ordering], anchor=local_anchor)
    return {
        "reference_index": nearest_index,
        "episode_index": episode,
        "frame_index": int(frames[nearest_index]),
        "state_distance": float(np.linalg.norm(normalized[nearest_index + 1] - normalized[0])),
        "gt_chunk": gt,
        "valid": valid,
    }


def _policy_to_baseline_ratio(policy: Mapping[str, Any], baseline: Mapping[str, Any]) -> dict[str, Any]:
    """Return policy/no-op metric ratios; lower values are better and zero baselines are undefined."""
    ratios: dict[str, Any] = {"interpretation": "policy/no-op; lower is better; null means zero or unavailable no-op metric", "windows": {}}
    for horizon in WINDOWS:
        key = str(horizon)
        row: dict[str, float | None] = {}
        for name, value in policy["windows"][key].items():
            reference = baseline["windows"][key].get(name)
            if name == "valid_steps":
                continue
            row[name] = None if value is None or reference is None or float(reference) == 0.0 else float(value) / float(reference)
        ratios["windows"][key] = row
    return ratios


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping): return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)): return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray): return _json_safe(value.tolist())
    if isinstance(value, np.generic): return _json_safe(value.item())
    if isinstance(value, Path): return str(value)
    if isinstance(value, float) and not math.isfinite(value): raise ValueError("JSON must not contain NaN or Inf")
    return value


def _write_npz(path: Path | str, arrays: Mapping[str, Any]) -> Path:
    checked = {name: np.asarray(value) for name, value in arrays.items()}
    for name, value in checked.items():
        if value.dtype == object: raise ValueError(f"{name} must not require pickle")
        if np.issubdtype(value.dtype, np.number) and not np.isfinite(value).all(): raise ValueError(f"{name} must be finite")
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.stem}.", suffix=".npz"
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as file:
            np.savez_compressed(file, **checked)
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return path


def write_inputs_archive(path: Path | str, inputs: Mapping[str, Any]) -> Path:
    document = dict(inputs)
    metadata = document.pop("metadata_json", {})
    document["metadata_json"] = np.asarray(json.dumps(_json_safe(metadata), allow_nan=False, sort_keys=True))
    return _write_npz(path, document)


def write_predictions_archive(path: Path | str, predictions: Mapping[str, Any], metadata: Mapping[str, Any]) -> Path:
    document = dict(predictions)
    document["metadata_json"] = np.asarray(json.dumps(_json_safe(dict(metadata)), allow_nan=False, sort_keys=True))
    return _write_npz(path, document)


def _decode_image_cell(cell: Any, dataset_root: Path) -> np.ndarray:
    if isinstance(cell, Mapping):
        raw, relative = cell.get("bytes"), cell.get("path")
    else:
        raw, relative = getattr(cell, "bytes", None), getattr(cell, "path", None)
    if raw:
        from PIL import Image
        with Image.open(io.BytesIO(raw)) as image: return np.ascontiguousarray(np.asarray(image.convert("RGB"), dtype=np.uint8))
    if relative: return read_saved_rgb(dataset_root / str(relative))
    raise ValueError("dataset image cell has neither bytes nor path")


def _load_pick01_frames(dataset_root: Path, episodes: Sequence[int]) -> dict[int, list[dict[str, Any]]]:
    try:
        import pyarrow.parquet as pq
    except ImportError as error:
        raise RuntimeError("prepare requires pyarrow to read downloaded parquet data") from error
    selected = set(int(item) for item in episodes); grouped = {episode: [] for episode in selected}
    for parquet in sorted(dataset_root.rglob("*.parquet")):
        table = pq.read_table(parquet)
        columns = set(table.column_names)
        needed = {"episode_index", "frame_index", "observation.state", "actions", "observation.images.camera0", "observation.images.camera1"}
        if not needed <= columns: continue
        for row in table.to_pylist():
            episode = int(row["episode_index"])
            if episode in selected: grouped[episode].append(row)
    missing = sorted(episode for episode, rows in grouped.items() if not rows)
    if missing: raise ValueError(f"missing requested pick_01 episodes: {missing}")
    for rows in grouped.values(): rows.sort(key=lambda row: int(row["frame_index"]))
    return grouped


def load_state_quantiles(path: Path | str = DEFAULT_NORM_STATS) -> tuple[np.ndarray, np.ndarray]:
    """Load the deployed pick_0102 state q01/q99 values used by quantile normalization."""
    source = Path(path)
    try:
        document = json.loads(source.read_text(encoding="utf-8"))
        state = document["norm_stats"]["state"]
    except (OSError, json.JSONDecodeError, KeyError, TypeError) as error:
        raise ValueError(f"could not load deployed state quantiles: {source}") from error
    q01 = _finite(state["q01"], shape=(ACTION_DIM,), name="state q01").astype(np.float32)
    q99 = _finite(state["q99"], shape=(ACTION_DIM,), name="state q99").astype(np.float32)
    if np.any(q99 <= q01):
        raise ValueError("state q99 must exceed q01 in every dimension")
    return q01, q99


def prepare_inputs(dataset_root: Path | str, real_obs_dir: Path | str, output_dir: Path | str, *, episodes: Sequence[int] = tuple(range(10)), norm_stats_path: Path | str = DEFAULT_NORM_STATS) -> Path:
    """Prepare the exact frame-0 anchors, GT chunks and real start-pose proxies."""
    dataset_root, output_dir = Path(dataset_root), Path(output_dir)
    grouped = _load_pick01_frames(dataset_root, episodes)
    states=[]; camera0=[]; camera1=[]; chunks=[]; masks=[]; episode_ids=[]
    reference_states=[]; reference_actions=[]; reference_episodes=[]; reference_frames=[]
    for episode in sorted(grouped):
        rows = grouped[episode]; first = rows[0]
        if int(first["frame_index"]) != 0: raise ValueError(f"episode {episode} lacks frame 0")
        all_actions = np.asarray([row["actions"] for row in rows], dtype=np.float32)
        gt, valid = build_gt_chunk(all_actions)
        state = _finite(first["observation.state"], shape=(20,), name="dataset state").astype(np.float32)
        states.append(state); camera0.append(_decode_image_cell(first["observation.images.camera0"], dataset_root)); camera1.append(_decode_image_cell(first["observation.images.camera1"], dataset_root)); chunks.append(gt); masks.append(valid); episode_ids.append(episode)
        for row in rows:
            reference_states.append(_finite(row["observation.state"], shape=(ACTION_DIM,), name="reference state").astype(np.float32))
            reference_actions.append(_finite(row["actions"], shape=(ACTION_DIM,), name="reference action").astype(np.float32))
            reference_episodes.append(episode)
            reference_frames.append(int(row["frame_index"]))
    real = load_real_observations(real_obs_dir); start = (real[0].left_pose, real[0].right_pose)
    state_q01, state_q99 = load_state_quantiles(norm_stats_path)
    arrays = {
        "dataset_states": np.stack(states), "dataset_images_camera0": np.stack(camera0), "dataset_images_camera1": np.stack(camera1),
        "dataset_gt": np.stack(chunks), "dataset_valid": np.stack(masks), "dataset_episode_indices": np.asarray(episode_ids, dtype=np.int64),
        "reference_states": np.stack(reference_states), "reference_actions": np.stack(reference_actions),
        "reference_episode_indices": np.asarray(reference_episodes, dtype=np.int64), "reference_frame_indices": np.asarray(reference_frames, dtype=np.int64),
        "state_q01": state_q01, "state_q99": state_q99,
        "real_states": np.stack([build_real_state_proxy(start, item) for item in real]), "real_images_camera0": np.stack([item.camera0_rgb for item in real]), "real_images_camera1": np.stack([item.camera1_rgb for item in real]),
        "real_step_ids": np.asarray([item.step for item in real], dtype=np.int64),
        "metadata_json": {"dataset": "KaiyueChen/pick_01", "episodes": list(episode_ids), "anchor_frame": 0, "deployment_action_horizon": 50, "checkpoint_action_horizon": 50, "real_episode_start_pose": "first saved observation proxy", "real_has_paired_gt": False, "reference_frames": len(reference_states), "state_quantiles_path": str(Path(norm_stats_path).resolve()), "state_quantiles": "pick_0102 q01/q99"},
    }
    return write_inputs_archive(output_dir / "inputs.npz", arrays)


def _load_archive(path: Path | str) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as archive:
        result = {name: archive[name].copy() for name in archive.files}
    result["metadata_json"] = json.loads(str(result["metadata_json"]))
    return result


def _jax_backend(require_cuda: bool) -> tuple[Any, str, list[str]]:
    import jax
    backend, devices = str(jax.default_backend()), [str(device) for device in jax.devices()]
    if require_cuda and backend not in {"gpu", "cuda"}: raise RuntimeError(f"CUDA JAX backend is required, got backend={backend!r}, devices={devices!r}")
    return jax, backend, devices


def predict_robot_action_chunk(
    policy: Any,
    observation: Mapping[str, Any],
    task: str,
    *,
    seed: int,
    num_steps: int,
    jax_module: Any,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample one policy chunk, synchronise it, and return model/robot actions."""
    sampled = policy.predict_action_chunk(observation, task, seed=seed, num_steps=num_steps)
    normalized = np.asarray(jax_module.block_until_ready(sampled)[0], dtype=np.float32)
    robot = np.asarray(policy.unnormalize_actions(normalized), dtype=np.float32)
    expected_model = (policy.config.action_horizon, policy.config.action_dim)
    expected_robot = (policy.config.action_horizon, policy.config.robot_action_dim)
    if normalized.shape != expected_model or robot.shape != expected_robot or not np.isfinite(robot).all():
        raise ValueError("Pi0.5 returned an unexpected action shape or non-finite action")
    return normalized, robot


def warmup_policy(
    policy: Any,
    observation: Mapping[str, Any],
    task: str,
    *,
    seed: int,
    num_steps: int,
    jax_module: Any,
) -> None:
    """Consume the configured live-equivalent warmup sample and discard it."""
    predict_robot_action_chunk(
        policy, observation, task, seed=seed, num_steps=num_steps, jax_module=jax_module
    )


def infer(inputs_path: Path | str, output_dir: Path | str, *, config_path: Path | str = DEFAULT_CONFIG, require_cuda: bool = True) -> Path:
    """Lazily load Pi0.5, synchronise each sample, and atomically checkpoint progress."""
    inputs = _load_archive(inputs_path); output_dir = Path(output_dir); archive = output_dir / "predictions.npz"
    for import_root in (ROOT, ROOT / "deploy_pi05" / "src"):
        if str(import_root) not in sys.path: sys.path.insert(0, str(import_root))
    jax, backend, devices = _jax_backend(require_cuda)
    from deploy_pi05.deployment import load_deployment_config, make_policy_config, section
    from deploy_pi05.policy import Pi05RemotePolicy
    config_file = Path(config_path).resolve(); config = load_deployment_config(config_file, "pi05")
    policy_config = make_policy_config(config, config_file)
    if policy_config.state_dim != 20 or policy_config.robot_action_dim != 20 or policy_config.action_horizon != HORIZON: raise ValueError("offline evaluator requires deployed dual-arm 20D/H=50 contract")
    policy = Pi05RemotePolicy(policy_config); prompt = str(section(config, "observation")["language_prompt"]); seed = int(config.get("seed", 0)); num_steps = int(config.get("num_steps", 10))
    warmup_runs = int(section(config, "runtime").get("warmup_runs", 1))
    if warmup_runs != 1:
        raise ValueError(f"offline evaluator requires runtime.warmup_runs=1, got {warmup_runs}")
    if len(inputs["dataset_states"]) == 0:
        raise ValueError("warmup requires the first dataset observation")
    first_raw = {
        "observation.state": np.asarray(inputs["dataset_states"][0], dtype=np.float32),
        "observation.images.camera0": np.asarray(inputs["dataset_images_camera0"][0], dtype=np.uint8),
        "observation.images.camera1": np.asarray(inputs["dataset_images_camera1"][0], dtype=np.uint8),
    }
    warmup_policy(policy, first_raw, prompt, seed=seed, num_steps=num_steps, jax_module=jax)
    collected = {"dataset_normalized": [], "dataset_robot": [], "real_normalized": [], "real_robot": []}
    metadata = {"complete": False, "completed_dataset": 0, "completed_real": 0, "jax_backend": backend, "jax_devices": devices, "config_path": str(config_file), "seed": seed, "num_steps": num_steps, "warmup_runs": warmup_runs, "action_horizon": HORIZON}
    def checkpoint() -> None:
        arrays = {key: np.stack(values) if values else np.empty((0, HORIZON, policy_config.action_dim if key.endswith("normalized") else ACTION_DIM), dtype=np.float32) for key, values in collected.items()}
        write_predictions_archive(archive, arrays, metadata)
    for label, state_key, image0_key, image1_key in (("dataset", "dataset_states", "dataset_images_camera0", "dataset_images_camera1"), ("real", "real_states", "real_images_camera0", "real_images_camera1")):
        for state, image0, image1 in zip(inputs[state_key], inputs[image0_key], inputs[image1_key], strict=True):
            raw = {"observation.state": np.asarray(state, dtype=np.float32), "observation.images.camera0": np.asarray(image0, dtype=np.uint8), "observation.images.camera1": np.asarray(image1, dtype=np.uint8)}
            normalized, robot = predict_robot_action_chunk(
                policy, raw, prompt, seed=seed, num_steps=num_steps, jax_module=jax
            )
            if normalized.shape != (HORIZON, policy_config.action_dim) or robot.shape != (HORIZON, ACTION_DIM): raise ValueError("Pi0.5 returned an unexpected action shape")
            collected[f"{label}_normalized"].append(np.asarray(normalized, dtype=np.float32)); collected[f"{label}_robot"].append(np.asarray(robot, dtype=np.float32))
            metadata[f"completed_{label}"] += 1; checkpoint()
    metadata["complete"] = True; checkpoint(); return archive


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else ["sample"]
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields, lineterminator="\n"); writer.writeheader(); writer.writerows(rows)
    return path


def write_report(inputs_path: Path | str, predictions_path: Path | str, output_dir: Path | str) -> dict[str, Path]:
    """Write finite paired metrics plus explicitly non-paired real prediction ranges."""
    inputs, predictions, output = _load_archive(inputs_path), _load_archive(predictions_path), Path(output_dir)
    count = min(len(inputs["dataset_gt"]), len(predictions["dataset_robot"]))
    paired = compute_native_metrics(predictions["dataset_robot"][:count], inputs["dataset_gt"][:count], inputs["dataset_valid"][:count]) if count else {"windows": {str(window): {"valid_steps": 0} for window in WINDOWS}}
    no_op = compute_native_metrics(
        build_noop_chunks(inputs["dataset_states"][:count]),
        inputs["dataset_gt"][:count],
        inputs["dataset_valid"][:count],
    ) if count else {"windows": {str(window): {"valid_steps": 0} for window in WINDOWS}}
    episode_rows=[]
    for index in range(count):
        individual = compute_native_metrics(predictions["dataset_robot"][index:index+1], inputs["dataset_gt"][index:index+1], inputs["dataset_valid"][index:index+1])
        for horizon, row in individual["windows"].items(): episode_rows.append({"episode_index": int(inputs["dataset_episode_indices"][index]), "horizon": int(horizon), **row})
    real_rows=[]
    for index, action in enumerate(predictions["real_robot"]):
        nearest = nearest_reference_chunk(
            inputs["real_states"][index], inputs["reference_states"], inputs["reference_actions"],
            inputs["reference_episode_indices"], inputs["reference_frame_indices"],
            inputs["state_q01"], inputs["state_q99"],
        )
        h20 = compute_native_metrics(action[None], nearest["gt_chunk"][None], nearest["valid"][None])["windows"]["20"]
        real_rows.append({
            "real_observation_index": index,
            "step_id": int(inputs.get("real_step_ids", np.arange(len(predictions["real_robot"])))[index]),
            "comparison": "nearest_state_unpaired", "paired_gt": False,
            "nearest_reference_episode": nearest["episode_index"], "nearest_reference_frame": nearest["frame_index"],
            "nearest_state_distance": nearest["state_distance"],
            "action_min": float(action.min()), "action_max": float(action.max()),
            "left_xyz_range_mm": float(np.ptp(action[:, :3]) * 1000.0), "right_xyz_range_mm": float(np.ptp(action[:, 10:13]) * 1000.0),
            **{f"h20_{name}": value for name, value in h20.items()},
        })
    summary = {"paired_metrics": paired, "no_op_baseline_metrics": no_op, "policy_to_noop_ratio": _policy_to_baseline_ratio(paired, no_op), "completed_dataset_predictions": count, "completed_real_predictions": len(predictions["real_robot"]), "limitations": ["Dataset metrics are paired only for frame-0 pick_01 anchors.", "The checkpoint and deployment action horizons are both 50; H=1/10/20 windows are diagnostic prefixes of the same 50-step prediction.", "Saved real observations use the first saved pose as a state proxy and nearest-state same-episode GT only as an unpaired descriptive comparison."], "provenance": inputs["metadata_json"], "prediction_metadata": predictions["metadata_json"]}
    output.mkdir(parents=True, exist_ok=True); summary_path=output/"summary.json"
    with summary_path.open("w", encoding="utf-8") as file: json.dump(_json_safe(summary), file, allow_nan=False, indent=2, sort_keys=True); file.write("\n")
    return {"summary_json": summary_path, "per_episode_csv": _write_csv(output/"per_episode.csv", episode_rows), "real_obs_actions_csv": _write_csv(output/"real_obs_actions.csv", real_rows)}


def _build_arg_parser() -> argparse.ArgumentParser:
    parser=argparse.ArgumentParser(description=__doc__); commands=parser.add_subparsers(dest="command", required=True)
    for name in ("prepare", "run"):
        sub=commands.add_parser(name); sub.add_argument("--dataset-root", type=Path, required=True); sub.add_argument("--real-obs-dir", type=Path, default=DEFAULT_REAL_OBS_DIR); sub.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR); sub.add_argument("--episodes", default="0-9"); sub.add_argument("--norm-stats", type=Path, default=DEFAULT_NORM_STATS)
        if name == "run": sub.add_argument("--require-cuda", action=argparse.BooleanOptionalAction, default=True)
    infer_parser=commands.add_parser("infer"); infer_parser.add_argument("--inputs", type=Path, default=DEFAULT_OUTPUT_DIR/"inputs.npz"); infer_parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR); infer_parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG); infer_parser.add_argument("--require-cuda", action=argparse.BooleanOptionalAction, default=True)
    report=commands.add_parser("report"); report.add_argument("--inputs", type=Path, default=DEFAULT_OUTPUT_DIR/"inputs.npz"); report.add_argument("--predictions", type=Path, default=DEFAULT_OUTPUT_DIR/"predictions.npz"); report.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser


def _episodes(value: str) -> tuple[int, ...]:
    values=[]
    for item in value.split(","):
        start, separator, end=item.partition("-")
        values.extend(range(int(start), int(end)+1) if separator else [int(start)])
    return tuple(sorted(set(values)))


def main(argv: Sequence[str] | None = None) -> int:
    args=_build_arg_parser().parse_args(argv)
    if args.command in {"prepare", "run"}: prepare_inputs(args.dataset_root, args.real_obs_dir, args.output_dir, episodes=_episodes(args.episodes), norm_stats_path=args.norm_stats)
    if args.command in {"infer", "run"}:
        inputs = args.output_dir/"inputs.npz" if args.command == "run" else args.inputs
        infer(inputs, args.output_dir, config_path=getattr(args, "config", DEFAULT_CONFIG), require_cuda=getattr(args, "require_cuda", True))
    if args.command in {"report", "run"}:
        inputs = args.output_dir/"inputs.npz" if args.command == "run" else args.inputs
        predictions = args.output_dir/"predictions.npz" if args.command == "run" else args.predictions
        write_report(inputs, predictions, args.output_dir)
    return 0


if __name__ == "__main__": raise SystemExit(main())
