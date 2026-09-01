from __future__ import annotations

import importlib.util
import json
import csv
import sys
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
from scipy.spatial.transform import Rotation


MODULE_PATH = Path(__file__).resolve().parents[1] / "tools" / "eval_pi05_deco_obs_offline.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("eval_pi05_deco_obs_offline", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


def _write_snapshot(
    root: Path,
    step: int,
    *,
    left_pose: np.ndarray,
    right_pose: np.ndarray,
    left_gripper: float,
    right_gripper: float,
    timestamp: float,
    camera_rgb: np.ndarray,
) -> None:
    step_dir = root / f"step_{step:06d}"
    step_dir.mkdir(parents=True)
    _write_json(step_dir / "robot0_eef_pos.json", [left_pose[:3].tolist()])
    _write_json(step_dir / "robot0_eef_rot_axis_angle.json", [left_pose[3:].tolist()])
    _write_json(step_dir / "robot0_gripper_width.json", [[left_gripper]])
    _write_json(step_dir / "robot1_eef_pos.json", [right_pose[:3].tolist()])
    _write_json(step_dir / "robot1_eef_rot_axis_angle.json", [right_pose[3:].tolist()])
    _write_json(step_dir / "robot1_gripper_width.json", [[right_gripper]])
    _write_json(step_dir / "timestamp.json", [timestamp])
    assert cv2.imwrite(
        str(step_dir / "camera1_rgb.jpg"),
        np.ascontiguousarray(camera_rgb[..., ::-1]),
    )


def _identity_rotation_6d() -> np.ndarray:
    return np.asarray([1.0, 0.0, 0.0, 0.0, 1.0, 0.0], dtype=np.float64)


def _rotation_6d_columns(matrix: np.ndarray) -> np.ndarray:
    return np.concatenate((matrix[:, 0], matrix[:, 1]))


def _right_actions(rows: int) -> np.ndarray:
    actions = np.zeros((rows, 10), dtype=np.float64)
    actions[:, 3:9] = _identity_rotation_6d()
    return actions


def _server_rotation_angle(rotation_6d: np.ndarray) -> float:
    first = rotation_6d[:3]
    second = rotation_6d[3:]
    first = first / np.linalg.norm(first)
    second = second / np.linalg.norm(second)
    second = second - np.dot(first, second) * first
    second = second / np.linalg.norm(second)
    matrix = np.stack((first, second, np.cross(first, second)), axis=-1)
    return float(np.arccos(np.clip((np.trace(matrix) - 1.0) / 2.0, -1.0, 1.0)))


def _server_safety_verdict(actions: np.ndarray, limits: dict[str, float]) -> bool:
    """Independent replica of vbvla_safety's float32 semantic checks for one arm."""
    raw_action_array = np.asarray(actions)
    action_array = raw_action_array.astype(np.float32).astype(np.float64)
    translation = np.linalg.norm(action_array[:, :3], axis=1)
    rotation = np.asarray(
        [_server_rotation_angle(action[3:9]) for action in action_array], dtype=np.float64
    )
    gripper = raw_action_array[:, 9]
    return not bool(
        np.any(translation > limits["max_pos_delta"])
        or np.any(rotation > limits["max_rot_delta"])
        or np.any(gripper < limits["min_gripper"])
        or np.any(gripper > limits["max_gripper"])
    )


def test_load_deco_observations_sorts_numeric_steps_and_decodes_camera1_as_rgb(tmp_path: Path):
    left = np.zeros(6)
    right = np.asarray([0.4, 0.0, 0.0, 0.0, 0.0, 0.0])
    _write_snapshot(
        tmp_path,
        32,
        left_pose=left,
        right_pose=right,
        left_gripper=0.12,
        right_gripper=0.08,
        timestamp=12.0,
        camera_rgb=np.asarray([[[0, 255, 0]]], dtype=np.uint8),
    )
    _write_snapshot(
        tmp_path,
        2,
        left_pose=left,
        right_pose=right,
        left_gripper=0.11,
        right_gripper=0.07,
        timestamp=10.0,
        camera_rgb=np.asarray([[[255, 0, 0]]], dtype=np.uint8),
    )

    observations = _load_module().load_deco_observations(tmp_path)

    assert [observation.step for observation in observations] == [2, 32]
    assert observations[0].timestamp == pytest.approx(10.0)
    assert observations[0].camera1_rgb.dtype == np.uint8
    assert observations[0].camera1_rgb.shape == (1, 1, 3)
    assert int(np.argmax(observations[0].camera1_rgb[0, 0])) == 0
    assert int(np.argmax(observations[1].camera1_rgb[0, 0])) == 1


def test_build_server_state_uses_first_pose_proxy_and_right_projection_contract(tmp_path: Path):
    left_start = np.asarray([0.1, 0.2, 0.3, 0.0, 0.0, 0.0])
    right_start = np.asarray([-0.2, 0.2, 0.3, 0.0, 0.0, 0.0])
    _write_snapshot(
        tmp_path,
        0,
        left_pose=left_start,
        right_pose=right_start,
        left_gripper=0.12,
        right_gripper=0.07,
        timestamp=1.0,
        camera_rgb=np.zeros((1, 1, 3), dtype=np.uint8),
    )
    observations = _load_module().load_deco_observations(tmp_path)
    observation = observations[0]

    state = _load_module().build_server_state(
        (observation.left_pose, observation.right_pose), observation
    )

    assert state.shape == (20,)
    np.testing.assert_allclose(state[:6], 0.0)
    assert state[6] == pytest.approx(0.12)
    np.testing.assert_allclose(state[7:13], 0.0)
    assert state[13] == pytest.approx(0.07)
    np.testing.assert_allclose(state[14:17], [0.3, 0.0, 0.0])
    from deploy_pi05.right_arm_adapter import project_right_observation

    projected = project_right_observation({"observation.state": state})
    np.testing.assert_allclose(projected["observation.state"], state[7:14])


def test_actions_to_absolute_waypoints_uses_column_rotation6d_and_chained_local_transforms():
    actions = _right_actions(2)
    actions[0, :3] = [1.0, 0.0, 0.0]
    actions[0, 3:9] = _rotation_6d_columns(Rotation.from_euler("z", 90, degrees=True).as_matrix())
    actions[0, 9] = 0.2
    actions[1, :3] = [1.0, 0.0, 0.0]
    actions[1, 9] = 0.3

    waypoints = _load_module().actions_to_absolute_waypoints(actions, np.zeros(6))

    assert waypoints.shape == (2, 7)
    np.testing.assert_allclose(waypoints[:, :3], [[1.0, 0.0, 0.0], [1.0, 1.0, 0.0]], atol=1e-7)
    np.testing.assert_allclose(waypoints[:, 3:6], [[0.0, 0.0, np.pi / 2]] * 2, atol=1e-7)
    np.testing.assert_allclose(waypoints[:, 6], [0.2, 0.3])


def test_action_safety_metrics_accepts_exact_server_boundaries_and_reports_violations():
    limits = {
        "max_pos_delta": 0.03,
        "max_rot_delta": 0.5,
        "min_gripper": -0.05,
        "max_gripper": 1.05,
    }
    actions = _right_actions(2)
    actions[0, :3] = [0.03, 0.0, 0.0]
    actions[0, 3:9] = _rotation_6d_columns(Rotation.from_rotvec([0.0, 0.0, 0.5]).as_matrix())
    actions[0, 9] = -0.05
    actions[1, 9] = 1.05

    safe = _load_module().action_safety_metrics(actions, limits)

    assert safe["safe"] is _server_safety_verdict(actions, limits)
    assert safe["max_translation_delta"] == pytest.approx(0.03)
    assert safe["max_rotation_delta"] == pytest.approx(0.5)
    assert safe["min_gripper"] == pytest.approx(-0.05)
    assert safe["max_gripper"] == pytest.approx(1.05)

    unsafe_actions = actions.copy()
    unsafe_actions[1, :3] = [0.030001, 0.0, 0.0]
    unsafe_actions[1, 9] = 1.050001
    unsafe = _load_module().action_safety_metrics(unsafe_actions, limits)

    assert unsafe["safe"] is False
    assert {(item["kind"], item["step"]) for item in unsafe["violations"]} >= {
        ("translation_delta", 1),
        ("gripper", 1),
    }


@pytest.mark.parametrize("direction", [np.asarray([1.0, 1.0, 1.0]), np.asarray([1.0, 5.0, -3.0])])
def test_action_safety_metrics_matches_server_float32_translation_boundary_for_arbitrary_directions(
    direction,
):
    limits = {
        "max_pos_delta": 0.03,
        "max_rot_delta": 0.5,
        "min_gripper": -0.05,
        "max_gripper": 1.05,
    }
    actions = _right_actions(1)
    actions[0, :3] = direction / np.linalg.norm(direction) * limits["max_pos_delta"]

    metrics = _load_module().action_safety_metrics(actions, limits)

    assert metrics["safe"] is _server_safety_verdict(actions, limits)

    over_limit = _right_actions(1)
    over_limit[0, :3] = direction / np.linalg.norm(direction) * 0.0301
    assert _server_safety_verdict(over_limit, limits) is False
    assert _load_module().action_safety_metrics(over_limit, limits)["safe"] is False


@pytest.mark.parametrize(
    "axis",
    [
        np.asarray([1.0, 1.0, 1.0]),
        np.asarray([1.0, 5.0, -3.0]),
        np.asarray([1e-8, 1.0, 0.0]),
    ],
)
def test_action_safety_metrics_matches_server_float32_rotation_boundary_for_arbitrary_axes(axis):
    limits = {
        "max_pos_delta": 0.03,
        "max_rot_delta": 0.5,
        "min_gripper": -0.05,
        "max_gripper": 1.05,
    }
    actions = _right_actions(1)
    matrix = Rotation.from_rotvec(axis / np.linalg.norm(axis) * limits["max_rot_delta"]).as_matrix()
    actions[0, 3:9] = _rotation_6d_columns(matrix)

    metrics = _load_module().action_safety_metrics(actions, limits)

    assert metrics["safe"] is _server_safety_verdict(actions, limits)
    assert metrics["max_rotation_delta"] == pytest.approx(
        _server_rotation_angle(
            np.asarray(actions[0, 3:9], dtype=np.float32).astype(np.float64)
        ),
        abs=1e-12,
    )

    over_limit = _right_actions(1)
    over_matrix = Rotation.from_rotvec(axis / np.linalg.norm(axis) * 0.501).as_matrix()
    over_limit[0, 3:9] = _rotation_6d_columns(over_matrix)
    assert _server_safety_verdict(over_limit, limits) is False
    unsafe = _load_module().action_safety_metrics(over_limit, limits)
    assert unsafe["safe"] is False
    assert unsafe["violations"][0]["kind"] == "rotation_delta"


def test_pure_helpers_reject_nonfinite_or_wrong_shaped_actions():
    module = _load_module()
    with pytest.raises(ValueError, match=r"shape \(H,10\)"):
        module.actions_to_absolute_waypoints(np.zeros((2, 9)), np.zeros(6))
    actions = _right_actions(1)
    actions[0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        module.action_safety_metrics(
            actions,
            {"max_pos_delta": 0.03, "max_rot_delta": 0.5, "min_gripper": -0.05, "max_gripper": 1.05},
        )
    with pytest.raises(ValueError, match="nonempty"):
        module.action_safety_metrics(
            np.empty((0, 10)),
            {"max_pos_delta": 0.03, "max_rot_delta": 0.5, "min_gripper": -0.05, "max_gripper": 1.05},
        )


def test_action_safety_metrics_uses_original_float64_gripper_for_server_structure_bounds():
    actions = _right_actions(1)
    actions[0, 9] = 1.050000001

    metrics = _load_module().action_safety_metrics(
        actions,
        {"max_pos_delta": 0.03, "max_rot_delta": 0.5, "min_gripper": -0.05, "max_gripper": 1.05},
    )

    assert metrics["safe"] is False
    assert metrics["max_gripper"] == pytest.approx(1.050000001)
    assert metrics["violations"][0]["kind"] == "gripper"


def test_action_safety_metrics_rejects_nonfloating_actions_before_numeric_coercion():
    action = np.asarray([[0, 0, 0, 1, 0, 0, 0, 1, 0, 0]], dtype=np.int32)

    with pytest.raises(ValueError, match="dtype"):
        _load_module().action_safety_metrics(
            action,
            {"max_pos_delta": 0.03, "max_rot_delta": 0.5, "min_gripper": -0.05, "max_gripper": 1.05},
        )


def test_action_safety_metrics_rejects_values_outside_float32_range():
    actions = _right_actions(1)
    actions[0, 0] = float(np.finfo(np.float32).max) * 2.0

    with pytest.raises(ValueError, match="float32 range"):
        _load_module().action_safety_metrics(
            actions,
            {"max_pos_delta": 0.03, "max_rot_delta": 0.5, "min_gripper": -0.05, "max_gripper": 1.05},
        )


def test_action_safety_metrics_accepts_float32_server_boundary_action():
    actions = _right_actions(1).astype(np.float32)
    actions[0, 9] = np.float32(1.05)

    metrics = _load_module().action_safety_metrics(
        actions,
        {"max_pos_delta": 0.03, "max_rot_delta": 0.5, "min_gripper": -0.05, "max_gripper": 1.05},
    )

    assert metrics["safe"] is True


def test_decoder_uses_cv2_without_pillow_runtime_dependency():
    source = MODULE_PATH.read_text(encoding="utf-8")

    assert "cv2.imread" in source
    assert "PIL" not in source


def _archive_arrays(chunks: int = 2, horizon: int = 2) -> dict[str, np.ndarray]:
    identity = _identity_rotation_6d().astype(np.float32)
    right = np.zeros((chunks, horizon, 10), dtype=np.float32)
    right[..., 3:9] = identity
    wire = np.zeros((chunks, horizon, 20), dtype=np.float32)
    wire[..., 3:9] = identity
    wire[..., 13:19] = identity
    waypoints = np.zeros((chunks, horizon, 7), dtype=np.float32)
    return {
        "states": np.zeros((chunks, 20), dtype=np.float32),
        "normalized_actions": right.copy(),
        "right_actions": right,
        "wire_actions": wire,
        "absolute_waypoints": waypoints,
        "step_ids": np.arange(chunks, dtype=np.int64) * 32,
        "timestamps": np.arange(chunks, dtype=np.float64),
    }


def test_task2_cli_has_safe_infer_and_report_defaults(tmp_path: Path):
    module = _load_module()
    parser = module._build_arg_parser()

    infer = parser.parse_args(["infer"])
    report = parser.parse_args(["report", "--artifact", str(tmp_path / "predictions.npz")])

    assert infer.command == "infer"
    assert infer.config.name == "deploy_pi05_right.yaml"
    assert infer.require_cuda is True
    assert infer.obs_dir.name == "eval_obs_20260901_143909"
    assert report.command == "report"
    assert report.deco_trace is None
    assert report.output_dir.name == "eval_obs_20260901_143909"


def test_require_cuda_accepts_jax_gpu_backend_label(monkeypatch):
    module = _load_module()
    device = SimpleNamespace(platform="gpu", device_kind="NVIDIA GPU")
    fake_jax = SimpleNamespace(default_backend=lambda: "gpu", devices=lambda: (device,))
    monkeypatch.setitem(sys.modules, "jax", fake_jax)

    returned_jax, backend, devices = module._require_cuda()

    assert returned_jax is fake_jax
    assert backend == "gpu"
    assert devices == (device,)


def test_prediction_archive_is_pickle_free_and_validates_metadata_and_shapes(tmp_path: Path):
    module = _load_module()
    archive_path = tmp_path / "predictions.npz"
    metadata = {"complete": True, "completed_chunks": 2, "action_horizon": 2}

    module.write_prediction_archive(archive_path, _archive_arrays(), metadata)
    loaded_arrays, loaded_metadata = module.load_prediction_archive(archive_path)

    assert loaded_metadata == metadata
    assert loaded_arrays["wire_actions"].shape == (2, 2, 20)
    with np.load(archive_path, allow_pickle=False) as archive:
        assert archive["metadata_json"].dtype.kind in {"U", "S"}

    np.savez_compressed(archive_path, metadata_json=np.asarray(json.dumps(metadata)))
    with pytest.raises(ValueError, match="missing required arrays"):
        module.load_prediction_archive(archive_path)


def test_time_interpolation_uses_common_seconds_interval():
    module = _load_module()
    source_time = np.asarray([3.0, 5.0])
    source_values = np.asarray([[0.0, 0.0, 0.0], [2.0, 4.0, 6.0]])
    target_time = np.asarray([2.0, 3.0, 4.0, 5.0, 6.0])

    times, interpolated = module.interpolate_on_common_seconds(
        source_time, source_values, target_time
    )

    np.testing.assert_allclose(times, [3.0, 4.0, 5.0])
    np.testing.assert_allclose(interpolated, [[0.0, 0.0, 0.0], [1.0, 2.0, 3.0], [2.0, 4.0, 6.0]])


def test_report_writes_csv_summary_and_pngs_with_optional_deco_trace(tmp_path: Path):
    module = _load_module()
    artifact = tmp_path / "predictions.npz"
    output = tmp_path / "report"
    arrays = _archive_arrays()
    arrays["absolute_waypoints"][0, :, :3] = [[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]]
    arrays["absolute_waypoints"][1, :, :3] = [[0.0, 0.0, 0.0], [0.0, 0.1, 0.0]]
    module.write_prediction_archive(
        artifact,
        arrays,
        {"complete": True, "completed_chunks": 2, "action_horizon": 2, "control_hz": 10.0},
    )
    trace = tmp_path / "chunk_trace.jsonl"
    trace.write_text(
        "".join(
            json.dumps(record) + "\n"
            for record in (
                {
                    "time": 0.0,
                    "action_timestamps": [0.0, 0.1],
                    "absolute_waypoints": [[0.0] * 7, [0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]],
                },
                {
                    "time": 1.0,
                    "action_timestamps": [1.0, 1.1],
                    "absolute_waypoints": [[0.0] * 7, [0.0, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0]],
                },
            )
        ),
        encoding="utf-8",
    )

    summary = module.run_report(artifact, output, deco_trace=trace)

    assert summary["chunks"] == 2
    assert "deco_comparison" in summary
    assert summary["deco_comparison"]["chunks_compared"] == 2
    with (output / "trajectory.csv").open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    assert len(rows) == 4
    assert set(rows[0]) >= {"chunk_index", "step_id", "action_index", "absolute_x", "gripper", "safe"}
    saved_summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    assert saved_summary == module.json_safe(summary)
    assert (output / "right_start_chunk.png").stat().st_size > 0
    assert (output / "right_all_chunks.png").stat().st_size > 0


def test_task2_source_has_no_robot_bridge_or_eager_jax_imports():
    source = MODULE_PATH.read_text(encoding="utf-8")

    assert "RobotBridgeClient" not in source
    assert "pi05_client" not in source
    assert "bridge_client" not in source
    assert "\nimport jax\n" not in source
