import ast
import csv
import importlib.util
import io
import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


MODULE_PATH = Path(__file__).resolve().parents[1] / "tools" / "eval_pi05_pick01_action_gt.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("eval_pi05_pick01_action_gt", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _rot6d(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float64)
    return np.concatenate((matrix[:, 0], matrix[:, 1])).astype(np.float32)


def _identity_actions(shape: tuple[int, ...]) -> np.ndarray:
    actions = np.zeros(shape + (20,), dtype=np.float32)
    actions[..., 3:9] = _rot6d(np.eye(3))
    actions[..., 13:19] = _rot6d(np.eye(3))
    return actions


def _jpeg_bytes(color: tuple[int, int, int]) -> bytes:
    image_module = pytest.importorskip("PIL.Image")
    buffer = io.BytesIO()
    image_module.new("RGB", (2, 1), color).save(buffer, format="JPEG", quality=100, subsampling=0)
    return buffer.getvalue()


def test_build_gt_chunk_uses_frame_zero_actions_and_masks_terminal_and_padding():
    actions = _identity_actions((4,))
    actions[:, 0] = [1.0, 2.0, 3.0, 0.0]
    actions[3] = 0.0  # terminal sentinel; it remains recorded but is never scored.

    module = _load_module()
    gt, valid = module.build_gt_chunk(actions, horizon=6)

    np.testing.assert_allclose(gt[:4], actions)
    np.testing.assert_allclose(gt[4:], 0.0)
    np.testing.assert_array_equal(valid, [True, True, True, False, False, False])


def test_rotation6d_geodesic_and_native_windows_use_expected_units():
    pred = _identity_actions((1, 50))
    gt = _identity_actions((1, 50))
    valid = np.ones((1, 50), dtype=bool)
    pred[0, :, 0] = 0.001
    pred[0, :, 9] = 0.002
    angle = np.deg2rad(90.0)
    yaw = np.asarray(
        [[np.cos(angle), -np.sin(angle), 0.0], [np.sin(angle), np.cos(angle), 0.0], [0.0, 0.0, 1.0]]
    )
    pred[0, :, 3:9] = _rot6d(yaw)

    module = _load_module()
    assert module.rotation_geodesic_deg(_rot6d(np.eye(3)), _rot6d(np.eye(3))) == pytest.approx(0.0)
    assert module.rotation_geodesic_deg(_rot6d(yaw), _rot6d(np.eye(3))) == pytest.approx(90.0)
    metrics = module.compute_native_metrics(pred, gt, valid)

    for horizon in (1, 10, 20, 50):
        row = metrics["windows"][str(horizon)]
        assert row["left_translation_l2_mm"] == pytest.approx(1.0)
        assert row["left_rotation_geodesic_deg"] == pytest.approx(90.0)
        assert row["left_gripper_mae_mm"] == pytest.approx(2.0)
        assert row["overall_mae"] == pytest.approx((0.001 + 0.002 + 4.0) / 20.0)


def test_real_state_proxy_matches_deployed_20d_relative_pose_contract():
    module = _load_module()
    start = (np.zeros(6, dtype=np.float32), np.zeros(6, dtype=np.float32))
    observation = {
        "left_pose": np.asarray([1.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        "right_pose": np.asarray([0.0, 2.0, 0.0, 0.0, 0.0, 0.0]),
        "left_gripper": 0.012,
        "right_gripper": 0.034,
    }

    state = module.build_real_state_proxy(start, observation)

    assert state.shape == (20,)
    np.testing.assert_allclose(state[:3], [1.0, 0.0, 0.0])
    assert state[6] == pytest.approx(0.012)
    np.testing.assert_allclose(state[7:10], [0.0, 2.0, 0.0])
    assert state[13] == pytest.approx(0.034)
    np.testing.assert_allclose(state[14:17], [1.0, -2.0, 0.0])


def test_pillow_rgb_image_decode_preserves_saved_channel_order(tmp_path: Path):
    image_path = tmp_path / "camera0_rgb.jpg"
    image_path.write_bytes(_jpeg_bytes((250, 5, 1)))

    module = _load_module()
    rgb = module.read_saved_rgb(image_path)

    assert rgb.dtype == np.uint8
    assert rgb.shape == (1, 2, 3)
    assert int(rgb[0, 0, 0]) > int(rgb[0, 0, 2])


def test_strict_archives_and_report_are_pickle_free_and_finite(tmp_path: Path):
    module = _load_module()
    inputs = {
        "dataset_states": np.zeros((1, 20), dtype=np.float32),
        "dataset_images_camera0": np.zeros((1, 2, 2, 3), dtype=np.uint8),
        "dataset_images_camera1": np.zeros((1, 2, 2, 3), dtype=np.uint8),
        "dataset_gt": _identity_actions((1, 50)),
        "dataset_valid": np.ones((1, 50), dtype=bool),
        "dataset_episode_indices": np.asarray([0], dtype=np.int64),
        "real_states": np.zeros((0, 20), dtype=np.float32),
        "real_images_camera0": np.zeros((0, 2, 2, 3), dtype=np.uint8),
        "real_images_camera1": np.zeros((0, 2, 2, 3), dtype=np.uint8),
        "metadata_json": {"checkpoint_action_horizon": 50, "deployment_action_horizon": 50},
    }
    inputs_path = module.write_inputs_archive(tmp_path / "inputs.npz", inputs)
    with np.load(inputs_path, allow_pickle=False) as archive:
        assert json.loads(str(archive["metadata_json"]))["deployment_action_horizon"] == 50

    predictions_path = module.write_predictions_archive(
        tmp_path / "predictions.npz",
        {"dataset_normalized": _identity_actions((1, 50)), "dataset_robot": _identity_actions((1, 50)),
         "real_normalized": np.zeros((0, 50, 20), dtype=np.float32), "real_robot": np.zeros((0, 50, 20), dtype=np.float32)},
        {"complete": True, "completed_dataset": 1, "completed_real": 0},
    )
    paths = module.write_report(inputs_path, predictions_path, tmp_path)
    summary = json.loads(paths["summary_json"].read_text(encoding="utf-8"))
    assert summary["paired_metrics"]["windows"]["50"]["valid_steps"] == 50
    assert summary["provenance"]["checkpoint_action_horizon"] == 50
    assert not any("H<=20" in item for item in summary["limitations"])
    assert paths["per_episode_csv"].exists() and paths["real_obs_actions_csv"].exists()


def test_tool_has_no_robot_or_network_imports_and_cli_exposes_cuda_switch():
    source = MODULE_PATH.read_text(encoding="utf-8")
    imports = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.append(node.module)
    assert not any(name.startswith(("real_world", "socket", "requests", "websocket")) for name in imports)

    module = _load_module()
    args = module._build_arg_parser().parse_args(["infer", "--no-require-cuda"])
    assert args.require_cuda is False


def test_warmup_consumes_one_dataset_prediction_and_blocks_before_evaluation():
    class Policy:
        def __init__(self):
            self.calls = []
            self.config = SimpleNamespace(action_horizon=50, action_dim=20, robot_action_dim=20)

        def predict_action_chunk(self, observation, task, *, seed, num_steps):
            self.calls.append((observation, task, seed, num_steps))
            return np.zeros((1, 50, 20), dtype=np.float32)

        def unnormalize_actions(self, action):
            return np.asarray(action, dtype=np.float32)

    class Jax:
        def __init__(self):
            self.blocked = []

        def block_until_ready(self, value):
            self.blocked.append(value)
            return value

    module = _load_module()
    policy, jax = Policy(), Jax()
    raw = {"observation.state": np.zeros(20, dtype=np.float32)}

    module.warmup_policy(policy, raw, "pick", seed=7, num_steps=10, jax_module=jax)

    assert policy.calls == [(raw, "pick", 7, 10)]
    assert len(jax.blocked) == 1


def test_npz_archive_uses_same_directory_atomic_replace(tmp_path: Path, monkeypatch):
    module = _load_module()
    target = tmp_path / "nested" / "artifact.npz"
    original_replace = os.replace
    replacements = []

    def tracking_replace(source, destination):
        replacements.append((Path(source), Path(destination)))
        return original_replace(source, destination)

    monkeypatch.setattr(module.os, "replace", tracking_replace)
    module.write_inputs_archive(target, {"values": np.asarray([1.0], dtype=np.float32)})

    assert target.exists()
    assert replacements and replacements[0][1] == target
    assert replacements[0][0].parent == target.parent
    assert replacements[0][0].suffix == ".npz"
    assert not replacements[0][0].exists()


def test_run_cli_accepts_no_require_cuda_switch():
    module = _load_module()

    args = module._build_arg_parser().parse_args(["run", "--dataset-root", "dataset", "--no-require-cuda"])

    assert args.require_cuda is False


def test_noop_baseline_keeps_identity_rotations_and_current_gripper_widths():
    module = _load_module()
    states = np.zeros((1, 20), dtype=np.float32)
    states[0, 6], states[0, 13] = 0.012, 0.034

    baseline = module.build_noop_chunks(states)

    assert baseline.shape == (1, 50, 20)
    np.testing.assert_allclose(baseline[0, 0, 3:9], _rot6d(np.eye(3)))
    np.testing.assert_allclose(baseline[0, -1, 13:19], _rot6d(np.eye(3)))
    np.testing.assert_allclose(baseline[0, :, 9], 0.012)
    np.testing.assert_allclose(baseline[0, :, 19], 0.034)


def test_nearest_reference_uses_quantile_normalized_state_and_stays_in_episode():
    module = _load_module()
    states = np.zeros((3, 20), dtype=np.float32)
    states[:, 0] = [0.0, 0.8, 0.9]
    actions = _identity_actions((3,))
    actions[:, 0] = [10.0, 20.0, 30.0]
    episodes = np.asarray([0, 1, 1], dtype=np.int64)
    frames = np.asarray([5, 3, 4], dtype=np.int64)
    real = np.zeros(20, dtype=np.float32)
    real[0] = 0.88

    match = module.nearest_reference_chunk(
        real, states, actions, episodes, frames, np.zeros(20), np.ones(20)
    )

    assert match["episode_index"] == 1
    assert match["frame_index"] == 4
    assert match["state_distance"] == pytest.approx(0.04, abs=1e-5)
    assert match["gt_chunk"][0, 0] == pytest.approx(30.0)
    assert match["valid"][0]


def test_report_includes_noop_baseline_ratio_and_unpaired_nearest_state_comparison(tmp_path: Path):
    module = _load_module()
    gt = _identity_actions((1, 50))
    gt[0, :, 0] = 0.01
    states = np.zeros((1, 20), dtype=np.float32)
    inputs = {
        "dataset_states": states,
        "dataset_images_camera0": np.zeros((1, 2, 2, 3), dtype=np.uint8),
        "dataset_images_camera1": np.zeros((1, 2, 2, 3), dtype=np.uint8),
        "dataset_gt": gt,
        "dataset_valid": np.ones((1, 50), dtype=bool),
        "dataset_episode_indices": np.asarray([0], dtype=np.int64),
        "reference_states": states,
        "reference_actions": gt[:, 0],
        "reference_episode_indices": np.asarray([0], dtype=np.int64),
        "reference_frame_indices": np.asarray([7], dtype=np.int64),
        "state_q01": np.zeros(20, dtype=np.float32),
        "state_q99": np.ones(20, dtype=np.float32),
        "real_states": states,
        "real_images_camera0": np.zeros((1, 2, 2, 3), dtype=np.uint8),
        "real_images_camera1": np.zeros((1, 2, 2, 3), dtype=np.uint8),
        "real_step_ids": np.asarray([42], dtype=np.int64),
        "metadata_json": {},
    }
    inputs_path = module.write_inputs_archive(tmp_path / "inputs.npz", inputs)
    predictions_path = module.write_predictions_archive(
        tmp_path / "predictions.npz",
        {"dataset_normalized": gt, "dataset_robot": gt, "real_normalized": gt, "real_robot": gt},
        {"complete": True},
    )

    paths = module.write_report(inputs_path, predictions_path, tmp_path)

    summary = json.loads(paths["summary_json"].read_text(encoding="utf-8"))
    assert summary["no_op_baseline_metrics"]["windows"]["20"]["left_translation_l2_mm"] == pytest.approx(10.0)
    assert summary["policy_to_noop_ratio"]["windows"]["20"]["left_translation_l2_mm"] == pytest.approx(0.0)
    real_row = next(csv.DictReader(paths["real_obs_actions_csv"].open(encoding="utf-8")))
    assert real_row["comparison"] == "nearest_state_unpaired"
    assert real_row["paired_gt"] == "False"
    assert real_row["nearest_reference_episode"] == "0"
    assert real_row["nearest_reference_frame"] == "7"
    assert float(real_row["h20_left_translation_l2_mm"]) == pytest.approx(0.0)
