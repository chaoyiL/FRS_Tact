from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

import eval_pick_tube_rdp_offline as evaluator


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


def _write_rgb(path: Path, rgb: tuple[int, int, int]) -> None:
    image = np.full((12, 16, 3), rgb, dtype=np.uint8)
    assert cv2.imwrite(str(path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))


def _write_snapshot(root: Path, step: int, *, camera0_rgb: tuple[int, int, int]) -> None:
    directory = root / f"step_{step:06d}"
    directory.mkdir()
    _write_json(directory / "timestamp.json", [1000.0 + step])
    _write_json(directory / "robot0_eef_pos.json", [0.1, 0.2, 0.3])
    _write_json(directory / "robot0_eef_rot_axis_angle.json", [0.0, 0.0, 0.0])
    _write_json(directory / "robot1_eef_pos.json", [0.4, 0.5, 0.6])
    _write_json(directory / "robot1_eef_rot_axis_angle.json", [0.0, 0.0, 0.0])
    _write_json(directory / "robot0_gripper_width.json", [0.01])
    _write_json(directory / "robot1_gripper_width.json", [0.02])
    for filename, rgb in {
        "camera0_rgb.jpg": camera0_rgb,
        "camera1_rgb.jpg": (0, 255, 0),
        "camera0_left_tactile.jpg": (0, 0, 255),
        "camera0_right_tactile.jpg": (255, 255, 0),
        "camera1_left_tactile.jpg": (255, 0, 255),
        "camera1_right_tactile.jpg": (0, 255, 255),
    }.items():
        _write_rgb(directory / filename, rgb)


@pytest.fixture
def snapshot_dir(tmp_path: Path) -> Path:
    _write_snapshot(tmp_path, 10, camera0_rgb=(255, 0, 0))
    _write_snapshot(tmp_path, 2, camera0_rgb=(255, 0, 0))
    return tmp_path


def test_load_snapshots_orders_numerically_and_decodes_rgb(snapshot_dir: Path) -> None:
    snapshots = evaluator.load_snapshots(snapshot_dir)

    assert [snapshot.step for snapshot in snapshots] == [2, 10]
    assert snapshots[0].images["observation.images.camera0"].shape == (224, 224, 3)
    np.testing.assert_allclose(
        snapshots[0].images["observation.images.camera0"][0, 0], [255, 0, 0], atol=2
    )


def test_build_server_state_uses_first_snapshot_as_relative_origin(snapshot_dir: Path) -> None:
    snapshots = evaluator.load_snapshots(snapshot_dir)

    state0 = evaluator.build_server_state(
        (snapshots[0].left_pose, snapshots[0].right_pose), snapshots[0]
    )

    np.testing.assert_allclose(state0[:6], 0, atol=1e-6)
    np.testing.assert_allclose(state0[7:13], 0, atol=1e-6)
    assert state0.shape == (20,)
    assert state0[6] == pytest.approx(snapshots[0].left_gripper)
    assert state0[13] == pytest.approx(snapshots[0].right_gripper)


class _FakeRuntime:
    def __init__(self) -> None:
        self.reset_calls = 0
        self.observations: list[dict[str, np.ndarray]] = []

    def reset(self) -> None:
        self.reset_calls += 1

    def predict(self, observation: dict[str, np.ndarray]) -> tuple[np.ndarray, bool]:
        self.observations.append(observation)
        return np.asarray([[0, 0, 0, 1, 0, 0, 0, 1, 0, 0.04]], dtype=np.float32), True


def test_predict_independent_snapshots_resets_each_snapshot(snapshot_dir: Path) -> None:
    snapshots = evaluator.load_snapshots(snapshot_dir)
    runtime = _FakeRuntime()

    results = evaluator.predict_independent_snapshots(
        runtime, evaluator.SINGLE_RIGHT_ARM_7X10, snapshots, seed=7
    )

    assert runtime.reset_calls == len(snapshots)
    assert results["policy_actions"].shape == (len(snapshots), 10)
    assert results["wire_actions"].shape == (len(snapshots), 20)
    np.testing.assert_array_equal(results["wire_actions"][:, :3], 0)
    assert results["states"].shape == (len(snapshots), 20)
    assert results["right_poses"].shape == (len(snapshots), 6)
    assert results["step_ids"].dtype == np.int64
    assert results["timestamps"].dtype == np.float64
    assert results["latency_ms"].dtype == np.float64


def test_write_reports_writes_all_snapshot_artifacts(snapshot_dir: Path, tmp_path: Path) -> None:
    snapshots = evaluator.load_snapshots(snapshot_dir)
    results = evaluator.predict_independent_snapshots(
        _FakeRuntime(), evaluator.SINGLE_RIGHT_ARM_7X10, snapshots, seed=7
    )

    paths = evaluator.write_reports(tmp_path / "reports", results)

    assert set(paths) == {
        "predictions", "trajectory_csv", "summary", "action_overview",
        "snapshot_responses",
    }
    assert all(path.is_file() and path.stat().st_size > 0 for path in paths.values())


def test_predict_independent_snapshots_synchronizes_cuda_before_latency(
    snapshot_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import time
    import torch
    from types import SimpleNamespace

    snapshots = evaluator.load_snapshots(snapshot_dir)[:1]
    runtime = _FakeRuntime()
    runtime.device = SimpleNamespace(type="cuda")
    synchronized: list[object] = []
    clock_reads = 0

    def perf_counter() -> float:
        nonlocal clock_reads
        clock_reads += 1
        if clock_reads == 1:
            return 10.0
        assert synchronized == [runtime.device]
        return 10.125

    monkeypatch.setattr(time, "perf_counter", perf_counter)
    monkeypatch.setattr(torch.cuda, "synchronize", synchronized.append)

    results = evaluator.predict_independent_snapshots(
        runtime, evaluator.SINGLE_RIGHT_ARM_7X10, snapshots, seed=7
    )

    assert clock_reads == 2
    np.testing.assert_allclose(results["latency_ms"], [125.0])


def test_write_reports_renders_decodable_right_snapshot_response_plot(
    snapshot_dir: Path, tmp_path: Path
) -> None:
    snapshots = evaluator.load_snapshots(snapshot_dir)
    results = evaluator.predict_independent_snapshots(
        _FakeRuntime(), evaluator.SINGLE_RIGHT_ARM_7X10, snapshots, seed=7
    )

    paths = evaluator.write_reports(tmp_path / "reports", results)

    response_plot = paths["snapshot_responses"]
    assert response_plot.name == "right_snapshot_responses.png"
    decoded = cv2.imread(str(response_plot), cv2.IMREAD_COLOR)
    assert decoded is not None and decoded.size > 0


def test_write_reports_includes_snapshot_action_metrics(
    snapshot_dir: Path, tmp_path: Path
) -> None:
    snapshots = evaluator.load_snapshots(snapshot_dir)
    results = evaluator.predict_independent_snapshots(
        _FakeRuntime(), evaluator.SINGLE_RIGHT_ARM_7X10, snapshots, seed=7
    )

    paths = evaluator.write_reports(tmp_path / "reports", results)

    header = paths["trajectory_csv"].read_text(encoding="utf-8").splitlines()[0].split(",")
    assert {"translation_norm", "rotation_angle", "gripper_command", "recorded_right_gripper", "latency_ms"} <= set(header)
    summary = json.loads(paths["summary"].read_text(encoding="utf-8"))
    assert summary["state"] == {"finite": True, "shape": [len(snapshots), 20]}
    for name in ("translation_norm", "rotation_angle", "gripper_delta"):
        assert set(summary[name]) == {"min", "max", "mean"}
        assert all(np.isfinite(value) for value in summary[name].values())
    assert set(summary["latency_ms"]) == {"min", "max", "mean", "p95"}


def test_write_reports_preserves_deterministic_response_metrics(tmp_path: Path) -> None:
    import csv

    states = np.zeros((2, 20), dtype=np.float32)
    states[:, 13] = [0.1, 0.2]
    right_poses = np.asarray(
        [[1, 2, 3, 0, 0, np.pi / 2], [2, 2, 3, 0, 0, np.pi / 2]], dtype=np.float32
    )
    policy_actions = np.asarray(
        [[1, 0, 0, 0, 1, 0, -1, 0, 0, 0.3], [1, 0, 0, 0, 1, 0, -1, 0, 0, 0.4]],
        dtype=np.float32,
    )
    results = {
        "states": states,
        "policy_actions": policy_actions,
        "wire_actions": np.zeros((2, 20), dtype=np.float32),
        "right_poses": right_poses,
        "step_ids": np.asarray([2, 10], dtype=np.int64),
        "timestamps": np.asarray([1.0, 2.0], dtype=np.float64),
        "latency_ms": np.asarray([1.0, 5.0], dtype=np.float64),
    }

    starts, endpoints = evaluator.right_snapshot_response_points(
        right_poses, policy_actions[:, :3]
    )
    np.testing.assert_allclose(starts[0], [1, 2, 3])
    np.testing.assert_allclose(endpoints[0], [1, 3, 3], atol=1e-6)

    paths = evaluator.write_reports(tmp_path / "reports", results)
    with paths["trajectory_csv"].open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    assert "gripper_delta" in rows[0]
    assert float(rows[0]["gripper_delta"]) == pytest.approx(0.2)
    summary = json.loads(paths["summary"].read_text(encoding="utf-8"))
    assert summary["rotation_angle"]["mean"] == pytest.approx(np.pi / 2)
    assert summary["latency_ms"] == {
        "min": 1.0, "mean": 3.0, "p95": 4.8, "max": 5.0,
    }
