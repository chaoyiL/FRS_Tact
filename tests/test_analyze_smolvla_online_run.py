from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "tools" / "analyze_smolvla_online_run.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("analyze_smolvla_online_run", MODULE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _actions(offset: float = 0.0) -> list[list[float]]:
    values = np.zeros((20, 20), dtype=np.float64)
    values[:, 0] = offset + np.arange(20)
    values[:, 2] = 0.01
    values[:, 9] = np.where(np.arange(20) % 2 == 0, 0.08, 0.10)
    values[:, 19] = 0.11
    return values.tolist()


def _chunk_row(*, obs_seq: int, vla_action=None, frs_action=None) -> dict:
    raw = vla_action if vla_action is not None else frs_action
    return {
        "time": 100.0 + obs_seq,
        "obs_seq": obs_seq,
        "vla_action": vla_action,
        "frs_action": frs_action,
        "prediction_source": "frs",
        "selected_raw_actions": [row[:] for row in (raw if raw is not None else _actions())[:10]],
        "absolute_waypoints": [[0.0, 0.0, -0.4 - 0.01 * index, 0.0, 0.0, 0.0, 0.1] * 2 for index in range(10)],
        "action_timestamps": [101.0 + index for index in range(20)],
    }


def _write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def _write_observation(root: Path, step: int, *, left_x: float, timestamp: float) -> None:
    step_dir = root / f"step_{step:06d}"
    _write_json(step_dir / "robot0_eef_pos.json", [[left_x, 2.0, 3.0]])
    _write_json(step_dir / "robot0_eef_rot_axis_angle.json", [[0.0, 0.0, np.pi / 2]])
    _write_json(step_dir / "robot0_gripper_width.json", [[0.04]])
    _write_json(step_dir / "robot1_eef_pos.json", [[0.0, 1.0, 0.0]])
    _write_json(step_dir / "robot1_eef_rot_axis_angle.json", [[0.0, 0.0, 0.0]])
    _write_json(step_dir / "robot1_gripper_width.json", [[0.05]])
    _write_json(step_dir / "timestamp.json", [timestamp])


def test_chunk_parser_uses_the_sole_non_null_full_action_not_logger_source(tmp_path):
    module = _load_module()
    trace_path = tmp_path / "chunk_trace.jsonl"
    vla = _actions(10.0)
    legacy = _actions(20.0)
    rows = [
        _chunk_row(obs_seq=1, vla_action=vla),
        _chunk_row(obs_seq=2, frs_action=legacy),
    ]
    trace_path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")

    chunks = module.load_chunk_trace(trace_path)

    assert [chunk.obs_seq for chunk in chunks] == [1, 2]
    assert chunks[0].prediction_field == "vla_action"
    assert chunks[1].prediction_field == "frs_action"
    assert chunks[1].prediction_source == "frs"
    assert chunks[0].raw_actions.shape == (20, 20)
    assert chunks[0].selected_actions.shape == (10, 20)
    np.testing.assert_allclose(chunks[1].raw_actions, legacy)


@pytest.mark.parametrize("vla_action, frs_action", [(_actions(), _actions(1.0)), (None, None)])
def test_chunk_parser_rejects_ambiguous_or_missing_full_actions(tmp_path, vla_action, frs_action):
    module = _load_module()
    trace_path = tmp_path / "chunk_trace.jsonl"
    trace_path.write_text(json.dumps(_chunk_row(obs_seq=1, vla_action=vla_action, frs_action=frs_action)))

    with pytest.raises(ValueError, match="exactly one.*vla_action.*frs_action"):
        module.load_chunk_trace(trace_path)


def test_chunk_parser_requires_the_full_twenty_by_twenty_action_shape(tmp_path):
    module = _load_module()
    trace_path = tmp_path / "chunk_trace.jsonl"
    row = _chunk_row(obs_seq=1, vla_action=_actions()[:19])
    trace_path.write_text(json.dumps(row))

    with pytest.raises(ValueError, match=r"shape \(20, 20\)"):
        module.load_chunk_trace(trace_path)


@pytest.mark.parametrize("selected_count", [9, 10])
def test_chunk_parser_requires_exact_first_ten_selected_actions(tmp_path, selected_count):
    module = _load_module()
    trace_path = tmp_path / "chunk_trace.jsonl"
    row = _chunk_row(obs_seq=1, vla_action=_actions())
    row["selected_raw_actions"] = row["selected_raw_actions"][:selected_count]
    if selected_count == 10:
        row["selected_raw_actions"][4][2] = 999.0
    trace_path.write_text(json.dumps(row))

    with pytest.raises(ValueError, match="selected_raw_actions"):
        module.load_chunk_trace(trace_path)


@pytest.mark.parametrize("bad_value", [True, "1.0"])
def test_chunk_parser_rejects_boolean_and_string_action_values(tmp_path, bad_value):
    module = _load_module()
    trace_path = tmp_path / "chunk_trace.jsonl"
    row = _chunk_row(obs_seq=1, vla_action=_actions())
    row["vla_action"][0][0] = bad_value
    row["selected_raw_actions"][0][0] = bad_value
    trace_path.write_text(json.dumps(row))

    with pytest.raises(ValueError, match="finite numeric"):
        module.load_chunk_trace(trace_path)


def test_saved_observations_sort_numeric_steps_and_reconstruct_20d_state(tmp_path):
    module = _load_module()
    saved = tmp_path / "saved"
    _write_observation(saved, 10, left_x=2.0, timestamp=30.0)
    _write_observation(saved, 2, left_x=1.0, timestamp=20.0)

    observations = module.load_saved_observations(saved)
    state = module.reconstruct_state(observations[1], observations[0])

    assert [observation.step for observation in observations] == [2, 10]
    assert [observation.timestamp for observation in observations] == [20.0, 30.0]
    assert state.shape == (20,)
    np.testing.assert_allclose(state[:7], [0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.04], atol=1e-6)
    np.testing.assert_allclose(state[7:14], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.05], atol=1e-6)
    np.testing.assert_allclose(state[14:], [2.0, 1.0, 3.0, 0.0, 0.0, np.pi / 2], atol=1e-6)


def test_action_chain_reports_local_raw_z_absolute_quest_z_and_frame_mismatch(tmp_path):
    module = _load_module()
    trace_path = tmp_path / "chunk_trace.jsonl"
    trace_path.write_text(json.dumps(_chunk_row(obs_seq=1, vla_action=_actions())) + "\n")
    controller_path = tmp_path / "controller_trace.jsonl"
    controller_path.write_text(
        json.dumps(
            {
                "pose_frame": "quest",
                "samples": [
                    {
                        "wall_time": 101.0,
                        "ee_pose_left_z": -0.45,
                        "target_pose_left_z": -0.37,
                    }
                ],
            }
        )
        + "\n"
    )
    saved = tmp_path / "saved"
    _write_observation(saved, 0, left_x=1.0, timestamp=88.5)

    report = module.analyze_action_chain(
        module.load_chunk_trace(trace_path),
        module.load_controller_trace(controller_path),
        module.load_saved_observations(saved),
    )
    chunk = report["chunks"][0]

    assert report["controller_target_frame_mismatch"] is True
    np.testing.assert_allclose(chunk["raw_left_local_z"], np.full(20, 0.01))
    np.testing.assert_allclose(chunk["absolute_left_quest_z"], [-0.4 - 0.01 * index for index in range(10)])
    assert chunk["left_close_count"] == 10
    assert chunk["right_close_count"] == 0
    assert chunk["cumulative_raw_left_local_z"][-1] == pytest.approx(0.2)
    assert chunk["controller_actual_left_quest_z"] == [-0.45]
    assert chunk["controller_target_left_robot_z"] == [-0.37]
    assert chunk["saved_step"] == 0
    assert chunk["saved_timestamp"] == 88.5
    assert chunk["chunk_timestamp"] == 101.0


def test_action_chain_rejects_missing_or_nonconforming_saved_observation_mapping(tmp_path):
    module = _load_module()
    trace_path = tmp_path / "chunk_trace.jsonl"
    trace_path.write_text(json.dumps(_chunk_row(obs_seq=1, vla_action=_actions())) + "\n")
    controller_path = tmp_path / "controller_trace.jsonl"
    controller_path.write_text(
        json.dumps(
            {
                "pose_frame": "quest",
                "samples": [{"wall_time": 101.0, "ee_pose_left_z": -0.45}],
            }
        )
        + "\n"
    )
    saved = tmp_path / "saved"
    _write_observation(saved, 20, left_x=1.0, timestamp=77.0)

    with pytest.raises(ValueError, match="one-to-one"):
        module.analyze_action_chain(
            module.load_chunk_trace(trace_path),
            module.load_controller_trace(controller_path),
            module.load_saved_observations(saved),
        )


def test_action_chain_rejects_duplicate_chunk_observation_sequences(tmp_path):
    module = _load_module()
    trace_path = tmp_path / "chunk_trace.jsonl"
    row = _chunk_row(obs_seq=1, vla_action=_actions())
    trace_path.write_text(json.dumps(row) + "\n" + json.dumps(row) + "\n")
    controller_path = tmp_path / "controller_trace.jsonl"
    controller_path.write_text(
        json.dumps(
            {
                "pose_frame": "quest",
                "samples": [{"wall_time": 101.0, "ee_pose_left_z": -0.45}],
            }
        )
        + "\n"
    )
    saved = tmp_path / "saved"
    _write_observation(saved, 0, left_x=1.0, timestamp=77.0)

    with pytest.raises(ValueError, match="duplicate chunk obs_seq"):
        module.analyze_action_chain(
            module.load_chunk_trace(trace_path),
            module.load_controller_trace(controller_path),
            module.load_saved_observations(saved),
        )


def test_action_chain_preserves_two_argument_interface_without_saved_provenance(tmp_path):
    module = _load_module()
    trace_path = tmp_path / "chunk_trace.jsonl"
    trace_path.write_text(json.dumps(_chunk_row(obs_seq=1, vla_action=_actions())) + "\n")
    controller_path = tmp_path / "controller_trace.jsonl"
    controller_path.write_text(
        json.dumps(
            {
                "pose_frame": "quest",
                "samples": [{"wall_time": 101.0, "ee_pose_left_z": -0.45}],
            }
        )
        + "\n"
    )

    report = module.analyze_action_chain(
        module.load_chunk_trace(trace_path), module.load_controller_trace(controller_path)
    )

    assert report["chunks"][0]["saved_step"] is None
    assert report["chunks"][0]["saved_timestamp"] is None


def test_chunk_parser_rejects_numeric_string_timestamp(tmp_path):
    module = _load_module()
    trace_path = tmp_path / "chunk_trace.jsonl"
    row = _chunk_row(obs_seq=1, vla_action=_actions())
    row["time"] = "101.0"
    trace_path.write_text(json.dumps(row))

    with pytest.raises(ValueError, match="finite number"):
        module.load_chunk_trace(trace_path)
