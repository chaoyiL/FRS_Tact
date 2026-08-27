from types import SimpleNamespace
from pathlib import Path

import numpy as np
import pytest
import yaml
import deploy_pi05.frs_runtime as frs_runtime_module

from deploy_pi05.frs_config import (
    GripperHysteresisConfig,
    Task1MotionGainConfig,
    parse_gripper_hysteresis_config,
    parse_task1_motion_gain_config,
    parse_task_switch,
    validate_frs_config_section,
)
from deploy_pi05.deployment import print_startup_summary
from deploy_pi05.frs_runtime import (
    FRSRuntime,
    _apply_task1_motion_gain,
    _select_latched_vla_translation,
)


def _config() -> GripperHysteresisConfig:
    return GripperHysteresisConfig(
        left_close_threshold=0.08,
        left_reopen_threshold=0.10,
        left_closed_command=0.01,
        right_close_threshold=0.09,
        right_reopen_threshold=0.10,
        right_closed_command=0.01,
    )


def _rows() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    selected = np.arange(20, dtype=np.float32) + 100.0
    vla_normalized = np.arange(20, dtype=np.float32) + 200.0
    vla_action = np.zeros(20, dtype=np.float32)
    return selected, vla_normalized, vla_action


def test_shared_gripper_config_parses_exact_hysteresis_contract() -> None:
    parsed = parse_gripper_hysteresis_config(
        {
            "gripper": {
                "left_close_threshold": 0.08,
                "left_reopen_threshold": 0.10,
                "left_closed_command": 0.01,
                "right_close_threshold": 0.09,
                "right_reopen_threshold": 0.10,
                "right_closed_command": 0.01,
            }
        }
    )

    assert parsed == _config()


@pytest.mark.parametrize(
    ("raw", "expected"),
    [({}, 0), ({"task": 0}, 0), ({"task": 1}, 1)],
)
def test_task_switch_defaults_off_and_accepts_only_task1(raw: dict, expected: int) -> None:
    assert parse_task_switch(raw) == expected


@pytest.mark.parametrize("value", [-1, 2, True, "1"])
def test_task_switch_rejects_invalid_values(value: object) -> None:
    with pytest.raises(ValueError, match="task must be 0 or 1"):
        parse_task_switch({"task": value})


def test_task1_motion_gains_parse_from_shared_yaml() -> None:
    parsed = parse_task1_motion_gain_config(
        {
            "task": 1,
            "task1": {
                "approach_translation_gain": 1.2,
                "translation_gain": 1.5,
                "rotation_gain": 1.0,
            },
        }
    )

    assert parsed == Task1MotionGainConfig(
        approach_translation_gain=1.2,
        translation_gain=1.5,
        rotation_gain=1.0,
    )


@pytest.mark.parametrize(
    "field", ["approach_translation_gain", "translation_gain", "rotation_gain"]
)
@pytest.mark.parametrize(
    "value", [0.0, -1.0, float("nan"), float("inf"), True, "1.5", 3.1]
)
def test_task1_motion_gains_reject_invalid_values(field: str, value: object) -> None:
    raw = {
        "approach_translation_gain": 1.2,
        "translation_gain": 1.5,
        "rotation_gain": 1.0,
    }
    raw[field] = value

    with pytest.raises(ValueError, match=field):
        parse_task1_motion_gain_config({"task": 1, "task1": raw})


def test_task0_motion_gains_are_identity() -> None:
    assert parse_task1_motion_gain_config({"task": 0}) == Task1MotionGainConfig(
        approach_translation_gain=1.0,
        translation_gain=1.0,
        rotation_gain=1.0,
    )


def test_task1_client_validation_requires_exactly_80_hz_controller_frequency() -> None:
    config_path = Path(__file__).parents[1] / "deploy_pi05/configs/deploy_pi05_frs.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config["control"]["controller_frequency"] = 79.9

    with pytest.raises(
        ValueError, match="Task 1 control.controller_frequency must be 80.0"
    ):
        validate_frs_config_section(config)


def test_task1_runtime_requires_explicit_motion_gain() -> None:
    with pytest.raises(ValueError, match="task1_motion_gain is required when task is 1"):
        FRSRuntime(
            {},
            config_path=Path("deploy.yaml"),
            policy=SimpleNamespace(),
            source_sample_steps=10,
            gripper_hysteresis=_config(),
            task=1,
        )


def test_task1_runtime_accepts_explicit_motion_gain_before_loading(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    motion_gain = Task1MotionGainConfig(
        approach_translation_gain=1.2,
        translation_gain=1.5,
        rotation_gain=1.0,
    )

    def stop_after_gain_gate(*_args, **_kwargs):
        raise RuntimeError("runtime loading reached")

    monkeypatch.setattr(frs_runtime_module, "parse_frs_config", stop_after_gain_gate)

    with pytest.raises(RuntimeError, match="runtime loading reached"):
        FRSRuntime(
            {},
            config_path=Path("deploy.yaml"),
            policy=SimpleNamespace(),
            source_sample_steps=10,
            gripper_hysteresis=_config(),
            task=1,
            task1_motion_gain=motion_gain,
        )


@pytest.mark.parametrize(
    ("latched", "left_gain", "right_gain"),
    [
        pytest.param((True, False), 1.5, 1.2, id="left-latched"),
        pytest.param((False, False), 1.2, 1.2, id="both-open"),
        pytest.param((True, True), 1.5, 1.5, id="both-closed"),
        pytest.param((False, True), 1.2, 1.5, id="left-reopened"),
    ],
)
def test_task1_motion_gain_uses_each_arm_latch_phase(
    latched: tuple[bool, bool], left_gain: float, right_gain: float
) -> None:
    action = np.arange(20, dtype=np.float32)

    scaled = _apply_task1_motion_gain(
        action,
        latched=latched,
        config=Task1MotionGainConfig(
            approach_translation_gain=1.2,
            translation_gain=1.5,
            rotation_gain=1.0,
        ),
    )

    np.testing.assert_allclose(scaled[0:3], action[0:3] * left_gain)
    np.testing.assert_allclose(scaled[10:13], action[10:13] * right_gain)
    np.testing.assert_array_equal(scaled[3:9], action[3:9])
    np.testing.assert_array_equal(scaled[13:19], action[13:19])
    np.testing.assert_array_equal(scaled[[9, 19]], action[[9, 19]])


def test_task1_startup_summary_advertises_shared_motion_contract(capsys) -> None:
    config = {
        "task": 1,
        "connection": {"address": "127.0.0.1", "port": 26421},
        "control": {
            "control_frequency": 10.0,
            "controller_frequency": 80.0,
            "dispatch_lead_time_s": 0.04,
        },
        "task1": {
            "approach_translation_gain": 1.2,
            "translation_gain": 1.5,
            "rotation_gain": 1.0,
            "left_min_lift_height_m": 0.10,
            "right_preclose_forward_m": 0.005,
        },
        "frs": {
            "checkpoint": "/tmp/frs-checkpoint",
            "tactile_encoder_checkpoint": "/tmp/tactile-encoder",
            "tactile_keys": ["left", "right"],
        },
    }
    policy_config = SimpleNamespace(
        assets_dir="/tmp/assets",
        asset_id="two-tubes",
        camera_map={},
        checkpoint="/tmp/pi05-checkpoint",
        state_dim=20,
        action_dim=20,
        robot_action_dim=20,
        action_horizon=50,
    )

    print_startup_summary(
        config, policy_config, mode="frs", backend="cpu", devices=()
    )

    output = capsys.readouterr().out
    assert "dispatch_lead_s=0.04" in output
    assert "approach_gain=1.2" in output
    assert "translation_gain=1.5" in output
    assert "left_min_lift_m=0.1" in output
    assert "right_preclose_forward_m=0.005" in output


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("left_close_threshold", float("nan")),
        ("left_reopen_threshold", float("inf")),
        ("right_close_threshold", True),
        ("right_reopen_threshold", "0.10"),
        ("left_closed_command", 0.009),
        ("right_closed_command", 0.041),
    ],
)
def test_shared_gripper_config_rejects_invalid_values(field: str, value: object) -> None:
    raw = dict(vars(SimpleNamespace(**_config().__dict__)))
    raw[field] = value

    with pytest.raises(ValueError, match=field):
        parse_gripper_hysteresis_config({"gripper": raw})


@pytest.mark.parametrize("side", ["left", "right"])
def test_shared_gripper_config_requires_close_below_reopen(side: str) -> None:
    raw = dict(_config().__dict__)
    raw[f"{side}_close_threshold"] = raw[f"{side}_reopen_threshold"]

    with pytest.raises(ValueError, match=f"{side}_close_threshold"):
        parse_gripper_hysteresis_config({"gripper": raw})


def test_left_latch_uses_vla_translation_but_preserves_frs_rotation_and_right_arm() -> None:
    selected, vla_normalized, vla_action = _rows()
    vla_action[9] = 0.08
    vla_action[19] = 0.11

    protected, latched = _select_latched_vla_translation(
        selected,
        vla_normalized,
        vla_action,
        (False, False),
        _config(),
    )

    assert latched == (True, False)
    np.testing.assert_array_equal(protected[0:3], vla_normalized[0:3])
    np.testing.assert_array_equal(protected[3:10], selected[3:10])
    np.testing.assert_array_equal(protected[10:20], selected[10:20])


def test_latch_deadband_persists_across_calls_until_reopen_threshold() -> None:
    selected, vla_normalized, vla_action = _rows()
    vla_action[[9, 19]] = (0.09, 0.095)

    held, latched = _select_latched_vla_translation(
        selected,
        vla_normalized,
        vla_action,
        (True, True),
        _config(),
    )

    assert latched == (True, True)
    np.testing.assert_array_equal(held[0:3], vla_normalized[0:3])
    np.testing.assert_array_equal(held[10:13], vla_normalized[10:13])
    np.testing.assert_array_equal(held[3:10], selected[3:10])
    np.testing.assert_array_equal(held[13:20], selected[13:20])

    vla_action[[9, 19]] = (0.10, 0.10)
    reopened, latched = _select_latched_vla_translation(
        selected,
        vla_normalized,
        vla_action,
        latched,
        _config(),
    )

    assert latched == (False, False)
    np.testing.assert_array_equal(reopened, selected)


def test_right_arm_can_latch_independently() -> None:
    selected, vla_normalized, vla_action = _rows()
    vla_action[[9, 19]] = (0.11, 0.09)

    protected, latched = _select_latched_vla_translation(
        selected,
        vla_normalized,
        vla_action,
        (False, False),
        _config(),
    )

    assert latched == (False, True)
    np.testing.assert_array_equal(protected[0:10], selected[0:10])
    np.testing.assert_array_equal(protected[10:13], vla_normalized[10:13])
    np.testing.assert_array_equal(protected[13:20], selected[13:20])


def test_episode_reset_clears_latches_while_chunk_clear_preserves_them() -> None:
    runtime = object.__new__(FRSRuntime)
    runtime._vla_translation_latched = (True, True)
    runtime._clear_chunk_state()
    assert runtime._vla_translation_latched == (True, True)

    runtime._encode_observation = lambda _observation: np.zeros((4, 1), dtype=np.float32)
    runtime.history = SimpleNamespace(window=2, stride=1, token_shape=(4, 1))
    runtime.reset_episode({})

    assert runtime._vla_translation_latched == (False, False)
