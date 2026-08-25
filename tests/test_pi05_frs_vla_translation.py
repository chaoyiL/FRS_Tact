from types import SimpleNamespace

import numpy as np
import pytest

from deploy_pi05.frs_config import (
    GripperHysteresisConfig,
    parse_gripper_hysteresis_config,
)
from deploy_pi05.frs_runtime import FRSRuntime, _select_latched_vla_translation


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
