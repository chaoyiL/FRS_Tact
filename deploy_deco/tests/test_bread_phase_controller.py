import numpy as np
import pytest

from deploy_deco.bread_phase_controller import BreadPhaseController, BreadPhaseTimeout


IDENTITY_6D = np.array([1, 0, 0, 0, 1, 0], dtype=np.float32)


def _state(right_gripper: float = 0.12, left_gripper: float = 0.31) -> np.ndarray:
    state = np.zeros(20, dtype=np.float32)
    state[6] = left_gripper
    state[13] = right_gripper
    return state


def _action() -> np.ndarray:
    return np.arange(20, dtype=np.float32) / 10.0


def test_phase0_masks_left_arm_and_keeps_current_left_gripper():
    controller = BreadPhaseController(timeout_s=15.0)
    state = _state()

    filtered = controller.apply(state, _action(), now_s=0.0)

    np.testing.assert_array_equal(filtered[0:3], np.zeros(3, dtype=np.float32))
    np.testing.assert_array_equal(filtered[3:9], IDENTITY_6D)
    assert filtered[9] == pytest.approx(state[6])
    np.testing.assert_array_equal(filtered[10:20], _action()[10:20])
    assert controller.phase == 0


def test_phase1_masks_right_arm_and_uses_open_right_gripper():
    controller = BreadPhaseController(timeout_s=15.0)
    controller.apply(_state(right_gripper=0.08), _action(), now_s=0.0)
    controller.apply(_state(right_gripper=0.10), _action(), now_s=0.1)
    filtered = controller.apply(_state(right_gripper=0.11), _action(), now_s=0.2)

    assert controller.phase == 1
    np.testing.assert_array_equal(filtered[0:10], _action()[0:10])
    np.testing.assert_array_equal(filtered[10:13], np.zeros(3, dtype=np.float32))
    np.testing.assert_array_equal(filtered[13:19], IDENTITY_6D)
    assert filtered[19] == pytest.approx(0.12)


def test_phase0_requires_close_then_two_consecutive_open_observations():
    controller = BreadPhaseController(timeout_s=15.0)
    controller.apply(_state(right_gripper=0.11), _action(), now_s=0.0)
    controller.apply(_state(right_gripper=0.08), _action(), now_s=0.1)
    controller.apply(_state(right_gripper=0.10), _action(), now_s=0.2)
    assert controller.phase == 0

    controller.apply(_state(right_gripper=0.09), _action(), now_s=0.3)
    controller.apply(_state(right_gripper=0.10), _action(), now_s=0.4)
    assert controller.phase == 0

    controller.apply(_state(right_gripper=0.11), _action(), now_s=0.5)
    assert controller.phase == 1


def test_phase0_timeout_stops_without_advancing_phase():
    controller = BreadPhaseController(timeout_s=15.0)
    controller.apply(_state(right_gripper=0.08), _action(), now_s=0.0)

    with pytest.raises(BreadPhaseTimeout):
        controller.apply(_state(right_gripper=0.12), _action(), now_s=15.1)
    assert controller.phase == 0
