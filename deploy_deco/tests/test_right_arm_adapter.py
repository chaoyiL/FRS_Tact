import numpy as np
import pytest

from deploy_deco.right_arm_adapter import expand_right_action, project_right_observation


def observation() -> dict:
    return {
        "observation.state": np.arange(20, dtype=np.float32),
        "observation.images.camera0": np.zeros((2, 3, 3), dtype=np.uint8),
        "observation.images.camera1": np.zeros((2, 3, 3), dtype=np.uint8),
    }


def test_project_right_observation_uses_indices_7_through_13():
    source = observation()
    projected = project_right_observation(source)
    np.testing.assert_array_equal(projected["observation.state"], np.arange(7, 14))
    assert projected["observation.state"] is not source["observation.state"]


def test_expand_right_action_holds_left_and_preserves_right():
    source = observation()
    right = np.arange(20, dtype=np.float32).reshape(2, 10)
    wire = expand_right_action(right, source)
    np.testing.assert_array_equal(wire[:, 10:], right)
    np.testing.assert_array_equal(wire[:, :3], 0.0)
    np.testing.assert_array_equal(
        wire[:, 3:9], np.array([[1, 0, 0, 0, 1, 0]] * 2, dtype=np.float32)
    )
    np.testing.assert_array_equal(wire[:, 9], 6.0)


def test_project_rejects_wrong_server_state_shape():
    source = observation()
    source["observation.state"] = np.zeros(7, dtype=np.float32)
    with pytest.raises(ValueError, match=r"shape \(20,\)"):
        project_right_observation(source)


def test_expand_rejects_wrong_action_width():
    with pytest.raises(ValueError, match=r"\(H,10\)"):
        expand_right_action(np.zeros((32, 20), dtype=np.float32), observation())


def test_expand_rejects_nonfinite_action():
    action = np.zeros((32, 10), dtype=np.float32)
    action[0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        expand_right_action(action, observation())
