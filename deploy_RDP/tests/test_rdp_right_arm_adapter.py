import numpy as np
import pytest

from reactive_diffusion_policy.deploy.right_arm_adapter import (
    expand_right_action,
    project_right_state,
)


def test_project_right_state_uses_right_slice():
    state = np.arange(20, dtype=np.float32)
    projected = project_right_state({"observation.state": state})
    np.testing.assert_array_equal(projected, state[7:14])
    assert not np.shares_memory(projected, state)


@pytest.mark.parametrize(
    "state",
    [
        np.arange(19, dtype=np.float32),
        np.arange(21, dtype=np.float32),
        np.array([np.nan] + list(range(19)), dtype=np.float32),
        np.array([np.inf] + list(range(19)), dtype=np.float32),
    ],
)
def test_project_right_state_rejects_invalid_bridge_state(state):
    with pytest.raises(ValueError):
        project_right_state({"observation.state": state})


def test_expand_right_action_builds_bimanual_wire_action():
    state = np.arange(20, dtype=np.float32)
    right = np.arange(20, dtype=np.float32).reshape(2, 10)
    result = expand_right_action(right, {"observation.state": state})
    assert result.shape == (2, 20)
    np.testing.assert_array_equal(result[:, 10:], right)
    np.testing.assert_array_equal(result[:, :3], 0)
    np.testing.assert_array_equal(result[:, 3:9], [[1, 0, 0, 0, 1, 0]] * 2)
    np.testing.assert_array_equal(result[:, 9], state[6])


def test_expand_right_action_requires_horizon_by_ten_action():
    state = np.arange(20, dtype=np.float32)
    with pytest.raises(ValueError):
        expand_right_action(np.zeros((10,), dtype=np.float32), {"observation.state": state})


def test_expand_right_action_rejects_nonfinite_action():
    state = np.arange(20, dtype=np.float32)
    right = np.zeros((1, 10), dtype=np.float32)
    right[0, 0] = np.nan

    with pytest.raises(ValueError, match="finite"):
        expand_right_action(right, {"observation.state": state})
