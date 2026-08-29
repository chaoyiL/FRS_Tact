from __future__ import annotations

import numpy as np
import pytest


def _server_observation() -> dict[str, np.ndarray]:
    return {
        "observation.state": np.arange(20, dtype=np.float32),
        "observation.images.camera0": np.full((3, 4, 3), 255, dtype=np.uint8),
        "observation.images.camera1": np.full((3, 4, 3), 17, dtype=np.uint8),
    }


def test_project_right_observation_uses_right_state_without_mutating_images() -> None:
    from deploy_pi05.right_arm_adapter import project_right_observation

    source = _server_observation()
    projected = project_right_observation(source)

    np.testing.assert_array_equal(projected["observation.state"], np.arange(7, 14))
    np.testing.assert_array_equal(
        projected["observation.images.camera0"],
        source["observation.images.camera0"],
    )
    np.testing.assert_array_equal(
        projected["observation.images.camera1"],
        source["observation.images.camera1"],
    )
    np.testing.assert_array_equal(source["observation.images.camera0"], 255)


def test_expand_right_action_holds_left_and_places_model_action_on_right() -> None:
    from deploy_pi05.right_arm_adapter import expand_right_action

    source = _server_observation()
    right = np.arange(20, dtype=np.float32).reshape(2, 10)

    wire = expand_right_action(right, source)

    assert wire.shape == (2, 20)
    np.testing.assert_array_equal(wire[:, 10:], right)
    np.testing.assert_array_equal(wire[:, :3], 0.0)
    np.testing.assert_array_equal(
        wire[:, 3:9],
        np.array([[1, 0, 0, 0, 1, 0]] * 2, dtype=np.float32),
    )
    np.testing.assert_array_equal(wire[:, 9], source["observation.state"][6])


@pytest.mark.parametrize(
    ("state", "message"),
    [
        (np.zeros(7, dtype=np.float32), r"shape \(20,\)"),
        (np.full(20, np.nan, dtype=np.float32), "finite"),
    ],
)
def test_project_right_observation_rejects_invalid_server_state(
    state: np.ndarray, message: str
) -> None:
    from deploy_pi05.right_arm_adapter import project_right_observation

    source = _server_observation()
    source["observation.state"] = state

    with pytest.raises(ValueError, match=message):
        project_right_observation(source)


def test_expand_right_action_rejects_invalid_model_action() -> None:
    from deploy_pi05.right_arm_adapter import expand_right_action

    action = np.zeros((50, 10), dtype=np.float32)
    action[0, 0] = np.nan
    with pytest.raises(ValueError, match=r"finite with shape \(H,10\)"):
        expand_right_action(action, _server_observation())


def test_plain_pi05_loop_projects_right_input_and_expands_wire_action() -> None:
    from types import SimpleNamespace

    from deploy_pi05.pi05_client import run_legacy_loop

    raw = _server_observation()
    sent: list[tuple[np.ndarray, int]] = []

    class Bridge:
        @staticmethod
        def receive_observation(*, timeout: float):
            assert timeout == 1.5
            return 42, raw

        @staticmethod
        def send_action(action: np.ndarray, obs_seq: int) -> None:
            sent.append((action, obs_seq))

    class Policy:
        config = SimpleNamespace(
            state_action_profile="single-right-arm-7x10",
            state_dim=7,
            action_horizon=2,
            action_dim=10,
            robot_action_dim=10,
        )
        expected = np.arange(20, dtype=np.float32).reshape(2, 10)

        @staticmethod
        def predict_action_chunk(observation, task: str, *, seed: int, num_steps: int):
            np.testing.assert_array_equal(observation["observation.state"], np.arange(7, 14))
            assert "observation.images.camera0" not in observation
            np.testing.assert_array_equal(
                observation["observation.images.camera1"],
                raw["observation.images.camera1"],
            )
            assert task == "insert"
            assert (seed, num_steps) == (3, 4)
            return Policy.expected[None]

        @staticmethod
        def unnormalize_actions(action: np.ndarray) -> np.ndarray:
            return action

    run_legacy_loop(
        Bridge(),
        Policy(),
        task="insert",
        image_keys=("observation.images.camera1",),
        observation_timeout_s=1.5,
        seed=3,
        sample_steps=4,
        max_iterations=1,
        saver=None,
    )

    assert sent[0][1] == 42
    np.testing.assert_array_equal(sent[0][0][:, 10:], Policy.expected)
    np.testing.assert_array_equal(sent[0][0][:, 9], raw["observation.state"][6])
