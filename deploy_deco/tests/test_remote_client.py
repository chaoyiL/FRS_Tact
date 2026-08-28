from __future__ import annotations

from pathlib import Path

import numpy as np

from deploy_deco import bridge_client, policy, remote_client


def test_bounded_loop_waits_for_post_action_observation_before_stop(monkeypatch) -> None:
    events: list[tuple[str, int | str] | str] = []
    observations = iter(
        [
            (0, {"frame": "warmup"}),
            (1, {"frame": "action-input"}),
            (2, {"frame": "post-action"}),
        ]
    )
    config = {
        "checkpoint": "/tmp/deco.ts",
        "device": "cuda:0",
        "connection": {
            "address": "127.0.0.1",
            "port": 26421,
            "observation_timeout_s": 1.25,
            "require_token": False,
        },
        "observation": {},
        "control": {},
        "runtime": {"auto_start": True, "warmup_runs": 0},
    }

    class ProtocolFaithfulBridge:
        def __init__(self, **kwargs) -> None:
            pass

        def send_config(self, server_config: dict) -> None:
            pass

        def receive_observation(self, timeout: float | None = None):
            obs_seq, observation = next(observations)
            events.append(("observation", obs_seq))
            assert timeout == 1.25
            return obs_seq, observation

        def send_state(self, state: str) -> None:
            events.append(("state", state))

        def send_action(self, action: np.ndarray, obs_seq: int) -> None:
            events.append(("action", obs_seq))

        def close(self) -> None:
            events.append("close")

    class FakePolicy:
        state_dim = 20
        action_dim = 20
        action_horizon = 32
        expected_sample_hz = 30.0

        def __init__(self, checkpoint: Path, *, device: str, verify_hash: bool) -> None:
            pass

        def predict(self, observation: dict, *, seed: int) -> np.ndarray:
            return np.zeros((32, 20), dtype=np.float32)

    monkeypatch.setattr(remote_client, "check", lambda _path: config)
    monkeypatch.setattr(remote_client, "resolve_checkpoint", lambda _config: Path("/tmp/deco.ts"))
    monkeypatch.setattr(remote_client, "make_server_config", lambda _config: {})
    monkeypatch.setattr(bridge_client, "RobotBridgeClient", ProtocolFaithfulBridge)
    monkeypatch.setattr(policy, "DECOPolicy", FakePolicy)

    remote_client.run(Path("unused.yaml"), max_iterations_override=1)

    assert events == [
        ("observation", 0),
        ("state", "start"),
        ("observation", 1),
        ("action", 1),
        ("observation", 2),
        ("state", "stop"),
        "close",
    ]


def test_legacy_loop_reads_next_observation_without_waiting_for_action_ack(
    monkeypatch,
) -> None:
    events: list[object] = []
    observations = iter(
        [
            (0, {"frame": "warmup"}),
            (1, {"frame": "first"}),
            (2, {"frame": "second"}),
            (3, {"frame": "post-action"}),
        ]
    )
    config = {
        "checkpoint": "/tmp/deco.ts",
        "device": "cuda:0",
        "seed": 7,
        "connection": {
            "address": "127.0.0.1",
            "port": 26421,
            "observation_timeout_s": 1.25,
            "require_token": False,
        },
        "observation": {},
        "control": {},
        "runtime": {"auto_start": True, "warmup_runs": 0},
    }

    class FakeBridge:
        def __init__(self, **kwargs) -> None:
            events.append(("connect", kwargs["address"], kwargs["port"]))

        def send_config(self, server_config: dict) -> None:
            events.append(("config", server_config))

        def receive_observation(self, timeout: float | None = None):
            obs_seq, observation = next(observations)
            events.append(("observation", obs_seq, timeout))
            return obs_seq, observation

        def send_state(self, state: str) -> None:
            events.append(("state", state))

        def send_action(self, action: np.ndarray, obs_seq: int) -> None:
            events.append(("action", obs_seq, action.shape))

        def receive_action_ack(self, obs_seq: int, timeout: float) -> None:
            raise AssertionError("legacy DECO must not wait for a generic action_ack")

        def close(self) -> None:
            events.append("close")

    class FakePolicy:
        state_dim = 20
        action_dim = 20
        action_horizon = 32
        expected_sample_hz = 30.0

        def __init__(self, checkpoint: Path, *, device: str, verify_hash: bool) -> None:
            events.append(("policy", checkpoint, device, verify_hash))

        def predict(self, observation: dict, *, seed: int) -> np.ndarray:
            events.append(("predict", observation["frame"], seed))
            return np.zeros((32, 20), dtype=np.float32)

    monkeypatch.setattr(remote_client, "check", lambda _path: config)
    monkeypatch.setattr(remote_client, "resolve_checkpoint", lambda _config: Path("/tmp/deco.ts"))
    monkeypatch.setattr(remote_client, "make_server_config", lambda _config: {"legacy": True})
    monkeypatch.setattr(bridge_client, "RobotBridgeClient", FakeBridge)
    monkeypatch.setattr(policy, "DECOPolicy", FakePolicy)

    remote_client.run(Path("unused.yaml"), max_iterations_override=2)

    assert [(event[0], event[1]) for event in events if isinstance(event, tuple) and event[0] == "action"] == [
        ("action", 1),
        ("action", 2),
    ]


def test_right_arm_loop_projects_state_and_sends_bimanual_action(monkeypatch) -> None:
    predicted_states: list[np.ndarray] = []
    predicted_camera0: list[np.ndarray] = []
    predicted_camera1: list[np.ndarray] = []
    sent_actions: list[tuple[np.ndarray, int]] = []
    camera0 = np.full((4, 5, 3), 255, dtype=np.uint8)
    camera1 = np.full((4, 5, 3), 17, dtype=np.uint8)
    observations = iter(
        [
            (
                index,
                {
                    "observation.state": np.arange(20, dtype=np.float32),
                    "observation.images.camera0": camera0.copy(),
                    "observation.images.camera1": camera1.copy(),
                },
            )
            for index in range(3)
        ]
    )
    config = {
        "checkpoint": "/tmp/deco.ts",
        "device": "cuda:0",
        "seed": 7,
        "model": {"state_action_profile": "single-right-arm-7x10"},
        "connection": {
            "address": "127.0.0.1",
            "port": 26421,
            "observation_timeout_s": 1.25,
            "require_token": False,
        },
        "observation": {
            "single_arm_mode": True,
            "controlled_arm": "right",
            "black_camera0": True,
        },
        "control": {},
        "runtime": {"auto_start": True, "warmup_runs": 1},
    }
    right_action = np.tile(np.arange(10, dtype=np.float32), (32, 1))

    class FakeBridge:
        def __init__(self, **kwargs) -> None:
            pass

        def send_config(self, server_config: dict) -> None:
            pass

        def receive_observation(self, timeout: float | None = None):
            return next(observations)

        def send_state(self, state: str) -> None:
            pass

        def send_action(self, action: np.ndarray, obs_seq: int) -> None:
            sent_actions.append((action.copy(), obs_seq))

        def close(self) -> None:
            pass

    class FakeRightPolicy:
        state_dim = 7
        action_dim = 10
        action_horizon = 32
        expected_sample_hz = 30.0

        def __init__(self, checkpoint, *, device, verify_hash):
            pass

        def predict(self, observation, *, seed):
            predicted_states.append(observation["observation.state"].copy())
            predicted_camera0.append(observation["observation.images.camera0"].copy())
            predicted_camera1.append(observation["observation.images.camera1"].copy())
            return right_action.copy()

    monkeypatch.setattr(remote_client, "check", lambda _path: config)
    monkeypatch.setattr(remote_client, "resolve_checkpoint", lambda _config: Path("/tmp/deco.ts"))
    monkeypatch.setattr(remote_client, "make_server_config", lambda _config: {"right": True})
    monkeypatch.setattr(bridge_client, "RobotBridgeClient", FakeBridge)
    monkeypatch.setattr(policy, "DECOPolicy", FakeRightPolicy)

    remote_client.run(Path("unused.yaml"), max_iterations_override=1)

    sent_action, obs_seq = sent_actions[0]
    assert len(predicted_states) == 2
    for predicted_state in predicted_states:
        np.testing.assert_array_equal(predicted_state, np.arange(7, 14, dtype=np.float32))
    for image in predicted_camera0:
        np.testing.assert_array_equal(image, np.zeros_like(camera0))
    for image in predicted_camera1:
        np.testing.assert_array_equal(image, camera1)
    assert obs_seq == 1
    assert sent_action.shape == (32, 20)
    np.testing.assert_array_equal(sent_action[:, 10:], right_action)
    np.testing.assert_array_equal(sent_action[:, [3, 7]], 1.0)
    np.testing.assert_array_equal(sent_action[:, 9], 6.0)
