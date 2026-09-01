from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from deploy_deco import bridge_client, policy, remote_client
from deploy_deco.artifact import TACTILE_FIELD_ORDER


def test_right_gripper_bias_shifts_only_column_19_and_clips() -> None:
    action = np.zeros((2, 20), dtype=np.float32)
    action[:, 9] = [0.08, 0.09]
    action[:, 19] = [0.10, 0.119]
    original = action.copy()

    adjusted = remote_client.apply_right_gripper_bias(action, 0.005)

    np.testing.assert_allclose(adjusted[:, 19], [0.105, 0.1208])
    np.testing.assert_array_equal(adjusted[:, 9], original[:, 9])
    np.testing.assert_array_equal(action, original)


def test_zero_right_gripper_bias_returns_an_equal_copy() -> None:
    action = np.arange(40, dtype=np.float32).reshape(2, 20)

    adjusted = remote_client.apply_right_gripper_bias(action, 0.0)

    np.testing.assert_array_equal(adjusted, action)
    assert adjusted is not action


def test_gripper_biases_shift_left_and_right_independently() -> None:
    action = np.zeros((2, 20), dtype=np.float32)
    action[:, 9] = [0.08, 0.07]
    action[:, 19] = [0.10, 0.12]
    original = action.copy()

    adjusted = remote_client.apply_gripper_biases(action, -0.005, -0.005)

    np.testing.assert_allclose(adjusted[:, 9], [0.075, 0.0677])
    np.testing.assert_allclose(adjusted[:, 19], [0.095, 0.115])
    np.testing.assert_array_equal(adjusted[:, :9], original[:, :9])
    np.testing.assert_array_equal(action, original)


def test_bounded_loop_waits_for_post_action_observation_before_stop(monkeypatch) -> None:
    events: list[tuple[str, int | str] | str] = []
    sent_actions: list[np.ndarray] = []
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
        "control": {
            "left_gripper_bias_m": -0.005,
            "right_gripper_bias_m": 0.005,
        },
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
            sent_actions.append(action.copy())

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
            action = np.zeros((32, 20), dtype=np.float32)
            action[:, 9] = 0.08
            action[:, 19] = 0.10
            return action

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
    np.testing.assert_allclose(sent_actions[0][:, 19], 0.105)
    np.testing.assert_allclose(sent_actions[0][:, 9], 0.075)


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


def test_server_dry_run_stops_without_post_action_observation(monkeypatch) -> None:
    events: list[tuple[str, int | str] | str] = []
    observations = iter([(0, {"frame": "warmup"}), (1, {"frame": "action-input"})])
    config = {
        "checkpoint": "/tmp/deco.ts",
        "device": "cuda:0",
        "connection": {"address": "127.0.0.1", "port": 26421, "require_token": False},
        "observation": {},
        "control": {},
        "runtime": {"auto_start": True, "warmup_runs": 0},
    }

    class FakeBridge:
        def __init__(self, **kwargs) -> None:
            pass

        def send_config(self, server_config: dict) -> None:
            pass

        def receive_observation(self, timeout: float | None = None):
            obs_seq, observation = next(observations)
            events.append(("observation", obs_seq))
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
    monkeypatch.setattr(bridge_client, "RobotBridgeClient", FakeBridge)
    monkeypatch.setattr(policy, "DECOPolicy", FakePolicy)

    remote_client.run(
        Path("unused.yaml"), max_iterations_override=1, server_dry_run=True
    )

    assert events == [
        ("observation", 0),
        ("state", "start"),
        ("observation", 1),
        ("action", 1),
        ("state", "stop"),
        "close",
    ]


def test_server_dry_run_requires_positive_max_iterations(monkeypatch) -> None:
    monkeypatch.setattr(remote_client, "check", lambda _path: {"runtime": {}})

    with pytest.raises(ValueError, match="max-iterations"):
        remote_client.run(Path("unused.yaml"), server_dry_run=True)


def test_observe_only_requires_six_stream_stage2_artifact(monkeypatch) -> None:
    config = {
        "checkpoint": "/tmp/deco.ts",
        "device": "cuda:0",
        "connection": {"address": "127.0.0.1", "port": 26421, "require_token": False},
        "observation": {"black_camera0": True},
        "control": {},
        "runtime": {"auto_start": True, "warmup_runs": 0},
    }

    class Stage1Policy:
        state_dim = 20
        action_dim = 20
        action_horizon = 32
        expected_sample_hz = 30.0
        image_keys = ("observation.images.camera0", "observation.images.camera1")
        tactile_keys = ()

        def __init__(self, checkpoint: Path, *, device: str, verify_hash: bool) -> None:
            pass

    class BridgeMustNotConnect:
        def __init__(self, **kwargs) -> None:
            raise AssertionError("Stage 1 observe-only must reject before connecting")

    monkeypatch.setattr(remote_client, "check", lambda _path: config)
    monkeypatch.setattr(remote_client, "resolve_checkpoint", lambda _config: Path("/tmp/deco.ts"))
    monkeypatch.setattr(bridge_client, "RobotBridgeClient", BridgeMustNotConnect)
    monkeypatch.setattr(policy, "DECOPolicy", Stage1Policy)

    with pytest.raises(ValueError, match="Stage 2"):
        remote_client.run(Path("unused.yaml"), observe_only=True)


def test_observe_only_requires_black_camera0_before_connecting(monkeypatch) -> None:
    config = {
        "checkpoint": "/tmp/deco.ts",
        "device": "cuda:0",
        "connection": {"address": "127.0.0.1", "port": 26421, "require_token": False},
        "observation": {"black_camera0": False},
        "control": {},
        "runtime": {"auto_start": True, "warmup_runs": 0},
    }

    class Stage2Policy:
        state_dim = 7
        action_dim = 10
        action_horizon = 32
        expected_sample_hz = 30.0
        image_keys = ("observation.images.camera0", "observation.images.camera1")
        tactile_keys = TACTILE_FIELD_ORDER

        def __init__(self, checkpoint: Path, *, device: str, verify_hash: bool) -> None:
            pass

    class BridgeMustNotConnect:
        def __init__(self, **kwargs) -> None:
            raise AssertionError("observe-only must reject before connecting")

    monkeypatch.setattr(remote_client, "check", lambda _path: config)
    monkeypatch.setattr(remote_client, "resolve_checkpoint", lambda _config: Path("/tmp/deco.ts"))
    monkeypatch.setattr(bridge_client, "RobotBridgeClient", BridgeMustNotConnect)
    monkeypatch.setattr(policy, "DECOPolicy", Stage2Policy)

    with pytest.raises(ValueError, match="black_camera0"):
        remote_client.run(Path("unused.yaml"), observe_only=True)


def test_observe_only_projects_and_saves_tactile_observation(monkeypatch, tmp_path) -> None:
    events: list[tuple[str, str] | str] = []
    saved: dict[str, object] = {}
    receive_calls = 0
    camera0 = np.full((3, 4, 3), 255, dtype=np.uint8)
    tactile_left_0 = np.full((3, 4, 3), 37, dtype=np.uint8)
    observation = {
        "observation.state": np.arange(20, dtype=np.float32),
        "observation.images.camera0": camera0,
        "observation.images.camera1": np.full((3, 4, 3), 21, dtype=np.uint8),
        TACTILE_FIELD_ORDER[0]: tactile_left_0,
        TACTILE_FIELD_ORDER[1]: np.full((3, 4, 3), 38, dtype=np.uint8),
        TACTILE_FIELD_ORDER[2]: np.full((3, 4, 3), 39, dtype=np.uint8),
        TACTILE_FIELD_ORDER[3]: np.full((3, 4, 3), 40, dtype=np.uint8),
    }
    config = {
        "checkpoint": "/tmp/deco.ts",
        "device": "cuda:0",
        "seed": 3,
        "model": {"state_action_profile": "single-right-arm-7x10"},
        "connection": {"address": "127.0.0.1", "port": 26421, "require_token": False},
        "observation": {"black_camera0": True},
        "control": {},
        "runtime": {"auto_start": False, "warmup_runs": 0},
    }

    class FakeBridge:
        def __init__(self, **kwargs) -> None:
            pass

        def send_config(self, server_config: dict) -> None:
            pass

        def receive_observation(self, timeout: float | None = None):
            nonlocal receive_calls
            receive_calls += 1
            return 0, observation

        def send_state(self, state: str) -> None:
            events.append(("state", state))

        def send_action(self, action: np.ndarray, obs_seq: int) -> None:
            events.append("action")

        def close(self) -> None:
            events.append("close")

    class FakePolicy:
        state_dim = 7
        action_dim = 10
        action_horizon = 32
        expected_sample_hz = 30.0
        image_keys = ("observation.images.camera0", "observation.images.camera1")
        tactile_keys = TACTILE_FIELD_ORDER

        def __init__(self, checkpoint: Path, *, device: str, verify_hash: bool) -> None:
            pass

        def predict(self, policy_observation: dict, *, seed: int) -> np.ndarray:
            saved["predicted_observation"] = policy_observation
            return np.zeros((32, 10), dtype=np.float32)

    def save(output_root: Path, saved_observation: dict, saved_policy, action: np.ndarray) -> Path:
        saved.update(
            output_root=output_root,
            observation=saved_observation,
            policy=saved_policy,
            action=action,
        )
        return tmp_path / "observe_only_test"

    monkeypatch.setattr(remote_client, "check", lambda _path: config)
    monkeypatch.setattr(remote_client, "resolve_checkpoint", lambda _config: Path("/tmp/deco.ts"))
    monkeypatch.setattr(remote_client, "make_server_config", lambda _config: {})
    monkeypatch.setattr(remote_client, "save_observe_only_bundle", save)
    monkeypatch.setattr(bridge_client, "RobotBridgeClient", FakeBridge)
    monkeypatch.setattr(policy, "DECOPolicy", FakePolicy)

    remote_client.run(Path("unused.yaml"), observe_only=True)

    assert receive_calls == 1
    assert events == [("state", "stop"), "close"]
    saved_observation = saved["observation"]
    assert isinstance(saved_observation, dict)
    assert tuple(
        key
        for key in saved_observation
        if key.startswith("observation.images") or key in TACTILE_FIELD_ORDER
    ) == (*FakePolicy.image_keys, *TACTILE_FIELD_ORDER)
    np.testing.assert_array_equal(
        saved_observation["observation.images.camera0"], np.zeros_like(camera0)
    )
    assert saved_observation[TACTILE_FIELD_ORDER[0]] is tactile_left_0
    assert saved["predicted_observation"] is saved_observation


def test_save_observe_only_bundle_writes_pngs_and_summary(tmp_path) -> None:
    from PIL import Image

    class FakePolicy:
        image_keys = ("observation.images.camera0", "observation.images.camera1")
        tactile_keys = TACTILE_FIELD_ORDER

    observation = {
        "observation.state": np.array([1.0, 2.0, 3.0], dtype=np.float32),
        "observation.images.camera0": np.zeros((2, 3, 3), dtype=np.uint8),
        "observation.images.camera1": np.ones((2, 3, 3), dtype=np.float32),
        TACTILE_FIELD_ORDER[0]: np.full((2, 3, 3), 2, dtype=np.uint8),
        TACTILE_FIELD_ORDER[1]: np.full((2, 3, 3), 0.5, dtype=np.float32),
        TACTILE_FIELD_ORDER[2]: np.full((2, 3, 3), 4, dtype=np.uint8),
        TACTILE_FIELD_ORDER[3]: np.full((2, 3, 3), 0.25, dtype=np.float32),
    }
    action = np.arange(20, dtype=np.float32).reshape(2, 10)

    bundle = remote_client.save_observe_only_bundle(tmp_path, observation, FakePolicy(), action)

    expected_keys = (*FakePolicy.image_keys, *FakePolicy.tactile_keys)
    assert bundle.parent == tmp_path
    assert sorted(path.name for path in bundle.iterdir()) == sorted(
        [*(key.replace(".", "_") + ".png" for key in expected_keys), "summary.json"]
    )
    for key in expected_keys:
        with Image.open(bundle / (key.replace(".", "_") + ".png")) as image:
            image.load()
            assert image.mode == "RGB"
            assert image.size == (3, 2)
    summary = json.loads((bundle / "summary.json").read_text())
    assert [item["key"] for item in summary["images"]] == list(expected_keys)
    assert summary["state"] == {"shape": [3], "min": 1.0, "max": 3.0}
    assert summary["action"] == {"shape": [2, 10], "min": 0.0, "max": 19.0}
