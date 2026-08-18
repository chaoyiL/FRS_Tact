from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from deploy_pi05_frs import pi05_client


def _observation() -> dict[str, np.ndarray]:
    return {
        "observation.state": np.zeros((2,), dtype=np.float32),
        "observation.images.camera0": np.zeros((4, 5, 3), dtype=np.uint8),
    }


class FakePolicy:
    config = SimpleNamespace(action_horizon=2, action_dim=3, robot_action_dim=2, state_dim=2)
    robot_image_keys = ("observation.images.camera0",)

    def __init__(self, *, fail_on_prediction: int | None = None) -> None:
        self.fail_on_prediction = fail_on_prediction
        self.predictions = 0

    def predict_action_chunk(self, observation, task, *, seed, num_steps):
        self.predictions += 1
        if self.predictions == self.fail_on_prediction:
            raise RuntimeError("inference failed")
        return np.arange(6, dtype=np.float32).reshape(1, 2, 3)

    def unnormalize_actions(self, actions):
        return np.asarray(actions[..., :2], dtype=np.float32)


class FakeBridge:
    def __init__(self, *, observations, fail_ack: bool = False) -> None:
        self.observations = list(observations)
        self.fail_ack = fail_ack
        self.events: list[tuple[object, ...]] = []
        self.sent_configs: list[dict[str, object]] = []

    def receive_observation(self, timeout=None):
        obs_seq, observation = self.observations.pop(0)
        self.events.append(("receive", obs_seq))
        return obs_seq, observation

    def send_action(self, action, obs_seq, *, trace=None):
        self.events.append(("send_action", obs_seq, tuple(action.shape)))

    def receive_action_ack(self, obs_seq, timeout):
        self.events.append(("ack", obs_seq))
        if self.fail_ack:
            raise RuntimeError("ack failed")

    def send_config(self, config):
        self.sent_configs.append(config)
        self.events.append(("config",))

    def send_state(self, state):
        self.events.append(("state", state))

    def close(self):
        self.events.append(("close",))


class FakeSaver:
    def __init__(self, *args, **kwargs) -> None:
        self.submissions: list[tuple[int, int]] = []
        self.started = False
        self.closed = False

    def start(self) -> None:
        self.started = True

    def submit(self, iteration, obs_seq, observation) -> None:
        self.submissions.append((iteration, obs_seq))

    def close(self) -> None:
        self.closed = True


def test_predict_robot_action_chunk_returns_full_float32_robot_chunk():
    action = pi05_client.predict_robot_action_chunk(
        FakePolicy(), _observation(), "pick", seed=0, num_steps=10
    )

    assert action.shape == (2, 2)
    assert action.dtype == np.float32
    assert np.isfinite(action).all()


@pytest.mark.parametrize(
    "unnormalized",
    [
        np.zeros((2, 1), dtype=np.float32),
        np.full((2, 2), np.nan, dtype=np.float32),
    ],
)
def test_predict_robot_action_chunk_rejects_invalid_robot_output(unnormalized):
    policy = FakePolicy()
    policy.unnormalize_actions = lambda actions: unnormalized

    with pytest.raises(ValueError, match="robot action"):
        pi05_client.predict_robot_action_chunk(policy, _observation(), "pick", seed=0, num_steps=10)


def test_legacy_loop_waits_for_matching_ack_before_next_observation():
    bridge = FakeBridge(observations=[(7, _observation()), (8, _observation())])
    pi05_client.run_legacy_loop(
        bridge,
        FakePolicy(),
        task="pick",
        image_keys=FakePolicy.robot_image_keys,
        observation_timeout_s=1.0,
        action_ack_timeout_s=2.0,
        seed=0,
        sample_steps=10,
        max_iterations=2,
        saver=FakeSaver(),
    )

    assert bridge.events == [
        ("receive", 7),
        ("send_action", 7, (2, 2)),
        ("ack", 7),
        ("receive", 8),
        ("send_action", 8, (2, 2)),
        ("ack", 8),
    ]


def test_legacy_loop_stops_at_max_iterations():
    bridge = FakeBridge(observations=[(7, _observation()), (8, _observation())])
    saver = FakeSaver()

    pi05_client.run_legacy_loop(
        bridge,
        FakePolicy(),
        task="pick",
        image_keys=FakePolicy.robot_image_keys,
        observation_timeout_s=1.0,
        action_ack_timeout_s=2.0,
        seed=0,
        sample_steps=10,
        max_iterations=1,
        saver=saver,
    )

    assert [event[0] for event in bridge.events] == ["receive", "send_action", "ack"]
    assert saver.submissions == [(1, 7)]


def _run_config() -> dict[str, object]:
    return {
        "connection": {
            "address": "127.0.0.1",
            "port": 26421,
            "action_ack_timeout_s": 2.0,
        },
        "observation": {"language_prompt": "pick"},
        "control": {},
        "runtime": {"auto_start": True, "warmup_runs": 1, "max_iterations": 1},
        "logging": {},
        "seed": 0,
        "num_steps": 10,
    }


def _patch_run_dependencies(monkeypatch, bridge, policy, saver):
    config = _run_config()
    monkeypatch.setattr(pi05_client, "load_deployment_config", lambda path, mode: config)
    monkeypatch.setattr(pi05_client, "make_policy_config", lambda config, path: policy.config)
    monkeypatch.setattr(pi05_client, "make_server_config", lambda config, *, mode: {"data_type": "vision"})
    monkeypatch.setattr(pi05_client, "_make_policy", lambda policy_config: policy)
    monkeypatch.setattr(pi05_client, "RobotBridgeClient", lambda **kwargs: bridge)
    monkeypatch.setattr(pi05_client, "ObservationSaver", lambda *args: saver)


def test_run_sends_plain_server_config_and_closes_resources(monkeypatch, tmp_path):
    bridge = FakeBridge(observations=[(1, _observation()), (2, _observation())])
    policy = FakePolicy()
    saver = FakeSaver()
    _patch_run_dependencies(monkeypatch, bridge, policy, saver)

    pi05_client.run(tmp_path / "deploy_pi05.yaml")

    assert bridge.sent_configs == [{"data_type": "vision"}]
    assert "execution_protocol" not in bridge.sent_configs[0]
    assert bridge.events[-2:] == [("state", "stop"), ("close",)]
    assert saver.started and saver.closed


@pytest.mark.parametrize("failure", ["inference", "ack"])
def test_run_attempts_stop_and_close_when_inference_or_ack_raises(monkeypatch, tmp_path, failure):
    bridge = FakeBridge(
        observations=[(1, _observation()), (2, _observation())], fail_ack=failure == "ack"
    )
    policy = FakePolicy(fail_on_prediction=2 if failure == "inference" else None)
    saver = FakeSaver()
    _patch_run_dependencies(monkeypatch, bridge, policy, saver)

    with pytest.raises(RuntimeError, match=f"{failure} failed"):
        pi05_client.run(tmp_path / "deploy_pi05.yaml")

    assert bridge.events[-2:] == [("state", "stop"), ("close",)]
    assert saver.closed
