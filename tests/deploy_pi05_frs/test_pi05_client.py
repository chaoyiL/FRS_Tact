from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from deploy_pi05_frs import pi05_client


def _observation() -> dict[str, np.ndarray]:
    return {
        "observation.state": np.zeros((2,), dtype=np.float32),
        "observation.images.camera0": np.zeros((4, 5, 3), dtype=np.uint8),
    }


class FakePolicy:
    config = SimpleNamespace(action_horizon=2, action_dim=3, robot_action_dim=2, state_dim=2)
    robot_image_keys = ("observation.images.camera0",)

    def __init__(
        self,
        *,
        fail_on_prediction: int | None = None,
        events: list[tuple[object, ...]] | None = None,
    ) -> None:
        self.fail_on_prediction = fail_on_prediction
        self.predictions = 0
        self.events = events
        self.calls: list[tuple[object, str, int, int]] = []

    def predict_action_chunk(self, observation, task, *, seed, num_steps):
        self.predictions += 1
        self.calls.append((observation, task, seed, num_steps))
        if self.events is not None:
            self.events.append(("predict", seed, num_steps))
        if self.predictions == self.fail_on_prediction:
            raise RuntimeError("inference failed")
        return np.arange(6, dtype=np.float32).reshape(1, 2, 3)

    def unnormalize_actions(self, actions):
        return np.asarray(actions[..., :2], dtype=np.float32)


class FakeBridge:
    def __init__(
        self,
        *,
        observations,
        fail_ack: bool = False,
        events: list[tuple[object, ...]] | None = None,
        interrupt_on_receive: int | None = None,
        fail_config: bool = False,
        fail_stop: bool = False,
        fail_close: bool = False,
    ) -> None:
        self.observations = list(observations)
        self.fail_ack = fail_ack
        self.events = [] if events is None else events
        self.sent_configs: list[dict[str, object]] = []
        self.interrupt_on_receive = interrupt_on_receive
        self.receive_calls = 0
        self.fail_config = fail_config
        self.fail_stop = fail_stop
        self.fail_close = fail_close

    def receive_observation(self, timeout=None):
        self.receive_calls += 1
        if self.receive_calls == self.interrupt_on_receive:
            raise KeyboardInterrupt
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
        if self.fail_config:
            raise RuntimeError("config failed")

    def send_state(self, state):
        self.events.append(("state", state))
        if state == "stop" and self.fail_stop:
            raise RuntimeError("stop failed")

    def close(self):
        self.events.append(("close",))
        if self.fail_close:
            raise RuntimeError("close failed")


class FakeSaver:
    def __init__(
        self,
        *args,
        events: list[tuple[object, ...]] | None = None,
        fail_start: bool = False,
        fail_close: bool = False,
        **kwargs,
    ) -> None:
        self.submissions: list[tuple[int, int]] = []
        self.started = False
        self.closed = False
        self.events = events
        self.fail_start = fail_start
        self.fail_close = fail_close

    def start(self) -> None:
        self.started = True
        if self.events is not None:
            self.events.append(("saver_start",))
        if self.fail_start:
            raise RuntimeError("saver start failed")

    def submit(self, iteration, obs_seq, observation) -> None:
        self.submissions.append((iteration, obs_seq))

    def close(self) -> None:
        self.closed = True
        if self.events is not None:
            self.events.append(("saver_close",))
        if self.fail_close:
            raise RuntimeError("saver close failed")


def test_predict_robot_action_chunk_returns_full_float32_robot_chunk():
    policy = FakePolicy()
    action = pi05_client.predict_robot_action_chunk(
        policy, _observation(), "pick", seed=123, num_steps=17
    )

    assert action.shape == (2, 2)
    assert action.dtype == np.float32
    assert np.isfinite(action).all()
    assert policy.calls[0][2:] == (123, 17)


def test_predict_robot_action_chunk_rejects_wrong_model_output_shape():
    policy = FakePolicy()
    policy.predict_action_chunk = lambda *args, **kwargs: np.zeros((2, 2, 3), dtype=np.float32)

    with pytest.raises(ValueError, match="pi0.5 action"):
        pi05_client.predict_robot_action_chunk(policy, _observation(), "pick", seed=0, num_steps=10)


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
    return config


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


def test_run_warms_up_without_sending_action_then_confirms_before_start(monkeypatch, tmp_path):
    events: list[tuple[object, ...]] = []
    bridge = FakeBridge(observations=[(1, _observation()), (2, _observation())], events=events)
    policy = FakePolicy(events=events)
    saver = FakeSaver()
    config = _patch_run_dependencies(monkeypatch, bridge, policy, saver)
    config["runtime"] = {"auto_start": False, "warmup_runs": 1, "max_iterations": 1}
    monkeypatch.setattr("builtins.input", lambda prompt: events.append(("input", prompt)))

    pi05_client.run(tmp_path / "deploy_pi05.yaml")

    assert [event[0] for event in events] == [
        "config",
        "receive",
        "predict",
        "input",
        "state",
        "receive",
        "predict",
        "send_action",
        "ack",
        "state",
        "close",
    ]
    assert events[4] == ("state", "start")
    assert events[-2:] == [("state", "stop"), ("close",)]


def test_run_keyboard_interrupt_still_attempts_stop_and_close(monkeypatch, tmp_path):
    bridge = FakeBridge(observations=[(1, _observation())], interrupt_on_receive=2)
    policy = FakePolicy()
    saver = FakeSaver()
    _patch_run_dependencies(monkeypatch, bridge, policy, saver)

    pi05_client.run(tmp_path / "deploy_pi05.yaml")

    assert bridge.events[-2:] == [("state", "stop"), ("close",)]
    assert saver.closed


def test_run_zero_max_iterations_continues_until_interrupted(monkeypatch, tmp_path):
    bridge = FakeBridge(
        observations=[(1, _observation()), (2, _observation()), (3, _observation())],
        interrupt_on_receive=4,
    )
    policy = FakePolicy()
    saver = FakeSaver()
    config = _patch_run_dependencies(monkeypatch, bridge, policy, saver)
    config["runtime"] = {"auto_start": True, "warmup_runs": 1, "max_iterations": 0}

    pi05_client.run(tmp_path / "deploy_pi05.yaml")

    assert [event[0] for event in bridge.events].count("send_action") == 2
    assert bridge.events[-2:] == [("state", "stop"), ("close",)]


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


@pytest.mark.parametrize("auto_start", ["false", 1])
def test_run_rejects_invalid_auto_start_before_sending_start(
    monkeypatch, tmp_path, auto_start
):
    config = yaml.safe_load(pi05_client.DEFAULT_CONFIG.read_text(encoding="utf-8"))
    config["runtime"]["auto_start"] = auto_start
    config["runtime"]["warmup_runs"] = 1
    config["runtime"]["max_iterations"] = 1
    config["logging"]["save_observations"] = False
    config["connection"]["require_token"] = False
    config_path = tmp_path / "deploy.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    bridge = FakeBridge(observations=[(1, _observation()), (2, _observation())])
    policy = FakePolicy()
    monkeypatch.setattr(pi05_client, "make_policy_config", lambda config, path: policy.config)
    monkeypatch.setattr(pi05_client, "_make_policy", lambda policy_config: policy)
    monkeypatch.setattr(pi05_client, "RobotBridgeClient", lambda **kwargs: bridge)

    with pytest.raises(ValueError, match=r"runtime\.auto_start must be a boolean"):
        pi05_client.run(config_path)

    assert ("state", "start") not in bridge.events


def test_run_stops_before_saver_drain_then_closes_bridge(monkeypatch, tmp_path):
    events: list[tuple[object, ...]] = []
    bridge = FakeBridge(
        observations=[(1, _observation()), (2, _observation())], events=events
    )
    saver = FakeSaver(events=events)
    _patch_run_dependencies(monkeypatch, bridge, FakePolicy(events=events), saver)

    pi05_client.run(tmp_path / "deploy_pi05.yaml")

    assert events.index(("state", "stop")) < events.index(("saver_close",))
    assert events.index(("saver_close",)) < events.index(("close",))


@pytest.mark.parametrize("failure", ["stop", "saver_close"])
def test_run_cleanup_failures_do_not_block_later_cleanup(
    monkeypatch, tmp_path, failure
):
    events: list[tuple[object, ...]] = []
    bridge = FakeBridge(
        observations=[(1, _observation()), (2, _observation())],
        events=events,
        fail_stop=failure == "stop",
    )
    saver = FakeSaver(events=events, fail_close=failure == "saver_close")
    _patch_run_dependencies(monkeypatch, bridge, FakePolicy(), saver)

    pi05_client.run(tmp_path / "deploy_pi05.yaml")

    assert ("state", "stop") in events
    assert ("saver_close",) in events
    assert events[-1] == ("close",)


def test_run_treats_saver_constructor_failure_as_best_effort(monkeypatch, tmp_path):
    bridge = FakeBridge(observations=[(1, _observation()), (2, _observation())])
    policy = FakePolicy()
    _patch_run_dependencies(monkeypatch, bridge, policy, FakeSaver())

    def fail_saver(*args, **kwargs):
        raise OSError("output mkdir failed")

    monkeypatch.setattr(pi05_client, "ObservationSaver", fail_saver)

    pi05_client.run(tmp_path / "deploy_pi05.yaml")

    assert [event[0] for event in bridge.events].count("send_action") == 1
    assert bridge.events[-2:] == [("state", "stop"), ("close",)]


def test_run_treats_saver_start_failure_as_best_effort(monkeypatch, tmp_path):
    events: list[tuple[object, ...]] = []
    bridge = FakeBridge(
        observations=[(1, _observation()), (2, _observation())], events=events
    )
    saver = FakeSaver(events=events, fail_start=True)
    _patch_run_dependencies(monkeypatch, bridge, FakePolicy(), saver)

    pi05_client.run(tmp_path / "deploy_pi05.yaml")

    assert [event[0] for event in events].count("send_action") == 1
    assert events[-1] == ("close",)


def test_run_send_config_failure_still_stops_and_closes(monkeypatch, tmp_path):
    bridge = FakeBridge(observations=[], fail_config=True)
    saver = FakeSaver()
    _patch_run_dependencies(monkeypatch, bridge, FakePolicy(), saver)

    with pytest.raises(RuntimeError, match="config failed"):
        pi05_client.run(tmp_path / "deploy_pi05.yaml")

    assert bridge.events[-2:] == [("state", "stop"), ("close",)]
