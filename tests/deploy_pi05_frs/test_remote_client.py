from __future__ import annotations

import importlib
import sys
import types
from types import SimpleNamespace

import numpy as np
import pytest

from deploy_pi05_frs.frs_protocol import (
    FRSChunkEnd,
    FRSChunkStart,
    FRSSteerAck,
    FRSSteerRequest,
)


def _import_remote_client(monkeypatch):
    jax_module = types.ModuleType("jax")
    jax_module.default_backend = lambda: "cpu"
    bridge_module = types.ModuleType("deploy_pi05_frs.bridge_client")
    bridge_module.RobotBridgeClient = object
    runtime_module = types.ModuleType("deploy_pi05_frs.frs_runtime")
    runtime_module.FRSChunkReady = object
    runtime_module.FRSRuntime = object
    runtime_module.FRSSteerResult = object
    policy_module = types.ModuleType("deploy_pi05_frs.policy")
    policy_module.Pi05RemotePolicy = object
    monkeypatch.setitem(sys.modules, "jax", jax_module)
    monkeypatch.setitem(sys.modules, "deploy_pi05_frs.bridge_client", bridge_module)
    monkeypatch.setitem(sys.modules, "deploy_pi05_frs.frs_runtime", runtime_module)
    monkeypatch.setitem(sys.modules, "deploy_pi05_frs.policy", policy_module)
    monkeypatch.delitem(sys.modules, "deploy_pi05_frs.remote_client", raising=False)
    return importlib.import_module("deploy_pi05_frs.remote_client")


def _observation() -> dict[str, np.ndarray]:
    return {
        "observation.state": np.zeros((2,), dtype=np.float32),
        "observation.images.camera0": np.zeros((3, 4, 3), dtype=np.uint8),
        "observation.images.tactile0": np.zeros((3, 4, 3), dtype=np.uint8),
    }


def _messages(*, ack_request_id: int = 4):
    return [
        FRSChunkStart(
            obs_seq=2,
            chunk_id=3,
            observation=_observation(),
            observation_timestamp=100.0,
            control_dt=0.05,
            action_horizon=2,
            execution_mode="block",
            action_timestamps=None,
            nominal_chunk_end=None,
        ),
        FRSSteerRequest(
            chunk_id=3,
            request_id=4,
            action_index=1,
            target_timestamp=None,
            protection_applied=False,
            observation=_observation(),
        ),
        FRSSteerAck(
            chunk_id=3,
            request_id=ack_request_id,
            action_index=1,
            status="scheduled",
            scheduled_timestamp=100.1,
        ),
        FRSChunkEnd(chunk_id=3, reason="exhausted", scheduled_count=1, stale_count=0),
    ]


class FakePolicy:
    robot_image_keys = ("observation.images.camera0",)

    def __init__(self):
        self.config = SimpleNamespace(
            checkpoint="/models/pi05",
            state_dim=2,
            action_dim=3,
            robot_action_dim=2,
            action_horizon=2,
            empty_cameras=(),
        )


class FakeFRSRuntime:
    tactile_keys = ("observation.images.tactile0",)

    def __init__(self, policy, events):
        self.policy = policy
        self.events = events
        self.config = SimpleNamespace(steering_protection_interval_s=None)

    def reset_episode(self, observation):
        self.events.append(("reset_episode",))

    def warmup(self, observation, task, *, seed, sample_steps):
        self.events.append(("warmup", seed, sample_steps))

    def begin_chunk(self, chunk_id, observation, task, *, seed, num_steps):
        self.events.append(("begin_chunk", chunk_id))
        return SimpleNamespace(
            chunk_id=chunk_id,
            action_vla_normalized=np.zeros((2, 3), dtype=np.float32),
            action_vla=np.zeros((2, 2), dtype=np.float32),
            x_base=np.zeros((2, 3), dtype=np.float32),
            prediction_started_at=1.0,
            prediction_finished_at=2.0,
        )

    def steer_action(self, chunk_id, request_id, observation, action_index):
        self.events.append(("steer_action", chunk_id, request_id, action_index))
        return SimpleNamespace(
            chunk_id=chunk_id,
            request_id=request_id,
            action_index=action_index,
            action_vla_normalized=np.zeros((2, 3), dtype=np.float32),
            x_base=np.zeros((2, 3), dtype=np.float32),
            decoded_normalized=np.zeros((2, 3), dtype=np.float32),
            selected_normalized=np.zeros((3,), dtype=np.float32),
            selected_action=np.zeros((2,), dtype=np.float32),
            tactile_sequence_length=1,
            diagnostics=SimpleNamespace(
                tactile_change=0.0,
                delta_rms=0.0,
                max_normalized_action_abs=0.0,
            ),
            encode_started_at=1.0,
            encode_finished_at=1.1,
            decode_started_at=1.1,
            decode_finished_at=1.2,
        )

    def end_chunk(self, chunk_id):
        self.events.append(("end_chunk", chunk_id))


class FakeBridge:
    def __init__(
        self,
        messages,
        events,
        *,
        fail_config=False,
        fail_stop=False,
        fail_close=False,
    ):
        self.messages = list(messages)
        self.events = events
        self.fail_config = fail_config
        self.fail_stop = fail_stop
        self.fail_close = fail_close

    def send_config(self, config):
        self.events.append(("config", config.get("execution_protocol")))
        if self.fail_config:
            raise RuntimeError("config failed")

    def receive_observation(self, timeout=None):
        self.events.append(("warmup_receive",))
        return 1, _observation()

    def receive_frs_message(self, timeout=None):
        message = self.messages.pop(0)
        self.events.append(("receive_frs", type(message).__name__))
        return message

    def send_frs_chunk_ready(self, obs_seq, chunk_id, prediction_trace=None):
        self.events.append(("send_ready", obs_seq, chunk_id))

    def send_frs_steer_action(self, chunk_id, request_id, action_index, action, *, trace=None):
        self.events.append(("send_steer", chunk_id, request_id, action_index))

    def send_state(self, state):
        self.events.append(("state", state))
        if state == "stop" and self.fail_stop:
            raise RuntimeError("stop failed")

    def close(self):
        self.events.append(("bridge_close",))
        if self.fail_close:
            raise RuntimeError("bridge close failed")


class FakeSaver:
    def __init__(self, events, *, fail_start=False, fail_close=False):
        self.events = events
        self.fail_start = fail_start
        self.fail_close = fail_close

    def start(self):
        self.events.append(("saver_start",))
        if self.fail_start:
            raise RuntimeError("saver start failed")

    def submit(self, iteration, obs_seq, observation):
        self.events.append(("saver_submit", iteration, obs_seq))

    def close(self):
        self.events.append(("saver_close",))
        if self.fail_close:
            raise RuntimeError("saver close failed")


def _run_config(auto_start=True):
    return {
        "seed": 0,
        "num_steps": 10,
        "connection": {
            "address": "127.0.0.1",
            "port": 26421,
            "action_ack_timeout_s": 2.0,
            "require_token": False,
        },
        "observation": {
            "data_type": "vitac",
            "language_prompt": "pick",
            "single_arm_mode": False,
            "no_state_obs_mode": False,
        },
        "control": {
            "control_frequency": 20.0,
            "controller_frequency": 80.0,
            "steps_per_inference": 2,
            "action_horizon": 2,
        },
        "runtime": {"auto_start": auto_start, "warmup_runs": 1, "max_iterations": 1},
        "logging": {},
        "frs": {},
    }


def _patch_run(monkeypatch, remote_client, bridge, runtime, policy, saver):
    monkeypatch.setattr(remote_client, "load_config", lambda path: _run_config())
    monkeypatch.setattr(remote_client, "make_policy_config", lambda config, path: policy.config)
    monkeypatch.setattr(remote_client, "Pi05RemotePolicy", lambda config: policy)
    monkeypatch.setattr(
        remote_client,
        "FRSRuntime",
        lambda raw, *, config_path, policy, source_sample_steps: runtime,
    )
    monkeypatch.setattr(remote_client, "RobotBridgeClient", lambda **kwargs: bridge)
    monkeypatch.setattr(remote_client, "ObservationSaver", lambda *args: saver)


def test_run_frs_executes_chunk_and_waits_for_matching_ack(monkeypatch):
    remote_client = _import_remote_client(monkeypatch)
    events = []
    policy = FakePolicy()
    runtime = FakeFRSRuntime(policy, events)
    bridge = FakeBridge(_messages(), events)
    saver = FakeSaver(events)

    remote_client._run_frs(
        bridge,
        runtime,
        task="pick",
        image_keys=(*policy.robot_image_keys, *runtime.tactile_keys),
        observation_timeout_s=1.0,
        action_ack_timeout_s=2.0,
        seed=0,
        sample_steps=10,
        max_chunks=1,
        saver=saver,
    )

    assert [event[0] for event in events] == [
        "receive_frs",
        "saver_submit",
        "begin_chunk",
        "send_ready",
        "receive_frs",
        "steer_action",
        "send_steer",
        "receive_frs",
        "receive_frs",
        "end_chunk",
    ]


def test_run_frs_rejects_mismatched_steer_ack(monkeypatch):
    remote_client = _import_remote_client(monkeypatch)
    events = []
    policy = FakePolicy()
    runtime = FakeFRSRuntime(policy, events)

    with pytest.raises(RuntimeError, match="FRSSteerAck does not match"):
        remote_client._run_frs(
            FakeBridge(_messages(ack_request_id=99), events),
            runtime,
            task="pick",
            image_keys=(*policy.robot_image_keys, *runtime.tactile_keys),
            observation_timeout_s=1.0,
            action_ack_timeout_s=2.0,
            seed=0,
            sample_steps=10,
            max_chunks=1,
            saver=FakeSaver(events),
        )


@pytest.mark.parametrize("auto_start", [False, True])
def test_run_covers_warmup_start_frs_chunk_and_ordered_cleanup(
    monkeypatch, tmp_path, auto_start
):
    remote_client = _import_remote_client(monkeypatch)
    events = []
    policy = FakePolicy()
    runtime = FakeFRSRuntime(policy, events)
    bridge = FakeBridge(_messages(), events)
    saver = FakeSaver(events)
    _patch_run(monkeypatch, remote_client, bridge, runtime, policy, saver)
    monkeypatch.setattr(remote_client, "load_config", lambda path: _run_config(auto_start))
    monkeypatch.setattr("builtins.input", lambda prompt: events.append(("input",)))

    remote_client.run(tmp_path / "deploy.yaml")

    names = [event[0] for event in events]
    assert ("input" in names) is (not auto_start)
    assert names.index("warmup") < names.index("state") < names.index("begin_chunk")
    assert events[names.index("state")] == ("state", "start")
    stop_index = events.index(("state", "stop"))
    assert stop_index < events.index(("saver_close",)) < events.index(("bridge_close",))


def test_run_rejects_negative_max_iterations_before_bridge(monkeypatch, tmp_path):
    remote_client = _import_remote_client(monkeypatch)
    events = []
    policy = FakePolicy()
    runtime = FakeFRSRuntime(policy, events)
    bridge_calls = []
    monkeypatch.setattr(remote_client, "load_config", lambda path: _run_config())
    monkeypatch.setattr(remote_client, "make_policy_config", lambda config, path: policy.config)
    monkeypatch.setattr(remote_client, "Pi05RemotePolicy", lambda config: policy)
    monkeypatch.setattr(
        remote_client,
        "FRSRuntime",
        lambda raw, *, config_path, policy, source_sample_steps: runtime,
    )
    monkeypatch.setattr(
        remote_client,
        "RobotBridgeClient",
        lambda **kwargs: bridge_calls.append(kwargs),
    )

    with pytest.raises(ValueError, match="max_iterations must be non-negative"):
        remote_client.run(tmp_path / "deploy.yaml", -1)

    assert bridge_calls == []


@pytest.mark.parametrize("failure", ["constructor", "start"])
def test_run_treats_saver_startup_failures_as_best_effort(
    monkeypatch, tmp_path, failure
):
    remote_client = _import_remote_client(monkeypatch)
    events = []
    policy = FakePolicy()
    runtime = FakeFRSRuntime(policy, events)
    bridge = FakeBridge(_messages(), events)
    saver = FakeSaver(events, fail_start=failure == "start")
    _patch_run(monkeypatch, remote_client, bridge, runtime, policy, saver)

    if failure == "constructor":
        def fail_saver(*args):
            raise OSError("output mkdir failed")

        monkeypatch.setattr(remote_client, "ObservationSaver", fail_saver)

    remote_client.run(tmp_path / "deploy.yaml")

    assert ("state", "start") in events
    assert ("state", "stop") in events
    assert events[-1] == ("bridge_close",)


def test_run_preserves_body_error_when_all_cleanup_operations_fail(monkeypatch, tmp_path):
    remote_client = _import_remote_client(monkeypatch)
    events = []
    policy = FakePolicy()
    runtime = FakeFRSRuntime(policy, events)
    bridge = FakeBridge(
        [object()], events, fail_stop=True, fail_close=True
    )
    saver = FakeSaver(events, fail_close=True)
    _patch_run(monkeypatch, remote_client, bridge, runtime, policy, saver)

    with pytest.raises(RuntimeError, match="expected FRSChunkStart"):
        remote_client.run(tmp_path / "deploy.yaml")

    assert ("state", "stop") in events
    assert ("saver_close",) in events
    assert ("bridge_close",) in events


def test_run_send_config_failure_still_stops_and_closes(monkeypatch, tmp_path):
    remote_client = _import_remote_client(monkeypatch)
    events = []
    policy = FakePolicy()
    runtime = FakeFRSRuntime(policy, events)
    bridge = FakeBridge([], events, fail_config=True)
    saver = FakeSaver(events)
    _patch_run(monkeypatch, remote_client, bridge, runtime, policy, saver)

    with pytest.raises(RuntimeError, match="config failed"):
        remote_client.run(tmp_path / "deploy.yaml")

    assert events[-2:] == [("state", "stop"), ("bridge_close",)]
