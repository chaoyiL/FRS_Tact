from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from threading import Event

import msgpack
import numpy as np
import pytest

from deploy_baseline_pi05 import bridge_client, deployment, remote_client
from deploy_baseline_pi05.deployment import TACTILE_KEYS
from deploy_baseline_pi05.bridge_client import RobotBridgeClient
from deploy_baseline_pi05.protocol import (
    ScheduleChunkEnd,
    ScheduleChunkStart,
    ScheduleProtocolError,
    ScheduleSteerAck,
    ScheduleSteerRequest,
    parse_schedule_message,
)
from deploy_baseline_pi05.tests.test_config_checkpoint import _fake_asset_config


def _start(*, chunk_id: int = 7) -> dict[str, object]:
    return {
        "type": "frs_chunk_start",
        "obs_seq": 3,
        "chunk_id": chunk_id,
        "observation": {"observation.state": np.zeros(20, dtype=np.float32)},
        "observation_timestamp": 1.0,
        "control_dt": 0.1,
        "action_horizon": 50,
        "execution_mode": "block",
        "action_timestamps": None,
        "nominal_chunk_end": None,
    }


def _steer(*, chunk_id: int = 7, request_id: int = 11, action_index: int = 2) -> dict[str, object]:
    return {
        "type": "frs_steer_request",
        "chunk_id": chunk_id,
        "request_id": request_id,
        "action_index": action_index,
        "target_timestamp": None,
        "protection_applied": False,
        "observation": {"observation.state": np.ones(20, dtype=np.float32)},
    }


def _ack(*, chunk_id: int = 7, request_id: int = 11, action_index: int = 2) -> dict[str, object]:
    return {
        "type": "frs_steer_ack",
        "chunk_id": chunk_id,
        "request_id": request_id,
        "action_index": action_index,
        "status": "scheduled",
        "scheduled_timestamp": 1.2,
    }


def _end(*, chunk_id: int = 7) -> dict[str, object]:
    return {
        "type": "frs_chunk_end",
        "chunk_id": chunk_id,
        "reason": "exhausted",
        "scheduled_count": 1,
        "stale_count": 0,
    }


def _ready(chunk_id: int) -> SimpleNamespace:
    return SimpleNamespace(
        chunk_id=chunk_id,
        action_vla_normalized=np.zeros((1, 50, 20), dtype=np.float32),
        action_vla=np.ones((1, 50, 20), dtype=np.float32),
        prediction_started_at=1.0,
        prediction_finished_at=1.1,
    )


def test_protocol_parses_exact_scheduling_shapes_and_ndarrays() -> None:
    start = parse_schedule_message(_start())
    request = parse_schedule_message(_steer())
    ack = parse_schedule_message(_ack())
    end = parse_schedule_message(_end())

    assert isinstance(start, ScheduleChunkStart)
    assert isinstance(start.observation["observation.state"], np.ndarray)
    assert isinstance(request, ScheduleSteerRequest)
    assert isinstance(ack, ScheduleSteerAck)
    assert isinstance(end, ScheduleChunkEnd)

    invalid = _steer()
    invalid["extra"] = None
    with pytest.raises(ScheduleProtocolError, match="unexpected"):
        parse_schedule_message(invalid)
    invalid = _start()
    invalid["action_horizon"] = 49
    with pytest.raises(ScheduleProtocolError, match="action_horizon"):
        parse_schedule_message(invalid)


def test_bridge_auth_hello_timeout_and_full_physical_action_wire() -> None:
    class FakeSocket:
        def __init__(self) -> None:
            self.incoming = [
                msgpack.packb({"type": "hello", "protocol": "robot-bridge-v1"}),
                msgpack.packb(
                    {
                        "type": "obs",
                        "obs_seq": 41,
                        "obs": {"observation.state": np.zeros(20, dtype=np.float32)},
                    },
                    default=bridge_client._pack_array,
                ),
            ]
            self.sent: list[bytes] = []
            self.closed = False

        def recv(self, timeout: float | None = None) -> bytes:
            assert timeout in (10.0, 2.5)
            return self.incoming.pop(0)

        def send(self, payload: bytes) -> None:
            self.sent.append(payload)

        def close(self) -> None:
            self.closed = True

    socket = FakeSocket()
    calls: list[tuple[str, dict[str, str] | None]] = []

    def connect(uri: str, *, additional_headers=None, **_kwargs):
        calls.append((uri, additional_headers))
        return socket

    bridge = RobotBridgeClient("127.0.0.1", 26421, "secret", connect_factory=connect)
    server_config = {"execution_protocol": "frs_steering_v1"}
    bridge.send_config(server_config)
    obs_seq, observation = bridge.receive_observation(timeout=2.5)
    bridge.send_frs_chunk_ready(3, 7)
    bridge.send_frs_steer_action(7, 11, 2, np.zeros(20, dtype=np.float64))
    config_message, ready, action = [
        msgpack.unpackb(payload, object_hook=bridge_client._unpack_array)
        for payload in socket.sent
    ]

    assert calls == [("ws://127.0.0.1:26421", {"Authorization": "Bearer secret"})]
    assert config_message == {"type": "config", "config": server_config}
    assert obs_seq == 41
    assert observation["observation.state"].shape == (20,)
    assert ready["type"] == "frs_chunk_ready"
    assert action["type"] == "frs_steer_action"
    assert action["action"].shape == (20,)
    assert action["action"].dtype == np.dtype("float32")
    with pytest.raises(ValueError, match="20D"):
        bridge.send_frs_steer_action(7, 11, 2, np.zeros(19, dtype=np.float32))


def test_server_config_is_the_complete_direct_tactile_scheduling_contract() -> None:
    config = SimpleNamespace(
        observation=SimpleNamespace(
            data_type="vitac",
            language_prompt="trained prompt",
            single_arm_mode=False,
            no_state_obs_mode=False,
        ),
        control=SimpleNamespace(
            control_frequency=10.0,
            controller_frequency=80.0,
            steps_per_inference=50,
            action_horizon=50,
        ),
        tactile_encoder=SimpleNamespace(tactile_keys=TACTILE_KEYS),
    )

    assert deployment.make_server_config(config) == {
        "data_type": "vitac",
        "language_prompt": "trained prompt",
        "control_frequency": 10.0,
        "controller_frequency": 80.0,
        "single_arm_mode": False,
        "no_state_obs_mode": False,
        "steps_per_inference": 50,
        "action_horizon": 50,
        "observation_profile": "pi05_vitac_224",
        "execution_protocol": "frs_steering_v1",
        "steering_protection_interval_s": None,
        "frs_tactile_keys": list(TACTILE_KEYS),
    }


def test_schedule_lifecycle_preserves_order_and_duplicate_runtime_idempotence() -> None:
    events: list[object] = []
    messages = iter(
        [
            parse_schedule_message(_start()),
            parse_schedule_message(_steer()),
            parse_schedule_message(_ack()),
            parse_schedule_message(_steer()),
            parse_schedule_message(_ack()),
            parse_schedule_message(_end()),
        ]
    )

    class Bridge:
        def receive_schedule_message(self, timeout: float):
            events.append(("receive", timeout))
            return next(messages)

        def send_frs_chunk_ready(self, obs_seq: int, chunk_id: int, prediction_trace=None) -> None:
            events.append(("ready", obs_seq, chunk_id, prediction_trace))

        def send_frs_steer_action(self, chunk_id: int, request_id: int, action_index: int, action, *, trace=None) -> None:
            events.append(("action", chunk_id, request_id, action_index, np.asarray(action).shape, trace))

    class Runtime:
        def begin_chunk(self, chunk_id, observation, task, *, seed, num_steps):
            events.append(("begin", chunk_id, observation, task, seed, num_steps))
            return SimpleNamespace(chunk_id=chunk_id, action_vla_normalized=np.zeros((1, 50, 20), dtype=np.float32), action_vla=np.ones((1, 50, 20), dtype=np.float32), prediction_started_at=1.0, prediction_finished_at=1.1)

        def steer_action(self, chunk_id, request_id, observation, action_index):
            events.append(("steer", chunk_id, request_id, action_index, observation))
            return SimpleNamespace(chunk_id=chunk_id, request_id=request_id, action_index=action_index, action_vla_normalized=np.zeros((1, 50, 20), dtype=np.float32), decoded_normalized=np.ones((1, 50, 20), dtype=np.float32), selected_normalized=np.ones(20, dtype=np.float32), selected_action=np.zeros(20, dtype=np.float32), diagnostics=SimpleNamespace(delta_rms=0.25, max_normalized_action_abs=1.0), encode_started_at=1.2, encode_finished_at=1.3, decode_started_at=1.4, decode_finished_at=1.5)

        def end_chunk(self, chunk_id):
            events.append(("end", chunk_id))

    remote_client.run_schedule(
        Bridge(), Runtime(), task="pick", observation_timeout_s=2.0, action_ack_timeout_s=0.5,
        seed=0, sample_steps=10, max_iterations=1,
    )

    assert [event[0] for event in events if isinstance(event, tuple) and event[0] in {"begin", "ready", "steer", "action", "end"}] == [
        "begin", "ready", "steer", "action", "steer", "action", "end"
    ]
    assert [event[:4] for event in events if isinstance(event, tuple) and event[0] == "action"] == [
        ("action", 7, 11, 2), ("action", 7, 11, 2)
    ]
    ready = next(event for event in events if isinstance(event, tuple) and event[0] == "ready")
    action = next(event for event in events if isinstance(event, tuple) and event[0] == "action")
    assert ready[3]["coarse_normalized_action"].shape == (1, 50, 20)
    assert action[5]["coarse_normalized_action"].shape == (1, 50, 20)
    assert action[5]["selected_action"].shape == (20,)
    assert action[5]["delta_rms"] == 0.25


def test_schedule_fail_stops_on_runtime_failure_without_action() -> None:
    messages = iter([parse_schedule_message(_start()), parse_schedule_message(_steer())])

    class Bridge:
        actions = 0

        def receive_schedule_message(self, timeout: float):
            return next(messages)

        def send_frs_chunk_ready(self, *_args, **_kwargs) -> None:
            return None

        def send_frs_steer_action(self, *_args, **_kwargs) -> None:
            self.actions += 1

    class Runtime:
        def begin_chunk(self, *_args, **_kwargs):
            return _ready(7)

        def steer_action(self, *_args, **_kwargs):
            raise RuntimeError("decoder exploded")

    bridge = Bridge()
    with pytest.raises(RuntimeError, match="decoder exploded"):
        remote_client.run_schedule(
            bridge, Runtime(), task="pick", observation_timeout_s=2.0, action_ack_timeout_s=0.5,
            seed=0, sample_steps=10, max_iterations=1,
        )
    assert bridge.actions == 0


def test_schedule_rejects_mismatched_runtime_result_ids_before_any_action() -> None:
    messages = iter([parse_schedule_message(_start()), parse_schedule_message(_steer())])
    class Bridge:
        ready = 0
        actions = 0
        def receive_schedule_message(self, timeout: float): return next(messages)
        def send_frs_chunk_ready(self, *_args, **_kwargs) -> None: self.ready += 1
        def send_frs_steer_action(self, *_args, **_kwargs) -> None: self.actions += 1
    class Runtime:
        def begin_chunk(self, *_args, **_kwargs): return _ready(7)
        def steer_action(self, *_args, **_kwargs): return SimpleNamespace(chunk_id=7, request_id=99, action_index=2, selected_action=np.zeros(20, dtype=np.float32))
        def end_chunk(self, *_args, **_kwargs) -> None: return None
    bridge = Bridge()
    with pytest.raises(RuntimeError, match="request_id"):
        remote_client.run_schedule(bridge, Runtime(), task="pick", observation_timeout_s=2.0, action_ack_timeout_s=0.5, seed=0, sample_steps=10, max_iterations=1)
    assert bridge.ready == 1
    assert bridge.actions == 0


def test_schedule_rejects_mismatched_begin_chunk_before_chunk_ready() -> None:
    class Bridge:
        ready = 0
        def receive_schedule_message(self, timeout: float): return parse_schedule_message(_start())
        def send_frs_chunk_ready(self, *_args, **_kwargs) -> None: self.ready += 1
    class Runtime:
        def begin_chunk(self, *_args, **_kwargs): return SimpleNamespace(chunk_id=8)
    bridge = Bridge()
    with pytest.raises(RuntimeError, match="begin_chunk.*chunk_id"):
        remote_client.run_schedule(bridge, Runtime(), task="pick", observation_timeout_s=2.0, action_ack_timeout_s=0.5, seed=0, sample_steps=10, max_iterations=1)
    assert bridge.ready == 0


def test_warmup_drives_visual_tactile_and_decoder_on_the_server_observation() -> None:
    config = SimpleNamespace(runtime=SimpleNamespace(warmup_runs=1), source=SimpleNamespace(seed=0, sample_steps=10), observation=SimpleNamespace(language_prompt="pick"))
    events: list[tuple[object, ...]] = []
    class Runtime:
        def begin_chunk(self, chunk_id, observation, task, *, seed, num_steps):
            events.append(("begin", chunk_id, observation, task, seed, num_steps)); return SimpleNamespace(chunk_id=chunk_id)
        def steer_action(self, chunk_id, request_id, observation, action_index):
            events.append(("steer", chunk_id, request_id, observation, action_index)); return SimpleNamespace(chunk_id=chunk_id, request_id=request_id, action_index=action_index, selected_action=np.zeros(20, dtype=np.float32))
        def end_chunk(self, chunk_id): events.append(("end", chunk_id))
    observation = remote_client._warmup_observation()
    observation["observation.state"][:] = 7
    remote_client.warmup(Runtime(), config, observation)
    begin = events[0]
    warmed_observation = begin[2]
    assert [event[0] for event in events] == ["begin", "steer", "end"]
    assert warmed_observation is observation
    assert np.all(warmed_observation["observation.state"] == 7)


def test_run_wires_config_robot_observation_warmup_and_obs_seq_start(monkeypatch) -> None:
    config = remote_client.load_deployment_config(remote_client.DEFAULT_CONFIG)
    config = replace(
        config,
        connection=replace(config.connection, require_token=False),
        runtime=replace(config.runtime, auto_start=True, warmup_runs=1, max_iterations=1),
        logging=replace(config.logging, save_observations=False),
    )
    observation = remote_client._warmup_observation()
    observation["observation.state"][:] = 9
    events: list[tuple[object, ...]] = []
    messages = iter([parse_schedule_message(_start()), parse_schedule_message(_end())])

    class Runtime:
        def begin_chunk(self, chunk_id, frame, task, *, seed, num_steps):
            events.append(("begin", chunk_id, frame, task, seed, num_steps))
            return _ready(chunk_id)

        def steer_action(self, chunk_id, request_id, frame, action_index):
            events.append(("warmup_decode", chunk_id, request_id, frame, action_index))
            return SimpleNamespace(chunk_id=chunk_id, request_id=request_id, action_index=action_index)

        def end_chunk(self, chunk_id):
            events.append(("end", chunk_id))

    class Bridge:
        def __init__(self, *_args, **_kwargs):
            events.append(("connect_hello",))

        def send_config(self, value):
            events.append(("config", value))

        def receive_observation(self, timeout):
            events.append(("warmup_observation", timeout))
            return 41, observation

        def send_state(self, state, obs_seq=None):
            events.append(("state", state, obs_seq))

        def receive_schedule_message(self, timeout):
            events.append(("schedule_receive", timeout))
            return next(messages)

        def send_frs_chunk_ready(self, obs_seq, chunk_id, prediction_trace=None):
            events.append(("ready", obs_seq, chunk_id, prediction_trace))

        def send_frs_steer_action(self, *_args, **_kwargs):
            pytest.fail("warmup or empty schedule must not send a robot action")

        def close(self):
            events.append(("close",))

    monkeypatch.setattr(remote_client, "check", lambda _path: config)
    remote_client.run(
        remote_client.DEFAULT_CONFIG,
        bridge_factory=Bridge,
        runtime_factory=lambda _config: Runtime(),
    )

    names = [event[0] for event in events]
    assert names[:6] == [
        "connect_hello",
        "config",
        "warmup_observation",
        "begin",
        "warmup_decode",
        "end",
    ]
    assert events[3][2] is observation
    assert ("state", "start", 41) in events
    assert names.index("state") < names.index("schedule_receive")


def test_bounded_trace_saver_flushes_and_raises_on_full_queue(tmp_path: Path) -> None:
    entered, release = Event(), Event()
    persisted: list[dict[str, object]] = []
    def writer(payload: dict[str, object]) -> None:
        entered.set(); assert release.wait(1.0); persisted.append(payload)
    saver = remote_client.BoundedTraceSaver(tmp_path, queue_size=1, writer=writer)
    saver.start()
    saver.submit({"kind": "chunk", "iteration": 1})
    assert entered.wait(1.0)
    saver.submit({"kind": "steer", "iteration": 1})
    with pytest.raises(RuntimeError, match="queue is full"):
        saver.submit({"kind": "steer", "iteration": 2})
    release.set()
    saver.close()
    assert [item["kind"] for item in persisted] == ["chunk", "steer"]


def test_bounded_trace_saver_surfaces_background_write_failures(tmp_path: Path) -> None:
    def writer(_payload: dict[str, object]) -> None: raise OSError("disk failed")
    saver = remote_client.BoundedTraceSaver(tmp_path, queue_size=1, writer=writer)
    saver.start()
    saver.submit({"kind": "chunk", "iteration": 1})
    with pytest.raises(RuntimeError, match="trace saver failed"):
        saver.flush()
    with pytest.raises(RuntimeError, match="trace saver failed"):
        saver.close()


def test_bounded_trace_saver_uses_a_unique_session_directory(tmp_path: Path) -> None:
    first = remote_client.BoundedTraceSaver(tmp_path, queue_size=1, writer=lambda _payload: None)
    second = remote_client.BoundedTraceSaver(tmp_path, queue_size=1, writer=lambda _payload: None)
    first.start()
    second.start()
    assert first.session_dir != second.session_dir
    assert first.session_dir.parent == tmp_path
    first.close()
    second.close()


def test_schedule_samples_saved_chunk_traces_at_save_every() -> None:
    messages = iter([parse_schedule_message(_start(chunk_id=7)), parse_schedule_message(_end(chunk_id=7)), parse_schedule_message(_start(chunk_id=8)), parse_schedule_message(_end(chunk_id=8))])
    saved: list[dict[str, object]] = []
    class Bridge:
        def receive_schedule_message(self, _timeout: float): return next(messages)
        def send_frs_chunk_ready(self, *_args, **_kwargs) -> None: return None
    class Runtime:
        def begin_chunk(self, chunk_id, *_args, **_kwargs): return _ready(chunk_id)
        def end_chunk(self, *_args, **_kwargs) -> None: return None
    class Saver:
        def submit(self, payload: dict[str, object]) -> None: saved.append(payload)
    remote_client.run_schedule(Bridge(), Runtime(), task="pick", observation_timeout_s=2.0, action_ack_timeout_s=0.5, seed=0, sample_steps=10, max_iterations=2, saver=Saver(), save_every=2)
    assert [(item["kind"], item["iteration"]) for item in saved] == [("chunk", 2)]


def test_saver_close_preserves_control_failure_and_exposes_cleanup_failure() -> None:
    class BrokenSaver:
        def close(self) -> None: raise OSError("disk failed")
    control_error = RuntimeError("control failed")
    with pytest.raises(RuntimeError, match="control failed"):
        try:
            raise control_error
        except BaseException as active:
            remote_client._close_saver(BrokenSaver(), active)
            raise
    assert "trace saver close failed" in control_error.__notes__[0]


def test_check_mode_loads_config_and_never_constructs_bridge_or_runtime(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    config_path, _paths = _fake_asset_config(tmp_path)
    called: list[str] = []

    def forbidden(*_args, **_kwargs):
        called.append("forbidden")
        raise AssertionError("check mode must not connect or load model runtimes")

    monkeypatch.setattr(remote_client, "run", forbidden)
    assert remote_client.main(["--config", str(config_path), "--check"]) == 0
    assert called == []
    assert "sha256" in capsys.readouterr().out
