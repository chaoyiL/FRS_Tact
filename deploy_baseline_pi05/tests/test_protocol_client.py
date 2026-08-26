from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import msgpack
import numpy as np
import pytest

from deploy_baseline_pi05 import bridge_client, remote_client
from deploy_baseline_pi05.bridge_client import RobotBridgeClient
from deploy_baseline_pi05.protocol import (
    ScheduleChunkEnd,
    ScheduleChunkStart,
    ScheduleProtocolError,
    ScheduleSteerAck,
    ScheduleSteerRequest,
    parse_schedule_message,
)


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
            self.incoming = [msgpack.packb({"type": "hello", "protocol": "robot-bridge-v1"})]
            self.sent: list[bytes] = []
            self.closed = False

        def recv(self, timeout: float | None = None) -> bytes:
            assert timeout == 10.0
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
    bridge.send_frs_chunk_ready(3, 7)
    bridge.send_frs_steer_action(7, 11, 2, np.zeros(20, dtype=np.float64))
    ready, action = [
        msgpack.unpackb(payload, object_hook=bridge_client._unpack_array)
        for payload in socket.sent
    ]

    assert calls == [("ws://127.0.0.1:26421", {"Authorization": "Bearer secret"})]
    assert ready["type"] == "frs_chunk_ready"
    assert action["type"] == "frs_steer_action"
    assert action["action"].shape == (20,)
    assert action["action"].dtype == np.dtype("float32")
    with pytest.raises(ValueError, match="20D"):
        bridge.send_frs_steer_action(7, 11, 2, np.zeros(19, dtype=np.float32))


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
            return SimpleNamespace(chunk_id=chunk_id)

        def steer_action(self, chunk_id, request_id, observation, action_index):
            events.append(("steer", chunk_id, request_id, action_index, observation))
            return SimpleNamespace(selected_action=np.zeros(20, dtype=np.float32))

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
            return SimpleNamespace(chunk_id=7)

        def steer_action(self, *_args, **_kwargs):
            raise RuntimeError("decoder exploded")

    bridge = Bridge()
    with pytest.raises(RuntimeError, match="decoder exploded"):
        remote_client.run_schedule(
            bridge, Runtime(), task="pick", observation_timeout_s=2.0, action_ack_timeout_s=0.5,
            seed=0, sample_steps=10, max_iterations=1,
        )
    assert bridge.actions == 0


def test_check_mode_loads_config_and_never_constructs_bridge_or_runtime(monkeypatch, capsys) -> None:
    config = remote_client.check(remote_client.DEFAULT_CONFIG)
    config = replace(config, runtime=replace(config.runtime, max_iterations=1, auto_start=True))
    called: list[str] = []
    monkeypatch.setattr(remote_client, "check", lambda _path: config)

    def forbidden(*_args, **_kwargs):
        called.append("forbidden")
        raise AssertionError("check mode must not connect or load model runtimes")

    monkeypatch.setattr(remote_client, "run", forbidden)
    assert remote_client.main(["--check"]) == 0
    assert called == []
    assert "sha256" in capsys.readouterr().out
