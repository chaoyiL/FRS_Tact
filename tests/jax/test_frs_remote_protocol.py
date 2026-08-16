from __future__ import annotations

from collections import deque
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from deploy_smolvla import remote_client
from deploy_smolvla.frs_protocol import (
    FRSChunkEnd,
    FRSChunkStart,
    FRSSteerAck,
    FRSSteerRequest,
)
from deploy_smolvla.frs_runtime import FRSChunkReady, FRSDiagnostics, FRSSteerResult


STATE_DIM = 3
ACTION_DIM = 2
IMAGE_KEYS = ("camera", "tactile")


def observation(value: int = 0) -> dict[str, np.ndarray]:
    return {
        "observation.state": np.full((STATE_DIM,), value, dtype=np.float32),
        "camera": np.full((4, 4, 3), value, dtype=np.uint8),
        "tactile": np.full((4, 4, 3), value, dtype=np.uint8),
    }


def chunk_start(chunk_id: int, *, obs_seq: int | None = None) -> FRSChunkStart:
    return FRSChunkStart(
        obs_seq=chunk_id if obs_seq is None else obs_seq,
        chunk_id=chunk_id,
        observation=observation(chunk_id),
        observation_timestamp=100.0,
        control_dt=0.05,
        action_horizon=3,
        execution_mode="block",
        action_timestamps=None,
        nominal_chunk_end=None,
    )


def steer_request(
    chunk_id: int,
    request_id: int,
    action_index: int,
) -> FRSSteerRequest:
    return FRSSteerRequest(
        chunk_id=chunk_id,
        request_id=request_id,
        action_index=action_index,
        target_timestamp=123.5,
        protection_applied=True,
        observation=observation(request_id),
    )


def steer_ack(
    chunk_id: int,
    request_id: int,
    action_index: int,
    status: str = "scheduled",
) -> FRSSteerAck:
    return FRSSteerAck(
        chunk_id=chunk_id,
        request_id=request_id,
        action_index=action_index,
        status=status,  # type: ignore[arg-type]
        scheduled_timestamp=124.0 if status == "scheduled" else None,
    )


def chunk_end(chunk_id: int) -> FRSChunkEnd:
    return FRSChunkEnd(
        chunk_id=chunk_id,
        reason="exhausted",
        scheduled_count=1,
        stale_count=0,
    )


class ScriptedBridge:
    def __init__(self, inbound: list[object], events: list[tuple[Any, ...]]) -> None:
        self.inbound = deque(inbound)
        self.events = events
        self.sent: list[tuple[Any, ...]] = []
        self.timeouts: list[float] = []

    def receive_frs_message(self, timeout: float) -> object:
        self.timeouts.append(timeout)
        if not self.inbound:
            raise TimeoutError("script exhausted")
        message = self.inbound.popleft()
        self.events.append(("receive", type(message).__name__))
        return message

    def send_frs_chunk_ready(
        self,
        obs_seq: int,
        chunk_id: int,
        prediction_trace: dict[str, Any] | None = None,
    ) -> None:
        sent = ("ready", obs_seq, chunk_id, prediction_trace)
        self.events.append(sent)
        self.sent.append(sent)

    def send_frs_steer_action(
        self,
        chunk_id: int,
        request_id: int,
        action_index: int,
        action: np.ndarray,
        *,
        trace: dict[str, Any] | None = None,
    ) -> None:
        selected = np.asarray(action)
        assert selected.shape == (ACTION_DIM,)
        sent = (
            "action",
            chunk_id,
            request_id,
            action_index,
            np.array(selected, copy=True),
            trace,
        )
        self.events.append(sent)
        self.sent.append(sent)


class PolicySpy:
    tactile_keys = ("tactile",)

    def __init__(self, events: list[tuple[Any, ...]]) -> None:
        self.events = events
        self.begin_calls: list[tuple[int, int]] = []
        self.steer_calls: list[tuple[int, int, int]] = []
        self.end_calls: list[int] = []
        self.reset_calls = 0
        self.selected_action: np.ndarray | None = None
        self.policy = SimpleNamespace(
            config=SimpleNamespace(action_dim=ACTION_DIM, chunk_size=3)
        )

    def reset_episode(self, initial_observation: object) -> None:
        del initial_observation
        self.reset_calls += 1

    def begin_chunk(
        self,
        chunk_id: int,
        initial_observation: dict[str, Any],
        task: str,
        *,
        seed: int,
        jit: bool,
        num_steps: int | None,
    ) -> FRSChunkReady:
        assert tuple(initial_observation) == (*IMAGE_KEYS, "observation.state")
        assert task == "pick"
        assert jit is True
        assert num_steps == 4
        self.begin_calls.append((chunk_id, seed))
        self.events.append(("begin", chunk_id))
        chunk = np.zeros((1, 3, ACTION_DIM), dtype=np.float32)
        return FRSChunkReady(chunk_id, chunk, chunk, chunk, 1.0, 2.0)

    def steer_action(
        self,
        chunk_id: int,
        request_id: int,
        request_observation: dict[str, Any],
        action_index: int,
    ) -> FRSSteerResult:
        assert tuple(request_observation) == (*IMAGE_KEYS, "observation.state")
        self.steer_calls.append((chunk_id, request_id, action_index))
        self.events.append(("steer", chunk_id, request_id, action_index))
        chunk = np.zeros((1, 3, ACTION_DIM), dtype=np.float32)
        selected = np.asarray([request_id, action_index], dtype=np.float32)
        return FRSSteerResult(
            chunk_id=chunk_id,
            request_id=request_id,
            action_index=action_index,
            action_vla_normalized=chunk,
            x_base=chunk,
            decoded_normalized=chunk,
            selected_normalized=selected,
            selected_action=(
                selected if self.selected_action is None else self.selected_action
            ),
            tactile_sequence_length=1,
            diagnostics=FRSDiagnostics(0.1, 0.2, 0.3),
            encode_started_at=3.0,
            encode_finished_at=4.0,
            decode_started_at=5.0,
            decode_finished_at=6.0,
        )

    def end_chunk(self, chunk_id: int) -> None:
        self.end_calls.append(chunk_id)
        self.events.append(("end", chunk_id))


class SaverSpy:
    def __init__(self) -> None:
        self.calls: list[tuple[int, int, object]] = []

    def submit(self, iteration: int, obs_seq: int, obs: object) -> None:
        self.calls.append((iteration, obs_seq, obs))


def run_args(saver: SaverSpy, *, max_chunks: int = 1) -> dict[str, Any]:
    return {
        "task": "pick",
        "state_dim": STATE_DIM,
        "image_keys": IMAGE_KEYS,
        "empty_cameras": 0,
        "observation_timeout_s": 10.0,
        "action_ack_timeout_s": 2.0,
        "seed": 7,
        "jit": True,
        "num_steps": 4,
        "max_chunks": max_chunks,
        "observation_saver": saver,
    }


@pytest.mark.parametrize("status", ["scheduled", "stale"])
def test_frs_loop_orders_ready_request_selected_action_ack_and_end(status: str) -> None:
    events: list[tuple[Any, ...]] = []
    start = chunk_start(1, obs_seq=9)
    bridge = ScriptedBridge(
        [
            start,
            steer_request(1, 4, 2),
            steer_ack(1, 4, 2, status),
            chunk_end(1),
        ],
        events,
    )
    policy = PolicySpy(events)
    saver = SaverSpy()

    remote_client._run_frs_protocol(bridge, policy, **run_args(saver))

    assert [event[0] for event in events] == [
        "receive",
        "begin",
        "ready",
        "receive",
        "steer",
        "action",
        "receive",
        "receive",
        "end",
    ]
    np.testing.assert_array_equal(bridge.sent[1][4], np.asarray([4, 2], dtype=np.float32))
    assert policy.steer_calls == [(1, 4, 2)]
    assert policy.end_calls == [1]
    assert saver.calls == [(1, 9, start.observation)]
    assert bridge.timeouts == [10.0, 10.0, 2.0, 10.0]


def test_frs_loop_rejects_a_server_rejected_ack_without_ending_chunk() -> None:
    events: list[tuple[Any, ...]] = []
    bridge = ScriptedBridge(
        [chunk_start(1), steer_request(1, 4, 1), steer_ack(1, 4, 1, "rejected")],
        events,
    )
    policy = PolicySpy(events)

    with pytest.raises(RuntimeError, match="rejected"):
        remote_client._run_frs_protocol(bridge, policy, **run_args(SaverSpy()))

    assert policy.end_calls == []


@pytest.mark.parametrize(
    "ack",
    [
        steer_ack(2, 4, 1),
        steer_ack(1, 5, 1),
        steer_ack(1, 4, 2),
    ],
)
def test_frs_loop_rejects_an_ack_that_does_not_match_the_request(ack: FRSSteerAck) -> None:
    events: list[tuple[Any, ...]] = []
    bridge = ScriptedBridge([chunk_start(1), steer_request(1, 4, 1), ack], events)
    policy = PolicySpy(events)

    with pytest.raises(RuntimeError, match="acknowledgement.*does not match"):
        remote_client._run_frs_protocol(bridge, policy, **run_args(SaverSpy()))

    assert policy.end_calls == []


def test_frs_loop_rejects_request_and_end_chunk_ids_before_policy_calls() -> None:
    events: list[tuple[Any, ...]] = []
    bridge = ScriptedBridge([chunk_start(1), steer_request(2, 4, 1)], events)
    policy = PolicySpy(events)

    with pytest.raises(RuntimeError, match="chunk id"):
        remote_client._run_frs_protocol(bridge, policy, **run_args(SaverSpy()))

    assert policy.steer_calls == []
    assert policy.end_calls == []


def test_frs_loop_rejects_an_out_of_order_chunk_start_id() -> None:
    events: list[tuple[Any, ...]] = []
    bridge = ScriptedBridge(
        [chunk_start(2), chunk_end(2), chunk_start(1)],
        events,
    )
    policy = PolicySpy(events)

    with pytest.raises(RuntimeError, match="strictly increasing"):
        remote_client._run_frs_protocol(
            bridge,
            policy,
            **run_args(SaverSpy(), max_chunks=2),
        )

    assert policy.begin_calls == [(2, 7)]


def test_frs_loop_preserves_episode_baseline_across_chunks() -> None:
    events: list[tuple[Any, ...]] = []
    bridge = ScriptedBridge(
        [chunk_start(1), chunk_end(1), chunk_start(2), chunk_end(2)],
        events,
    )
    policy = PolicySpy(events)
    saver = SaverSpy()

    remote_client._run_frs_protocol(bridge, policy, **run_args(saver, max_chunks=2))

    assert policy.reset_calls == 0
    assert policy.begin_calls == [(1, 7), (2, 7)]
    assert policy.end_calls == [1, 2]
    assert [call[:2] for call in saver.calls] == [(1, 1), (2, 2)]


def test_frs_loop_omits_trace_failures_without_interrupting_actions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[tuple[Any, ...]] = []
    bridge = ScriptedBridge(
        [chunk_start(1), steer_request(1, 4, 1), steer_ack(1, 4, 1), chunk_end(1)],
        events,
    )
    policy = PolicySpy(events)

    def fail_trace(*args: object) -> None:
        del args
        raise ValueError("trace unavailable")

    monkeypatch.setattr(remote_client, "_build_frs_chunk_trace", fail_trace)
    monkeypatch.setattr(remote_client, "_build_frs_steer_trace", fail_trace)

    remote_client._run_frs_protocol(bridge, policy, **run_args(SaverSpy()))

    assert bridge.sent[0][3] is None
    assert bridge.sent[1][5] is None


def test_frs_loop_rejects_an_out_of_order_ack_instead_of_a_request() -> None:
    events: list[tuple[Any, ...]] = []
    bridge = ScriptedBridge([chunk_start(1), steer_ack(1, 4, 1)], events)
    policy = PolicySpy(events)

    with pytest.raises(RuntimeError, match="expected FRS steer request or chunk end"):
        remote_client._run_frs_protocol(bridge, policy, **run_args(SaverSpy()))

    assert policy.steer_calls == []
    assert policy.end_calls == []


def test_frs_loop_rejects_a_non_vector_selected_action_before_sending() -> None:
    events: list[tuple[Any, ...]] = []
    bridge = ScriptedBridge([chunk_start(1), steer_request(1, 4, 1)], events)
    policy = PolicySpy(events)
    policy.selected_action = np.zeros((1, ACTION_DIM), dtype=np.float32)

    with pytest.raises(RuntimeError, match="selected action must have shape"):
        remote_client._run_frs_protocol(bridge, policy, **run_args(SaverSpy()))

    assert [message for message in bridge.sent if message[0] == "action"] == []
