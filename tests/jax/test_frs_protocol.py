from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pytest

from deploy_smolvla.bridge_client import RobotBridgeClient, _Packer, _unpackb
from deploy_smolvla.frs_protocol import (
    FRSChunkEnd,
    FRSChunkStart,
    FRSProtocolError,
    FRSSteerAck,
    FRSSteerRequest,
    parse_frs_server_message,
)


class RecordingSocket:
    def __init__(self) -> None:
        self.sent: list[bytes] = []

    def send(self, payload: bytes) -> None:
        self.sent.append(payload)


@pytest.fixture
def socket() -> RecordingSocket:
    return RecordingSocket()


@pytest.fixture
def bridge(socket: RecordingSocket) -> RobotBridgeClient:
    client = object.__new__(RobotBridgeClient)
    client._websocket = socket
    client._packer = _Packer()
    return client


def unpack(payload: bytes) -> dict[str, Any]:
    message = _unpackb(payload)
    assert isinstance(message, dict)
    return message


def _rtc_start() -> dict[str, Any]:
    return {
        "type": "frs_chunk_start",
        "obs_seq": 2,
        "chunk_id": 3,
        "observation": {"observation.state": np.zeros((2,), dtype=np.float32)},
        "observation_timestamp": 100.0,
        "control_dt": 0.05,
        "action_horizon": 3,
        "execution_mode": "rtc",
        "action_timestamps": np.asarray([100.0, 100.05, 100.1], dtype=np.float64),
        "nominal_chunk_end": 100.15,
    }


def test_protocol_parses_rtc_chunk_start_with_exact_timestamps() -> None:
    message = _rtc_start()

    parsed = parse_frs_server_message(message)

    assert isinstance(parsed, FRSChunkStart)
    assert parsed.obs_seq == 2
    assert parsed.chunk_id == 3
    assert parsed.execution_mode == "rtc"
    assert parsed.action_horizon == 3
    assert parsed.action_timestamps is message["action_timestamps"]
    assert parsed.nominal_chunk_end == 100.15


def test_protocol_parses_block_chunk_start_only_with_null_timing() -> None:
    message = _rtc_start()
    message.update(
        execution_mode="block",
        action_timestamps=None,
        nominal_chunk_end=None,
    )

    parsed = parse_frs_server_message(message)

    assert isinstance(parsed, FRSChunkStart)
    assert parsed.execution_mode == "block"
    assert parsed.action_timestamps is None
    assert parsed.nominal_chunk_end is None


def test_protocol_preserves_the_full_steer_request_observation_mapping() -> None:
    observation: Mapping[str, Any] = {
        "observation.state": np.asarray([1.0, 2.0], dtype=np.float32),
        "server_only": {"nested": "untouched"},
    }
    message = {
        "type": "frs_steer_request",
        "chunk_id": 3,
        "request_id": 8,
        "action_index": 1,
        "target_timestamp": None,
        "protection_applied": True,
        "observation": observation,
    }

    parsed = parse_frs_server_message(message)

    assert isinstance(parsed, FRSSteerRequest)
    assert parsed.observation is observation
    assert parsed.target_timestamp is None
    assert parsed.protection_applied is True


@pytest.mark.parametrize(
    ("message", "expected_type"),
    [
        (
            {
                "type": "frs_steer_ack",
                "chunk_id": 3,
                "request_id": 8,
                "action_index": 1,
                "status": "scheduled",
                "scheduled_timestamp": 101.0,
            },
            FRSSteerAck,
        ),
        (
            {
                "type": "frs_chunk_end",
                "chunk_id": 3,
                "reason": "deadline",
                "scheduled_count": 1,
                "stale_count": 2,
            },
            FRSChunkEnd,
        ),
    ],
)
def test_protocol_parses_ack_and_end_enums(
    message: dict[str, Any], expected_type: type[FRSSteerAck] | type[FRSChunkEnd]
) -> None:
    assert isinstance(parse_frs_server_message(message), expected_type)


@pytest.mark.parametrize(
    "message",
    [
        {**_rtc_start(), "chunk_id": True},
        {**_rtc_start(), "obs_seq": True},
        {
            "type": "frs_steer_request",
            "chunk_id": 3,
            "request_id": True,
            "action_index": 1,
            "target_timestamp": None,
            "protection_applied": False,
            "observation": {},
        },
        {**_rtc_start(), "observation_timestamp": float("nan")},
        {**_rtc_start(), "control_dt": float("inf")},
        {**_rtc_start(), "action_timestamps": np.zeros((1, 3), dtype=np.float32)},
        {**_rtc_start(), "action_timestamps": np.zeros((2,), dtype=np.float32)},
        {**_rtc_start(), "nominal_chunk_end": None},
        {
            **_rtc_start(),
            "execution_mode": "block",
            "action_timestamps": np.zeros((3,), dtype=np.float32),
            "nominal_chunk_end": None,
        },
        {
            "type": "frs_steer_ack",
            "chunk_id": 3,
            "request_id": 8,
            "action_index": 1,
            "status": "stale",
            "scheduled_timestamp": 101.0,
        },
    ],
)
def test_protocol_rejects_malformed_wire_values(message: dict[str, Any]) -> None:
    with pytest.raises(FRSProtocolError):
        parse_frs_server_message(message)


def test_bridge_sends_only_rank_one_selected_action(
    bridge: RobotBridgeClient, socket: RecordingSocket
) -> None:
    bridge.send_frs_steer_action(3, 8, 4, np.zeros((14,), dtype=np.float32))

    message = unpack(socket.sent[-1])
    assert {key: value for key, value in message.items() if key != "action"} == {
        "type": "frs_steer_action",
        "chunk_id": 3,
        "request_id": 8,
        "action_index": 4,
        "trace": None,
    }
    np.testing.assert_array_equal(message["action"], np.zeros((14,), dtype=np.float32))


def test_bridge_sends_chunk_ready_with_the_optional_trace(
    bridge: RobotBridgeClient, socket: RecordingSocket
) -> None:
    trace = {"prediction": "ready"}

    bridge.send_frs_chunk_ready(2, 3, trace)

    assert unpack(socket.sent[-1]) == {
        "type": "frs_chunk_ready",
        "obs_seq": 2,
        "chunk_id": 3,
        "prediction_trace": trace,
    }


def test_bridge_receives_a_typed_frs_message(bridge: RobotBridgeClient) -> None:
    bridge._receive = lambda timeout: _rtc_start()

    message = bridge.receive_frs_message(timeout=0.5)

    assert isinstance(message, FRSChunkStart)
    assert message.chunk_id == 3


@pytest.mark.parametrize(
    "action",
    [
        np.zeros((1, 2), dtype=np.float32),
        np.asarray([1, 2], dtype=np.int64),
        np.asarray([np.nan], dtype=np.float32),
        np.asarray([float(np.finfo(np.float32).max) * 2], dtype=np.float64),
    ],
)
def test_bridge_rejects_invalid_selected_action(
    bridge: RobotBridgeClient, action: np.ndarray
) -> None:
    with pytest.raises(ValueError):
        bridge.send_frs_steer_action(3, 8, 4, action)


def test_bridge_serializes_an_isolated_float32_selected_action(
    bridge: RobotBridgeClient, socket: RecordingSocket
) -> None:
    action = np.asarray([1.25, 2.5], dtype=np.float64)

    bridge.send_frs_steer_action(3, 8, 4, action)
    action[:] = -1.0

    sent_action = unpack(socket.sent[-1])["action"]
    assert isinstance(sent_action, np.ndarray)
    assert sent_action.dtype == np.float32
    np.testing.assert_array_equal(sent_action, [1.25, 2.5])
