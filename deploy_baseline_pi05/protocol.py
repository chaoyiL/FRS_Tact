"""Strict parser for the robot bridge's ``frs_steering_v1`` scheduling wire."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np


class ScheduleProtocolError(ValueError):
    """Raised when a scheduling message does not match the bridge wire schema."""


@dataclass(frozen=True)
class ScheduleChunkStart:
    obs_seq: int
    chunk_id: int
    observation: Mapping[str, Any]
    observation_timestamp: float
    control_dt: float
    action_horizon: int
    execution_mode: Literal["rtc", "block"]
    action_timestamps: np.ndarray | None
    nominal_chunk_end: float | None


@dataclass(frozen=True)
class ScheduleSteerRequest:
    chunk_id: int
    request_id: int
    action_index: int
    target_timestamp: float | None
    protection_applied: bool
    observation: Mapping[str, Any]


@dataclass(frozen=True)
class ScheduleSteerAck:
    chunk_id: int
    request_id: int
    action_index: int
    status: Literal["scheduled", "stale", "rejected"]
    scheduled_timestamp: float | None


@dataclass(frozen=True)
class ScheduleChunkEnd:
    chunk_id: int
    reason: Literal["exhausted", "deadline", "no_future_action", "stopped"]
    scheduled_count: int
    stale_count: int


type ScheduleMessage = ScheduleChunkStart | ScheduleSteerRequest | ScheduleSteerAck | ScheduleChunkEnd

_START_KEYS = frozenset(("type", "obs_seq", "chunk_id", "observation", "observation_timestamp", "control_dt", "action_horizon", "execution_mode", "action_timestamps", "nominal_chunk_end"))
_STEER_KEYS = frozenset(("type", "chunk_id", "request_id", "action_index", "target_timestamp", "protection_applied", "observation"))
_ACK_KEYS = frozenset(("type", "chunk_id", "request_id", "action_index", "status", "scheduled_timestamp"))
_END_KEYS = frozenset(("type", "chunk_id", "reason", "scheduled_count", "stale_count"))


def _exact_keys(message: Mapping[str, Any], expected: frozenset[str]) -> None:
    missing = expected - set(message)
    extra = set(message) - expected
    if missing:
        raise ScheduleProtocolError(f"missing scheduling message fields: {sorted(missing)}")
    if extra:
        raise ScheduleProtocolError(f"unexpected scheduling message fields: {sorted(map(repr, extra))}")


def _id(message: Mapping[str, Any], name: str, *, positive: bool = False) -> int:
    value = message[name]
    if not isinstance(value, int) or isinstance(value, bool) or value < int(positive):
        qualifier = "positive" if positive else "nonnegative"
        raise ScheduleProtocolError(f"{name} must be a {qualifier} integer")
    return value


def _finite(message: Mapping[str, Any], name: str, *, nullable: bool = False) -> float | None:
    value = message[name]
    if nullable and value is None:
        return None
    if not isinstance(value, (float, np.floating)) or not math.isfinite(float(value)):
        qualifier = "null or a " if nullable else "a "
        raise ScheduleProtocolError(f"{name} must be {qualifier}finite float")
    return float(value)


def _observation(message: Mapping[str, Any]) -> Mapping[str, Any]:
    value = message["observation"]
    if not isinstance(value, Mapping):
        raise ScheduleProtocolError("observation must be a mapping")
    return value


def _parse_start(message: Mapping[str, Any]) -> ScheduleChunkStart:
    horizon = _id(message, "action_horizon", positive=True)
    if horizon != 50:
        raise ScheduleProtocolError("action_horizon must be 50 for direct Pi0.5 scheduling")
    mode = message["execution_mode"]
    if type(mode) is not str or mode not in ("rtc", "block"):
        raise ScheduleProtocolError("execution_mode must be 'rtc' or 'block'")
    timestamps = message["action_timestamps"]
    end = message["nominal_chunk_end"]
    if mode == "rtc":
        if not isinstance(timestamps, np.ndarray) or timestamps.dtype.kind != "f" or timestamps.shape != (horizon,) or not np.isfinite(timestamps).all():
            raise ScheduleProtocolError("RTC action_timestamps must be finite floating ndarray shaped [50]")
        if not isinstance(end, (float, np.floating)) or not math.isfinite(float(end)):
            raise ScheduleProtocolError("RTC nominal_chunk_end must be a finite float")
        parsed_timestamps, parsed_end = timestamps, float(end)
    else:
        if timestamps is not None or end is not None:
            raise ScheduleProtocolError("block action_timestamps and nominal_chunk_end must both be null")
        parsed_timestamps, parsed_end = None, None
    return ScheduleChunkStart(_id(message, "obs_seq"), _id(message, "chunk_id"), _observation(message), _finite(message, "observation_timestamp"), _finite(message, "control_dt"), horizon, mode, parsed_timestamps, parsed_end)


def _parse_steer(message: Mapping[str, Any]) -> ScheduleSteerRequest:
    protected = message["protection_applied"]
    if not isinstance(protected, bool):
        raise ScheduleProtocolError("protection_applied must be a bool")
    return ScheduleSteerRequest(_id(message, "chunk_id"), _id(message, "request_id"), _id(message, "action_index"), _finite(message, "target_timestamp", nullable=True), protected, _observation(message))


def _parse_ack(message: Mapping[str, Any]) -> ScheduleSteerAck:
    status = message["status"]
    if type(status) is not str or status not in ("scheduled", "stale", "rejected"):
        raise ScheduleProtocolError("invalid scheduling acknowledgement status")
    timestamp = _finite(message, "scheduled_timestamp", nullable=True)
    if (status == "scheduled") != (timestamp is not None):
        raise ScheduleProtocolError("scheduled_timestamp must be finite exactly when status is scheduled")
    return ScheduleSteerAck(_id(message, "chunk_id"), _id(message, "request_id"), _id(message, "action_index"), status, timestamp)


def _parse_end(message: Mapping[str, Any]) -> ScheduleChunkEnd:
    reason = message["reason"]
    if type(reason) is not str or reason not in ("exhausted", "deadline", "no_future_action", "stopped"):
        raise ScheduleProtocolError("invalid scheduling chunk end reason")
    return ScheduleChunkEnd(_id(message, "chunk_id"), reason, _id(message, "scheduled_count"), _id(message, "stale_count"))


def parse_schedule_message(message: Mapping[str, Any]) -> ScheduleMessage:
    """Strictly parse bridge scheduling only; it contains no FRS inference payloads."""
    if not isinstance(message, Mapping):
        raise ScheduleProtocolError(f"scheduling message must be a mapping, got {type(message)}")
    message_type = message.get("type")
    if type(message_type) is not str:
        raise ScheduleProtocolError("scheduling message type must be a built-in str")
    if message_type == "frs_chunk_start":
        _exact_keys(message, _START_KEYS)
        return _parse_start(message)
    if message_type == "frs_steer_request":
        _exact_keys(message, _STEER_KEYS)
        return _parse_steer(message)
    if message_type == "frs_steer_ack":
        _exact_keys(message, _ACK_KEYS)
        return _parse_ack(message)
    if message_type == "frs_chunk_end":
        _exact_keys(message, _END_KEYS)
        return _parse_end(message)
    raise ScheduleProtocolError(f"unsupported scheduling message type: {message_type!r}")
