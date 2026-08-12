"""Strict typed parser for FRS server wire messages."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal, TypeAlias

import numpy as np


class FRSProtocolError(ValueError):
    """Raised when an FRS wire message violates its schema."""


@dataclass(frozen=True)
class FRSChunkStart:
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
class FRSSteerRequest:
    chunk_id: int
    request_id: int
    action_index: int
    target_timestamp: float | None
    protection_applied: bool
    observation: Mapping[str, Any]


@dataclass(frozen=True)
class FRSSteerAck:
    chunk_id: int
    request_id: int
    action_index: int
    status: Literal["scheduled", "stale", "rejected"]
    scheduled_timestamp: float | None


@dataclass(frozen=True)
class FRSChunkEnd:
    chunk_id: int
    reason: Literal["exhausted", "deadline", "no_future_action", "stopped"]
    scheduled_count: int
    stale_count: int


FRSServerMessage: TypeAlias = FRSChunkStart | FRSSteerRequest | FRSSteerAck | FRSChunkEnd


def _field(message: Mapping[str, Any], name: str) -> Any:
    try:
        return message[name]
    except KeyError as error:
        raise FRSProtocolError(f"missing FRS message field: {name}") from error


def _nonnegative_id(message: Mapping[str, Any], name: str) -> int:
    value = _field(message, name)
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise FRSProtocolError(f"{name} must be a nonnegative integer")
    return value


def _positive_integer(message: Mapping[str, Any], name: str) -> int:
    value = _nonnegative_id(message, name)
    if value == 0:
        raise FRSProtocolError(f"{name} must be a positive integer")
    return value


def _finite_float(message: Mapping[str, Any], name: str) -> float:
    value = _field(message, name)
    if not isinstance(value, (float, np.floating)) or not math.isfinite(float(value)):
        raise FRSProtocolError(f"{name} must be a finite float")
    return float(value)


def _nullable_finite_float(message: Mapping[str, Any], name: str) -> float | None:
    value = _field(message, name)
    if value is None:
        return None
    if not isinstance(value, (float, np.floating)) or not math.isfinite(float(value)):
        raise FRSProtocolError(f"{name} must be null or a finite float")
    return float(value)


def _observation(message: Mapping[str, Any]) -> Mapping[str, Any]:
    observation = _field(message, "observation")
    if not isinstance(observation, Mapping):
        raise FRSProtocolError("observation must be a mapping")
    return observation


def _parse_chunk_start(message: Mapping[str, Any]) -> FRSChunkStart:
    action_horizon = _positive_integer(message, "action_horizon")
    execution_mode = _field(message, "execution_mode")
    if execution_mode not in ("rtc", "block"):
        raise FRSProtocolError("execution_mode must be 'rtc' or 'block'")

    action_timestamps = _field(message, "action_timestamps")
    nominal_chunk_end = _field(message, "nominal_chunk_end")
    if execution_mode == "rtc":
        if (
            not isinstance(action_timestamps, np.ndarray)
            or action_timestamps.ndim != 1
            or action_timestamps.shape != (action_horizon,)
            or action_timestamps.dtype.kind != "f"
            or not np.isfinite(action_timestamps).all()
        ):
            raise FRSProtocolError(
                "RTC action_timestamps must be a finite floating ndarray with shape [H]"
            )
        if (
            not isinstance(nominal_chunk_end, (float, np.floating))
            or not math.isfinite(float(nominal_chunk_end))
        ):
            raise FRSProtocolError("RTC nominal_chunk_end must be a finite float")
        parsed_timestamps: np.ndarray | None = action_timestamps
        parsed_end: float | None = float(nominal_chunk_end)
    else:
        if action_timestamps is not None or nominal_chunk_end is not None:
            raise FRSProtocolError(
                "block action_timestamps and nominal_chunk_end must both be null"
            )
        parsed_timestamps = None
        parsed_end = None

    return FRSChunkStart(
        obs_seq=_nonnegative_id(message, "obs_seq"),
        chunk_id=_nonnegative_id(message, "chunk_id"),
        observation=_observation(message),
        observation_timestamp=_finite_float(message, "observation_timestamp"),
        control_dt=_finite_float(message, "control_dt"),
        action_horizon=action_horizon,
        execution_mode=execution_mode,
        action_timestamps=parsed_timestamps,
        nominal_chunk_end=parsed_end,
    )


def _parse_steer_request(message: Mapping[str, Any]) -> FRSSteerRequest:
    protection_applied = _field(message, "protection_applied")
    if not isinstance(protection_applied, bool):
        raise FRSProtocolError("protection_applied must be a bool")
    return FRSSteerRequest(
        chunk_id=_nonnegative_id(message, "chunk_id"),
        request_id=_nonnegative_id(message, "request_id"),
        action_index=_nonnegative_id(message, "action_index"),
        target_timestamp=_nullable_finite_float(message, "target_timestamp"),
        protection_applied=protection_applied,
        observation=_observation(message),
    )


def _parse_steer_ack(message: Mapping[str, Any]) -> FRSSteerAck:
    status = _field(message, "status")
    if status not in ("scheduled", "stale", "rejected"):
        raise FRSProtocolError("invalid FRS steer acknowledgement status")
    scheduled_timestamp = _nullable_finite_float(message, "scheduled_timestamp")
    if (status == "scheduled") != (scheduled_timestamp is not None):
        raise FRSProtocolError(
            "scheduled_timestamp must be finite exactly when status is 'scheduled'"
        )
    return FRSSteerAck(
        chunk_id=_nonnegative_id(message, "chunk_id"),
        request_id=_nonnegative_id(message, "request_id"),
        action_index=_nonnegative_id(message, "action_index"),
        status=status,
        scheduled_timestamp=scheduled_timestamp,
    )


def _parse_chunk_end(message: Mapping[str, Any]) -> FRSChunkEnd:
    reason = _field(message, "reason")
    if reason not in ("exhausted", "deadline", "no_future_action", "stopped"):
        raise FRSProtocolError("invalid FRS chunk end reason")
    return FRSChunkEnd(
        chunk_id=_nonnegative_id(message, "chunk_id"),
        reason=reason,
        scheduled_count=_nonnegative_id(message, "scheduled_count"),
        stale_count=_nonnegative_id(message, "stale_count"),
    )


def parse_frs_server_message(message: Mapping[str, Any]) -> FRSServerMessage:
    """Parse one of the four FRS server message types without changing its payload."""

    if not isinstance(message, Mapping):
        raise FRSProtocolError(f"FRS message must be a mapping, got {type(message)}")
    message_type = message.get("type")
    if message_type == "frs_chunk_start":
        return _parse_chunk_start(message)
    if message_type == "frs_steer_request":
        return _parse_steer_request(message)
    if message_type == "frs_steer_ack":
        return _parse_steer_ack(message)
    if message_type == "frs_chunk_end":
        return _parse_chunk_end(message)
    raise FRSProtocolError(f"unsupported FRS server message type: {message_type!r}")
