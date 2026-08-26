"""Small binary client for the existing robot bridge scheduling wire."""

from __future__ import annotations

import functools
import ipaddress
import time
from collections.abc import Callable
from typing import Any
from urllib.parse import urlsplit, urlunsplit

import msgpack
import numpy as np

from .protocol import ScheduleMessage, parse_schedule_message


def _pack_array(value: Any) -> Any:
    if isinstance(value, (np.ndarray, np.generic)) and value.dtype.kind in ("V", "O", "c"):
        raise ValueError(f"unsupported NumPy dtype: {value.dtype}")
    if isinstance(value, np.ndarray):
        return {b"__ndarray__": True, b"data": value.tobytes(), b"dtype": value.dtype.str, b"shape": value.shape}
    if isinstance(value, np.generic):
        return {b"__npgeneric__": True, b"data": value.item(), b"dtype": value.dtype.str}
    return value


def _unpack_array(value: dict[bytes, Any]) -> Any:
    if b"__ndarray__" in value:
        try:
            dtype = np.dtype(value[b"dtype"])
            shape = tuple(value[b"shape"])
            data = value[b"data"]
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("invalid ndarray MessagePack payload") from error
        if dtype.kind in ("V", "O", "c") or any(not isinstance(dimension, int) or dimension < 0 for dimension in shape):
            raise ValueError("invalid ndarray MessagePack dtype or shape")
        if len(data) != int(np.prod(shape, dtype=np.int64)) * dtype.itemsize:
            raise ValueError("ndarray MessagePack byte count does not match its shape")
        return np.ndarray(buffer=data, dtype=dtype, shape=shape)
    if b"__npgeneric__" in value:
        return np.dtype(value[b"dtype"]).type(value[b"data"])
    return value


_Packer = functools.partial(msgpack.Packer, default=_pack_array)
_unpackb = functools.partial(msgpack.unpackb, object_hook=_unpack_array)


def _is_tunnel(host: str | None) -> bool:
    if host is None:
        return False
    normalized = host.rstrip(".").lower()
    return any(normalized == suffix or normalized.endswith(f".{suffix}") for suffix in ("ngrok-free.dev", "ngrok-free.app", "ngrok.app", "ngrok.io", "trycloudflare.com", "loca.lt", "localtunnel.me", "serveo.net", "localhost.run"))


def _is_local(host: str | None) -> bool:
    if host is None:
        return False
    normalized = host.rstrip(".").lower()
    if normalized == "localhost" or normalized.endswith(".local"):
        return True
    try:
        address = ipaddress.ip_address(normalized)
    except ValueError:
        return False
    return address.is_loopback or address.is_private or address.is_link_local


def build_websocket_uri(address: str, port: int, add_port: bool | None = None) -> str:
    address = str(address).strip()
    if not address:
        raise ValueError("robot WebSocket address must not be empty")
    has_scheme = "://" in address
    parsed = urlsplit(address if has_scheme else f"//{address}")
    if parsed.hostname is None:
        raise ValueError(f"invalid robot WebSocket address: {address!r}")
    if parsed.scheme in ("ws", "wss"):
        scheme = parsed.scheme
    elif parsed.scheme == "http":
        scheme = "ws"
    elif parsed.scheme == "https":
        scheme = "wss"
    elif not parsed.scheme:
        scheme = "wss" if _is_tunnel(parsed.hostname) else "ws"
    else:
        raise ValueError(f"unsupported WebSocket address scheme: {parsed.scheme!r}")
    add = (parsed.port is None and not _is_tunnel(parsed.hostname) and (not has_scheme or _is_local(parsed.hostname))) if add_port is None else add_port and parsed.port is None
    netloc = f"{parsed.netloc}:{port}" if add else parsed.netloc
    return urlunsplit((scheme, netloc, parsed.path, parsed.query, parsed.fragment))


class RobotBridgeClient:
    """Persistent bridge client with an injectable connector for CPU-only tests."""

    def __init__(self, address: str, port: int, token: str | None, *, add_port: bool | None = None, retry_interval_s: float = 1.0, ping_interval_s: float = 20.0, ping_timeout_s: float = 20.0, connect_factory: Callable[..., Any] | None = None) -> None:
        self.uri = build_websocket_uri(address, port, add_port)
        self.token = token
        self.retry_interval_s = retry_interval_s
        self.ping_interval_s = ping_interval_s
        self.ping_timeout_s = ping_timeout_s
        self._packer = _Packer()
        self._connect_factory = connect_factory or self._production_connect
        self._websocket = self._connect()
        try:
            hello = self._receive(timeout=10.0)
            if hello.get("type") != "hello" or hello.get("protocol") != "robot-bridge-v1":
                raise RuntimeError(f"unexpected robot bridge greeting: {hello}")
        except BaseException:
            self.close()
            raise

    @staticmethod
    def _production_connect(uri: str, **kwargs: Any) -> Any:
        from websockets.sync.client import connect

        return connect(uri, **kwargs)

    def _connect(self) -> Any:
        headers = None if not self.token else {"Authorization": f"Bearer {self.token}"}
        while True:
            try:
                return self._connect_factory(self.uri, additional_headers=headers, compression=None, max_size=None, ping_interval=self.ping_interval_s, ping_timeout=self.ping_timeout_s)
            except OSError:
                time.sleep(self.retry_interval_s)

    def _send(self, message: dict[str, Any]) -> None:
        self._websocket.send(self._packer.pack(message))

    def _receive(self, timeout: float | None = None) -> dict[str, Any]:
        raw = self._websocket.recv(timeout=timeout)
        if isinstance(raw, str):
            raise RuntimeError("robot bridge expects binary WebSocket frames")
        message = _unpackb(raw)
        if not isinstance(message, dict):
            raise RuntimeError(f"unexpected robot bridge payload: {type(message)}")
        return message

    def receive_schedule_message(self, timeout: float) -> ScheduleMessage:
        return parse_schedule_message(self._receive(timeout=timeout))

    def send_config(self, config: dict[str, Any]) -> None:
        if not isinstance(config, dict):
            raise ValueError("robot server config must be a dictionary")
        self._send({"type": "config", "config": dict(config)})

    def receive_observation(self, timeout: float | None = None) -> tuple[int, dict[str, Any]]:
        message = self._receive(timeout=timeout)
        if message.get("type") != "obs":
            raise RuntimeError(f"expected observation, received: {message.get('type')}")
        obs_seq = message.get("obs_seq")
        if not isinstance(obs_seq, int) or isinstance(obs_seq, bool) or obs_seq < 0:
            raise RuntimeError("observation obs_seq must be a nonnegative integer")
        observation = message.get("obs")
        if not isinstance(observation, dict):
            raise RuntimeError(f"observation must be a dictionary, got {type(observation)}")
        return obs_seq, observation

    def send_state(self, state: str, obs_seq: int | None = None) -> None:
        if state not in ("start", "stop"):
            raise ValueError("bridge state must be start or stop")
        message: dict[str, Any] = {"type": "state", "state": state}
        if obs_seq is not None:
            message["obs_seq"] = _nonnegative_id(obs_seq, "obs_seq")
        self._send(message)

    def send_frs_chunk_ready(self, obs_seq: int, chunk_id: int, prediction_trace: dict[str, Any] | None = None) -> None:
        self._send({"type": "frs_chunk_ready", "obs_seq": _nonnegative_id(obs_seq, "obs_seq"), "chunk_id": _nonnegative_id(chunk_id, "chunk_id"), "prediction_trace": _trace(prediction_trace, "prediction_trace")})

    def send_frs_steer_action(self, chunk_id: int, request_id: int, action_index: int, action: Any, *, trace: dict[str, Any] | None = None) -> None:
        selected = _physical_action(action)
        self._send({"type": "frs_steer_action", "chunk_id": _nonnegative_id(chunk_id, "chunk_id"), "request_id": _nonnegative_id(request_id, "request_id"), "action_index": _nonnegative_id(action_index, "action_index"), "action": selected, "trace": _trace(trace, "trace")})

    def close(self) -> None:
        try:
            self._websocket.close()
        except Exception:
            pass


def _nonnegative_id(value: Any, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return value


def _physical_action(action: Any) -> np.ndarray:
    value = np.asarray(action)
    if value.shape != (20,) or value.dtype.kind != "f":
        raise ValueError("direct decoder selected action must be a floating full physical 20D vector")
    if not np.isfinite(value).all():
        raise ValueError("direct decoder selected action must be finite")
    converted = np.asarray(value, dtype=np.float32)
    if not np.isfinite(converted).all():
        raise ValueError("direct decoder selected action must be float32-representable")
    return np.array(converted, copy=True)


def _trace(trace: dict[str, Any] | None, name: str) -> dict[str, Any] | None:
    if trace is None:
        return None
    if not isinstance(trace, dict):
        raise ValueError(f"{name} must be a dictionary or null")
    return dict(trace)
