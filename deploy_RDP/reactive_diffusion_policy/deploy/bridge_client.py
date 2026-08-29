"""Synchronous client for the VB3 robot WebSocket bridge."""

from __future__ import annotations

import functools
import ipaddress
import time
from typing import Any
from urllib.parse import urlsplit, urlunsplit

import msgpack
import numpy as np
from websockets.exceptions import InvalidStatus
from websockets.sync.client import ClientConnection, connect


def _pack_array(value: Any) -> Any:
    if isinstance(value, (np.ndarray, np.generic)) and value.dtype.kind in ("V", "O", "c"):
        raise ValueError(f"Unsupported NumPy dtype: {value.dtype}")
    if isinstance(value, np.ndarray):
        return {
            b"__ndarray__": True,
            b"data": value.tobytes(),
            b"dtype": value.dtype.str,
            b"shape": value.shape,
        }
    if isinstance(value, np.generic):
        return {
            b"__npgeneric__": True,
            b"data": value.item(),
            b"dtype": value.dtype.str,
        }
    return value


def _unpack_array(value: dict[bytes, Any]) -> Any:
    if b"__ndarray__" in value:
        return np.ndarray(
            buffer=value[b"data"],
            dtype=np.dtype(value[b"dtype"]),
            shape=value[b"shape"],
        )
    if b"__npgeneric__" in value:
        return np.dtype(value[b"dtype"]).type(value[b"data"])
    return value


_Packer = functools.partial(msgpack.Packer, default=_pack_array)
_unpackb = functools.partial(msgpack.unpackb, object_hook=_unpack_array)
_TUNNEL_HOST_SUFFIXES = (
    "ngrok-free.dev",
    "ngrok-free.app",
    "ngrok.app",
    "ngrok.io",
    "trycloudflare.com",
    "loca.lt",
    "localtunnel.me",
    "serveo.net",
    "localhost.run",
)


def _is_local_address(host: str | None) -> bool:
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


def _is_tunnel_host(host: str | None) -> bool:
    if host is None:
        return False
    normalized = host.rstrip(".").lower()
    return any(
        normalized == suffix or normalized.endswith(f".{suffix}")
        for suffix in _TUNNEL_HOST_SUFFIXES
    )


def build_websocket_uri(address: str, port: int, add_port: bool | None = None) -> str:
    address = str(address).strip()
    if not address:
        raise ValueError("Robot WebSocket address must not be empty")
    has_scheme = "://" in address
    parsed = urlsplit(address if has_scheme else f"//{address}")
    host = parsed.hostname
    if host is None:
        raise ValueError(f"Invalid robot WebSocket address: {address!r}")
    if parsed.scheme in ("ws", "wss"):
        scheme = parsed.scheme
    elif parsed.scheme == "http":
        scheme = "ws"
    elif parsed.scheme == "https":
        scheme = "wss"
    elif not parsed.scheme:
        scheme = "wss" if _is_tunnel_host(host) else "ws"
    else:
        raise ValueError(f"Unsupported WebSocket address scheme: {parsed.scheme!r}")

    if add_port is None:
        should_add_port = parsed.port is None and not _is_tunnel_host(host)
        if has_scheme and not _is_local_address(host):
            should_add_port = False
    else:
        should_add_port = add_port and parsed.port is None
    netloc = f"{parsed.netloc}:{port}" if should_add_port else parsed.netloc
    return urlunsplit((scheme, netloc, parsed.path, parsed.query, parsed.fragment))


class RobotBridgeClient:
    def __init__(
        self,
        address: str,
        port: int,
        token: str | None,
        add_port: bool | None = None,
        retry_interval_s: float = 1.0,
    ) -> None:
        self.uri = build_websocket_uri(address, port, add_port)
        self.token = token
        self.retry_interval_s = retry_interval_s
        self._packer = _Packer()
        self._websocket = self._connect()
        hello = self._receive(timeout=10.0)
        if hello.get("type") != "hello" or hello.get("protocol") != "robot-bridge-v1":
            raise RuntimeError(f"Unexpected robot bridge greeting: {hello}")

    def _connect(self) -> ClientConnection:
        headers = None if not self.token else {"Authorization": f"Bearer {self.token}"}
        while True:
            try:
                websocket = connect(
                    self.uri,
                    additional_headers=headers,
                    compression=None,
                    max_size=None,
                    ping_interval=None,
                )
                print(f"[bridge] Connected to {self.uri}")
                return websocket
            except OSError as error:
                print(
                    f"[bridge] Connection failed: {error!r}; "
                    f"retrying in {self.retry_interval_s:.1f}s"
                )
                time.sleep(self.retry_interval_s)
            except InvalidStatus as error:
                raise RuntimeError(
                    f"Robot bridge rejected HTTP {error.response.status_code}; check token"
                ) from error

    def _send(self, message: dict[str, Any]) -> None:
        self._websocket.send(self._packer.pack(message))

    def _receive(self, timeout: float | None = None) -> dict[str, Any]:
        raw_message = self._websocket.recv(timeout=timeout)
        if isinstance(raw_message, str):
            raise RuntimeError("Robot bridge expects binary WebSocket frames")
        message = _unpackb(raw_message)
        if not isinstance(message, dict):
            raise RuntimeError(f"Unexpected robot bridge payload: {type(message)}")
        return message

    def send_config(self, config: dict[str, Any]) -> None:
        self._send({"type": "config", "config": config})

    def send_state(self, state: str) -> None:
        self._send({"type": "state", "state": state})

    def receive_observation(self, timeout: float | None = None) -> tuple[int, dict[str, Any]]:
        message = self._receive(timeout=timeout)
        if message.get("type") != "obs":
            raise RuntimeError(f"Expected observation, received: {message.get('type')}")
        observation = message["obs"]
        if not isinstance(observation, dict):
            raise RuntimeError(f"Observation must be a dictionary, got {type(observation)}")
        return int(message["obs_seq"]), observation

    def send_action(self, action: np.ndarray, obs_seq: int) -> None:
        self._send({"type": "action", "obs_seq": int(obs_seq), "action": action})

    def receive_action_ack(self, obs_seq: int, timeout: float) -> None:
        message = self._receive(timeout=timeout)
        if message.get("type") != "action_ack" or message.get("obs_seq") != int(obs_seq):
            raise RuntimeError(f"Expected action_ack for observation {obs_seq}, got {message}")

    def close(self) -> None:
        self._websocket.close()
