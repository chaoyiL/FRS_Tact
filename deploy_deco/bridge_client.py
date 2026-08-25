"""Minimal client for the existing binary ``robot-bridge-v1`` protocol."""

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

_TUNNEL_HOST_SUFFIXES = (
    "ngrok-free.dev", "ngrok-free.app", "ngrok.app", "ngrok.io",
    "trycloudflare.com", "loca.lt", "localtunnel.me", "serveo.net", "localhost.run",
)


def _pack_array(obj: Any) -> Any:
    if isinstance(obj, (np.ndarray, np.generic)) and obj.dtype.kind in ("V", "O", "c"):
        raise ValueError(f"unsupported NumPy dtype: {obj.dtype}")
    if isinstance(obj, np.ndarray):
        return {
            b"__ndarray__": True,
            b"data": obj.tobytes(),
            b"dtype": obj.dtype.str,
            b"shape": obj.shape,
        }
    if isinstance(obj, np.generic):
        return {b"__npgeneric__": True, b"data": obj.item(), b"dtype": obj.dtype.str}
    return obj


def _unpack_array(obj: dict[bytes, Any]) -> Any:
    if b"__ndarray__" in obj:
        return np.ndarray(
            buffer=obj[b"data"], dtype=np.dtype(obj[b"dtype"]), shape=obj[b"shape"]
        )
    if b"__npgeneric__" in obj:
        return np.dtype(obj[b"dtype"]).type(obj[b"data"])
    return obj


_Packer = functools.partial(msgpack.Packer, default=_pack_array)
_unpackb = functools.partial(msgpack.unpackb, object_hook=_unpack_array)


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


def _websocket_scheme(scheme: str, host: str | None) -> str:
    if scheme in ("ws", "wss"):
        return scheme
    if scheme == "http":
        return "ws"
    if scheme == "https":
        return "wss"
    if not scheme:
        return "wss" if _is_tunnel_host(host) else "ws"
    raise ValueError(f"unsupported WebSocket address scheme: {scheme!r}")


def build_websocket_uri(address: str, port: int, add_port: bool | None) -> str:
    address = str(address).strip()
    if not address:
        raise ValueError("robot WebSocket address must not be empty")
    has_scheme = "://" in address
    parsed = urlsplit(address if has_scheme else f"//{address}")
    host = parsed.hostname
    if host is None:
        raise ValueError(f"invalid robot WebSocket address: {address!r}")
    scheme = _websocket_scheme(parsed.scheme, host)
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
        ping_interval_s: float = 20.0,
        ping_timeout_s: float = 20.0,
    ) -> None:
        self.uri = build_websocket_uri(address, port, add_port)
        self.token = token
        self.retry_interval_s = retry_interval_s
        self.ping_interval_s = ping_interval_s
        self.ping_timeout_s = ping_timeout_s
        self._packer = _Packer()
        self._websocket = self._connect()
        try:
            hello = self._receive(timeout=10.0)
            if hello.get("type") != "hello" or hello.get("protocol") != "robot-bridge-v1":
                raise RuntimeError(f"unexpected robot bridge greeting: {hello}")
        except BaseException:
            self.close()
            raise

    def _connect(self) -> ClientConnection:
        headers = None if not self.token else {"Authorization": f"Bearer {self.token}"}
        while True:
            try:
                websocket = connect(
                    self.uri,
                    additional_headers=headers,
                    compression=None,
                    max_size=None,
                    ping_interval=self.ping_interval_s,
                    ping_timeout=self.ping_timeout_s,
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
                    f"robot bridge rejected HTTP {error.response.status_code}; check token"
                ) from error

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

    def send_config(self, config: dict[str, Any]) -> None:
        self._send({"type": "config", "config": config})

    def send_state(self, state: str) -> None:
        self._send({"type": "state", "state": state})

    def receive_observation(self, timeout: float | None = None) -> tuple[int, dict[str, Any]]:
        message = self._receive(timeout=timeout)
        if message.get("type") != "obs":
            raise RuntimeError(f"expected observation, received: {message.get('type')}")
        observation = message.get("obs")
        if not isinstance(observation, dict):
            raise RuntimeError(f"observation must be a dictionary, got {type(observation)}")
        return int(message["obs_seq"]), observation

    def send_action(self, action: np.ndarray, obs_seq: int) -> None:
        action = np.asarray(action)
        if action.ndim != 2 or action.dtype.kind != "f" or not np.isfinite(action).all():
            raise ValueError("DECO action must be a finite rank-two floating array")
        self._send(
            {
                "type": "action",
                "obs_seq": int(obs_seq),
                "action": np.array(action, dtype=np.float32, copy=True),
            }
        )

    def receive_action_ack(self, obs_seq: int, timeout: float) -> None:
        message = self._receive(timeout=timeout)
        if message.get("type") != "action_ack" or message.get("obs_seq") != int(obs_seq):
            raise RuntimeError(f"invalid action acknowledgement: {message}")

    def close(self) -> None:
        self._websocket.close()
