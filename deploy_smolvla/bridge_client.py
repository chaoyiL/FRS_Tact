"""WebSocket client for the existing VB robot bridge protocol."""

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


def _pack_array(obj: Any) -> Any:
    if isinstance(obj, (np.ndarray, np.generic)) and obj.dtype.kind in ("V", "O", "c"):
        raise ValueError(f"Unsupported NumPy dtype: {obj.dtype}")
    if isinstance(obj, np.ndarray):
        return {
            b"__ndarray__": True,
            b"data": obj.tobytes(),
            b"dtype": obj.dtype.str,
            b"shape": obj.shape,
        }
    if isinstance(obj, np.generic):
        return {
            b"__npgeneric__": True,
            b"data": obj.item(),
            b"dtype": obj.dtype.str,
        }
    return obj


def _unpack_array(obj: dict[bytes, Any]) -> Any:
    if b"__ndarray__" in obj:
        return np.ndarray(buffer=obj[b"data"], dtype=np.dtype(obj[b"dtype"]), shape=obj[b"shape"])
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
    raise ValueError(f"Unsupported WebSocket address scheme: {scheme!r}")


def build_websocket_uri(address: str, port: int, add_port: bool | None) -> str:
    address = str(address).strip()
    if not address:
        raise ValueError("Robot WebSocket address must not be empty")

    has_scheme = "://" in address
    parsed = urlsplit(address if has_scheme else f"//{address}")
    host = parsed.hostname
    if host is None:
        raise ValueError(f"Invalid robot WebSocket address: {address!r}")
    specified_port = parsed.port
    scheme = _websocket_scheme(parsed.scheme, host)

    if add_port is None:
        should_add_port = specified_port is None and not _is_tunnel_host(host)
        if has_scheme and not _is_local_address(host):
            should_add_port = False
    else:
        should_add_port = add_port and specified_port is None

    netloc = parsed.netloc
    if should_add_port:
        netloc = f"{netloc}:{port}"
    return urlunsplit((scheme, netloc, parsed.path, parsed.query, parsed.fragment))


class RobotBridgeClient:
    """Persistent binary client compatible with ``vb3_robot_server/client/robot_client.py``."""

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
                    f"Robot bridge rejected the handshake with HTTP "
                    f"{error.response.status_code}; check token"
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
        if message.get("type") != "action_ack":
            raise RuntimeError(
                f"Expected action acknowledgement, received: {message.get('type')}"
            )
        acknowledged_obs_seq = message.get("obs_seq")
        if (
            not isinstance(acknowledged_obs_seq, int)
            or isinstance(acknowledged_obs_seq, bool)
            or acknowledged_obs_seq != int(obs_seq)
        ):
            raise RuntimeError(
                f"Expected action acknowledgement for observation {obs_seq}, "
                f"received: {acknowledged_obs_seq}"
            )

    def close(self) -> None:
        self._websocket.close()
