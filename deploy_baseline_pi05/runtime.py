"""Fail-stop direct tactile steering state machine for one Pi0.5 action chunk."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from hashlib import sha256
from numbers import Integral
from time import perf_counter
from typing import Any

import numpy as np

from .deployment import TACTILE_KEYS


_CHUNK_SHAPE = (1, 50, 20)
_TOKEN_SHAPE = (1, 4, 512)


def _immutable(value: Any) -> np.ndarray:
    array = np.ascontiguousarray(np.asarray(value, dtype=np.float32))
    return np.frombuffer(array.tobytes(order="C"), dtype=np.float32).reshape(array.shape)


def _finite_shape(value: Any, shape: tuple[int, ...], name: str) -> np.ndarray:
    try:
        array = np.asarray(value, dtype=np.float32)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric") from exc
    if array.shape != shape:
        raise ValueError(f"{name} must be shaped {shape}, got {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must be finite")
    return array


@dataclass(frozen=True)
class DirectChunkReady:
    chunk_id: int
    action_vla_normalized: np.ndarray
    action_vla: np.ndarray
    prediction_started_at: float
    prediction_finished_at: float


@dataclass(frozen=True)
class DirectSteerDiagnostics:
    delta_rms: float
    max_normalized_action_abs: float


@dataclass(frozen=True)
class DirectSteerResult:
    chunk_id: int
    request_id: int
    action_index: int
    action_vla_normalized: np.ndarray
    decoded_normalized: np.ndarray
    selected_normalized: np.ndarray
    selected_action: np.ndarray
    diagnostics: DirectSteerDiagnostics
    encode_started_at: float
    encode_finished_at: float
    decode_started_at: float
    decode_finished_at: float


@dataclass(frozen=True)
class _CachedRequest:
    chunk_id: int
    action_index: int
    tactile_digest: bytes
    result: DirectSteerResult


class DirectDecoderRuntime:
    """One visual Pi0.5 sample per chunk and one tactile decode per unique request."""

    def __init__(
        self,
        *,
        policy: Any,
        tactile_encoder: Any,
        decoder: Any,
        max_normalized_action_abs: float,
        max_normalized_delta_rms: float,
        device: str | None = None,
    ) -> None:
        for name, value in (("max_normalized_action_abs", max_normalized_action_abs), ("max_normalized_delta_rms", max_normalized_delta_rms)):
            if not isinstance(value, (int, float)) or isinstance(value, bool) or not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be a finite positive number")
        self.policy = policy
        self.tactile_encoder = tactile_encoder
        self.decoder = decoder
        self.max_normalized_action_abs = float(max_normalized_action_abs)
        self.max_normalized_delta_rms = float(max_normalized_delta_rms)
        self.device = device
        self.tactile_keys = tuple(getattr(tactile_encoder, "tactile_keys", TACTILE_KEYS))
        if self.tactile_keys != TACTILE_KEYS:
            raise ValueError("tactile encoder must use canonical tactile_keys")
        self._active_chunk_id: int | None = None
        self._coarse: np.ndarray | None = None
        self._cache: dict[int, _CachedRequest] = {}
        self._last_action_index: int | None = None

    def _clear(self) -> None:
        self._active_chunk_id = None
        self._coarse = None
        self._cache.clear()
        self._last_action_index = None

    def reset(self) -> None:
        self._clear()

    def cached_result(self, request_id: int) -> DirectSteerResult | None:
        cached = self._cache.get(request_id)
        return None if cached is None else cached.result

    def begin_chunk(
        self,
        chunk_id: int,
        observation: Mapping[str, Any],
        task: str,
        *,
        seed: int = 0,
        num_steps: int = 10,
    ) -> DirectChunkReady:
        if self._active_chunk_id is not None:
            raise RuntimeError("a direct steering chunk is already active")
        if isinstance(chunk_id, bool) or not isinstance(chunk_id, Integral):
            raise ValueError("chunk id must be an integer")
        prediction_started = perf_counter()
        coarse = _finite_shape(
            self.policy.predict_action_chunk(observation, task, seed=seed, num_steps=num_steps),
            _CHUNK_SHAPE,
            "Pi0.5 normalized action chunk",
        )
        prediction_finished = perf_counter()
        physical = _finite_shape(self.policy.unnormalize_actions(np.array(coarse, copy=True)), _CHUNK_SHAPE, "Pi0.5 physical action chunk")
        self._active_chunk_id = int(chunk_id)
        self._coarse = _immutable(coarse)
        return DirectChunkReady(
            chunk_id=int(chunk_id),
            action_vla_normalized=self._coarse,
            action_vla=_immutable(physical),
            prediction_started_at=prediction_started,
            prediction_finished_at=prediction_finished,
        )

    def _require_active(self, chunk_id: int) -> None:
        if self._active_chunk_id is None:
            raise RuntimeError("no active direct steering chunk")
        if chunk_id != self._active_chunk_id:
            raise ValueError("chunk id does not match the active chunk")

    def end_chunk(self, chunk_id: int) -> None:
        self._require_active(chunk_id)
        self._clear()

    def _payload_digest(self, observation: Mapping[str, Any]) -> bytes:
        missing = [key for key in self.tactile_keys if key not in observation]
        if missing:
            raise ValueError(f"observation is missing tactile keys: {missing}")
        digest = sha256()
        for key in self.tactile_keys:
            array = np.ascontiguousarray(np.asarray(observation[key]))
            for part in (key.encode(), array.dtype.str.encode(), repr(array.shape).encode(), array.tobytes(order="C")):
                digest.update(len(part).to_bytes(8, "big"))
                digest.update(part)
        return digest.digest()

    def _decode(self, coarse: np.ndarray, tactile: np.ndarray) -> Any:
        if hasattr(self.decoder, "decode"):
            return self.decoder.decode(coarse, tactile)
        try:
            import torch
        except ImportError as exc:  # pragma: no cover - deployment dependency guard
            raise RuntimeError("PyTorch is required for the deployment decoder") from exc
        target_device = self.device or next(self.decoder.parameters()).device
        with torch.inference_mode():
            output = self.decoder(
                torch.as_tensor(coarse, dtype=torch.float32, device=target_device),
                torch.as_tensor(tactile, dtype=torch.float32, device=target_device),
            )
        return output.detach().to(device="cpu", dtype=torch.float32).numpy()

    def steer_action(
        self,
        chunk_id: int,
        request_id: int,
        observation: Mapping[str, Any],
        action_index: int,
    ) -> DirectSteerResult:
        if isinstance(action_index, (bool, np.bool_)) or not isinstance(action_index, Integral):
            raise ValueError("action index must be an integer")
        action_index = int(action_index)
        if not 0 <= action_index < 50:
            raise ValueError("action index is outside [0,50)")
        tactile_digest = self._payload_digest(observation)
        cached = self._cache.get(request_id)
        if cached is not None:
            if (cached.chunk_id, cached.action_index, cached.tactile_digest) == (chunk_id, action_index, tactile_digest):
                return cached.result
            raise ValueError("conflicting duplicate direct steering request")
        self._require_active(chunk_id)
        if self._last_action_index is not None and action_index <= self._last_action_index:
            raise ValueError("unique action indices must be strictly increasing")
        assert self._coarse is not None
        encode_started = perf_counter()
        tactile = _finite_shape(self.tactile_encoder.encode(observation), _TOKEN_SHAPE, "tactile encoder tokens")
        encode_finished = perf_counter()
        decode_started = perf_counter()
        decoded = _finite_shape(self._decode(np.array(self._coarse, copy=True), np.array(tactile, copy=True)), _CHUNK_SHAPE, "direct decoder output")
        decode_finished = perf_counter()
        max_abs = float(np.max(np.abs(decoded)))
        if max_abs > self.max_normalized_action_abs:
            raise ValueError(f"direct decoder magnitude limit exceeded: {max_abs:.4f}")
        delta_rms = float(np.sqrt(np.mean(np.square(decoded - self._coarse, dtype=np.float64))))
        if delta_rms > self.max_normalized_delta_rms:
            raise ValueError(f"direct decoder delta limit exceeded: {delta_rms:.4f}")
        decoded_copy = _immutable(decoded)
        selected_normalized = _immutable(decoded_copy[0, action_index])
        selected_action = _immutable(_finite_shape(self.policy.unnormalize_actions(np.array(selected_normalized, copy=True)), (20,), "selected physical action"))
        result = DirectSteerResult(
            chunk_id=int(chunk_id), request_id=request_id, action_index=action_index,
            action_vla_normalized=self._coarse, decoded_normalized=decoded_copy,
            selected_normalized=selected_normalized, selected_action=selected_action,
            diagnostics=DirectSteerDiagnostics(delta_rms=delta_rms, max_normalized_action_abs=max_abs),
            encode_started_at=encode_started, encode_finished_at=encode_finished,
            decode_started_at=decode_started, decode_finished_at=decode_finished,
        )
        self._cache[request_id] = _CachedRequest(int(chunk_id), action_index, tactile_digest, result)
        self._last_action_index = action_index
        return result
