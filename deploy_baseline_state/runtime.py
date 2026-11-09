"""Structured-noise chunks with one state-only Flow decode per unique action request."""

from __future__ import annotations
from dataclasses import dataclass
from hashlib import sha256
from numbers import Integral
from time import perf_counter
import numpy as np
from .right_arm_adapter import expand_right_action, project_right_observation


def _immutable(value):
    array = np.ascontiguousarray(np.asarray(value, dtype=np.float32))
    return np.frombuffer(array.tobytes(), dtype=np.float32).reshape(array.shape)


def _finite_shape(value, shape, name):
    array = np.asarray(value, dtype=np.float32)
    if array.shape != shape or not np.isfinite(array).all():
        raise ValueError(f"{name} must be finite with shape {shape}")
    return array


@dataclass(frozen=True)
class StateChunkReady:
    chunk_id: int
    action_vla_normalized: np.ndarray
    action_vla: np.ndarray
    structured_noise: np.ndarray
    prediction_started_at: float
    prediction_finished_at: float


@dataclass(frozen=True)
class StateSteerDiagnostics:
    delta_rms: float
    max_normalized_action_abs: float


@dataclass(frozen=True)
class StateSteerResult:
    chunk_id: int
    request_id: int
    action_index: int
    action_vla_normalized: np.ndarray
    decoded_normalized: np.ndarray
    selected_normalized: np.ndarray
    selected_action: np.ndarray
    diagnostics: StateSteerDiagnostics
    encode_started_at: float
    encode_finished_at: float
    decode_started_at: float
    decode_finished_at: float


@dataclass(frozen=True)
class _CachedRequest:
    chunk_id: int
    action_index: int
    state_digest: bytes
    result: StateSteerResult


class StateDecoderRuntime:
    def __init__(
        self,
        *,
        policy,
        decoder,
        max_normalized_action_abs,
        max_normalized_delta_rms,
        device=None,
        action_dim=20,
    ):
        if type(action_dim) is not int or action_dim not in (10, 20):
            raise ValueError("action_dim must be 10 or 20")
        for value in (max_normalized_action_abs, max_normalized_delta_rms):
            if not np.isfinite(value) or value <= 0:
                raise ValueError("action limits must be finite and positive")
        self.policy, self.decoder, self.device = policy, decoder, device
        self.action_dim = action_dim
        self.state_dim = 7 if action_dim == 10 else 20
        self.max_normalized_action_abs = max_normalized_action_abs
        self.max_normalized_delta_rms = max_normalized_delta_rms
        self._chunk_shape = (1, 50, action_dim)
        self.reset()

    def reset(self):
        self._active_chunk_id = None
        self._coarse = None
        self._structured_noise = None
        self._cache = {}
        self._last_action_index = None

    def reset_episode(self, observation):
        self.reset()

    def cached_result(self, request_id):
        item = self._cache.get(request_id)
        return None if item is None else item.result

    def begin_chunk(self, chunk_id, observation, task, *, seed=0, num_steps=10):
        if self._active_chunk_id is not None:
            raise RuntimeError("an active chunk must end first")
        if isinstance(chunk_id, bool) or not isinstance(chunk_id, Integral):
            raise ValueError("invalid chunk id")
        started = perf_counter()
        model_observation = project_right_observation(observation) if self.action_dim == 10 else observation
        coarse, noise = self.policy.predict_and_reverse_action_chunk(
            model_observation, task, seed=seed, num_steps=num_steps
        )
        coarse = _finite_shape(coarse, self._chunk_shape, "VLA normalized action")
        noise = _finite_shape(noise, self._chunk_shape, "structured noise")
        physical = _finite_shape(
            self.policy.unnormalize_actions(coarse), self._chunk_shape, "VLA physical action"
        )
        if self.action_dim == 10:
            physical = expand_right_action(physical[0], observation)[None, ...]
        self._active_chunk_id = int(chunk_id)
        self._coarse = _immutable(coarse)
        self._structured_noise = _immutable(noise)
        return StateChunkReady(
            int(chunk_id), self._coarse, _immutable(physical), self._structured_noise, started, perf_counter()
        )

    def _require_active(self, chunk_id):
        if self._active_chunk_id is None:
            raise RuntimeError("no active state Flow chunk")
        if chunk_id != self._active_chunk_id:
            raise ValueError("chunk id does not match the active chunk")

    def end_chunk(self, chunk_id):
        self._require_active(chunk_id)
        self.reset()

    def _decode(self, state):
        coarse = np.array(self._coarse, copy=True)
        noise = np.array(self._structured_noise, copy=True)
        if not hasattr(self.decoder, "parameters"):
            return self.decoder.decode(noise, state=state, vla_action=coarse)
        import torch

        device = self.device or next(self.decoder.parameters()).device
        with torch.inference_mode():
            output = self.decoder.decode(
                torch.as_tensor(noise, device=device),
                state=torch.as_tensor(state, device=device),
                vla_action=torch.as_tensor(coarse, device=device),
            )
        return output.detach().cpu().float().numpy()

    def steer_action(self, chunk_id, request_id, observation, action_index):
        self._require_active(chunk_id)
        if (
            isinstance(action_index, bool)
            or not isinstance(action_index, Integral)
            or not 0 <= action_index < 50
        ):
            raise ValueError("action_index must be an integer in [0,50)")
        if "observation.state" not in observation:
            raise ValueError("missing observation.state")
        raw = np.ascontiguousarray(np.asarray(observation["observation.state"], dtype=np.float32))
        digest = sha256(raw.tobytes() + str(raw.shape).encode()).digest()
        cached = self._cache.get(request_id)
        if cached is not None:
            if (cached.chunk_id, cached.action_index, cached.state_digest) == (
                chunk_id,
                action_index,
                digest,
            ):
                return cached.result
            raise ValueError("conflicting duplicate state Flow request")
        if self._last_action_index is not None and action_index <= self._last_action_index:
            raise ValueError("unique action indices must be strictly increasing")
        encode_start = perf_counter()
        projected = project_right_observation(observation) if self.action_dim == 10 else observation
        state = _finite_shape(
            self.policy.normalize_state(np.asarray(projected["observation.state"])[None, :]),
            (1, self.state_dim),
            "normalized state",
        )
        encode_end = perf_counter()
        decode_start = perf_counter()
        decoded = _finite_shape(self._decode(state), self._chunk_shape, "state Flow output").copy()
        decoded[..., 9::10] = self._coarse[..., 9::10]
        maximum = float(np.abs(decoded).max())
        delta = float(np.sqrt(np.mean((decoded - self._coarse).astype(np.float64) ** 2)))
        if maximum > self.max_normalized_action_abs:
            raise ValueError("state Flow magnitude limit exceeded")
        if delta > self.max_normalized_delta_rms:
            raise ValueError("state Flow delta limit exceeded")
        selected = _immutable(decoded[0, int(action_index)])
        physical = _finite_shape(
            self.policy.unnormalize_actions(selected), (self.action_dim,), "selected physical action"
        )
        if self.action_dim == 10:
            physical = expand_right_action(physical[None, :], observation)[0]
        result = StateSteerResult(
            int(chunk_id),
            int(request_id),
            int(action_index),
            self._coarse,
            _immutable(decoded),
            selected,
            _immutable(physical),
            StateSteerDiagnostics(delta, maximum),
            encode_start,
            encode_end,
            decode_start,
            perf_counter(),
        )
        self._cache[request_id] = _CachedRequest(int(chunk_id), int(action_index), digest, result)
        self._last_action_index = int(action_index)
        return result
