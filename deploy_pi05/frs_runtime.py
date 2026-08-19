"""Stateful tactile FRS runtime driven by a frozen JAX pi0.5 source policy."""

from __future__ import annotations

import hashlib
import math
import time
from collections import deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from numbers import Real
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from tactile_encoder.utils.checkpoint import load_tactile_encoder
from tactile_encoder.utils.image_dataset import parse_image_to_unit
from tactile_encoder.utils.model import tactile_clip_config_from_dict
from tactile_encoder.utils.resnet import encode_resnet18
from train_pi05_frs.utils.checkpoint import load_checkpoint as load_frs_checkpoint
from train_pi05_frs.utils.data import resolve_tactile_window, tactile_change_from_tokens
from train_pi05_frs.utils.model import DECODER_INPUT_VERSION, decode_actions

from .frs_config import validate_frs_config_section
from .policy import Pi05RemotePolicy


@dataclass(frozen=True)
class FRSDiagnostics:
    tactile_change: float
    delta_rms: float
    max_normalized_action_abs: float


@dataclass(frozen=True)
class FRSChunkReady:
    chunk_id: int
    action_vla_normalized: np.ndarray
    action_vla: np.ndarray
    x_base: np.ndarray
    prediction_started_at: float
    prediction_finished_at: float


@dataclass(frozen=True)
class FRSSteerResult:
    chunk_id: int
    request_id: int
    action_index: int
    action_vla_normalized: np.ndarray
    x_base: np.ndarray
    decoded_normalized: np.ndarray
    selected_normalized: np.ndarray
    selected_action: np.ndarray
    tactile_sequence_length: int
    diagnostics: FRSDiagnostics
    encode_started_at: float
    encode_finished_at: float
    decode_started_at: float
    decode_finished_at: float


@dataclass(frozen=True)
class FRSConfig:
    checkpoint: Path
    tactile_encoder_checkpoint: Path
    tactile_keys: tuple[str, ...]
    tactile_window_divisor: int
    history_stride: int
    reverse_steps: int
    reverse_solver: str
    decode_steps: int
    decode_solver: str
    steering_protection_interval_s: float | None
    temporal_ensemble_coeff: float | None
    gripper_gain: tuple[float, float] | None
    verify_source_checkpoint_fingerprint: bool
    max_normalized_action_abs: float
    max_normalized_delta_rms: float


def _local_directory(value: Any, *, config_path: Path, name: str) -> Path:
    path = Path(str(value)).expanduser()
    path = (config_path.parent / path).resolve() if not path.is_absolute() else path.resolve()
    if not path.is_dir():
        raise FileNotFoundError(f"{name} directory does not exist: {path}")
    return path


def _optional_nonnegative(value: Any, *, name: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be null or a finite non-negative number")
    parsed = float(value)
    if not math.isfinite(parsed) or parsed < 0:
        raise ValueError(f"{name} must be null or a finite non-negative number")
    return parsed


def _gripper_gain_config(value: Any) -> tuple[float, float] | None:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise ValueError("frs.gripper_gain must be null or a mapping")
    missing = {"threshold", "gain"} - set(value)
    if missing:
        raise ValueError(f"missing frs.gripper_gain values: {sorted(missing)}")
    threshold = float(value["threshold"])
    gain = float(value["gain"])
    if not math.isfinite(threshold) or not math.isfinite(gain) or gain < 0:
        raise ValueError("frs.gripper_gain threshold/gain must be finite and gain non-negative")
    return threshold, gain


def parse_frs_config(raw: Mapping[str, Any], *, config_path: Path) -> FRSConfig:
    deprecated = {"gate_tau", "gate_temperature"}.intersection(raw)
    if deprecated:
        raise ValueError(f"deprecated deployment Gate values: {sorted(deprecated)}")
    required = {
        "checkpoint",
        "tactile_encoder_checkpoint",
        "tactile_keys",
        "tactile_window_divisor",
        "reverse_steps",
        "reverse_solver",
        "decode_steps",
        "decode_solver",
    }
    missing = required - set(raw)
    if missing:
        raise ValueError(f"missing FRS config values: {sorted(missing)}")
    keys = raw["tactile_keys"]
    if (
        not isinstance(keys, list | tuple)
        or not keys
        or any(not isinstance(key, str) or not key for key in keys)
        or len(set(keys)) != len(keys)
    ):
        raise ValueError("frs.tactile_keys must be a non-empty unique list of strings")
    divisor = int(raw["tactile_window_divisor"])
    stride = int(raw.get("history_stride", 1))
    reverse_steps = int(raw["reverse_steps"])
    decode_steps = int(raw["decode_steps"])
    if min(divisor, stride, reverse_steps, decode_steps) <= 0:
        raise ValueError("FRS divisor/stride/reverse_steps/decode_steps must be positive")
    reverse_solver = str(raw["reverse_solver"])
    decode_solver = str(raw["decode_solver"])
    if reverse_solver not in {"euler", "fireflow", "slerpflow"}:
        raise ValueError("frs.reverse_solver must be euler, fireflow, or slerpflow")
    if decode_solver not in {"euler", "fireflow"}:
        raise ValueError("frs.decode_solver must be euler or fireflow")
    max_abs = float(raw.get("max_normalized_action_abs", 8.0))
    max_delta = float(raw.get("max_normalized_delta_rms", 4.0))
    if not math.isfinite(max_abs) or not math.isfinite(max_delta) or min(max_abs, max_delta) <= 0:
        raise ValueError("FRS normalized-action safety limits must be finite and positive")
    return FRSConfig(
        checkpoint=_local_directory(raw["checkpoint"], config_path=config_path, name="FRS checkpoint"),
        tactile_encoder_checkpoint=_local_directory(
            raw["tactile_encoder_checkpoint"],
            config_path=config_path,
            name="tactile encoder checkpoint",
        ),
        tactile_keys=tuple(keys),
        tactile_window_divisor=divisor,
        history_stride=stride,
        reverse_steps=reverse_steps,
        reverse_solver=reverse_solver,
        decode_steps=decode_steps,
        decode_solver=decode_solver,
        steering_protection_interval_s=_optional_nonnegative(
            raw.get("steering_protection_interval_s"),
            name="frs.steering_protection_interval_s",
        ),
        temporal_ensemble_coeff=_optional_nonnegative(
            raw.get("temporal_ensemble_coeff"),
            name="frs.temporal_ensemble_coeff",
        ),
        gripper_gain=_gripper_gain_config(raw.get("gripper_gain")),
        verify_source_checkpoint_fingerprint=bool(raw.get("verify_source_checkpoint_fingerprint", False)),
        max_normalized_action_abs=max_abs,
        max_normalized_delta_rms=max_delta,
    )


def _checkpoint_fingerprint(checkpoint_dir: Path) -> str:
    checkpoint_dir = checkpoint_dir.expanduser().resolve()
    if checkpoint_dir.name == "params":
        checkpoint_dir = checkpoint_dir.parent
    params_dir = checkpoint_dir / "params"
    if not params_dir.is_dir():
        raise FileNotFoundError(f"pi0.5 checkpoint has no params directory: {checkpoint_dir}")
    candidates = sorted(path for path in params_dir.rglob("*") if path.is_file())
    digest = hashlib.sha256()
    for path in candidates:
        stat = path.stat()
        digest.update(str(path.relative_to(checkpoint_dir)).encode())
        digest.update(f":{stat.st_size}:{stat.st_mtime_ns}\n".encode())
    return digest.hexdigest()


def _source_cache_configurations(extra: Mapping[str, Any]) -> tuple[Mapping[str, Any], ...]:
    cache = extra.get("cache_configuration")
    if not isinstance(cache, Mapping):
        raise ValueError("FRS checkpoint is missing extra_metadata.cache_configuration")
    sources = cache.get("sources")
    if sources is None:
        return (cache,)
    if not isinstance(sources, Sequence) or not sources:
        raise ValueError("FRS checkpoint cache_configuration.sources is invalid")
    output = []
    for source in sources:
        configuration = source.get("configuration") if isinstance(source, Mapping) else None
        if not isinstance(configuration, Mapping):
            raise ValueError("FRS checkpoint contains an invalid source cache configuration")
        output.append(configuration)
    return tuple(output)


def _require_equal(actual: Any, expected: Any, name: str) -> None:
    matches = (
        math.isclose(float(actual), expected, rel_tol=1e-7, abs_tol=1e-9)
        if isinstance(expected, float)
        else actual == expected
    )
    if not matches:
        raise ValueError(f"FRS checkpoint mismatch for {name}: {actual!r} != {expected!r}")


class TactileHistory:
    """Fixed-size, training-equivalent tactile history with clamped early frames."""

    def __init__(self, *, window: int, stride: int, token_shape: tuple[int, int]) -> None:
        if window <= 0 or stride <= 0:
            raise ValueError("window and stride must be positive")
        self.window = int(window)
        self.stride = int(stride)
        self.token_shape = token_shape
        self._frames: deque[np.ndarray] = deque(maxlen=(window - 1) * stride + 1)

    def _copy(self, tokens: Any) -> np.ndarray:
        array = np.asarray(tokens, dtype=np.float32)
        if array.shape != self.token_shape or not np.isfinite(array).all():
            raise ValueError(f"invalid tactile tokens: expected {self.token_shape}, got {array.shape}")
        return np.array(array, copy=True)

    def reset(self, tokens: Any) -> None:
        self._frames = deque((self._copy(tokens),), maxlen=self._frames.maxlen)

    def append(self, tokens: Any) -> None:
        self._frames.append(self._copy(tokens))

    def window_tokens(self) -> np.ndarray:
        if not self._frames:
            raise RuntimeError("tactile history is not initialized")
        frames = tuple(self._frames)
        current = len(frames) - 1
        indices = [max(0, current - offset * self.stride) for offset in reversed(range(self.window))]
        return np.stack([frames[index] for index in indices], axis=0)


class FRSRuntime:
    """Validate FRS provenance and serve per-action tactile steering requests."""

    def __init__(
        self,
        raw_config: Mapping[str, Any],
        *,
        config_path: Path,
        policy: Pi05RemotePolicy,
        source_sample_steps: int,
    ) -> None:
        self.policy = policy
        self.config = parse_frs_config(raw_config, config_path=config_path)
        self.model, self.metadata = load_frs_checkpoint(self.config.checkpoint)
        self.encoder = load_tactile_encoder(self.config.tactile_encoder_checkpoint)
        encoder_config = tactile_clip_config_from_dict(self.encoder.metadata["tactile_clip_config"])
        self.embedding_dim = int(encoder_config.embedding_dim)
        self.image_size = int(encoder_config.tactile_image_size)
        if "tactile_resnet" not in self.encoder.params:
            raise KeyError("tactile encoder checkpoint is missing tactile_resnet params")

        def encode_external(params: Any, images: jax.Array) -> jax.Array:
            embeddings, _ = encode_resnet18(
                params,
                images,
                train=False,
                embedding_dim=self.embedding_dim,
            )
            return embeddings

        self._encode_tactile = jax.jit(encode_external)
        self._validate_contract(source_sample_steps=source_sample_steps)
        decoder = self.model.config
        self.history = TactileHistory(
            window=int(decoder.tactile_window),
            stride=self.config.history_stride,
            token_shape=(len(self.config.tactile_keys), self.embedding_dim),
        )
        self.baseline: np.ndarray | None = None
        self._episode_baseline: np.ndarray | None = None
        self._clear_chunk_state()

    @property
    def tactile_keys(self) -> tuple[str, ...]:
        return self.config.tactile_keys

    def _validate_contract(self, *, source_sample_steps: int) -> None:
        decoder = self.model.config
        pc = self.policy.config
        _require_equal(decoder.action_dim, pc.action_dim, "action_dim")
        _require_equal(decoder.action_horizon, pc.action_horizon, "action_horizon")
        _require_equal(decoder.num_tactile_tokens, len(self.config.tactile_keys), "tactile tokens")
        _require_equal(decoder.resnet_embedding_dim, self.embedding_dim, "embedding_dim")
        _require_equal(decoder.decoder_input_version, DECODER_INPUT_VERSION, "decoder_input_version")
        if decoder.state_conditioning:
            _require_equal(decoder.state_dim, pc.state_dim, "state_dim")
        expected_window = resolve_tactile_window(
            action_horizon=pc.action_horizon,
            window_divisor=self.config.tactile_window_divisor,
        )
        _require_equal(decoder.tactile_window, expected_window, "tactile_window")
        extra = self.metadata.get("extra_metadata")
        if not isinstance(extra, Mapping):
            raise ValueError("FRS checkpoint is missing extra_metadata")
        _require_equal(extra.get("loss_mode"), "gated", "loss_mode")
        _require_equal(int(extra.get("history_stride", 0)), self.config.history_stride, "history_stride")
        _require_equal(str(extra.get("aux_decode_solver")), self.config.decode_solver, "decode_solver")
        _require_equal(int(extra.get("aux_decode_steps", 0)), self.config.decode_steps, "decode_steps")
        deployed_fingerprint = None
        if self.config.verify_source_checkpoint_fingerprint:
            deployed_fingerprint = _checkpoint_fingerprint(self.policy.checkpoint)
        for index, source in enumerate(_source_cache_configurations(extra)):
            prefix = f"cache source {index}"
            _require_equal(source.get("base_model"), "pi0.5", f"{prefix} base_model")
            _require_equal(
                int(source.get("model_sample_steps", 0)), source_sample_steps, f"{prefix} sample_steps"
            )
            _require_equal(
                int(source.get("reverse_steps", 0)), self.config.reverse_steps, f"{prefix} reverse_steps"
            )
            _require_equal(
                source.get("reverse_solver"), self.config.reverse_solver, f"{prefix} reverse_solver"
            )
            _require_equal(source.get("norm_stats_asset_id"), pc.asset_id, f"{prefix} norm asset")
            _require_equal(
                bool(source.get("use_quantile_norm", False)), pc.use_quantile_norm, f"{prefix} norm mode"
            )
            if deployed_fingerprint is not None:
                _require_equal(
                    source.get("checkpoint_fingerprint"), deployed_fingerprint, f"{prefix} checkpoint"
                )

    def resolved_tactile_window(self) -> int:
        return resolve_tactile_window(
            action_horizon=self.policy.config.action_horizon,
            window_divisor=self.config.tactile_window_divisor,
        )

    def _prepare_tactile(self, image: Any) -> np.ndarray:
        return parse_image_to_unit(image, image_size=self.image_size)[None, ...]

    def _encode_observation(self, observation: Mapping[str, Any]) -> np.ndarray:
        missing = [key for key in self.config.tactile_keys if key not in observation]
        if missing:
            raise ValueError(f"robot observation is missing tactile keys: {missing}")
        images = np.concatenate(
            [self._prepare_tactile(observation[key]) for key in self.config.tactile_keys],
            axis=0,
        )
        embeddings = self._encode_tactile(
            self.encoder.params["tactile_resnet"],
            jnp.asarray(images, dtype=jnp.float32),
        )
        return np.asarray(jax.device_get(embeddings), dtype=np.float32)

    def _normalized_state(self, observation: Mapping[str, Any]) -> jax.Array | None:
        if not self.model.config.state_conditioning:
            return None
        if "observation.state" not in observation:
            raise ValueError("robot observation is missing observation.state")
        state = np.asarray(observation["observation.state"], dtype=np.float32)[None, :]
        normalized = self.policy.normalize_state(state)
        expected = (1, int(self.model.config.state_dim))
        if normalized.shape != expected or not np.isfinite(np.asarray(normalized)).all():
            raise ValueError(f"normalized FRS state must be finite with shape {expected}")
        return normalized

    @staticmethod
    def _readonly(value: Any) -> np.ndarray:
        array = np.array(jax.device_get(value), dtype=np.float32, copy=True)
        array.setflags(write=False)
        return array

    @staticmethod
    def _public(value: Any) -> np.ndarray:
        array = np.asarray(jax.device_get(value), dtype=np.float32)
        return np.frombuffer(array.tobytes(order="C"), dtype=np.float32).reshape(array.shape)

    def _model_chunk(self, value: Any, *, name: str) -> np.ndarray:
        array = self._readonly(value)
        expected = (1, self.policy.config.action_horizon, self.policy.config.action_dim)
        if array.shape != expected or not np.isfinite(array).all():
            raise ValueError(f"{name} must be finite with shape {expected}, got {array.shape}")
        return array

    def _clear_chunk_state(self) -> None:
        self._active_chunk_id: int | None = None
        self._action_vla_normalized: np.ndarray | None = None
        self._action_vla: np.ndarray | None = None
        self._x_base: np.ndarray | None = None
        self._x_base_device: jax.Array | None = None
        self._request_results: dict[int, tuple[int, int, bytes, FRSSteerResult]] = {}
        self._last_action_index: int | None = None
        self.last_diagnostics: FRSDiagnostics | None = None
        self.last_vla_normalized: np.ndarray | None = None
        self.last_frs_normalized: np.ndarray | None = None

    def reset_episode(self, initial_observation: Mapping[str, Any]) -> None:
        baseline = self._readonly(self._encode_observation(initial_observation))
        history = TactileHistory(
            window=self.history.window,
            stride=self.history.stride,
            token_shape=self.history.token_shape,
        )
        history.reset(baseline)
        self._episode_baseline = baseline
        self.baseline = self._readonly(baseline)
        self.history = history
        self._clear_chunk_state()

    def warmup(
        self,
        observation: Mapping[str, Any],
        task: str,
        *,
        seed: int,
        sample_steps: int,
    ) -> None:
        if self._episode_baseline is None:
            raise RuntimeError("reset_episode must be called before warmup")
        normalized = self.policy.predict_action_chunk(observation, task, seed=seed, num_steps=sample_steps)
        x_base = self.policy.reverse_action_chunk(
            observation,
            task,
            normalized,
            num_steps=self.config.reverse_steps,
            solver=self.config.reverse_solver,
        )
        kwargs: dict[str, Any] = {}
        state = self._normalized_state(observation)
        if state is not None:
            kwargs["state"] = state
        decoded = decode_actions(
            self.model,
            x_base,
            jnp.asarray(self.history.window_tokens()[None, ...]),
            num_steps=self.config.decode_steps,
            solver=self.config.decode_solver,
            **kwargs,
        )
        jax.block_until_ready(decoded)

    def begin_chunk(
        self,
        chunk_id: int,
        observation: Mapping[str, Any],
        task: str,
        *,
        seed: int,
        num_steps: int,
    ) -> FRSChunkReady:
        if self._episode_baseline is None:
            raise RuntimeError("reset_episode must be called before begin_chunk")
        if self._active_chunk_id is not None:
            raise RuntimeError(f"active chunk {self._active_chunk_id} must be ended first")
        started = time.time()
        normalized = self.policy.predict_action_chunk(observation, task, seed=seed, num_steps=num_steps)
        x_base = self.policy.reverse_action_chunk(
            observation,
            task,
            normalized,
            num_steps=self.config.reverse_steps,
            solver=self.config.reverse_solver,
        )
        normalized_array = self._model_chunk(normalized, name="normalized pi0.5 actions")
        x_base_array = self._model_chunk(x_base, name="reverse-flow base")
        robot_actions = self._readonly(self.policy.unnormalize_actions(normalized_array))
        expected_robot = (
            1,
            self.policy.config.action_horizon,
            self.policy.config.robot_action_dim,
        )
        if robot_actions.shape != expected_robot:
            raise ValueError(f"robot action chunk must have shape {expected_robot}")
        self._clear_chunk_state()
        self._active_chunk_id = chunk_id
        self._action_vla_normalized = normalized_array
        self._action_vla = robot_actions
        self._x_base = x_base_array
        self._x_base_device = jnp.asarray(x_base_array, dtype=jnp.float32)
        return FRSChunkReady(
            chunk_id=chunk_id,
            action_vla_normalized=self._public(normalized_array),
            action_vla=self._public(robot_actions),
            x_base=self._public(x_base_array),
            prediction_started_at=started,
            prediction_finished_at=time.time(),
        )

    def _require_active(self, chunk_id: int) -> None:
        if self._active_chunk_id is None:
            raise RuntimeError("FRS has no active chunk")
        if chunk_id != self._active_chunk_id:
            raise ValueError(f"chunk id {chunk_id} does not match active {self._active_chunk_id}")

    def end_chunk(self, chunk_id: int) -> None:
        self._require_active(chunk_id)
        self._clear_chunk_state()

    def _payload_hash(self, observation: Mapping[str, Any]) -> bytes:
        digest = hashlib.sha256()
        for key in (*self.config.tactile_keys, "observation.state"):
            if key not in observation:
                raise ValueError(f"robot observation is missing key: {key}")
            value = np.ascontiguousarray(np.asarray(observation[key]))
            digest.update(key.encode())
            digest.update(value.dtype.str.encode())
            digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
            digest.update(value.tobytes(order="C"))
        return digest.digest()

    def _validated_decoded(self, decoded: Any) -> tuple[np.ndarray, float, float]:
        assert self._action_vla_normalized is not None
        array = np.asarray(jax.device_get(decoded), dtype=np.float32)
        expected = (1, self.policy.config.action_horizon, self.policy.config.action_dim)
        if array.shape != expected or not np.isfinite(array).all():
            raise ValueError(f"FRS output must be finite with shape {expected}, got {array.shape}")
        delta = float(np.sqrt(np.mean(np.square(array - self._action_vla_normalized))))
        max_abs = float(np.max(np.abs(array)))
        if max_abs > self.config.max_normalized_action_abs:
            raise ValueError(
                f"FRS normalized action safety limit exceeded: {max_abs:.4f} > "
                f"{self.config.max_normalized_action_abs:.4f}"
            )
        if delta > self.config.max_normalized_delta_rms:
            raise ValueError(
                f"FRS normalized delta safety limit exceeded: {delta:.4f} > "
                f"{self.config.max_normalized_delta_rms:.4f}"
            )
        return array, delta, max_abs

    def steer_action(
        self,
        chunk_id: int,
        request_id: int,
        observation: Mapping[str, Any],
        action_index: int,
    ) -> FRSSteerResult:
        tactile_hash = self._payload_hash(observation)
        cached = self._request_results.get(request_id)
        if cached is not None:
            cached_chunk, cached_index, cached_hash, result = cached
            if (chunk_id, action_index, tactile_hash) != (
                cached_chunk,
                cached_index,
                cached_hash,
            ):
                raise ValueError(f"conflicting duplicate FRS request id {request_id}")
            return result
        self._require_active(chunk_id)
        horizon = self.policy.config.action_horizon
        if not isinstance(action_index, int) or isinstance(action_index, bool):
            raise ValueError("action_index must be an integer")
        if not 0 <= action_index < horizon:
            raise ValueError(f"action_index is outside [0,{horizon}): {action_index}")
        if self._last_action_index is not None and action_index <= self._last_action_index:
            raise ValueError("unique FRS requests must have strictly increasing action_index")
        assert self._episode_baseline is not None
        assert self._action_vla_normalized is not None
        assert self._x_base is not None
        assert self._x_base_device is not None

        encode_started = time.time()
        current = self._readonly(self._encode_observation(observation))
        self.history.append(current)
        tactile = jnp.asarray(self.history.window_tokens()[None, ...], dtype=jnp.float32)
        change = tactile_change_from_tokens(current[None, ...], self._episode_baseline[None, ...])
        encode_finished = time.time()
        kwargs: dict[str, Any] = {}
        state = self._normalized_state(observation)
        if state is not None:
            kwargs["state"] = state
        decode_started = time.time()
        decoded = decode_actions(
            self.model,
            self._x_base_device,
            tactile,
            num_steps=self.config.decode_steps,
            solver=self.config.decode_solver,
            **kwargs,
        )
        decoded_array, delta, max_abs = self._validated_decoded(decoded)
        decode_finished = time.time()
        selected = decoded_array[0, action_index]
        if self.config.temporal_ensemble_coeff is not None:
            candidates = [
                item.decoded_normalized[0, action_index] for *_, item in self._request_results.values()
            ]
            candidates.append(selected)
            ages = np.arange(len(candidates) - 1, -1, -1)
            weights = np.exp(-self.config.temporal_ensemble_coeff * ages)
            selected = np.average(np.stack(candidates), axis=0, weights=weights)
        selected_normalized = self._public(selected)
        robot_selected = np.asarray(self.policy.unnormalize_actions(selected_normalized), dtype=np.float32)
        expected_robot = (self.policy.config.robot_action_dim,)
        if robot_selected.shape != expected_robot or not np.isfinite(robot_selected).all():
            raise ValueError(f"selected robot action must be finite with shape {expected_robot}")
        if self.config.gripper_gain is not None:
            if robot_selected.shape != (20,):
                raise ValueError("gripper gain requires a 20-dimensional robot action")
            threshold, gain = self.config.gripper_gain
            robot_selected = robot_selected.copy()
            indices = np.asarray((9, 19))
            robot_selected[indices] = np.where(
                robot_selected[indices] < threshold,
                robot_selected[indices] - gain,
                robot_selected[indices],
            )
        diagnostics = FRSDiagnostics(float(change[0]), delta, max_abs)
        result = FRSSteerResult(
            chunk_id=chunk_id,
            request_id=request_id,
            action_index=action_index,
            action_vla_normalized=self._public(self._action_vla_normalized),
            x_base=self._public(self._x_base),
            decoded_normalized=self._public(decoded_array),
            selected_normalized=selected_normalized,
            selected_action=self._public(robot_selected),
            tactile_sequence_length=self.history.window,
            diagnostics=diagnostics,
            encode_started_at=encode_started,
            encode_finished_at=encode_finished,
            decode_started_at=decode_started,
            decode_finished_at=decode_finished,
        )
        self._request_results[request_id] = (chunk_id, action_index, tactile_hash, result)
        self._last_action_index = action_index
        self.last_diagnostics = diagnostics
        self.last_vla_normalized = self._readonly(self._action_vla_normalized)
        self.last_frs_normalized = self._readonly(decoded_array)
        return result
