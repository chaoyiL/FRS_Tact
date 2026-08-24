"""Stateful tactile Flow Re-Steering runtime for remote SmolVLA deployment."""

from __future__ import annotations

import hashlib
import logging
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

from modalities_eval.utils import EvalObservation
from train_encoder.utils.checkpoint import load_tactile_encoder
from train_encoder.utils.model import tactile_clip_config_from_dict
from train_encoder.utils.resnet import encode_resnet18
from train_smolvla_frs.utils.bimanual_schema import validate_bimanual_objective_metadata
from train_smolvla_frs.utils.checkpoint import load_checkpoint as load_frs_checkpoint
from train_smolvla_frs.utils.data import resolve_tactile_window, tactile_change_from_tokens
from train_smolvla_frs.utils.model import decode_actions
from train_vtsmolvla.preprocessing import prepare_tactile_batch
from utils.integration import REVERSE_INTEGRATION_VERSION
from utils.source_model import reverse_integrate_actions


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
    inactive_arm_xyz_threshold_m: float | None
    gripper_gain: tuple[float, float, float] | None
    verify_source_checkpoint_fingerprint: bool
    max_normalized_action_abs: float
    max_normalized_delta_rms: float


def _local_directory(value: Any, *, config_path: Path, name: str) -> Path:
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        path = (config_path.parent / path).resolve()
    else:
        path = path.resolve()
    if not path.is_dir():
        raise FileNotFoundError(f"{name} directory does not exist: {path}")
    return path


def _steering_protection_interval(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(
            "frs.steering_protection_interval_s must be null or a finite "
            "non-negative number"
        )
    interval = float(value)
    if not math.isfinite(interval) or interval < 0:
        raise ValueError(
            "frs.steering_protection_interval_s must be null or a finite "
            "non-negative number"
        )
    return interval


def _temporal_ensemble_coefficient(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(
            "frs.temporal_ensemble_coeff must be null or a finite non-negative number"
        )
    coefficient = float(value)
    if not math.isfinite(coefficient) or coefficient < 0:
        raise ValueError(
            "frs.temporal_ensemble_coeff must be null or a finite non-negative number"
        )
    return coefficient


def _inactive_arm_xyz_threshold(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(
            "frs.inactive_arm_xyz_threshold_m must be null or a finite positive number"
        )
    threshold = float(value)
    if not math.isfinite(threshold) or threshold <= 0:
        raise ValueError(
            "frs.inactive_arm_xyz_threshold_m must be null or a finite positive number"
        )
    return threshold


def _protect_inactive_arm_xyz(
    selected_normalized: Any,
    vla_normalized: Any,
    vla_action: Any,
    threshold_m: float,
) -> np.ndarray:
    selected = np.asarray(selected_normalized)
    vla_normalized_array = np.asarray(vla_normalized)
    vla_action_array = np.asarray(vla_action)
    expected = (20,)
    if (
        selected.shape != expected
        or vla_normalized_array.shape != expected
        or vla_action_array.shape != expected
    ):
        raise ValueError(
            "FRS inactive-arm XYZ protection requires a 20-dimensional action"
        )

    protected = np.array(selected, copy=True)
    action_threshold = np.asarray(threshold_m, dtype=vla_action_array.dtype)
    for start in (0, 10):
        xyz_slice = slice(start, start + 3)
        if np.max(np.abs(vla_action_array[xyz_slice])) < action_threshold:
            protected[xyz_slice] = vla_normalized_array[xyz_slice]
    return protected


def _gripper_gain_config(value: Any) -> tuple[float, float, float] | None:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise ValueError("frs.gripper_gain must be null or a mapping")
    names = ("threshold", "multiplier", "above_multiplier")
    missing = [name for name in names if name not in value]
    if missing:
        raise ValueError(f"Missing frs.gripper_gain config values: {missing}")
    parsed = []
    for name in names:
        raw = value[name]
        if isinstance(raw, bool) or not isinstance(raw, Real):
            raise ValueError(f"frs.gripper_gain.{name} must be a finite number")
        number = float(raw)
        if not math.isfinite(number):
            raise ValueError(f"frs.gripper_gain.{name} must be a finite number")
        parsed.append(number)
    threshold, multiplier, above_multiplier = parsed
    if multiplier <= 0:
        raise ValueError("frs.gripper_gain.multiplier must be positive")
    if above_multiplier <= 0:
        raise ValueError("frs.gripper_gain.above_multiplier must be positive")
    return threshold, multiplier, above_multiplier


def _positive_history_stride(raw: Mapping[str, Any]) -> int:
    if "history_stride" not in raw:
        return 1
    stride = int(raw["history_stride"])
    if stride <= 0:
        raise ValueError("frs.history_stride must be positive")
    return stride


def _reject_deprecated_gate_config(raw: Mapping[str, Any]) -> None:
    deprecated = {"gate_tau", "gate_temperature"}.intersection(raw)
    if deprecated:
        raise ValueError(f"Deprecated FRS gate config values: {sorted(deprecated)}")


def parse_frs_config(raw: Mapping[str, Any], *, config_path: Path) -> FRSConfig:
    _reject_deprecated_gate_config(raw)
    required = (
        "checkpoint",
        "tactile_encoder_checkpoint",
        "tactile_keys",
        "tactile_window_divisor",
        "reverse_steps",
        "reverse_solver",
        "decode_steps",
        "decode_solver",
    )
    missing = [key for key in required if key not in raw]
    if missing:
        raise ValueError(f"Missing FRS config values: {missing}")
    tactile_keys_value = raw["tactile_keys"]
    if (
        not isinstance(tactile_keys_value, list | tuple)
        or not tactile_keys_value
        or any(not isinstance(key, str) or not key for key in tactile_keys_value)
    ):
        raise ValueError("frs.tactile_keys must be a non-empty list of strings")
    tactile_keys = tuple(tactile_keys_value)
    if len(set(tactile_keys)) != len(tactile_keys):
        raise ValueError("frs.tactile_keys must not contain duplicates")

    tactile_window_divisor = int(raw["tactile_window_divisor"])
    history_stride = _positive_history_stride(raw)
    reverse_steps = int(raw["reverse_steps"])
    decode_steps = int(raw["decode_steps"])
    max_action_abs = float(raw.get("max_normalized_action_abs", 8.0))
    max_delta_rms = float(raw.get("max_normalized_delta_rms", 4.0))
    steering_protection_interval_s = _steering_protection_interval(
        raw.get("steering_protection_interval_s")
    )
    temporal_ensemble_coeff = _temporal_ensemble_coefficient(
        raw.get("temporal_ensemble_coeff")
    )
    inactive_arm_xyz_threshold_m = _inactive_arm_xyz_threshold(
        raw.get("inactive_arm_xyz_threshold_m")
    )
    gripper_gain = _gripper_gain_config(raw.get("gripper_gain"))
    if min(tactile_window_divisor, history_stride, reverse_steps, decode_steps) <= 0:
        raise ValueError(
            "frs tactile_window_divisor/history_stride/reverse_steps/decode_steps must be positive"
        )
    if max_action_abs <= 0 or max_delta_rms <= 0:
        raise ValueError("FRS normalized-action safety limits must be positive")
    reverse_solver = str(raw["reverse_solver"])
    decode_solver = str(raw["decode_solver"])
    if reverse_solver not in {"euler", "fireflow", "slerpflow"}:
        raise ValueError("frs.reverse_solver must be euler, fireflow, or slerpflow")
    if decode_solver not in {"euler", "fireflow"}:
        raise ValueError("frs.decode_solver must be euler or fireflow")

    return FRSConfig(
        checkpoint=_local_directory(raw["checkpoint"], config_path=config_path, name="FRS checkpoint"),
        tactile_encoder_checkpoint=_local_directory(
            raw["tactile_encoder_checkpoint"],
            config_path=config_path,
            name="tactile encoder checkpoint",
        ),
        tactile_keys=tactile_keys,
        tactile_window_divisor=tactile_window_divisor,
        history_stride=history_stride,
        reverse_steps=reverse_steps,
        reverse_solver=reverse_solver,
        decode_steps=decode_steps,
        decode_solver=decode_solver,
        steering_protection_interval_s=steering_protection_interval_s,
        temporal_ensemble_coeff=temporal_ensemble_coeff,
        inactive_arm_xyz_threshold_m=inactive_arm_xyz_threshold_m,
        gripper_gain=gripper_gain,
        verify_source_checkpoint_fingerprint=bool(raw.get("verify_source_checkpoint_fingerprint", True)),
        max_normalized_action_abs=max_action_abs,
        max_normalized_delta_rms=max_delta_rms,
    )


def validate_frs_config_section(config: Mapping[str, Any]) -> None:
    """Validate values that do not require checkpoint files to exist."""

    raw = config.get("frs")
    if raw is None:
        return
    if not isinstance(raw, Mapping):
        raise ValueError("frs must be a mapping")
    _reject_deprecated_gate_config(raw)
    if raw.get("enabled", True) is False:
        return
    for key in (
        "checkpoint",
        "tactile_encoder_checkpoint",
        "tactile_keys",
        "tactile_window_divisor",
        "reverse_steps",
        "reverse_solver",
        "decode_steps",
        "decode_solver",
    ):
        if key not in raw:
            raise ValueError(f"Missing config value frs.{key}")
    keys = raw["tactile_keys"]
    if not isinstance(keys, list | tuple) or not keys or any(not isinstance(key, str) or not key for key in keys):
        raise ValueError("frs.tactile_keys must be a non-empty list of strings")
    if len(set(keys)) != len(keys):
        raise ValueError("frs.tactile_keys must not contain duplicates")
    if (
        min(
            int(raw["tactile_window_divisor"]),
            _positive_history_stride(raw),
            int(raw["reverse_steps"]),
            int(raw["decode_steps"]),
        )
        <= 0
    ):
        raise ValueError(
            "frs tactile_window_divisor/history_stride/reverse_steps/decode_steps must be positive"
        )
    if str(raw["reverse_solver"]) not in {"euler", "fireflow", "slerpflow"}:
        raise ValueError("frs.reverse_solver must be euler, fireflow, or slerpflow")
    if str(raw["decode_solver"]) not in {"euler", "fireflow"}:
        raise ValueError("frs.decode_solver must be euler or fireflow")
    _steering_protection_interval(raw.get("steering_protection_interval_s"))
    _temporal_ensemble_coefficient(raw.get("temporal_ensemble_coeff"))
    _inactive_arm_xyz_threshold(raw.get("inactive_arm_xyz_threshold_m"))
    _gripper_gain_config(raw.get("gripper_gain"))
    if config.get("observation", {}).get("data_type") != "vitac":
        raise ValueError("FRS deployment requires observation.data_type='vitac'")
    control = config.get("control", {})
    if (
        control.get("steps_per_inference") is not None
        and control.get("action_horizon") is not None
        and int(control["steps_per_inference"]) != int(control["action_horizon"])
    ):
        raise ValueError(
            "FRS deployment requires steps_per_inference to equal action_horizon"
        )
    if control.get("action_horizon") is not None:
        resolve_tactile_window(
            action_horizon=int(control["action_horizon"]),
            window_divisor=int(raw["tactile_window_divisor"]),
        )


def _checkpoint_fingerprint(checkpoint_dir: Path) -> str:
    """Match the source-checkpoint identity stored by ``train_smolvla_frs.prepare_frs_caches``."""

    checkpoint_dir = checkpoint_dir.expanduser().resolve()
    if checkpoint_dir.name == "params":
        checkpoint_dir = checkpoint_dir.parent
    params_dir = checkpoint_dir / "params"
    model_file = checkpoint_dir / "model.safetensors"
    if params_dir.is_dir():
        candidates = sorted(path for path in params_dir.rglob("*") if path.is_file())
    elif model_file.is_file():
        candidates = [model_file]
    else:
        raise FileNotFoundError(
            f"Checkpoint params not found under {checkpoint_dir}: expected params/ or model.safetensors"
        )
    for name in ("config.json", "conversion_manifest.json"):
        path = checkpoint_dir / name
        if path.is_file():
            candidates.append(path)
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
        if not isinstance(source, Mapping) or not isinstance(source.get("configuration"), Mapping):
            raise ValueError("FRS checkpoint contains an invalid source cache configuration")
        output.append(source["configuration"])
    return tuple(output)


def _require_equal(actual: Any, expected: Any, name: str) -> None:
    if isinstance(expected, float):
        matches = math.isclose(float(actual), expected, rel_tol=1e-7, abs_tol=1e-9)
    else:
        matches = actual == expected
    if not matches:
        raise ValueError(f"FRS checkpoint mismatch for {name}: {actual!r} != {expected!r}")


class TactileHistory:
    """Episode-local embedding history with training-equivalent clamped indexing."""

    def __init__(self, *, window: int, stride: int, token_shape: tuple[int, int]) -> None:
        if window <= 0 or stride <= 0:
            raise ValueError("window and stride must be positive")
        self.window = int(window)
        self.stride = int(stride)
        self.token_shape = tuple(token_shape)
        self._frames: deque[np.ndarray] = deque(maxlen=(window - 1) * stride + 1)

    def _validated_copy(self, tokens: np.ndarray) -> np.ndarray:
        tokens = np.asarray(tokens, dtype=np.float32)
        if tokens.shape != self.token_shape:
            raise ValueError(f"expected tactile tokens {self.token_shape}, got {tokens.shape}")
        if not np.isfinite(tokens).all():
            raise ValueError("tactile encoder produced NaN or Inf")
        return np.array(tokens, copy=True)

    def reset(self, tokens: np.ndarray) -> None:
        prepared = self._validated_copy(tokens)
        self._frames = deque((prepared,), maxlen=self._frames.maxlen)

    def append(self, tokens: np.ndarray) -> None:
        self._frames.append(self._validated_copy(tokens))

    def window_tokens_after(self, tokens: np.ndarray) -> np.ndarray:
        """Preview the fixed window after an append without mutating history."""

        prepared = self._validated_copy(tokens)
        frames = (*self._frames, prepared)
        maxlen = self._frames.maxlen
        if maxlen is not None and len(frames) > maxlen:
            frames = frames[-maxlen:]
        current = len(frames) - 1
        indices = [
            max(0, current - offset * self.stride)
            for offset in reversed(range(self.window))
        ]
        return np.stack([frames[index] for index in indices], axis=0)

    def window_tokens(self) -> np.ndarray:
        if not self._frames:
            raise RuntimeError("FRS tactile history is not initialized")
        frames = tuple(self._frames)
        current = len(frames) - 1
        indices = [max(0, current - offset * self.stride) for offset in reversed(range(self.window))]
        return np.stack([frames[index] for index in indices], axis=0)


class FRSSteeringPolicy:
    """Load, validate and run FRS steering without silently falling back to VLA."""

    def __init__(
        self,
        raw_config: Mapping[str, Any],
        *,
        config_path: Path,
        policy: Any,
        source_sample_steps: int,
    ) -> None:
        self.policy = policy
        self.config = parse_frs_config(raw_config, config_path=config_path)
        if bool(getattr(policy.config, "use_tactile_encoder", False)):
            raise ValueError("FRS deployment requires a visual SmolVLA source checkpoint")
        if getattr(policy.config, "rtc_config", None) is not None and bool(
            getattr(policy.config.rtc_config, "enabled", False)
        ):
            raise ValueError("FRS deployment does not support RTC action stitching")
        if bool(getattr(policy.config, "adapt_to_pi_aloha", False)):
            raise ValueError("FRS deployment requires adapt_to_pi_aloha=false")

        self.model, self.metadata = load_frs_checkpoint(self.config.checkpoint)
        self.encoder = None
        if self.model.config.tactile_encoder_trainable:
            self.embedding_dim = int(self.model.config.resnet_embedding_dim)
            self.image_size = int(self.model.config.tactile_image_size)
        else:
            self.encoder = load_tactile_encoder(
                self.config.tactile_encoder_checkpoint
            )
            encoder_config = tactile_clip_config_from_dict(
                self.encoder.metadata["tactile_clip_config"]
            )
            self.embedding_dim = int(encoder_config.embedding_dim)
            self.image_size = int(encoder_config.tactile_image_size)
            if "tactile_resnet" not in self.encoder.params:
                raise KeyError(
                    "tactile encoder checkpoint is missing tactile_resnet params"
                )
        self._validate_contract(policy, source_sample_steps=source_sample_steps)

        decoder = self.model.config
        self.history = TactileHistory(
            window=int(decoder.tactile_window),
            stride=int(self.config.history_stride),
            token_shape=(len(self.config.tactile_keys), self.embedding_dim),
        )
        self.baseline: np.ndarray | None = None
        self._episode_baseline: np.ndarray | None = None
        self._clear_chunk_state()

        if self.model.config.tactile_encoder_trainable:
            from flax import nnx

            @nnx.jit
            def encode_embedded(model: Any, images: jax.Array) -> jax.Array:
                return model.encode_tactile_images(
                    images[None, None, ...]
                )[0, 0]

            self._encode_tactile = encode_embedded
        else:
            def encode_external(params: Any, images: jax.Array) -> jax.Array:
                embeddings, _ = encode_resnet18(
                    params,
                    images,
                    train=False,
                    embedding_dim=self.embedding_dim,
                )
                return embeddings

            self._encode_tactile = jax.jit(encode_external)

    def _validate_contract(self, policy: Any, *, source_sample_steps: int) -> None:
        decoder = self.model.config
        _require_equal(decoder.action_dim, int(policy.config.action_dim), "action_dim")
        if (
            getattr(self.config, "inactive_arm_xyz_threshold_m", None) is not None
            and int(policy.config.action_dim) != 20
        ):
            raise ValueError(
                "FRS inactive-arm XYZ protection requires a 20-dimensional action"
            )
        _require_equal(decoder.action_horizon, int(policy.config.chunk_size), "action_horizon")
        _require_equal(
            decoder.num_tactile_tokens,
            len(self.config.tactile_keys),
            "num_tactile_tokens",
        )
        _require_equal(decoder.resnet_embedding_dim, self.embedding_dim, "resnet_embedding_dim")
        if bool(getattr(decoder, "state_conditioning", False)):
            _require_equal(decoder.state_dim, int(policy.config.state_dim), "state_dim")
        extra = self.metadata.get("extra_metadata")
        if not isinstance(extra, Mapping):
            raise ValueError("FRS checkpoint is missing extra_metadata")
        if extra.get("loss_mode") == "bimanual_gated":
            validate_bimanual_objective_metadata(extra)
        else:
            _require_equal(extra.get("loss_mode"), "gated", "loss_mode")
        decoder_input_version = extra.get("decoder_input_version")
        if (
            not isinstance(decoder_input_version, int)
            or isinstance(decoder_input_version, bool)
            or decoder_input_version != 2
        ):
            raise ValueError(
                "FRS checkpoint mismatch for decoder_input_version: "
                f"{decoder_input_version!r} != 2"
            )
        _require_equal(
            int(extra.get("tactile_window", 0)),
            decoder.tactile_window,
            "tactile_window",
        )
        _require_equal(
            bool(extra.get("state_conditioning", False)),
            bool(getattr(decoder, "state_conditioning", False)),
            "state_conditioning",
        )
        # ``validation_steps`` records the training-time evaluation setup.
        # Runtime decoding may intentionally use a different step count.
        # ``validation_solver`` records how the training run selected/evaluated
        # this checkpoint. It is provenance, not a runtime decoder constraint.

        source_configs = _source_cache_configurations(extra)
        deployed_fingerprint = None
        if self.config.verify_source_checkpoint_fingerprint:
            deployed_fingerprint = _checkpoint_fingerprint(Path(policy.checkpoint))
        for index, source in enumerate(source_configs):
            prefix = f"cache source {index}"
            _require_equal(
                int(source.get("model_sample_steps", 0)),
                source_sample_steps,
                f"{prefix} sample_steps",
            )
            _require_equal(
                int(source.get("reverse_steps", 0)),
                self.config.reverse_steps,
                f"{prefix} reverse_steps",
            )
            _require_equal(
                source.get("reverse_solver"),
                self.config.reverse_solver,
                f"{prefix} reverse_solver",
            )
            _require_equal(
                source.get("normalization_source"),
                "checkpoint",
                f"{prefix} normalization_source",
            )
            _require_equal(
                int(source.get("reverse_integration_version", 0)),
                REVERSE_INTEGRATION_VERSION,
                f"{prefix} reverse integration version",
            )
            if deployed_fingerprint is not None:
                _require_equal(
                    source.get("checkpoint_fingerprint"),
                    deployed_fingerprint,
                    f"{prefix} source checkpoint fingerprint",
                )

    @property
    def tactile_keys(self) -> tuple[str, ...]:
        return self.config.tactile_keys

    def _encode_observation(self, observation: Mapping[str, Any]) -> np.ndarray:
        missing = [key for key in self.config.tactile_keys if key not in observation]
        if missing:
            raise ValueError(f"robot observation is missing FRS tactile keys: {missing}")
        images = np.concatenate(
            [prepare_tactile_batch(observation[key], self.image_size) for key in self.config.tactile_keys],
            axis=0,
        )
        image_array = jnp.asarray(images, dtype=jnp.float32)
        if self.model.config.tactile_encoder_trainable:
            embeddings = self._encode_tactile(self.model, image_array)
        else:
            assert self.encoder is not None
            embeddings = self._encode_tactile(
                self.encoder.params["tactile_resnet"],
                image_array,
            )
        return np.asarray(jax.device_get(embeddings), dtype=np.float32)

    def _uses_state(self) -> bool:
        return bool(
            getattr(getattr(self.model, "config", None), "state_conditioning", False)
        )

    def resolved_tactile_window(self) -> int:
        return resolve_tactile_window(
            action_horizon=int(self.policy.config.chunk_size),
            window_divisor=int(self.config.tactile_window_divisor),
        )

    def _normalized_state(self, observation: Mapping[str, Any]) -> jax.Array | None:
        if not self._uses_state():
            return None
        if "observation.state" not in observation:
            raise ValueError("robot observation is missing FRS observation.state")
        state = jnp.asarray(observation["observation.state"], dtype=jnp.float32)
        if state.ndim == 1:
            state = state[None, :]
        expected = (1, int(self.model.config.state_dim))
        if state.shape != expected:
            raise ValueError(f"FRS state must have shape {expected}, got {state.shape}")
        state = self.policy.preprocessor.normalize_state(state)
        if not bool(jnp.all(jnp.isfinite(state))):
            raise ValueError("normalized FRS state contains NaN or Inf")
        return state

    @staticmethod
    def _readonly_array(value: Any) -> np.ndarray:
        array = np.array(jax.device_get(value), dtype=np.float32, copy=True)
        array.setflags(write=False)
        return array

    @staticmethod
    def _immutable_public_array(value: Any) -> np.ndarray:
        array = np.asarray(jax.device_get(value), dtype=np.float32)
        return np.frombuffer(array.tobytes(order="C"), dtype=np.float32).reshape(array.shape)

    def _readonly_action_chunk(self, value: Any, *, name: str) -> np.ndarray:
        array = self._readonly_array(value)
        expected = (
            1,
            int(self.policy.config.chunk_size),
            int(self.policy.config.action_dim),
        )
        if array.shape != expected:
            raise ValueError(f"{name} must have shape {expected}, got {array.shape}")
        if not np.isfinite(array).all():
            raise ValueError(f"{name} must be finite")
        return array

    def _clear_chunk_state(self) -> None:
        self._active_chunk_id: int | None = None
        self._action_vla_normalized: np.ndarray | None = None
        self._action_vla: np.ndarray | None = None
        self._x_base: np.ndarray | None = None
        self._x_base_device: jax.Array | None = None
        self._tactile_sequence: list[np.ndarray] = []
        self._request_results: dict[int, tuple[int, int, bytes, FRSSteerResult]] = {}
        self._last_action_index: int | None = None
        self.last_diagnostics = None
        self.last_vla_normalized = None
        self.last_frs_normalized = None

    def reset_episode(self, initial_observation: Mapping[str, Any]) -> None:
        internal_baseline = self._readonly_array(self._encode_observation(initial_observation))
        prepared_history = TactileHistory(
            window=self.history.window,
            stride=self.history.stride,
            token_shape=self.history.token_shape,
        )
        prepared_history.reset(internal_baseline)
        public_baseline = self._readonly_array(internal_baseline)

        self._episode_baseline = internal_baseline
        self.baseline = public_baseline
        self.history = prepared_history
        self._clear_chunk_state()

    def reset(self, observation: Mapping[str, Any]) -> None:
        """Compatibility wrapper for the legacy one-shot steering path."""

        self.reset_episode(observation)

    def _require_episode_and_no_active_chunk(self) -> None:
        if self._episode_baseline is None:
            raise RuntimeError("FRS reset_episode() must be called before begin_chunk()")
        if self._active_chunk_id is not None:
            raise RuntimeError(f"FRS active chunk {self._active_chunk_id} must be ended first")

    def _activate_chunk(self, chunk_id: int, normalized: Any, x_base: Any) -> None:
        normalized_array = self._readonly_action_chunk(
            normalized,
            name="normalized VLA actions",
        )
        x_base_array = self._readonly_action_chunk(
            x_base,
            name="reverse-flow base",
        )
        action_array = self._readonly_action_chunk(
            self.policy.preprocessor.unnormalize_actions(normalized_array),
            name="robot-space VLA actions",
        )

        self._clear_chunk_state()
        self._active_chunk_id = chunk_id
        self._action_vla_normalized = normalized_array
        self._action_vla = action_array
        self._x_base = x_base_array
        # Keep reverse-flow base on device so each steer avoids a host→device copy.
        self._x_base_device = jnp.asarray(x_base_array, dtype=jnp.float32)

    def _make_chunk_ready(self, started: float, finished: float) -> FRSChunkReady:
        assert self._active_chunk_id is not None
        assert self._action_vla_normalized is not None
        assert self._action_vla is not None
        assert self._x_base is not None
        return FRSChunkReady(
            chunk_id=self._active_chunk_id,
            action_vla_normalized=self._readonly_array(self._action_vla_normalized),
            action_vla=self._readonly_array(self._action_vla),
            x_base=self._readonly_array(self._x_base),
            prediction_started_at=started,
            prediction_finished_at=finished,
        )

    def begin_chunk(
        self,
        chunk_id: int,
        initial_observation: Mapping[str, Any],
        task: str,
        *,
        seed: int,
        jit: bool,
        num_steps: int | None,
    ) -> FRSChunkReady:
        self._require_episode_and_no_active_chunk()
        started = time.time()
        normalized = self.policy.predict_action_chunk(
            initial_observation,
            task,
            seed=seed,
            jit=jit,
            num_steps=num_steps,
            normalized=True,
        )
        # print("[frs] Forward prediction finished.")

        x_base = self.policy.reverse_action_chunk(
            initial_observation,
            task,
            normalized,
            num_steps=self.config.reverse_steps,
            solver=self.config.reverse_solver,
        )
        # print("[frs] Reversal integration finished.")

        self._activate_chunk(chunk_id, normalized, x_base)
        return self._make_chunk_ready(started, time.time())

    def _require_active_chunk(self, chunk_id: int) -> None:
        if self._active_chunk_id is None:
            raise RuntimeError("FRS has no active chunk")
        if chunk_id != self._active_chunk_id:
            raise ValueError(
                f"FRS chunk id {chunk_id} does not match active chunk {self._active_chunk_id}"
            )

    def end_chunk(self, chunk_id: int) -> None:
        self._require_active_chunk(chunk_id)
        self._clear_chunk_state()

    def _tactile_payload_hash(self, observation: Mapping[str, Any]) -> bytes:
        digest = hashlib.sha256()
        for key in self.config.tactile_keys:
            if key not in observation:
                raise ValueError(f"robot observation is missing FRS tactile key: {key}")
            array = np.asarray(observation[key])
            if array.dtype.hasobject or array.dtype.fields is not None:
                raise ValueError(
                    f"FRS tactile key {key!r} must use a numeric non-structured dtype, "
                    f"got {array.dtype}"
                )
            contiguous = np.ascontiguousarray(array)
            for value in (key.encode(), contiguous.dtype.str.encode()):
                digest.update(len(value).to_bytes(8, "big"))
                digest.update(value)
            digest.update(len(contiguous.shape).to_bytes(8, "big"))
            for dimension in contiguous.shape:
                digest.update(int(dimension).to_bytes(8, "big", signed=True))
            payload = contiguous.tobytes(order="C")
            digest.update(len(payload).to_bytes(8, "big"))
            digest.update(payload)
        if self._uses_state():
            state = np.ascontiguousarray(np.asarray(observation["observation.state"]))
            payload = state.tobytes(order="C")
            digest.update(b"observation.state")
            digest.update(state.dtype.str.encode())
            digest.update(payload)
        return digest.digest()

    def _validated_decoded_chunk(self, decoded: Any) -> tuple[np.ndarray, float, float]:
        assert self._action_vla_normalized is not None
        # One host sync for the full chunk; callers reuse this array instead of
        # issuing another device_get for selected rows / public copies.
        decoded_array = np.asarray(jax.device_get(decoded), dtype=np.float32)
        expected = (
            1,
            int(self.policy.config.chunk_size),
            int(self.policy.config.action_dim),
        )
        if decoded_array.shape != expected:
            raise ValueError(f"FRS output must have shape {expected}, got {decoded_array.shape}")
        if expected[-1] == 20:
            decoded_array = np.array(decoded_array, copy=True)
            decoded_array[..., [9, 19]] = self._action_vla_normalized[..., [9, 19]]
        if not np.isfinite(decoded_array).all():
            raise ValueError("FRS action contains NaN or Inf")
        delta_rms = float(
            np.sqrt(np.mean(np.square(decoded_array - self._action_vla_normalized)))
        )
        max_abs = float(np.max(np.abs(decoded_array)))
        if max_abs > self.config.max_normalized_action_abs:
            raise ValueError(
                f"FRS normalized action safety limit exceeded: {max_abs:.4f} > "
                f"{self.config.max_normalized_action_abs:.4f}"
            )
        if delta_rms > self.config.max_normalized_delta_rms:
            raise ValueError(
                f"FRS normalized delta safety limit exceeded: {delta_rms:.4f} > "
                f"{self.config.max_normalized_delta_rms:.4f}"
            )
        return decoded_array, delta_rms, max_abs

    def steer_action(
        self,
        chunk_id: int,
        request_id: int,
        observation: Mapping[str, Any],
        action_index: int,
    ) -> FRSSteerResult:
        cached = self._request_results.get(request_id)
        if cached is not None:
            tactile_hash = self._tactile_payload_hash(observation)
            cached_chunk_id, cached_action_index, cached_hash, result = cached
            if (chunk_id, action_index, tactile_hash) != (
                cached_chunk_id,
                cached_action_index,
                cached_hash,
            ):
                raise ValueError(f"conflicting duplicate FRS request id {request_id}")
            return result

        self._require_active_chunk(chunk_id)
        horizon = int(self.policy.config.chunk_size)
        if not isinstance(action_index, int) or isinstance(action_index, bool):
            raise ValueError("FRS action_index must be an integer")
        if not 0 <= action_index < horizon:
            raise ValueError(
                f"FRS action_index is outside action horizon [0, {horizon}): {action_index}"
            )
        if self._last_action_index is not None and action_index <= self._last_action_index:
            raise ValueError(
                "FRS action_index must be strictly increasing for unique requests: "
                f"{action_index} <= {self._last_action_index}"
            )
        if len(self._request_results) >= horizon:
            raise ValueError(f"FRS tactile sequence exceeds action horizon {horizon}")
        tactile_hash = self._tactile_payload_hash(observation)
        assert self._episode_baseline is not None
        assert self._action_vla_normalized is not None
        assert self._action_vla is not None
        assert self._x_base is not None
        assert self._x_base_device is not None

        encode_started_at = time.time()
        current = self._readonly_array(self._encode_observation(observation))
        encode_finished_at = time.time()
        if current.shape != self._episode_baseline.shape:
            raise ValueError(
                f"expected tactile tokens {self._episode_baseline.shape}, got {current.shape}"
            )
        if not np.isfinite(current).all():
            raise ValueError("tactile encoder produced NaN or Inf")
        tactile_sequence = self.history.window_tokens_after(current)
        tactile = jnp.expand_dims(jnp.asarray(tactile_sequence), axis=0)
        change = tactile_change_from_tokens(
            current[None, ...],
            self._episode_baseline[None, ...],
        )
        decode_started_at = time.time()
        decode_kwargs: dict[str, Any] = {}
        normalized_state = self._normalized_state(observation)
        if normalized_state is not None:
            decode_kwargs["state"] = normalized_state
        decoded = decode_actions(
            self.model,
            self._x_base_device,
            tactile,
            num_steps=self.config.decode_steps,
            solver=self.config.decode_solver,
            **decode_kwargs,
        )
        decoded_array, delta_rms, max_abs = self._validated_decoded_chunk(decoded)
        decode_finished_at = time.time()
        diagnostics = FRSDiagnostics(
            tactile_change=float(change[0]),
            delta_rms=delta_rms,
            max_normalized_action_abs=max_abs,
        )
        # Hot path: only the selected action leaves as a fresh public buffer.
        # Full-chunk fields are isolated once from the single host sync above.
        selected = decoded_array[0, action_index]
        ensemble_coeff = getattr(self.config, "temporal_ensemble_coeff", None)
        if ensemble_coeff is not None:
            candidates = [
                cached_result.decoded_normalized[0, action_index]
                for *_, cached_result in self._request_results.values()
            ]
            candidates.append(selected)
            ages = np.arange(len(candidates) - 1, -1, -1)
            weights = np.power(math.exp(-float(ensemble_coeff)), ages)
            selected = np.average(np.stack(candidates), axis=0, weights=weights)
        inactive_threshold = getattr(
            self.config,
            "inactive_arm_xyz_threshold_m",
            None,
        )
        if inactive_threshold is not None:
            selected = _protect_inactive_arm_xyz(
                selected,
                self._action_vla_normalized[0, action_index],
                self._action_vla[0, action_index],
                inactive_threshold,
            )
        if selected.shape == (20,):
            selected = np.array(selected, copy=True)
            selected[[9, 19]] = self._action_vla_normalized[
                0,
                action_index,
                [9, 19],
            ]
        selected_normalized = self._immutable_public_array(selected)
        robot_selected = np.asarray(
            self.policy.preprocessor.unnormalize_actions(selected_normalized),
            dtype=np.float32,
        )
        expected_selected_shape = (int(self.policy.config.action_dim),)
        if robot_selected.shape != expected_selected_shape:
            raise ValueError(
                f"robot-space selected action must have shape {expected_selected_shape}, "
                f"got {robot_selected.shape}"
            )
        if not np.isfinite(robot_selected).all():
            raise ValueError("robot-space selected action must be finite")
        if robot_selected.shape == (20,):
            robot_selected = robot_selected.copy()
            robot_selected[[9, 19]] = self._action_vla[0, action_index, [9, 19]]
        selected_action = self._immutable_public_array(robot_selected)

        result = FRSSteerResult(
            chunk_id=chunk_id,
            request_id=request_id,
            action_index=action_index,
            action_vla_normalized=self._immutable_public_array(self._action_vla_normalized),
            x_base=self._immutable_public_array(self._x_base),
            decoded_normalized=self._immutable_public_array(decoded_array),
            selected_normalized=selected_normalized,
            selected_action=selected_action,
            tactile_sequence_length=len(tactile_sequence),
            diagnostics=diagnostics,
            encode_started_at=encode_started_at,
            encode_finished_at=encode_finished_at,
            decode_started_at=decode_started_at,
            decode_finished_at=decode_finished_at,
        )
        next_last_vla_normalized = self._readonly_array(self._action_vla_normalized)
        next_last_frs_normalized = self._readonly_array(decoded_array)

        # Commit only after every validation, conversion and allocation that can
        # fail, so an unsuccessful request never advances episode history.
        self.history.append(current)
        self._request_results[request_id] = (chunk_id, action_index, tactile_hash, result)
        self._last_action_index = action_index
        self.last_diagnostics = diagnostics
        self.last_vla_normalized = next_last_vla_normalized
        self.last_frs_normalized = next_last_frs_normalized
        return result

    def _snapshot_live_state(self) -> tuple[Any, ...]:
        return (
            self._episode_baseline,
            self.baseline,
            self.history,
            self._active_chunk_id,
            self._action_vla_normalized,
            self._action_vla,
            self._x_base,
            self._x_base_device,
            self._tactile_sequence,
            self._request_results,
            self._last_action_index,
            self.last_diagnostics,
            self.last_vla_normalized,
            self.last_frs_normalized,
        )

    def _restore_live_state(self, snapshot: tuple[Any, ...]) -> None:
        (
            self._episode_baseline,
            self.baseline,
            self.history,
            self._active_chunk_id,
            self._action_vla_normalized,
            self._action_vla,
            self._x_base,
            self._x_base_device,
            self._tactile_sequence,
            self._request_results,
            self._last_action_index,
            self.last_diagnostics,
            self.last_vla_normalized,
            self.last_frs_normalized,
        ) = snapshot

    def warmup_all_tactile_lengths(self) -> None:
        if self._episode_baseline is None:
            raise RuntimeError("FRS reset_episode() must be called before warmup")
        snapshot = self._snapshot_live_state()
        try:
            horizon = int(self.policy.config.chunk_size)
            checkpoint_window = int(self.model.config.tactile_window)
            window = self.resolved_tactile_window()
            if window != checkpoint_window:
                logging.getLogger(__name__).warning(
                    "FRS runtime tactile window %d (horizon %d / divisor %d) "
                    "differs from checkpoint tactile window %d; "
                    "warming %d concrete lengths",
                    window,
                    horizon,
                    int(self.config.tactile_window_divisor),
                    checkpoint_window,
                    window,
                )
            baseline = np.asarray(self._episode_baseline, dtype=np.float32)
            if self._x_base_device is not None:
                synthetic_x_base = self._x_base_device
            elif self._x_base is None:
                synthetic_x_base = jnp.zeros(
                    (
                        1,
                        horizon,
                        int(self.policy.config.action_dim),
                    ),
                    dtype=jnp.float32,
                )
            else:
                synthetic_x_base = jnp.asarray(self._x_base, dtype=jnp.float32)
            for length in range(1, window + 1):
                tactile = jnp.expand_dims(
                    jnp.asarray(np.stack([baseline] * length)),
                    axis=0,
                )
                warmup_kwargs: dict[str, Any] = {}
                if self._uses_state():
                    warmup_kwargs["state"] = jnp.zeros(
                        (1, int(self.model.config.state_dim)), dtype=jnp.float32
                    )
                warmed = decode_actions(
                    self.model,
                    synthetic_x_base,
                    tactile,
                    num_steps=self.config.decode_steps,
                    solver=self.config.decode_solver,
                    **warmup_kwargs,
                )
                jax.block_until_ready(warmed)
        finally:
            self._restore_live_state(snapshot)

    @staticmethod
    def _eval_observation(policy: Any, observation: Mapping[str, Any], task: str) -> EvalObservation:
        batch = policy.preprocessor.prepare(observation, task)
        return EvalObservation(
            images=batch["images"],
            image_masks=batch["image_masks"],
            language_tokens=batch["language_tokens"],
            language_masks=batch["language_masks"],
            state=batch["state"],
            image_keys=tuple(policy.config.image_keys),
        )

    # steering entrypoint
    def steer(
        self,
        policy: Any,
        observation: Mapping[str, Any],
        task: str,
        vla_actions: jax.Array,
        *,
        update_history: bool = True,
    ) -> jax.Array:
        if self.baseline is None:
            raise RuntimeError("FRS reset() must be called with the episode baseline first")
        current = self._encode_observation(observation)
        if update_history:
            self.history.append(current)

        tactile_seq = self.history.window_tokens()[None, ...]
        change = tactile_change_from_tokens(current[None, ...], self._episode_baseline[None, ...])

        eval_observation = self._eval_observation(policy, observation, task)
        x_base = reverse_integrate_actions(
            policy,
            eval_observation,
            vla_actions,
            num_steps=self.config.reverse_steps,
            solver=self.config.reverse_solver,
        )

        legacy_decode_kwargs: dict[str, Any] = {}
        if self._uses_state():
            legacy_decode_kwargs["state"] = eval_observation.state
        refined = decode_actions(
            self.model,
            x_base,
            jnp.asarray(tactile_seq, dtype=jnp.float32),
            num_steps=self.config.decode_steps,
            solver=self.config.decode_solver,
            **legacy_decode_kwargs,
        )
        refined_np = np.asarray(jax.device_get(refined), dtype=np.float32)
        vla_np = np.asarray(jax.device_get(vla_actions), dtype=np.float32)

        if refined_np.shape != vla_np.shape:
            raise ValueError(f"FRS output shape {refined_np.shape} != VLA shape {vla_np.shape}")
        if not np.isfinite(refined_np).all():
            raise ValueError("FRS action contains NaN or Inf")
        delta_rms = float(np.sqrt(np.mean(np.square(refined_np - vla_np))))
        max_abs = float(np.max(np.abs(refined_np)))
        if max_abs > self.config.max_normalized_action_abs:
            raise ValueError(
                f"FRS normalized action safety limit exceeded: {max_abs:.4f} > "
                f"{self.config.max_normalized_action_abs:.4f}"
            )
        if delta_rms > self.config.max_normalized_delta_rms:
            raise ValueError(
                f"FRS normalized delta safety limit exceeded: {delta_rms:.4f} > "
                f"{self.config.max_normalized_delta_rms:.4f}"
            )

        self.last_diagnostics = FRSDiagnostics(
            tactile_change=float(change[0]),
            delta_rms=delta_rms,
            max_normalized_action_abs=max_abs,
        )
        self.last_vla_normalized = np.array(vla_np, copy=True)
        self.last_frs_normalized = np.array(refined_np, copy=True)
        return jnp.asarray(refined_np)


FRSRuntime = FRSSteeringPolicy
