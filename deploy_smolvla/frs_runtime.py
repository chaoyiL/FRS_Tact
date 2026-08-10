"""Stateful tactile Flow Re-Steering runtime for remote SmolVLA deployment."""

from __future__ import annotations

import hashlib
import math
from collections import deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from modalities_eval.utils import EvalObservation
from tactile_encoder.utils.checkpoint import load_tactile_encoder
from tactile_encoder.utils.model import tactile_clip_config_from_dict
from tactile_encoder.utils.resnet import encode_resnet18
from train_frs.utils.checkpoint import load_checkpoint as load_frs_checkpoint
from train_frs.utils.data import gate_weights_from_change, tactile_change_from_tokens
from train_frs.utils.model import decode_actions
from train_vtsmolvla.preprocessing import prepare_tactile_batch
from utils.integration import REVERSE_INTEGRATION_VERSION
from utils.source_model import reverse_integrate_actions


@dataclass(frozen=True)
class FRSDiagnostics:
    tactile_change: float
    gate_weight: float
    delta_rms: float
    max_normalized_action_abs: float


@dataclass(frozen=True)
class FRSConfig:
    checkpoint: Path
    tactile_encoder_checkpoint: Path
    tactile_keys: tuple[str, ...]
    history_stride: int
    gate_tau: float
    gate_temperature: float
    reverse_steps: int
    reverse_solver: str
    decode_steps: int
    decode_solver: str
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


def parse_frs_config(raw: Mapping[str, Any], *, config_path: Path) -> FRSConfig:
    required = (
        "checkpoint",
        "tactile_encoder_checkpoint",
        "tactile_keys",
        "history_stride",
        "gate_tau",
        "gate_temperature",
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

    history_stride = int(raw["history_stride"])
    reverse_steps = int(raw["reverse_steps"])
    decode_steps = int(raw["decode_steps"])
    gate_temperature = float(raw["gate_temperature"])
    max_action_abs = float(raw.get("max_normalized_action_abs", 8.0))
    max_delta_rms = float(raw.get("max_normalized_delta_rms", 4.0))
    if min(history_stride, reverse_steps, decode_steps) <= 0:
        raise ValueError("frs history_stride/reverse_steps/decode_steps must be positive")
    if gate_temperature <= 0:
        raise ValueError("frs.gate_temperature must be positive")
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
        history_stride=history_stride,
        gate_tau=float(raw["gate_tau"]),
        gate_temperature=gate_temperature,
        reverse_steps=reverse_steps,
        reverse_solver=reverse_solver,
        decode_steps=decode_steps,
        decode_solver=decode_solver,
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
    if raw.get("enabled", True) is False:
        return
    for key in (
        "checkpoint",
        "tactile_encoder_checkpoint",
        "tactile_keys",
        "history_stride",
        "gate_tau",
        "gate_temperature",
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
            int(raw["history_stride"]),
            int(raw["reverse_steps"]),
            int(raw["decode_steps"]),
        )
        <= 0
    ):
        raise ValueError("frs history_stride/reverse_steps/decode_steps must be positive")
    if float(raw["gate_temperature"]) <= 0:
        raise ValueError("frs.gate_temperature must be positive")
    if str(raw["reverse_solver"]) not in {"euler", "fireflow", "slerpflow"}:
        raise ValueError("frs.reverse_solver must be euler, fireflow, or slerpflow")
    if str(raw["decode_solver"]) not in {"euler", "fireflow"}:
        raise ValueError("frs.decode_solver must be euler or fireflow")
    if config.get("observation", {}).get("data_type") != "vitac":
        raise ValueError("FRS deployment requires observation.data_type='vitac'")


def _checkpoint_fingerprint(checkpoint_dir: Path) -> str:
    """Match the source-checkpoint identity stored by ``train_frs.prepare``."""

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

    def reset(self, tokens: np.ndarray) -> None:
        self._frames.clear()
        self.append(tokens)

    def append(self, tokens: np.ndarray) -> None:
        tokens = np.asarray(tokens, dtype=np.float32)
        if tokens.shape != self.token_shape:
            raise ValueError(f"expected tactile tokens {self.token_shape}, got {tokens.shape}")
        if not np.isfinite(tokens).all():
            raise ValueError("tactile encoder produced NaN or Inf")
        self._frames.append(np.array(tokens, copy=True))

    def window_tokens(self) -> np.ndarray:
        if not self._frames:
            raise RuntimeError("FRS tactile history is not initialized")
        frames = tuple(self._frames)
        current = len(frames) - 1
        indices = [max(0, current - offset * self.stride) for offset in reversed(range(self.window))]
        return np.stack([frames[index] for index in indices], axis=0)


class FRSRuntime:
    """Load, validate and run FRS steering without silently falling back to VLA."""

    def __init__(
        self,
        raw_config: Mapping[str, Any],
        *,
        config_path: Path,
        policy: Any,
        source_sample_steps: int,
    ) -> None:
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
        self.encoder = load_tactile_encoder(self.config.tactile_encoder_checkpoint)
        encoder_config = tactile_clip_config_from_dict(self.encoder.metadata["tactile_clip_config"])
        self.embedding_dim = int(encoder_config.embedding_dim)
        self.image_size = int(encoder_config.tactile_image_size)
        if "tactile_resnet" not in self.encoder.params:
            raise KeyError("tactile encoder checkpoint is missing tactile_resnet params")
        self._validate_contract(policy, source_sample_steps=source_sample_steps)
        decoder = self.model.config
        self.history = TactileHistory(
            window=int(decoder.tactile_window),
            stride=self.config.history_stride,
            token_shape=(len(self.config.tactile_keys), self.embedding_dim),
        )
        self.baseline: np.ndarray | None = None
        self.last_diagnostics: FRSDiagnostics | None = None
        self.last_vla_normalized: np.ndarray | None = None
        self.last_frs_normalized: np.ndarray | None = None

        def encode(params: Any, images: jax.Array) -> jax.Array:
            embeddings, _ = encode_resnet18(
                params,
                images,
                train=False,
                embedding_dim=self.embedding_dim,
            )
            return embeddings

        self._encode_tactile = jax.jit(encode)

    def _validate_contract(self, policy: Any, *, source_sample_steps: int) -> None:
        decoder = self.model.config
        _require_equal(decoder.action_dim, int(policy.config.action_dim), "action_dim")
        _require_equal(decoder.action_horizon, int(policy.config.chunk_size), "action_horizon")
        _require_equal(
            decoder.num_tactile_tokens,
            len(self.config.tactile_keys),
            "num_tactile_tokens",
        )
        _require_equal(decoder.resnet_embedding_dim, self.embedding_dim, "resnet_embedding_dim")
        if not bool(decoder.gate_conditioning):
            raise ValueError("FRS checkpoint must have explicit gate conditioning enabled")

        extra = self.metadata.get("extra_metadata")
        if not isinstance(extra, Mapping):
            raise ValueError("FRS checkpoint is missing extra_metadata")
        _require_equal(extra.get("loss_mode"), "gated", "loss_mode")
        loss_weighting_version = int(extra.get("loss_weighting_version", 0))
        if loss_weighting_version not in {2, 3, 4}:
            raise ValueError(
                "unsupported FRS loss_weighting_version: " f"{loss_weighting_version}; expected one of 2, 3, 4"
            )
        if loss_weighting_version >= 4:
            low_threshold = float(extra.get("rank_low_gate_threshold", -1.0))
            high_threshold = float(extra.get("rank_high_gate_threshold", -1.0))
            if not 0.0 <= low_threshold < 0.5 < high_threshold <= 1.0:
                raise ValueError(
                    "invalid v4 three-region gate thresholds: " f"low={low_threshold}, high={high_threshold}"
                )
        _require_equal(
            int(extra.get("history_stride", 0)),
            self.config.history_stride,
            "history_stride",
        )
        _require_equal(
            int(extra.get("tactile_window", 0)),
            decoder.tactile_window,
            "tactile_window",
        )
        _require_equal(bool(extra.get("gate_conditioning", False)), True, "gate_conditioning")
        _require_equal(float(extra.get("gate_tau")), self.config.gate_tau, "gate_tau")
        _require_equal(
            float(extra.get("gate_temperature")),
            self.config.gate_temperature,
            "gate_temperature",
        )
        _require_equal(
            int(extra.get("validation_steps", 0)),
            self.config.decode_steps,
            "decode_steps",
        )
        _require_equal(extra.get("validation_solver"), self.config.decode_solver, "decode_solver")

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
        embeddings = self._encode_tactile(
            self.encoder.params["tactile_resnet"],
            jnp.asarray(images, dtype=jnp.float32),
        )
        return np.asarray(jax.device_get(embeddings), dtype=np.float32)

    def reset(self, observation: Mapping[str, Any]) -> None:
        baseline = self._encode_observation(observation)
        self.baseline = np.array(baseline, copy=True)
        self.history.reset(baseline)
        self.last_diagnostics = None
        self.last_vla_normalized = None
        self.last_frs_normalized = None

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
        change = tactile_change_from_tokens(current[None, ...], self.baseline[None, ...])
        gate = gate_weights_from_change(change, tau=self.config.gate_tau, temperature=self.config.gate_temperature)
        eval_observation = self._eval_observation(policy, observation, task)
        x_base = reverse_integrate_actions(
            policy,
            eval_observation,
            vla_actions,
            num_steps=self.config.reverse_steps,
            solver=self.config.reverse_solver,
        )
        refined = decode_actions(
            self.model,
            x_base,
            jnp.asarray(tactile_seq, dtype=jnp.float32),
            jnp.asarray(gate, dtype=jnp.float32),
            num_steps=self.config.decode_steps,
            solver=self.config.decode_solver,
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
            gate_weight=float(gate[0]),
            delta_rms=delta_rms,
            max_normalized_action_abs=max_abs,
        )
        self.last_vla_normalized = np.array(vla_np, copy=True)
        self.last_frs_normalized = np.array(refined_np, copy=True)
        return jnp.asarray(refined_np)
