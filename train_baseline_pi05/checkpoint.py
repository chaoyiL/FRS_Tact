"""Strict, atomic checkpoints for the direct tactile action decoder."""

from __future__ import annotations

import os
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from train_baseline_pi05.model import DirectDecoderConfig, DirectTactileActionDecoder


_SCHEMA = "direct_tactile_action_decoder"
_VERSION = 1
_REQUIRED_KEYS = {
    "schema",
    "version",
    "run_kind",
    "mode",
    "epoch",
    "global_step",
    "decoder_config",
    "decoder_state",
    "metrics",
    "source_contract",
}
_OPTIONAL_RESUME_KEYS = {"optimizer_state", "scheduler_state", "rng_state", "best_state"}


def _checkpoint_path(directory_or_path: Path, filename: str) -> Path:
    path = Path(directory_or_path)
    return path if path.suffix == ".pt" else path / filename


def _integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer.")
    return value


def _scalar_mapping(value: object, name: str) -> dict[str, int | float]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping.")
    result: dict[str, int | float] = {}
    for key, item in value.items():
        if not isinstance(key, str) or isinstance(item, bool) or not isinstance(item, (int, float)):
            raise ValueError(f"{name} must contain string keys and numeric scalar values.")
        result[key] = item
    return result


def _primitive(value: object) -> bool:
    if value is None or isinstance(value, (str, int, float, bool)):
        return True
    if isinstance(value, (list, tuple)):
        return all(_primitive(item) for item in value)
    if isinstance(value, Mapping):
        return all(isinstance(key, str) and _primitive(item) for key, item in value.items())
    return False


def _weights_only_payload(value: object, name: str) -> Any:
    """Copy a value into built-in containers accepted by ``torch.load(weights_only=True)``."""
    if value is None or isinstance(value, (str, int, float, bool, Tensor)):
        return value
    if isinstance(value, list):
        return [_weights_only_payload(item, name) for item in value]
    if isinstance(value, tuple):
        return tuple(_weights_only_payload(item, name) for item in value)
    if isinstance(value, Mapping):
        result: dict[str | int, Any] = {}
        for key, item in value.items():
            if isinstance(key, bool) or not isinstance(key, (str, int)):
                raise ValueError(f"{name} has a non-primitive mapping key.")
            result[key] = _weights_only_payload(item, name)
        return result
    raise ValueError(f"{name} contains a value incompatible with weights_only loading.")


def _source_contract(value: object) -> dict[str, Any]:
    if not isinstance(value, Mapping) or not _primitive(value):
        raise ValueError("source_contract must be a primitive mapping.")
    return dict(value)


def _cpu_state_dict(model: DirectTactileActionDecoder) -> dict[str, Tensor]:
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def _atomic_save(payload: dict[str, Any], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False
        ) as temporary:
            temporary_name = temporary.name
            torch.save(payload, temporary)
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_name, path)
    except Exception:
        if temporary_name is not None:
            Path(temporary_name).unlink(missing_ok=True)
        raise
    return path


def _payload(
    model: DirectTactileActionDecoder,
    config: DirectDecoderConfig,
    *,
    epoch: int,
    global_step: int,
    metrics: Mapping[str, int | float],
    source_contract: Mapping[str, object],
) -> dict[str, Any]:
    config.validate()
    if model.config != config:
        raise ValueError("model configuration does not match checkpoint configuration.")
    return {
        "schema": _SCHEMA,
        "version": _VERSION,
        "run_kind": "formal",
        "mode": "action_tactile",
        "epoch": _integer(epoch, "epoch"),
        "global_step": _integer(global_step, "global_step"),
        "decoder_config": config.to_primitive(),
        "decoder_state": _cpu_state_dict(model),
        "metrics": _scalar_mapping(metrics, "metrics"),
        "source_contract": _source_contract(source_contract),
    }


def save_best_checkpoint(
    directory_or_path: Path,
    model: DirectTactileActionDecoder,
    config: DirectDecoderConfig,
    *,
    epoch: int,
    global_step: int,
    metrics: Mapping[str, int | float],
    source_contract: Mapping[str, object],
) -> Path:
    """Atomically write the portable, best-model checkpoint as ``best.pt``."""
    return _atomic_save(
        _payload(
            model, config, epoch=epoch, global_step=global_step, metrics=metrics,
            source_contract=source_contract,
        ),
        _checkpoint_path(directory_or_path, "best.pt"),
    )


def save_last_checkpoint(
    directory_or_path: Path,
    model: DirectTactileActionDecoder,
    config: DirectDecoderConfig,
    *,
    epoch: int,
    global_step: int,
    metrics: Mapping[str, int | float],
    source_contract: Mapping[str, object],
    optimizer: torch.optim.Optimizer | None = None,
    scheduler_state: Mapping[str, object] | None = None,
    rng_state: Mapping[str, object] | None = None,
    best_state: Mapping[str, object] | None = None,
) -> Path:
    """Atomically write ``last.pt``, including optional state needed for resume."""
    payload = _payload(
        model, config, epoch=epoch, global_step=global_step, metrics=metrics,
        source_contract=source_contract,
    )
    if optimizer is not None:
        payload["optimizer_state"] = _weights_only_payload(optimizer.state_dict(), "optimizer_state")
    if scheduler_state is not None:
        payload["scheduler_state"] = _weights_only_payload(scheduler_state, "scheduler_state")
    if rng_state is not None:
        payload["rng_state"] = _weights_only_payload(rng_state, "rng_state")
    if best_state is not None:
        payload["best_state"] = _weights_only_payload(best_state, "best_state")
    return _atomic_save(payload, _checkpoint_path(directory_or_path, "last.pt"))


def _validate_payload(raw: object) -> dict[str, Any]:
    if (
        not isinstance(raw, dict)
        or not _REQUIRED_KEYS.issubset(raw)
        or not set(raw).issubset(_REQUIRED_KEYS | _OPTIONAL_RESUME_KEYS)
    ):
        raise ValueError("checkpoint has an invalid schema.")
    if raw["schema"] != _SCHEMA or raw["version"] != _VERSION:
        raise ValueError("checkpoint schema/version is unsupported.")
    if raw["run_kind"] != "formal" or raw["mode"] != "action_tactile":
        raise ValueError("checkpoint run contract is invalid.")
    _integer(raw["epoch"], "epoch")
    _integer(raw["global_step"], "global_step")
    DirectDecoderConfig.from_primitive(raw["decoder_config"])
    _scalar_mapping(raw["metrics"], "metrics")
    _source_contract(raw["source_contract"])
    state = raw["decoder_state"]
    if not isinstance(state, Mapping) or not all(
        isinstance(key, str) and isinstance(value, Tensor) for key, value in state.items()
    ):
        raise ValueError("decoder_state must be a tensor state dictionary.")
    for name in _OPTIONAL_RESUME_KEYS & set(raw):
        _weights_only_payload(raw[name], name)
    return raw


def load_decoder_checkpoint(
    path: Path, *, map_location: str | torch.device = "cpu"
) -> tuple[DirectTactileActionDecoder, dict[str, Any]]:
    """Load a portable decoder checkpoint with safe deserialization and strict state."""
    try:
        raw = torch.load(Path(path), map_location=map_location, weights_only=True)
    except Exception as exc:
        raise ValueError("checkpoint could not be safely loaded.") from exc
    payload = _validate_payload(raw)
    config = DirectDecoderConfig.from_primitive(payload["decoder_config"])
    decoder = DirectTactileActionDecoder(config)
    try:
        decoder.load_state_dict(payload["decoder_state"], strict=True)
    except (RuntimeError, TypeError, AttributeError) as exc:
        raise ValueError("checkpoint decoder state is incompatible with the fixed contract.") from exc
    return decoder, payload
