"""Strict weights-only loader for formal direct tactile decoder checkpoints."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from .direct_decoder import DirectDecoderConfig, DirectTactileActionDecoder
from .deployment import DeploymentConfig, expected_source_contract


_REQUIRED = {"schema", "version", "run_kind", "mode", "epoch", "global_step", "decoder_config", "decoder_state", "metrics", "source_contract"}
_OPTIONAL = {"optimizer_state", "scheduler_state", "rng_state", "best_state"}


def _payload(raw: object) -> dict[str, Any]:
    if not isinstance(raw, dict) or not _REQUIRED.issubset(raw) or not set(raw).issubset(_REQUIRED | _OPTIONAL):
        raise ValueError("checkpoint has an invalid schema.")
    if raw["schema"] != "direct_tactile_action_decoder" or raw["version"] != 1:
        raise ValueError("checkpoint schema/version is unsupported.")
    if raw["run_kind"] != "formal" or raw["mode"] != "action_tactile":
        raise ValueError("checkpoint run contract is invalid.")
    config = DirectDecoderConfig.from_primitive(raw["decoder_config"])
    if not isinstance(raw["decoder_state"], Mapping) or not all(isinstance(key, str) and isinstance(value, Tensor) for key, value in raw["decoder_state"].items()):
        raise ValueError("decoder_state must be a tensor state dictionary.")
    if not isinstance(raw["source_contract"], Mapping):
        raise ValueError("source_contract must be a mapping.")
    raw["decoder_config"] = config.to_primitive()
    return raw


def _resolve_identity(value: object) -> object:
    if isinstance(value, str):
        return str(Path(value).expanduser().resolve())
    if isinstance(value, Mapping):
        return {key: _resolve_identity(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_resolve_identity(item) for item in value]
    return value


def _expected(value: Mapping[str, object] | DeploymentConfig) -> Mapping[str, object]:
    return expected_source_contract(value) if isinstance(value, DeploymentConfig) else value


def _validate_source(actual: object, expected_source: Mapping[str, object] | DeploymentConfig) -> None:
    if not isinstance(actual, Mapping):
        raise ValueError("source_contract must be a mapping.")
    expected = _resolve_identity(_expected(expected_source))
    resolved_actual = _resolve_identity(actual)
    if not isinstance(expected, Mapping) or not isinstance(resolved_actual, Mapping):
        raise ValueError("source_contract is invalid.")
    for group, fields in (("pi", ("checkpoint", "norm_stats_dir", "norm_stats_asset_id", "variant", "model_action_width", "sample_steps")), ("encoder", ("checkpoint", "key_order"))):
        actual_group = resolved_actual.get(group)
        expected_group = expected.get(group)
        if not isinstance(actual_group, Mapping) or not isinstance(expected_group, Mapping):
            raise ValueError(f"source_contract.{group} is missing.")
        for field in fields:
            if actual_group.get(field) != expected_group.get(field):
                raise ValueError(f"source_contract.{group}.{field} does not match deployment configuration.")


def load_decoder(path: Path, *, device: str | torch.device = "cpu", expected_source: Mapping[str, object] | DeploymentConfig) -> DirectTactileActionDecoder:
    """Load, strictly validate, freeze, and evaluate a portable best checkpoint."""
    try:
        raw = torch.load(Path(path), map_location=device, weights_only=True)
    except Exception as exc:
        raise ValueError("checkpoint could not be safely loaded.") from exc
    payload = _payload(raw)
    _validate_source(payload["source_contract"], expected_source)
    decoder = DirectTactileActionDecoder(DirectDecoderConfig.from_primitive(payload["decoder_config"]))
    try:
        decoder.load_state_dict(payload["decoder_state"], strict=True)
    except (RuntimeError, TypeError, AttributeError) as exc:
        raise ValueError("checkpoint decoder state is incompatible with the fixed contract.") from exc
    decoder.to(device).eval()
    for parameter in decoder.parameters():
        parameter.requires_grad_(False)
    return decoder
