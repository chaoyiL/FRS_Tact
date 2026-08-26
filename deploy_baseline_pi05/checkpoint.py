"""Strict weights-only loader for formal direct tactile decoder checkpoints."""

from __future__ import annotations

from collections.abc import Mapping
from math import isfinite
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from .direct_decoder import DirectDecoderConfig, DirectTactileActionDecoder
from .deployment import DeploymentConfig, expected_source_contract


_REQUIRED = {"schema", "version", "run_kind", "mode", "epoch", "global_step", "decoder_config", "decoder_state", "metrics", "source_contract"}


def _nonnegative_integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer.")
    return value


def _metrics(value: object) -> Mapping[str, int | float]:
    if not isinstance(value, Mapping):
        raise ValueError("metrics must be a mapping.")
    for key, item in value.items():
        if not isinstance(key, str) or isinstance(item, bool) or not isinstance(item, (int, float)) or not isfinite(item):
            raise ValueError("metrics must contain string keys and finite numeric scalar values.")
    return value


def _payload(raw: object) -> dict[str, Any]:
    if not isinstance(raw, dict) or set(raw) != _REQUIRED:
        raise ValueError("checkpoint has an invalid schema.")
    if raw["schema"] != "direct_tactile_action_decoder" or raw["version"] != 1:
        raise ValueError("checkpoint schema/version is unsupported.")
    if raw["run_kind"] != "formal" or raw["mode"] != "action_tactile":
        raise ValueError("checkpoint run contract is invalid.")
    _nonnegative_integer(raw["epoch"], "epoch")
    _nonnegative_integer(raw["global_step"], "global_step")
    _metrics(raw["metrics"])
    config = DirectDecoderConfig.from_primitive(raw["decoder_config"])
    if not isinstance(raw["decoder_state"], Mapping) or not all(isinstance(key, str) and isinstance(value, Tensor) for key, value in raw["decoder_state"].items()):
        raise ValueError("decoder_state must be a tensor state dictionary.")
    if not isinstance(raw["source_contract"], Mapping):
        raise ValueError("source_contract must be a mapping.")
    raw["decoder_config"] = config.to_primitive()
    return raw


def _expected(value: Mapping[str, object] | DeploymentConfig) -> Mapping[str, object]:
    return expected_source_contract(value) if isinstance(value, DeploymentConfig) else value


def _matches_identity(actual: object, expected: object, *, path: bool = False) -> bool:
    """Compare checkpoint identities, resolving only a checkpoint path from the payload."""
    if path:
        if not isinstance(actual, str) or not isinstance(expected, str):
            return False
        actual = str(Path(actual).expanduser().resolve())
    if isinstance(expected, Mapping):
        return isinstance(actual, Mapping) and all(
            key in actual and _matches_identity(actual[key], item)
            for key, item in expected.items()
        )
    if isinstance(expected, list):
        return (
            isinstance(actual, list)
            and len(actual) == len(expected)
            and all(_matches_identity(item, wanted) for item, wanted in zip(actual, expected, strict=True))
        )
    return type(actual) is type(expected) and actual == expected


def _validate_source(actual: object, expected_source: Mapping[str, object] | DeploymentConfig) -> None:
    if not isinstance(actual, Mapping):
        raise ValueError("source_contract must be a mapping.")
    expected = _expected(expected_source)
    if not isinstance(expected, Mapping):
        raise ValueError("source_contract is invalid.")
    for group in ("pi", "encoder"):
        actual_group = actual.get(group)
        expected_group = expected.get(group)
        if not isinstance(actual_group, Mapping) or not isinstance(expected_group, Mapping):
            raise ValueError(f"source_contract.{group} is missing.")
        for field, expected_value in expected_group.items():
            if field not in actual_group or not _matches_identity(
                actual_group[field], expected_value,
                path=(group, field) in {("pi", "checkpoint"), ("pi", "norm_stats_dir"), ("encoder", "checkpoint")},
            ):
                raise ValueError(f"source_contract.{group}.{field} does not match deployment configuration.")
    for group in ("action_cache", "tactile_cache"):
        if group not in expected:
            continue
        actual_group = actual.get(group)
        expected_group = expected[group]
        if not isinstance(actual_group, Mapping) or not isinstance(expected_group, Mapping):
            raise ValueError(f"source_contract.{group} is missing.")
        for field, expected_value in expected_group.items():
            if field not in actual_group or not _matches_identity(
                actual_group[field], expected_value, path=field == "path"
            ):
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
