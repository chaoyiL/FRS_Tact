"""Dependency-light validation for the FRS deployment configuration."""

from __future__ import annotations

from collections.abc import Mapping
from numbers import Integral
from typing import Any


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return value


def _boolean(value: Any, name: str) -> bool:
    if type(value) is not bool:
        raise ValueError(f"{name} must be a boolean")
    return value


def _integer(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be an integer")
    return int(value)


def validate_frs_config_section(config: Mapping[str, Any]) -> None:
    """Validate the FRS profile without importing JAX or training modules."""
    raw = _mapping(config.get("frs"), "frs")
    if not _boolean(raw.get("enabled", True), "frs.enabled"):
        raise ValueError("deploy_pi05_frs requires frs.enabled=true")
    _boolean(
        raw.get("verify_source_checkpoint_fingerprint", False),
        "frs.verify_source_checkpoint_fingerprint",
    )
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
        raise ValueError(f"missing FRS config values: {missing}")
    observation = _mapping(config.get("observation"), "observation")
    if observation.get("data_type") != "vitac":
        raise ValueError("FRS deployment requires observation.data_type='vitac'")
    control = _mapping(config.get("control"), "control")
    steps = _integer(control.get("steps_per_inference"), "control.steps_per_inference")
    horizon = _integer(control.get("action_horizon"), "control.action_horizon")
    if steps != horizon:
        raise ValueError("FRS deployment requires steps_per_inference == action_horizon")
    divisor = _integer(raw["tactile_window_divisor"], "frs.tactile_window_divisor")
    if horizon <= 0:
        raise ValueError(f"action_horizon must be positive, got {horizon}.")
    if divisor <= 0:
        raise ValueError(f"window_divisor must be positive, got {divisor}.")
    if horizon % divisor:
        raise ValueError(
            f"action_horizon ({horizon}) must be divisible by window_divisor ({divisor})."
        )
