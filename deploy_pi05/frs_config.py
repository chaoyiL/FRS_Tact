"""Dependency-light validation for the FRS deployment configuration."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import math
from numbers import Integral
from numbers import Real
from typing import Any


@dataclass(frozen=True)
class GripperHysteresisConfig:
    left_close_threshold: float
    left_reopen_threshold: float
    left_closed_command: float
    right_close_threshold: float
    right_reopen_threshold: float
    right_closed_command: float


@dataclass(frozen=True)
class Task1MotionGainConfig:
    approach_translation_gain: float
    translation_gain: float
    rotation_gain: float


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


def parse_task_switch(config: Mapping[str, Any]) -> int:
    task = config.get("task", 0)
    if (
        isinstance(task, bool)
        or not isinstance(task, Integral)
        or int(task) not in (0, 1)
    ):
        raise ValueError("task must be 0 or 1")
    return int(task)


def parse_task1_motion_gain_config(
    config: Mapping[str, Any],
) -> Task1MotionGainConfig:
    if parse_task_switch(config) != 1:
        return Task1MotionGainConfig(
            approach_translation_gain=1.0,
            translation_gain=1.0,
            rotation_gain=1.0,
        )
    raw = _mapping(config.get("task1"), "task1")
    return Task1MotionGainConfig(
        approach_translation_gain=_bounded_float(
            raw.get("approach_translation_gain"),
            "task1.approach_translation_gain",
            minimum=0.1,
            maximum=3.0,
        ),
        translation_gain=_bounded_float(
            raw.get("translation_gain"),
            "task1.translation_gain",
            minimum=0.1,
            maximum=3.0,
        ),
        rotation_gain=_bounded_float(
            raw.get("rotation_gain"),
            "task1.rotation_gain",
            minimum=0.1,
            maximum=2.0,
        ),
    )


def _bounded_float(value: Any, name: str, *, minimum: float, maximum: float) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite number")
    parsed = float(value)
    if not math.isfinite(parsed) or not minimum <= parsed <= maximum:
        raise ValueError(f"{name} must be in [{minimum}, {maximum}]")
    return parsed


def parse_gripper_hysteresis_config(
    config: Mapping[str, Any],
) -> GripperHysteresisConfig:
    raw = _mapping(config.get("gripper"), "gripper")
    values: dict[str, float] = {}
    for side in ("left", "right"):
        close_name = f"{side}_close_threshold"
        reopen_name = f"{side}_reopen_threshold"
        closed_name = f"{side}_closed_command"
        close = _bounded_float(
            raw.get(close_name),
            f"gripper.{close_name}",
            minimum=0.0,
            maximum=1.05,
        )
        reopen = _bounded_float(
            raw.get(reopen_name),
            f"gripper.{reopen_name}",
            minimum=0.0,
            maximum=1.05,
        )
        if close >= reopen:
            raise ValueError(
                f"gripper.{close_name} must be less than gripper.{reopen_name}"
            )
        values[close_name] = close
        values[reopen_name] = reopen
        values[closed_name] = _bounded_float(
            raw.get(closed_name),
            f"gripper.{closed_name}",
            minimum=0.01,
            maximum=0.04,
        )
    return GripperHysteresisConfig(**values)


def validate_frs_config_section(config: Mapping[str, Any]) -> None:
    """Validate the FRS profile without importing JAX or training modules."""
    raw = _mapping(config.get("frs"), "frs")
    task = parse_task_switch(config)
    parse_task1_motion_gain_config(config)
    if not _boolean(raw.get("enabled", True), "frs.enabled"):
        raise ValueError("deploy_pi05 requires frs.enabled=true")
    parse_gripper_hysteresis_config(config)
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
    if task == 1:
        controller_frequency = control.get("controller_frequency")
        if (
            isinstance(controller_frequency, bool)
            or not isinstance(controller_frequency, Real)
            or not math.isfinite(float(controller_frequency))
            or float(controller_frequency) != 80.0
        ):
            raise ValueError("Task 1 control.controller_frequency must be 80.0")
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
