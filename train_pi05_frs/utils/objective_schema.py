"""Contract metadata for the scalar composite FRS training objective."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

COMPOSITE_GATED_LOSS_MODE = "composite_gated"
COMPOSITE_GATED_OBJECTIVE_VERSION = 2
COMPOSITE_GATED_ENDPOINT_POLICY = "scalar_three_region_arm9_vla_gripper"
COMPOSITE_GATED_DECODE_POLICY = "scalar_three_region_arm9_vla_gripper"

LEARNED_RESIDUAL_GATED_LOSS_MODE = "learned_residual_gated"
LEARNED_RESIDUAL_GATED_OBJECTIVE_VERSION = 3
LEARNED_RESIDUAL_GATED_ARCHITECTURE = "single_arm_vla_residual_gate_v1"


def composite_gated_objective_metadata() -> dict[str, Any]:
    return {
        "loss_objective_version": COMPOSITE_GATED_OBJECTIVE_VERSION,
        "endpoint_policy": COMPOSITE_GATED_ENDPOINT_POLICY,
        "decode_policy": COMPOSITE_GATED_DECODE_POLICY,
        "action_dim": 10,
        "steered_action_dim": 9,
        "gripper_index": 9,
        "gripper_policy": "vla_endpoint_preserved",
    }


def validate_composite_gated_objective_metadata(extra: Mapping[str, Any]) -> None:
    expected = {
        "loss_mode": COMPOSITE_GATED_LOSS_MODE,
        **composite_gated_objective_metadata(),
    }
    for field, value in expected.items():
        if extra.get(field) != value:
            raise ValueError(
                f"composite_gated checkpoint metadata {field} mismatch: "
                f"{extra.get(field)!r} != {value!r}"
            )


def learned_residual_gated_objective_metadata(
    *,
    oracle_safe_mse_threshold: float,
    oracle_repair_mse_threshold: float,
    residual_bound: float,
) -> dict[str, Any]:
    safe = float(oracle_safe_mse_threshold)
    repair = float(oracle_repair_mse_threshold)
    bound = float(residual_bound)
    if not 0.0 <= safe < repair:
        raise ValueError(
            "oracle thresholds must satisfy 0 <= safe < repair, got "
            f"{safe}, {repair}"
        )
    if not math.isfinite(bound) or bound <= 0.0:
        raise ValueError(f"residual_bound must be finite and positive, got {bound}")
    return {
        "loss_objective_version": LEARNED_RESIDUAL_GATED_OBJECTIVE_VERSION,
        "model_architecture": LEARNED_RESIDUAL_GATED_ARCHITECTURE,
        "action_dim": 10,
        "steered_action_dim": 9,
        "gripper_index": 9,
        "gripper_policy": "vla_runtime_preserved",
        "gate_granularity": "chunk",
        "residual_parameterization": "bounded_normalized_vla_additive",
        "gate_label_policy": "arm9_chunk_mse_two_threshold",
        "oracle_safe_mse_threshold": safe,
        "oracle_repair_mse_threshold": repair,
        "residual_bound": bound,
    }


def validate_learned_residual_gated_objective_metadata(
    extra: Mapping[str, Any],
    *,
    oracle_safe_mse_threshold: float | None = None,
    oracle_repair_mse_threshold: float | None = None,
    residual_bound: float | None = None,
) -> None:
    def value(field: str, override: float | None) -> float:
        raw = extra.get(field) if override is None else override
        if raw is None:
            raise ValueError(
                f"learned_residual_gated checkpoint metadata missing {field}"
            )
        try:
            return float(raw)
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"learned_residual_gated checkpoint metadata {field} "
                f"must be numeric, got {raw!r}"
            ) from error

    safe = value("oracle_safe_mse_threshold", oracle_safe_mse_threshold)
    repair = value("oracle_repair_mse_threshold", oracle_repair_mse_threshold)
    bound = value("residual_bound", residual_bound)
    expected = {
        "loss_mode": LEARNED_RESIDUAL_GATED_LOSS_MODE,
        **learned_residual_gated_objective_metadata(
            oracle_safe_mse_threshold=safe,
            oracle_repair_mse_threshold=repair,
            residual_bound=bound,
        ),
    }
    for field, value in expected.items():
        if extra.get(field) != value:
            raise ValueError(
                f"learned_residual_gated checkpoint metadata {field} mismatch: "
                f"{extra.get(field)!r} != {value!r}"
            )
