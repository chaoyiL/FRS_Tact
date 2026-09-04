"""Contract metadata for the scalar composite FRS training objective."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

COMPOSITE_GATED_LOSS_MODE = "composite_gated"
COMPOSITE_GATED_OBJECTIVE_VERSION = 2
COMPOSITE_GATED_ENDPOINT_POLICY = "scalar_three_region_arm9_vla_gripper"
COMPOSITE_GATED_DECODE_POLICY = "scalar_three_region_arm9_vla_gripper"


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
