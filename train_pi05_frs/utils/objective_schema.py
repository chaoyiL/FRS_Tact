"""Contract metadata for the scalar composite FRS training objective."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

COMPOSITE_GATED_LOSS_MODE = "composite_gated"
COMPOSITE_GATED_OBJECTIVE_VERSION = 1
COMPOSITE_GATED_ENDPOINT_POLICY = "scalar_three_region_composite"
COMPOSITE_GATED_DECODE_POLICY = "scalar_three_region_composite"


def composite_gated_objective_metadata() -> dict[str, Any]:
    return {
        "loss_objective_version": COMPOSITE_GATED_OBJECTIVE_VERSION,
        "endpoint_policy": COMPOSITE_GATED_ENDPOINT_POLICY,
        "decode_policy": COMPOSITE_GATED_DECODE_POLICY,
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
