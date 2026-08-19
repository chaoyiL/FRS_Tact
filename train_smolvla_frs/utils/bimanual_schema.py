"""Fixed schema for the bimanual FRS objective."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

BIMANUAL_LOSS_MODE = "bimanual_gated"
BIMANUAL_OBJECTIVE_VERSION = 2
BIMANUAL_ACTION_DIM = 20
LEFT_ACTION_SLICE = slice(0, 10)
RIGHT_ACTION_SLICE = slice(10, 20)
LEFT_WRIST_TOKEN_INDICES = (0, 1)
RIGHT_WRIST_TOKEN_INDICES = (2, 3)


def _validate_index_groups(
    metadata: Mapping[str, Any],
    *,
    field_name: str,
    expected: Mapping[str, tuple[int, int]],
) -> None:
    actual = metadata.get(field_name)
    if not isinstance(actual, Mapping) or set(actual) != set(expected):
        raise ValueError(f"Bimanual objective metadata has invalid {field_name}.")
    for name, expected_indices in expected.items():
        try:
            actual_indices = tuple(actual[name])
        except TypeError as error:
            raise ValueError(
                f"Bimanual objective metadata has invalid {field_name}.{name}."
            ) from error
        if actual_indices != expected_indices:
            raise ValueError(f"Bimanual objective metadata has invalid {field_name}.{name}.")


def validate_bimanual_objective_metadata(metadata: Mapping[str, Any]) -> None:
    """Reject checkpoint metadata incompatible with the fixed bimanual contract."""

    if metadata.get("loss_mode") != BIMANUAL_LOSS_MODE:
        raise ValueError("Bimanual objective metadata has invalid loss_mode.")
    if metadata.get("loss_objective_version") != BIMANUAL_OBJECTIVE_VERSION:
        raise ValueError("Bimanual objective metadata has invalid loss_objective_version.")
    _validate_index_groups(
        metadata,
        field_name="action_slices",
        expected={"left": (LEFT_ACTION_SLICE.start, LEFT_ACTION_SLICE.stop), "right": (RIGHT_ACTION_SLICE.start, RIGHT_ACTION_SLICE.stop)},
    )
    _validate_index_groups(
        metadata,
        field_name="wrist_token_indices",
        expected={"left": LEFT_WRIST_TOKEN_INDICES, "right": RIGHT_WRIST_TOKEN_INDICES},
    )
