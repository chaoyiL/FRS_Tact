"""Fixed schema for the bimanual FRS objective."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

BIMANUAL_LOSS_MODE = "bimanual_gated"
BIMANUAL_OBJECTIVE_VERSION = 2
BIMANUAL_ACTION_DIM = 20
LEFT_ACTION_SLICE = slice(0, 10)
RIGHT_ACTION_SLICE = slice(10, 20)
LEFT_WRIST_TOKEN_INDICES = (0, 1)
RIGHT_WRIST_TOKEN_INDICES = (2, 3)
BIMANUAL_TACTILE_KEY_BASENAMES = (
    "tactile_left_0",
    "tactile_right_0",
    "tactile_left_1",
    "tactile_right_1",
)


def validate_bimanual_tactile_keys(
    tactile_keys: Sequence[object],
    *,
    field_name: str = "model.tactile_keys",
) -> tuple[str, ...]:
    """Require the fixed per-wrist token order while allowing observation prefixes."""

    if isinstance(tactile_keys, (str, bytes)):
        raise ValueError(
            f"{field_name} must contain exactly {BIMANUAL_TACTILE_KEY_BASENAMES!r} in order."
        )
    actual = tuple(str(key) for key in tactile_keys)
    basenames = tuple(key.rsplit(".", 1)[-1] for key in actual)
    if basenames != BIMANUAL_TACTILE_KEY_BASENAMES:
        raise ValueError(
            f"{field_name} must contain exactly {BIMANUAL_TACTILE_KEY_BASENAMES!r} "
            f"in order, got {basenames!r}."
        )
    return actual


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
        if (
            len(actual_indices) != len(expected_indices)
            or any(type(index) is not int for index in actual_indices)
            or actual_indices != expected_indices
        ):
            raise ValueError(f"Bimanual objective metadata has invalid {field_name}.{name}.")


def validate_bimanual_objective_metadata(metadata: Mapping[str, Any]) -> None:
    """Reject checkpoint metadata incompatible with the fixed bimanual contract."""

    if metadata.get("loss_mode") != BIMANUAL_LOSS_MODE:
        raise ValueError("Bimanual objective metadata has invalid loss_mode.")
    objective_version = metadata.get("loss_objective_version")
    if (
        type(objective_version) is not int
        or objective_version != BIMANUAL_OBJECTIVE_VERSION
    ):
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
