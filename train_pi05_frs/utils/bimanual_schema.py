"""Fixed schema for Pi0.5's physical bimanual FRS objective."""

from collections.abc import Mapping, Sequence
from typing import Any


BIMANUAL_LOSS_MODE = "bimanual_gated"
BIMANUAL_OBJECTIVE_VERSION = 2
LOSS_WEIGHTING_VERSION = 7
STEERED_ACTION_DIM = 20
LEFT_ACTION_SLICE = slice(0, 10)
RIGHT_ACTION_SLICE = slice(10, 20)
LEFT_WRIST_TOKEN_INDICES = (0, 1)
RIGHT_WRIST_TOKEN_INDICES = (2, 3)
PADDED_TAIL_POLICY = "vla_endpoint_masked"
BIMANUAL_TACTILE_KEY_BASENAMES = (
    "tactile_left_0",
    "tactile_right_0",
    "tactile_left_1",
    "tactile_right_1",
)


def validate_bimanual_tactile_keys(
    tactile_keys: Sequence[object], *, field_name: str = "model.tactile_keys"
) -> tuple[str, ...]:
    if isinstance(tactile_keys, (str, bytes)):
        raise ValueError(f"{field_name} must contain the fixed bimanual tactile key order")
    actual = tuple(str(key) for key in tactile_keys)
    basenames = tuple(key.rsplit(".", 1)[-1] for key in actual)
    if basenames != BIMANUAL_TACTILE_KEY_BASENAMES:
        raise ValueError(
            f"{field_name} must contain {BIMANUAL_TACTILE_KEY_BASENAMES!r}, got {basenames!r}"
        )
    return actual


def bimanual_objective_metadata(*, action_dim: int) -> dict[str, object]:
    if action_dim < STEERED_ACTION_DIM:
        raise ValueError(f"bimanual objective requires action_dim >= {STEERED_ACTION_DIM}")
    return {
        "loss_mode": BIMANUAL_LOSS_MODE,
        "loss_objective_version": BIMANUAL_OBJECTIVE_VERSION,
        "loss_weighting_version": LOSS_WEIGHTING_VERSION,
        "action_dim": int(action_dim),
        "steered_action_dim": STEERED_ACTION_DIM,
        "action_slices": {"left": [0, 10], "right": [10, 20]},
        "wrist_token_indices": {"left": [0, 1], "right": [2, 3]},
        "padded_tail_policy": PADDED_TAIL_POLICY,
    }


def validate_bimanual_objective_metadata(
    metadata: Mapping[str, Any], *, action_dim: int
) -> None:
    expected = bimanual_objective_metadata(action_dim=action_dim)
    for field in (
        "loss_mode",
        "loss_objective_version",
        "loss_weighting_version",
        "action_dim",
        "steered_action_dim",
        "action_slices",
        "wrist_token_indices",
        "padded_tail_policy",
    ):
        if metadata.get(field) != expected[field]:
            raise ValueError(f"bimanual objective metadata has invalid {field}")
