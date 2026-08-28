"""Explicit state/action contracts supported by the LeRobot DECO adapter."""

from __future__ import annotations

from dataclasses import dataclass


DUAL_ARM_PROFILE = "dual-arm-20x20"
SINGLE_RIGHT_ARM_PROFILE = "single-right-arm-7x10"


_ACTION_COMPONENTS = (
    "delta.x",
    "delta.y",
    "delta.z",
    "rotation_column_0.x",
    "rotation_column_0.y",
    "rotation_column_0.z",
    "rotation_column_1.x",
    "rotation_column_1.y",
    "rotation_column_1.z",
    "gripper_width",
)


@dataclass(frozen=True)
class StateActionProfile:
    name: str
    state_dim: int
    action_dim: int
    state_columns: tuple[str, ...]
    action_columns: tuple[str, ...]
    state_layout: str
    controlled_arms: tuple[str, ...]


DUAL_ARM_20X20 = StateActionProfile(
    name=DUAL_ARM_PROFILE,
    state_dim=20,
    action_dim=20,
    state_columns=(
        "robot0.relative_start.x",
        "robot0.relative_start.y",
        "robot0.relative_start.z",
        "robot0.relative_start.rx",
        "robot0.relative_start.ry",
        "robot0.relative_start.rz",
        "robot0.gripper_width",
        "robot1.relative_start.x",
        "robot1.relative_start.y",
        "robot1.relative_start.z",
        "robot1.relative_start.rx",
        "robot1.relative_start.ry",
        "robot1.relative_start.rz",
        "robot1.gripper_width",
        "left_relative_to_right.x",
        "left_relative_to_right.y",
        "left_relative_to_right.z",
        "left_relative_to_right.rx",
        "left_relative_to_right.ry",
        "left_relative_to_right.rz",
    ),
    action_columns=tuple(
        f"robot{robot_index}.{component}"
        for robot_index in range(2)
        for component in _ACTION_COMPONENTS
    ),
    state_layout="relative_start_pose6d_gripper_plus_left_relative_right",
    controlled_arms=("left", "right"),
)


SINGLE_RIGHT_ARM_7X10 = StateActionProfile(
    name=SINGLE_RIGHT_ARM_PROFILE,
    state_dim=7,
    action_dim=10,
    state_columns=(
        "right_arm.relative_start.x",
        "right_arm.relative_start.y",
        "right_arm.relative_start.z",
        "right_arm.relative_start.rx",
        "right_arm.relative_start.ry",
        "right_arm.relative_start.rz",
        "right_arm.gripper_width",
    ),
    action_columns=tuple(f"right_arm.{component}" for component in _ACTION_COMPONENTS),
    state_layout="single_right_relative_start_pose6d_gripper",
    controlled_arms=("right",),
)


PROFILES = {
    profile.name: profile
    for profile in (DUAL_ARM_20X20, SINGLE_RIGHT_ARM_7X10)
}


def resolve_state_action_profile(
    requested: str | None,
    state_shape: tuple[int, ...],
    action_shape: tuple[int, ...],
) -> StateActionProfile:
    """Resolve and validate an explicit profile without guessing single-arm handedness."""

    if requested is not None:
        try:
            profile = PROFILES[requested]
        except KeyError as exc:
            raise ValueError(
                f"Unsupported state_action_profile={requested!r}; "
                f"expected one of {sorted(PROFILES)}"
            ) from exc
        expected = ((profile.state_dim,), (profile.action_dim,))
        if (state_shape, action_shape) != expected:
            raise ValueError(
                f"state_action_profile={requested!r} requires state/action shapes "
                f"{expected}, got {(state_shape, action_shape)}"
            )
        return profile

    if state_shape == (DUAL_ARM_20X20.state_dim,) and action_shape == (
        DUAL_ARM_20X20.action_dim,
    ):
        return DUAL_ARM_20X20

    if state_shape == (SINGLE_RIGHT_ARM_7X10.state_dim,) and action_shape == (
        SINGLE_RIGHT_ARM_7X10.action_dim,
    ):
        raise ValueError(
            "The 7D state / 10D action contract requires an explicit handedness. "
            f"Regenerate the manifest with --state-action-profile {SINGLE_RIGHT_ARM_PROFILE}."
        )

    raise ValueError(
        "Unsupported LeRobot state/action shapes: "
        f"state={state_shape}, action={action_shape}"
    )
