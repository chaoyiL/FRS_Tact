"""Pure helpers for the pick-tube bimanual relative-action v2 contract."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


ACTION_REPRESENTATION_VERSION = 2
ACTION_CONTRACT = "bimanual_relative_pose20d_v2"
TERMINAL_ACTION_POLICY = "canonical_relative_noop_v2"

LOW_TRANSLATION_DELTA_M = 0.0005
LOW_ROTATION_DELTA_DEG = 0.25
LOW_GRIPPER_DELTA_M = 0.0005
HIGH_TRANSLATION_DELTA_M = 0.0008
HIGH_ROTATION_DELTA_DEG = 0.4
HIGH_GRIPPER_DELTA_M = 0.0008
IDLE_ENTRY_FRAMES = 8
IDLE_EXIT_FRAMES = 2

ROTATION_EPSILON = 1e-6
IDENTITY_ROTATION_6D = np.asarray([1, 0, 0, 0, 1, 0], dtype=np.float32)


@dataclass(frozen=True)
class CanonicalEpisodeActions:
    action_raw: np.ndarray
    action: np.ndarray
    action_valid: np.ndarray
    idle_arm_mask: np.ndarray


def canonical_noop_from_state(state: np.ndarray) -> np.ndarray:
    """Return a 20D relative no-op with grippers held at the current widths."""
    state_array = np.asarray(state)
    if state_array.shape != (20,):
        raise ValueError(f"state must have shape (20,), got {state_array.shape}")
    if not np.isfinite(state_array[[6, 13]]).all():
        raise ValueError("state gripper widths must be finite")

    dtype = np.result_type(state_array.dtype, np.float32)
    noop = np.zeros(20, dtype=dtype)
    noop[3:9] = IDENTITY_ROTATION_6D
    noop[9] = state_array[6]
    noop[13:19] = IDENTITY_ROTATION_6D
    noop[19] = state_array[13]
    return noop


def _rotation_angle_degrees(rotation_6d: np.ndarray, *, row: int, arm: str) -> float:
    first = np.asarray(rotation_6d[:3], dtype=np.float64)
    second = np.asarray(rotation_6d[3:], dtype=np.float64)
    if not np.isfinite(first).all() or not np.isfinite(second).all():
        raise ValueError(f"degenerate rotation at row {row} for {arm} arm: non-finite basis")

    first_norm = np.linalg.norm(first)
    second_norm = np.linalg.norm(second)
    cross_norm = np.linalg.norm(np.cross(first, second))
    if (
        first_norm < ROTATION_EPSILON
        or second_norm < ROTATION_EPSILON
        or cross_norm < ROTATION_EPSILON
    ):
        raise ValueError(f"degenerate rotation at row {row} for {arm} arm")

    basis_x = first / first_norm
    second_orthogonal = second - np.dot(basis_x, second) * basis_x
    second_orthogonal_norm = np.linalg.norm(second_orthogonal)
    if second_orthogonal_norm < ROTATION_EPSILON:
        raise ValueError(f"degenerate rotation at row {row} for {arm} arm")
    basis_y = second_orthogonal / second_orthogonal_norm
    basis_z = np.cross(basis_x, basis_y)
    rotation = np.stack((basis_x, basis_y, basis_z), axis=1)
    cosine = np.clip((np.trace(rotation) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


def _motion_threshold_masks(action: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    length = action.shape[0]
    low_motion = np.zeros((length, 2), dtype=bool)
    high_motion = np.zeros((length, 2), dtype=bool)
    arm_layout = ((0, 3, 9, "left"), (10, 13, 19, "right"))

    # The source terminal action is invalid by policy and is excluded from
    # physical-motion classification and nonterminal rotation validation.
    for row in range(max(0, length - 1)):
        for arm_index, (position_start, rotation_start, gripper_index, arm_name) in enumerate(arm_layout):
            translation = float(np.linalg.norm(action[row, position_start:position_start + 3]))
            rotation = _rotation_angle_degrees(
                action[row, rotation_start:rotation_start + 6], row=row, arm=arm_name
            )
            previous_row = max(0, row - 1)
            gripper = float(abs(action[row, gripper_index] - action[previous_row, gripper_index]))
            if not np.isfinite((translation, gripper)).all():
                raise ValueError(f"non-finite motion at row {row} for {arm_name} arm")
            low_motion[row, arm_index] = (
                translation < LOW_TRANSLATION_DELTA_M
                and rotation < LOW_ROTATION_DELTA_DEG
                and gripper < LOW_GRIPPER_DELTA_M
            )
            high_motion[row, arm_index] = (
                translation > HIGH_TRANSLATION_DELTA_M
                or rotation > HIGH_ROTATION_DELTA_DEG
                or gripper > HIGH_GRIPPER_DELTA_M
            )
    return low_motion, high_motion


def _idle_hysteresis(low_motion: np.ndarray, high_motion: np.ndarray) -> np.ndarray:
    idle_mask = np.zeros_like(low_motion)
    for arm_index in range(2):
        idle = False
        entry_count = 0
        exit_count = 0
        other_arm_index = 1 - arm_index
        for row in range(low_motion.shape[0]):
            if idle:
                if high_motion[row, arm_index]:
                    exit_count += 1
                else:
                    exit_count = 0
                if exit_count >= IDLE_EXIT_FRAMES:
                    idle = False
                    entry_count = 0
                    exit_count = 0
            else:
                if low_motion[row, arm_index] and high_motion[row, other_arm_index]:
                    entry_count += 1
                else:
                    entry_count = 0
                if entry_count >= IDLE_ENTRY_FRAMES:
                    idle = True
                    exit_count = 0
            idle_mask[row, arm_index] = idle
    return idle_mask


def _canonicalize_arm(action: np.ndarray, state: np.ndarray, rows: np.ndarray, arm: int) -> None:
    action_offset = arm * 10
    state_gripper_index = 6 if arm == 0 else 13
    action[rows, action_offset:action_offset + 3] = 0
    action[rows, action_offset + 3:action_offset + 9] = IDENTITY_ROTATION_6D
    action[rows, action_offset + 9] = state[rows, state_gripper_index]


def canonicalize_episode_actions(
    state: np.ndarray, action: np.ndarray
) -> CanonicalEpisodeActions:
    """Validate and canonicalize one episode without mutating caller arrays."""
    state_array = np.asarray(state)
    action_array = np.asarray(action)
    if state_array.ndim != 2 or state_array.shape[1:] != (20,):
        raise ValueError(f"state must have shape [T,20], got {state_array.shape}")
    if action_array.shape != state_array.shape:
        raise ValueError(
            f"action must match state shape {state_array.shape}, got {action_array.shape}"
        )
    if state_array.shape[0] == 0:
        raise ValueError("episode must contain at least one row")
    if not np.issubdtype(action_array.dtype, np.floating):
        raise ValueError(f"action must use a floating dtype, got {action_array.dtype}")

    action_raw = np.array(action_array, copy=True)
    canonical_action = np.array(action_array, copy=True)
    action_valid = np.ones(action_array.shape[0], dtype=bool)
    action_valid[-1] = False

    low_motion, high_motion = _motion_threshold_masks(action_raw)
    idle_arm_mask = _idle_hysteresis(low_motion, high_motion)
    idle_arm_mask[-1] = False
    for arm_index in range(2):
        rows = idle_arm_mask[:, arm_index]
        _canonicalize_arm(canonical_action, state_array, rows, arm_index)

    canonical_action[-1] = canonical_noop_from_state(state_array[-1]).astype(
        canonical_action.dtype, copy=False
    )
    return CanonicalEpisodeActions(
        action_raw=action_raw,
        action=canonical_action,
        action_valid=action_valid,
        idle_arm_mask=idle_arm_mask,
    )
