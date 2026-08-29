"""Runtime phase gate for the single-weight Bread DECO policy."""

from __future__ import annotations

import time

import numpy as np


class BreadPhaseTimeout(RuntimeError):
    """The bread/right-arm phase did not finish before its deadline."""


class BreadPhaseController:
    """Mask inactive arm commands and advance only after right-gripper release."""

    _IDENTITY_6D = np.array([1, 0, 0, 0, 1, 0], dtype=np.float32)

    def __init__(self, timeout_s: float = 15.0) -> None:
        if timeout_s <= 0:
            raise ValueError("timeout_s must be positive")
        self.timeout_s = float(timeout_s)
        self.phase = 0
        self._phase0_started_at: float | None = None
        self._saw_right_closed = False
        self._right_open_streak = 0

    def apply(
        self,
        state: np.ndarray,
        action: np.ndarray,
        *,
        now_s: float | None = None,
    ) -> np.ndarray:
        """Update the phase state and return a safe 20D action command."""
        current_time = time.monotonic() if now_s is None else float(now_s)
        state_array = self.observe(state, now_s=current_time)
        return self.mask(state_array, action)

    def observe(self, state: np.ndarray, *, now_s: float | None = None) -> np.ndarray:
        """Consume exactly one measured robot state and update the phase."""
        current_time = time.monotonic() if now_s is None else float(now_s)
        state_array = np.asarray(state, dtype=np.float32)
        if state_array.shape != (20,):
            raise ValueError(f"state must have shape (20,), got {state_array.shape}")
        if self.phase == 0:
            self._update_phase0(state_array, current_time)
        return state_array

    def mask(self, state: np.ndarray, action: np.ndarray) -> np.ndarray:
        """Mask one command without consuming the observation again."""
        state_array = np.asarray(state, dtype=np.float32)
        action_array = np.asarray(action, dtype=np.float32)
        if state_array.shape != (20,):
            raise ValueError(f"state must have shape (20,), got {state_array.shape}")
        if action_array.shape != (20,):
            raise ValueError(f"action must have shape (20,), got {action_array.shape}")
        filtered = action_array.copy()
        if self.phase == 0:
            filtered[0:3] = 0.0
            filtered[3:9] = self._IDENTITY_6D
            filtered[9] = state_array[6]
        else:
            filtered[10:13] = 0.0
            filtered[13:19] = self._IDENTITY_6D
            filtered[19] = 0.12
        return filtered

    def mask_chunk(self, state: np.ndarray, actions: np.ndarray) -> np.ndarray:
        actions_array = np.asarray(actions, dtype=np.float32)
        if actions_array.ndim != 2 or actions_array.shape[1] != 20:
            raise ValueError(f"actions must have shape (T, 20), got {actions_array.shape}")
        return np.stack([self.mask(state, action) for action in actions_array], axis=0)

    def _update_phase0(self, state: np.ndarray, now_s: float) -> None:
        if self._phase0_started_at is None:
            self._phase0_started_at = now_s
        if now_s - self._phase0_started_at > self.timeout_s:
            raise BreadPhaseTimeout(
                f"bread phase exceeded {self.timeout_s:g}s; refusing to advance to phase 1"
            )

        right_gripper = float(state[13])
        if not self._saw_right_closed:
            self._saw_right_closed = right_gripper <= 0.09
            return
        if right_gripper >= 0.10:
            self._right_open_streak += 1
        else:
            self._right_open_streak = 0
        if self._right_open_streak >= 2:
            self.phase = 1
