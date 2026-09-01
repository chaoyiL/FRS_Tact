"""Adapters between the single-right policy and the bimanual bridge wire format."""

from collections.abc import Mapping
from typing import Any

import numpy as np


def _bridge_state(observation: Mapping[str, Any]) -> np.ndarray:
    state = np.asarray(observation["observation.state"], dtype=np.float32)
    if state.shape != (20,):
        raise ValueError("bridge observation.state must have shape (20,)")
    if not np.isfinite(state).all():
        raise ValueError("bridge observation.state must contain only finite values")
    return state


def project_right_state(observation: Mapping[str, Any]) -> np.ndarray:
    """Project a 20D bimanual bridge state to the policy's 7D right-arm state."""
    return _bridge_state(observation)[7:14].copy()


def expand_right_action(action: Any, observation: Mapping[str, Any]) -> np.ndarray:
    """Expand a policy ``[H, 10]`` right-arm action to a bimanual ``[H, 20]`` action."""
    state = _bridge_state(observation)
    right = np.asarray(action, dtype=np.float32)
    if right.ndim != 2 or right.shape[1] != 10:
        raise ValueError("right action must have shape [H, 10]")
    if not np.isfinite(right).all():
        raise ValueError("right action must contain only finite values")

    left = np.empty((right.shape[0], 10), dtype=np.float32)
    left[:, :3] = 0
    left[:, 3:9] = np.array([1, 0, 0, 0, 1, 0], dtype=np.float32)
    left[:, 9] = state[6]
    return np.concatenate((left, right), axis=1)
