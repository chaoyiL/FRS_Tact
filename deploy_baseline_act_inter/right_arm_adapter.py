"""Adapt a single-right-arm Pi0.5 policy to the bimanual robot wire contract."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np


def _server_state(observation: Mapping[str, Any]) -> np.ndarray:
    state = np.asarray(observation.get("observation.state"), dtype=np.float32)
    if state.shape != (20,) or not np.isfinite(state).all():
        raise ValueError(
            f"bimanual server state must be finite with shape (20,), got {state.shape}"
        )
    return state


def project_right_observation(observation: Mapping[str, Any]) -> dict[str, Any]:
    """Project the bimanual server observation into the right-arm model contract."""

    result = dict(observation)
    result["observation.state"] = _server_state(observation)[7:14].copy()
    return result


def expand_right_action(action: Any, observation: Mapping[str, Any]) -> np.ndarray:
    """Place a 10D right-arm chunk in the 20D wire action and hold the left arm."""

    right = np.asarray(action, dtype=np.float32)
    if right.ndim != 2 or right.shape[1] != 10 or not np.isfinite(right).all():
        raise ValueError(
            f"right-arm action must be finite with shape (H,10), got {right.shape}"
        )
    state = _server_state(observation)
    left = np.zeros_like(right)
    left[:, 3] = 1.0
    left[:, 7] = 1.0
    left[:, 9] = state[6]
    return np.ascontiguousarray(np.concatenate((left, right), axis=1))
