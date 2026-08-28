"""Adapt a single-right-arm policy to the bimanual robot wire contract."""

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


def project_right_observation(
    observation: Mapping[str, Any], *, black_camera0: bool = False
) -> dict[str, Any]:
    result = dict(observation)
    result["observation.state"] = _server_state(observation)[7:14].copy()
    if black_camera0:
        key = "observation.images.camera0"
        if key not in observation:
            raise ValueError(f"right-arm observation is missing {key}")
        result[key] = np.zeros_like(np.asarray(observation[key]))
    return result


def expand_right_action(action: Any, observation: Mapping[str, Any]) -> np.ndarray:
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
    return np.concatenate((left, right), axis=1)
