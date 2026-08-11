"""Base-model-agnostic helpers for reading `LeRobotDataset` samples.

Extracted from `lerobot.policies.smolvla_jax.data` (which re-exports these for backward
compatibility) so that other base models -- e.g. `lerobot.policies.pi05_jax` -- can parse the
same LeRobot sample dicts without importing SmolVLA's config/preprocessing code. These functions
only depend on the dataset's feature/sample schema, not on any particular policy.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import torch


def resolve_action_key(features: Mapping[str, Any], action_key: str | None = None) -> str:
    """Resolve both current ``action`` and legacy/custom ``actions`` feature names."""

    if action_key is not None:
        if action_key not in features:
            raise KeyError(f"action feature {action_key!r} is absent from the dataset")
        return action_key
    matches = [key for key in ("action", "actions") if key in features]
    if len(matches) != 1:
        raise ValueError(
            "could not unambiguously find the dataset action feature; pass action_key explicitly"
        )
    return matches[0]


def action_delta_timestamps(action_key: str, chunk_size: int, fps: int) -> dict[str, list[float]]:
    if fps <= 0:
        raise ValueError(f"dataset FPS must be positive, got {fps}")
    return {action_key: [index / fps for index in range(chunk_size)]}


def to_numpy(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def lerobot_sample_to_observation(sample: Mapping[str, Any]) -> dict[str, np.ndarray]:
    return {key: to_numpy(value) for key, value in sample.items() if key.startswith("observation.")}
