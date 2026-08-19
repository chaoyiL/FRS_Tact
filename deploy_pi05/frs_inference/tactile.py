"""Small tactile helpers required by online FRS deployment."""

from __future__ import annotations

import numpy as np

def resolve_tactile_window(*, action_horizon: int, window_divisor: int) -> int:
    """Return window = action_horizon // window_divisor (must divide evenly)."""

    if action_horizon <= 0:
        raise ValueError(f"action_horizon must be positive, got {action_horizon}.")
    if window_divisor <= 0:
        raise ValueError(f"window_divisor must be positive, got {window_divisor}.")
    if action_horizon % window_divisor != 0:
        raise ValueError(
            f"action_horizon ({action_horizon}) must be divisible by "
            f"window_divisor ({window_divisor})."
        )
    window = action_horizon // window_divisor
    if window <= 0:
        raise ValueError(f"Resolved tactile window must be positive, got {window}.")
    return window

def _l2_normalize(vectors: np.ndarray, *, eps: float = 1e-8) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=-1, keepdims=True)
    return vectors / np.maximum(norms, eps)


def tactile_change_from_tokens(
    current_tokens: np.ndarray,
    baseline_tokens: np.ndarray,
) -> np.ndarray:
    """Per-sample tactile change ``s = mean_i(1 - cos)`` for tokens ``[B, 4, D]``."""

    if current_tokens.ndim != 3 or baseline_tokens.ndim != 3:
        raise ValueError(
            f"Expected tokens [B, 4, D], got current={current_tokens.shape}, "
            f"baseline={baseline_tokens.shape}."
        )
    if current_tokens.shape != baseline_tokens.shape:
        raise ValueError(
            f"current/baseline shape mismatch: {current_tokens.shape} vs {baseline_tokens.shape}."
        )
    current_n = _l2_normalize(current_tokens.astype(np.float32))
    baseline_n = _l2_normalize(baseline_tokens.astype(np.float32))
    cosine = np.sum(current_n * baseline_n, axis=-1)  # [B, 4]
    return np.mean(1.0 - cosine, axis=-1).astype(np.float32)
