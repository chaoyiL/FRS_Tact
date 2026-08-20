"""Pure NumPy metrics for bimanual Gate regions and error quadrants."""

from __future__ import annotations

from typing import Any

import numpy as np


BIMANUAL_QUADRANTS = ("low_low", "high_low", "low_high", "high_high")
BIMANUAL_WRISTS = ("left", "right")
_WRIST_METRICS = (
    "mse_gt",
    "mse_vla",
    "mse_vla_gt",
    "gt_gain",
    "relative_gt_error",
    "vla_preserve_ratio",
    "rank_satisfied_frac",
)


def _as_bimanual_array(value: Any) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != 2:
        raise ValueError("bimanual quadrant inputs must all have shape [N, 2]")
    if not np.all(np.isfinite(array)):
        raise ValueError("bimanual quadrant inputs must have matching finite values")
    return array


def _validate_thresholds(low_threshold: float, high_threshold: float) -> tuple[float, float]:
    low = float(low_threshold)
    high = float(high_threshold)
    if not np.isfinite(low) or not np.isfinite(high) or not (0.0 <= low < high <= 1.0):
        raise ValueError(
            "low_threshold and high_threshold must be finite, satisfy 0 <= low < high <= 1"
        )
    return low, high


def _quadrant_masks(gates: np.ndarray, low: float, high: float) -> dict[str, np.ndarray]:
    is_low = gates <= low
    is_high = gates >= high
    return {
        "low_low": is_low[:, 0] & is_low[:, 1],
        "high_low": is_high[:, 0] & is_low[:, 1],
        "low_high": is_low[:, 0] & is_high[:, 1],
        "high_high": is_high[:, 0] & is_high[:, 1],
    }


def bimanual_quadrant_metrics(
    *,
    mse_gt: Any,
    mse_vla: Any,
    mse_vla_gt: Any,
    gate_weights: Any,
    low_threshold: float,
    high_threshold: float,
    ranking_margin: float = 0.0,
) -> dict[str, dict[str, object]]:
    """Aggregate per-wrist errors by independent low/high Gate quadrants.

    All metric inputs have shape ``[N, 2]`` in fixed left/right order. Threshold
    boundaries are inclusive; samples with a mid-region Gate are not assigned to
    a quadrant. Empty quadrants retain their shape with NaN metric values.
    """
    low, high = _validate_thresholds(low_threshold, high_threshold)
    margin = float(ranking_margin)
    if not np.isfinite(margin) or margin < 0:
        raise ValueError("ranking_margin must be finite and non-negative")

    arrays = tuple(
        _as_bimanual_array(value) for value in (mse_gt, mse_vla, mse_vla_gt, gate_weights)
    )
    if len({array.shape for array in arrays}) != 1:
        raise ValueError("bimanual quadrant inputs must have matching finite values")
    gt, vla, baseline, gates = arrays

    output: dict[str, dict[str, object]] = {}
    for quadrant, mask in _quadrant_masks(gates, low, high).items():
        group: dict[str, object] = {"n": int(np.count_nonzero(mask))}
        for wrist_index, wrist in enumerate(BIMANUAL_WRISTS):
            if not np.any(mask):
                group[wrist] = {metric: float("nan") for metric in _WRIST_METRICS}
                continue

            mean_gt = float(np.mean(gt[mask, wrist_index]))
            mean_vla = float(np.mean(vla[mask, wrist_index]))
            mean_baseline = float(np.mean(baseline[mask, wrist_index]))
            denominator = max(mean_baseline, 1e-8)
            group[wrist] = {
                "mse_gt": mean_gt,
                "mse_vla": mean_vla,
                "mse_vla_gt": mean_baseline,
                "gt_gain": mean_baseline - mean_gt,
                "relative_gt_error": mean_gt / denominator,
                "vla_preserve_ratio": mean_vla / denominator,
                "rank_satisfied_frac": float(
                    np.mean(gt[mask, wrist_index] + margin <= vla[mask, wrist_index])
                ),
            }
        output[quadrant] = group
    return output


def flatten_bimanual_quadrant_metrics(
    metrics: dict[str, dict[str, object]], *, prefix: str = "val_quadrant"
) -> dict[str, float | int]:
    """Flatten nested quadrant metrics into logger/JSON-friendly scalar keys."""
    flattened: dict[str, float | int] = {}
    for quadrant in BIMANUAL_QUADRANTS:
        group = metrics[quadrant]
        flattened[f"{prefix}_{quadrant}_n"] = int(group["n"])
        for wrist in BIMANUAL_WRISTS:
            wrist_metrics = group[wrist]
            for metric in _WRIST_METRICS:
                value = wrist_metrics[metric]
                flattened[f"{prefix}_{quadrant}_{metric}_{wrist}"] = float(value)
    return flattened


def bimanual_gate_region_counts(
    gate_weights: Any, *, low_threshold: float, high_threshold: float
) -> np.ndarray:
    """Return a 3x3 left-by-right count matrix for low/mid/high Gate regions."""
    low, high = _validate_thresholds(low_threshold, high_threshold)
    gates = _as_bimanual_array(gate_weights)
    regions = np.full(gates.shape, 1, dtype=np.intp)  # middle region
    regions[gates <= low] = 0
    regions[gates >= high] = 2
    counts = np.zeros((3, 3), dtype=np.intp)
    np.add.at(counts, (regions[:, 0], regions[:, 1]), 1)
    return counts
