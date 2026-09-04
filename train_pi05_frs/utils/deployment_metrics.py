"""Deployment-aligned endpoint metrics for single-right-arm FRS checkpoints."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np


def _validate_actions(
    frs_actions: np.ndarray,
    gt_actions: np.ndarray,
    vla_actions: np.ndarray,
    gate_weights: np.ndarray,
    *,
    gripper_index: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    frs = np.asarray(frs_actions, dtype=np.float32)
    gt = np.asarray(gt_actions, dtype=np.float32)
    vla = np.asarray(vla_actions, dtype=np.float32)
    gates = np.asarray(gate_weights, dtype=np.float32)
    if frs.shape != gt.shape or frs.shape != vla.shape:
        raise ValueError(
            f"FRS/GT/VLA action shape mismatch: {frs.shape}, {gt.shape}, {vla.shape}"
        )
    if frs.ndim != 3:
        raise ValueError(f"actions must have shape [samples, horizon, dims], got {frs.shape}")
    if not 0 <= int(gripper_index) < frs.shape[-1]:
        raise ValueError(
            f"gripper_index {gripper_index} is outside action width {frs.shape[-1]}"
        )
    if gates.shape != (frs.shape[0],):
        raise ValueError(
            f"gate_weights must have shape {(frs.shape[0],)}, got {gates.shape}"
        )
    if not all(np.isfinite(value).all() for value in (frs, gt, vla, gates)):
        raise ValueError("actions and gate weights must be finite")
    return frs, gt, vla, gates


def _per_sample_mse(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    return np.mean(np.square(first - second), axis=(1, 2), dtype=np.float64)


def _summary(
    frs: np.ndarray,
    gt: np.ndarray,
    vla: np.ndarray,
    gates: np.ndarray,
    *,
    low_gate_threshold: float,
    high_gate_threshold: float,
    low_gate_safety_margin: float,
    rank_margin: float,
    repair_margin: float,
) -> dict[str, float | int]:
    low = gates <= float(low_gate_threshold)
    high = gates >= float(high_gate_threshold)
    mid = ~(low | high)
    if not np.any(low):
        raise ValueError("low-Gate region is empty")
    if not np.any(high):
        raise ValueError("high-Gate region is empty")

    mse_frs_gt = _per_sample_mse(frs, gt)
    mse_frs_vla = _per_sample_mse(frs, vla)
    mse_vla_gt = _per_sample_mse(vla, gt)
    low_safe = np.minimum(mse_frs_gt[low], mse_frs_vla[low]) <= float(
        low_gate_safety_margin
    )
    high_rank = (
        mse_frs_gt[high] + float(rank_margin) <= mse_frs_vla[high]
    )
    high_repair = (
        mse_frs_gt[high] + float(repair_margin) <= mse_vla_gt[high]
    )

    result: dict[str, float | int] = {
        "sample_count": int(len(gates)),
        "n_low": int(np.count_nonzero(low)),
        "n_mid": int(np.count_nonzero(mid)),
        "n_high": int(np.count_nonzero(high)),
        "mse_frs_gt": float(np.mean(mse_frs_gt)),
        "mse_frs_vla": float(np.mean(mse_frs_vla)),
        "mse_vla_gt": float(np.mean(mse_vla_gt)),
        "gt_gain": float(np.mean(mse_vla_gt - mse_frs_gt)),
        "relative_gt_error": float(
            np.mean(mse_frs_gt) / max(float(np.mean(mse_vla_gt)), 1e-8)
        ),
        "low_safe_frac": float(np.mean(low_safe)),
        "low_unsafe_frac": float(1.0 - np.mean(low_safe)),
        "high_gain": float(np.mean(mse_vla_gt[high] - mse_frs_gt[high])),
        "high_rank_satisfied_frac": float(np.mean(high_rank)),
        "high_repair_satisfied_frac": float(np.mean(high_repair)),
    }
    for region_name, mask in (("low", low), ("mid", mid), ("high", high)):
        if not np.any(mask):
            continue
        result[f"{region_name}_mse_frs_gt"] = float(np.mean(mse_frs_gt[mask]))
        result[f"{region_name}_mse_frs_vla"] = float(np.mean(mse_frs_vla[mask]))
        result[f"{region_name}_mse_vla_gt"] = float(np.mean(mse_vla_gt[mask]))
        result[f"{region_name}_gain"] = float(
            np.mean(mse_vla_gt[mask] - mse_frs_gt[mask])
        )
    return result


def deployment_aligned_single_hand_metrics(
    frs_actions: np.ndarray,
    gt_actions: np.ndarray,
    vla_actions: np.ndarray,
    gate_weights: np.ndarray,
    *,
    gripper_index: int,
    low_gate_threshold: float,
    high_gate_threshold: float,
    low_gate_safety_margin: float,
    rank_margin: float,
    repair_margin: float,
) -> Mapping[str, dict[str, float | int]]:
    """Return arm-only and exact runtime endpoint metrics.

    ``arm9`` removes the physical gripper dimension. ``runtime10`` first
    replaces raw FRS gripper output with the frozen VLA gripper, matching the
    deployment runtime, and then evaluates the complete physical action.
    """

    frs, gt, vla, gates = _validate_actions(
        frs_actions,
        gt_actions,
        vla_actions,
        gate_weights,
        gripper_index=gripper_index,
    )
    arm_frs = np.delete(frs, gripper_index, axis=-1)
    arm_gt = np.delete(gt, gripper_index, axis=-1)
    arm_vla = np.delete(vla, gripper_index, axis=-1)
    runtime_frs = np.array(frs, copy=True)
    runtime_frs[..., gripper_index] = vla[..., gripper_index]
    kwargs = {
        "low_gate_threshold": low_gate_threshold,
        "high_gate_threshold": high_gate_threshold,
        "low_gate_safety_margin": low_gate_safety_margin,
        "rank_margin": rank_margin,
        "repair_margin": repair_margin,
    }
    return {
        "arm9": _summary(arm_frs, arm_gt, arm_vla, gates, **kwargs),
        "runtime10": _summary(runtime_frs, gt, vla, gates, **kwargs),
    }
