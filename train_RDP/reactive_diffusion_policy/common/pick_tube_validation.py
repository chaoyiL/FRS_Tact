"""Episode splitting and physical validation metrics for pick-tube v2."""

from __future__ import annotations

from contextlib import contextmanager
import json
import math
from pathlib import Path
import random
from collections.abc import Mapping, Sequence

import einops
import numpy as np
import torch
from omegaconf import OmegaConf

from reactive_diffusion_policy.common.artifact_manifest import stable_json_digest
from reactive_diffusion_policy.common.pick_tube_action_contract import (
    DUAL_ARM_PROFILE,
    HIGH_ROTATION_DELTA_DEG,
    HIGH_TRANSLATION_DELTA_M,
    LOW_ROTATION_DELTA_DEG,
    LOW_TRANSLATION_DELTA_M,
    resolve_state_action_profile,
)


def _arm_layout_for(
    action_dim: int,
    state_action_profile: str | None,
) -> dict[str, tuple[slice, slice, int]]:
    if state_action_profile is None:
        if action_dim != 20:
            raise ValueError(
                "10D validation requires state_action_profile=single-right-arm-7x10"
            )
        state_action_profile = DUAL_ARM_PROFILE
    profile = resolve_state_action_profile(state_action_profile)
    if profile.action_dim != action_dim:
        raise ValueError(
            f"{profile.name} requires {profile.action_dim}D actions, got {action_dim}D"
        )
    return {
        arm_name: (
            slice(arm_index * 10, arm_index * 10 + 3),
            slice(arm_index * 10 + 3, arm_index * 10 + 9),
            arm_index * 10 + 9,
        )
        for arm_index, arm_name in enumerate(profile.controlled_arms)
    }


def build_episode_split_manifest(
    episode_sources: Sequence[int] | np.ndarray,
    *,
    val_ratio: float,
    seed: int,
) -> dict:
    """Return a deterministic source-stratified episode split."""
    sources = np.asarray(episode_sources)
    if sources.ndim != 1 or sources.size == 0:
        raise ValueError("episode_sources must be a non-empty one-dimensional array")
    if not 0.0 <= float(val_ratio) < 1.0:
        raise ValueError("val_ratio must be in [0, 1)")

    validation_ids: list[int] = []
    rng = np.random.default_rng(int(seed))
    if val_ratio > 0:
        for source in sorted(np.unique(sources).tolist()):
            source_ids = np.flatnonzero(sources == source)
            if source_ids.size < 2:
                raise ValueError(
                    f"source {source!r} needs at least two episodes for a held-out split"
                )
            requested = max(1, int(round(source_ids.size * float(val_ratio))))
            count = min(requested, source_ids.size - 1)
            validation_ids.extend(
                int(value)
                for value in np.sort(rng.choice(source_ids, size=count, replace=False))
            )

    validation_ids = sorted(validation_ids)
    validation_set = set(validation_ids)
    train_ids = [
        episode_id
        for episode_id in range(int(sources.size))
        if episode_id not in validation_set
    ]
    identity = {
        "seed": int(seed),
        "val_ratio": float(val_ratio),
        "episode_sources": sources.tolist(),
        "train_episode_ids": train_ids,
        "validation_episode_ids": validation_ids,
    }
    return {**identity, "split_digest": stable_json_digest(identity)}


def _as_numpy(value) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        tensor = value.detach().cpu()
        if tensor.dtype == torch.bfloat16:
            tensor = tensor.float()
        return tensor.numpy()
    return np.asarray(value)


def _rotation_matrix(rotation_6d: np.ndarray) -> np.ndarray:
    return _rotation_matrices(rotation_6d)


def _rotation_matrices(rotation_6d: np.ndarray) -> np.ndarray:
    """Convert one or more row-major 6D rotations with legacy fallbacks."""
    rotation_array = np.asarray(rotation_6d, dtype=np.float64)
    if rotation_array.ndim < 1 or rotation_array.shape[-1] != 6:
        raise ValueError("rotation_6d must have shape [..., 6]")

    first_raw = rotation_array[..., :3]
    first_norm = np.linalg.norm(first_raw, axis=-1, keepdims=True)
    first_valid = np.isfinite(first_raw).all(axis=-1) & (
        first_norm[..., 0] >= 1e-12
    )
    with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
        normalized_first = first_raw / np.where(
            first_valid[..., None], first_norm, 1.0
        )
    first = np.where(
        first_valid[..., None],
        normalized_first,
        np.asarray([1.0, 0.0, 0.0]),
    )

    second_raw = rotation_array[..., 3:]
    with np.errstate(invalid="ignore", over="ignore"):
        second = second_raw - np.sum(first * second_raw, axis=-1, keepdims=True) * first
    second_norm = np.linalg.norm(second, axis=-1, keepdims=True)
    second_valid = np.isfinite(second).all(axis=-1) & (
        second_norm[..., 0] >= 1e-12
    )
    fallback_axis = np.eye(3)[np.argmin(np.abs(first), axis=-1)]
    fallback_second = (
        fallback_axis
        - np.sum(first * fallback_axis, axis=-1, keepdims=True) * first
    )
    fallback_norm = np.linalg.norm(fallback_second, axis=-1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
        normalized_second = second / np.where(
            second_valid[..., None], second_norm, 1.0
        )
    second = np.where(
        second_valid[..., None],
        normalized_second,
        fallback_second / fallback_norm,
    )
    third = np.cross(first, second)
    return np.stack((first, second, third), axis=-2)


def _geodesic_degrees(first: np.ndarray, second: np.ndarray) -> float:
    return float(_geodesic_degrees_batch(first, second))


def _geodesic_degrees_batch(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    relative = np.matmul(np.swapaxes(first, -1, -2), second)
    cosine = np.clip(
        (np.trace(relative, axis1=-2, axis2=-1) - 1.0) / 2.0,
        -1.0,
        1.0,
    )
    return np.degrees(np.arccos(cosine))


def _mean_or_nan(values: Sequence[float]) -> float:
    return float(np.mean(values)) if len(values) else float("nan")


def _p95_or_nan(values: Sequence[float]) -> float:
    return float(np.percentile(values, 95)) if len(values) else float("nan")


def _p50_or_nan(values: Sequence[float]) -> float:
    return float(np.percentile(values, 50)) if len(values) else float("nan")


@contextmanager
def preserve_global_rng_state(seed: int):
    """Run a deterministic validation sample without advancing global RNGs."""
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.random.get_rng_state()
    cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    try:
        random.seed(int(seed))
        np.random.seed(int(seed) % (2**32))
        torch.manual_seed(int(seed))
        yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.random.set_rng_state(torch_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)


def compute_idle_rollout_metrics(
    target,
    prediction,
    idle_mask,
    horizon: int = 29,
    *,
    valid_mask=None,
    state_action_profile: str | None = None,
) -> dict[str, float]:
    """Measure physical errors from unnormalized 10D or 20D actions."""
    target_array = _as_numpy(target).astype(np.float64, copy=False)
    prediction_array = _as_numpy(prediction).astype(np.float64, copy=False)
    idle_array = _as_numpy(idle_mask).astype(bool, copy=False)
    if target_array.shape != prediction_array.shape:
        raise ValueError("target and prediction must have matching shapes")
    if target_array.ndim == 2:
        target_array = target_array[None]
        prediction_array = prediction_array[None]
    if target_array.ndim != 3:
        raise ValueError("target and prediction must have shape [B, T, A]")
    arm_layout = _arm_layout_for(
        target_array.shape[-1],
        state_action_profile,
    )
    if idle_array.ndim == 2:
        idle_array = idle_array[None]
    if idle_array.shape != (*target_array.shape[:-1], len(arm_layout)):
        raise ValueError(
            f"idle_mask must have shape [B, T, {len(arm_layout)}]"
        )
    if horizon < 1 or horizon > target_array.shape[-2]:
        raise ValueError("horizon must be positive and no longer than the action sequence")

    if valid_mask is None:
        valid_array = np.ones(target_array.shape[:-1], dtype=bool)
    else:
        valid_array = _as_numpy(valid_mask).astype(bool, copy=False)
        if valid_array.ndim == 1:
            valid_array = valid_array[None]
        if valid_array.shape != target_array.shape[:-1]:
            raise ValueError("valid_mask must have shape [B, T]")

    start = target_array.shape[-2] - int(horizon)
    target_array = target_array[:, start:]
    prediction_array = prediction_array[:, start:]
    idle_array = idle_array[:, start:]
    valid_array = valid_array[:, start:]

    idle_counts = {}
    active_counts = {}
    integrated_counts = {}
    for arm_index, arm in enumerate(arm_layout):
        idle_valid = valid_array & idle_array[..., arm_index]
        idle_counts[arm] = int(np.count_nonzero(idle_valid))
        active_counts[arm] = int(np.count_nonzero(valid_array & ~idle_array[..., arm_index]))
        integrated_counts[arm] = int(np.count_nonzero(np.all(idle_valid, axis=1)))

    def allocate_metric_values(counts):
        values = np.empty(sum(counts.values()), dtype=np.float64)
        views = {}
        cursors = {}
        offset = 0
        for arm in arm_layout:
            count = counts[arm]
            views[arm] = values[offset : offset + count]
            cursors[arm] = 0
            offset += count
        return values, views, cursors

    (
        all_integrated_translation,
        integrated_translation,
        integrated_cursor,
    ) = allocate_metric_values(integrated_counts)
    all_integrated_rotation, integrated_rotation, _ = allocate_metric_values(
        integrated_counts
    )
    (
        all_idle_step_translation,
        idle_step_translation,
        idle_cursor,
    ) = allocate_metric_values(idle_counts)
    all_idle_step_rotation, idle_step_rotation, _ = allocate_metric_values(
        idle_counts
    )
    (
        all_active_translation,
        active_translation,
        active_cursor,
    ) = allocate_metric_values(active_counts)
    all_active_rotation, active_rotation, _ = allocate_metric_values(active_counts)

    translation_bias_sum = {
        phase: {arm: np.zeros(3, dtype=np.float64) for arm in arm_layout}
        for phase in ("idle", "active")
    }
    gripper_error_sum = {
        phase: {arm: 0.0 for arm in arm_layout}
        for phase in ("idle", "active")
    }
    micro_target_count = 0
    micro_predicted_count = 0
    predicted_micro_count = 0
    true_predicted_micro_count = 0

    metric_batch_size = 8192
    identity_rotation = np.eye(3)
    for arm_index, (arm, (position_slice, rotation_slice, gripper_index)) in enumerate(
        arm_layout.items()
    ):
        for batch_start in range(0, target_array.shape[0], metric_batch_size):
            batch_end = min(batch_start + metric_batch_size, target_array.shape[0])
            target_chunk = target_array[batch_start:batch_end]
            prediction_chunk = prediction_array[batch_start:batch_end]
            valid_chunk = valid_array[batch_start:batch_end]
            idle_chunk = idle_array[batch_start:batch_end, :, arm_index]
            idle_valid = valid_chunk & idle_chunk
            active_valid = valid_chunk & ~idle_chunk

            target_position = target_chunk[..., position_slice]
            predicted_position = prediction_chunk[..., position_slice]
            translation_error_vector = (
                predicted_position - target_position
            ) * 1000.0
            translation_error = np.linalg.norm(
                translation_error_vector, axis=-1
            )
            target_rotation = _rotation_matrices(target_chunk[..., rotation_slice])
            predicted_rotation = _rotation_matrices(
                prediction_chunk[..., rotation_slice]
            )
            rotation_error = _geodesic_degrees_batch(
                target_rotation, predicted_rotation
            )
            width_error = np.abs(
                prediction_chunk[..., gripper_index]
                - target_chunk[..., gripper_index]
            ) * 1000.0

            idle_count = int(np.count_nonzero(idle_valid))
            idle_start = idle_cursor[arm]
            idle_end = idle_start + idle_count
            idle_step_translation[arm][idle_start:idle_end] = translation_error[
                idle_valid
            ]
            idle_step_rotation[arm][idle_start:idle_end] = rotation_error[idle_valid]
            idle_cursor[arm] = idle_end

            active_count = int(np.count_nonzero(active_valid))
            active_start = active_cursor[arm]
            active_end = active_start + active_count
            active_translation[arm][active_start:active_end] = translation_error[
                active_valid
            ]
            active_rotation[arm][active_start:active_end] = rotation_error[
                active_valid
            ]
            active_cursor[arm] = active_end

            translation_bias_sum["idle"][arm] += np.sum(
                translation_error_vector[idle_valid], axis=0
            )
            translation_bias_sum["active"][arm] += np.sum(
                translation_error_vector[active_valid], axis=0
            )
            gripper_error_sum["idle"][arm] += float(
                np.sum(width_error[idle_valid])
            )
            gripper_error_sum["active"][arm] += float(
                np.sum(width_error[active_valid])
            )

            target_motion_translation = np.linalg.norm(target_position, axis=-1)
            target_motion_rotation = _geodesic_degrees_batch(
                identity_rotation, target_rotation
            )
            predicted_motion_translation = np.linalg.norm(
                predicted_position, axis=-1
            )
            predicted_motion_rotation = _geodesic_degrees_batch(
                identity_rotation, predicted_rotation
            )
            target_is_micro = (
                (
                    (target_motion_translation >= LOW_TRANSLATION_DELTA_M)
                    | (target_motion_rotation >= LOW_ROTATION_DELTA_DEG)
                )
                & (target_motion_translation <= HIGH_TRANSLATION_DELTA_M)
                & (target_motion_rotation <= HIGH_ROTATION_DELTA_DEG)
                & active_valid
            )
            predicted_is_micro = (
                (
                    (predicted_motion_translation >= LOW_TRANSLATION_DELTA_M)
                    | (predicted_motion_rotation >= LOW_ROTATION_DELTA_DEG)
                )
                & active_valid
            )
            micro_target_count += int(np.count_nonzero(target_is_micro))
            micro_predicted_count += int(
                np.count_nonzero(target_is_micro & predicted_is_micro)
            )
            predicted_micro_count += int(np.count_nonzero(predicted_is_micro))
            true_predicted_micro_count += int(
                np.count_nonzero(predicted_is_micro & target_is_micro)
            )

            integrated_mask = np.all(idle_valid, axis=1)
            if np.any(integrated_mask):
                integrated_count = int(np.count_nonzero(integrated_mask))
                integrated_start = integrated_cursor[arm]
                integrated_end = integrated_start + integrated_count
                integrated_translation[arm][integrated_start:integrated_end] = (
                    np.linalg.norm(
                        np.sum(predicted_position[integrated_mask], axis=1)
                        - np.sum(target_position[integrated_mask], axis=1),
                        axis=-1,
                    )
                    * 1000.0
                )
                target_total_rotation = np.broadcast_to(
                    identity_rotation,
                    (integrated_count, 3, 3),
                ).copy()
                prediction_total_rotation = target_total_rotation.copy()
                for time_index in range(target_array.shape[1]):
                    target_total_rotation = np.matmul(
                        target_total_rotation,
                        target_rotation[integrated_mask, time_index],
                    )
                    prediction_total_rotation = np.matmul(
                        prediction_total_rotation,
                        predicted_rotation[integrated_mask, time_index],
                    )
                integrated_rotation[arm][integrated_start:integrated_end] = (
                    _geodesic_degrees_batch(
                        target_total_rotation, prediction_total_rotation
                    )
                )
                integrated_cursor[arm] = integrated_end

    for arm in arm_layout:
        if idle_cursor[arm] != idle_counts[arm]:
            raise RuntimeError("idle metric preallocation count mismatch")
        if active_cursor[arm] != active_counts[arm]:
            raise RuntimeError("active metric preallocation count mismatch")
        if integrated_cursor[arm] != integrated_counts[arm]:
            raise RuntimeError("integrated metric preallocation count mismatch")

    def mean_from_sum_or_nan(total, count):
        if not count:
            if np.ndim(total):
                return np.full(np.shape(total), np.nan)
            return float("nan")
        return np.asarray(total) / count

    metrics: dict[str, float] = {
        "val_idle_translation_29_mm": _mean_or_nan(all_integrated_translation),
        "val_idle_rotation_29_deg": _mean_or_nan(all_integrated_rotation),
        "val_idle_translation_step_p95_mm": _p95_or_nan(
            all_idle_step_translation
        ),
        "val_idle_rotation_step_p95_deg": _p95_or_nan(all_idle_step_rotation),
        "val_micro_motion_recall": (
            float(micro_predicted_count / micro_target_count)
            if micro_target_count
            else float("nan")
        ),
        "val_micro_motion_precision": (
            float(true_predicted_micro_count / predicted_micro_count)
            if predicted_micro_count
            else float("nan")
        ),
    }
    metrics["val_idle_translation_p95_mm"] = metrics[
        "val_idle_translation_step_p95_mm"
    ]
    metrics["val_idle_rotation_p95_deg"] = metrics[
        "val_idle_rotation_step_p95_deg"
    ]
    for arm in arm_layout:
        metrics.update(
            {
                f"val_idle_{arm}_translation_29_mm": _mean_or_nan(
                    integrated_translation[arm]
                ),
                f"val_idle_{arm}_rotation_29_deg": _mean_or_nan(
                    integrated_rotation[arm]
                ),
                f"val_idle_{arm}_translation_step_p95_mm": _p95_or_nan(
                    idle_step_translation[arm]
                ),
                f"val_idle_{arm}_rotation_step_p95_deg": _p95_or_nan(
                    idle_step_rotation[arm]
                ),
                f"val_active_{arm}_translation_mae_mm": _mean_or_nan(
                    active_translation[arm]
                ),
                f"val_active_{arm}_rotation_mae_deg": _mean_or_nan(
                    active_rotation[arm]
                ),
            }
        )
        for phase, translations, rotations in (
            ("idle", idle_step_translation, idle_step_rotation),
            ("active", active_translation, active_rotation),
        ):
            phase_count = (
                idle_counts[arm] if phase == "idle" else active_counts[arm]
            )
            mean_bias = mean_from_sum_or_nan(
                translation_bias_sum[phase][arm], phase_count
            )
            metrics.update(
                {
                    f"val_{phase}_{arm}_translation_bias_x_mm": float(
                        mean_bias[0]
                    ),
                    f"val_{phase}_{arm}_translation_bias_y_mm": float(
                        mean_bias[1]
                    ),
                    f"val_{phase}_{arm}_translation_bias_z_mm": float(
                        mean_bias[2]
                    ),
                    f"val_{phase}_{arm}_translation_mae_mm": _mean_or_nan(
                        translations[arm]
                    ),
                    f"val_{phase}_{arm}_translation_p50_mm": _p50_or_nan(
                        translations[arm]
                    ),
                    f"val_{phase}_{arm}_translation_p95_mm": _p95_or_nan(
                        translations[arm]
                    ),
                    f"val_{phase}_{arm}_rotation_mae_deg": _mean_or_nan(
                        rotations[arm]
                    ),
                    f"val_{phase}_{arm}_rotation_p50_deg": _p50_or_nan(
                        rotations[arm]
                    ),
                    f"val_{phase}_{arm}_rotation_p95_deg": _p95_or_nan(
                        rotations[arm]
                    ),
                    f"val_{phase}_{arm}_gripper_mae_mm": float(
                        mean_from_sum_or_nan(
                            gripper_error_sum[phase][arm], phase_count
                        )
                    ),
                }
            )
    metrics["val_active_translation_mae_mm"] = _mean_or_nan(
        all_active_translation
    )
    metrics["val_active_rotation_mae_deg"] = _mean_or_nan(
        all_active_rotation
    )
    for phase, all_translations, all_rotations, counts in (
        ("idle", all_idle_step_translation, all_idle_step_rotation, idle_counts),
        ("active", all_active_translation, all_active_rotation, active_counts),
    ):
        phase_count = sum(counts.values())
        mean_bias = mean_from_sum_or_nan(
            sum(
                (translation_bias_sum[phase][arm] for arm in arm_layout),
                np.zeros(3, dtype=np.float64),
            ),
            phase_count,
        )
        metrics.update(
            {
                f"val_{phase}_translation_bias_x_mm": float(mean_bias[0]),
                f"val_{phase}_translation_bias_y_mm": float(mean_bias[1]),
                f"val_{phase}_translation_bias_z_mm": float(mean_bias[2]),
                f"val_{phase}_translation_mae_mm": _mean_or_nan(all_translations),
                f"val_{phase}_translation_p50_mm": _p50_or_nan(all_translations),
                f"val_{phase}_translation_p95_mm": _p95_or_nan(all_translations),
                f"val_{phase}_rotation_mae_deg": _mean_or_nan(all_rotations),
                f"val_{phase}_rotation_p50_deg": _p50_or_nan(all_rotations),
                f"val_{phase}_rotation_p95_deg": _p95_or_nan(all_rotations),
                f"val_{phase}_gripper_mae_mm": float(
                    mean_from_sum_or_nan(
                        sum(gripper_error_sum[phase].values()), phase_count
                    )
                ),
            }
        )
    return metrics


def compute_deployment_window_metrics(
    target,
    prediction,
    idle_mask,
    *,
    phase_start,
    phase_count,
    valid_mask=None,
    state_action_profile: str | None = None,
) -> dict[str, float]:
    """Measure idle metrics over the action phase executed by deployment."""
    if phase_start < 0:
        raise ValueError("phase_start must be non-negative")
    if phase_count < 1:
        raise ValueError("phase_count must be positive")

    target_array = _as_numpy(target)
    if target_array.ndim not in (2, 3):
        raise ValueError("target must have shape [T, A] or [B, T, A]")
    horizon = target_array.shape[-2]
    phase_end = phase_start + phase_count
    if phase_end > horizon:
        raise ValueError("deployment phase must fit within the action sequence")

    window = slice(phase_start, phase_end)
    metrics = compute_idle_rollout_metrics(
        target[..., window, :],
        prediction[..., window, :],
        idle_mask[..., window, :],
        horizon=phase_count,
        valid_mask=(None if valid_mask is None else valid_mask[..., window]),
        state_action_profile=state_action_profile,
    )
    return {
        name.replace("val_", "val_deploy_", 1).replace("_29_", "_window_"): value
        for name, value in metrics.items()
    }


def build_canonical_noop_actions(actions: torch.Tensor) -> torch.Tensor:
    """Return relative-action no-ops while retaining each gripper target."""
    if not isinstance(actions, torch.Tensor):
        raise TypeError("actions must be a torch.Tensor")
    if actions.ndim < 1 or actions.shape[-1] not in (10, 20):
        raise ValueError("actions must use a contiguous 10D or 20D layout")
    if not actions.is_contiguous():
        raise ValueError("actions must use a contiguous 10D or 20D layout")

    noops = actions.clone()
    rotation = actions.new_tensor([1, 0, 0, 0, 1, 0])
    for arm_start in range(0, actions.shape[-1], 10):
        noops[..., arm_start : arm_start + 3] = 0
        noops[..., arm_start + 3 : arm_start + 9] = rotation
    return noops


def compute_contiguous_300_step_drift(
    target,
    prediction,
    idle_mask,
    *,
    valid_mask=None,
    state_action_profile: str | None = None,
) -> dict[str, float]:
    """Measure drift from genuine contiguous 300-step action trajectories."""
    target_array = _as_numpy(target)
    arm_layout = _arm_layout_for(target_array.shape[-1], state_action_profile)
    if target_array.ndim < 2 or target_array.shape[-2] < 300:
        raise ValueError("300 contiguous action steps are required for drift metrics")
    metrics = compute_idle_rollout_metrics(
        target,
        prediction,
        idle_mask,
        horizon=300,
        valid_mask=valid_mask,
        state_action_profile=state_action_profile,
    )
    result = {
        "val_idle_translation_300_mm": metrics["val_idle_translation_29_mm"],
        "val_idle_rotation_300_deg": metrics["val_idle_rotation_29_deg"],
    }
    for arm in arm_layout:
        result[f"val_idle_{arm}_translation_300_mm"] = metrics[
            f"val_idle_{arm}_translation_29_mm"
        ]
        result[f"val_idle_{arm}_rotation_300_deg"] = metrics[
            f"val_idle_{arm}_rotation_29_deg"
        ]
    return result


def evaluate_checkpoint_feasibility(
    *,
    idle_translation_29_mm: float,
    idle_rotation_29_deg: float,
    idle_translation_p95_mm: float,
    idle_rotation_p95_deg: float,
    active_translation_mm: float,
    active_translation_baseline_mm: float | None,
    active_rotation_deg: float,
    active_rotation_baseline_deg: float | None,
    micro_motion_recall: float,
    max_active_degradation: float = 0.05,
    min_micro_motion_recall: float = 0.40,
) -> dict[str, float | bool]:
    """Apply hard gates and calculate the idle score used by top-k selection."""
    score = float(idle_translation_29_mm) + float(idle_rotation_29_deg) / 0.5
    usable_translation_baseline = (
        active_translation_baseline_mm is not None
        and math.isfinite(float(active_translation_baseline_mm))
        and float(active_translation_baseline_mm) > 0
    )
    usable_rotation_baseline = (
        active_rotation_baseline_deg is not None
        and math.isfinite(float(active_rotation_baseline_deg))
        and float(active_rotation_baseline_deg) > 0
    )
    translation_degradation = (
        float(active_translation_mm) / float(active_translation_baseline_mm) - 1.0
        if usable_translation_baseline
        else float("inf")
    )
    rotation_degradation = (
        float(active_rotation_deg) / float(active_rotation_baseline_deg) - 1.0
        if usable_rotation_baseline
        else float("inf")
    )
    feasible = bool(
        usable_translation_baseline
        and usable_rotation_baseline
        and math.isfinite(score)
        and math.isfinite(float(micro_motion_recall))
        and translation_degradation <= float(max_active_degradation) + 1e-12
        and rotation_degradation <= float(max_active_degradation) + 1e-12
        and float(micro_motion_recall) >= float(min_micro_motion_recall)
    )
    deployable = bool(
        feasible
        and float(idle_translation_29_mm) < 1.0
        and float(idle_rotation_29_deg) < 0.5
        and float(idle_translation_p95_mm) < 0.05
        and float(idle_rotation_p95_deg) < 0.03
    )
    return {
        "val_active_translation_degradation": translation_degradation,
        "val_active_rotation_degradation": rotation_degradation,
        "val_micro_motion_recall": float(micro_motion_recall),
        "val_idle_score": score,
        "val_checkpoint_feasible": feasible,
        "val_deployable": deployable,
    }


def _select(config, key: str):
    if OmegaConf.is_config(config):
        return OmegaConf.select(config, key)
    current = config
    for part in key.split("."):
        if not isinstance(current, Mapping) or part not in current:
            return None
        current = current[part]
    return current


def load_active_metric_baselines(config) -> dict[str, float] | None:
    """Load optional frozen-v1 active metrics used by deployment gates."""
    baseline_path = _select(config, "validation.baseline_json")
    if not baseline_path:
        return None
    path = Path(str(baseline_path)).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"validation baseline_json does not exist: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"validation baseline_json is invalid JSON: {path}") from error
    if not isinstance(value, Mapping):
        raise ValueError("validation baseline_json must contain a JSON object")
    controlled_arms = _select(config, "task.controlled_arms")
    active_arm = str(controlled_arms[0]) if controlled_arms else "left"
    required = {
        "translation_mm": f"val_active_{active_arm}_translation_mae_mm",
        "rotation_deg": f"val_active_{active_arm}_rotation_mae_deg",
    }
    baselines: dict[str, float] = {}
    for output_key, input_key in required.items():
        raw = value.get(input_key)
        if isinstance(raw, bool) or not isinstance(raw, (int, float)):
            raise ValueError(
                f"validation baseline_json field {input_key!r} must be a number"
            )
        baseline = float(raw)
        if not math.isfinite(baseline) or baseline <= 0:
            raise ValueError(
                f"validation baseline_json field {input_key!r} must be finite and positive"
            )
        baselines[output_key] = baseline
    return baselines


def _is_finite_positive_baseline(value) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(float(value))
        and float(value) > 0
    )


def resolve_active_metric_baselines(
    *,
    external_baselines: dict[str, float] | None,
    auto_translation_baseline_mm,
    auto_rotation_baseline_deg,
    active_translation_mm,
    active_rotation_deg,
    epoch,
    auto_baseline_epoch=None,
) -> dict[str, float | int | str | bool | None]:
    """Prefer external baselines, otherwise freeze the first valid active metrics."""
    if external_baselines is not None:
        return {
            "translation_mm": external_baselines["translation_mm"],
            "rotation_deg": external_baselines["rotation_deg"],
            "source": "external",
            "epoch": None,
            "calibrated": False,
        }
    if (
        _is_finite_positive_baseline(auto_translation_baseline_mm)
        and _is_finite_positive_baseline(auto_rotation_baseline_deg)
    ):
        return {
            "translation_mm": float(auto_translation_baseline_mm),
            "rotation_deg": float(auto_rotation_baseline_deg),
            "source": "auto",
            "epoch": auto_baseline_epoch,
            "calibrated": False,
        }
    if (
        _is_finite_positive_baseline(active_translation_mm)
        and _is_finite_positive_baseline(active_rotation_deg)
    ):
        return {
            "translation_mm": float(active_translation_mm),
            "rotation_deg": float(active_rotation_deg),
            "source": "auto",
            "epoch": int(epoch),
            "calibrated": True,
        }
    return {
        "translation_mm": None,
        "rotation_deg": None,
        "source": None,
        "epoch": None,
        "calibrated": False,
    }


def _action_contract_identity(config) -> tuple[object, object]:
    version = _select(config, "task.action_representation_version")
    contract = _select(config, "task.action_contract")
    if version is None:
        version = _select(config, "artifacts.action_representation_version")
    if contract is None:
        contract = _select(config, "artifacts.action_contract")
    return version, contract


def validate_resume_action_contract(current_config, checkpoint_config) -> None:
    """Reject a checkpoint whose action contract differs from this run."""
    current = _action_contract_identity(current_config)
    checkpoint = _action_contract_identity(checkpoint_config)
    if current == (None, None) and checkpoint == (None, None):
        return
    if current != checkpoint:
        raise ValueError(
            "cannot resume across action-contract version boundaries: "
            f"current={current!r}, checkpoint={checkpoint!r}"
        )


def reconstruct_at_actions(model, batch: Mapping[str, torch.Tensor]) -> torch.Tensor:
    """Decode deterministic posterior-mode AT reconstructions in physical units."""
    normalized = model.normalizer["action"].normalize(batch["action"])
    state_representation = model.encoder(model.preprocess(normalized / model.act_scale))
    if model.use_vq:
        latent, _, _ = model.quant_state_with_vq(state_representation)
    else:
        latent, _ = model.quant_state_without_vq(
            state_representation,
            sample=False,
        )
        latent = model.postprocess_quant_state_without_vq(latent)
    if model.use_rnn_decoder:
        temporal = model.get_temporal_cond(batch["extended_obs"]).to(model.device)
        decoded = model.decoder(latent, temporal)
    else:
        decoded = model.decoder(latent)
    normalized_prediction = einops.rearrange(
        decoded,
        "N (T A) -> N T A",
        T=model.input_dim_h,
        A=model.input_dim_w,
    ) * model.act_scale
    return model.normalizer["action"].unnormalize(normalized_prediction)
