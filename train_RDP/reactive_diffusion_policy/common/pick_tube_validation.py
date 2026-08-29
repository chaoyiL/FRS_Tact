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
    HIGH_ROTATION_DELTA_DEG,
    HIGH_TRANSLATION_DELTA_M,
    LOW_ROTATION_DELTA_DEG,
    LOW_TRANSLATION_DELTA_M,
)


_ARM_LAYOUT = {
    "left": (slice(0, 3), slice(3, 9), 9),
    "right": (slice(10, 13), slice(13, 19), 19),
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
    first = np.asarray(rotation_6d[:3], dtype=np.float64)
    second = np.asarray(rotation_6d[3:], dtype=np.float64)
    first_norm = np.linalg.norm(first)
    if not np.isfinite(first).all() or first_norm < 1e-12:
        first = np.asarray([1.0, 0.0, 0.0])
    else:
        first = first / first_norm
    second = second - np.dot(first, second) * first
    second_norm = np.linalg.norm(second)
    if not np.isfinite(second).all() or second_norm < 1e-12:
        axis = np.eye(3)[int(np.argmin(np.abs(first)))]
        second = axis - np.dot(first, axis) * first
        second_norm = np.linalg.norm(second)
    second = second / second_norm
    third = np.cross(first, second)
    return np.stack((first, second, third), axis=-2)


def _geodesic_degrees(first: np.ndarray, second: np.ndarray) -> float:
    relative = first.T @ second
    cosine = float(np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0))
    return math.degrees(math.acos(cosine))


def _mean_or_nan(values: Sequence[float]) -> float:
    return float(np.mean(values)) if values else float("nan")


def _p95_or_nan(values: Sequence[float]) -> float:
    return float(np.percentile(values, 95)) if values else float("nan")


def _p50_or_nan(values: Sequence[float]) -> float:
    return float(np.percentile(values, 50)) if values else float("nan")


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
) -> dict[str, float]:
    """Measure bimanual physical errors from unnormalized 20D actions."""
    target_array = _as_numpy(target).astype(np.float64, copy=False)
    prediction_array = _as_numpy(prediction).astype(np.float64, copy=False)
    idle_array = _as_numpy(idle_mask).astype(bool, copy=False)
    if target_array.shape != prediction_array.shape or target_array.shape[-1] != 20:
        raise ValueError("target and prediction must have matching [..., T, 20] shapes")
    if target_array.ndim == 2:
        target_array = target_array[None]
        prediction_array = prediction_array[None]
    if target_array.ndim != 3:
        raise ValueError("target and prediction must have shape [B, T, 20]")
    if idle_array.ndim == 2:
        idle_array = idle_array[None]
    if idle_array.shape != (*target_array.shape[:-1], 2):
        raise ValueError("idle_mask must have shape [B, T, 2]")
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

    integrated_translation: dict[str, list[float]] = {
        arm: [] for arm in _ARM_LAYOUT
    }
    integrated_rotation: dict[str, list[float]] = {
        arm: [] for arm in _ARM_LAYOUT
    }
    idle_step_translation: dict[str, list[float]] = {
        arm: [] for arm in _ARM_LAYOUT
    }
    idle_step_rotation: dict[str, list[float]] = {
        arm: [] for arm in _ARM_LAYOUT
    }
    active_translation: dict[str, list[float]] = {
        arm: [] for arm in _ARM_LAYOUT
    }
    active_rotation: dict[str, list[float]] = {
        arm: [] for arm in _ARM_LAYOUT
    }
    translation_bias = {
        phase: {arm: [] for arm in _ARM_LAYOUT} for phase in ("idle", "active")
    }
    gripper_error = {
        phase: {arm: [] for arm in _ARM_LAYOUT} for phase in ("idle", "active")
    }
    micro_target_count = 0
    micro_predicted_count = 0
    predicted_micro_count = 0
    true_predicted_micro_count = 0

    for arm_index, (arm, (position_slice, rotation_slice, gripper_index)) in enumerate(
        _ARM_LAYOUT.items()
    ):
        for batch_index in range(target_array.shape[0]):
            target_total_rotation = np.eye(3)
            prediction_total_rotation = np.eye(3)
            target_total_translation = np.zeros(3)
            prediction_total_translation = np.zeros(3)
            has_idle = False
            is_idle_rollout = bool(
                np.all(
                    valid_array[batch_index]
                    & idle_array[batch_index, :, arm_index]
                )
            )
            for time_index in range(target_array.shape[1]):
                if not valid_array[batch_index, time_index]:
                    continue
                target_position = target_array[batch_index, time_index, position_slice]
                predicted_position = prediction_array[
                    batch_index, time_index, position_slice
                ]
                target_rotation = _rotation_matrix(
                    target_array[batch_index, time_index, rotation_slice]
                )
                predicted_rotation = _rotation_matrix(
                    prediction_array[batch_index, time_index, rotation_slice]
                )
                translation_error = float(
                    np.linalg.norm(predicted_position - target_position)
                )
                translation_error_vector = (
                    predicted_position - target_position
                ) * 1000.0
                rotation_error = _geodesic_degrees(target_rotation, predicted_rotation)
                width_error = abs(
                    prediction_array[batch_index, time_index, gripper_index]
                    - target_array[batch_index, time_index, gripper_index]
                ) * 1000.0
                if idle_array[batch_index, time_index, arm_index]:
                    has_idle = True
                    target_total_translation += target_position
                    prediction_total_translation += predicted_position
                    target_total_rotation = target_total_rotation @ target_rotation
                    prediction_total_rotation = (
                        prediction_total_rotation @ predicted_rotation
                    )
                    idle_step_translation[arm].append(translation_error * 1000.0)
                    idle_step_rotation[arm].append(rotation_error)
                    translation_bias["idle"][arm].append(translation_error_vector)
                    gripper_error["idle"][arm].append(float(width_error))
                else:
                    active_translation[arm].append(translation_error * 1000.0)
                    active_rotation[arm].append(rotation_error)
                    translation_bias["active"][arm].append(translation_error_vector)
                    gripper_error["active"][arm].append(float(width_error))
                    target_motion_translation = float(np.linalg.norm(target_position))
                    target_motion_rotation = _geodesic_degrees(
                        np.eye(3), target_rotation
                    )
                    predicted_motion_translation = float(
                        np.linalg.norm(predicted_position)
                    )
                    predicted_motion_rotation = _geodesic_degrees(
                        np.eye(3), predicted_rotation
                    )
                    target_is_micro = (
                        (
                            target_motion_translation >= LOW_TRANSLATION_DELTA_M
                            or target_motion_rotation >= LOW_ROTATION_DELTA_DEG
                        )
                        and target_motion_translation <= HIGH_TRANSLATION_DELTA_M
                        and target_motion_rotation <= HIGH_ROTATION_DELTA_DEG
                    )
                    predicted_is_micro = (
                        predicted_motion_translation >= LOW_TRANSLATION_DELTA_M
                        or predicted_motion_rotation >= LOW_ROTATION_DELTA_DEG
                    )
                    if target_is_micro:
                        micro_target_count += 1
                        micro_predicted_count += int(predicted_is_micro)
                    if predicted_is_micro:
                        predicted_micro_count += 1
                        true_predicted_micro_count += int(target_is_micro)
            if has_idle and is_idle_rollout:
                integrated_translation[arm].append(
                    float(
                        np.linalg.norm(
                            prediction_total_translation - target_total_translation
                        )
                        * 1000.0
                    )
                )
                integrated_rotation[arm].append(
                    _geodesic_degrees(
                        target_total_rotation, prediction_total_rotation
                    )
                )

    all_integrated_translation = sum(integrated_translation.values(), [])
    all_integrated_rotation = sum(integrated_rotation.values(), [])
    all_idle_step_translation = sum(idle_step_translation.values(), [])
    all_idle_step_rotation = sum(idle_step_rotation.values(), [])
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
    for arm in _ARM_LAYOUT:
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
            biases = translation_bias[phase][arm]
            mean_bias = (
                np.mean(np.stack(biases), axis=0)
                if biases
                else np.full(3, np.nan)
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
                    f"val_{phase}_{arm}_gripper_mae_mm": _mean_or_nan(
                        gripper_error[phase][arm]
                    ),
                }
            )
    metrics["val_active_translation_mae_mm"] = _mean_or_nan(
        sum(active_translation.values(), [])
    )
    metrics["val_active_rotation_mae_deg"] = _mean_or_nan(
        sum(active_rotation.values(), [])
    )
    for phase, translations, rotations in (
        ("idle", idle_step_translation, idle_step_rotation),
        ("active", active_translation, active_rotation),
    ):
        all_biases = sum(translation_bias[phase].values(), [])
        mean_bias = (
            np.mean(np.stack(all_biases), axis=0)
            if all_biases
            else np.full(3, np.nan)
        )
        all_translations = sum(translations.values(), [])
        all_rotations = sum(rotations.values(), [])
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
                f"val_{phase}_gripper_mae_mm": _mean_or_nan(
                    sum(gripper_error[phase].values(), [])
                ),
            }
        )
    return metrics


def compute_contiguous_300_step_drift(
    target,
    prediction,
    idle_mask,
    *,
    valid_mask=None,
) -> dict[str, float]:
    """Measure drift from genuine contiguous 300-step action trajectories."""
    target_array = _as_numpy(target)
    if target_array.ndim < 2 or target_array.shape[-2] < 300:
        raise ValueError("300 contiguous action steps are required for drift metrics")
    metrics = compute_idle_rollout_metrics(
        target,
        prediction,
        idle_mask,
        horizon=300,
        valid_mask=valid_mask,
    )
    result = {
        "val_idle_translation_300_mm": metrics["val_idle_translation_29_mm"],
        "val_idle_rotation_300_deg": metrics["val_idle_rotation_29_deg"],
    }
    for arm in _ARM_LAYOUT:
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
    min_micro_motion_recall: float = 0.95,
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
    required = {
        "translation_mm": "val_active_left_translation_mae_mm",
        "rotation_deg": "val_active_left_rotation_mae_deg",
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
