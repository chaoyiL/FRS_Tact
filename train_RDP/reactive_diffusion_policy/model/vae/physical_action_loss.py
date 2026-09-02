import math
from collections.abc import Mapping

import torch
import torch.nn.functional as F

from reactive_diffusion_policy.common.pick_tube_action_contract import (
    HIGH_ROTATION_DELTA_DEG,
    HIGH_TRANSLATION_DELTA_M,
    LOW_ROTATION_DELTA_DEG,
    LOW_TRANSLATION_DELTA_M,
)


def action_arm_slices(action_dim: int):
    if action_dim not in (10, 20):
        raise ValueError(
            f"physical_v2 requires a 10D single-arm or 20D bimanual action, got {action_dim}D"
        )
    return tuple(
        (slice(offset, offset + 3), slice(offset + 3, offset + 9), offset + 9)
        for offset in range(0, action_dim, 10)
    )


_DEFAULT_WEIGHTS = {
    "position_scale": 1e-3,
    "rotation_scale": math.radians(1.0),
    "gripper_scale": 5e-3,
    "idle_position_scale": 1e-4,
    "idle_rotation_scale": math.radians(0.05),
    "position_weight": 1.0,
    "rotation_weight": 1.0,
    "gripper_weight": 1.0,
    "idle_weight": 1.0,
    "degenerate_weight": 1.0,
    "rot6_aux_weight": 0.0,
    "micro_motion_weight": 0.0,
}


def _rotation_compute_dtype(*values: torch.Tensor) -> torch.dtype:
    dtype = values[0].dtype
    for value in values[1:]:
        dtype = torch.promote_types(dtype, value.dtype)
    return torch.float32 if dtype in (torch.float16, torch.bfloat16) else dtype


def project_rotation_6d(
        rotation_6d: torch.Tensor,
        eps: float = 1e-6) -> tuple[torch.Tensor, torch.Tensor]:
    """Project two 3D basis vectors to SO(3) with stable Gram-Schmidt."""
    if rotation_6d.shape[-1] != 6:
        raise ValueError("rotation_6d must have a final dimension of 6")

    first, second = rotation_6d.split(3, dim=-1)
    first_norm = torch.linalg.vector_norm(first, dim=-1, keepdim=True)
    raw_second_norm = torch.linalg.vector_norm(second, dim=-1, keepdim=True)
    default_first = torch.zeros_like(first)
    default_first[..., 0] = 1
    normalized_first = first / first_norm.clamp_min(eps)
    basis_first = torch.where(first_norm > eps, normalized_first, default_first)

    orthogonal_second = second - (basis_first * second).sum(dim=-1, keepdim=True) * basis_first
    second_norm = torch.linalg.vector_norm(orthogonal_second, dim=-1, keepdim=True)

    canonical_axes = torch.eye(3, dtype=rotation_6d.dtype, device=rotation_6d.device)
    fallback_index = basis_first.abs().argmin(dim=-1)
    fallback_second = canonical_axes[fallback_index]
    fallback_second = fallback_second - (
        fallback_second * basis_first
    ).sum(dim=-1, keepdim=True) * basis_first
    fallback_second = F.normalize(fallback_second, dim=-1, eps=eps)
    normalized_second = orthogonal_second / second_norm.clamp_min(eps)
    basis_second = torch.where(second_norm > eps, normalized_second, fallback_second)
    basis_third = torch.linalg.cross(basis_first, basis_second, dim=-1)

    matrix = torch.stack((basis_first, basis_second, basis_third), dim=-2)
    first_penalty = F.relu(eps - first_norm.squeeze(-1)) / eps
    second_penalty = F.relu(eps - raw_second_norm.squeeze(-1)) / eps
    relative_orthogonal_norm = second_norm / raw_second_norm.clamp_min(eps)
    collinearity_eps = 1e-3
    collinear_penalty = (
        F.relu(collinearity_eps - relative_orthogonal_norm.squeeze(-1))
        / collinearity_eps
    )
    degeneracy_penalty = first_penalty + second_penalty + collinear_penalty
    return matrix, degeneracy_penalty


def _geodesic_angle(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
    relative = first.transpose(-1, -2) @ second
    cosine = (relative.diagonal(dim1=-2, dim2=-1).sum(dim=-1) - 1.0) / 2.0
    clamp_eps = 1e-7
    cosine = cosine.clamp(-1.0 + clamp_eps, 1.0 - clamp_eps)
    return torch.acos(cosine)


def _scaled_huber(value: torch.Tensor, scale: float) -> torch.Tensor:
    if scale <= 0:
        raise ValueError("physical loss scales must be positive")
    scaled = value / scale
    return F.smooth_l1_loss(scaled, torch.zeros_like(scaled), reduction="none")


def _masked_mean(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask = mask.to(device=value.device, dtype=value.dtype)
    return (value * mask).sum() / mask.sum().clamp_min(1.0)


def _arm_mean(values: list[torch.Tensor]) -> torch.Tensor:
    return torch.stack(values).mean()


def _resolve_weights(weights: Mapping[str, float] | None) -> dict[str, float]:
    resolved = dict(_DEFAULT_WEIGHTS)
    if weights is not None:
        resolved.update(weights)
    resolved["rot6_aux_weight"] = min(max(float(resolved["rot6_aux_weight"]), 0.0), 0.1)
    return resolved


def compute_physical_action_loss(
        target: torch.Tensor,
        prediction: torch.Tensor,
        valid_mask: torch.Tensor,
        idle_arm_mask: torch.Tensor,
        weights: Mapping[str, float] | None = None) -> dict[str, torch.Tensor]:
    """Compute mask-aware physical reconstruction losses for 10D or 20D actions."""
    if target.shape != prediction.shape:
        raise ValueError("target and prediction must have matching shapes")
    arm_slices = action_arm_slices(target.shape[-1])
    if valid_mask.shape != target.shape[:-1]:
        raise ValueError("valid_mask must match target batch/time dimensions")
    if idle_arm_mask.shape != (*target.shape[:-1], len(arm_slices)):
        raise ValueError(
            f"idle_arm_mask must have shape [..., {len(arm_slices)}]"
        )

    resolved = _resolve_weights(weights)
    valid_mask = valid_mask.to(device=target.device, dtype=torch.bool)
    idle_arm_mask = idle_arm_mask.to(device=target.device, dtype=torch.bool)
    rotation_compute_dtype = _rotation_compute_dtype(target, prediction)
    with torch.autocast(device_type=prediction.device.type, enabled=False):
        identity_6d = target.new_tensor(
            [1, 0, 0, 0, 1, 0], dtype=rotation_compute_dtype
        )
        identity_matrix, _ = project_rotation_6d(identity_6d)

    position_terms = []
    rotation_terms = []
    gripper_terms = []
    idle_terms = []
    micro_motion_terms = []
    degenerate_terms = []
    rot6_aux_terms = []
    for arm_index, (position_slice, rotation_slice, gripper_index) in enumerate(arm_slices):
        target_position = target[..., position_slice]
        predicted_position = prediction[..., position_slice]
        position_error = torch.linalg.vector_norm(
            predicted_position - target_position, dim=-1
        )
        position_value = _scaled_huber(position_error, resolved["position_scale"])
        micro_motion_position_value = _scaled_huber(
            position_error, LOW_TRANSLATION_DELTA_M
        )
        position_terms.append(_masked_mean(position_value, valid_mask))

        with torch.autocast(device_type=prediction.device.type, enabled=False):
            target_rotation_6d = target[..., rotation_slice].to(rotation_compute_dtype)
            predicted_rotation_6d = prediction[..., rotation_slice].to(rotation_compute_dtype)
            target_rotation, _ = project_rotation_6d(target_rotation_6d)
            predicted_rotation, degeneracy = project_rotation_6d(predicted_rotation_6d)
            rotation_error = _geodesic_angle(target_rotation, predicted_rotation)
            rotation_value = _scaled_huber(
                rotation_error, resolved["rotation_scale"]
            )
            micro_motion_rotation_value = _scaled_huber(
                rotation_error, math.radians(LOW_ROTATION_DELTA_DEG)
            )
            rotation_terms.append(_masked_mean(rotation_value, valid_mask))
            degenerate_terms.append(_masked_mean(degeneracy, valid_mask))

            raw_rotation_error = F.smooth_l1_loss(
                predicted_rotation_6d,
                target_rotation_6d,
                reduction="none",
            ).mean(dim=-1)
            rot6_aux_terms.append(_masked_mean(raw_rotation_error, valid_mask))

            idle_rotation_error = _geodesic_angle(identity_matrix, predicted_rotation)
            idle_rotation_value = _scaled_huber(
                idle_rotation_error, resolved["idle_rotation_scale"]
            )
            # The contract's 0.25-degree lower bound is below BF16's useful
            # resolution around an identity rotation matrix. Keep target
            # classification in the same FP32/FP64 island as geodesic loss.
            target_rotation_error = _geodesic_angle(
                identity_matrix, target_rotation
            )

        gripper_error = (prediction[..., gripper_index] - target[..., gripper_index]).abs()
        gripper_terms.append(_masked_mean(
            _scaled_huber(gripper_error, resolved["gripper_scale"]), valid_mask
        ))

        idle_mask = valid_mask & idle_arm_mask[..., arm_index]
        target_translation = torch.linalg.vector_norm(target_position, dim=-1)
        target_is_micro_motion = (
            (
                (target_translation >= LOW_TRANSLATION_DELTA_M)
                | (target_rotation_error >= math.radians(LOW_ROTATION_DELTA_DEG))
            )
            & (target_translation <= HIGH_TRANSLATION_DELTA_M)
            & (target_rotation_error <= math.radians(HIGH_ROTATION_DELTA_DEG))
        )
        micro_motion_mask = (
            valid_mask & ~idle_arm_mask[..., arm_index] & target_is_micro_motion
        )
        micro_motion_terms.append(_masked_mean(
            micro_motion_position_value + micro_motion_rotation_value,
            micro_motion_mask,
        ))
        idle_position_error = torch.linalg.vector_norm(predicted_position, dim=-1)
        idle_value = (
            _scaled_huber(idle_position_error, resolved["idle_position_scale"])
            + idle_rotation_value
        )
        idle_terms.append(_masked_mean(idle_value, idle_mask))

    position_loss = _arm_mean(position_terms)
    rotation_loss = _arm_mean(rotation_terms)
    gripper_loss = _arm_mean(gripper_terms)
    idle_loss = _arm_mean(idle_terms)
    micro_motion_loss = _arm_mean(micro_motion_terms)
    degenerate_loss = _arm_mean(degenerate_terms)
    rot6_aux_loss = _arm_mean(rot6_aux_terms)
    total = (
        float(resolved["position_weight"]) * position_loss
        + float(resolved["rotation_weight"]) * rotation_loss
        + float(resolved["gripper_weight"]) * gripper_loss
        + float(resolved["idle_weight"]) * idle_loss
        + float(resolved["degenerate_weight"]) * degenerate_loss
        + float(resolved["rot6_aux_weight"]) * rot6_aux_loss
        + float(resolved["micro_motion_weight"]) * micro_motion_loss
    )
    return {
        "loss": total,
        "position_loss": position_loss,
        "rotation_loss": rotation_loss,
        "gripper_loss": gripper_loss,
        "idle_loss": idle_loss,
        "micro_motion_loss": micro_motion_loss,
        "degenerate_loss": degenerate_loss,
        "rot6_aux_loss": rot6_aux_loss,
    }


# Backward-compatible public name used by existing dual-arm callers and tests.
compute_bimanual_physical_loss = compute_physical_action_loss
