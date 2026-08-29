import math
from collections.abc import Mapping

import torch
import torch.nn.functional as F


_ARM_SLICES = ((slice(0, 3), slice(3, 9), 9), (slice(10, 13), slice(13, 19), 19))
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
}


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


def compute_bimanual_physical_loss(
        target: torch.Tensor,
        prediction: torch.Tensor,
        valid_mask: torch.Tensor,
        idle_arm_mask: torch.Tensor,
        weights: Mapping[str, float] | None = None) -> dict[str, torch.Tensor]:
    """Compute mask-aware physical-domain reconstruction losses for 20D actions."""
    if target.shape != prediction.shape or target.shape[-1] != 20:
        raise ValueError("target and prediction must have matching [..., 20] shapes")
    if valid_mask.shape != target.shape[:-1]:
        raise ValueError("valid_mask must match target batch/time dimensions")
    if idle_arm_mask.shape != (*target.shape[:-1], 2):
        raise ValueError("idle_arm_mask must have shape [..., 2]")

    resolved = _resolve_weights(weights)
    valid_mask = valid_mask.to(device=target.device, dtype=torch.bool)
    idle_arm_mask = idle_arm_mask.to(device=target.device, dtype=torch.bool)
    identity_6d = target.new_tensor([1, 0, 0, 0, 1, 0])
    identity_matrix, _ = project_rotation_6d(identity_6d)

    position_terms = []
    rotation_terms = []
    gripper_terms = []
    idle_terms = []
    degenerate_terms = []
    rot6_aux_terms = []
    for arm_index, (position_slice, rotation_slice, gripper_index) in enumerate(_ARM_SLICES):
        target_position = target[..., position_slice]
        predicted_position = prediction[..., position_slice]
        position_error = torch.linalg.vector_norm(
            predicted_position - target_position, dim=-1
        )
        position_terms.append(_masked_mean(
            _scaled_huber(position_error, resolved["position_scale"]), valid_mask
        ))

        target_rotation_6d = target[..., rotation_slice]
        predicted_rotation_6d = prediction[..., rotation_slice]
        target_rotation, _ = project_rotation_6d(target_rotation_6d)
        predicted_rotation, degeneracy = project_rotation_6d(predicted_rotation_6d)
        rotation_error = _geodesic_angle(target_rotation, predicted_rotation)
        rotation_terms.append(_masked_mean(
            _scaled_huber(rotation_error, resolved["rotation_scale"]), valid_mask
        ))
        degenerate_terms.append(_masked_mean(degeneracy, valid_mask))

        gripper_error = (prediction[..., gripper_index] - target[..., gripper_index]).abs()
        gripper_terms.append(_masked_mean(
            _scaled_huber(gripper_error, resolved["gripper_scale"]), valid_mask
        ))

        raw_rotation_error = F.smooth_l1_loss(
            predicted_rotation_6d,
            target_rotation_6d,
            reduction="none",
        ).mean(dim=-1)
        rot6_aux_terms.append(_masked_mean(raw_rotation_error, valid_mask))

        idle_mask = valid_mask & idle_arm_mask[..., arm_index]
        idle_position_error = torch.linalg.vector_norm(predicted_position, dim=-1)
        idle_rotation_error = _geodesic_angle(identity_matrix, predicted_rotation)
        idle_value = (
            _scaled_huber(idle_position_error, resolved["idle_position_scale"])
            + _scaled_huber(idle_rotation_error, resolved["idle_rotation_scale"])
        )
        idle_terms.append(_masked_mean(idle_value, idle_mask))

    position_loss = _arm_mean(position_terms)
    rotation_loss = _arm_mean(rotation_terms)
    gripper_loss = _arm_mean(gripper_terms)
    idle_loss = _arm_mean(idle_terms)
    degenerate_loss = _arm_mean(degenerate_terms)
    rot6_aux_loss = _arm_mean(rot6_aux_terms)
    total = (
        float(resolved["position_weight"]) * position_loss
        + float(resolved["rotation_weight"]) * rotation_loss
        + float(resolved["gripper_weight"]) * gripper_loss
        + float(resolved["idle_weight"]) * idle_loss
        + float(resolved["degenerate_weight"]) * degenerate_loss
        + float(resolved["rot6_aux_weight"]) * rot6_aux_loss
    )
    return {
        "loss": total,
        "position_loss": position_loss,
        "rotation_loss": rotation_loss,
        "gripper_loss": gripper_loss,
        "idle_loss": idle_loss,
        "degenerate_loss": degenerate_loss,
        "rot6_aux_loss": rot6_aux_loss,
    }
