import math

import pytest
import torch

from reactive_diffusion_policy.common.normalize_util import get_action_normalizer
from reactive_diffusion_policy.model.vae.model import VAE
from reactive_diffusion_policy.model.vae.physical_action_loss import (
    compute_bimanual_physical_loss,
    project_rotation_6d,
)


IDENTITY_6D = torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float64)
DEFAULT_WEIGHTS = {
    "position_scale": 1e-3,
    "rotation_scale": math.radians(1.0),
    "gripper_scale": 5e-3,
    "idle_position_scale": 1e-4,
    "idle_rotation_scale": math.radians(0.05),
    "idle_weight": 1.0,
    "degenerate_weight": 1.0,
    "rot6_aux_weight": 0.05,
}


def _identity_actions(batch=1, horizon=3, dtype=torch.float64):
    actions = torch.zeros((batch, horizon, 20), dtype=dtype)
    actions[..., 3:9] = IDENTITY_6D.to(dtype)
    actions[..., 13:19] = IDENTITY_6D.to(dtype)
    return actions


def _z_rotation_10d_action(
    angle_degrees: float,
    *,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Return a single-arm action whose 6D rotation is a Z-axis rotation."""
    angle = math.radians(angle_degrees)
    action = torch.zeros((1, 1, 10), device=device, dtype=dtype)
    action[..., 3:9] = torch.tensor(
        [math.cos(angle), -math.sin(angle), 0, math.sin(angle), math.cos(angle), 0],
        device=device,
        dtype=dtype,
    )
    return action


def _single_arm_rotation_loss(
    target: torch.Tensor,
    prediction: torch.Tensor,
) -> dict[str, torch.Tensor]:
    return compute_bimanual_physical_loss(
        target,
        prediction,
        torch.ones(target.shape[:2], device=target.device, dtype=torch.bool),
        torch.zeros((*target.shape[:2], 1), device=target.device, dtype=torch.bool),
        DEFAULT_WEIGHTS,
    )


def _loss(target, prediction, valid_mask=None, idle_arm_mask=None, weights=None):
    if valid_mask is None:
        valid_mask = torch.ones(target.shape[:2], dtype=torch.bool)
    if idle_arm_mask is None:
        idle_arm_mask = torch.zeros((*target.shape[:2], 2), dtype=torch.bool)
    return compute_bimanual_physical_loss(
        target,
        prediction,
        valid_mask,
        idle_arm_mask,
        DEFAULT_WEIGHTS if weights is None else weights,
    )


@pytest.mark.parametrize("angle_degrees", [0.05, 0.5, 1.2])
def test_bf16_cpu_autocast_preserves_small_10d_rotation_loss(angle_degrees):
    target = _z_rotation_10d_action(0.0)
    prediction = _z_rotation_10d_action(angle_degrees)

    with torch.autocast("cpu", enabled=False):
        reference = _single_arm_rotation_loss(target, prediction)["rotation_loss"]
    with torch.autocast("cpu", dtype=torch.bfloat16):
        actual = _single_arm_rotation_loss(target, prediction)["rotation_loss"]

    assert actual.dtype == torch.float32
    assert actual > 0
    torch.testing.assert_close(actual, reference, atol=1e-6, rtol=1e-6)


def test_bf16_cpu_autocast_preserves_10d_rotation_gradients():
    target = _z_rotation_10d_action(0.0)
    prediction = _z_rotation_10d_action(1.2).requires_grad_()

    with torch.autocast("cpu", dtype=torch.bfloat16):
        rotation_loss = _single_arm_rotation_loss(target, prediction)["rotation_loss"]
    rotation_loss.backward()

    rotation_gradient = prediction.grad[..., 3:9]
    assert torch.isfinite(rotation_gradient).all()
    assert torch.count_nonzero(rotation_gradient) > 0


@pytest.mark.skipif(
    not torch.cuda.is_available() or not torch.cuda.is_bf16_supported(),
    reason="CUDA BF16 support is required",
)
@pytest.mark.parametrize("angle_degrees", [0.05, 0.5, 1.2])
def test_bf16_cuda_autocast_preserves_small_10d_rotation_loss(angle_degrees):
    target = _z_rotation_10d_action(0.0, device="cuda")
    prediction = _z_rotation_10d_action(angle_degrees, device="cuda")

    with torch.autocast("cuda", enabled=False):
        reference = _single_arm_rotation_loss(target, prediction)["rotation_loss"]
    with torch.autocast("cuda", dtype=torch.bfloat16):
        actual = _single_arm_rotation_loss(target, prediction)["rotation_loss"]

    assert actual.dtype == torch.float32
    assert actual > 0
    torch.testing.assert_close(actual, reference, atol=1e-6, rtol=1e-6)


def test_project_rotation_6d_maps_identity_to_a_valid_rotation():
    matrix, penalty = project_rotation_6d(IDENTITY_6D)

    torch.testing.assert_close(matrix, torch.eye(3, dtype=torch.float64))
    torch.testing.assert_close(penalty, torch.zeros_like(penalty))


def test_project_rotation_6d_projects_nonorthogonal_inputs_to_so3():
    rotation_6d = torch.tensor(
        [[2.0, 0.1, -0.2, 0.7, 1.5, 0.3], [0.2, 1.0, 0.4, 1.2, -0.3, 0.8]],
        dtype=torch.float64,
    )

    matrix, penalty = project_rotation_6d(rotation_6d)
    identity = torch.eye(3, dtype=torch.float64).expand_as(matrix)

    assert torch.isfinite(matrix).all()
    assert torch.isfinite(penalty).all()
    assert torch.linalg.matrix_norm(matrix.transpose(-1, -2) @ matrix - identity).max() < 1e-5
    assert (torch.linalg.det(matrix) - 1).abs().max() < 1e-5


def test_physical_loss_masks_invalid_and_padded_timesteps():
    target = _identity_actions(horizon=3)
    clean_losses = _loss(target, target)
    prediction = target.clone()
    prediction[:, 1:, :3] = 1000.0
    prediction[:, 1:, 3:9] = 0.0
    valid_mask = torch.tensor([[True, False, False]])

    losses = _loss(target, prediction, valid_mask=valid_mask)

    for name in (
        "position_loss",
        "rotation_loss",
        "gripper_loss",
        "idle_loss",
        "degenerate_loss",
        "rot6_aux_loss",
        "loss",
    ):
        torch.testing.assert_close(losses[name], clean_losses[name])


def test_physical_loss_averages_left_and_right_arms_symmetrically():
    target = _identity_actions(horizon=2)
    left_prediction = target.clone()
    right_prediction = target.clone()
    left_prediction[..., 0] = 0.002
    right_prediction[..., 10] = 0.002

    left_loss = _loss(target, left_prediction)["position_loss"]
    right_loss = _loss(target, right_prediction)["position_loss"]

    torch.testing.assert_close(left_loss, right_loss)


def test_physical_loss_uses_so3_geodesic_rotation_error():
    target = _identity_actions(horizon=1)
    prediction = target.clone()
    prediction[..., 3:9] = torch.tensor([0, -1, 0, 1, 0, 0], dtype=torch.float64)

    losses = _loss(target, prediction)

    assert losses["rotation_loss"] > 0
    assert losses["position_loss"] == 0


def test_physical_loss_reports_quantitative_geodesic_angle_without_dead_zone():
    target = _identity_actions(horizon=1)
    prediction = target.clone()
    angle = math.radians(1.0)
    prediction[..., 3:9] = torch.tensor(
        [math.cos(angle), -math.sin(angle), 0, math.sin(angle), math.cos(angle), 0],
        dtype=torch.float64,
    )

    rotation_loss = _loss(target, prediction)["rotation_loss"]
    clean_rotation_loss = _loss(target, target)["rotation_loss"]
    clamp_angle = math.acos(1.0 - 1e-7)
    clamp_huber = 0.5 * (clamp_angle / math.radians(1.0)) ** 2
    expected_increase = (0.5 - clamp_huber) / 2.0

    torch.testing.assert_close(
        rotation_loss - clean_rotation_loss,
        torch.tensor(expected_increase, dtype=torch.float64),
        atol=1e-6,
        rtol=0,
    )


def test_idle_loss_responds_only_for_masked_arm_and_timestep():
    target = _identity_actions(horizon=2)
    prediction = target.clone()
    prediction[:, 0, 0] = 0.0002
    prediction[:, 0, 3:9] = torch.tensor([0, -1, 0, 1, 0, 0], dtype=torch.float64)
    idle_mask = torch.tensor([[[True, False], [False, False]]])

    masked = _loss(target, prediction, idle_arm_mask=idle_mask)["idle_loss"]
    unmasked = _loss(target, prediction)["idle_loss"]

    assert masked > 0
    torch.testing.assert_close(unmasked, torch.zeros_like(unmasked))


def test_nearly_collinear_projection_has_finite_loss_and_gradients():
    target = _identity_actions(horizon=1)
    prediction = target.clone()
    prediction[..., 3:9] = torch.tensor(
        [1.0, 0.0, 0.0, 1.0, 1e-10, 0.0], dtype=torch.float64
    )
    prediction.requires_grad_(True)

    losses = _loss(target, prediction)
    losses["loss"].backward()

    assert losses["degenerate_loss"] > 0
    assert torch.isfinite(losses["loss"])
    assert torch.isfinite(prediction.grad).all()


def test_degeneracy_penalty_detects_scale_invariant_collinearity():
    rotation_6d = torch.tensor([1.0, 0, 0, 1e9, 1.0, 0], dtype=torch.float64)

    _, penalty = project_rotation_6d(rotation_6d)

    assert penalty > 0


def test_rot6_auxiliary_weight_is_capped_at_point_one():
    target = _identity_actions(horizon=1)
    prediction = target.clone()
    prediction[..., 3] += 0.2
    high_weights = dict(DEFAULT_WEIGHTS, rot6_aux_weight=10.0)
    capped_weights = dict(DEFAULT_WEIGHTS, rot6_aux_weight=0.1)

    high = _loss(target, prediction, weights=high_weights)["loss"]
    capped = _loss(target, prediction, weights=capped_weights)["loss"]

    torch.testing.assert_close(high, capped)


def test_vae_physical_v2_returns_named_metrics_and_scalar_loss():
    vae = VAE(
        horizon=2,
        shape_meta={"action": {"shape": [20]}, "extended_obs": {}},
        n_latent_dims=4,
        n_embed=2,
        mlp_layer_num=0,
        use_vq=False,
        eval=False,
        device="cpu",
        action_loss_version="physical_v2",
        physical_loss_weights=DEFAULT_WEIGHTS,
    )
    actions = _identity_actions(batch=4, horizon=2, dtype=torch.float32)
    fit_actions = actions.reshape(-1, 20).numpy().copy()
    fit_actions[:, [9, 19]] = torch.linspace(0.01, 0.08, len(fit_actions)).numpy()[:, None]
    normalizer = get_action_normalizer(
        fit_actions,
        bimanual_contiguous=True,
        version="zero_centered_v2",
    )
    from reactive_diffusion_policy.model.common.normalizer import LinearNormalizer

    wrapped = LinearNormalizer()
    wrapped["action"] = normalizer
    vae.set_normalizer(wrapped)
    batch = {
        "action": actions,
        "valid_mask": torch.ones((4, 2), dtype=torch.bool),
        "idle_arm_mask": torch.zeros((4, 2, 2), dtype=torch.bool),
    }

    result = vae.compute_loss_and_metric(batch)

    assert result["loss"].ndim == 0
    for name in (
        "position_loss",
        "rotation_loss",
        "gripper_loss",
        "idle_loss",
        "degenerate_loss",
        "rot6_aux_loss",
        "kl_loss",
        "rep_loss",
    ):
        assert name in result


def test_vae_physical_v2_projects_rotation_before_returning_actions():
    vae = VAE(
        horizon=1,
        shape_meta={"action": {"shape": [20]}, "extended_obs": {}},
        n_latent_dims=4,
        n_embed=2,
        mlp_layer_num=0,
        use_vq=False,
        eval=False,
        device="cpu",
        action_loss_version="physical_v2",
    )
    fit_actions = _identity_actions(batch=4, horizon=1, dtype=torch.float32).reshape(-1, 20)
    fit_actions[:, 9] = torch.linspace(0.01, 0.04, 4)
    fit_actions[:, 19] = torch.linspace(0.02, 0.05, 4)
    normalizer = get_action_normalizer(
        fit_actions.numpy(), bimanual_contiguous=True, version="zero_centered_v2"
    )
    from reactive_diffusion_policy.model.common.normalizer import LinearNormalizer

    wrapped = LinearNormalizer()
    wrapped["action"] = normalizer
    vae.set_normalizer(wrapped)
    physical_output = fit_actions[0].clone()
    physical_output[3:9] = torch.tensor([2.0, 0.1, -0.2, 0.7, 1.5, 0.3])
    normalized_output = normalizer.normalize(physical_output).detach()

    class FixedDecoder(torch.nn.Module):
        def forward(self, latent):
            return normalized_output.expand(latent.shape[0], -1)

    vae.decoder = FixedDecoder()
    returned = vae.get_action_from_latent(torch.zeros((2, 4)))
    physical = normalizer.unnormalize(returned).detach()
    rows = physical[..., 3:9].reshape(2, 1, 2, 3)
    third = torch.linalg.cross(rows[..., 0, :], rows[..., 1, :], dim=-1)
    matrix = torch.cat((rows, third.unsqueeze(-2)), dim=-2)
    identity = torch.eye(3).expand_as(matrix)

    assert torch.linalg.matrix_norm(matrix.transpose(-1, -2) @ matrix - identity).max() < 1e-5
    assert (torch.linalg.det(matrix) - 1).abs().max() < 1e-5



def test_single_right_physical_loss_and_vae_support_10d_actions():
    target = torch.zeros((2, 3, 10), dtype=torch.float32)
    target[..., 3:9] = IDENTITY_6D.float()
    prediction = target.clone()
    prediction[..., 0] = 0.001
    valid_mask = torch.ones((2, 3), dtype=torch.bool)
    idle_mask = torch.ones((2, 3, 1), dtype=torch.bool)

    losses = compute_bimanual_physical_loss(
        target,
        prediction,
        valid_mask,
        idle_mask,
        DEFAULT_WEIGHTS,
    )
    assert losses["position_loss"] > 0
    assert losses["idle_loss"] > 0

    vae = VAE(
        horizon=3,
        shape_meta={"action": {"shape": [10]}, "extended_obs": {}},
        n_latent_dims=4,
        n_embed=2,
        mlp_layer_num=0,
        use_vq=False,
        eval=False,
        device="cpu",
        action_loss_version="physical_v2",
        physical_loss_weights=DEFAULT_WEIGHTS,
    )
    fit_actions = target.reshape(-1, 10).numpy().copy()
    fit_actions[:, 9] = torch.linspace(0.01, 0.06, len(fit_actions)).numpy()
    normalizer = get_action_normalizer(fit_actions, version="zero_centered_v2")
    from reactive_diffusion_policy.model.common.normalizer import LinearNormalizer

    wrapped = LinearNormalizer()
    wrapped["action"] = normalizer
    vae.set_normalizer(wrapped)
    result = vae.compute_loss_and_metric(
        {
            "action": target,
            "valid_mask": valid_mask,
            "idle_arm_mask": idle_mask,
        }
    )
    assert result["loss"].ndim == 0
