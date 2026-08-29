import inspect

import torch
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from torch import nn

from reactive_diffusion_policy.model.common.normalizer import (
    LinearNormalizer,
    SingleFieldLinearNormalizer,
)
from reactive_diffusion_policy.model.vae.model import VAE
from reactive_diffusion_policy.policy.latent_diffusion_unet_image_policy import (
    LatentDiffusionUnetImagePolicy,
)


def _make_at():
    return VAE(
        horizon=4,
        shape_meta={"action": {"shape": [2]}, "extended_obs": {}},
        n_latent_dims=4,
        n_embed=2,
        mlp_layer_num=0,
        use_vq=False,
        eval=False,
        device="cpu",
    )


def _at_parameters(at):
    modules = (at.encoder, at.decoder, at.quant, at.post_quant)
    return [parameter for module in modules for parameter in module.parameters()]


class _TinyObsEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(3, 4)

    def output_shape(self):
        return (4,)

    def forward(self, obs):
        return self.proj(obs["state"])


class _TinyDenoiser(nn.Module):
    def __init__(self):
        super().__init__()
        self.bias = nn.Parameter(torch.tensor(0.0))

    def forward(self, sample, timestep, local_cond=None, global_cond=None):
        return sample + self.bias


def _make_policy():
    at = _make_at()
    policy = LatentDiffusionUnetImagePolicy(
        at=at,
        use_latent_action_before_vq=False,
        shape_meta={
            "action": {"shape": [2]},
            "obs": {"state": {"shape": [3], "type": "low_dim"}},
        },
        noise_scheduler=DDPMScheduler(
            num_train_timesteps=4,
            prediction_type="epsilon",
        ),
        obs_encoder=_TinyObsEncoder(),
        horizon=4,
        n_action_steps=2,
        n_obs_steps=1,
        diffusion_step_embed_dim=8,
        down_dims=(8, 16),
        n_groups=1,
    )
    policy.model = _TinyDenoiser()
    normalizer = LinearNormalizer()
    for key in ("action", "latent_action", "state"):
        normalizer[key] = SingleFieldLinearNormalizer.create_identity()
    policy.set_normalizer(normalizer)
    return policy


def test_non_vq_posterior_mode_is_deterministic():
    at = _make_at()
    actions = torch.randn(3, 4, 2)
    encoded = at.encoder(at.preprocess(actions))

    first, _ = at.quant_state_without_vq(encoded, sample=False)
    second, _ = at.quant_state_without_vq(encoded, sample=False)

    torch.testing.assert_close(first, second, rtol=0, atol=0)


def test_non_vq_posterior_sampling_remains_the_default():
    sample_parameter = inspect.signature(
        VAE.quant_state_without_vq
    ).parameters["sample"]

    assert sample_parameter.default is True


def test_at_validation_posterior_mode_is_deterministic_and_reports_statistics():
    at = _make_at()
    normalizer = LinearNormalizer()
    normalizer["action"] = SingleFieldLinearNormalizer.create_identity()
    at.set_normalizer(normalizer)
    batch = {"action": torch.randn(3, 4, 2), "extended_obs": {}}

    first = at.compute_loss_and_metric(batch, sample_posterior=False)
    second = at.compute_loss_and_metric(batch, sample_posterior=False)

    torch.testing.assert_close(first["loss"], second["loss"], rtol=0, atol=0)
    assert first["posterior_mean"] == second["posterior_mean"]
    assert first["posterior_std"] == second["posterior_std"]
    assert first["posterior_std"] > 0


def test_ldp_latent_target_encoding_is_deterministic_and_detached():
    policy = _make_policy()
    actions = torch.randn(3, 4, 2)

    first = policy.encode_latent_target(actions)
    second = policy.encode_latent_target(actions)

    torch.testing.assert_close(first, second, rtol=0, atol=0)
    assert not first.requires_grad


def test_ldp_construction_freezes_every_action_tokenizer_parameter():
    policy = _make_policy()

    assert list(policy.at.parameters())
    assert all(not parameter.requires_grad for parameter in policy.at.parameters())


def test_ldp_backward_leaves_action_tokenizer_gradients_none():
    policy = _make_policy()
    batch = {
        "obs": {"state": torch.randn(2, 1, 3)},
        "action": torch.randn(2, 4, 2),
    }

    policy.compute_loss(batch).backward()

    assert all(parameter.grad is None for parameter in _at_parameters(policy.at))


def test_ldp_compute_loss_accepts_full_v2_dataset_batch():
    policy = _make_policy()
    batch = {
        "obs": {"state": torch.randn(2, 1, 3)},
        "action": torch.randn(2, 4, 2),
        "extended_obs": {},
        "valid_mask": torch.tensor(
            [[True, True, True, False], [True, True, False, False]]
        ),
        "idle_arm_mask": torch.zeros((2, 4, 2), dtype=torch.bool),
    }

    loss = policy.compute_loss(batch)

    assert torch.isfinite(loss)


def test_ldp_train_keeps_action_tokenizer_in_eval_mode():
    policy = _make_policy()
    policy.at.train()

    policy.train()

    assert policy.training
    assert not policy.at.encoder.training
    assert not policy.at.decoder.training
    assert not policy.at.quant.training
    assert not policy.at.post_quant.training


def test_ldp_optimizer_contains_only_ldp_parameters():
    policy = _make_policy()
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)
    optimizer_parameters = {
        id(parameter)
        for group in optimizer.param_groups
        for parameter in group["params"]
    }
    at_parameters = {id(parameter) for parameter in _at_parameters(policy.at)}

    assert optimizer_parameters
    assert optimizer_parameters.isdisjoint(at_parameters)
