"""Offline numerical regression against the user's untouched RDP sources."""
import copy
import importlib.util
from pathlib import Path

import pytest
import torch
from torch import nn
from diffusers import DDPMScheduler

from rdp_baseline.model import VAE
from rdp_baseline.policy import LatentDiffusionUnetImagePolicy
from rdp_baseline.workspace_ldp import fit_latent_normalizer
from reactive_diffusion_policy.model.common.normalizer import LinearNormalizer, SingleFieldLinearNormalizer
from reactive_diffusion_policy.dataset.base_dataset import BaseImageDataset
from omegaconf import OmegaConf

torch.set_num_threads(1)
ORIGINAL = Path(__file__).parents[1] / 'reactive_diffusion_policy-main/reactive_diffusion_policy'
if not ORIGINAL.is_dir():
    ORIGINAL = Path(__file__).parent / 'fixtures/rdp_upstream'


def original_module(name, relative):
    spec = importlib.util.spec_from_file_location(name, ORIGINAL / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def shape_meta(action_dim=10, touch_dim=2):
    return {'action': {'shape': [action_dim]}, 'obs': {'state': {'shape': [3], 'type': 'low_dim'}},
            'extended_obs': {'touch': {'shape': [touch_dim], 'type': 'low_dim'}}}


def make_at(cls=VAE, meta=None):
    return cls(horizon=4, shape_meta=shape_meta() if meta is None else meta, n_latent_dims=4, n_embed=2,
               use_rnn_decoder=True, rnn_latent_dims=8, device='cpu', kl_multiplier=.03)


def normalizer():
    norm = LinearNormalizer()
    for key in ('action', 'state', 'touch', 'latent_action'):
        norm[key] = SingleFieldLinearNormalizer.create_identity()
    return norm


def batch(meta=None):
    meta = shape_meta() if meta is None else meta
    return {'action': torch.randn(2, 4, meta['action']['shape'][0]), 'obs': {'state': torch.randn(2, 2, 3)},
            'extended_obs': {'touch': torch.randn(2, 4, meta['extended_obs']['touch']['shape'][0])}}


class TinyObsEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(3, 4)

    def output_shape(self):
        return (4,)

    def forward(self, obs):
        return self.linear(obs['state'])


def make_policy(cls=LatentDiffusionUnetImagePolicy, meta=None):
    return cls(at=make_at(meta=meta), use_latent_action_before_vq=False,
               shape_meta=shape_meta() if meta is None else meta,
               noise_scheduler=DDPMScheduler(num_train_timesteps=10),
               obs_encoder=TinyObsEncoder(), horizon=4, n_action_steps=2, n_obs_steps=2,
               diffusion_step_embed_dim=8, down_dims=(8, 16), n_groups=2)


@pytest.mark.parametrize('action_dim,touch_dim', [(10, 2), (20, 30)])
def test_at_original_l1_kl_numerical_and_gradient_equivalence(action_dim,touch_dim):
    meta = shape_meta(action_dim,touch_dim)
    original = original_module('original_at', 'model/vae/model.py').VAE
    model, reference = make_at(meta=meta), make_at(original,meta=meta)
    reference.load_state_dict(copy.deepcopy(model.state_dict()))
    for item in (model, reference):
        item.set_normalizer(normalizer())
    data = batch(meta)
    torch.manual_seed(42)
    actual = model.compute_loss_and_metric(data)
    torch.manual_seed(42)
    expected = reference.compute_loss_and_metric(data)
    assert actual['loss'].item() == expected['loss'].item()
    assert actual['loss'].item() == pytest.approx(float(actual['encoder_loss']) + .03 * float(actual['kl_loss']))
    actual['loss'].backward()
    expected['loss'].backward()
    for a, b in zip(model.optim_params, reference.optim_params):
        torch.testing.assert_close(a.grad, b.grad, rtol=0, atol=0)
    before = model.decoder.fc.weight.detach().clone()
    torch.optim.AdamW(model.optim_params, lr=1e-3).step()
    assert not torch.equal(before, model.decoder.fc.weight)
    model.eval()
    with torch.no_grad():
        assert torch.isfinite(model.compute_loss_and_metric(data)['loss'])


def test_posterior_samples_even_in_eval():
    model = make_at().eval()
    encoded = model.encoder(model.preprocess(batch()['action']))
    first, posterior = model.quant_state_without_vq(encoded)
    second, _ = model.quant_state_without_vq(encoded)
    assert not torch.equal(first, second)
    assert not torch.equal(first, posterior.mode().flatten(1))


@pytest.mark.parametrize('action_dim,touch_dim', [(10, 2), (20, 30)])
def test_ldp_original_sampled_target_loss_and_frozen_at(action_dim,touch_dim):
    meta = shape_meta(action_dim,touch_dim)
    original = original_module('original_ldp', 'policy/latent_diffusion_unet_image_policy.py').LatentDiffusionUnetImagePolicy
    # The untouched upstream policy mutates its argument to latent dimensions.
    policy, reference = make_policy(meta=meta), make_policy(original,meta=copy.deepcopy(meta))
    reference.load_state_dict({k: copy.deepcopy(v) for k, v in policy.state_dict().items()
                               if k != '_extra_state'})
    reference.at.load_state_dict(copy.deepcopy(policy.at.state_dict()))
    for item in (policy, reference):
        item.set_normalizer(normalizer())
    data = batch(meta)
    torch.manual_seed(7)
    loss = policy.compute_loss(data)
    torch.manual_seed(7)
    expected = reference.compute_loss(data)
    torch.testing.assert_close(loss, expected, rtol=0, atol=0)
    loss.backward()
    assert any(p.grad is not None for p in policy.model.parameters())
    assert all(p.grad is None and not p.requires_grad for p in policy.at.optim_params)
    policy.train()
    assert not policy.at.encoder.training


@pytest.mark.parametrize('action_dim,touch_dim', [(10, 2), (20, 30)])
def test_policy_does_not_mutate_meta_and_decodes_full_action_dim(action_dim,touch_dim):
    meta = shape_meta(action_dim,touch_dim)
    policy = make_policy(meta=meta)
    policy.set_normalizer(normalizer())
    assert meta == shape_meta(action_dim,touch_dim)
    assert policy.action_dim == 2
    with torch.no_grad():
        out = policy.predict_action(batch(meta)['obs'], 1, batch(meta)['extended_obs'])
    assert out['action_pred'].shape == (2, 4, action_dim)


def test_at_checkpoint_retains_normalizer_and_loads_original_payload():
    model = make_at()
    norm = normalizer()
    norm['action'] = SingleFieldLinearNormalizer.create_fit(torch.randn(10, 10) * 8)
    model.set_normalizer(norm)
    fresh = make_at()
    fresh.load_state_dict(copy.deepcopy(model.state_dict()))
    for key, value in norm.state_dict().items():
        torch.testing.assert_close(fresh.normalizer.state_dict()[key], value)
    original = original_module('original_at_state', 'model/vae/model.py').VAE
    fresh.load_state_dict(make_at(original).state_dict())


def test_ldp_checkpoint_restores_frozen_at_without_external_weights():
    policy = make_policy()
    policy.set_normalizer(normalizer())
    fresh = make_policy()
    assert not torch.equal(fresh.at.decoder.fc.weight, policy.at.decoder.fc.weight)
    fresh.load_state_dict(copy.deepcopy(policy.state_dict()))
    for a, b in zip(fresh.at.optim_params, policy.at.optim_params):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    for key, value in policy.at.normalizer.state_dict().items():
        torch.testing.assert_close(fresh.at.normalizer.state_dict()[key], value)


def test_latent_normalizer_samples_lowdim_only_and_matches_streaming_stats():
    class Dataset:
        def __len__(self):
            return 7

        def __getitem__(self, index):
            raise AssertionError('latent fit must not decode RGB')

        def get_lowdim_batch(self, indices):
            return {'action': torch.arange(len(indices) * 40).reshape(-1, 4, 10).float().numpy() / 100}

    at = make_at()
    norm = normalizer()
    at.set_normalizer(norm)
    captured = []
    quant = at.quant_state_without_vq

    def capture(state):
        latent, posterior = quant(state)
        captured.append(latent.reshape(-1, at.n_embed).detach())
        return latent, posterior

    at.quant_state_without_vq = capture
    result = fit_latent_normalizer(Dataset(), at, norm, batch_size=3)
    expected = SingleFieldLinearNormalizer.create_fit(torch.cat(captured))
    for key in ('min', 'max', 'mean', 'std'):
        torch.testing.assert_close(result['latent_action'].get_input_stats()[key], expected.get_input_stats()[key])
    torch.testing.assert_close(result['latent_action'].params_dict['scale'], expected.params_dict['scale'])


class TinyDataset(BaseImageDataset):
    def __init__(self, **kwargs):
        pass

    def __len__(self):
        return 10

    def __getitem__(self, index):
        generator = torch.Generator().manual_seed(index)
        return {'action': torch.randn(4, 10, generator=generator),
                'obs': {'state': torch.randn(2, 3, generator=generator)},
                'extended_obs': {'touch': torch.randn(4, 2, generator=generator)}}

    def get_lowdim_batch(self, indices):
        return {'action': torch.stack([self[i]['action'] for i in indices]).numpy()}

    def get_normalizer(self):
        return normalizer()

    def get_validation_dataset(self):
        return self


@pytest.mark.parametrize('role', ['at', 'ldp'])
def test_workspace_cpu_train_val_resume_and_partial_accumulation(tmp_path, monkeypatch, role):
    from rdp_baseline.workspace_at import TrainATWorkspace
    from rdp_baseline.workspace_ldp import TrainDiffusionUnetImageWorkspace
    at_cfg = dict(_target_='rdp_baseline.model.VAE', horizon=4, shape_meta=shape_meta(),
                  n_latent_dims=4, n_embed=2, use_rnn_decoder=True, rnn_latent_dims=8, device='cpu')
    policy_cfg = at_cfg if role == 'at' else dict(
        _target_='rdp_baseline.policy.LatentDiffusionUnetImagePolicy', at=at_cfg,
        use_latent_action_before_vq=False, shape_meta=shape_meta(),
        noise_scheduler=dict(_target_='diffusers.DDPMScheduler', num_train_timesteps=4),
        obs_encoder=dict(_target_=f'{__name__}.TinyObsEncoder'),
        horizon=4, n_action_steps=2, n_obs_steps=2, diffusion_step_embed_dim=8,
        down_dims=[8, 16], n_groups=2)
    cfg = OmegaConf.create(dict(
        name='baseline_latent', policy=policy_cfg, shape_meta=shape_meta(),
        optimizer=dict(_target_='torch.optim.AdamW', lr=.001, weight_decay=.0001),
        ema=dict(_target_='reactive_diffusion_policy.model.diffusion.ema_model.EMAModel'),
        task=dict(dataset=dict(_target_=f'{__name__}.TinyDataset', val_ratio=.1,
                               obs_temporal_downsample_ratio=1)),
        dataloader=dict(batch_size=2, shuffle=False), val_dataloader=dict(batch_size=2),
        logging=dict(project='offline-baseline-test', mode='disabled'),
        checkpoint=dict(save_last_ckpt=True, save_last_snapshot=False,
                        topk=dict(monitor_key='train_loss', mode='min', k=1,
                                  format_str='epoch={epoch:04d}-train_loss={train_loss:.3f}.ckpt')),
        training=dict(seed=42, device='cpu', resume=False, num_epochs=1, debug=False,
                      gradient_accumulate_every=2, max_train_steps=3, max_val_steps=1,
                      lr_scheduler='constant', lr_warmup_steps=0, checkpoint_every=1,
                      val_every=1, sample_every=1, use_ema=True, freeze_encoder=False,
                      tqdm_interval_sec=10, latent_normalizer_batch_size=3)))
    # Dataset kwargs mirror launcher configuration; the tiny fixture ignores them.
    cls = TrainATWorkspace if role == 'at' else TrainDiffusionUnetImageWorkspace
    workspace = cls(cfg, output_dir=str(tmp_path))
    workspace.run()
    assert workspace.optimizer_step == 2  # batches [0,1], then partial [2]
    assert workspace.global_step == 3 and workspace.epoch == 1
    assert 'val_loss' in (tmp_path / 'logs.json.txt').read_text()
    saved_norm = copy.deepcopy(workspace.model.normalizer.state_dict())
    cfg.training.num_epochs = 2
    cfg.training.resume = True
    if role == 'ldp':
        def forbidden_refit(*args, **kwargs):
            raise AssertionError('resume must retain the fitted latent coordinate system')
        monkeypatch.setattr('rdp_baseline.workspace_ldp.fit_latent_normalizer', forbidden_refit)
    resumed = cls(cfg, output_dir=str(tmp_path))
    resumed.run()
    assert resumed.optimizer_step == 4
    assert resumed.global_step == 6 and resumed.epoch == 2
    if role == 'ldp':
        for name, value in saved_norm.items():
            torch.testing.assert_close(resumed.model.normalizer.state_dict()[name], value, rtol=0, atol=0)
