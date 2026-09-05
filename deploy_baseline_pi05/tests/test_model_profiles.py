from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np
import pytest
import torch

from deploy_baseline_pi05.deployment import TACTILE_KEYS, load_deployment_config

RIGHT_KEYS = ("observation.images.tactile_left_1", "observation.images.tactile_right_1")
CONFIG = Path(__file__).resolve().parents[1] / "configs/deploy_baseline_pi05.yaml"
sys.path.insert(0, str(CONFIG.parents[1] / "src"))


@pytest.mark.parametrize("width,keys", [(10, RIGHT_KEYS), (20, TACTILE_KEYS)])
def test_training_best_checkpoint_matches_deployed_forward(tmp_path, width, keys):
    from train_baseline_pi05.checkpoint import save_best_checkpoint
    from train_baseline_pi05.model import DirectDecoderConfig, DirectTactileActionDecoder
    from deploy_baseline_pi05.checkpoint import load_decoder

    torch.set_num_threads(1)
    training = DirectTactileActionDecoder(DirectDecoderConfig(action_dim=width, tactile_keys=keys)).eval()
    source = {"pi": {"model_action_width": width}, "encoder": {"key_order": list(keys)}}
    checkpoint = save_best_checkpoint(tmp_path, training, training.config, epoch=2, global_step=3,
                                      metrics={"validation_loss": 0.2}, source_contract=source)
    deployed = load_decoder(checkpoint, expected_source=source)
    training.requires_grad_(False)
    coarse, tactile = torch.randn(2, 50, width), torch.randn(2, len(keys), 512)
    with torch.inference_mode():
        torch.testing.assert_close(deployed(coarse, tactile), training(coarse, tactile), rtol=0, atol=0)
    assert not deployed.training
    assert all(not parameter.requires_grad for parameter in deployed.parameters())


@pytest.mark.parametrize("keys", [RIGHT_KEYS, TACTILE_KEYS])
def test_tactile_encoding_batches_in_model_order_with_wire_mapping(monkeypatch, keys):
    import jax.numpy as jnp
    import deploy_baseline_pi05.tactile_encoder as module

    monkeypatch.setattr(module, "load_tactile_encoder", lambda path: SimpleNamespace(
        params={"tactile_resnet": {}}, metadata={"tactile_clip_config": {}}))
    monkeypatch.setattr(module, "tactile_clip_config_from_dict", lambda raw: SimpleNamespace(
        embedding_dim=512, tactile_image_size=224))

    def encode(variables, images, *, train, embedding_dim):
        assert train is False
        # Vary feature ratios, since RMS normalization removes overall scale.
        means = images.mean(axis=(1, 2, 3))
        return jnp.ones((len(keys), 512)).at[:, 0].set(means), {}

    monkeypatch.setattr(module, "encode_resnet18", encode)
    # Deliberately insert mapping entries in reverse model order.
    key_map = {key: f"wire.sensor_{index}" for index, key in reversed(list(enumerate(keys)))}
    encoder = module.FrozenTactileEncoder("unused", tactile_keys=keys, key_map=key_map)
    observation = {key_map[key]: np.full((224, 224, 3), 20 + 30 * index, dtype=np.uint8)
                   for index, key in enumerate(keys)}
    tokens = encoder.encode(observation)
    assert tokens.shape == (1, len(keys), 512)
    np.testing.assert_allclose(tokens[0, :, 0] / tokens[0, :, 1],
                               (20 + 30 * np.arange(len(keys))) / 255, rtol=1e-5)
    np.testing.assert_allclose(np.square(tokens).mean(axis=-1), 1, rtol=1e-6)


@pytest.mark.parametrize("width,state_dim,model_width,quantiles", [(10, 7, 10, True), (10, 7, 32, False), (20, 20, 20, True)])
def test_policy_matches_training_noise_and_normalization_and_reuses_jit(monkeypatch, width, state_dim, model_width, quantiles):
    import flax.nnx as nnx
    import deploy_baseline_pi05.policy as module
    from lerobot.policies.pi05_jax.normalize import NormStats
    from lerobot.policies.pi05_jax.transforms import Unnormalize
    from train_baseline_pi05.source_model import fixed_noise

    config = load_deployment_config(CONFIG)
    camera_map = {"right_wrist_0_rgb": "observation.images.camera1"} if width == 10 else config.source.camera_map
    config = replace(config, source=replace(config.source, action_dim=width, state_dim=state_dim,
                     model_action_dim=model_width, camera_map=camera_map),
                     norm_stats=replace(config.norm_stats, use_quantile_norm=quantiles))
    stats = {"state": NormStats(mean=np.ones(state_dim), std=np.full(state_dim, 2.),
                               q01=np.full(state_dim, -2.), q99=np.full(state_dim, 4.)),
             "actions": NormStats(mean=np.arange(width, dtype=np.float32), std=np.full(width, 2.),
                                 q01=np.full(width, -3.), q99=np.full(width, 5.))}
    traces = []
    tokenized_states = []

    class Model(nnx.Module):
        def sample_actions(self, rng, observation, *, noise, num_steps):
            traces.append(num_steps)
            assert observation.state.shape == (1, model_width)
            assert tuple(observation.images) == tuple(sorted(camera_map))
            return noise

    class Tokenizer:
        def __init__(self, length):
            self.length = length

        def tokenize(self, prompt, state):
            tokenized_states.append(state.copy())
            return np.zeros(self.length, np.int32), np.ones(self.length, bool)

    monkeypatch.setattr(module, "resolve_checkpoint", lambda value: value)
    monkeypatch.setattr(module, "load_pi0", lambda *args, **kwargs: Model())
    monkeypatch.setattr(module, "load_norm_stats", lambda *args: stats)
    monkeypatch.setattr(module, "PaligemmaTokenizer", Tokenizer)
    policy = module.Pi05VisualPolicy(config)
    observation = {"observation.state": np.arange(state_dim, dtype=np.float32),
                   **{key: np.full((224, 224, 3), 80, np.uint8) for key in camera_map.values()}}
    for _ in range(2):
        predicted = policy.predict_action_chunk(observation, "pick up tube")
        expected = np.asarray(fixed_noise(1, seed=0, horizon=50, action_dim=model_width))[..., :width]
        np.testing.assert_array_equal(predicted, expected)
    assert traces == [10]
    raw_state = observation["observation.state"].astype(stats["state"].mean.dtype)
    expected_state = ((raw_state + 2) / (6 + 1e-6) * 2 - 1
                      if quantiles else (raw_state - 1) / (2 + 1e-6))
    np.testing.assert_allclose(tokenized_states[0], expected_state)
    physical = policy.unnormalize_actions(predicted)
    expected_physical = Unnormalize({"actions": stats["actions"]}, use_quantiles=quantiles)({"actions": expected})["actions"]
    np.testing.assert_allclose(physical, expected_physical, rtol=1e-6, atol=1e-6)
    if width == 10:
        with pytest.raises(ValueError, match="state"):
            policy.predict_action_chunk({**observation, "observation.state": np.zeros(14, np.float32)}, "task")


def test_tactile_resize_matches_training_cache_float_resize():
    from train_baseline_pi05.tactile_cache import _raw_image_to_unit
    from deploy_baseline_pi05.tactile_encoder import FrozenTactileEncoder

    encoder = object.__new__(FrozenTactileEncoder)
    encoder.image_size = 224
    raw = np.random.default_rng(17).integers(0, 256, size=(137, 263, 3), dtype=np.uint8)
    np.testing.assert_array_equal(encoder._prepare_image(raw), _raw_image_to_unit(raw))
