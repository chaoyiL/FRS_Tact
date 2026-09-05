import importlib
from types import SimpleNamespace

import numpy as np
import pytest

from train_baseline_pi05.source_model import validate_pi05_model


@pytest.mark.parametrize("width", (10, 20, 32))
def test_source_accepts_single_bimanual_and_padded_actions(width):
    model = SimpleNamespace(pi05=True, action_dim=width, action_horizon=50)
    assert validate_pi05_model(model, action_horizon=50) == width


def test_frozen_sampler_compiles_once_and_preserves_sampling_contract():
    import flax.nnx as nnx
    import jax
    import jax.numpy as jnp
    from train_baseline_pi05.source_model import make_frozen_sampler, sample_coarse_actions

    traced = []

    class Source(nnx.Module):
        def __init__(self):
            self.weight = nnx.Param(jnp.asarray(0.25))

        def sample_actions(self, rng, observation, *, noise, num_steps):
            traced.append(num_steps)
            return jax.lax.fori_loop(0, num_steps, lambda _, x: x + self.weight.value * observation, noise)

    model = Source()
    sampler = make_frozen_sampler(model)
    noise = jnp.ones((2, 50, 10))
    first = sample_coarse_actions(model, jax.random.key(0), jnp.asarray(2.0), noise, 2, sampler=sampler)
    second = sample_coarse_actions(model, jax.random.key(0), jnp.asarray(4.0), noise, 2, sampler=sampler)
    assert traced == [2]  # Different observations reuse the same compiled loop.
    np.testing.assert_array_equal(first, np.full((2, 50, 10), 2.0))
    np.testing.assert_array_equal(second, np.full((2, 50, 10), 3.0))
    different_steps = sample_coarse_actions(model, jax.random.key(0), jnp.asarray(2.0), noise, 3, sampler=sampler)
    assert traced == [2, 3]
    np.testing.assert_array_equal(different_steps, np.full((2, 50, 10), 2.5))


@pytest.mark.parametrize("action_dim,state_dim,cameras", (
    (10, 7, {"right_wrist_0_rgb": "observation.images.camera1"}),
    (20, 20, {"left_wrist_0_rgb": "observation.images.camera0", "right_wrist_0_rgb": "observation.images.camera1"}),
))
def test_processor_preserves_real_state_and_only_configured_cameras(monkeypatch, tmp_path, action_dim, state_dim, cameras):
    from train_baseline_pi05.runtime_path import activate_vendored_lerobot
    activate_vendored_lerobot()
    # Cache-producer tests temporarily import this module with fake runtimes.
    inputs = importlib.reload(importlib.import_module("train_baseline_pi05.policy_inputs"))
    from lerobot.policies.pi05_jax.normalize import NormStats

    metadata = SimpleNamespace(
        root=tmp_path, revision=None, fps=10,
        features={"observation.state": {"shape": [state_dim]}, "actions": {"shape": [action_dim]}},
    )
    monkeypatch.setattr(inputs, "LeRobotDatasetMetadata", lambda *a, **kw: metadata)
    states_seen = []

    class Tokenizer:
        def __init__(self, *a):
            pass

        def tokenize(self, prompt, state=None):
            states_seen.append(np.asarray(state))
            return np.zeros(200, dtype=np.int32), np.ones(200, dtype=bool)

    monkeypatch.setattr(inputs, "PaligemmaTokenizer", Tokenizer)

    def stats(dim):
        return NormStats(mean=np.zeros(dim), std=np.ones(dim), q01=-np.ones(dim), q99=np.ones(dim))

    processor = inputs.Pi05SampleProcessor(
        dataset_repo_id="local/test", dataset_root=tmp_path, action_key="actions",
        camera_map=cameras, state_stats=stats(state_dim), action_stats=stats(action_dim), action_dim=action_dim,
    )
    sample = {
        "observation.state": np.zeros(state_dim, dtype=np.float32),
        "actions": np.ones((50, action_dim), dtype=np.float32),
        "task": "test",
        **{name: np.zeros((3, 224, 224), dtype=np.float32) for name in cameras.values()},
    }
    obs, target, _ = processor.prepare_sample(sample)
    assert set(obs.images) == set(cameras)
    assert tuple(processor.config.image_keys) == tuple(cameras)
    assert target.shape == (50, action_dim)
    assert states_seen[0].shape == (state_dim,)
