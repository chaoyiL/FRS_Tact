from __future__ import annotations

from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
import pytest

from deploy_smolvla.remote_client import (
    _prepare_observation,
    _remaining_action_chunk,
    _robot_image_keys,
    _robot_tactile_keys,
    _validate_observation_mode,
)
from lerobot.policies.smolvla_jax.configuration import JaxSmolVLAConfig
from lerobot.policies.smolvla_jax.modeling import JaxSmolVLA, normalize_tactile_embeddings
from lerobot.policies.smolvla_jax.policy import JaxSmolVLAPolicy


def _tactile_policy():
    config = SimpleNamespace(
        image_keys=("observation.images.camera1", "observation.images.camera2"),
        use_tactile_encoder=True,
        tactile_keys=(
            "observation.images.tactile_left_0",
            "observation.images.tactile_right_0",
        ),
    )
    return SimpleNamespace(config=config)


def test_tactile_embedding_normalization_has_unit_rms() -> None:
    embeddings = jnp.asarray([[[1.0, 2.0, 3.0], [2.0, 4.0, 8.0]]])
    normalized = normalize_tactile_embeddings(embeddings)
    rms = jnp.sqrt(jnp.mean(jnp.square(normalized), axis=-1))
    np.testing.assert_allclose(rms, np.ones((1, 2)), rtol=1e-5, atol=1e-5)


def test_cached_tactile_embeddings_keep_trainable_projection() -> None:
    config = JaxSmolVLAConfig(
        use_tactile_encoder=True,
        tactile_encoder_path="unused-for-cached-input",
        tactile_keys=("left", "right"),
        tactile_num_tokens=2,
        tactile_embedding_dim=3,
        text_hidden_size=4,
    )
    model = JaxSmolVLA(config)
    params = {
        "model.tactile_proj.weight": jnp.arange(12, dtype=jnp.float32).reshape(4, 3) / 10,
        "model.tactile_proj.bias": jnp.arange(4, dtype=jnp.float32),
    }
    cached = jnp.asarray([[[1.0, 2.0, 3.0], [2.0, 1.0, 4.0]]], dtype=jnp.float16)
    actual = model.embed_tactile(params, tactile_embeddings=cached)
    expected = (
        normalize_tactile_embeddings(cached) @ params["model.tactile_proj.weight"].T
        + params["model.tactile_proj.bias"]
    )
    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)


def test_remote_observation_keeps_and_requires_tactile_images() -> None:
    policy = _tactile_policy()
    rename_map = {
        "observation.images.camera0": "observation.images.camera1",
        "observation.images.camera1": "observation.images.camera2",
    }
    image_keys = _robot_image_keys(policy, rename_map)
    tactile_keys = _robot_tactile_keys(policy, rename_map)
    observation = {
        "observation.state": np.zeros((20,), dtype=np.float32),
        **{key: np.zeros((8, 8, 3), dtype=np.uint8) for key in image_keys},
    }

    prepared = _prepare_observation(
        observation,
        state_dim=20,
        image_keys=image_keys,
        empty_cameras=0,
        required_image_keys=tactile_keys,
    )

    assert image_keys == (
        "observation.images.camera0",
        "observation.images.camera1",
        "observation.images.tactile_left_0",
        "observation.images.tactile_right_0",
    )
    assert set(tactile_keys).issubset(prepared)

    observation.pop(tactile_keys[0])
    with pytest.raises(ValueError, match="required tactile"):
        _prepare_observation(
            observation,
            state_dim=20,
            image_keys=image_keys,
            empty_cameras=0,
            required_image_keys=tactile_keys,
        )


def test_remote_observation_mode_must_match_checkpoint() -> None:
    _validate_observation_mode("vitac", use_tactile_encoder=True)
    _validate_observation_mode("vision", use_tactile_encoder=False)

    with pytest.raises(ValueError, match="requires observation.data_type='vitac'"):
        _validate_observation_mode("vision", use_tactile_encoder=True)
    with pytest.raises(ValueError, match="requires observation.data_type='vision'"):
        _validate_observation_mode("vitac", use_tactile_encoder=False)


def test_rtc_previous_chunk_is_shifted_by_executed_steps() -> None:
    chunk = np.arange(20, dtype=np.float32).reshape(10, 2)
    remaining = _remaining_action_chunk(chunk, 3)

    np.testing.assert_array_equal(remaining, chunk[3:])
    assert remaining.shape == (7, 2)
    assert not np.shares_memory(remaining, chunk)


def test_policy_inference_accepts_cached_tactile_embeddings() -> None:
    captured = {}

    class FakeModel:
        def sample_actions(self, params, *args, **kwargs):
            del params, args
            captured.update(kwargs)
            return jnp.zeros((1, 2, 1), dtype=jnp.float32)

    class FakePreprocessor:
        def prepare(self, observation, task):
            del observation, task
            return {
                "images": jnp.zeros((1, 1, 2, 2, 3)),
                "image_masks": jnp.ones((1, 1), dtype=bool),
                "language_tokens": jnp.ones((1, 2), dtype=jnp.int32),
                "language_masks": jnp.ones((1, 2), dtype=bool),
                "state": jnp.zeros((1, 1)),
                "tactile_embeddings": jnp.ones((1, 2, 3)),
                "tactile_masks": jnp.ones((1, 2), dtype=bool),
            }

    policy = object.__new__(JaxSmolVLAPolicy)
    policy.config = SimpleNamespace(
        chunk_size=2,
        max_action_dim=1,
        num_steps=2,
        rtc_config=None,
        adapt_to_pi_aloha=False,
    )
    policy.params = {}
    policy.model = FakeModel()
    policy.preprocessor = FakePreprocessor()
    policy._compiled_samples = {}

    policy.predict_action_chunk({}, "task", jit=False, normalized=True)

    assert captured["tactile_images"] is None
    assert captured["tactile_embeddings"].shape == (1, 2, 3)


def test_select_action_advances_chunk_seed() -> None:
    policy = object.__new__(JaxSmolVLAPolicy)
    policy.config = SimpleNamespace(n_action_steps=1)
    seeds = []

    def predict(observation, task, *, seed, jit, **kwargs):
        del observation, task, jit, kwargs
        seeds.append(seed)
        return jnp.asarray([[seed]], dtype=jnp.float32)

    policy.predict_action_chunk = predict
    policy.reset()
    policy.select_action({}, "task", seed=10, jit=False)
    policy.select_action({}, "task", seed=10, jit=False)

    assert seeds == [10, 11]
