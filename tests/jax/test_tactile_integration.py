from __future__ import annotations

from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
import pytest

from deploy_smolvla.remote_client import _prepare_observation, _robot_image_keys, _robot_tactile_keys
from lerobot.policies.smolvla_jax.configuration import JaxSmolVLAConfig
from lerobot.policies.smolvla_jax.modeling import JaxSmolVLA, normalize_tactile_embeddings


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
