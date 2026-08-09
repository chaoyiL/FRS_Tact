from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from lerobot.policies.smolvla_jax.configuration import JaxSmolVLAConfig
from lerobot.policies.smolvla_jax.modeling import JaxSmolVLA
from lerobot.policies.smolvla_jax.training import JaxSmolVLATrainer


class TinyVisionModel(JaxSmolVLA):
    def __init__(self) -> None:
        super().__init__(
            JaxSmolVLAConfig(
                resize_height=32,
                resize_width=32,
                vision_patch_size=4,
                connector_scale_factor=1,
                text_hidden_size=4,
                max_state_dim=2,
                max_action_dim=2,
                action_dim=2,
                chunk_size=2,
                num_steps=1,
            )
        )
        self.fail_on_embed_image = False

    def embed_image(self, params, image):
        del params
        if self.fail_on_embed_image:
            raise AssertionError("cached rollout invoked embed_image")
        values = jnp.mean(image, axis=(1, 2, 3))[:, None, None]
        tokens = jnp.arange(64, dtype=jnp.float32)[None, :, None]
        channels = jnp.arange(self.config.text_hidden_size, dtype=jnp.float32)[
            None, None, :
        ]
        return values + tokens + channels

    def embed_language(self, params, tokens):
        del params
        return jnp.broadcast_to(
            tokens[..., None].astype(jnp.float32),
            (*tokens.shape, self.config.text_hidden_size),
        )

    def embed_suffix(self, params, noisy_actions, timestep):
        del params, timestep
        batch = noisy_actions.shape[0]
        embeddings = jnp.zeros(
            (batch, self.config.chunk_size, self.config.text_hidden_size),
            dtype=jnp.float32,
        )
        masks = jnp.ones((batch, self.config.chunk_size), dtype=jnp.bool_)
        return embeddings, masks, masks

    def transformer(
        self,
        params,
        prefix_hidden,
        expert_hidden,
        attention_mask,
        position_ids,
        *,
        cache=None,
        fill_cache=False,
    ):
        del params, attention_mask, position_ids, cache, fill_cache
        return prefix_hidden, expert_hidden, ()

    def _linear(self, params, name, value, *, bias=False, **kwargs):
        del params, bias, kwargs
        if name == "model.state_proj":
            return jnp.zeros(
                (value.shape[0], self.config.text_hidden_size), dtype=jnp.float32
            )
        if name == "model.action_out_proj":
            return jnp.zeros(
                (*value.shape[:-1], self.config.max_action_dim), dtype=jnp.float32
            )
        raise AssertionError(f"unexpected linear layer {name}")


@pytest.fixture
def model_inputs():
    images = jnp.arange(2 * 2 * 3 * 32 * 32, dtype=jnp.float32).reshape(
        2, 2, 3, 32, 32
    )
    return {
        "images": images,
        "image_masks": jnp.asarray([[True, False], [True, True]]),
        "language_tokens": jnp.asarray([[1, 2], [3, 4]], dtype=jnp.int32),
        "language_masks": jnp.ones((2, 2), dtype=jnp.bool_),
        "state": jnp.zeros((2, 2), dtype=jnp.float32),
    }


def _cached_tokens(model: TinyVisionModel, images: jax.Array) -> jax.Array:
    return jnp.stack(
        [model.embed_image({}, images[:, index]) for index in range(images.shape[1])],
        axis=1,
    ).astype(jnp.bfloat16)


def test_cached_vision_embeddings_match_live_prefix_exactly(model_inputs) -> None:
    model = TinyVisionModel()
    online_prefix = model.embed_prefix({}, **model_inputs)
    cached = jnp.stack(
        [
            model.embed_image({}, model_inputs["images"][:, index])
            for index in range(2)
        ],
        axis=1,
    )
    offline_inputs = dict(model_inputs)
    offline_inputs["images"] = None
    cached_prefix = model.embed_prefix(
        {}, **offline_inputs, vision_embeddings=cached
    )

    for online, offline in zip(online_prefix, cached_prefix, strict=True):
        np.testing.assert_array_equal(np.asarray(online), np.asarray(offline))


def test_live_and_cached_vision_sources_are_mutually_exclusive(model_inputs) -> None:
    model = TinyVisionModel()
    cached = _cached_tokens(model, model_inputs["images"])

    with pytest.raises(
        ValueError, match="exactly one of images or vision_embeddings is required"
    ):
        model.embed_prefix({}, **model_inputs, vision_embeddings=cached)

    missing_inputs = dict(model_inputs)
    missing_inputs["images"] = None
    with pytest.raises(
        ValueError, match="exactly one of images or vision_embeddings is required"
    ):
        model.embed_prefix({}, **missing_inputs)


@pytest.mark.parametrize(
    ("shape", "message"),
    [
        ((2, 1, 64, 4), "same number of cameras"),
        ((2, 2, 63, 4), "vision_embeddings must be"),
        ((2, 2, 64, 5), "vision_embeddings must be"),
    ],
)
def test_cached_vision_embeddings_validate_shape(
    model_inputs, shape: tuple[int, ...], message: str
) -> None:
    model = TinyVisionModel()
    offline_inputs = dict(model_inputs)
    offline_inputs["images"] = None

    with pytest.raises(ValueError, match=message):
        model.embed_prefix(
            {}, **offline_inputs, vision_embeddings=jnp.zeros(shape, dtype=jnp.bfloat16)
        )


def test_cached_vision_embeddings_produce_finite_loss(model_inputs) -> None:
    model = TinyVisionModel()
    batch = {key: value for key, value in model_inputs.items() if key != "images"}
    batch.update(
        vision_embeddings=_cached_tokens(model, model_inputs["images"]),
        actions=jnp.zeros((2, 2, 2), dtype=jnp.float32),
    )

    loss = model.loss(
        {},
        batch,
        jax.random.key(0),
        noise=jnp.ones((2, 2, 2), dtype=jnp.float32),
        time=jnp.full((2,), 0.5, dtype=jnp.float32),
    )

    assert bool(jnp.isfinite(loss))


def test_cached_validation_rollout_does_not_embed_images(model_inputs) -> None:
    model = TinyVisionModel()
    batch = {key: value for key, value in model_inputs.items() if key != "images"}
    batch.update(
        vision_embeddings=_cached_tokens(model, model_inputs["images"]),
        actions=jnp.zeros((2, 2, 2), dtype=jnp.float32),
    )
    model.fail_on_embed_image = True
    trainer = object.__new__(JaxSmolVLATrainer)
    trainer.model = model
    trainer.config = model.config

    metrics = trainer._eval_batch(
        {}, batch, jax.random.key(1), rollout=True, rollout_steps=1
    )

    assert bool(jnp.isfinite(metrics["loss"]))
    assert bool(jnp.isfinite(metrics["action_mse"]))
