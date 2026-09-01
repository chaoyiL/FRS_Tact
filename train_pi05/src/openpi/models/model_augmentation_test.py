import jax
import jax.numpy as jnp

from openpi.models import model as _model


def _observation() -> _model.Observation:
    image = jnp.linspace(-1.0, 1.0, 2 * 8 * 8 * 3, dtype=jnp.float32).reshape(2, 8, 8, 3)
    return _model.Observation(
        images={"left": image, "right": image},
        image_masks={"left": jnp.ones(2, dtype=jnp.bool_), "right": jnp.ones(2, dtype=jnp.bool_)},
        state=jnp.zeros((2, 7), dtype=jnp.float32),
    )


def test_balanced_light_v2_is_jittable_and_shared_across_cameras():
    observation = _observation()
    preprocess = jax.jit(
        lambda rng, obs: _model.preprocess_observation(
            rng,
            obs,
            train=True,
            image_keys=("left", "right"),
            image_resolution=(8, 8),
            image_augmentation="balanced-light-v2",
        )
    )

    output = preprocess(jax.random.key(7), observation)

    assert output.images["left"].shape == (2, 8, 8, 3)
    assert jnp.allclose(output.images["left"], output.images["right"])
    assert jnp.all(output.images["left"] >= -1.0)
    assert jnp.all(output.images["left"] <= 1.0)


def test_balanced_light_v2_is_disabled_for_validation():
    observation = _observation()
    output = _model.preprocess_observation(
        None,
        observation,
        train=False,
        image_keys=("left", "right"),
        image_resolution=(8, 8),
        image_augmentation="balanced-light-v2",
    )

    assert jnp.array_equal(output.images["left"], observation.images["left"])
    assert jnp.array_equal(output.images["right"], observation.images["right"])
