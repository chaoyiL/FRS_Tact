def test_visual_package_is_discoverable():
    import importlib.util

    assert importlib.util.find_spec("train_smolvla") is not None


def test_visual_package_does_not_load_tactile_modules():
    import sys

    import train_smolvla

    assert "tactile_encoder" not in sys.modules
    assert "train_vtsmolvla" not in sys.modules


def test_visual_config_has_no_tactile_fields():
    from dataclasses import fields

    from train_smolvla import JaxSmolVLAConfig

    assert not {field.name for field in fields(JaxSmolVLAConfig) if "tactile" in field.name}


def test_visual_sample_actions_forwards_state_mask_contract():
    import jax.numpy as jnp
    import pytest

    from train_smolvla.modeling import JaxSmolVLA

    model = object.__new__(JaxSmolVLA)
    captured = {}

    def capture_context(*args, **kwargs):
        captured.update(kwargs)
        raise RuntimeError("captured prefix arguments")

    model.build_prefix_context = capture_context
    state_mask = jnp.asarray([False], dtype=jnp.bool_)
    with pytest.raises(RuntimeError, match="captured prefix arguments"):
        model.sample_actions(
            {},
            jnp.zeros((1, 1, 3, 2, 2)),
            jnp.ones((1, 1), dtype=jnp.bool_),
            jnp.zeros((1, 1), dtype=jnp.int32),
            jnp.ones((1, 1), dtype=jnp.bool_),
            jnp.zeros((1, 1)),
            jnp.zeros((2,), dtype=jnp.uint32),
            state_mask=state_mask,
            noise=jnp.zeros((1, 1, 1)),
        )

    assert captured["state_mask"] is state_mask
