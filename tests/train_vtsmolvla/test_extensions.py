from __future__ import annotations

import jax.numpy as jnp
import numpy as np


def test_vt_trainer_preserves_tactile_parameter_keys_through_portable_round_trip(
    tmp_path,
) -> None:
    from train_vtsmolvla.checkpoint import (
        load_safetensors_params,
        save_portable_params,
    )
    from train_vtsmolvla.configuration import VTSmolVLAConfig
    from train_vtsmolvla.modeling import VTJaxSmolVLA
    from train_vtsmolvla.training import VTJaxSmolVLATrainer

    config = VTSmolVLAConfig(
        use_tactile_encoder=True,
        tactile_encoder_path="unused-because-encoder-params-exist",
        tactile_keys=("touch",),
        tactile_num_tokens=1,
        tactile_embedding_dim=3,
        text_hidden_size=4,
    )
    encoder_key = "model.tactile_encoder.params/conv_init/kernel"
    trainer = VTJaxSmolVLATrainer(
        VTJaxSmolVLA(config),
        {encoder_key: jnp.ones((1,), dtype=jnp.float32)},
    )

    assert encoder_key in trainer.frozen_params
    assert "model.tactile_proj.weight" in trainer.state.params
    assert "model.tactile_proj.bias" in trainer.state.params

    destination = save_portable_params(trainer.full_params, tmp_path / "portable")
    restored = load_safetensors_params(destination)
    assert encoder_key in restored
    assert restored["model.tactile_proj.weight"].shape == (4, 3)
    assert restored["model.tactile_proj.bias"].shape == (4,)


def test_vt_model_explicit_tactile_arguments_forward_to_visual_prefix_seam(
    monkeypatch,
) -> None:
    from train_smolvla.modeling import JaxSmolVLA
    from train_vtsmolvla.configuration import VTSmolVLAConfig
    from train_vtsmolvla.modeling import VTJaxSmolVLA

    captured = {}

    def record_prefix(self, *args, prefix_inputs=None, **kwargs):
        del self, args, kwargs
        captured.update(prefix_inputs)
        return (
            jnp.zeros((1, 1, 4), dtype=jnp.float32),
            jnp.ones((1, 1), dtype=jnp.bool_),
            jnp.zeros((1, 1), dtype=jnp.bool_),
        )

    monkeypatch.setattr(JaxSmolVLA, "embed_prefix", record_prefix)
    model = VTJaxSmolVLA(
        VTSmolVLAConfig(
            use_tactile_encoder=True,
            tactile_keys=("touch",),
            tactile_num_tokens=1,
        )
    )
    tactile_embeddings = jnp.ones((1, 1, 3), dtype=jnp.float32)
    tactile_masks = jnp.ones((1, 1), dtype=jnp.bool_)

    model.embed_prefix(
        {},
        jnp.zeros((1, 1, 3, 2, 2), dtype=jnp.float32),
        jnp.ones((1, 1), dtype=jnp.bool_),
        jnp.ones((1, 1), dtype=jnp.int32),
        jnp.ones((1, 1), dtype=jnp.bool_),
        jnp.zeros((1, 1), dtype=jnp.float32),
        tactile_embeddings=tactile_embeddings,
        tactile_masks=tactile_masks,
    )

    np.testing.assert_array_equal(captured["tactile_embeddings"], tactile_embeddings)
    np.testing.assert_array_equal(captured["tactile_masks"], tactile_masks)
    assert captured["tactile_images"] is None
