from __future__ import annotations

import pathlib
import tempfile
import unittest

import jax
import jax.numpy as jnp
from flax import nnx

from utils.checkpoint import load_checkpoint
from utils.checkpoint import save_checkpoint
from utils.model import DecoderConfig
from utils.model import SelfAttentionFlowDecoder
from utils.model import decode_actions
from utils.model import decode_euler
from utils.model import flow_matching_loss_per_sample
from utils.model import make_optimizer
from utils.model import train_step


class DecoderModelTest(unittest.TestCase):
    def make_model(self) -> SelfAttentionFlowDecoder:
        return SelfAttentionFlowDecoder(
            DecoderConfig(action_dim=3, action_horizon=5, model_dim=16, depth=1, num_heads=4),
            rngs=nnx.Rngs(0),
        )

    def test_shape_finite_gradient_and_decode(self):
        model = self.make_model()
        x_base = jax.random.normal(jax.random.key(1), (4, 5, 3))
        target = x_base + 0.25
        t = jnp.linspace(0.1, 0.9, 4)
        loss = flow_matching_loss_per_sample(model, x_base, target, t)
        decoded = decode_euler(model, x_base, num_steps=4)
        decoded_fireflow = decode_actions(model, x_base, num_steps=4, solver="fireflow")
        self.assertEqual(loss.shape, (4,))
        self.assertEqual(decoded.shape, target.shape)
        self.assertEqual(decoded_fireflow.shape, target.shape)
        self.assertTrue(bool(jnp.all(jnp.isfinite(loss))))
        self.assertTrue(bool(jnp.all(jnp.isfinite(decoded_fireflow))))
        optimizer = make_optimizer(model, learning_rate=1e-3, weight_decay=0.0)
        step_loss = train_step(model, optimizer, x_base, target, jax.random.key(2))
        self.assertTrue(bool(jnp.isfinite(step_loss)))

    def test_attention_is_not_causal(self):
        model = self.make_model()
        original = jnp.zeros((1, 5, 3), dtype=jnp.float32)
        changed = original.at[:, -1, :].set(10.0)
        t = jnp.asarray([0.5], dtype=jnp.float32)
        first_original = model(original, t)[:, 0, :]
        first_changed = model(changed, t)[:, 0, :]
        self.assertGreater(float(jnp.max(jnp.abs(first_original - first_changed))), 1e-7)

    def test_checkpoint_round_trip(self):
        model = self.make_model()
        x = jnp.ones((2, 5, 3), dtype=jnp.float32)
        t = jnp.asarray([0.25, 0.75], dtype=jnp.float32)
        expected = model(x, t)
        with tempfile.TemporaryDirectory() as directory:
            checkpoint_dir = pathlib.Path(directory)
            save_checkpoint(checkpoint_dir, model, epoch=3, metrics={"val_mse": 0.5})
            restored, metadata = load_checkpoint(checkpoint_dir)
            self.assertTrue(jnp.array_equal(expected, restored(x, t)))
            self.assertEqual(metadata["epoch"], 3)


if __name__ == "__main__":
    unittest.main()
