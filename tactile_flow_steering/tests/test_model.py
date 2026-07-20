from __future__ import annotations

import pathlib
import tempfile
import unittest

import jax
import jax.numpy as jnp
from flax import nnx

from tactile_flow_steering.utils.checkpoint import load_checkpoint
from tactile_flow_steering.utils.checkpoint import save_checkpoint
from tactile_flow_steering.utils.model import DecoderConfig
from tactile_flow_steering.utils.model import TactileConditionedFlowDecoder
from tactile_flow_steering.utils.model import decode_actions
from tactile_flow_steering.utils.model import decode_euler
from tactile_flow_steering.utils.model import flow_matching_loss_per_sample
from tactile_flow_steering.utils.model import make_optimizer
from tactile_flow_steering.utils.model import train_step


class ConditionedDecoderModelTest(unittest.TestCase):
    def make_model(self) -> TactileConditionedFlowDecoder:
        return TactileConditionedFlowDecoder(
            DecoderConfig(
                action_dim=3,
                action_horizon=5,
                tactile_token_dim=8,
                model_dim=16,
                depth=2,
                num_heads=4,
            ),
            rngs=nnx.Rngs(0),
        )

    def test_shape_finite_gradient_and_decode(self):
        model = self.make_model()
        x_base = jax.random.normal(jax.random.key(1), (4, 5, 3))
        target = x_base + 0.25
        tactile = jax.random.normal(jax.random.key(3), (4, 2, 8))
        t = jnp.linspace(0.1, 0.9, 4)
        loss = flow_matching_loss_per_sample(model, x_base, target, t, tactile)
        decoded = decode_euler(model, x_base, tactile, num_steps=4)
        decoded_fireflow = decode_actions(
            model, x_base, tactile, num_steps=4, solver="fireflow"
        )
        self.assertEqual(loss.shape, (4,))
        self.assertEqual(decoded.shape, target.shape)
        self.assertEqual(decoded_fireflow.shape, target.shape)
        self.assertTrue(bool(jnp.all(jnp.isfinite(loss))))
        self.assertTrue(bool(jnp.all(jnp.isfinite(decoded_fireflow))))
        optimizer = make_optimizer(model, learning_rate=1e-3, weight_decay=0.0)
        step_loss = train_step(model, optimizer, x_base, target, tactile, jax.random.key(2))
        self.assertTrue(bool(jnp.isfinite(step_loss)))

    def test_tactile_tokens_change_output(self):
        model = self.make_model()
        x_t = jax.random.normal(jax.random.key(4), (2, 5, 3))
        t = jnp.asarray([0.3, 0.7], dtype=jnp.float32)
        tactile_a = jax.random.normal(jax.random.key(5), (2, 2, 8))
        tactile_b = tactile_a + 5.0
        velocity_a = model(x_t, t, tactile_a)
        velocity_b = model(x_t, t, tactile_b)
        self.assertGreater(float(jnp.max(jnp.abs(velocity_a - velocity_b))), 1e-4)

        x_base = jax.random.normal(jax.random.key(6), (2, 5, 3))
        decoded_a = decode_euler(model, x_base, tactile_a, num_steps=3)
        decoded_b = decode_euler(model, x_base, tactile_b, num_steps=3)
        self.assertGreater(float(jnp.max(jnp.abs(decoded_a - decoded_b))), 1e-4)

    def test_checkpoint_round_trip(self):
        model = self.make_model()
        x = jnp.ones((2, 5, 3), dtype=jnp.float32)
        t = jnp.asarray([0.25, 0.75], dtype=jnp.float32)
        tactile = jnp.ones((2, 2, 8), dtype=jnp.float32)
        expected = model(x, t, tactile)
        with tempfile.TemporaryDirectory() as directory:
            checkpoint_dir = pathlib.Path(directory)
            save_checkpoint(checkpoint_dir, model, epoch=3, metrics={"val_mse": 0.5})
            restored, metadata = load_checkpoint(checkpoint_dir)
            self.assertTrue(jnp.array_equal(expected, restored(x, t, tactile)))
            self.assertEqual(metadata["epoch"], 3)
            self.assertEqual(metadata["decoder_config"]["tactile_token_dim"], 8)


if __name__ == "__main__":
    unittest.main()
