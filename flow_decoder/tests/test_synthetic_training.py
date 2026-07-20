from __future__ import annotations

import unittest

import jax
import jax.numpy as jnp
from flax import nnx

from flow_decoder.utils.model import DecoderConfig
from flow_decoder.utils.model import SelfAttentionFlowDecoder
from flow_decoder.utils.model import flow_matching_loss_per_sample
from flow_decoder.utils.model import make_optimizer
from flow_decoder.utils.model import train_step


class SyntheticTrainingTest(unittest.TestCase):
    def test_paired_flow_loss_decreases(self):
        model = SelfAttentionFlowDecoder(
            DecoderConfig(action_dim=2, action_horizon=4, model_dim=16, depth=1, num_heads=4),
            rngs=nnx.Rngs(4),
        )
        optimizer = make_optimizer(model, learning_rate=3e-3, weight_decay=0.0)
        x_base = jax.random.normal(jax.random.key(5), (32, 4, 2))
        target = x_base + jnp.asarray([0.4, -0.2])[None, None, :]
        fixed_t = jnp.full((32,), 0.5)
        initial = float(jnp.mean(flow_matching_loss_per_sample(model, x_base, target, fixed_t)))
        key = jax.random.key(6)
        for step in range(40):
            train_step(model, optimizer, x_base, target, jax.random.fold_in(key, step))
        final = float(jnp.mean(flow_matching_loss_per_sample(model, x_base, target, fixed_t)))
        self.assertLess(final, initial * 0.5)


if __name__ == "__main__":
    unittest.main()
