from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest import mock

import jax
import jax.numpy as jnp
import numpy as np

from tactile_flow_steering.utils.data import TactileConditionedBatches
from tactile_flow_steering.utils.data import resolve_dataset_repo_id
from tactile_flow_steering.utils.data import tactile_token_dim_from_encoder


class FakePairs:
    def __init__(self):
        self.manifest = {
            "configuration": {"dataset_repo_id": "org/demo", "dataset_root": None},
        }
        self.arrays = {
            "dataset_index": np.asarray([10, 11, 12], dtype=np.int64),
            "x_base": np.zeros((3, 2, 3), dtype=np.float32),
            "target": np.zeros((3, 2, 3), dtype=np.float32),
            "gt_action": np.ones((3, 2, 3), dtype=np.float32),
            "split": np.asarray([0, 0, 1], dtype=np.int8),
        }

    def batches(self, split, *, batch_size, shuffle, seed):
        del shuffle, seed
        indices = np.asarray([0, 1] if split == "train" else [2], dtype=np.int64)
        for start in range(0, len(indices), batch_size):
            batch = indices[start : start + batch_size]
            yield (
                batch,
                self.arrays["x_base"][batch],
                self.arrays["target"][batch],
                self.arrays["gt_action"][batch],
            )


class DataHelpersTest(unittest.TestCase):
    def test_tactile_token_dim_from_encoder_no_history(self):
        bundle = SimpleNamespace(
            metadata={
                "tactile_clip_config": {
                    "embedding_dim": 512,
                    "tactile_image_count": 2,
                    "tactile_history": 0,
                    "gru_hidden_dim": 256,
                }
            }
        )
        self.assertEqual(tactile_token_dim_from_encoder(bundle), 1024)

    def test_resolve_dataset_repo_id(self):
        pairs = FakePairs()
        self.assertEqual(resolve_dataset_repo_id(pairs), "org/demo")
        self.assertEqual(resolve_dataset_repo_id(pairs, dataset_repo_id="other/id"), "other/id")

    def test_encode_indices_shape_and_stop_gradient(self):
        fake_dataset = mock.Mock()

        def get_images(index, keys, *, as_float=True):
            del as_float
            return {key: np.full((8, 8, 3), float(index % 5) / 10.0, dtype=np.float32) for key in keys}

        fake_dataset.get_images.side_effect = get_images

        class FakeBundle:
            def encode(self, tactile_images, *, train=False):
                del train
                # Depend on inputs so gradients would be nonzero without stop_gradient.
                reduced = jnp.mean(tactile_images, axis=(1, 2, 3, 4))
                return jnp.tile(reduced[:, None], (1, 4))

        conditioner = TactileConditionedBatches.__new__(TactileConditionedBatches)
        conditioner.pairs = FakePairs()
        conditioner.bundle = FakeBundle()
        conditioner.tactile_token_dim = 4
        conditioner.image_size = 8
        conditioner.dataset = fake_dataset
        conditioner.left_keys = ("tactile_left_0", "tactile_right_0")
        conditioner.right_keys = ("tactile_left_1", "tactile_right_1")

        tokens = conditioner.encode_indices([10, 11])
        self.assertEqual(tokens.shape, (2, 2, 4))

        images = jnp.asarray(np.random.default_rng(0).normal(size=(2, 2, 8, 8, 3)).astype(np.float32))
        grads = jax.grad(lambda x: jnp.sum(conditioner._encode_side(x)))(images)
        self.assertTrue(bool(jnp.allclose(grads, 0.0)))

        batch = next(conditioner.batches("train", batch_size=2, shuffle=False, seed=0))
        indices, x_base, gt_action, tactile_tokens = batch
        self.assertEqual(list(indices), [0, 1])
        self.assertEqual(x_base.shape, (2, 2, 3))
        self.assertEqual(gt_action.shape, (2, 2, 3))
        self.assertEqual(tactile_tokens.shape, (2, 2, 4))


if __name__ == "__main__":
    unittest.main()
