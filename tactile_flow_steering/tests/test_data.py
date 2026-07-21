from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest import mock

import jax
import jax.numpy as jnp
import numpy as np

from tactile_flow_steering.utils.data import NUM_TACTILE_STREAMS
from tactile_flow_steering.utils.data import TACTILE_KEYS
from tactile_flow_steering.utils.data import TactileConditionedBatches
from tactile_flow_steering.utils.data import gate_weights_from_change
from tactile_flow_steering.utils.data import load_tactile_windows
from tactile_flow_steering.utils.data import resolve_dataset_repo_id
from tactile_flow_steering.utils.data import resolve_tactile_window
from tactile_flow_steering.utils.data import resnet_embedding_dim_from_encoder
from tactile_flow_steering.utils.data import tactile_change_from_tokens
from tactile_flow_steering.utils.data import window_frame_indices


class FakePairs:
    def __init__(self):
        self.manifest = {
            "configuration": {"dataset_repo_id": "org/demo", "dataset_root": None},
            "action_horizon": 6,
        }
        self.arrays = {
            "dataset_index": np.asarray([10, 11, 12], dtype=np.int64),
            "episode_index": np.asarray([0, 0, 0], dtype=np.int64),
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
    def test_resolve_tactile_window(self):
        self.assertEqual(resolve_tactile_window(action_horizon=50, window_divisor=1), 50)
        self.assertEqual(resolve_tactile_window(action_horizon=50, window_divisor=2), 25)
        with self.assertRaises(ValueError):
            resolve_tactile_window(action_horizon=50, window_divisor=3)

    def test_resnet_embedding_dim_from_encoder(self):
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
        self.assertEqual(resnet_embedding_dim_from_encoder(bundle), 512)

    def test_resolve_dataset_repo_id(self):
        pairs = FakePairs()
        self.assertEqual(resolve_dataset_repo_id(pairs), "org/demo")
        self.assertEqual(resolve_dataset_repo_id(pairs, dataset_repo_id="other/id"), "other/id")

    def test_window_frame_indices_clamps_to_episode_start(self):
        fake_dataset = mock.Mock()
        fake_dataset.indices_for_episode.return_value = (100, 101, 102, 103, 104)
        indices = window_frame_indices(
            fake_dataset,
            dataset_index=102,
            episode_index=0,
            window=4,
            history_stride=1,
        )
        self.assertEqual(indices, (100, 100, 101, 102))

    def test_gate_weights_from_change(self):
        low = gate_weights_from_change(np.asarray([0.0], dtype=np.float32), tau=0.5, temperature=0.1)
        high = gate_weights_from_change(np.asarray([1.0], dtype=np.float32), tau=0.5, temperature=0.1)
        mid = gate_weights_from_change(np.asarray([0.5], dtype=np.float32), tau=0.5, temperature=0.1)
        self.assertLess(float(low[0]), 0.05)
        self.assertGreater(float(high[0]), 0.95)
        self.assertAlmostEqual(float(mid[0]), 0.5, places=5)

    def test_tactile_change_identical_is_zero(self):
        tokens = np.random.default_rng(0).normal(size=(2, 4, 8)).astype(np.float32)
        change = tactile_change_from_tokens(tokens, tokens)
        self.assertTrue(np.allclose(change, 0.0, atol=1e-6))

    def test_load_tactile_windows_dedupes_frames(self):
        fake_dataset = mock.Mock()
        fake_dataset.indices_for_episode.return_value = tuple(range(0, 20))
        calls: list[int] = []

        def get_images(index, keys, *, as_float=True):
            calls.append(int(index))
            dtype = np.float32 if as_float else np.uint8
            fill = (index % 5) * (1.0 if as_float else 10)
            return {key: np.full((4, 4, 3), fill, dtype=dtype) for key in keys}

        fake_dataset.get_images.side_effect = get_images
        # Overlapping windows on contiguous frames should share decoded indices.
        images = load_tactile_windows(
            fake_dataset,
            [(12, 0), (13, 0)],
            tactile_window=3,
            history_stride=1,
            tactile_keys=TACTILE_KEYS,
            load_threads=1,
            as_float=False,
        )
        self.assertEqual(images.shape, (2, 3, 4, 4, 4, 3))
        self.assertEqual(images.dtype, np.uint8)
        self.assertEqual(sorted(calls), [10, 11, 12, 13])

    def test_encode_cache_indices_shape_and_stop_gradient(self):
        fake_dataset = mock.Mock()
        fake_dataset.indices_for_episode.return_value = tuple(range(0, 20))

        def get_images(index, keys, *, as_float=True):
            del as_float
            return {
                key: np.full((8, 8, 3), float(index % 5) / 10.0, dtype=np.float32)
                for key in keys
            }

        fake_dataset.get_images.side_effect = get_images

        conditioner = TactileConditionedBatches.__new__(TactileConditionedBatches)
        conditioner.pairs = FakePairs()
        conditioner.bundle = SimpleNamespace(params={"tactile_resnet": "unused"})
        conditioner.tactile_window = 3
        conditioner.history_stride = 1
        conditioner.encode_batch_size = 64
        conditioner.resnet_embedding_dim = 4
        conditioner.image_size = 8
        conditioner.dataset = fake_dataset
        conditioner.tactile_keys = TACTILE_KEYS
        conditioner.load_threads = 1
        conditioner.pipeline_prefetch = 1
        conditioner.num_workers = 0
        conditioner._mp_loader = None
        conditioner.episode_baselines = {
            0: np.ones((4, 4), dtype=np.float32),
        }

        def fake_encode(images):
            images = jnp.asarray(images)
            reduced = jnp.mean(images, axis=(1, 2, 3))
            return jnp.tile(reduced[:, None], (1, 4))

        conditioner._encode_images_frozen = lambda images: jax.lax.stop_gradient(
            fake_encode(images)
        )

        seq = conditioner.encode_cache_indices([0, 1])
        self.assertEqual(seq.shape, (2, 3, NUM_TACTILE_STREAMS, 4))

        images = jnp.asarray(
            np.random.default_rng(0).normal(size=(2, 8, 8, 3)).astype(np.float32)
        )
        grads = jax.grad(lambda x: jnp.sum(conditioner._encode_images_frozen(x)))(images)
        self.assertTrue(bool(jnp.allclose(grads, 0.0)))

        batch = next(conditioner.batches("train", batch_size=2, shuffle=False, seed=0))
        indices, x_base, predicted, gt_action, tactile_seq = batch
        self.assertEqual(list(indices), [0, 1])
        self.assertEqual(x_base.shape, (2, 2, 3))
        self.assertEqual(predicted.shape, (2, 2, 3))
        self.assertEqual(gt_action.shape, (2, 2, 3))
        self.assertEqual(tactile_seq.shape, (2, 3, 4, 4))

        weights = conditioner.gate_weights_for_cache_indices(
            indices,
            np.asarray(tactile_seq[:, -1, :, :]),
            tau=0.5,
            temperature=0.1,
        )
        self.assertEqual(weights.shape, (2,))


if __name__ == "__main__":
    unittest.main()
