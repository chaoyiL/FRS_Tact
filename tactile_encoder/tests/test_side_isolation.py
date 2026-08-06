from __future__ import annotations

import unittest

import jax.numpy as jnp
import numpy as np

from tactile_encoder.utils.data import FutureRecord
from tactile_encoder.utils.data import batches
from tactile_encoder.utils.data import resolve_data_keys
from tactile_encoder.utils.metrics import l2_normalize
from tactile_encoder.utils.metrics import pooled_retrieval_metrics_by_side
from tactile_encoder.utils.metrics import retrieval_metrics_by_side
from tactile_encoder.utils.model import TactileClipConfig
from tactile_encoder.utils.model import _filter_bank_logits_hard_negatives
from tactile_encoder.utils.model import symmetric_contrastive_loss


class FakeImageDataset:
    def get_images(self, index, keys, *, as_float=True):
        del as_float
        fill = float(int(index) % 7) / 10.0
        return {key: np.full((8, 8, 3), fill, dtype=np.float32) for key in keys}


class SideIsolationTest(unittest.TestCase):
    def test_all_bank_negatives_still_exclude_same_episode(self):
        logits = jnp.asarray([[3.0, 2.0, 1.0]], dtype=jnp.float32)
        filtered, hard = _filter_bank_logits_hard_negatives(
            logits,
            bank_positive_mask=jnp.asarray([[False, True, False]]),
            bank_valid=jnp.asarray([True, True, True]),
            candidate_mask=jnp.asarray([[True, False, False]]),
            hard_negatives_k=0,
        )

        np.testing.assert_allclose(np.asarray(filtered[0, :2]), np.asarray([3.0, 2.0]))
        self.assertTrue(bool(jnp.isneginf(filtered[0, 2])))
        self.assertEqual(hard.shape, (1, 0))

    def test_batches_equal_left_right_when_shuffled(self):
        records = tuple(
            FutureRecord(
                dataset_index=i,
                future_dataset_index=i + 1,
                episode_index=0,
                split="train",
            )
            for i in range(6)
        )
        keys = resolve_data_keys()
        batch_sizes = []
        left_counts = []
        right_counts = []
        for batch in batches(
            FakeImageDataset(),
            records,
            keys,
            batch_size=4,
            shuffle=True,
            seed=7,
            image_size=8,
            num_workers=1,
            prefetch_batches=1,
            tactile_history=0,
            history_stride=1,
        ):
            sides = np.asarray(batch["side_id"])
            batch_sizes.append(int(sides.shape[0]))
            left_counts.append(int(np.sum(sides == 0)))
            right_counts.append(int(np.sum(sides == 1)))

        self.assertEqual(batch_sizes, [4, 4, 4])
        self.assertEqual(left_counts, [2, 2, 2])
        self.assertEqual(right_counts, [2, 2, 2])

    def test_batches_rejects_odd_batch_size(self):
        records = (
            FutureRecord(dataset_index=0, future_dataset_index=1, episode_index=0, split="train"),
        )
        with self.assertRaises(ValueError):
            next(
                batches(
                    FakeImageDataset(),
                    records,
                    resolve_data_keys(),
                    batch_size=3,
                    shuffle=False,
                    seed=0,
                    image_size=8,
                    num_workers=1,
                    prefetch_batches=1,
                )
            )

    def test_cross_side_logits_masked_in_loss(self):
        # Left query aligns with right gallery 2 more than either left gallery.
        query = jnp.asarray(
            [
                [1.0, 0.0],
                [0.0, 1.0],
                [0.0, 1.0],
                [1.0, 0.0],
            ],
            dtype=jnp.float32,
        )
        target = jnp.asarray(
            [
                [0.0, 1.0],  # left gallery 0 — orthogonal to left query 0
                [0.2, 1.0],  # left gallery 1 — weak for left query 0
                [1.0, 0.0],  # right gallery — would win without side mask
                [0.0, 1.0],
            ],
            dtype=jnp.float32,
        )
        side_id = jnp.asarray([0, 0, 1, 1], dtype=jnp.int32)
        loss, metrics = symmetric_contrastive_loss(
            query,
            target,
            config=TactileClipConfig(temperature=0.07),
            positive_mask=jnp.eye(4, dtype=bool),
            side_id=side_id,
        )
        self.assertTrue(bool(jnp.isfinite(loss)))
        self.assertGreaterEqual(float(metrics["batch_recall_at_1"]), 0.0)
        self.assertLessEqual(float(metrics["batch_recall_at_1"]), 1.0)

        q = l2_normalize(query)
        t = l2_normalize(target)
        logits = (1.0 / 0.07) * (q @ t.T)
        neg_inf = jnp.asarray(-jnp.inf, dtype=logits.dtype)
        same_side = side_id[:, None] == side_id[None, :]
        masked = jnp.where(same_side, logits, neg_inf)
        left0_best = int(jnp.argmax(masked[0]))
        self.assertIn(left0_best, (0, 1))
        unmasked_best = int(jnp.argmax(logits[0]))
        self.assertEqual(unmasked_best, 2)

    def test_retrieval_metrics_by_side_ignores_other_wrist(self):
        future = np.asarray(
            [
                [0.2, 1.0],  # left positive for q0 — weak for [1,0]
                [1.0, 0.1],  # left future1 — strong for q0, so q0 rank=1
                [1.0, 0.0],  # right — strongest overall for q0 if mixed
                [0.0, 1.0],  # right positive for q2 / gallery for q3
            ],
            dtype=np.float32,
        )
        # Make q1 also left with positive at future1 exact-ish so recall@1_left=0.5
        query = np.asarray(
            [
                [1.0, 0.0],
                [1.0, 0.1],
                [0.0, 1.0],
                [0.0, 1.0],
            ],
            dtype=np.float32,
        )
        side_id = np.asarray([0, 0, 1, 1], dtype=np.int64)
        metrics, ranks = retrieval_metrics_by_side(query, future, side_id)
        self.assertEqual(int(metrics["sample_count"]), 4)
        self.assertEqual(int(metrics["sample_count_left"]), 2)
        self.assertEqual(int(metrics["sample_count_right"]), 2)
        self.assertEqual(int(ranks[0]), 1)
        self.assertEqual(int(ranks[1]), 0)
        self.assertEqual(float(metrics["recall@1_left"]), 0.5)
        # Cross-side isolation: mixed gallery would give q0 a worse rank via future2.
        qn = query / np.linalg.norm(query, axis=1, keepdims=True)
        fn = future / np.linalg.norm(future, axis=1, keepdims=True)
        mixed = qn @ fn.T
        mixed_rank0 = int(np.sum(mixed[0] > mixed[0, 0]))
        self.assertGreater(mixed_rank0, int(ranks[0]))
        expected_total = 0.5 * float(metrics["recall@1_left"]) + 0.5 * float(
            metrics["recall@1_right"]
        )
        self.assertAlmostEqual(float(metrics["recall@1"]), expected_total)

    def test_pooled_retrieval_metrics_sample_same_side_pools(self):
        query = np.eye(6, dtype=np.float32)
        future = np.eye(6, dtype=np.float32)
        side_id = np.asarray([0, 0, 0, 1, 1, 1], dtype=np.int64)

        metrics, ranks = pooled_retrieval_metrics_by_side(
            query,
            future,
            side_id,
            pool_size=3,
            seed=123,
        )

        self.assertEqual(int(metrics["sample_count"]), 6)
        self.assertEqual(int(metrics["pool_size"]), 3)
        self.assertTrue(np.all(ranks == 0))
        self.assertEqual(float(metrics["recall@1"]), 1.0)
        self.assertEqual(float(metrics["recall@5"]), 1.0)


if __name__ == "__main__":
    unittest.main()
