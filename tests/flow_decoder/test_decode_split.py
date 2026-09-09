from __future__ import annotations

import unittest

import numpy as np

from decode_tests.split import indices_for_episodes
from decode_tests.split import split_episodes_three_way
from decode_tests.train import build_parser
from utils.model import resolve_peak_learning_rate


class DecodeEpisodeSplitTest(unittest.TestCase):
    def test_split_is_episode_disjoint_and_deterministic(self):
        episode_rows = np.repeat(np.arange(10), 3)
        train_a, val_a, test_a = split_episodes_three_way(episode_rows, seed=7)
        train_b, val_b, test_b = split_episodes_three_way(episode_rows, seed=7)

        self.assertEqual((train_a, val_a, test_a), (train_b, val_b, test_b))
        self.assertEqual((len(train_a), len(val_a), len(test_a)), (8, 1, 1))
        self.assertFalse(set(train_a) & set(val_a))
        self.assertFalse(set(train_a) & set(test_a))
        self.assertFalse(set(val_a) & set(test_a))
        self.assertEqual(set(train_a) | set(val_a) | set(test_a), set(range(10)))

        train_indices = indices_for_episodes(episode_rows, train_a)
        val_indices = indices_for_episodes(episode_rows, val_a)
        test_indices = indices_for_episodes(episode_rows, test_a)
        self.assertFalse(set(train_indices) & set(val_indices))
        self.assertFalse(set(train_indices) & set(test_indices))
        self.assertFalse(set(val_indices) & set(test_indices))
        self.assertEqual(
            len(train_indices) + len(val_indices) + len(test_indices),
            len(episode_rows),
        )

    def test_training_defaults_match_legacy_decoder(self):
        args = build_parser().parse_args(
            ["--cache-dir", "cache", "--output-dir", "output"]
        )
        self.assertEqual(args.model_dim, 256)
        self.assertEqual(args.depth, 6)
        self.assertEqual(args.num_heads, 4)
        self.assertEqual(args.mlp_ratio, 4)
        self.assertEqual(args.lr_reference_dim, 256)
        self.assertEqual(args.batch_size, 256)
        self.assertEqual(args.epochs, 1000)

    def test_learning_rate_scales_from_reference_width(self):
        self.assertAlmostEqual(
            resolve_peak_learning_rate(3e-4, model_dim=256, lr_reference_dim=256),
            3e-4,
        )
        self.assertAlmostEqual(
            resolve_peak_learning_rate(3e-4, model_dim=128, lr_reference_dim=256),
            3e-4 * np.sqrt(2.0),
        )


if __name__ == "__main__":
    unittest.main()
