"""Tests for utils.cache.build_records.

`build_records` was moved out of the per-model prepare scripts into utils/cache.py so every
action-cache producers share one record-selection/split implementation instead of drifting apart.
It had no coverage before that move. These tests need only numpy + stdlib (utils/cache.py pulls in
nothing else), so they run anywhere -- unlike tests/jax/*.

The duck-typed `metadata` contract is the point of the move, so it is covered explicitly: real
`LeRobotDatasetMetadata` stores the episode bounds as numpy arrays, while an independently
installed upstream `lerobot` (or a test double) may use plain ints. Both must work.
"""

from __future__ import annotations

import unittest

import numpy as np

from utils.cache import build_records


class FakeMetadata:
    """Minimal stand-in: only what build_records' docstring says it needs."""

    def __init__(self, lengths, *, as_numpy: bool = False):
        self.total_episodes = len(lengths)
        self.episodes = []
        start = 0
        for length in lengths:
            bounds = {"dataset_from_index": start, "dataset_to_index": start + length}
            if as_numpy:
                # Matches how the real LeRobotDatasetMetadata stores these.
                bounds = {key: np.asarray([value]) for key, value in bounds.items()}
            self.episodes.append(bounds)
            start += length


DEFAULTS = {
    "val_fraction": 0.1,
    "split_seed": 42,
    "frame_stride": 5,
    "max_episodes": None,
    "max_samples": None,
    "action_horizon": 50,
    "drop_tail_action_chunks": 1,
}


def build(metadata, **overrides):
    return build_records(metadata, **{**DEFAULTS, **overrides})


class BuildRecordsTest(unittest.TestCase):
    def test_sample_count_matches_stride_after_dropping_tail(self):
        # 10 episodes x 390 frames, horizon 50, drop 1 chunk -> 340 usable, stride 5 -> 68 each.
        records, _, _ = build(FakeMetadata([390] * 10))
        self.assertEqual(len(records), 68 * 10)

    def test_numpy_and_int_metadata_agree(self):
        """The duck-typed contract: array-valued bounds must behave like plain ints."""
        plain, _, _ = build(FakeMetadata([390] * 10))
        arrays, _, _ = build(FakeMetadata([390] * 10, as_numpy=True))
        self.assertEqual(
            [record.dataset_index for record in plain],
            [record.dataset_index for record in arrays],
        )

    def test_train_val_episode_disjoint_and_labels_consistent(self):
        records, train_episodes, val_episodes = build(FakeMetadata([390] * 10))
        self.assertTrue(train_episodes and val_episodes)
        self.assertFalse(set(train_episodes) & set(val_episodes))
        for record in records:
            expected = train_episodes if record.split == "train" else val_episodes
            self.assertIn(record.episode_index, expected)

    def test_deterministic_for_a_given_seed(self):
        first, train_a, val_a = build(FakeMetadata([390] * 10))
        second, train_b, val_b = build(FakeMetadata([390] * 10))
        self.assertEqual(
            [record.dataset_index for record in first],
            [record.dataset_index for record in second],
        )
        self.assertEqual((train_a, val_a), (train_b, val_b))

    def test_dataset_indices_are_global_and_ordered_within_episode(self):
        """Indices must be dataset-global offsets, not per-episode frame numbers -- the caches key
        into the dataset by these."""
        records, _, _ = build(FakeMetadata([390] * 3), val_fraction=0.34)
        by_episode: dict[int, list[int]] = {}
        for record in records:
            by_episode.setdefault(record.episode_index, []).append(record.dataset_index)
        self.assertEqual(min(by_episode[0]), 0)
        self.assertEqual(min(by_episode[1]), 390)
        self.assertEqual(min(by_episode[2]), 780)
        for indices in by_episode.values():
            self.assertEqual(indices, sorted(indices))

    def test_episodes_shorter_than_dropped_tail_are_skipped_not_fatal(self):
        records, _, _ = build(FakeMetadata([390] * 8 + [30, 30]), val_fraction=0.2, split_seed=1)
        self.assertEqual(sorted({record.episode_index for record in records}), list(range(8)))

    def test_frame_stride_of_one_keeps_every_usable_frame(self):
        records, _, _ = build(FakeMetadata([390] * 10), frame_stride=1)
        self.assertEqual(len(records), (390 - 50) * 10)

    def test_drop_tail_zero_keeps_episode_tails(self):
        records, _, _ = build(FakeMetadata([390] * 10), frame_stride=1, drop_tail_action_chunks=0)
        self.assertEqual(len(records), 390 * 10)

    def test_max_samples_subsamples_but_keeps_both_splits(self):
        records, _, _ = build(FakeMetadata([390] * 10), max_samples=100)
        self.assertEqual(len(records), 100)
        splits = {record.split for record in records}
        self.assertEqual(splits, {"train", "val"})

    def test_max_episodes_limits_episodes(self):
        records, train_episodes, val_episodes = build(FakeMetadata([390] * 10), max_episodes=4)
        self.assertLessEqual(len(set(train_episodes) | set(val_episodes)), 4)
        self.assertTrue(all(record.episode_index < 4 for record in records))

    def test_rejects_invalid_arguments(self):
        for overrides in (
            {"frame_stride": 0},
            {"frame_stride": -1},
            {"max_episodes": 1},
            {"action_horizon": 0},
            {"drop_tail_action_chunks": -1},
            {"val_fraction": 0.0},
            {"val_fraction": 1.0},
        ):
            with self.subTest(**overrides), self.assertRaises(ValueError):
                build(FakeMetadata([390] * 10), **overrides)

    def test_rejects_single_episode_dataset(self):
        with self.assertRaises(ValueError):
            build(FakeMetadata([390]))

    def test_raises_when_every_episode_is_dropped(self):
        with self.assertRaises(ValueError):
            build(FakeMetadata([30] * 10))


if __name__ == "__main__":
    unittest.main()
