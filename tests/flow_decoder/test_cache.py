from __future__ import annotations

import unittest
import pathlib
import tempfile

import numpy as np

from utils.cache import CACHE_VERSION
from utils.cache import SampleRecord
from utils.cache import atomic_write_json
from utils.cache import create_cache_arrays
from utils.cache import finalize_partial_cache
from utils.cache import flush_arrays
from utils.cache import limit_records
from utils.cache import load_manifest
from utils.cache import open_cache_arrays
from utils.cache import split_episodes


class EpisodeSplitTest(unittest.TestCase):
    def test_split_is_disjoint_deterministic_and_eighty_twenty(self):
        train_a, val_a = split_episodes(range(10), val_fraction=0.2, seed=7)
        train_b, val_b = split_episodes(range(10), val_fraction=0.2, seed=7)
        self.assertEqual((train_a, val_a), (train_b, val_b))
        self.assertEqual(len(train_a), 8)
        self.assertEqual(len(val_a), 2)
        self.assertFalse(set(train_a) & set(val_a))
        self.assertEqual(set(train_a) | set(val_a), set(range(10)))

    def test_sample_limit_keeps_both_splits(self):
        records = [SampleRecord(i, i // 10, "train" if i < 80 else "val") for i in range(100)]
        selected = limit_records(records, max_samples=10, seed=3)
        self.assertEqual(len(selected), 10)
        self.assertIn("train", {record.split for record in selected})
        self.assertIn("val", {record.split for record in selected})
        self.assertEqual(selected, sorted(selected, key=lambda record: record.dataset_index))

    def test_incomplete_memmap_cache_can_be_reopened_for_resume(self):
        records = [SampleRecord(10, 0, "train"), SampleRecord(20, 1, "val")]
        with tempfile.TemporaryDirectory() as directory:
            cache_dir = pathlib.Path(directory)
            arrays = create_cache_arrays(cache_dir, records, action_horizon=2, action_dim=3)
            arrays["x_base"][0] = 7.0
            flush_arrays(arrays)
            atomic_write_json(
                cache_dir / "manifest.json",
                {
                    "version": CACHE_VERSION,
                    "status": "incomplete",
                    "completed_samples": 1,
                    "sample_count": 2,
                },
            )
            manifest = load_manifest(cache_dir, require_complete=False)
            reopened = open_cache_arrays(cache_dir, mode="r+")
            self.assertEqual(manifest["completed_samples"], 1)
            self.assertTrue(np.all(reopened["x_base"][0] == 7.0))
            reopened["x_base"][1] = 9.0
            flush_arrays(reopened)
            self.assertTrue(np.all(np.load(cache_dir / "x_base.npy")[1] == 9.0))

    def test_finalize_partial_cache_truncates_and_resplits(self):
        records = [
            SampleRecord(10, 0, "train"),
            SampleRecord(20, 0, "train"),
            SampleRecord(30, 1, "val"),
            SampleRecord(40, 2, "train"),
        ]
        with tempfile.TemporaryDirectory() as directory:
            cache_dir = pathlib.Path(directory)
            arrays = create_cache_arrays(cache_dir, records, action_horizon=2, action_dim=3)
            arrays["x_base"][:] = np.arange(24, dtype=np.float32).reshape(4, 2, 3)
            arrays["target"][:] = arrays["x_base"][:] + 1.0
            arrays["inversion_mse"][:] = np.asarray([1.0, 2.0, 3.0, 0.0], dtype=np.float32)
            flush_arrays(arrays)
            atomic_write_json(
                cache_dir / "manifest.json",
                {
                    "version": CACHE_VERSION,
                    "status": "incomplete",
                    "completed_samples": 3,
                    "sample_count": 4,
                    "configuration": {"val_fraction": 0.5, "split_seed": 0},
                    "train_episodes": [0, 2],
                    "val_episodes": [1],
                },
            )

            manifest = finalize_partial_cache(cache_dir, resplit=True)
            reopened = open_cache_arrays(cache_dir)

            self.assertEqual(manifest["status"], "complete")
            self.assertEqual(manifest["sample_count"], 3)
            self.assertEqual(manifest["completed_samples"], 3)
            self.assertEqual(reopened["x_base"].shape[0], 3)
            self.assertEqual(manifest["train_sample_count"] + manifest["val_sample_count"], 3)
            self.assertEqual(set(manifest["train_episodes"]) & set(manifest["val_episodes"]), set())
            self.assertAlmostEqual(manifest["mean_source_inversion_mse"], 2.0)


if __name__ == "__main__":
    unittest.main()
