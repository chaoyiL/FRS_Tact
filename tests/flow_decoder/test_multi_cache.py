from __future__ import annotations

import pathlib

import numpy as np

from utils.cache import (
    CACHE_VERSION,
    MultiCachedPairs,
    SampleRecord,
    atomic_write_json,
    create_cache_arrays,
    flush_arrays,
    records_digest,
)


def _write_cache(path: pathlib.Path, *, offset: int, train_count: int = 2) -> None:
    records = [
        *[
            SampleRecord(offset + 10 + index, 0, "train")
            for index in range(train_count)
        ],
        SampleRecord(offset + 20, 1, "val"),
    ]
    arrays = create_cache_arrays(path, records, action_horizon=2, action_dim=2)
    for row in range(len(records)):
        arrays["x_base"][row] = offset + row
        arrays["target"][row] = offset + row + 100
        arrays["gt_action"][row] = offset + row + 200
    flush_arrays(arrays)
    atomic_write_json(
        path / "manifest.json",
        {
            "version": CACHE_VERSION,
            "status": "complete",
            "completed_samples": len(records),
            "sample_count": len(records),
            "train_sample_count": train_count,
            "val_sample_count": 1,
            "action_horizon": 2,
            "action_dim": 2,
            "configuration": {"dataset_repo_id": f"demo/source_{offset}"},
            "records_sha256": records_digest(records),
        },
    )


def test_multi_cache_preserves_source_local_indices(tmp_path: pathlib.Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    _write_cache(first, offset=0)
    _write_cache(second, offset=1000)
    pairs = MultiCachedPairs([first, second], source_names=["demo/first", "demo/second"])

    assert pairs.manifest["sample_count"] == 6
    assert pairs.indices("train").tolist() == [0, 1, 3, 4]
    source, local = pairs.source_and_local_indices([0, 2, 3, 5])
    assert source.tolist() == [0, 0, 1, 1]
    assert local.tolist() == [0, 2, 0, 2]
    assert pairs.metadata_values([0, 3, 5], "dataset_index").tolist() == [10, 1010, 1020]

    indices, x_base, predicted, gt = next(pairs.batches("train", batch_size=4, shuffle=False, seed=0))
    assert indices.tolist() == [0, 1, 3, 4]
    np.testing.assert_array_equal(x_base[:, 0, 0], [0, 1, 1000, 1001])
    np.testing.assert_array_equal(predicted[:, 0, 0], [100, 101, 1100, 1101])
    np.testing.assert_array_equal(gt[:, 0, 0], [200, 201, 1200, 1201])


def test_multi_cache_source_balanced_batches_oversample_small_sources(
    tmp_path: pathlib.Path,
) -> None:
    large = tmp_path / "large"
    small = tmp_path / "small"
    _write_cache(large, offset=0, train_count=4)
    _write_cache(small, offset=1000, train_count=1)
    pairs = MultiCachedPairs([large, small], source_names=["large", "small"])

    assert pairs.source_batch_quotas(4).tolist() == [2, 2]
    assert pairs.batch_count("train", batch_size=4, source_balanced=True) == 2
    batches = list(
        pairs.batches(
            "train",
            batch_size=4,
            shuffle=True,
            seed=7,
            source_balanced=True,
        )
    )
    assert len(batches) == 2
    all_sources = []
    for indices, *_ in batches:
        sources, _ = pairs.source_and_local_indices(indices)
        assert np.bincount(sources, minlength=2).tolist() == [2, 2]
        all_sources.extend(sources.tolist())
    assert np.bincount(all_sources, minlength=2).tolist() == [4, 4]
