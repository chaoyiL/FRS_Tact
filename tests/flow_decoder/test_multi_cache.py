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


def _write_cache(path: pathlib.Path, *, offset: int) -> None:
    records = [
        SampleRecord(offset + 10, 0, "train"),
        SampleRecord(offset + 11, 0, "train"),
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
            "train_sample_count": 2,
            "val_sample_count": 1,
            "action_horizon": 2,
            "action_dim": 2,
            "state_dim": 1,
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

    indices, x_base, predicted, gt, state = next(
        pairs.batches("train", batch_size=4, shuffle=False, seed=0)
    )
    assert indices.tolist() == [0, 1, 3, 4]
    np.testing.assert_array_equal(x_base[:, 0, 0], [0, 1, 1000, 1001])
    np.testing.assert_array_equal(predicted[:, 0, 0], [100, 101, 1100, 1101])
    np.testing.assert_array_equal(gt[:, 0, 0], [200, 201, 1200, 1201])
    np.testing.assert_array_equal(state[:, 0], [0, 0, 0, 0])
