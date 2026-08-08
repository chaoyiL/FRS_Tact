from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from lerobot.datasets import io_utils


def test_load_nested_dataset_forwards_parquet_column_projection(
    tmp_path: Path,
    monkeypatch,
) -> None:
    parquet_path = tmp_path / "episodes" / "chunk-000" / "file-000.parquet"
    parquet_path.parent.mkdir(parents=True)
    pq.write_table(
        pa.Table.from_pylist(
            [
                {"episode_index": 0, "stats/value": [1.0], "unused": "drop-me"},
                {"episode_index": 1, "stats/value": [2.0], "unused": "drop-me"},
            ]
        ),
        parquet_path,
    )
    original = io_utils.Dataset.from_parquet
    captured: list[list[str] | None] = []

    def capture_from_parquet(*args, **kwargs):
        captured.append(kwargs.get("columns"))
        return original(*args, **kwargs)

    monkeypatch.setattr(io_utils.Dataset, "from_parquet", staticmethod(capture_from_parquet))

    result = io_utils.load_nested_dataset(
        parquet_path.parents[1],
        episodes=[1],
        columns=["episode_index", "stats/value"],
    )

    assert captured == [["episode_index", "stats/value"]]
    assert result.column_names == ["episode_index", "stats/value"]
    assert result["episode_index"] == [1]
