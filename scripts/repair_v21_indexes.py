#!/usr/bin/env python

"""Repair LeRobot v2.1 row indexes on a materialized conversion copy."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


INDEX_COLUMNS = ("index", "episode_index", "frame_index")


def _replace_integer_column(table: pa.Table, name: str, values: range) -> pa.Table:
    column_index = table.schema.get_field_index(name)
    return table.set_column(column_index, name, pa.array(values, type=pa.int64()))


def repair_v21_indexes(root: str | Path) -> tuple[int, int]:
    root = Path(root)
    info_path = root / "meta" / "info.json"
    if not info_path.is_file():
        raise FileNotFoundError(info_path)
    info = json.loads(info_path.read_text(encoding="utf-8"))
    if info.get("codebase_version") != "v2.1":
        raise ValueError(f"Expected v2.1 at {root}, got {info.get('codebase_version')!r}")

    paths = sorted((root / "data").glob("*/*.parquet"))
    expected_episodes = int(info.get("total_episodes", -1))
    if len(paths) != expected_episodes:
        raise ValueError(f"Expected {expected_episodes} episode parquet files, found {len(paths)}")

    row_counts: list[int] = []
    for episode_index, path in enumerate(paths):
        parquet = pq.ParquetFile(path)
        missing = [name for name in INDEX_COLUMNS if name not in parquet.schema_arrow.names]
        if missing:
            raise ValueError(f"Episode {episode_index} is missing index columns: {missing}")
        row_counts.append(parquet.metadata.num_rows)
    expected_frames = int(info.get("total_frames", -1))
    if sum(row_counts) != expected_frames:
        raise ValueError(f"Expected {expected_frames} frames, found {sum(row_counts)}")

    global_start = 0
    for episode_index, (path, length) in enumerate(zip(paths, row_counts, strict=True)):
        temporary = path.with_name(f".{path.name}.index-repair")
        try:
            table = pq.read_table(path)
            table = _replace_integer_column(
                table, "index", range(global_start, global_start + length)
            )
            table = table.set_column(
                table.schema.get_field_index("episode_index"),
                "episode_index",
                pa.array([episode_index] * length, type=pa.int64()),
            )
            table = _replace_integer_column(table, "frame_index", range(length))
            pq.write_table(table, temporary, compression="zstd")
            os.replace(temporary, path)
            global_start += length
        finally:
            temporary.unlink(missing_ok=True)

    return len(paths), global_start


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    episodes, frames = repair_v21_indexes(args.root)
    print(f"Repaired {episodes} episodes and {frames} frames at {args.root}")


if __name__ == "__main__":
    main()
