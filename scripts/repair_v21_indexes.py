#!/usr/bin/env python

"""Repair LeRobot v2.1 row indexes on a materialized conversion copy."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


INDEX_COLUMNS = ("index", "episode_index", "frame_index")
REQUIRED_STAT_FIELDS = ("min", "max", "mean", "std", "count")


def _replace_integer_column(table: pa.Table, name: str, values: range) -> pa.Table:
    column_index = table.schema.get_field_index(name)
    return table.set_column(column_index, name, pa.array(values, type=pa.int64()))


def _quantile(values: list[int], quantile: float) -> float:
    position = quantile * (len(values) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(values[lower])
    fraction = position - lower
    return float(values[lower] + (values[upper] - values[lower]) * fraction)


def _repaired_scalar_stats(
    values: list[int], template: object, *, feature_name: str
) -> dict[str, object]:
    if not isinstance(template, dict):
        raise ValueError(f"{feature_name} episode stats must be an object")
    missing = [name for name in REQUIRED_STAT_FIELDS if name not in template]
    if missing:
        raise ValueError(f"{feature_name} episode stats are missing fields: {missing}")
    count = len(values)
    mean = math.fsum(values) / count
    variance = math.fsum((value - mean) ** 2 for value in values) / count
    repaired = dict(template)
    repaired.update(
        {
            "min": [float(min(values))],
            "max": [float(max(values))],
            "mean": [mean],
            "std": [math.sqrt(variance)],
            "count": [count],
        }
    )
    for field in template:
        if not (field.startswith("q") and field[1:].isdigit()):
            continue
        percentile = int(field[1:])
        if not 0 <= percentile <= 100:
            raise ValueError(f"Invalid quantile field {feature_name}.{field}")
        repaired[field] = [_quantile(values, percentile / 100)]
    return repaired


def _load_episode_stats(path: Path, expected_episodes: int) -> list[dict[str, object]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(rows) != expected_episodes:
        raise ValueError(
            f"Expected {expected_episodes} episode stats rows, found {len(rows)}"
        )
    for episode_index, row in enumerate(rows):
        if not isinstance(row, dict) or row.get("episode_index") != episode_index:
            raise ValueError(
                f"Episode stats rows must be ordered by contiguous episode_index; "
                f"row {episode_index} is {row!r}"
            )
        feature_stats = row.get("stats")
        if not isinstance(feature_stats, dict):
            raise ValueError(f"Episode {episode_index} stats must be an object")
        for name in INDEX_COLUMNS:
            _repaired_scalar_stats([0], feature_stats.get(name), feature_name=name)
    return rows


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
    episode_stats_path = root / "meta" / "episodes_stats.jsonl"
    episode_stats_rows = _load_episode_stats(episode_stats_path, expected_episodes)

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
            indexes = list(range(global_start, global_start + length))
            frame_indexes = list(range(length))
            feature_stats = episode_stats_rows[episode_index]["stats"]
            assert isinstance(feature_stats, dict)
            feature_stats["index"] = _repaired_scalar_stats(
                indexes, feature_stats["index"], feature_name="index"
            )
            feature_stats["episode_index"] = _repaired_scalar_stats(
                [episode_index] * length,
                feature_stats["episode_index"],
                feature_name="episode_index",
            )
            feature_stats["frame_index"] = _repaired_scalar_stats(
                frame_indexes, feature_stats["frame_index"], feature_name="frame_index"
            )
            global_start += length
        finally:
            temporary.unlink(missing_ok=True)

    stats_temporary = episode_stats_path.with_name(
        f".{episode_stats_path.name}.index-repair"
    )
    try:
        stats_temporary.write_text(
            "".join(json.dumps(row, separators=(",", ":")) + "\n" for row in episode_stats_rows),
            encoding="utf-8",
        )
        os.replace(stats_temporary, episode_stats_path)
    finally:
        stats_temporary.unlink(missing_ok=True)

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
