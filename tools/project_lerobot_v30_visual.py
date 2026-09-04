#!/usr/bin/env python
"""Project a local LeRobot v3.0 dataset to selected visual streams in place.

Each data parquet is replaced atomically, one file at a time.  This makes the
operation restartable and bounds temporary storage to one projected parquet.
Dataset-level metadata is updated only after every data parquet is valid.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
from typing import Any

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from datasets import Features, Image


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as file:
        value = json.load(file)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.visual.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as file:
        json.dump(value, file, ensure_ascii=False, indent=4)
        file.write("\n")
        file.flush()
        os.fsync(file.fileno())
    os.replace(temporary, path)


def _visual_keys(features: dict[str, Any]) -> set[str]:
    return {
        key
        for key, feature in features.items()
        if isinstance(feature, dict) and feature.get("dtype") in {"image", "video"}
    }


def _project_data_parquet(path: Path, keep_visual_keys: set[str]) -> tuple[int, int]:
    original_size = path.stat().st_size
    original_rows = pq.read_metadata(path).num_rows
    columns = pq.read_schema(path).names
    missing = keep_visual_keys - set(columns)
    if missing:
        raise ValueError(f"{path} is missing required visual columns: {sorted(missing)}")
    projected_columns = [
        key
        for key in columns
        if not key.startswith("observation.images.") or key in keep_visual_keys
    ]
    if projected_columns == columns:
        return original_size, original_size

    frame = pd.read_parquet(path, columns=projected_columns)
    schema = pa.Schema.from_pandas(frame)
    features = Features.from_arrow_schema(schema)
    for key in keep_visual_keys:
        features[key] = Image()

    temporary = path.with_name(f".{path.name}.visual.tmp")
    try:
        frame.to_parquet(temporary, index=False, schema=features.arrow_schema)
        if pq.read_metadata(temporary).num_rows != original_rows:
            raise ValueError(f"Row count changed while projecting {path}")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return original_size, path.stat().st_size


def _project_episode_parquet(path: Path, removed_visual_keys: set[str]) -> None:
    columns = pq.read_schema(path).names
    prefixes = tuple(
        prefix
        for key in removed_visual_keys
        for prefix in (f"stats/{key}/", f"videos/{key}/")
    )
    projected_columns = [
        key for key in columns if not prefixes or not key.startswith(prefixes)
    ]
    if projected_columns == columns:
        return
    original_rows = pq.read_metadata(path).num_rows
    frame = pd.read_parquet(path, columns=projected_columns)
    temporary = path.with_name(f".{path.name}.visual.tmp")
    try:
        frame.to_parquet(temporary, index=False)
        if pq.read_metadata(temporary).num_rows != original_rows:
            raise ValueError(f"Episode row count changed while projecting {path}")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _remove_video_directories(root: Path, removed_visual_keys: set[str]) -> None:
    videos_root = (root / "videos").resolve()
    if not videos_root.is_dir():
        return
    for key in sorted(removed_visual_keys):
        target = (videos_root / key).resolve()
        if videos_root not in target.parents:
            raise ValueError(f"Unsafe video path resolved outside dataset: {target}")
        if target.is_dir():
            shutil.rmtree(target)


def project_dataset(root: Path, keep_visual_keys: list[str], *, check: bool = False) -> None:
    root = root.expanduser().resolve()
    info_path = root / "meta" / "info.json"
    stats_path = root / "meta" / "stats.json"
    if not info_path.is_file():
        raise FileNotFoundError(f"Missing LeRobot metadata: {info_path}")
    info = _read_json(info_path)
    if info.get("codebase_version") != "v3.0":
        raise ValueError(
            f"Expected LeRobot v3.0 at {root}, got {info.get('codebase_version')!r}"
        )
    features = info.get("features")
    if not isinstance(features, dict):
        raise ValueError(f"Invalid features mapping: {info_path}")

    requested = set(keep_visual_keys)
    if not requested or len(requested) != len(keep_visual_keys):
        raise ValueError("Visual keys must be non-empty and unique")
    available = _visual_keys(features)
    missing = requested - available
    if missing:
        raise ValueError(
            f"Requested visual keys are absent from {root}: {sorted(missing)}; "
            f"available={sorted(available)}"
        )

    data_paths = sorted((root / "data").glob("*/*.parquet"))
    if not data_paths:
        raise FileNotFoundError(f"No data parquet files found below {root / 'data'}")
    parquet_visual_keys = {
        key
        for path in data_paths
        for key in pq.read_schema(path).names
        if key.startswith("observation.images.")
    }
    removed = (available | parquet_visual_keys) - requested
    if check:
        if removed:
            raise ValueError(f"Dataset still contains unselected visual keys: {sorted(removed)}")
        print(f"visual-only dataset check passed: {root} cameras={sorted(requested)}")
        return

    before_bytes = 0
    after_bytes = 0
    for index, path in enumerate(data_paths, start=1):
        before, after = _project_data_parquet(path, requested)
        before_bytes += before
        after_bytes += after
        print(
            f"[visual-only] data parquet {index}/{len(data_paths)}: {path.name} "
            f"{before / 1024**2:.1f} -> {after / 1024**2:.1f} MiB",
            flush=True,
        )

    for path in sorted((root / "meta" / "episodes").glob("*/*.parquet")):
        _project_episode_parquet(path, removed)
    _remove_video_directories(root, removed)

    if stats_path.is_file():
        stats = _read_json(stats_path)
        _atomic_write_json(
            stats_path,
            {key: value for key, value in stats.items() if key not in removed},
        )

    info["features"] = {
        key: value for key, value in features.items() if key not in removed
    }
    if not any(
        feature.get("dtype") == "video" for feature in info["features"].values()
    ):
        info["video_path"] = None
    _atomic_write_json(info_path, info)

    project_dataset(root, keep_visual_keys, check=True)
    print(
        f"[visual-only] completed: {root} data parquet size "
        f"{before_bytes / 1024**3:.2f} -> {after_bytes / 1024**3:.2f} GiB",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument(
        "--keep-visual-key",
        dest="keep_visual_keys",
        action="append",
        required=True,
    )
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    project_dataset(args.root, args.keep_visual_keys, check=args.check)


if __name__ == "__main__":
    main()
