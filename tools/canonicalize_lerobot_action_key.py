"""Rename legacy LeRobot ``actions`` features to canonical ``action`` in-place.

The v2.1 -> v3.0 converter preserves custom feature names.  Official LeRobot
SmolVLA training, however, requires the singular ``action`` key.  This tool
updates the v3 metadata and rewrites affected parquet shards atomically without
loading an entire shard into memory.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq


LEGACY_KEY = "actions"
CANONICAL_KEY = "action"
LEGACY_STATS_PREFIX = "stats/actions/"
CANONICAL_STATS_PREFIX = "stats/action/"


def _renamed_column(name: str) -> str:
    if name == LEGACY_KEY:
        return CANONICAL_KEY
    if name.startswith(LEGACY_STATS_PREFIX):
        return CANONICAL_STATS_PREFIX + name[len(LEGACY_STATS_PREFIX) :]
    return name


def _rename_metadata_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            _renamed_column(key): _rename_metadata_json(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_rename_metadata_json(item) for item in value]
    if isinstance(value, str):
        return _renamed_column(value)
    return value


def _renamed_schema_metadata(metadata: dict[bytes, bytes] | None) -> dict[bytes, bytes] | None:
    if metadata is None:
        return None
    renamed = dict(metadata)
    for key in (b"huggingface", b"pandas"):
        raw = renamed.get(key)
        if raw is None:
            continue
        try:
            value = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            continue
        renamed[key] = json.dumps(
            _rename_metadata_json(value), separators=(",", ":")
        ).encode("utf-8")
    return renamed


def _planned_parquet_columns(path: Path) -> tuple[list[str], list[str]]:
    current = pq.ParquetFile(path).schema_arrow.names
    renamed = [_renamed_column(name) for name in current]
    if len(renamed) != len(set(renamed)):
        raise ValueError(
            f"refusing to create duplicate parquet columns in {path}: "
            f"{current} -> {renamed}"
        )
    return current, renamed


def _rewrite_parquet(path: Path, renamed_columns: list[str]) -> None:
    parquet = pq.ParquetFile(path)
    old_schema = parquet.schema_arrow
    new_schema = pa.schema(
        [field.with_name(name) for field, name in zip(old_schema, renamed_columns, strict=True)],
        metadata=_renamed_schema_metadata(old_schema.metadata),
    )
    temporary = path.with_name(f".{path.name}.action-key.tmp")
    try:
        with pq.ParquetWriter(temporary, new_schema, compression="snappy") as writer:
            for batch in parquet.iter_batches(batch_size=65_536):
                table = pa.Table.from_batches([batch]).rename_columns(renamed_columns)
                table = table.replace_schema_metadata(new_schema.metadata)
                writer.write_table(table)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise TypeError(f"expected a JSON object: {path}")
    return value


def _write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.action-key.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=4)
            stream.write("\n")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _rename_mapping_key(mapping: dict[str, Any], *, source: Path) -> bool:
    if LEGACY_KEY not in mapping:
        return False
    if CANONICAL_KEY in mapping:
        raise ValueError(f"both 'actions' and 'action' exist in {source}; refusing to overwrite")
    items = [
        (CANONICAL_KEY if key == LEGACY_KEY else key, value)
        for key, value in mapping.items()
    ]
    mapping.clear()
    mapping.update(items)
    return True


def canonicalize_dataset(root: Path, *, check_only: bool = False) -> int:
    root = root.expanduser().resolve()
    info_path = root / "meta" / "info.json"
    stats_path = root / "meta" / "stats.json"
    if not info_path.is_file():
        raise FileNotFoundError(f"missing LeRobot metadata: {info_path}")

    info = _load_json(info_path)
    if info.get("codebase_version") != "v3.0":
        raise ValueError(f"expected LeRobot v3.0 at {root}, got {info.get('codebase_version')!r}")
    features = info.get("features")
    if not isinstance(features, dict):
        raise TypeError(f"features must be an object: {info_path}")
    info_changed = _rename_mapping_key(features, source=info_path)

    stats: dict[str, Any] | None = None
    stats_changed = False
    if stats_path.is_file():
        stats = _load_json(stats_path)
        stats_changed = _rename_mapping_key(stats, source=stats_path)

    parquet_changes: list[tuple[Path, list[str]]] = []
    for path in sorted(root.rglob("*.parquet")):
        current, renamed = _planned_parquet_columns(path)
        if current != renamed:
            parquet_changes.append((path, renamed))

    needs_change = info_changed or stats_changed or bool(parquet_changes)
    if check_only:
        state = "needs migration" if needs_change else "canonical"
        print(f"{root}: {state}; parquet_shards={len(parquet_changes)}")
        return 1 if needs_change else 0

    # Commit the large data files first and info.json last.  If interrupted,
    # rerunning is safe and metadata never advertises a completed migration
    # before every parquet shard has been rewritten.
    for path, renamed in parquet_changes:
        print(f"rewrite {path}")
        _rewrite_parquet(path, renamed)
    if stats_changed and stats is not None:
        _write_json_atomic(stats_path, stats)
    if info_changed:
        _write_json_atomic(info_path, info)

    print(
        f"{root}: canonical action key ready; "
        f"parquet_shards_rewritten={len(parquet_changes)}"
    )
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("roots", nargs="+", type=Path, help="LeRobot v3.0 dataset roots")
    parser.add_argument("--check", action="store_true", help="report only; do not modify files")
    args = parser.parse_args()

    statuses = [canonicalize_dataset(root, check_only=args.check) for root in args.roots]
    if args.check and any(statuses):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
