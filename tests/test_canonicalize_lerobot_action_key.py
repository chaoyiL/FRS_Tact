from __future__ import annotations

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from tools.canonicalize_lerobot_action_key import canonicalize_dataset


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _dataset(root: Path) -> None:
    _write_json(
        root / "meta/info.json",
        {
            "codebase_version": "v3.0",
            "features": {
                "observation.state": {"dtype": "float32", "shape": [2]},
                "actions": {"dtype": "float32", "shape": [2]},
            },
        },
    )
    _write_json(root / "meta/stats.json", {"actions": {"mean": [1.0, 2.0]}})
    data = root / "data/chunk-000/file-000.parquet"
    data.parent.mkdir(parents=True)
    pq.write_table(
        pa.table(
            {
                "observation.state": [[0.0, 0.0]],
                "actions": [[1.0, 2.0]],
            },
            metadata={
                b"huggingface": json.dumps(
                    {"info": {"features": {"actions": {"_type": "List"}}}}
                )
            },
        ),
        data,
    )
    episodes = root / "meta/episodes/chunk-000/file-000.parquet"
    episodes.parent.mkdir(parents=True)
    pq.write_table(
        pa.table({"episode_index": [0], "stats/actions/mean": [[1.0, 2.0]]}),
        episodes,
    )


def test_canonicalize_dataset_updates_metadata_data_and_episode_stats(tmp_path: Path) -> None:
    _dataset(tmp_path)

    assert canonicalize_dataset(tmp_path) == 0

    info = json.loads((tmp_path / "meta/info.json").read_text(encoding="utf-8"))
    stats = json.loads((tmp_path / "meta/stats.json").read_text(encoding="utf-8"))
    assert "action" in info["features"] and "actions" not in info["features"]
    assert "action" in stats and "actions" not in stats
    assert pq.read_schema(tmp_path / "data/chunk-000/file-000.parquet").names == [
        "observation.state",
        "action",
    ]
    hf_metadata = pq.read_schema(tmp_path / "data/chunk-000/file-000.parquet").metadata[
        b"huggingface"
    ]
    assert b'"action"' in hf_metadata and b'"actions"' not in hf_metadata
    assert pq.read_schema(tmp_path / "meta/episodes/chunk-000/file-000.parquet").names == [
        "episode_index",
        "stats/action/mean",
    ]
    assert canonicalize_dataset(tmp_path, check_only=True) == 0


def test_canonicalize_dataset_refuses_action_collision(tmp_path: Path) -> None:
    _dataset(tmp_path)
    info_path = tmp_path / "meta/info.json"
    info = json.loads(info_path.read_text(encoding="utf-8"))
    info["features"]["action"] = info["features"]["actions"]
    _write_json(info_path, info)

    with pytest.raises(ValueError, match="both 'actions' and 'action'"):
        canonicalize_dataset(tmp_path)
