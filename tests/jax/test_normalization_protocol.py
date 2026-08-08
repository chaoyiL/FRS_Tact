from __future__ import annotations

import json
import shutil
from dataclasses import replace
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from lerobot.datasets.io_utils import load_nested_dataset as real_load_nested_dataset
from lerobot.policies.smolvla_jax.data import DatasetSource
from lerobot.policies.smolvla_jax import normalization_protocol as protocol_module
from lerobot.policies.smolvla_jax.normalization_protocol import (
    NORMALIZATION_MANIFEST_FILENAME,
    PREPROCESSOR_STATS_FILENAME,
    build_or_validate_normalization_protocol,
)


DIM = 20


def _episode_row(
    episode_index: int,
    *,
    state_mean: float,
    action_mean: float,
    count: int,
    dim: int = DIM,
) -> dict[str, object]:
    row: dict[str, object] = {"episode_index": episode_index}
    for feature, mean in (("observation.state", state_mean), ("actions", action_mean)):
        prefix = f"stats/{feature}"
        row[f"{prefix}/mean"] = [mean] * dim
        row[f"{prefix}/std"] = [0.0] * dim
        row[f"{prefix}/min"] = [mean] * dim
        row[f"{prefix}/max"] = [mean] * dim
        row[f"{prefix}/count"] = [count]
    return row


def _write_episode_metadata(root: Path, rows: list[dict[str, object]]) -> None:
    info_path = root / "meta" / "info.json"
    info_path.parent.mkdir(parents=True, exist_ok=True)
    if not info_path.exists():
        info_path.write_text(
            json.dumps(
                {
                    "codebase_version": "v3.0",
                    "fps": 30,
                    "features": {
                        "observation.state": {"dtype": "float32", "shape": [DIM]},
                        "actions": {"dtype": "float32", "shape": [DIM]},
                    },
                }
            )
            + "\n",
            encoding="utf-8",
        )
    path = root / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), path)


def _write_split(path: Path, sources: list[DatasetSource], *, val_ids: list[int] | None = None) -> Path:
    path.write_text(
        json.dumps(
            {
                "version": 1,
                "val_fraction": 0.1,
                "datasets": [
                    {
                        "repo_id": source.repo_id,
                        "revision": source.revision,
                        "train_episodes": list(source.episodes or []),
                        "val_episodes": list(val_ids or []),
                    }
                    for source in sources
                ],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def _four_sources(tmp_path: Path) -> tuple[list[DatasetSource], list[tuple[float, float, int]]]:
    sources: list[DatasetSource] = []
    selected: list[tuple[float, float, int]] = []
    for source_index in range(4):
        root = tmp_path / f"dataset-{source_index}"
        first = (float(source_index), float(10 + source_index), source_index + 1)
        second = (float(source_index + 4), float(14 + source_index), source_index + 3)
        selected.extend((first, second))
        _write_episode_metadata(
            root,
            [
                _episode_row(0, state_mean=first[0], action_mean=first[1], count=first[2]),
                _episode_row(1, state_mean=second[0], action_mean=second[1], count=second[2]),
                _episode_row(2, state_mean=10_000.0, action_mean=-10_000.0, count=1000),
            ],
        )
        sources.append(
            DatasetSource(
                repo_id=f"org/dataset-{source_index}",
                root=root,
                revision=f"revision-{source_index}",
                episodes=[0, 1],
                action_key="actions",
                rename_map={
                    "observation.images.camera0": "observation.images.camera1",
                },
            )
        )
    return sources, selected


def test_protocol_aggregates_only_selected_train_episode_metadata_across_four_sources(
    tmp_path: Path,
) -> None:
    sources, selected = _four_sources(tmp_path)
    split_path = _write_split(tmp_path / "data_split.json", sources, val_ids=[2])

    result = build_or_validate_normalization_protocol(
        tmp_path / "shared-protocol",
        split_path=split_path,
        sources=sources,
        state_dim=DIM,
        action_dim=DIM,
    )

    total_count = sum(count for _, _, count in selected)
    expected_state = sum(state * count for state, _, count in selected) / total_count
    expected_action = sum(action * count for _, action, count in selected) / total_count
    np.testing.assert_allclose(result.stats["observation.state"]["mean"], expected_state)
    np.testing.assert_allclose(result.stats["action"]["mean"], expected_action)
    np.testing.assert_array_equal(result.stats["observation.state"]["count"], [total_count])
    assert float(result.stats["observation.state"]["mean"][0]) != 10_000.0
    assert result.split_path.read_bytes() == split_path.read_bytes()
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert manifest["algorithm_version"] == 1
    assert len(manifest["datasets"]) == 4
    assert manifest["dimensions"] == {"action": DIM, "observation.state": DIM}
    assert all(entry["train_episodes"] == [0, 1] for entry in manifest["datasets"])
    assert all(entry["selected_stats_sha256"] for entry in manifest["datasets"])
    assert manifest["canonical_stats_sha256"]


def test_local_protocol_does_not_eagerly_load_global_stats_or_all_episode_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "dataset"
    _write_episode_metadata(
        root,
        [_episode_row(0, state_mean=1.0, action_mean=2.0, count=2)],
    )
    # A real LeRobotDatasetMetadata construction eagerly loads global stats and
    # unfiltered episode metadata. Local protocol creation must use info.json plus
    # the predicate-pushed stats read instead.
    monkeypatch.setattr(
        "lerobot.policies.smolvla_jax.data.LeRobotDatasetMetadata",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("eager metadata container must not be constructed")
        ),
    )
    source = DatasetSource(
        repo_id="org/dataset",
        root=root,
        revision="revision",
        episodes=[0],
        action_key="actions",
    )
    split_path = _write_split(tmp_path / "data_split.json", [source])

    result = build_or_validate_normalization_protocol(
        tmp_path / "protocol",
        split_path=split_path,
        sources=[source],
        state_dim=DIM,
        action_dim=DIM,
    )

    np.testing.assert_array_equal(result.stats["action"]["mean"], np.full(DIM, 2.0))


def test_protocol_pushes_episode_and_stats_columns_into_parquet_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "dataset"
    row = _episode_row(0, state_mean=1.0, action_mean=2.0, count=2)
    row["tasks"] = [0]
    row["videos/observation.images.camera1/chunk_index"] = 0
    _write_episode_metadata(root, [row])
    source = DatasetSource(
        repo_id="org/dataset",
        root=root,
        revision="revision",
        episodes=[0],
        action_key="actions",
    )
    split_path = _write_split(tmp_path / "data_split.json", [source])
    calls: list[dict[str, object]] = []

    def capture_load(*args, **kwargs):
        calls.append(dict(kwargs))
        return real_load_nested_dataset(*args, **kwargs)

    monkeypatch.setattr(protocol_module, "load_nested_dataset", capture_load)

    build_or_validate_normalization_protocol(
        tmp_path / "protocol",
        split_path=split_path,
        sources=[source],
        state_dim=DIM,
        action_dim=DIM,
    )

    assert calls == [
        {
            "episodes": [0],
            "columns": [
                "episode_index",
                *[
                    f"stats/{feature}/{stat}"
                    for feature in ("observation.state", "actions")
                    for stat in ("min", "max", "mean", "std", "count")
                ],
            ],
        }
    ]


def test_identical_protocol_is_reused_byte_for_byte(tmp_path: Path) -> None:
    sources, _ = _four_sources(tmp_path)
    split_path = _write_split(tmp_path / "data_split.json", sources, val_ids=[2])
    protocol_dir = tmp_path / "shared-protocol"
    first = build_or_validate_normalization_protocol(
        protocol_dir,
        split_path=split_path,
        sources=sources,
        state_dim=DIM,
        action_dim=DIM,
    )
    before = {path.name: path.read_bytes() for path in protocol_dir.iterdir()}

    second = build_or_validate_normalization_protocol(
        protocol_dir,
        split_path=split_path,
        sources=sources,
        state_dim=DIM,
        action_dim=DIM,
    )

    after = {path.name: path.read_bytes() for path in protocol_dir.iterdir()}
    assert before == after
    assert first.manifest_path == second.manifest_path
    assert json.loads(first.manifest_path.read_text())["canonical_stats_sha256"] == json.loads(
        second.manifest_path.read_text()
    )["canonical_stats_sha256"]


@pytest.mark.parametrize("winner_matches", [True, False])
def test_concurrent_protocol_publish_validates_winner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    winner_matches: bool,
) -> None:
    root = tmp_path / "dataset"
    _write_episode_metadata(
        root,
        [_episode_row(0, state_mean=1.0, action_mean=2.0, count=2)],
    )
    source = DatasetSource(
        repo_id="org/dataset",
        root=root,
        revision="revision",
        episodes=[0],
        action_key="actions",
    )
    split_path = _write_split(tmp_path / "data_split.json", [source])
    protocol_dir = tmp_path / "shared-protocol"

    def concurrent_winner(staging: str | Path, destination: str | Path) -> None:
        shutil.copytree(staging, destination)
        if not winner_matches:
            (Path(destination) / "data_split.json").write_text(
                '{"version": 1, "datasets": []}\n', encoding="utf-8"
            )
        raise FileExistsError("another creator won")

    monkeypatch.setattr(protocol_module, "_rename_noreplace", concurrent_winner)

    if winner_matches:
        result = build_or_validate_normalization_protocol(
            protocol_dir,
            split_path=split_path,
            sources=[source],
            state_dim=DIM,
            action_dim=DIM,
        )
        assert result.manifest_path == protocol_dir / NORMALIZATION_MANIFEST_FILENAME
        assert result.split_path.read_bytes() == split_path.read_bytes()
    else:
        with pytest.raises(ValueError, match="split.*mismatch"):
            build_or_validate_normalization_protocol(
                protocol_dir,
                split_path=split_path,
                sources=[source],
                state_dim=DIM,
                action_dim=DIM,
            )
        assert (protocol_dir / "data_split.json").read_text(encoding="utf-8").startswith(
            '{"version": 1, "datasets": []}'
        )

    assert not list(tmp_path.glob(".shared-protocol.staging-*"))


@pytest.mark.parametrize(
    "case",
    [
        "duplicate_requested_episode",
        "missing_requested_episode",
        "duplicate_metadata_episode",
        "missing_stats",
        "non_finite_stats",
        "wrong_shape_stats",
    ],
)
def test_protocol_rejects_incomplete_or_invalid_episode_stats(tmp_path: Path, case: str) -> None:
    root = tmp_path / "dataset"
    rows = [
        _episode_row(0, state_mean=1.0, action_mean=2.0, count=2),
        _episode_row(1, state_mean=3.0, action_mean=4.0, count=3),
    ]
    episodes = [0, 1]
    if case == "duplicate_requested_episode":
        episodes = [0, 0]
    elif case == "missing_requested_episode":
        episodes = [0, 99]
    elif case == "duplicate_metadata_episode":
        rows.append(dict(rows[0]))
    elif case == "missing_stats":
        rows[0].pop("stats/actions/std")
        episodes = [0]
    elif case == "non_finite_stats":
        rows[0]["stats/observation.state/mean"] = [float("nan")] * DIM
        episodes = [0]
    elif case == "wrong_shape_stats":
        rows[0]["stats/actions/mean"] = [1.0] * (DIM - 1)
        episodes = [0]
    _write_episode_metadata(root, rows)
    source = DatasetSource(
        repo_id="org/dataset",
        root=root,
        revision="revision",
        episodes=episodes,
        action_key="actions",
    )
    split_path = _write_split(tmp_path / "data_split.json", [source])

    with pytest.raises(ValueError, match="episode|stats|finite|shape|unique|coverage"):
        build_or_validate_normalization_protocol(
            tmp_path / "protocol",
            split_path=split_path,
            sources=[source],
            state_dim=DIM,
            action_dim=DIM,
        )
    assert not (tmp_path / "protocol").exists()


def test_existing_protocol_corruption_is_not_overwritten(tmp_path: Path) -> None:
    sources, _ = _four_sources(tmp_path)
    split_path = _write_split(tmp_path / "data_split.json", sources, val_ids=[2])
    protocol_dir = tmp_path / "shared-protocol"
    build_or_validate_normalization_protocol(
        protocol_dir,
        split_path=split_path,
        sources=sources,
        state_dim=DIM,
        action_dim=DIM,
    )
    stats_path = protocol_dir / PREPROCESSOR_STATS_FILENAME
    stats_path.write_bytes(b"corrupt")

    with pytest.raises(ValueError, match="corrupt|digest|hash|asset"):
        build_or_validate_normalization_protocol(
            protocol_dir,
            split_path=split_path,
            sources=sources,
            state_dim=DIM,
            action_dim=DIM,
        )
    assert stats_path.read_bytes() == b"corrupt"


def test_existing_protocol_rejects_split_source_and_metadata_drift(tmp_path: Path) -> None:
    sources, _ = _four_sources(tmp_path)
    split_path = _write_split(tmp_path / "data_split.json", sources, val_ids=[2])
    protocol_dir = tmp_path / "shared-protocol"
    build_or_validate_normalization_protocol(
        protocol_dir,
        split_path=split_path,
        sources=sources,
        state_dim=DIM,
        action_dim=DIM,
    )

    split_payload = json.loads(split_path.read_text())
    split_payload["split_seed"] = 999
    split_path.write_text(json.dumps(split_payload, sort_keys=True) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="split|digest|mismatch"):
        build_or_validate_normalization_protocol(
            protocol_dir,
            split_path=split_path,
            sources=sources,
            state_dim=DIM,
            action_dim=DIM,
        )

    split_path = _write_split(tmp_path / "data_split.json", sources, val_ids=[2])
    drifted_sources = list(sources)
    drifted_sources[0] = replace(drifted_sources[0], revision="different-revision")
    with pytest.raises(ValueError, match="source|revision|split|mismatch"):
        build_or_validate_normalization_protocol(
            protocol_dir,
            split_path=split_path,
            sources=drifted_sources,
            state_dim=DIM,
            action_dim=DIM,
        )

    drifted_action_sources = list(sources)
    drifted_action_sources[0] = replace(drifted_action_sources[0], action_key=None)
    with pytest.raises(ValueError, match="action|source|manifest|mismatch"):
        build_or_validate_normalization_protocol(
            protocol_dir,
            split_path=split_path,
            sources=drifted_action_sources,
            state_dim=DIM,
            action_dim=DIM,
        )

    _write_episode_metadata(
        Path(sources[0].root),
        [
            _episode_row(0, state_mean=123.0, action_mean=10.0, count=1),
            _episode_row(1, state_mean=4.0, action_mean=14.0, count=3),
            _episode_row(2, state_mean=10_000.0, action_mean=-10_000.0, count=1000),
        ],
    )
    with pytest.raises(ValueError, match="stats|digest|drift|mismatch"):
        build_or_validate_normalization_protocol(
            protocol_dir,
            split_path=split_path,
            sources=sources,
            state_dim=DIM,
            action_dim=DIM,
        )


def test_per_episode_drift_is_detected_even_when_source_aggregate_is_unchanged(
    tmp_path: Path,
) -> None:
    root = tmp_path / "dataset"
    source = DatasetSource(
        repo_id="org/dataset",
        root=root,
        revision="revision",
        episodes=[0, 1],
        action_key="actions",
    )
    split_path = _write_split(tmp_path / "data_split.json", [source])
    _write_episode_metadata(
        root,
        [
            _episode_row(0, state_mean=0.0, action_mean=10.0, count=1),
            _episode_row(1, state_mean=2.0, action_mean=12.0, count=1),
        ],
    )
    protocol_dir = tmp_path / "protocol"
    build_or_validate_normalization_protocol(
        protocol_dir,
        split_path=split_path,
        sources=[source],
        state_dim=DIM,
        action_dim=DIM,
    )

    # Swapping the episode-level statistics leaves the aggregate exactly equal,
    # but provenance must still identify that selected metadata changed.
    _write_episode_metadata(
        root,
        [
            _episode_row(0, state_mean=2.0, action_mean=12.0, count=1),
            _episode_row(1, state_mean=0.0, action_mean=10.0, count=1),
        ],
    )
    with pytest.raises(ValueError, match="stats|digest|drift|mismatch"):
        build_or_validate_normalization_protocol(
            protocol_dir,
            split_path=split_path,
            sources=[source],
            state_dim=DIM,
            action_dim=DIM,
        )

def test_existing_protocol_with_missing_manifest_fails_closed(tmp_path: Path) -> None:
    sources, _ = _four_sources(tmp_path)
    split_path = _write_split(tmp_path / "data_split.json", sources, val_ids=[2])
    protocol_dir = tmp_path / "shared-protocol"
    build_or_validate_normalization_protocol(
        protocol_dir,
        split_path=split_path,
        sources=sources,
        state_dim=DIM,
        action_dim=DIM,
    )
    (protocol_dir / NORMALIZATION_MANIFEST_FILENAME).unlink()

    with pytest.raises(ValueError, match="missing|manifest|artifact"):
        build_or_validate_normalization_protocol(
            protocol_dir,
            split_path=split_path,
            sources=sources,
            state_dim=DIM,
            action_dim=DIM,
        )
