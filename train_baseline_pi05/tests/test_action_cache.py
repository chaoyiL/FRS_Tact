from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from train_baseline_pi05.action_cache import ActionCache, ActionCacheWriter, SampleRecord, build_records


class _Metadata:
    def __init__(self, episode_lengths: list[int]) -> None:
        self.episodes: list[dict[str, int]] = []
        start = 0
        for length in episode_lengths:
            self.episodes.append({"dataset_from_index": start, "dataset_to_index": start + length})
            start += length
        self.total_episodes = len(self.episodes)


@pytest.fixture
def metadata() -> _Metadata:
    return _Metadata([8] * 10)


@pytest.fixture
def records() -> tuple[SampleRecord, ...]:
    return (
        SampleRecord(dataset_index=10, episode_index=1, frame_index=10, split_id=0),
        SampleRecord(dataset_index=11, episode_index=1, frame_index=11, split_id=0),
        SampleRecord(dataset_index=20, episode_index=2, frame_index=20, split_id=1),
    )


@pytest.fixture
def manifest() -> dict[str, object]:
    return {
        "dataset_identity": {"repo_id": "example/dataset", "revision": "abc123"},
        "split": {"fractions": [0.8, 0.1, 0.1], "seed": 42},
        "source_checkpoint": "pi05/checkpoint",
        "norm_stats": "pi05/norm-stats",
        "sample_steps": 10,
        "noise_seed": 0,
        "source_model_action_width": 32,
        "decoder_action_width": 20,
        "action_space": "normalized_joint_action",
    }


def test_episode_split_is_disjoint_deterministic_and_strided(metadata: _Metadata) -> None:
    first = build_records(metadata, split_seed=42, fractions=(0.8, 0.1, 0.1), frame_stride=3)
    second = build_records(metadata, split_seed=42, fractions=(0.8, 0.1, 0.1), frame_stride=3)

    assert first == second
    by_split = {
        split_id: {record.episode_index for record in first if record.split_id == split_id}
        for split_id in (0, 1, 2)
    }
    assert by_split[0].isdisjoint(by_split[1] | by_split[2])
    assert by_split[1].isdisjoint(by_split[2])
    assert all(by_split[split_id] for split_id in (0, 1, 2))
    assert [record.frame_index for record in first if record.episode_index == 0] == [0, 3, 6]


@pytest.mark.parametrize("fractions", ((-0.1, 0.6, 0.5), (0.8, 0.1, 0.2)))
def test_record_builder_rejects_invalid_fractions(metadata: _Metadata, fractions: tuple[float, ...]) -> None:
    with pytest.raises(ValueError, match="fractions"):
        build_records(metadata, split_seed=42, fractions=fractions, frame_stride=1)


def test_record_builder_allows_zero_fraction_and_omits_that_split(metadata: _Metadata) -> None:
    records = build_records(metadata, split_seed=42, fractions=(1.0, 0.0, 0.0), frame_stride=1)

    assert {record.split_id for record in records} == {0}
    assert {record.episode_index for record in records} == set(range(metadata.total_episodes))


def test_action_cache_schema_tail_mask_and_reader_indices(
    tmp_path: Path, manifest: dict[str, object], records: tuple[SampleRecord, ...]
) -> None:
    writer = ActionCacheWriter.create(tmp_path, sample_count=3, horizon=50, action_dim=20, manifest=manifest)
    valid = np.ones((3, 50), dtype=bool)
    valid[2, 4:] = False
    writer.write_batch(
        0,
        coarse=np.zeros((3, 50, 20), dtype=np.float32),
        expert=np.ones((3, 50, 20), dtype=np.float32),
        valid=valid,
        records=records,
    )
    writer.finalize()

    assert {path.name for path in tmp_path.iterdir()} == {
        "coarse_actions.npy",
        "expert_actions.npy",
        "valid_masks.npy",
        "dataset_indices.npy",
        "episode_indices.npy",
        "split_ids.npy",
        "manifest.json",
        ".writer.lock",
    }
    assert not (tmp_path / "x_base.npy").exists()
    cache = ActionCache.open(tmp_path)
    assert cache.coarse_actions.dtype == np.float32
    assert cache.expert_actions.dtype == np.float32
    assert cache.valid_masks.dtype == np.bool_
    assert cache.dataset_indices.dtype == np.int64
    assert cache.episode_indices.dtype == np.int64
    assert cache.split_ids.dtype == np.uint8
    assert cache.coarse_actions.shape == (3, 50, 20)
    assert cache.expert_actions.shape == (3, 50, 20)
    assert cache.valid_masks.shape == (3, 50)
    assert not cache.valid_masks[2, 4:].any()
    assert np.array_equal(cache.indices("train"), np.array([0, 1], dtype=np.int64))
    assert np.array_equal(cache.indices("validation"), np.array([2], dtype=np.int64))
    assert not cache.indices("test").size


def test_writer_persists_progress_resumes_only_matching_manifest_and_finalizes(
    tmp_path: Path, manifest: dict[str, object], records: tuple[SampleRecord, ...]
) -> None:
    writer = ActionCacheWriter.create(tmp_path, sample_count=3, horizon=2, action_dim=20, manifest=manifest)
    writer.write_batch(
        0,
        coarse=np.zeros((1, 2, 20), dtype=np.float32),
        expert=np.ones((1, 2, 20), dtype=np.float32),
        valid=np.ones((1, 2), dtype=bool),
        records=records[:1],
    )
    progress = json.loads((tmp_path / "manifest.json").read_text())
    assert progress["status"] == "incomplete"
    assert progress["completed_samples"] == 1
    writer.close()
    with pytest.raises(ValueError, match="immutable"):
        ActionCacheWriter.resume(tmp_path, {**manifest, "noise_seed": 1})

    resumed = ActionCacheWriter.resume(tmp_path, manifest)
    resumed.write_batch(
        1,
        coarse=np.zeros((2, 2, 20), dtype=np.float32),
        expert=np.ones((2, 2, 20), dtype=np.float32),
        valid=np.ones((2, 2), dtype=bool),
        records=records[1:],
    )
    resumed.finalize()
    finished = json.loads((tmp_path / "manifest.json").read_text())
    assert finished["status"] == "complete"
    assert finished["completed_samples"] == 3
    assert isinstance(finished["records_sha256"], str)


def test_writer_lock_rejects_concurrent_create_and_resume_then_releases(
    tmp_path: Path, manifest: dict[str, object]
) -> None:
    writer = ActionCacheWriter.create(tmp_path, sample_count=3, horizon=2, action_dim=20, manifest=manifest)

    with pytest.raises(RuntimeError, match="locked"):
        ActionCacheWriter.create(tmp_path, sample_count=3, horizon=2, action_dim=20, manifest=manifest)
    with pytest.raises(RuntimeError, match="locked"):
        ActionCacheWriter.resume(tmp_path, manifest)

    writer.close()
    resumed = ActionCacheWriter.resume(tmp_path, manifest)
    resumed.close()


def test_writer_rejects_batch_bounds_shapes_and_nonfinite_values(
    tmp_path: Path, manifest: dict[str, object], records: tuple[SampleRecord, ...]
) -> None:
    writer = ActionCacheWriter.create(tmp_path, sample_count=3, horizon=2, action_dim=20, manifest=manifest)
    with pytest.raises(ValueError, match="start"):
        writer.write_batch(
            1,
            coarse=np.zeros((1, 2, 20), dtype=np.float32),
            expert=np.zeros((1, 2, 20), dtype=np.float32),
            valid=np.ones((1, 2), dtype=bool),
            records=records[:1],
        )
    with pytest.raises(ValueError, match="shape"):
        writer.write_batch(
            0,
            coarse=np.zeros((1, 2, 19), dtype=np.float32),
            expert=np.zeros((1, 2, 20), dtype=np.float32),
            valid=np.ones((1, 2), dtype=bool),
            records=records[:1],
        )
    with pytest.raises(ValueError, match="finite"):
        writer.write_batch(
            0,
            coarse=np.full((1, 2, 20), np.nan, dtype=np.float32),
            expert=np.zeros((1, 2, 20), dtype=np.float32),
            valid=np.ones((1, 2), dtype=bool),
            records=records[:1],
        )


def test_open_rejects_incomplete_or_corrupt_cache(
    tmp_path: Path, manifest: dict[str, object], records: tuple[SampleRecord, ...]
) -> None:
    writer = ActionCacheWriter.create(tmp_path, sample_count=3, horizon=2, action_dim=20, manifest=manifest)
    with pytest.raises(ValueError, match="complete"):
        ActionCache.open(tmp_path)
    writer.write_batch(
        0,
        coarse=np.zeros((3, 2, 20), dtype=np.float32),
        expert=np.zeros((3, 2, 20), dtype=np.float32),
        valid=np.ones((3, 2), dtype=bool),
        records=records,
    )
    writer.finalize()
    (tmp_path / "valid_masks.npy").unlink()
    with pytest.raises((FileNotFoundError, ValueError), match="valid_masks|array"):
        ActionCache.open(tmp_path)
