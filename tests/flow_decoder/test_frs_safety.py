from __future__ import annotations

import dataclasses
import pathlib

import jax.numpy as jnp
import numpy as np
import pytest
import torch

from lerobot.policies.smolvla_jax.configuration import JaxSmolVLAConfig
from prepare import _ActionCacheRecordDataset
from prepare import _create_batch_loader
from prepare import _prepare_observation_batch
from prepare import _require_finite_cache_batch
from tactile_flow_steering.train import _existing_run_artifacts
from tactile_flow_steering.train import _validate_resume_cache
from tools.compare_frs_reverse_solvers import mean_ratio
from tools.compare_frs_reverse_solvers import summarize_inversion_mse
from tools.train_frs import resolve_resume_mode
from utils.cache import SampleRecord


class _FakeBatchPreprocessor:
    rename_map = {}

    def prepare(self, observation, tasks):
        batch = len(tasks)
        return {
            "images": jnp.asarray(observation["observation.images.camera1"])[:, None],
            "image_masks": jnp.ones((batch, 1), dtype=jnp.bool_),
            "language_tokens": jnp.ones((batch, 3), dtype=jnp.int32),
            "language_masks": jnp.ones((batch, 3), dtype=jnp.bool_),
            "state": jnp.asarray(observation["observation.state"]),
        }

    def normalize_actions(self, actions):
        return actions + 1.0


class _FakeBatchModel:
    action_key = "actions"
    config = dataclasses.replace(
        JaxSmolVLAConfig(),
        image_keys=("observation.images.camera1",),
        chunk_size=2,
        action_dim=3,
        max_action_dim=4,
    )
    preprocessor = _FakeBatchPreprocessor()

    def image_keys_for_sample(self, sample):
        assert "observation.images.camera1" in sample
        return ("observation.images.camera1",)


class _SpawnFakeDataset:
    """Top-level fake so a spawn DataLoader can import and unpickle it."""

    def __getitem__(self, index):
        return {
            "observation.state": torch.full((4,), index, dtype=torch.float32),
            "observation.images.camera1": torch.zeros(3, 4, 4, dtype=torch.uint8),
            "actions": torch.zeros(2, 3),
            "actions_is_pad": torch.zeros(2, dtype=torch.bool),
            "task": "pick",
        }


def test_cache_batch_finite_check_rejects_nan() -> None:
    _require_finite_cache_batch(actions=np.ones((2, 3), dtype=np.float32))
    with pytest.raises(FloatingPointError, match=r"x_base.*\(1, 2\)"):
        values = np.ones((2, 3), dtype=np.float32)
        values[1, 2] = np.nan
        _require_finite_cache_batch(x_base=values)


def test_action_cache_record_dataset_filters_and_preserves_indices() -> None:
    class FakeDataset:
        def __getitem__(self, index):
            return {
                "observation.state": torch.full((4,), index),
                "observation.images.camera1": torch.zeros(3, 4, 4),
                "actions": torch.zeros(2, 3),
                "actions_is_pad": torch.zeros(2, dtype=torch.bool),
                "task": "pick",
                "episode_index": torch.tensor(99),
            }

    records = [SampleRecord(dataset_index=7, episode_index=0, split="train")]
    dataset = _ActionCacheRecordDataset(FakeDataset(), records, action_key="actions")
    sample = dataset[0]
    assert sample["__frs_dataset_index__"] == 7
    assert "episode_index" not in sample
    assert "observation.state" in sample


def test_action_cache_prepares_whole_batch_and_keeps_order() -> None:
    raw = {
        "__frs_dataset_index__": torch.tensor([7, 12]),
        "observation.state": torch.arange(8, dtype=torch.float32).reshape(2, 4),
        "observation.images.camera1": torch.zeros(2, 3, 4, 4),
        "actions": torch.zeros(2, 2, 3),
        "actions_is_pad": torch.zeros(2, 2, dtype=torch.bool),
        "task": ["pick", "pick"],
    }
    indices, observation, actions = _prepare_observation_batch(_FakeBatchModel(), raw)
    assert indices == [7, 12]
    assert observation.images.shape == (2, 1, 3, 4, 4)
    np.testing.assert_array_equal(actions, np.ones((2, 2, 3), dtype=np.float32))


def test_action_cache_spawn_loader_keeps_workers_cpu_only() -> None:
    records = [
        SampleRecord(dataset_index=index, episode_index=0, split="train")
        for index in range(4)
    ]
    loader, iterator, worker_batch_size = _create_batch_loader(
        _SpawnFakeDataset(),
        records,
        action_key="actions",
        batch_size=2,
        num_workers=2,
        prefetch_factor=1,
        worker_timeout_seconds=30.0,
    )
    try:
        assert worker_batch_size == 1
        first = next(iterator)
        second = next(iterator)
        assert first["__frs_dataset_index__"].tolist() == [0, 1]
        assert second["__frs_dataset_index__"].tolist() == [2, 3]
    finally:
        del iterator
        del loader


def test_fresh_output_guard_ignores_logs_but_finds_training_state(
    tmp_path: pathlib.Path,
) -> None:
    (tmp_path / "pipeline_20260101.log").write_text("safe", encoding="utf-8")
    assert _existing_run_artifacts(tmp_path) == ()

    history = tmp_path / "history.csv"
    history.write_text("epoch\n", encoding="utf-8")
    assert _existing_run_artifacts(tmp_path) == (history,)


def test_resume_cache_provenance_must_match() -> None:
    manifest = {
        "records_sha256": "records",
        "configuration": {"reverse_solver": "slerpflow"},
    }
    metadata = {
        "extra_metadata": {
            "cache_records_sha256": "records",
            "cache_configuration": {"reverse_solver": "slerpflow"},
        }
    }
    _validate_resume_cache(metadata, manifest)

    bad = {
        "extra_metadata": {
            "cache_records_sha256": "records",
            "cache_configuration": {"reverse_solver": "fireflow"},
        }
    }
    with pytest.raises(ValueError, match="different action-cache configuration"):
        _validate_resume_cache(bad, manifest)


def test_solver_ab_summary_counts_nonfinite_and_computes_ratio() -> None:
    fire = summarize_inversion_mse(np.asarray([1.0, 2.0, 3.0]))
    slerp = summarize_inversion_mse(np.asarray([0.5, 1.0, np.nan]))
    assert fire["mean"] == 2.0
    assert slerp["nonfinite_count"] == 1
    assert mean_ratio(slerp, fire) == pytest.approx(0.375)


def test_resume_auto_uses_last_checkpoint_when_available(tmp_path: pathlib.Path) -> None:
    assert not resolve_resume_mode("auto", output_dir=tmp_path)
    last = tmp_path / "last"
    last.mkdir()
    (last / "checkpoint.json").write_text("{}", encoding="utf-8")
    assert resolve_resume_mode("auto", output_dir=tmp_path)
    assert not resolve_resume_mode("false", output_dir=tmp_path)
