from __future__ import annotations

import threading
import time
from pathlib import Path

import jax.numpy as jnp
import numpy as np

import prepare_pi05
from tools import prepare_frs_pi05_cache
from utils.cache import SampleRecord

# NOTE: the former `test_resize_skips_work_for_native_pi05_resolution` is gone with the
# hand-written `_resize` helper it covered. Resizing now goes through openpi's
# `transforms.ResizeImages` -> `image_tools.resize_with_pad`, which always runs (a jitted no-op
# resize at 224x224), so there is no short-circuit left to assert.


def test_parallel_sample_loading_preserves_order(monkeypatch) -> None:
    thread_ids: set[int] = set()
    lock = threading.Lock()

    class FakeDataset:
        def __getitem__(self, index: int):
            with lock:
                thread_ids.add(threading.get_ident())
            time.sleep(0.01)
            return {"index": index}

    class FakeModel:
        def prepare_sample(self, sample):
            index = sample["index"]
            return index, jnp.asarray(sample["actions"], dtype=jnp.float32), ""

    class FakeActionDataset:
        def __getitem__(self, index):
            rows = [np.asarray([value], dtype=np.float32) for value in range(index.start, index.stop)]
            return {"actions": rows}

    monkeypatch.setattr(prepare_pi05, "stack_observations", lambda values: values)
    records = [SampleRecord(index, 0, "train") for index in range(8)]

    observations, actions = prepare_pi05._load_observation_batch(
        FakeModel(),
        FakeDataset(),
        FakeActionDataset(),
        records,
        action_key="actions",
        action_horizon=3,
        episode_end_indices={0: 20},
        load_workers=4,
    )

    assert observations == list(range(8))
    np.testing.assert_array_equal(
        np.asarray(actions[..., 0]),
        np.arange(8)[:, None] + np.arange(3)[None, :],
    )
    assert len(thread_ids) > 1


def test_action_window_loading_merges_ranges_and_clamps_episode_tail() -> None:
    class FakeActionDataset:
        def __init__(self):
            self.slices = []

        def __getitem__(self, index):
            self.slices.append(index)
            rows = [np.asarray([value, -value], dtype=np.float32) for value in range(index.start, index.stop)]
            return {"actions": rows}

    dataset = FakeActionDataset()
    records = [SampleRecord(10, 1, "train"), SampleRecord(1, 0, "train")]

    windows = prepare_pi05._load_action_windows(
        dataset,
        records,
        action_key="actions",
        action_horizon=4,
        episode_end_indices={0: 5, 1: 12},
    )

    assert dataset.slices == [slice(1, 5), slice(10, 12)]
    np.testing.assert_array_equal(windows[0, :, 0], [10, 11, 11, 11])
    np.testing.assert_array_equal(windows[1, :, 0], [1, 2, 3, 4])


def test_config_loader_restores_pi05_only_once_for_all_sources(monkeypatch, tmp_path: Path) -> None:
    shared_model = object()
    load_calls = []
    prepare_calls = []

    monkeypatch.setattr(
        prepare_frs_pi05_cache,
        "load_pi0",
        lambda checkpoint, config: load_calls.append((checkpoint, config)) or shared_model,
    )
    monkeypatch.setattr(
        prepare_frs_pi05_cache,
        "prepare_cache",
        lambda **kwargs: prepare_calls.append(kwargs) or {},
    )
    config = {
        "checkpoint": "gs://example/pi05",
        "datasets": [
            {"repo_id": "org/one", "root": str(tmp_path / "one")},
            {"repo_id": "org/two", "root": str(tmp_path / "two")},
        ],
        "action_cache": {"root": str(tmp_path / "cache"), "load_workers": 3},
        "model": {
            "camera_map": {"left_wrist_0_rgb": "observation.images.camera1"},
            "action_dim": 32,
            "action_horizon": 50,
        },
        "norm_stats": {"dir": "gs://example/assets", "asset_id": "robot"},
    }

    prepare_frs_pi05_cache.prepare_from_config(config)

    assert len(load_calls) == 1
    assert len(prepare_calls) == 2
    assert all(call["loaded_model"] is shared_model for call in prepare_calls)
    assert all(call["load_workers"] == 3 for call in prepare_calls)
