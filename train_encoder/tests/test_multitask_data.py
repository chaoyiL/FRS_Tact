from __future__ import annotations

import dataclasses
from types import SimpleNamespace

import numpy as np
import pytest

from train_encoder.utils.data import batches, build_future_records, future_records_digest, resolve_data_keys
from train_encoder.utils.tasks import normalize_tasks, task_repo_ids


class TaskDataset:
    # Two shards belong to the first task; its budget must apply across both.
    def episodes_by_repo(self):
        return {
            "pick_a": tuple(range(200)),
            "pick_b": tuple(range(200, 400)),
            "two": tuple(range(400, 800)),
            "insert": tuple(range(800, 1200)),
            "press": tuple(range(1200, 1600)),
        }

    def indices_for_episode(self, episode):
        return tuple(range(episode * 6, (episode + 1) * 6))

    def get_images(self, index, keys, *, as_float=True):
        episode = index // 6
        if episode >= 800:
            assert not set(keys) & {"left_image", "tactile_left_0", "tactile_right_0"}
        return {key: np.zeros((2, 2, 3), dtype=np.float32) for key in keys}


def task_config():
    return [
        {
            "name": name,
            "dataset_repo_ids": repos,
            "sides": sides,
            "num_train_episodes": 300,
            "num_val_episodes": 30,
        }
        for name, repos, sides in (
            ("pick", ["pick_a", "pick_b"], ["left", "right"]),
            ("two", ["two"], ["left", "right"]),
            ("insert", ["insert"], ["right"]),
            ("press", ["press"], ["right"]),
        )
    ]


def test_exact_task_budgets_disjoint_and_reproducible():
    dataset = TaskDataset()
    kwargs = {"tasks": task_config(), "frame_stride": 2, "split_seed": 7}
    records = build_future_records(dataset, **kwargs)
    assert len(records.train_episodes) == 1200
    assert len(records.val_episodes) == 120
    assert not set(records.train_episodes) & set(records.val_episodes)
    for selection in records.task_selections:
        assert len(selection["train_episodes"]) == 300
        assert len(selection["val_episodes"]) == 30
    assert records == build_future_records(dataset, **kwargs)
    assert records != build_future_records(dataset, **{**kwargs, "split_seed": 8})
    for record in records.records:
        assert record.dataset_index // 6 == record.future_dataset_index // 6
        assert record.side_ids == ((0, 1) if record.episode_index < 800 else (1,))


def test_thread_batches_never_request_inactive_wrist():
    dataset = TaskDataset()
    records = build_future_records(dataset, tasks=task_config(), frame_stride=5).split_records("train")
    loaded = list(
        batches(
            dataset,
            records,
            resolve_data_keys(),
            batch_size=32,
            shuffle=True,
            seed=5,
            num_workers=2,
            prefetch_batches=2,
        )
    )
    episodes = np.concatenate([batch["episode_index"] for batch in loaded])
    sides = np.concatenate([batch["side_id"] for batch in loaded])
    assert sum(len(batch["side_id"]) for batch in loaded) == sum(len(record.side_ids) for record in records)
    assert np.all(sides[episodes >= 800] == 1)
    assert set(sides[episodes < 800]) == {0, 1}


def test_mp_batches_receive_only_active_wrist_indices(monkeypatch):
    from train_encoder.utils import mp_batches

    dataset = TaskDataset()
    records = build_future_records(dataset, tasks=task_config(), frame_stride=5).split_records("train")

    def inspect_indices(**kwargs):
        selected = kwargs["sample_index"]
        assert len(selected) == sum(len(record.side_ids) for record in records)
        assert all(side in records[index].side_ids for index, side in selected)
        return iter(())

    monkeypatch.setattr(mp_batches, "iter_mp_batches", inspect_indices)
    assert (
        list(
            batches(
                dataset,
                records,
                resolve_data_keys(),
                batch_size=32,
                shuffle=True,
                seed=5,
                loader="mp",
                dataset_repo_ids=task_repo_ids(task_config()),
                num_workers=2,
            )
        )
        == []
    )


def test_shortage_fails_instead_of_repeating_episodes():
    tasks = task_config()
    tasks[-1]["num_train_episodes"] = 401
    with pytest.raises(ValueError, match="press.*needs 431.*only 400"):
        build_future_records(TaskDataset(), tasks=tasks)


def test_side_selection_changes_checkpoint_digest():
    record = build_future_records(TaskDataset(), tasks=task_config()).records[0]
    assert future_records_digest([record]) != future_records_digest(
        [dataclasses.replace(record, side_ids=(1,))]
    )


def test_duplicate_task_sources_rejected():
    tasks = task_config()
    tasks[-1]["dataset_repo_ids"] = ["insert"]
    with pytest.raises(ValueError, match="unique across tasks"):
        normalize_tasks(tasks)


def test_single_wrist_image_datasets_keep_global_episode_and_frame_indices(monkeypatch):
    from train_encoder.utils import image_dataset

    raw_keys = [image_dataset.RAW_CAMERA1_KEY, *image_dataset.RAW_TACTILE_KEYS[2:]]

    class Rows:
        def __init__(self, rows):
            self.rows = rows
            self.column_names = list(rows[0])

        def with_transform(self, value):
            return self

        def with_format(self, value):
            return self

        def select_columns(self, keys):
            return Rows([{key: row[key] for key in keys} for row in self.rows])

        def __getitem__(self, index):
            return self.rows[index]

    class LeRobotDataset:
        def __init__(self, repo_id, **kwargs):
            self.meta = SimpleNamespace(video_keys=[], image_keys=raw_keys, total_episodes=3)
            self.episode_data_index = {"from": [0, 2, 4], "to": [2, 4, 6]}
            self.hf_dataset = Rows(
                [
                    dict(
                        episode_index=i // 2,
                        timestamp=i / 30,
                        index=i,
                        **{key: np.zeros((2, 2, 3), dtype=np.uint8) for key in raw_keys},
                    )
                    for i in range(6)
                ]
            )

        def __len__(self):
            return 6

    monkeypatch.setattr(
        image_dataset,
        "_import_lerobot_dataset_module",
        lambda: SimpleNamespace(LeRobotDataset=LeRobotDataset),
    )
    dataset = image_dataset.create_image_dataset(["right_a", "right_b"], image_size=2).dataset
    assert dataset.episodes_by_repo() == {"right_a": (0, 1, 2), "right_b": (3, 4, 5)}
    assert dataset.indices_for_episode(4) == (8, 9)
    frame = dataset.get_images(8, ["right_image", "tactile_left_1", "tactile_right_1"])
    assert frame["episode_index"] == 4
    assert frame["dataset_index"] == 8
    assert dataset[8]["right_image"].shape == (2, 2, 3)
    with pytest.raises(KeyError, match="lacks requested wrist images"):
        dataset.get_images(8, ["left_image"])
