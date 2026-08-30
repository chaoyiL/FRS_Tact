from types import SimpleNamespace

import pytest

from train_smolvla.torch_train import CombinedLeRobotDataset, dataset_sources


def test_dataset_sources_accepts_frs_style_list() -> None:
    config = {
        "dataset": {},
        "datasets": [
            {"repo_id": "org/one", "root": "/data/one"},
            {"repo_id": "org/two", "root": "/data/two", "revision": "v3"},
        ],
    }

    assert dataset_sources(config) == [
        {"repo_id": "org/one", "root": "/data/one", "revision": None},
        {"repo_id": "org/two", "root": "/data/two", "revision": "v3"},
    ]


def test_dataset_sources_rejects_unsupported_weight_instead_of_ignoring_it() -> None:
    config = {
        "dataset": {},
        "datasets": [{"repo_id": "org/one", "root": "/data/one", "weight": 2.0}],
    }

    with pytest.raises(ValueError, match=r"unknown datasets\[0\] fields.*weight"):
        dataset_sources(config)


def test_dataset_sources_rejects_duplicate_repositories() -> None:
    config = {
        "dataset": {},
        "datasets": [
            {"repo_id": "org/one", "root": "/data/one"},
            {"repo_id": "org/one", "root": "/data/one-again"},
        ],
    }

    with pytest.raises(ValueError, match="duplicate dataset repo_id"):
        dataset_sources(config)


class _FakeDataset:
    def __init__(self, name: str, lengths: list[int]):
        self._items = [f"{name}-{index}" for index in range(sum(lengths))]
        starts = []
        stops = []
        cursor = 0
        for length in lengths:
            starts.append(cursor)
            cursor += length
            stops.append(cursor)
        self.meta = SimpleNamespace(
            stats={"observation.state": {"source": name}},
            episodes={
                "dataset_from_index": starts,
                "dataset_to_index": stops,
                "tasks": [[name] for _ in lengths],
            },
        )
        self.episodes = list(range(len(lengths)))
        self.num_episodes = len(lengths)

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, index: int) -> str:
        return self._items[index]


def test_combined_dataset_concatenates_frames_episodes_and_stats() -> None:
    one = _FakeDataset("one", [2, 1])
    two = _FakeDataset("two", [1, 2])
    aggregate_calls = []

    def aggregate(stats):
        aggregate_calls.append(stats)
        return {"combined": len(stats)}

    combined = CombinedLeRobotDataset([one, two], aggregate)

    assert len(combined) == 6
    assert combined.num_frames == 6
    assert combined.num_episodes == 4
    assert [combined[index] for index in range(6)] == [
        "one-0",
        "one-1",
        "one-2",
        "two-0",
        "two-1",
        "two-2",
    ]
    assert combined.meta.episodes["dataset_from_index"] == [0, 2, 3, 4]
    assert combined.meta.episodes["dataset_to_index"] == [2, 3, 4, 6]
    assert combined.meta.episodes["tasks"] == [["one"], ["one"], ["two"], ["two"]]
    assert combined.meta.stats == {"combined": 2}
    assert len(aggregate_calls[0]) == 2


def test_combined_single_dataset_preserves_stats_without_aggregation() -> None:
    dataset = _FakeDataset("one", [2, 1])

    def aggregate(_stats):
        raise AssertionError("single-dataset stats must not be aggregated")

    combined = CombinedLeRobotDataset([dataset], aggregate)

    assert combined.meta.stats == dataset.meta.stats


def test_combined_multi_dataset_converts_tensor_stats_before_aggregation() -> None:
    class FakeTensor:
        def detach(self):
            return self

        def cpu(self):
            return self

        def numpy(self):
            return "numpy-value"

    one = _FakeDataset("one", [1])
    two = _FakeDataset("two", [1])
    one.meta.stats = {"observation.images.camera0": {"mean": FakeTensor()}}
    two.meta.stats = {"observation.images.camera0": {"mean": FakeTensor()}}

    def aggregate(stats):
        assert [item["observation.images.camera0"]["mean"] for item in stats] == [
            "numpy-value",
            "numpy-value",
        ]
        return {"combined": True}

    combined = CombinedLeRobotDataset([one, two], aggregate)

    assert combined.meta.stats == {"combined": True}
