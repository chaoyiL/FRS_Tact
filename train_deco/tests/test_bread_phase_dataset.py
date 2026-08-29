import numpy as np
import pytest
import torch
from torch.utils.data import Dataset

from train_deco.bread_phase.dataset import (
    BreadPhaseDataset,
    build_bread_phase_datasets,
    derive_bread_phase_labels,
)


def _actions(length=20):
    actions = np.zeros((length, 20), dtype=np.float32)
    actions[:, 19] = 0.12
    return actions


def _valid_actions():
    actions = _actions()
    actions[2:5, 19] = 0.08  # right close and remain closed
    actions[5, 19] = 0.11  # right reopen
    actions[7:12, 0] = 0.003  # sustained left translation
    return actions


def test_phase_labels_wait_for_right_reopen_then_left_motion():
    labels = derive_bread_phase_labels(_valid_actions())

    assert labels[:7].tolist() == [0] * 7
    assert labels[7:].tolist() == [1] * 13


def test_phase_labels_ignore_left_adjustment_before_right_reopen():
    actions = _valid_actions()
    actions[0:5, 0] = 0.003

    labels = derive_bread_phase_labels(actions)

    assert labels[:7].tolist() == [0] * 7
    assert labels[7:].tolist() == [1] * 13


@pytest.mark.parametrize(
    "actions, message",
    [
        (_actions(), "right gripper close"),
        (np.where(np.indices((20, 20))[1] == 19, 0.08, _actions()), "right gripper reopen"),
    ],
)
def test_phase_labels_reject_missing_required_events(actions, message):
    with pytest.raises(ValueError, match=message):
        derive_bread_phase_labels(actions)


def test_phase_labels_reject_left_motion_before_right_reopen():
    actions = _valid_actions()
    actions[1:6, 0] = 0.003
    actions[7:12, 0] = 0.0

    with pytest.raises(ValueError, match="left motion"):
        derive_bread_phase_labels(actions)


class _FakeBase(Dataset):
    def __init__(self, split="train"):
        self.split = split
        self.index = [(0, row) for row in range(19)]
        self.task_ids = ["source"]
        self.metadata = {"source": "fake"}
        self.stats = {}
        self.manifest = {}
        self.source_chunk_size = 32
        self._episode = {"actions": _valid_actions()}

    def __len__(self):
        return len(self.index)

    def _load_episode(self, episode_id):
        assert episode_id == 0
        return self._episode

    def __getitem__(self, index):
        return {"task_index": torch.tensor(99), "source_index": torch.tensor(index)}


def test_wrapper_replaces_task_index_and_balances_train_deterministically():
    dataset = BreadPhaseDataset(_FakeBase("train"), balance_train=True)

    assert len(dataset) == 24  # 7 phase-0 frames, 12 phase-1 frames, then phase-0 repeats
    assert dataset.task_ids == ["right_bread", "left_ketchup"]
    assert dataset[0]["task_index"].item() == 0
    assert dataset[7]["task_index"].item() == 1
    assert dataset[-1]["task_index"].item() == 0


def test_wrapper_preserves_validation_distribution():
    dataset = BreadPhaseDataset(_FakeBase("val"), balance_train=True)

    assert len(dataset) == 19
    assert sum(dataset.phase_labels) == 12


def test_build_datasets_balances_only_train_split():
    train, val = build_bread_phase_datasets(_FakeBase("train"), _FakeBase("val"))

    assert len(train) == 24
    assert len(val) == 19
