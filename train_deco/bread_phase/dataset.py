"""Phase labels for the two-stage Bread task.

The single policy is conditioned on a phase ID: 0 drives the right arm to
handle bread, and 1 drives the left arm to handle ketchup.  Labels are derived
solely from recorded actions so no external annotation file is required.
"""

from __future__ import annotations

from collections import Counter

import numpy as np
import torch
from torch.utils.data import Dataset


RIGHT_GRIPPER_INDEX = 19
LEFT_TRANSLATION = slice(0, 3)
RIGHT_CLOSE_THRESHOLD = 0.09
RIGHT_REOPEN_THRESHOLD = 0.10
LEFT_MOTION_PER_FRAME = 0.001
LEFT_MOTION_PATH = 0.01
LEFT_MOTION_WINDOW = 10


def _left_motion_starts(actions: np.ndarray) -> np.ndarray:
    deltas = np.asarray(actions[:, LEFT_TRANSLATION], dtype=np.float32)
    per_frame = np.linalg.norm(deltas, axis=1) > LEFT_MOTION_PER_FRAME
    starts = np.zeros(len(actions), dtype=bool)
    for row in np.flatnonzero(per_frame):
        end = min(len(actions), row + LEFT_MOTION_WINDOW)
        path = float(np.linalg.norm(deltas[row:end], axis=1).sum())
        starts[row] = path >= LEFT_MOTION_PATH
    return starts


def derive_bread_phase_labels(actions: np.ndarray) -> np.ndarray:
    """Return phase IDs for every action row in one Bread episode.

    ``0`` lasts through the right-arm bread phase. ``1`` begins with the first
    sustained left translation after the right gripper has closed and reopened.
    """

    actions = np.asarray(actions)
    if actions.ndim != 2 or actions.shape[1] != 20:
        raise ValueError(f"Bread phase labeling expects actions shaped [T, 20], got {actions.shape}")
    if len(actions) < 2:
        raise ValueError("Bread phase labeling requires at least two action rows")

    right_gripper = actions[:, RIGHT_GRIPPER_INDEX]
    closes = np.flatnonzero(right_gripper <= RIGHT_CLOSE_THRESHOLD)
    if len(closes) == 0:
        raise ValueError("Bread phase labeling missing right gripper close (<= 0.09)")
    close_row = int(closes[0])

    reopens = np.flatnonzero(
        right_gripper[close_row + 1 :] >= RIGHT_REOPEN_THRESHOLD
    )
    if len(reopens) == 0:
        raise ValueError("Bread phase labeling missing right gripper reopen (>= 0.10)")
    reopen_row = close_row + 1 + int(reopens[0])

    motion_starts = np.flatnonzero(_left_motion_starts(actions))
    before_reopen = motion_starts[motion_starts < reopen_row]
    if len(before_reopen):
        raise ValueError("Bread phase labeling found left motion before right gripper reopen")
    after_reopen = motion_starts[motion_starts > reopen_row]
    if len(after_reopen) == 0:
        raise ValueError("Bread phase labeling missing sustained left motion after right gripper reopen")
    left_start = int(after_reopen[0])

    labels = np.zeros(len(actions), dtype=np.int64)
    labels[left_start:] = 1
    return labels


class BreadPhaseDataset(Dataset):
    """Replace source task IDs with Bread phase IDs and balance train samples."""

    task_ids = ["right_bread", "left_ketchup"]

    def __init__(self, dataset: Dataset, *, balance_train: bool = True):
        if not hasattr(dataset, "index") or not hasattr(dataset, "_load_episode"):
            raise TypeError("BreadPhaseDataset requires a LeRobotVisionDECODataset-like source")
        self.dataset = dataset
        self.split = getattr(dataset, "split", "train")
        self.metadata = dict(getattr(dataset, "metadata", {}))
        self.metadata.update({"bread_phase_version": "bread-phase-v1", "phase_count": 2})
        self.stats = getattr(dataset, "stats", None)
        self.manifest = getattr(dataset, "manifest", None)
        self.source_chunk_size = getattr(dataset, "source_chunk_size", None)

        phase_by_key: dict[tuple[int, int], int] = {}
        processed_episodes: set[int] = set()
        for episode_id, _ in dataset.index:
            if episode_id in processed_episodes:
                continue
            processed_episodes.add(episode_id)
            episode = dataset._load_episode(episode_id)
            if "actions" not in episode:
                raise ValueError(f"Bread phase source episode={episode_id} has no actions")
            labels = derive_bread_phase_labels(episode["actions"])
            for row, phase_id in enumerate(labels):
                phase_by_key[(episode_id, row)] = int(phase_id)

        self._source_indices = list(range(len(dataset)))
        self._phase_by_source_index = {}
        for source_index, key in enumerate(dataset.index):
            if key not in phase_by_key:
                raise ValueError(f"Bread phase source index is outside episode actions: {key}")
            self._phase_by_source_index[source_index] = phase_by_key[key]

        if balance_train and self.split == "train":
            self._source_indices = self._balanced_indices(self._source_indices)
        self.phase_labels = [self._phase_by_source_index[index] for index in self._source_indices]

    def _balanced_indices(self, source_indices: list[int]) -> list[int]:
        by_phase = {
            phase: [index for index in source_indices if self._phase_by_source_index[index] == phase]
            for phase in (0, 1)
        }
        counts = Counter({phase: len(indices) for phase, indices in by_phase.items()})
        if not counts[0] or not counts[1]:
            raise ValueError("Bread phase training split must contain samples from both phases")
        target = max(counts.values())
        balanced = list(source_indices)
        for phase in (0, 1):
            deficit = target - counts[phase]
            if deficit:
                indices = by_phase[phase]
                balanced.extend(indices[offset % len(indices)] for offset in range(deficit))
        return balanced

    def __len__(self) -> int:
        return len(self._source_indices)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        source_index = self._source_indices[index]
        sample = dict(self.dataset[source_index])
        sample["task_index"] = torch.tensor(
            self._phase_by_source_index[source_index], dtype=torch.long
        )
        return sample


def build_bread_phase_datasets(
    train_dataset: Dataset, val_dataset: Dataset
) -> tuple[BreadPhaseDataset, BreadPhaseDataset]:
    """Wrap a LeRobot train/validation pair with the Bread phase contract."""

    return (
        BreadPhaseDataset(train_dataset, balance_train=True),
        BreadPhaseDataset(val_dataset, balance_train=False),
    )
