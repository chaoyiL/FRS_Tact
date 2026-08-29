from pathlib import Path

import pytest

from reactive_diffusion_policy.common.checkpoint_util import PeriodicCheckpointManager
from reactive_diffusion_policy.workspace.base_workspace import BaseWorkspace


def test_resume_advances_saved_end_of_epoch_state():
    workspace = BaseWorkspace(cfg=None)
    workspace.epoch = 4
    workspace.global_step = 116789

    workspace.advance_training_state_for_resume()

    assert workspace.epoch == 5
    assert workspace.global_step == 116790
    assert workspace.get_remaining_epochs(10) == 5


def test_epoch_target_is_total_for_fresh_and_completed_runs():
    workspace = BaseWorkspace(cfg=None)
    workspace.epoch = 0
    workspace.global_step = 0

    assert workspace.get_remaining_epochs(10) == 10

    workspace.epoch = 10
    assert workspace.get_remaining_epochs(10) == 0
    assert workspace.get_remaining_epochs(5) == 0


def test_negative_epoch_target_is_rejected():
    workspace = BaseWorkspace(cfg=None)
    workspace.epoch = 0

    with pytest.raises(ValueError, match="non-negative"):
        workspace.get_remaining_epochs(-1)


def test_checkpoint_schedule_always_saves_final_epoch():
    workspace = BaseWorkspace(cfg=None)

    workspace.epoch = 0
    assert workspace.should_save_checkpoint(10, local_epoch_idx=0, num_epochs_to_run=10)

    workspace.epoch = 1
    assert not workspace.should_save_checkpoint(10, local_epoch_idx=0, num_epochs_to_run=9)

    workspace.epoch = 9
    assert workspace.should_save_checkpoint(10, local_epoch_idx=8, num_epochs_to_run=9)


def test_checkpoint_schedule_rejects_invalid_interval():
    workspace = BaseWorkspace(cfg=None)
    workspace.epoch = 0

    with pytest.raises(ValueError, match="positive"):
        workspace.should_save_checkpoint(0, local_epoch_idx=0, num_epochs_to_run=1)


def test_periodic_checkpoint_retention_is_independent_from_topk(tmp_path):
    save_dir = tmp_path / "periodic"
    save_dir.mkdir()
    manager = PeriodicCheckpointManager(save_dir, keep=2)

    for epoch in (0, 2, 4):
        path = manager.get_ckpt_path(epoch)
        Path(path).write_bytes(b"checkpoint")
        manager.prune(path)

    assert [path.name for path in sorted(save_dir.glob("*.ckpt"))] == [
        "epoch=0002.ckpt",
        "epoch=0004.ckpt",
    ]
