from pathlib import Path
import math

import pytest
from omegaconf import OmegaConf

from reactive_diffusion_policy.common.checkpoint_util import PeriodicCheckpointManager
from reactive_diffusion_policy.workspace.base_workspace import BaseWorkspace
from reactive_diffusion_policy.workspace.train_at_workspace import (
    TrainATWorkspace,
    build_release_validation,
    get_deployment_phase_window,
    should_update_deployable_checkpoint,
)
from reactive_diffusion_policy.workspace.train_diffusion_unet_image_workspace import (
    TrainDiffusionUnetImageWorkspace,
)


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


@pytest.mark.parametrize(
    ("passed", "score", "best_score", "expected"),
    [
        (True, 1.0, math.inf, True),
        (True, 0.9, 1.0, True),
        (False, 0.1, math.inf, False),
        (True, 1.0, 1.0, False),
        (True, 1.1, 1.0, False),
        (True, math.nan, math.inf, False),
        (True, math.inf, math.inf, False),
    ],
)
def test_deployable_checkpoint_requires_passing_finite_improvement(
    passed, score, best_score, expected
):
    assert should_update_deployable_checkpoint(passed, score, best_score) is expected


@pytest.mark.parametrize(
    "workspace_class", [TrainATWorkspace, TrainDiffusionUnetImageWorkspace]
)
def test_deployable_best_score_survives_checkpoint_round_trip(tmp_path, workspace_class):
    cfg = OmegaConf.create({"training": {"use_ema": False}})
    workspace = workspace_class.__new__(workspace_class)
    BaseWorkspace.__init__(workspace, cfg, output_dir=str(tmp_path))
    workspace.global_step = 3
    workspace.optimizer_step = 2
    workspace.epoch = 1
    workspace.best_deploy_idle_score = 0.42

    checkpoint = tmp_path / "deployable-state.ckpt"
    workspace.save_checkpoint(path=checkpoint, use_thread=False)

    restored = workspace_class.__new__(workspace_class)
    BaseWorkspace.__init__(restored, cfg, output_dir=str(tmp_path))
    restored.best_deploy_idle_score = math.inf
    restored.load_checkpoint(path=checkpoint)

    assert restored.best_deploy_idle_score == pytest.approx(0.42)


def test_release_validation_is_omegaconf_safe_and_fail_closed_without_evidence():
    release = build_release_validation(
        passed=False,
        deployment_slow_update_interval=16,
        score=math.inf,
        epoch=7,
        metrics={"val_deploy_idle_score": math.inf, "val_deployable": False},
    )
    cfg = OmegaConf.create({"release_validation": release})

    assert OmegaConf.to_container(cfg, resolve=True) == {
        "release_validation": {
            "passed": False,
            "deployment_slow_update_interval": 16,
            "score": None,
            "epoch": 7,
            "metrics": {"val_deploy_idle_score": None, "val_deployable": False},
        }
    }

    with pytest.raises(ValueError, match="exactly 16"):
        build_release_validation(
            passed=False,
            deployment_slow_update_interval=15,
            score=None,
            epoch=7,
            metrics={},
        )


def test_deployment_release_window_is_always_slow16():
    cfg = OmegaConf.create(
        {
            "n_obs_steps": 2,
            "dataset_obs_temporal_downsample_ratio": 2,
            "validation": {"deployment_slow_update_interval": 16},
        }
    )

    assert get_deployment_phase_window(cfg) == (3, 16)

    cfg.validation.deployment_slow_update_interval = 15
    with pytest.raises(ValueError, match="exactly 16"):
        get_deployment_phase_window(cfg)
