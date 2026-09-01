from pathlib import Path
import math

import pytest
from omegaconf import OmegaConf

from reactive_diffusion_policy.common.pick_tube_validation import (
    resolve_active_metric_baselines,
)
from reactive_diffusion_policy.common.checkpoint_util import PeriodicCheckpointManager
from reactive_diffusion_policy.workspace.base_workspace import BaseWorkspace
from reactive_diffusion_policy.workspace.train_at_workspace import (
    TrainATWorkspace,
    build_release_validation,
    get_deployment_phase_window,
    merge_noop_idle_metrics,
    namespace_deployment_release_metrics,
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
    workspace.active_translation_baseline_mm = 1.25
    workspace.active_rotation_baseline_deg = 2.5
    workspace.active_baseline_source = "auto"
    workspace.active_baseline_epoch = 1

    checkpoint = tmp_path / "deployable-state.ckpt"
    workspace.save_checkpoint(path=checkpoint, use_thread=False)

    restored = workspace_class.__new__(workspace_class)
    BaseWorkspace.__init__(restored, cfg, output_dir=str(tmp_path))
    restored.best_deploy_idle_score = math.inf
    restored.active_translation_baseline_mm = None
    restored.active_rotation_baseline_deg = None
    restored.active_baseline_source = None
    restored.active_baseline_epoch = None
    restored.load_checkpoint(path=checkpoint)

    assert restored.best_deploy_idle_score == pytest.approx(0.42)
    assert restored.active_translation_baseline_mm == pytest.approx(1.25)
    assert restored.active_rotation_baseline_deg == pytest.approx(2.5)
    assert restored.active_baseline_source == "auto"
    assert restored.active_baseline_epoch == 1


@pytest.mark.parametrize(
    "workspace_class", [TrainATWorkspace, TrainDiffusionUnetImageWorkspace]
)
def test_old_checkpoint_without_auto_baseline_state_calibrates_after_resume(
    tmp_path, workspace_class
):
    cfg = OmegaConf.create({"training": {"use_ema": False}})
    legacy = workspace_class.__new__(workspace_class)
    BaseWorkspace.__init__(legacy, cfg, output_dir=str(tmp_path))
    legacy.global_step = 3
    legacy.optimizer_step = 2
    legacy.epoch = 1
    legacy.best_deploy_idle_score = math.inf
    checkpoint = tmp_path / "legacy-state.ckpt"
    legacy.save_checkpoint(
        path=checkpoint,
        use_thread=False,
        include_keys=[
            "global_step",
            "optimizer_step",
            "epoch",
            "best_deploy_idle_score",
        ],
    )

    restored = workspace_class.__new__(workspace_class)
    BaseWorkspace.__init__(restored, cfg, output_dir=str(tmp_path))
    restored.active_translation_baseline_mm = None
    restored.active_rotation_baseline_deg = None
    restored.active_baseline_source = None
    restored.active_baseline_epoch = None
    restored.load_checkpoint(path=checkpoint)
    baseline = resolve_active_metric_baselines(
        external_baselines=None,
        auto_translation_baseline_mm=restored.active_translation_baseline_mm,
        auto_rotation_baseline_deg=restored.active_rotation_baseline_deg,
        auto_baseline_epoch=restored.active_baseline_epoch,
        active_translation_mm=1.25,
        active_rotation_deg=2.5,
        epoch=restored.epoch,
    )

    assert baseline["calibrated"] is True
    assert baseline["source"] == "auto"
    assert baseline["epoch"] == 1


def test_release_validation_is_omegaconf_safe_and_fail_closed_without_evidence():
    release = build_release_validation(
        passed=False,
        deployment_slow_update_interval=16,
        phase_start=3,
        score=math.inf,
        epoch=7,
        metrics={"val_deploy_idle_score": math.inf, "val_deployable": False},
    )
    cfg = OmegaConf.create({"release_validation": release})

    assert OmegaConf.to_container(cfg, resolve=True) == {
        "release_validation": {
            "passed": False,
            "deployment_slow_update_interval": 16,
            "phase_start": 3,
            "score": None,
            "epoch": 7,
            "active_baseline_source": None,
            "active_baseline_epoch": None,
            "metrics": {"val_deploy_idle_score": None, "val_deployable": False},
        }
    }

    with pytest.raises(ValueError, match="exactly 16"):
        build_release_validation(
            passed=False,
            deployment_slow_update_interval=15,
            phase_start=3,
            score=None,
            epoch=7,
            metrics={},
        )


def test_release_validation_records_auto_baseline_evidence():
    release = build_release_validation(
        passed=False,
        deployment_slow_update_interval=16,
        phase_start=3,
        score=0.1,
        epoch=7,
        active_baseline_source="auto",
        active_baseline_epoch=4,
        metrics={},
    )

    assert OmegaConf.to_container(OmegaConf.create(release), resolve=True) == {
        "passed": False,
        "deployment_slow_update_interval": 16,
        "phase_start": 3,
        "score": 0.1,
        "epoch": 7,
        "active_baseline_source": "auto",
        "active_baseline_epoch": 4,
        "metrics": {},
    }


def test_release_validation_records_external_baseline_without_epoch():
    release = build_release_validation(
        passed=True,
        deployment_slow_update_interval=16,
        phase_start=3,
        score=0.1,
        epoch=7,
        active_baseline_source="external",
        active_baseline_epoch=4,
        metrics={},
    )

    assert release["active_baseline_source"] == "external"
    assert release["active_baseline_epoch"] is None


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


@pytest.mark.parametrize("invalid", [16.9, 16.0, "16", True])
def test_deployment_phase_contract_rejects_non_integer_interval(invalid):
    cfg = OmegaConf.create(
        {
            "n_obs_steps": 2,
            "dataset_obs_temporal_downsample_ratio": 2,
            "validation": {"deployment_slow_update_interval": invalid},
        }
    )

    with pytest.raises(ValueError, match="integer"):
        get_deployment_phase_window(cfg)

    with pytest.raises(ValueError, match="integer"):
        build_release_validation(
            passed=False,
            deployment_slow_update_interval=invalid,
            phase_start=3,
            score=None,
            epoch=1,
            metrics={},
        )


@pytest.mark.parametrize("phase_start", [3.0, "3", True, 5])
def test_release_validation_requires_the_exact_integer_phase_start(phase_start):
    with pytest.raises(ValueError, match="phase_start"):
        build_release_validation(
            passed=False,
            deployment_slow_update_interval=16,
            phase_start=phase_start,
            score=None,
            epoch=1,
            metrics={},
        )


@pytest.mark.parametrize("key", ["n_obs_steps", "dataset_obs_temporal_downsample_ratio"])
@pytest.mark.parametrize("invalid", [2.0, "2", True])
def test_deployment_phase_contract_rejects_non_integer_observation_inputs(key, invalid):
    cfg = OmegaConf.create(
        {
            "n_obs_steps": 2,
            "dataset_obs_temporal_downsample_ratio": 2,
            "validation": {"deployment_slow_update_interval": 16},
        }
    )
    cfg[key] = invalid

    with pytest.raises(ValueError, match="integer"):
        get_deployment_phase_window(cfg)


@pytest.mark.parametrize(
    ("n_obs_steps", "ratio"),
    [(1, 2), (2, 3)],
)
def test_deployment_phase_contract_rejects_noncanonical_start(n_obs_steps, ratio):
    cfg = OmegaConf.create(
        {
            "n_obs_steps": n_obs_steps,
            "dataset_obs_temporal_downsample_ratio": ratio,
            "validation": {"deployment_slow_update_interval": 16},
        }
    )

    with pytest.raises(ValueError, match="phase_start"):
        get_deployment_phase_window(cfg)


def test_noop_metric_merge_preserves_real_active_and_micro_evidence():
    real_metrics = {
        "val_deploy_idle_translation_step_p95_mm": 0.01,
        "val_deploy_active_right_translation_mae_mm": 1.0,
        "val_deploy_micro_motion_recall": 0.99,
    }
    noop_metrics = {
        "val_deploy_idle_translation_step_p95_mm": 0.02,
        "val_deploy_active_right_translation_mae_mm": math.nan,
        "val_deploy_micro_motion_recall": math.nan,
    }

    merged = merge_noop_idle_metrics(real_metrics, noop_metrics)
    release_metrics = namespace_deployment_release_metrics(
        {
            "val_active_translation_degradation": 0.01,
            "val_active_rotation_degradation": 0.02,
            "val_micro_motion_recall": 0.99,
            "val_idle_score": 0.03,
            "val_checkpoint_feasible": True,
            "val_deployable": True,
        }
    )
    evidence = build_release_validation(
        passed=True,
        deployment_slow_update_interval=16,
        phase_start=3,
        score=release_metrics["val_deploy_idle_score"],
        epoch=1,
        metrics={**merged, **release_metrics},
    )

    assert merged["val_deploy_active_right_translation_mae_mm"] == 1.0
    assert merged["val_deploy_micro_motion_recall"] == 0.99
    assert merged["val_deploy_noop_idle_translation_step_p95_mm"] == 0.02
    assert "val_deploy_noop_active_right_translation_mae_mm" not in merged
    assert "val_deploy_noop_micro_motion_recall" not in merged
    assert evidence["metrics"]["val_deploy_active_right_translation_mae_mm"] == 1.0
    assert evidence["metrics"]["val_deploy_micro_motion_recall"] == 0.99


def test_release_metric_namespace_preserves_historical_and_deployment_values():
    full = {
        "val_active_translation_degradation": 0.5,
        "val_active_rotation_degradation": 0.6,
        "val_micro_motion_recall": 0.7,
    }
    deployment = namespace_deployment_release_metrics(
        {
            "val_active_translation_degradation": 0.01,
            "val_active_rotation_degradation": 0.02,
            "val_micro_motion_recall": 0.99,
            "val_idle_score": 0.03,
            "val_checkpoint_feasible": True,
            "val_deployable": True,
        }
    )
    evidence = build_release_validation(
        passed=True,
        deployment_slow_update_interval=16,
        phase_start=3,
        score=deployment["val_deploy_idle_score"],
        epoch=1,
        metrics={**full, **deployment},
    )

    assert evidence["metrics"]["val_micro_motion_recall"] == 0.7
    assert evidence["metrics"]["val_active_translation_degradation"] == 0.5
    assert evidence["metrics"]["val_deploy_micro_motion_recall"] == 0.99
    assert evidence["metrics"]["val_deploy_active_translation_degradation"] == 0.01
    assert evidence["metrics"]["val_deploy_active_rotation_degradation"] == 0.02
    assert evidence["metrics"]["val_deploy_checkpoint_feasible"] is True
