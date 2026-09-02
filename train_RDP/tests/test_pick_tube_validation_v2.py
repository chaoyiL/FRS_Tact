import json
import math
import random
from pathlib import Path

import numpy as np
import pytest
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from reactive_diffusion_policy.common.pick_tube_validation import (
    build_canonical_noop_actions,
    build_episode_split_manifest,
    compute_contiguous_300_step_drift,
    compute_deployment_window_metrics,
    compute_idle_rollout_metrics,
    evaluate_checkpoint_feasibility,
    load_active_metric_baselines,
    preserve_global_rng_state,
    resolve_active_metric_baselines,
    validate_resume_action_contract,
)


IDENTITY_6D = np.asarray([1, 0, 0, 0, 1, 0], dtype=np.float64)


def _rotation_6d_z(degrees):
    angle = math.radians(degrees)
    return np.asarray(
        [math.cos(angle), math.sin(angle), 0, -math.sin(angle), math.cos(angle), 0],
        dtype=np.float64,
    )


def _neutral_actions(batch=1, horizon=29):
    actions = np.zeros((batch, horizon, 20), dtype=np.float64)
    actions[..., 3:9] = IDENTITY_6D
    actions[..., 13:19] = IDENTITY_6D
    return actions


def test_episode_split_is_deterministic_stratified_and_disjoint():
    sources = np.repeat(np.arange(6), 10)

    first = build_episode_split_manifest(sources, val_ratio=0.1, seed=42)
    second = build_episode_split_manifest(sources, val_ratio=0.1, seed=42)

    assert first == second
    assert len(first["validation_episode_ids"]) == 6
    assert set(first["train_episode_ids"]).isdisjoint(first["validation_episode_ids"])
    assert sorted(first["train_episode_ids"] + first["validation_episode_ids"]) == list(
        range(60)
    )
    assert {sources[index] for index in first["validation_episode_ids"]} == set(range(6))
    assert len(first["split_digest"]) == 64


def test_idle_metrics_integrate_translation_and_compose_so3_rotations():
    target = _neutral_actions()
    prediction = target.copy()
    prediction[..., 10] = 0.001
    prediction[..., 13:19] = _rotation_6d_z(1.0)
    idle_mask = np.zeros((1, 29, 2), dtype=bool)
    idle_mask[..., 1] = True

    metrics = compute_idle_rollout_metrics(target, prediction, idle_mask, horizon=29)

    assert metrics["val_idle_translation_29_mm"] == pytest.approx(29.0)
    assert metrics["val_idle_rotation_29_deg"] == pytest.approx(29.0, abs=1e-5)
    assert metrics["val_idle_translation_step_p95_mm"] == pytest.approx(1.0)
    assert metrics["val_idle_rotation_step_p95_deg"] == pytest.approx(1.0)
    assert metrics["val_idle_right_translation_29_mm"] == pytest.approx(29.0)
    assert math.isnan(metrics["val_idle_left_translation_29_mm"])


def test_deployment_window_metrics_only_measure_the_deployed_phase():
    target = _neutral_actions(horizon=32)
    idle_mask = np.zeros((1, 32, 2), dtype=bool)
    idle_mask[..., 0] = True
    prediction = target.copy()
    prediction[:, 3, 3:9] = _rotation_6d_z(2.0)
    prediction[:, 18, 3:9] = _rotation_6d_z(4.0)

    metrics = compute_deployment_window_metrics(
        target,
        prediction,
        idle_mask,
        phase_start=3,
        phase_count=16,
    )

    assert metrics["val_deploy_idle_rotation_step_p95_deg"] == pytest.approx(2.5)
    assert metrics["val_deploy_idle_rotation_window_deg"] == pytest.approx(6.0)
    assert metrics["val_deploy_idle_left_rotation_window_deg"] == pytest.approx(6.0)
    assert "val_deploy_idle_rotation_29_deg" not in metrics
    assert "val_deploy_idle_left_rotation_29_deg" not in metrics

    prediction = target.copy()
    prediction[:, 19:32, 3:9] = _rotation_6d_z(10.0)
    metrics = compute_deployment_window_metrics(
        target,
        prediction,
        idle_mask,
        phase_start=3,
        phase_count=16,
    )

    assert metrics["val_deploy_idle_rotation_step_p95_deg"] == pytest.approx(0.0)


@pytest.mark.parametrize(
    ("phase_start", "phase_count"),
    [(-1, 16), (0, 0), (17, 16)],
)
def test_deployment_window_metrics_reject_invalid_phase_bounds(
    phase_start, phase_count
):
    target = _neutral_actions(horizon=32)
    idle_mask = np.ones((1, 32, 2), dtype=bool)

    with pytest.raises(ValueError):
        compute_deployment_window_metrics(
            target,
            target.copy(),
            idle_mask,
            phase_start=phase_start,
            phase_count=phase_count,
        )


@pytest.mark.parametrize("action_dim", [10, 20])
def test_canonical_noop_actions_replace_pose_and_preserve_grippers(action_dim):
    actions = torch.randn(2, 3, action_dim)
    original = actions.clone()

    noops = build_canonical_noop_actions(actions)

    torch.testing.assert_close(noops[..., 0::10], torch.zeros_like(noops[..., 0::10]))
    torch.testing.assert_close(noops[..., 1::10], torch.zeros_like(noops[..., 1::10]))
    torch.testing.assert_close(noops[..., 2::10], torch.zeros_like(noops[..., 2::10]))
    expected_rotation = torch.tensor(
        [1, 0, 0, 0, 1, 0], dtype=actions.dtype
    ).expand_as(noops[..., 3:9])
    for arm_start in range(0, action_dim, 10):
        torch.testing.assert_close(
            noops[..., arm_start + 3 : arm_start + 9],
            expected_rotation[..., :6],
        )
        torch.testing.assert_close(noops[..., arm_start + 9], original[..., arm_start + 9])
    torch.testing.assert_close(actions, original)


@pytest.mark.parametrize(
    ("actions", "error"),
    [
        (torch.zeros(2, 3, 9), ValueError),
        (torch.zeros(2, 3, 30), ValueError),
        (torch.zeros(2, 10, 3).transpose(1, 2), ValueError),
        (np.zeros((2, 3, 10)), TypeError),
    ],
)
def test_canonical_noop_actions_reject_unsupported_layouts(actions, error):
    with pytest.raises(error):
        build_canonical_noop_actions(actions)


def test_metrics_report_active_errors_and_micro_motion_recall():
    target = _neutral_actions(horizon=4)
    prediction = target.copy()
    idle_mask = np.ones((1, 4, 2), dtype=bool)
    idle_mask[..., 0] = False
    target[..., 0] = 0.0006
    prediction[:, :3, 0] = 0.0006
    prediction[:, 3, 0] = 0.0
    prediction[..., 9] = 0.002

    metrics = compute_idle_rollout_metrics(target, prediction, idle_mask, horizon=4)

    assert metrics["val_active_left_translation_mae_mm"] == pytest.approx(0.15)
    assert metrics["val_active_left_translation_bias_x_mm"] == pytest.approx(-0.15)
    assert metrics["val_active_left_translation_bias_y_mm"] == pytest.approx(0.0)
    assert metrics["val_active_left_translation_bias_z_mm"] == pytest.approx(0.0)
    assert metrics["val_active_left_translation_p50_mm"] == pytest.approx(0.0)
    assert metrics["val_active_left_translation_p95_mm"] == pytest.approx(0.51)
    assert metrics["val_active_left_rotation_p50_deg"] == pytest.approx(0.0)
    assert metrics["val_active_left_gripper_mae_mm"] == pytest.approx(2.0)
    assert metrics["val_micro_motion_recall"] == pytest.approx(0.75)


@pytest.mark.parametrize(
    ("translation", "rotation", "recall", "feasible"),
    [
        (1.05, 2.10, 0.40, True),
        (1.051, 2.0, 0.40, False),
        (1.0, 2.101, 0.40, False),
        (1.0, 2.0, 0.399, False),
    ],
)
def test_checkpoint_feasibility_enforces_separate_active_and_micro_motion_limits(
    translation, rotation, recall, feasible
):
    result = evaluate_checkpoint_feasibility(
        idle_translation_29_mm=0.4,
        idle_rotation_29_deg=0.1,
        idle_translation_p95_mm=0.04,
        idle_rotation_p95_deg=0.02,
        active_translation_mm=translation,
        active_translation_baseline_mm=1.0,
        active_rotation_deg=rotation,
        active_rotation_baseline_deg=2.0,
        micro_motion_recall=recall,
    )

    assert result["val_active_translation_degradation"] == pytest.approx(
        translation / 1.0 - 1.0
    )
    assert result["val_active_rotation_degradation"] == pytest.approx(
        rotation / 2.0 - 1.0
    )
    assert result["val_idle_score"] == pytest.approx(0.6)
    assert result["val_checkpoint_feasible"] is feasible
    assert result["val_deployable"] is feasible


@pytest.mark.parametrize(
    ("recall", "feasible"),
    [
        (0.4681236080178174, True),
        (0.255707, False),
    ],
)
def test_micro_motion_gate_accepts_new_run_but_rejects_old_run(recall, feasible):
    result = evaluate_checkpoint_feasibility(
        idle_translation_29_mm=0.4,
        idle_rotation_29_deg=0.1,
        idle_translation_p95_mm=0.04,
        idle_rotation_p95_deg=0.02,
        active_translation_mm=1.0,
        active_translation_baseline_mm=1.0,
        active_rotation_deg=2.0,
        active_rotation_baseline_deg=2.0,
        micro_motion_recall=recall,
    )

    assert result["val_checkpoint_feasible"] is feasible
    assert result["val_deployable"] is feasible


@pytest.mark.parametrize(
    ("metric_name", "metric_value"),
    [
        ("idle_translation_29_mm", 1.0),
        ("idle_rotation_29_deg", 0.5),
        ("idle_translation_p95_mm", 0.05),
        ("idle_rotation_p95_deg", 0.03),
    ],
)
def test_deployable_enforces_strict_idle_release_limits(metric_name, metric_value):
    values = {
        "idle_translation_29_mm": 0.9,
        "idle_rotation_29_deg": 0.4,
        "idle_translation_p95_mm": 0.04,
        "idle_rotation_p95_deg": 0.02,
        "active_translation_mm": 1.0,
        "active_translation_baseline_mm": 1.0,
        "active_rotation_deg": 2.0,
        "active_rotation_baseline_deg": 2.0,
        "micro_motion_recall": 1.0,
    }
    values[metric_name] = metric_value

    result = evaluate_checkpoint_feasibility(**values)

    assert result["val_checkpoint_feasible"] is True
    assert result["val_deployable"] is False


def test_v2_baseline_json_is_optional_and_keeps_units_separate(tmp_path):
    config = OmegaConf.create(
        {
            "task": {"action_representation_version": 2},
            "validation": {"baseline_json": None},
        }
    )

    assert load_active_metric_baselines(config) is None

    baseline_path = tmp_path / "frozen-v1.json"
    baseline_path.write_text(
        json.dumps(
            {
                "val_active_left_translation_mae_mm": 1.25,
                "val_active_left_rotation_mae_deg": 2.5,
            }
        ),
        encoding="utf-8",
    )
    config.validation.baseline_json = str(baseline_path)

    assert load_active_metric_baselines(config) == {
        "translation_mm": 1.25,
        "rotation_deg": 2.5,
    }


def test_missing_baseline_keeps_checkpoint_fail_closed():
    result = evaluate_checkpoint_feasibility(
        idle_translation_29_mm=0.1,
        idle_rotation_29_deg=0.1,
        idle_translation_p95_mm=0.01,
        idle_rotation_p95_deg=0.01,
        active_translation_mm=0.1,
        active_translation_baseline_mm=None,
        active_rotation_deg=0.1,
        active_rotation_baseline_deg=None,
        micro_motion_recall=1.0,
    )

    assert result["val_checkpoint_feasible"] is False
    assert result["val_deployable"] is False


def test_auto_baseline_calibrates_valid_metrics_once_and_freezes():
    calibrated = resolve_active_metric_baselines(
        external_baselines=None,
        auto_translation_baseline_mm=None,
        auto_rotation_baseline_deg=None,
        active_translation_mm=1.25,
        active_rotation_deg=2.5,
        epoch=4,
    )

    assert calibrated == {
        "translation_mm": 1.25,
        "rotation_deg": 2.5,
        "source": "auto",
        "epoch": 4,
        "calibrated": True,
    }

    frozen = resolve_active_metric_baselines(
        external_baselines=None,
        auto_translation_baseline_mm=calibrated["translation_mm"],
        auto_rotation_baseline_deg=calibrated["rotation_deg"],
        auto_baseline_epoch=calibrated["epoch"],
        active_translation_mm=0.5,
        active_rotation_deg=0.75,
        epoch=9,
    )

    assert frozen == {
        "translation_mm": 1.25,
        "rotation_deg": 2.5,
        "source": "auto",
        "epoch": 4,
        "calibrated": False,
    }


@pytest.mark.parametrize(
    ("translation", "rotation"),
    [(math.nan, 2.5), (1.25, 0.0)],
)
def test_auto_baseline_skips_invalid_active_metrics(translation, rotation):
    result = resolve_active_metric_baselines(
        external_baselines=None,
        auto_translation_baseline_mm=None,
        auto_rotation_baseline_deg=None,
        active_translation_mm=translation,
        active_rotation_deg=rotation,
        epoch=4,
    )

    assert result == {
        "translation_mm": None,
        "rotation_deg": None,
        "source": None,
        "epoch": None,
        "calibrated": False,
    }


def test_external_baseline_has_priority_over_auto_baseline_state():
    result = resolve_active_metric_baselines(
        external_baselines={"translation_mm": 3.0, "rotation_deg": 4.0},
        auto_translation_baseline_mm=1.25,
        auto_rotation_baseline_deg=2.5,
        active_translation_mm=0.5,
        active_rotation_deg=0.75,
        epoch=9,
    )

    assert result == {
        "translation_mm": 3.0,
        "rotation_deg": 4.0,
        "source": "external",
        "epoch": None,
        "calibrated": False,
    }


def test_auto_baseline_calibration_forces_that_release_non_deployable():
    baseline = resolve_active_metric_baselines(
        external_baselines=None,
        auto_translation_baseline_mm=None,
        auto_rotation_baseline_deg=None,
        active_translation_mm=1.0,
        active_rotation_deg=2.0,
        epoch=4,
    )
    release = evaluate_checkpoint_feasibility(
        idle_translation_29_mm=0.1,
        idle_rotation_29_deg=0.1,
        idle_translation_p95_mm=0.01,
        idle_rotation_p95_deg=0.01,
        active_translation_mm=1.0,
        active_translation_baseline_mm=baseline["translation_mm"],
        active_rotation_deg=2.0,
        active_rotation_baseline_deg=baseline["rotation_deg"],
        micro_motion_recall=1.0,
    )

    release["val_deployable"] = bool(
        release["val_deployable"] and not baseline["calibrated"]
    )
    assert release["val_deployable"] is False


def test_seeded_validation_is_repeatable_and_preserves_global_rng():
    random.seed(101)
    np.random.seed(102)
    torch.manual_seed(103)
    expected_after = (random.random(), np.random.rand(), torch.rand(3))

    random.seed(101)
    np.random.seed(102)
    torch.manual_seed(103)
    seeded_values = []
    for _ in range(2):
        with preserve_global_rng_state(7):
            seeded_values.append((random.random(), np.random.rand(), torch.rand(3)))
    actual_after = (random.random(), np.random.rand(), torch.rand(3))

    assert seeded_values[0][0] == seeded_values[1][0]
    assert seeded_values[0][1] == seeded_values[1][1]
    torch.testing.assert_close(seeded_values[0][2], seeded_values[1][2], rtol=0, atol=0)
    assert actual_after[0] == expected_after[0]
    assert actual_after[1] == expected_after[1]
    torch.testing.assert_close(actual_after[2], expected_after[2], rtol=0, atol=0)


def test_contiguous_300_step_drift_requires_real_contiguous_actions():
    target = _neutral_actions(horizon=300)
    prediction = target.copy()
    prediction[..., 10] = 1e-6
    idle_mask = np.zeros((1, 300, 2), dtype=bool)
    idle_mask[..., 1] = True

    metrics = compute_contiguous_300_step_drift(target, prediction, idle_mask)

    assert metrics["val_idle_translation_300_mm"] == pytest.approx(0.3)
    with pytest.raises(ValueError, match="300 contiguous"):
        compute_contiguous_300_step_drift(
            target[:, :32], prediction[:, :32], idle_mask[:, :32]
        )


def test_resume_rejects_action_contract_version_mismatch():
    current = OmegaConf.create(
        {"task": {"action_representation_version": 2, "action_contract": "v2"}}
    )
    legacy = OmegaConf.create(
        {"task": {"action_representation_version": 1, "action_contract": "v1"}}
    )

    validate_resume_action_contract(current, current)
    with pytest.raises(ValueError, match="action-contract version"):
        validate_resume_action_contract(current, legacy)


def test_pick_tube_v2_configs_select_feasible_idle_score():
    OmegaConf.register_new_resolver("eval", eval, replace=True)
    config_dir = str(
        Path(__file__).resolve().parents[1] / "reactive_diffusion_policy" / "config"
    )
    with initialize_config_dir(version_base=None, config_dir=config_dir):
        at_cfg = compose(config_name="train_pick_tube_at_workspace")
        ldp_cfg = compose(config_name="train_pick_tube_ldp_workspace")
        single_right_at_cfg = compose(
            config_name="train_pick_tube_single_right_at_workspace"
        )
        single_right_ldp_cfg = compose(
            config_name="train_pick_tube_single_right_ldp_workspace"
        )

    for cfg, action_contract in (
        (at_cfg, "bimanual_relative_pose20d_v2"),
        (ldp_cfg, "bimanual_relative_pose20d_v2"),
        (single_right_at_cfg, "single_right_relative_pose10d_v2"),
        (single_right_ldp_cfg, "single_right_relative_pose10d_v2"),
    ):
        assert cfg.task.dataset.val_ratio == 0.1
        assert cfg.task.action_representation_version == 2
        assert cfg.task.action_contract == action_contract
        assert cfg.checkpoint.topk.monitor_key == "val_deploy_idle_score"
        assert cfg.checkpoint.topk.mode == "min"
        assert cfg.validation.max_active_degradation == 0.05
        assert cfg.validation.min_micro_motion_recall == 0.40
        assert cfg.validation.baseline_json is None
        assert list(cfg.validation.seeds) == [0]
        assert cfg.validation.deployment_slow_update_interval == 16
        assert cfg.checkpoint.periodic.keep == 10


def test_v2_experiment_launcher_uses_fresh_20_epoch_runs():
    script = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "run_pick_tube_rdp_experiments.sh"
    ).read_text(encoding="utf-8")

    assert '"AT_EPOCHS=20"' in script
    assert '"LDP_EPOCHS=20"' in script
    assert '"LDP_EPOCH=15"' not in script
    assert '"RESUME=false"' in script
    assert "pca%d_armwise_rdp_zarr_v2" in script
    assert "BASELINE_JSON" in script


def test_pick_tube_launchers_allow_missing_baseline():
    root = Path(__file__).resolve().parents[1]
    generic_launcher = root / "scripts" / "train_pick_tube_single_gpu.sh"
    generic_script = generic_launcher.read_text(encoding="utf-8")
    assert "BASELINE_JSON is required" not in generic_script
    assert "AT/LDP will auto-calibrate on the first valid deployment validation" in generic_script

    recovery_only_launchers = (
        root / "scripts" / "run_pick_tube_rdp_experiments.sh",
        root / "scripts" / "train_pick_tube_server.sh",
        root / "train_pick_tube_rdp.sh",
    )

    for launcher in recovery_only_launchers:
        script = launcher.read_text(encoding="utf-8")
        assert "BASELINE_JSON is required" not in script
        assert "checkpoints will remain non-deployable" in script


def test_single_right_launcher_requires_deployable_at_and_ldp_checkpoints():
    script = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "train_pick_tube_single_right_gpu.sh"
    ).read_text(encoding="utf-8")

    assert "AT_CKPT=${AT_CKPT:-${AT_DIR}/checkpoints/deployable.ckpt}" in script
    assert "${AT_DIR}/checkpoints/latest.ckpt" not in script
    assert "latest is recovery-only" in script
    assert "AT deployable checkpoint not found" in script
    assert "${LDP_DIR}/checkpoints/deployable.ckpt" in script
    assert "AT/LDP will auto-calibrate on the first valid deployment validation" in script
    assert "checkpoints will remain non-deployable" not in script


def test_generic_launcher_requires_deployable_at_and_ldp_checkpoints():
    script = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "train_pick_tube_single_gpu.sh"
    ).read_text(encoding="utf-8")

    assert "AT_CKPT=${AT_CKPT:-${AT_DIR}/checkpoints/deployable.ckpt}" in script
    assert "${AT_DIR}/checkpoints/latest.ckpt" not in script
    assert "latest is recovery-only" in script
    assert "AT deployable checkpoint not found" in script
    assert "${LDP_DIR}/checkpoints/deployable.ckpt" in script



def test_single_right_metrics_and_baseline_use_right_arm(tmp_path):
    target = np.zeros((1, 29, 10), dtype=np.float64)
    target[..., 3:9] = IDENTITY_6D
    prediction = target.copy()
    prediction[..., 0] = 0.001
    idle_mask = np.ones((1, 29, 1), dtype=bool)

    metrics = compute_idle_rollout_metrics(
        target,
        prediction,
        idle_mask,
        horizon=29,
        state_action_profile="single-right-arm-7x10",
    )
    assert metrics["val_idle_right_translation_29_mm"] == pytest.approx(29.0)
    assert "val_idle_left_translation_29_mm" not in metrics

    baseline_path = tmp_path / "single-right-baseline.json"
    baseline_path.write_text(
        json.dumps(
            {
                "val_active_right_translation_mae_mm": 1.5,
                "val_active_right_rotation_mae_deg": 2.5,
            }
        ),
        encoding="utf-8",
    )
    config = OmegaConf.create(
        {
            "task": {"controlled_arms": ["right"]},
            "validation": {"baseline_json": str(baseline_path)},
        }
    )
    assert load_active_metric_baselines(config) == {
        "translation_mm": 1.5,
        "rotation_deg": 2.5,
    }
