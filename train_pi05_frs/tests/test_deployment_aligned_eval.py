from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from train_pi05_frs.tools.evaluate_deployment_checkpoints import (
    _flatten_metrics,
    discover_validation_checkpoints,
)
from train_pi05_frs.utils.deployment_metrics import (
    deployment_aligned_single_hand_metrics,
)


def test_deployment_metrics_exclude_raw_frs_gripper() -> None:
    gt = np.zeros((2, 2, 10), dtype=np.float32)
    vla = np.ones_like(gt)
    vla[..., 9] = 2.0
    frs = np.array(vla, copy=True)
    frs[1, ..., :9] = 0.0
    frs[..., 9] = 99.0
    gates = np.asarray([0.1, 0.9], dtype=np.float32)

    metrics = deployment_aligned_single_hand_metrics(
        frs,
        gt,
        vla,
        gates,
        gripper_index=9,
        low_gate_threshold=0.3,
        high_gate_threshold=0.7,
        low_gate_safety_margin=0.03,
        low_gate_regression_margin=0.005,
        rank_margin=0.0,
        repair_margin=0.0,
    )

    arm9 = metrics["arm9"]
    assert arm9["mse_frs_gt"] == pytest.approx(0.5)
    assert arm9["mse_frs_vla"] == pytest.approx(0.5)
    assert arm9["mse_vla_gt"] == pytest.approx(1.0)
    assert arm9["high_gain"] == pytest.approx(1.0)
    assert arm9["high_rank_satisfied_frac"] == pytest.approx(1.0)
    assert arm9["high_repair_satisfied_frac"] == pytest.approx(1.0)
    assert arm9["low_safe_frac"] == pytest.approx(1.0)
    assert arm9["low_gate_regression_frac"] == pytest.approx(0.0)
    assert arm9["high_gate_harm_p95"] == pytest.approx(0.0)

    runtime10 = metrics["runtime10"]
    assert runtime10["mse_frs_gt"] == pytest.approx(0.85)
    assert runtime10["mse_frs_vla"] == pytest.approx(0.45)
    assert runtime10["mse_vla_gt"] == pytest.approx(1.3)
    assert runtime10["high_gain"] == pytest.approx(0.9)
    assert runtime10["high_rank_satisfied_frac"] == pytest.approx(1.0)
    assert runtime10["high_repair_satisfied_frac"] == pytest.approx(1.0)
    assert runtime10["low_safe_frac"] == pytest.approx(1.0)
    assert runtime10["low_gate_regression_frac"] == pytest.approx(0.0)
    assert runtime10["high_gate_harm_p95"] == pytest.approx(0.0)


def test_deployment_low_gate_safety_means_direct_vla_preservation() -> None:
    gt = np.zeros((2, 1, 10), dtype=np.float32)
    vla = np.ones_like(gt)
    frs = np.array(vla, copy=True)
    frs[0, ..., :9] = 0.0
    gates = np.asarray([0.1, 0.9], dtype=np.float32)

    metrics = deployment_aligned_single_hand_metrics(
        frs,
        gt,
        vla,
        gates,
        gripper_index=9,
        low_gate_threshold=0.3,
        high_gate_threshold=0.7,
        low_gate_safety_margin=0.03,
        low_gate_regression_margin=0.005,
        rank_margin=0.0,
        repair_margin=0.0,
    )

    assert metrics["arm9"]["low_safe_frac"] == pytest.approx(0.0)
    assert metrics["arm9"]["low_unsafe_frac"] == pytest.approx(1.0)


def test_deployment_checkpoint_feasibility_uses_new_harm_constraints() -> None:
    values = {
        "low_unsafe_frac": 0.01,
        "low_gate_regression_frac": 0.06,
        "high_gain": 0.1,
        "high_rank_satisfied_frac": 0.9,
        "high_repair_satisfied_frac": 0.9,
        "high_gate_harm_p95": 0.02,
    }

    row = _flatten_metrics(
        epoch=1,
        checkpoint=Path("checkpoint"),
        summaries={"arm9": values},
        max_low_gate_unsafe_frac=0.05,
        min_high_gate_gain=0.0,
        min_high_gate_repair_satisfied_frac=0.8,
        max_high_gate_harm_p95=0.03,
        max_low_gate_regression_frac=0.05,
    )

    assert row["arm9_checkpoint_feasible"] == 0


def test_deployment_metrics_reject_empty_gate_regions() -> None:
    actions = np.zeros((2, 3, 10), dtype=np.float32)
    with pytest.raises(ValueError, match="high-Gate region is empty"):
        deployment_aligned_single_hand_metrics(
            actions,
            actions,
            actions,
            np.asarray([0.1, 0.2]),
            gripper_index=9,
            low_gate_threshold=0.3,
            high_gate_threshold=0.7,
            low_gate_safety_margin=0.03,
            low_gate_regression_margin=0.005,
            rank_margin=0.0,
            repair_margin=0.0,
        )


def _write_generation(
    run_dir: Path,
    name: str,
    *,
    epoch: int,
    with_validation: bool,
) -> Path:
    generation = run_dir / ".checkpoint-generations" / name
    generation.mkdir(parents=True)
    metrics = {"val_mse_gt": 0.1} if with_validation else {"train_loss": 0.2}
    (generation / "checkpoint.json").write_text(
        json.dumps({"epoch": epoch, "metrics": metrics}),
        encoding="utf-8",
    )
    return generation


def test_discover_validation_checkpoints_prefers_evaluated_generation(
    tmp_path: Path,
) -> None:
    unevaluated = _write_generation(
        tmp_path, "epoch-10-last", epoch=10, with_validation=False
    )
    evaluated = _write_generation(
        tmp_path, "epoch-10-best", epoch=10, with_validation=True
    )
    epoch_15 = _write_generation(
        tmp_path, "epoch-15-last", epoch=15, with_validation=True
    )

    selected = discover_validation_checkpoints(tmp_path, epochs=(10, 15))

    assert selected == {10: evaluated, 15: epoch_15}
    assert selected[10] != unevaluated


def test_discover_validation_checkpoints_reports_missing_epoch(tmp_path: Path) -> None:
    _write_generation(tmp_path, "epoch-10", epoch=10, with_validation=True)
    with pytest.raises(ValueError, match=r"missing validation checkpoints.*15"):
        discover_validation_checkpoints(tmp_path, epochs=(10, 15))
