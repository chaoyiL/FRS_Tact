from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from train_pi05_frs.utils.history_plot import HISTORY_FIELDS
from train_pi05_frs.utils.metrics import EvaluationResult


def write_bimanual_history_fixture(path: Path) -> Path:
    quadrants = ("low_low", "high_low", "low_high", "high_high")
    wrists = ("left", "right")
    fields = list(HISTORY_FIELDS) + [
        "train_gate_w_left",
        "train_gate_w_right",
        "train_loss_total",
        "train_loss_decode",
        "train_loss_rank",
        "train_loss_low_safety",
        "train_loss_repair",
        "train_loss_composite_fm",
        "val_composite_fm",
        "val_mse_vla_gt",
        "val_gt_gain",
        "val_relative_gt_error",
        "checkpoint_selection_feasible",
    ]
    fields.extend(
        f"val_{field}_{wrist}"
        for wrist in wrists
        for field in (
            "low_safe_frac",
            "rank_satisfied_high_frac",
            "gate_w",
            "gate_w_p10",
            "gate_w_p50",
            "gate_w_p90",
            "tactile_change_p10",
            "tactile_change_p50",
            "tactile_change_p90",
            "n_low_w",
            "n_mid_w",
            "n_high_w",
        )
    )
    fields.extend(f"val_quadrant_{quadrant}_n" for quadrant in quadrants)
    fields.extend(
        f"val_quadrant_{quadrant}_{metric}_{wrist}"
        for quadrant in quadrants
        for wrist in wrists
        for metric in (
            "mse_gt",
            "mse_vla",
            "mse_vla_gt",
            "gt_gain",
            "relative_gt_error",
            "vla_preserve_ratio",
            "rank_satisfied_frac",
        )
    )
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for epoch in (1, 2):
            row = {field: 0.2 for field in fields}
            row.update({"epoch": epoch, "val_quadrant_high_high_n": 0})
            for wrist in wrists:
                row.update(
                    {
                        f"val_gate_w_{wrist}": 0.5,
                        f"val_gate_w_p10_{wrist}": 0.2,
                        f"val_gate_w_p50_{wrist}": 0.5,
                        f"val_gate_w_p90_{wrist}": 0.8,
                        f"val_tactile_change_p10_{wrist}": 0.1,
                        f"val_tactile_change_p50_{wrist}": 0.2,
                        f"val_tactile_change_p90_{wrist}": 0.3,
                        f"val_n_low_w_{wrist}": 2,
                        f"val_n_mid_w_{wrist}": 1,
                        f"val_n_high_w_{wrist}": 2,
                    }
                )
            writer.writerow(row)
    return path


def write_legacy_history_fixture(path: Path) -> Path:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=("epoch", "train_flow_loss", "val_flow_loss"))
        writer.writeheader()
        writer.writerow({"epoch": 1, "train_flow_loss": 0.3, "val_flow_loss": 0.2})
    return path


def make_bimanual_evaluation_result(*, action_dim: int) -> EvaluationResult:
    samples, horizon = 4, 3
    zeros = np.zeros(samples, dtype=np.float64)
    actions = np.zeros((samples, horizon, action_dim), dtype=np.float32)
    left_gate = np.asarray((0.85, 0.85, 0.15, 0.15))
    right_gate = np.asarray((0.15, 0.15, 0.85, 0.85))
    return EvaluationResult(
        target="gt", flow_loss=0.0, mse=0.0, rmse=0.0, mae=0.0,
        flow_loss_gt=0.0, mse_gt=0.0, rmse_gt=0.0, mae_gt=0.0,
        flow_loss_pred=0.0, mse_pred=0.0, rmse_pred=0.0, mae_pred=0.0,
        cache_indices=np.arange(samples), sample_flow_loss=zeros, sample_mse=zeros,
        sample_rmse=zeros, sample_mae=zeros, sample_mse_gt=zeros, sample_mae_gt=zeros,
        sample_mse_pred=zeros, sample_mae_pred=zeros, predictions=actions,
        sample_gate_w_left=left_gate, sample_gate_w_right=right_gate,
        sample_mse_gt_left=zeros, sample_mse_gt_right=zeros,
        sample_mse_vla_left=zeros, sample_mse_vla_right=zeros,
        sample_mse_vla_gt_left=zeros, sample_mse_vla_gt_right=zeros,
        bimanual_gate_region_counts=np.asarray(((0, 0, 2), (0, 0, 0), (2, 0, 0))),
        gt_actions=actions, vla_actions=actions,
    )


def test_bimanual_plot_bundle_writes_stable_filenames(tmp_path: Path) -> None:
    from train_pi05_frs.utils.bimanual_visualize import plot_bimanual_diagnostics

    history = write_bimanual_history_fixture(tmp_path / "history.csv")
    result = make_bimanual_evaluation_result(action_dim=32)
    paths = plot_bimanual_diagnostics(history, result, output_dir=tmp_path)
    assert {path.name for path in paths} == {
        "training_overview.png", "bimanual_behavior.png",
        "gate_diagnostics.png", "bimanual_action_examples.png",
    }
    assert all(path.stat().st_size > 0 for path in paths)


def test_legacy_history_still_writes_training_curves(tmp_path: Path) -> None:
    from train_pi05_frs.utils.history_plot import plot_training_history

    history = write_legacy_history_fixture(tmp_path / "history.csv")
    output = plot_training_history(history)
    assert output.name == "training_curves.png"
    assert output.stat().st_size > 0
