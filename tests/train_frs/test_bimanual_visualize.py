from __future__ import annotations

import csv
import inspect
from pathlib import Path
from unittest import mock

import numpy as np
import pytest

import train_smolvla_frs.utils.bimanual_visualize as bimanual_visualize
from train_smolvla_frs.utils.history_plot import HISTORY_FIELDS
from train_smolvla_frs.utils.metrics import EvaluationResult


def _write_bimanual_history(
    path: Path,
    *,
    sample_counts: dict[str, int] | None = None,
    include_overview_fields: bool = True,
) -> None:
    overview_fields = (
        "val_low_safe_frac_left",
        "val_low_safe_frac_right",
        "val_rank_satisfied_high_frac_left",
        "val_rank_satisfied_high_frac_right",
    )
    fieldnames = HISTORY_FIELDS + (overview_fields if include_overview_fields else ())
    sample_counts = sample_counts or {}
    rows = []
    for epoch, scale in ((1, 1.0), (2, 0.8)):
        row = dict.fromkeys(fieldnames, "")
        row.update(
            {
                "epoch": epoch,
                "train_loss_total": 0.5 * scale,
                "train_loss_gt_fm": 0.2 * scale,
                "train_loss_vla_fm": 0.1 * scale,
                "train_loss_composite_fm": 0.3 * scale,
                "train_loss_low_safety": 0.04 * scale,
                "train_loss_rank": 0.03 * scale,
                "train_loss_repair": 0.02 * scale,
                "train_gate_w_left": 0.8,
                "train_gate_w_right": 0.2,
                "val_composite_fm": 0.25 * scale,
            }
        )
        if include_overview_fields:
            row.update(
                {
                    "val_low_safe_frac_left": 0.95,
                    "val_low_safe_frac_right": 0.90,
                    "val_rank_satisfied_high_frac_left": 0.85,
                    "val_rank_satisfied_high_frac_right": 0.82,
                }
            )
        for quadrant in ("low_low", "high_low", "low_high", "high_high"):
            row[f"val_quadrant_{quadrant}_n"] = sample_counts.get(quadrant, 24)
            for wrist in ("left", "right"):
                row[f"val_quadrant_{quadrant}_mse_gt_{wrist}"] = 0.2 * scale
                row[f"val_quadrant_{quadrant}_mse_vla_{wrist}"] = 0.4 * scale
                row[f"val_quadrant_{quadrant}_mse_vla_gt_{wrist}"] = 0.5 * scale
                row[f"val_quadrant_{quadrant}_gt_gain_{wrist}"] = 0.3 * scale
                row[f"val_quadrant_{quadrant}_relative_gt_error_{wrist}"] = 0.4
                row[f"val_quadrant_{quadrant}_vla_preserve_ratio_{wrist}"] = 0.8
                row[f"val_quadrant_{quadrant}_rank_satisfied_frac_{wrist}"] = 0.9
        rows.append(row)

    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def test_overview_and_behavior_render_expected_panels_and_low_sample_notice(tmp_path: Path) -> None:
    history = tmp_path / "history.csv"
    _write_bimanual_history(
        history,
        sample_counts={"high_low": 8, "high_high": 0},
    )
    real_subplots = bimanual_visualize.plt.subplots
    figures = {}

    def capture_overview(*args, **kwargs):
        figure, axes = real_subplots(*args, **kwargs)
        figures["overview"] = figure
        return figure, axes

    def capture_behavior(*args, **kwargs):
        figure, axes = real_subplots(*args, **kwargs)
        figures["behavior"] = figure
        return figure, axes

    with mock.patch.object(
        bimanual_visualize.plt,
        "subplots",
        side_effect=capture_overview,
    ) as overview_subplots:
        overview = bimanual_visualize.plot_bimanual_training_overview(
            history,
            output_path=tmp_path / "training_overview.png",
        )
    with mock.patch.object(
        bimanual_visualize.plt,
        "subplots",
        side_effect=capture_behavior,
    ) as behavior_subplots:
        behavior = bimanual_visualize.plot_bimanual_behavior(
            history,
            output_path=tmp_path / "bimanual_behavior.png",
        )

    assert overview.is_file() and overview.stat().st_size > 0
    assert behavior.is_file() and behavior.stat().st_size > 0
    assert overview_subplots.call_args.args[:2] == (6, 1)
    assert behavior_subplots.call_args.args[:2] == (4, 2)
    assert any(
        "Insufficient samples" in text.get_text()
        for text in figures["behavior"].axes[2].texts
    )
    high_low_right = figures["behavior"].axes[3]
    low_high_left = figures["behavior"].axes[4]
    for axis in (high_low_right, low_high_left):
        lines = {line.get_label(): line for line in axis.lines}
        assert lines["VLA preserved (expected)"].get_alpha() == 1.0
        assert lines["VLA preserved (expected)"].get_linewidth() > lines[
            "FRS vs GT (reference)"
        ].get_linewidth()
        assert lines["FRS vs GT (reference)"].get_alpha() < 1.0
    high_low_left_text = "\n".join(
        text.get_text() for text in figures["behavior"].axes[2].texts
    )
    assert "MSE(FRS,GT)=" in high_low_left_text
    assert "MSE(FRS,VLA)=" in high_low_left_text
    assert "MSE(VLA,GT)=" in high_low_left_text
    assert "gain=" in high_low_left_text
    assert "rank satisfied=" in high_low_left_text
    high_high_right_text = "\n".join(
        text.get_text() for text in figures["behavior"].axes[7].texts
    )
    assert "No validation samples" in high_high_right_text
    assert "Insufficient samples" not in high_high_right_text
    assert "MSE(" not in high_high_right_text


@pytest.mark.parametrize(
    "plotter",
    (
        bimanual_visualize.plot_bimanual_training_overview,
        bimanual_visualize.plot_bimanual_behavior,
    ),
)
def test_bimanual_plots_reject_legacy_history_with_specific_error(
    tmp_path: Path,
    plotter,
) -> None:
    history = tmp_path / "history.csv"
    with history.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=("epoch", "train_loss_total"))
        writer.writeheader()
        writer.writerow({"epoch": 1, "train_loss_total": 0.3})

    with pytest.raises(ValueError, match="^bimanual history fields are absent$"):
        plotter(history, output_path=tmp_path / "unused.png")


def test_overview_requires_per_wrist_feasibility_history_fields(tmp_path: Path) -> None:
    history = tmp_path / "history.csv"
    _write_bimanual_history(history, include_overview_fields=False)

    with pytest.raises(ValueError, match="^bimanual history fields are absent$"):
        bimanual_visualize.plot_bimanual_training_overview(
            history,
            output_path=tmp_path / "training_overview.png",
        )

    output = bimanual_visualize.plot_bimanual_behavior(
        history,
        output_path=tmp_path / "bimanual_behavior.png",
    )
    assert output.is_file()


def _bimanual_result_with_mixed_quadrants() -> EvaluationResult:
    cache_indices = np.asarray([10, 11, 12, 13, 14, 15, 16], dtype=np.int64)
    actions = np.zeros((len(cache_indices), 3, 20), dtype=np.float32)
    for position in range(len(cache_indices)):
        actions[position, :, 9] = position + 1
        actions[position, :, 19] = 10 + position
    prediction = actions + 0.25
    gt_action = actions + 0.5
    vla_action = actions
    left_gate = np.asarray([0.85, 0.85, 0.85, 0.15, 0.15, 0.15, 0.75])
    right_gate = np.asarray([0.15, 0.15, 0.15, 0.85, 0.85, 0.85, 0.25])
    left_mse_vla = np.asarray([0.9, 0.5, 0.2, 0.9, 0.4, 0.1, 99.0])
    right_mse_vla = np.asarray([0.2, 0.5, 0.9, 0.8, 0.4, 0.1, 99.0])
    zeros = np.zeros(len(cache_indices), dtype=np.float64)
    return EvaluationResult(
        target="gt",
        flow_loss=0.0,
        mse=0.0,
        rmse=0.0,
        mae=0.0,
        flow_loss_gt=0.0,
        mse_gt=0.0,
        rmse_gt=0.0,
        mae_gt=0.0,
        flow_loss_pred=0.0,
        mse_pred=0.0,
        rmse_pred=0.0,
        mae_pred=0.0,
        mse_vla_gt=0.0,
        gt_gain=0.0,
        relative_gt_error=0.0,
        cache_indices=cache_indices,
        sample_flow_loss=zeros,
        sample_mse=zeros,
        sample_rmse=zeros,
        sample_mae=zeros,
        sample_mse_gt=zeros,
        sample_mae_gt=zeros,
        sample_mse_pred=zeros,
        sample_mae_pred=zeros,
        sample_mse_vla_gt=zeros,
        sample_gt_gain=zeros,
        sample_relative_gt_error=zeros,
        predictions=prediction,
        sample_gate_w_left=left_gate,
        sample_gate_w_right=right_gate,
        sample_tactile_change_left=np.asarray([0.9, 0.8, 0.7, 0.1, 0.2, 0.3, 0.5]),
        sample_tactile_change_right=np.asarray([0.1, 0.2, 0.3, 0.9, 0.8, 0.7, 0.5]),
        sample_mse_vla_left=left_mse_vla,
        sample_mse_vla_right=right_mse_vla,
        bimanual_gate_region_counts=np.asarray([[0, 0, 3], [0, 1, 0], [3, 0, 0]]),
        gt_actions=gt_action,
        vla_actions=vla_action,
        gate_low_threshold=0.2,
        gate_high_threshold=0.8,
    )


def test_gate_diagnostics_and_action_examples_render_retained_bimanual_actions(tmp_path: Path) -> None:
    history = tmp_path / "history.csv"
    _write_bimanual_history(history)
    result = _bimanual_result_with_mixed_quadrants()
    figures = {}
    real_subplots = bimanual_visualize.plt.subplots

    def capture_subplots(*args, **kwargs):
        figure, axes = real_subplots(*args, **kwargs)
        figures[len(figures)] = figure
        return figure, axes

    class Pairs:
        manifest = {"action_horizon": 3, "action_dim": 20}

    with mock.patch.object(bimanual_visualize.plt, "subplots", side_effect=capture_subplots):
        gate_plot = bimanual_visualize.plot_gate_diagnostics(
            history,
            result=result,
            output_path=tmp_path / "gate_diagnostics.png",
        )
        action_plot = bimanual_visualize.plot_bimanual_action_examples(
            result,
            Pairs(),  # type: ignore[arg-type]
            output_path=tmp_path / "bimanual_action_examples.png",
        )

    assert gate_plot.is_file() and gate_plot.stat().st_size > 0
    assert action_plot.is_file() and action_plot.stat().st_size > 0
    gate_figure, action_figure = figures.values()
    heatmap = next(image for axis in gate_figure.axes for image in axis.images)
    assert heatmap.get_array().shape == (3, 3)
    count_axis = gate_figure.axes[2]
    assert [patch.get_height() for patch in count_axis.patches] == [3, 1, 3, 3, 1, 3]
    action_heatmaps = [
        image
        for axis in action_figure.axes
        for image in axis.images
    ]
    assert all(image.get_array().shape == (3, 20) for image in action_heatmaps)
    labels = "\n".join(
        label.get_text()
        for axis in action_figure.axes
        if axis.get_legend() is not None
        for label in axis.get_legend().get_texts()
    )
    assert "gripper 9" in labels
    assert "gripper 19" in labels
    titles = "\n".join(axis.get_title(loc="left") for axis in action_figure.axes)
    assert "high/low median cache=11" in titles
    assert "high/low worst cache=12" in titles
    assert "low/high median cache=14" in titles
    assert "low/high worst cache=13" in titles
    assert "model" not in inspect.signature(bimanual_visualize.plot_gate_diagnostics).parameters
    assert "model" not in inspect.signature(bimanual_visualize.plot_bimanual_action_examples).parameters
