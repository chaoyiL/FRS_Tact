from __future__ import annotations

import csv
from pathlib import Path
from unittest import mock

import pytest

import train_smolvla_frs.utils.bimanual_visualize as bimanual_visualize
from train_smolvla_frs.utils.history_plot import HISTORY_FIELDS


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
