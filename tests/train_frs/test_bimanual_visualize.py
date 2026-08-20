from __future__ import annotations

import csv
import inspect
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pytest

import train_smolvla_frs.utils.bimanual_visualize as bimanual_visualize
from train_smolvla_frs.utils.history_plot import HISTORY_FIELDS
from train_smolvla_frs.utils.metrics import EvaluationResult
from utils.cache import MultiCachedPairs


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
    count_fields = tuple(
        f"val_n_{region}_w_{wrist}"
        for wrist in ("left", "right")
        for region in ("low", "mid", "high")
    )
    fieldnames = (
        HISTORY_FIELDS
        + (overview_fields if include_overview_fields else ())
        + count_fields
    )
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
                "train_loss_decode": 0.05 * scale,
                "train_loss_rank": 0.03 * scale,
                "train_loss_repair": 0.02 * scale,
                "train_gate_w_left": 0.8,
                "train_gate_w_right": 0.2,
                "val_composite_fm": 0.25 * scale,
                "val_mse_gt": 0.18 * scale,
                "val_mse_pred": 0.08 * scale,
                "val_mse_vla_gt": 0.50,
                "val_gt_gain": 0.50 - 0.18 * scale,
                "val_relative_gt_error": 0.36 * scale,
                "checkpoint_selection_feasible": int(epoch == 2),
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
        for wrist, gate_base, tactile_base in (
            ("left", 0.8, 0.7),
            ("right", 0.2, 0.3),
        ):
            row[f"val_gate_w_{wrist}"] = gate_base * scale
            row[f"val_gate_w_p10_{wrist}"] = gate_base * scale - 0.1
            row[f"val_gate_w_p50_{wrist}"] = gate_base * scale
            row[f"val_gate_w_p90_{wrist}"] = gate_base * scale + 0.1
            row[f"val_tactile_change_p10_{wrist}"] = tactile_base * scale - 0.1
            row[f"val_tactile_change_p50_{wrist}"] = tactile_base * scale
            row[f"val_tactile_change_p90_{wrist}"] = tactile_base * scale + 0.1
            row[f"val_n_low_w_{wrist}"] = epoch + (0 if wrist == "left" else 3)
            row[f"val_n_mid_w_{wrist}"] = epoch + (1 if wrist == "left" else 4)
            row[f"val_n_high_w_{wrist}"] = epoch + (2 if wrist == "left" else 5)
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
    overview_axes = figures["overview"].axes
    assert len(overview_axes) == 7
    overview_labels = [
        {line.get_label() for line in axis.lines}
        for axis in overview_axes
    ]
    assert {
        "train total",
        "train composite FM",
        "train decode",
        "train rank",
        "train low safety",
        "train repair",
    }.issubset(overview_labels[0])
    assert {"train composite FM", "validation composite FM"}.issubset(
        overview_labels[1]
    )
    assert {
        "MSE(FRS, GT)",
        "MSE(FRS, VLA)",
        "MSE(VLA, GT) frozen baseline",
    }.issubset(overview_labels[2])
    assert {"GT gain", "relative GT error", "zero gain", "VLA baseline"}.issubset(
        overview_labels[3]
    )
    assert {
        "left high-rank satisfied",
        "right high-rank satisfied",
        "left low safe",
        "right low safe",
        "checkpoint feasible",
        "minimum rank",
        "minimum safe",
    }.issubset(overview_labels[4])
    gate_axis = overview_axes[5]
    count_axis = overview_axes[6]
    assert {
        "left Gate mean",
        "left Gate p10",
        "left Gate p50",
        "left Gate p90",
        "right Gate mean",
        "right Gate p10",
        "right Gate p50",
        "right Gate p90",
    }.issubset(overview_labels[5])
    assert {
        "left low samples",
        "left mid samples",
        "left high samples",
        "right low samples",
        "right mid samples",
        "right high samples",
    }.issubset(overview_labels[6])
    assert not overview_labels[5].intersection(overview_labels[6])
    assert gate_axis.get_ylabel() == "Gate weight"
    assert count_axis.get_ylabel() == "samples"
    assert gate_axis.get_ylim() == pytest.approx((-0.05, 1.05))
    assert gate_axis.get_shared_x_axes().joined(gate_axis, count_axis)
    assert count_axis.get_legend() is None
    assert {
        label.get_text()
        for label in gate_axis.get_legend().get_texts()
    } >= overview_labels[5] | overview_labels[6]
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
    behavior_titles = [axis.get_title(loc="left") for axis in figures["behavior"].axes]
    assert "High Gate: approach GT" in behavior_titles[2]
    assert "Low Gate: preserve VLA" in behavior_titles[3]


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


def test_behavior_titles_state_expected_behavior_for_each_wrist(tmp_path: Path) -> None:
    history = tmp_path / "history.csv"
    _write_bimanual_history(history)
    captured = {}
    real_subplots = bimanual_visualize.plt.subplots

    def capture_subplots(*args, **kwargs):
        figure, axes = real_subplots(*args, **kwargs)
        captured["figure"] = figure
        return figure, axes

    with mock.patch.object(
        bimanual_visualize.plt,
        "subplots",
        side_effect=capture_subplots,
    ):
        bimanual_visualize.plot_bimanual_behavior(
            history,
            output_path=tmp_path / "behavior.png",
        )

    titles = [axis.get_title(loc="left") for axis in captured["figure"].axes]
    assert "High Gate: approach GT" in titles[2]
    assert "Low Gate: preserve VLA" in titles[3]


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
        sample_mse_gt_left=np.asarray([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]),
        sample_mse_gt_right=np.asarray([0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]),
        sample_mse_vla_gt_left=np.asarray([1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7]),
        sample_mse_vla_gt_right=np.asarray([1.7, 1.6, 1.5, 1.4, 1.3, 1.2, 1.1]),
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
        arrays = {"episode_index": np.arange(100, 117, dtype=np.int64)}

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
    gate_axis = gate_figure.axes[0]
    tactile_axis = gate_figure.axes[1]
    for axis in (gate_axis, tactile_axis, count_axis):
        assert all(list(line.get_xdata()) == [1, 2] for line in axis.lines)
    assert {line.get_label() for line in gate_axis.lines} == {
        "left median",
        "right median",
    }
    assert {line.get_label() for line in tactile_axis.lines} == {
        "left median",
        "right median",
    }
    assert {line.get_label() for line in count_axis.lines} == {
        "left low",
        "left mid",
        "left high",
        "right low",
        "right mid",
        "right high",
    }
    assert len(gate_axis.collections) == 2
    assert len(tactile_axis.collections) == 2
    action_heatmaps = [
        image
        for axis in action_figure.axes
        for image in axis.images
    ]
    assert all(image.get_array().shape == (3, 20) for image in action_heatmaps)
    for axis in (action_figure.axes[0], action_figure.axes[1]):
        assert "FRS−VLA" in {line.get_label() for line in axis.lines}
    heatmap_axes = [axis for axis in action_figure.axes if axis.images]
    for axis in heatmap_axes:
        assert {text.get_text() for text in axis.texts} >= {
            "left gripper 9",
            "right gripper 19",
        }
        vertical_markers = {
            float(np.asarray(line.get_xdata())[0])
            for line in axis.lines
            if np.asarray(line.get_xdata()).shape == (2,)
            and np.asarray(line.get_xdata())[0] == np.asarray(line.get_xdata())[1]
        }
        assert {9.0, 19.0}.issubset(vertical_markers)
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
    assert "global_cache=11" in titles
    assert "local_cache=11" in titles
    assert "episode=111" in titles
    assert "w_left=0.850" in titles
    assert "w_right=0.150" in titles
    assert "left MSE(FRS,GT)=" in titles
    assert "left MSE(FRS,VLA)=" in titles
    assert "left MSE(VLA,GT)=" in titles
    assert "right MSE(FRS,GT)=" in titles
    assert "right MSE(FRS,VLA)=" in titles
    assert "right MSE(VLA,GT)=" in titles
    assert "model" not in inspect.signature(bimanual_visualize.plot_gate_diagnostics).parameters
    assert "model" not in inspect.signature(bimanual_visualize.plot_bimanual_action_examples).parameters


def test_action_examples_map_multicache_global_indices_to_source_local_identity(
    tmp_path: Path,
) -> None:
    result = _bimanual_result_with_mixed_quadrants()
    pairs = object.__new__(MultiCachedPairs)
    pairs.source_names = ("alpha", "beta")
    pairs.sources = (
        SimpleNamespace(arrays={"episode_index": np.arange(10, dtype=np.int64)}),
        SimpleNamespace(arrays={"episode_index": np.arange(100, 110, dtype=np.int64)}),
    )
    pairs._starts = np.asarray([0, 10], dtype=np.int64)
    pairs._stops = np.asarray([10, 20], dtype=np.int64)
    pairs.manifest = {
        "sample_count": 20,
        "action_horizon": 3,
        "action_dim": 20,
    }
    captured = {}
    real_subplots = bimanual_visualize.plt.subplots

    def capture_subplots(*args, **kwargs):
        figure, axes = real_subplots(*args, **kwargs)
        captured["figure"] = figure
        return figure, axes

    with mock.patch.object(
        bimanual_visualize.plt,
        "subplots",
        side_effect=capture_subplots,
    ):
        bimanual_visualize.plot_bimanual_action_examples(
            result,
            pairs,
            output_path=tmp_path / "multi.png",
        )

    titles = "\n".join(
        axis.get_title(loc="left") for axis in captured["figure"].axes
    )
    assert "source=beta" in titles
    assert "global_cache=11" in titles
    assert "local_cache=1" in titles
    assert "episode=101" in titles


def test_action_example_titles_include_episode_gates_and_per_wrist_mses(
    tmp_path: Path,
) -> None:
    result = _bimanual_result_with_mixed_quadrants()
    captured = {}
    real_subplots = bimanual_visualize.plt.subplots

    class Pairs:
        manifest = {"action_horizon": 3, "action_dim": 20}
        arrays = {"episode_index": np.arange(100, 117, dtype=np.int64)}

    def capture_subplots(*args, **kwargs):
        figure, axes = real_subplots(*args, **kwargs)
        captured["figure"] = figure
        return figure, axes

    with mock.patch.object(
        bimanual_visualize.plt,
        "subplots",
        side_effect=capture_subplots,
    ):
        bimanual_visualize.plot_bimanual_action_examples(
            result,
            Pairs(),  # type: ignore[arg-type]
            output_path=tmp_path / "metadata.png",
        )

    titles = "\n".join(
        axis.get_title(loc="left") for axis in captured["figure"].axes
    )
    for expected in (
        "episode=111",
        "w_left=0.850",
        "w_right=0.150",
        "left MSE(FRS,GT)=",
        "left MSE(FRS,VLA)=",
        "left MSE(VLA,GT)=",
        "right MSE(FRS,GT)=",
        "right MSE(FRS,VLA)=",
        "right MSE(VLA,GT)=",
    ):
        assert expected in titles


def test_action_wrist_panels_include_direct_frs_vla_distance(tmp_path: Path) -> None:
    result = _bimanual_result_with_mixed_quadrants()
    captured = {}
    real_subplots = bimanual_visualize.plt.subplots

    class Pairs:
        manifest = {"action_horizon": 3, "action_dim": 20}
        arrays = {"episode_index": np.arange(100, 117, dtype=np.int64)}

    def capture_subplots(*args, **kwargs):
        figure, axes = real_subplots(*args, **kwargs)
        captured["figure"] = figure
        return figure, axes

    with mock.patch.object(
        bimanual_visualize.plt,
        "subplots",
        side_effect=capture_subplots,
    ):
        bimanual_visualize.plot_bimanual_action_examples(
            result,
            Pairs(),  # type: ignore[arg-type]
            output_path=tmp_path / "distances.png",
        )

    for row in range(4):
        for column in range(2):
            axis = captured["figure"].axes[row * 4 + column]
            assert "FRS−VLA" in {line.get_label() for line in axis.lines}


def test_action_heatmaps_mark_both_gripper_dimensions(tmp_path: Path) -> None:
    result = _bimanual_result_with_mixed_quadrants()
    captured = {}
    real_subplots = bimanual_visualize.plt.subplots

    class Pairs:
        manifest = {"action_horizon": 3, "action_dim": 20}
        arrays = {"episode_index": np.arange(100, 117, dtype=np.int64)}

    def capture_subplots(*args, **kwargs):
        figure, axes = real_subplots(*args, **kwargs)
        captured["figure"] = figure
        return figure, axes

    with mock.patch.object(
        bimanual_visualize.plt,
        "subplots",
        side_effect=capture_subplots,
    ):
        bimanual_visualize.plot_bimanual_action_examples(
            result,
            Pairs(),  # type: ignore[arg-type]
            output_path=tmp_path / "heatmaps.png",
        )

    heatmap_axes = [axis for axis in captured["figure"].axes if axis.images]
    assert len(heatmap_axes) == 4
    for axis in heatmap_axes:
        assert {text.get_text() for text in axis.texts} >= {
            "left gripper 9",
            "right gripper 19",
        }
        vertical_markers = {
            float(np.asarray(line.get_xdata())[0])
            for line in axis.lines
            if np.asarray(line.get_xdata()).shape == (2,)
            and np.asarray(line.get_xdata())[0] == np.asarray(line.get_xdata())[1]
        }
        assert {9.0, 19.0}.issubset(vertical_markers)


def test_action_examples_warn_and_placeholder_when_cache_mapping_is_unavailable(
    tmp_path: Path,
) -> None:
    result = _bimanual_result_with_mixed_quadrants()
    captured = {}
    real_subplots = bimanual_visualize.plt.subplots

    class UnmappedPairs:
        manifest = {"action_horizon": 3, "action_dim": 20}

    def capture_subplots(*args, **kwargs):
        figure, axes = real_subplots(*args, **kwargs)
        captured["figure"] = figure
        return figure, axes

    with (
        mock.patch.object(
            bimanual_visualize.plt,
            "subplots",
            side_effect=capture_subplots,
        ),
        pytest.warns(RuntimeWarning, match="cannot safely map action example"),
    ):
        output = bimanual_visualize.plot_bimanual_action_examples(
            result,
            UnmappedPairs(),  # type: ignore[arg-type]
            output_path=tmp_path / "unmapped.png",
        )

    assert output.is_file()
    texts = "\n".join(
        text.get_text()
        for axis in captured["figure"].axes
        for text in axis.texts
    )
    assert "Metadata unavailable; example skipped" in texts


def test_readme_states_bimanual_plot_prerequisites() -> None:
    readme = Path("train_smolvla_frs/README.md").read_text(encoding="utf-8")

    assert "`write_plots: true`" in readme
    assert "validation event" in readme
    assert "successful plotting" in readme
