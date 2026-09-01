"""Stable-name dashboards for the scalar-Gate single-hand FRS objective."""

from __future__ import annotations

import csv
import math
import os
import pathlib
import warnings
from typing import Any

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from train_pi05_frs.utils.metrics import EvaluationResult


_HISTORY_FIELDS = frozenset(
    {
        "train_loss_total",
        "train_loss_composite_fm",
        "train_loss_decode",
        "train_loss_rank",
        "train_loss_low_safety",
        "train_loss_repair",
        "val_composite_fm",
        "val_mse_gt",
        "val_mse_pred",
        "val_mse_vla_gt",
        "val_gt_gain",
        "val_relative_gt_error",
        "val_gate_w_mean",
        "val_gate_w_p10",
        "val_gate_w_p50",
        "val_gate_w_p90",
        "val_tactile_change_mean",
        "val_tactile_change_p10",
        "val_tactile_change_p50",
        "val_tactile_change_p90",
        "val_gate_n_low",
        "val_gate_n_mid",
        "val_gate_n_high",
        "val_low_safe_frac",
        "val_high_gate_rank_satisfied_frac",
        "val_high_gate_repair_satisfied_frac",
        "checkpoint_selection_feasible",
    }
)


def _read_rows(history_path: pathlib.Path) -> list[dict[str, float | int]]:
    by_epoch: dict[int, dict[str, float | int]] = {}
    with pathlib.Path(history_path).open(encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        if reader.fieldnames is None or not _HISTORY_FIELDS.issubset(reader.fieldnames):
            raise ValueError("single-hand composite history fields are absent")
        for raw in reader:
            epoch_text = (raw.get("epoch") or "").strip()
            if not epoch_text:
                continue
            row: dict[str, float | int] = {"epoch": int(epoch_text)}
            for field in _HISTORY_FIELDS:
                text = (raw.get(field) or "").strip()
                try:
                    row[field] = float(text) if text else math.nan
                except ValueError:
                    row[field] = math.nan
            by_epoch[int(row["epoch"])] = row
    if not by_epoch:
        raise ValueError(f"No training history rows found in {history_path}.")
    return [by_epoch[epoch] for epoch in sorted(by_epoch)]


def _save(fig: Any, path: pathlib.Path) -> pathlib.Path:
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    try:
        fig.savefig(temporary, format=path.suffix.lstrip("."), dpi=150)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
        plt.close(fig)
    return path


def _series(
    axis: Any,
    rows: list[dict[str, float | int]],
    field: str,
    label: str,
    color: str,
    *,
    linestyle: str = "-",
    alpha: float = 1.0,
) -> None:
    points = [
        (int(row["epoch"]), float(row.get(field, math.nan))) for row in rows
    ]
    points = [(epoch, value) for epoch, value in points if math.isfinite(value)]
    if points:
        axis.plot(
            [point[0] for point in points],
            [point[1] for point in points],
            label=label,
            color=color,
            linestyle=linestyle,
            alpha=alpha,
            linewidth=2,
            marker="o",
            markersize=3.5,
        )


def _finish(axis: Any, title: str, ylabel: str) -> None:
    axis.set_title(title, loc="left", fontsize=10, pad=6)
    axis.set_ylabel(ylabel)
    axis.grid(True, alpha=0.28)
    if axis.get_legend_handles_labels()[0]:
        axis.legend(loc="best", fontsize=8, framealpha=0.9)


def plot_single_hand_training_overview(
    history_path: pathlib.Path,
    *,
    output_path: pathlib.Path,
    min_rank_satisfied: float = 0.8,
    min_low_safe: float = 0.9,
) -> pathlib.Path:
    rows = _read_rows(history_path)
    fig, axes = plt.subplots(6, 1, figsize=(11, 20), sharex=True)
    fig.subplots_adjust(left=0.10, right=0.91, top=0.97, bottom=0.06, hspace=0.45)
    for field, label, color in (
        ("train_loss_total", "train total", "#4C72B0"),
        ("train_loss_composite_fm", "composite FM", "#8172B2"),
        ("train_loss_decode", "decode", "#C44E52"),
        ("train_loss_rank", "rank", "#DD8452"),
        ("train_loss_low_safety", "low safety", "#55A868"),
        ("train_loss_repair", "repair", "#937860"),
    ):
        _series(axes[0], rows, field, label, color)
    _finish(axes[0], "Training objective components", "loss")

    _series(axes[1], rows, "train_loss_composite_fm", "train", "#8172B2")
    _series(axes[1], rows, "val_composite_fm", "validation", "#C44E52")
    _finish(axes[1], "Composite endpoint flow matching", "loss")

    for field, label, color in (
        ("val_mse_gt", "MSE(FRS, GT)", "#C44E52"),
        ("val_mse_pred", "MSE(FRS, VLA)", "#4C72B0"),
        ("val_mse_vla_gt", "MSE(VLA, GT), frozen", "#555555"),
    ):
        _series(axes[2], rows, field, label, color)
    _finish(axes[2], "10D validation decode errors", "MSE")

    _series(axes[3], rows, "val_gt_gain", "GT gain", "#55A868")
    _series(
        axes[3], rows, "val_relative_gt_error", "relative GT error", "#8172B2"
    )
    axes[3].axhline(0, color="#555555", linestyle="--", linewidth=1.2)
    axes[3].axhline(1, color="#999999", linestyle=":", linewidth=1.2)
    _finish(axes[3], "Improvement over frozen VLA", "value")

    _series(axes[4], rows, "val_low_safe_frac", "low-Gate safe", "#4C72B0")
    _series(
        axes[4],
        rows,
        "val_high_gate_rank_satisfied_frac",
        "high-Gate rank satisfied",
        "#DD8452",
    )
    _series(
        axes[4],
        rows,
        "val_high_gate_repair_satisfied_frac",
        "high-Gate repair satisfied",
        "#55A868",
    )
    _series(
        axes[4],
        rows,
        "checkpoint_selection_feasible",
        "checkpoint feasible",
        "#222222",
        linestyle=":",
    )
    axes[4].axhline(min_rank_satisfied, color="#555555", linestyle="--")
    axes[4].axhline(min_low_safe, color="#999999", linestyle=":")
    axes[4].set_ylim(-0.05, 1.05)
    _finish(axes[4], "Safety, ranking, repair, and checkpoint constraints", "fraction")

    count_axis = axes[5].twinx()
    for field, label, linestyle in (
        ("val_gate_w_mean", "Gate mean", "-"),
        ("val_gate_w_p10", "Gate p10", "--"),
        ("val_gate_w_p50", "Gate p50", ":"),
        ("val_gate_w_p90", "Gate p90", "-."),
    ):
        _series(axes[5], rows, field, label, "#4C72B0", linestyle=linestyle)
    for field, label, color in (
        ("val_gate_n_low", "low samples", "#55A868"),
        ("val_gate_n_mid", "mid samples", "#DD8452"),
        ("val_gate_n_high", "high samples", "#C44E52"),
    ):
        _series(count_axis, rows, field, label, color, alpha=0.72)
    axes[5].set_ylim(-0.05, 1.05)
    _finish(axes[5], "Validation Gate distribution and region counts", "Gate weight")
    count_axis.set_ylabel("samples")
    count_axis.set_ylim(bottom=0)
    count_axis.grid(False)
    handles, labels = axes[5].get_legend_handles_labels()
    extra_handles, extra_labels = count_axis.get_legend_handles_labels()
    axes[5].legend(
        handles + extra_handles,
        labels + extra_labels,
        loc="upper center",
        ncol=4,
        fontsize=7,
    )
    axes[5].set_xlabel("epoch")
    fig.suptitle("Single-hand FRS training overview", fontsize=15)
    return _save(fig, output_path)


def _required_samples(result: EvaluationResult) -> dict[str, np.ndarray]:
    values: dict[str, np.ndarray] = {}
    for field in (
        "sample_gate_w",
        "sample_tactile_change",
        "sample_mse_gt",
        "sample_mse_pred",
        "sample_mse_vla_gt",
        "sample_gt_gain",
        "sample_relative_gt_error",
    ):
        value = getattr(result, field)
        if value is None:
            raise ValueError(f"single-hand diagnostics require {field}")
        values[field] = np.asarray(value, dtype=np.float64)
    return values


def _region_masks(result: EvaluationResult, gates: np.ndarray) -> tuple[tuple[str, np.ndarray], ...]:
    low = gates <= float(result.gate_low_threshold)
    high = gates >= float(result.gate_high_threshold)
    return ("low", low), ("mid", ~(low | high)), ("high", high)


def plot_single_hand_behavior(
    result: EvaluationResult, *, output_path: pathlib.Path
) -> pathlib.Path:
    samples = _required_samples(result)
    masks = _region_masks(result, samples["sample_gate_w"])
    fig, axes = plt.subplots(3, 2, figsize=(13, 13))
    fig.subplots_adjust(left=0.08, right=0.97, top=0.93, bottom=0.06, hspace=0.40)
    for row, (region, mask) in enumerate(masks):
        count = int(np.count_nonzero(mask))
        if count == 0:
            for axis in axes[row]:
                axis.set_axis_off()
                axis.text(0.5, 0.5, f"No {region}-Gate samples", ha="center", va="center")
            continue
        error_means = [
            float(np.mean(samples[name][mask]))
            for name in ("sample_mse_gt", "sample_mse_pred", "sample_mse_vla_gt")
        ]
        axes[row, 0].bar(
            ["FRS-GT", "FRS-VLA", "VLA-GT"],
            error_means,
            color=["#C44E52", "#4C72B0", "#777777"],
        )
        _finish(axes[row, 0], f"{region.title()} Gate (n={count}): endpoint errors", "MSE")
        relative_gt = samples["sample_relative_gt_error"][mask]
        preserve_ratio = samples["sample_mse_pred"][mask] / np.maximum(
            samples["sample_mse_vla_gt"][mask], 1e-8
        )
        axes[row, 1].boxplot(
            [relative_gt, preserve_ratio],
            tick_labels=["relative GT error", "VLA preserve ratio"],
            showfliers=False,
        )
        axes[row, 1].axhline(1, color="#777777", linestyle=":")
        axes[row, 1].text(
            0.02,
            0.97,
            f"mean GT gain={np.mean(samples['sample_gt_gain'][mask]):.3g}",
            transform=axes[row, 1].transAxes,
            va="top",
            fontsize=9,
        )
        _finish(
            axes[row, 1],
            f"{region.title()} Gate (n={count}): normalized behavior",
            "ratio",
        )
    fig.suptitle("Single-hand FRS behavior by Gate region", fontsize=15)
    return _save(fig, output_path)


def plot_single_hand_gate_diagnostics(
    history_path: pathlib.Path,
    result: EvaluationResult,
    *,
    output_path: pathlib.Path,
) -> pathlib.Path:
    rows = _read_rows(history_path)
    samples = _required_samples(result)
    gates = samples["sample_gate_w"]
    changes = samples["sample_tactile_change"]
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.subplots_adjust(left=0.08, right=0.96, top=0.91, bottom=0.09, hspace=0.38)
    for field, label, linestyle in (
        ("val_gate_w_p10", "p10", "--"),
        ("val_gate_w_p50", "median", "-"),
        ("val_gate_w_p90", "p90", "-."),
    ):
        _series(axes[0, 0], rows, field, label, "#4C72B0", linestyle=linestyle)
    axes[0, 0].axhline(result.gate_low_threshold, color="#55A868", linestyle=":")
    axes[0, 0].axhline(result.gate_high_threshold, color="#C44E52", linestyle=":")
    axes[0, 0].set_ylim(-0.05, 1.05)
    _finish(axes[0, 0], "Gate p10/median/p90 history", "Gate weight")

    for field, label, linestyle in (
        ("val_tactile_change_p10", "p10", "--"),
        ("val_tactile_change_p50", "median", "-"),
        ("val_tactile_change_p90", "p90", "-."),
    ):
        _series(axes[0, 1], rows, field, label, "#8172B2", linestyle=linestyle)
    _finish(axes[0, 1], "Tactile-change p10/median/p90 history", "change")

    masks = _region_masks(result, gates)
    axes[1, 0].bar(
        [name for name, _ in masks],
        [int(np.count_nonzero(mask)) for _, mask in masks],
        color=["#55A868", "#DD8452", "#C44E52"],
    )
    _finish(axes[1, 0], "Latest Gate-region sample counts", "samples")
    axes[1, 0].set_xlabel("Gate region")

    scatter = axes[1, 1].scatter(
        gates,
        samples["sample_gt_gain"],
        c=changes,
        cmap="viridis",
        alpha=0.72,
        edgecolors="none",
    )
    axes[1, 1].axhline(0, color="#777777", linestyle=":")
    axes[1, 1].axvline(result.gate_low_threshold, color="#55A868", linestyle="--")
    axes[1, 1].axvline(result.gate_high_threshold, color="#C44E52", linestyle="--")
    axes[1, 1].set_xlabel("Gate weight")
    _finish(axes[1, 1], "Latest Gate vs GT gain", "MSE(VLA,GT) - MSE(FRS,GT)")
    fig.colorbar(scatter, ax=axes[1, 1], fraction=0.046, pad=0.04, label="tactile change")
    fig.suptitle("Single-hand FRS Gate diagnostics", fontsize=15)
    return _save(fig, output_path)


def _example_identity(pairs: Any | None, cache_index: int) -> str:
    if pairs is None:
        return f"global_cache={cache_index}"
    if hasattr(pairs, "source_and_local_indices") and hasattr(pairs, "metadata_values"):
        source_indices, local_indices = pairs.source_and_local_indices([cache_index])
        episodes = pairs.metadata_values([cache_index], "episode_index")
        source = int(source_indices[0])
        source_names = getattr(pairs, "source_names")
        return (
            f"source={source_names[source]} cache={cache_index} "
            f"local={int(local_indices[0])} episode={int(episodes[0])}"
        )
    arrays = getattr(pairs, "arrays", None)
    if isinstance(arrays, dict) and "episode_index" in arrays:
        return f"cache={cache_index} episode={int(np.asarray(arrays['episode_index'])[cache_index])}"
    return f"global_cache={cache_index}"


def _median_and_worst(mask: np.ndarray, error: np.ndarray) -> tuple[tuple[str, int] | None, ...]:
    positions = np.flatnonzero(mask)
    if not len(positions):
        return None, None
    values = error[positions]
    median = int(positions[np.argmin(np.abs(values - np.median(values)))])
    worst = int(positions[np.argmax(values)])
    return ("median", median), ("worst", worst)


def plot_single_hand_action_examples(
    result: EvaluationResult,
    pairs: Any | None = None,
    *,
    output_path: pathlib.Path,
) -> pathlib.Path:
    samples = _required_samples(result)
    if result.predictions is None or result.gt_actions is None or result.vla_actions is None:
        raise ValueError("single-hand action examples require retained FRS, GT, and VLA actions")
    prediction, gt_action, vla_action = (
        np.asarray(value, dtype=np.float64)
        for value in (result.predictions, result.gt_actions, result.vla_actions)
    )
    if prediction.ndim != 3 or gt_action.shape != prediction.shape or vla_action.shape != prediction.shape:
        raise ValueError("retained single-hand actions must share shape [samples, horizon, action_dim]")
    masks = dict(_region_masks(result, samples["sample_gate_w"]))
    selections = [
        ("low", choice)
        for choice in _median_and_worst(masks["low"], samples["sample_mse_pred"])
    ] + [
        ("high", choice)
        for choice in _median_and_worst(masks["high"], samples["sample_mse_gt"])
    ]
    fig, axes = plt.subplots(4, 4, figsize=(18, 16), squeeze=False)
    fig.subplots_adjust(left=0.06, right=0.98, top=0.93, bottom=0.06, hspace=0.55, wspace=0.30)
    steps = np.arange(prediction.shape[1])
    gripper = min(9, prediction.shape[-1] - 1)
    for row, (region, choice) in enumerate(selections):
        if choice is None:
            for axis in axes[row]:
                axis.set_axis_off()
                axis.text(0.5, 0.5, f"No {region}-Gate example", ha="center", va="center")
            continue
        label, position = choice
        cache_index = int(result.cache_indices[position])
        identity = _example_identity(pairs, cache_index)
        header = (
            f"{region} {label}: {identity}, Gate={samples['sample_gate_w'][position]:.3f}\n"
            f"MSE FRS-GT={samples['sample_mse_gt'][position]:.3g}, "
            f"FRS-VLA={samples['sample_mse_pred'][position]:.3g}, "
            f"VLA-GT={samples['sample_mse_vla_gt'][position]:.3g}"
        )
        for left, right, curve_label, linestyle in (
            (prediction, gt_action, "FRS-GT", "-"),
            (vla_action, gt_action, "VLA-GT", "--"),
            (prediction, vla_action, "FRS-VLA", ":"),
        ):
            axes[row, 0].plot(
                steps,
                np.linalg.norm(left[position] - right[position], axis=1),
                label=curve_label,
                linestyle=linestyle,
                marker="o",
            )
        _finish(axes[row, 0], header, "per-step L2")
        axes[row, 0].set_xlabel("horizon step")

        for column, difference, title in (
            (1, prediction[position] - vla_action[position], "FRS - VLA"),
            (2, prediction[position] - gt_action[position], "FRS - GT"),
        ):
            image = axes[row, column].imshow(difference, aspect="auto", cmap="coolwarm")
            axes[row, column].axvline(gripper, color="#222222", linestyle="--")
            axes[row, column].set_title(title, loc="left", fontsize=10)
            axes[row, column].set_xlabel("action dimension")
            axes[row, column].set_ylabel("horizon step")
            fig.colorbar(image, ax=axes[row, column], fraction=0.046, pad=0.04)

        for values, curve_label, linestyle in (
            (prediction, "FRS", "-"),
            (gt_action, "GT", "--"),
            (vla_action, "VLA", ":"),
        ):
            axes[row, 3].plot(
                steps,
                values[position, :, gripper],
                label=curve_label,
                linestyle=linestyle,
            )
        _finish(axes[row, 3], f"Gripper action dim {gripper}", "action")
        axes[row, 3].set_xlabel("horizon step")
    fig.suptitle("Single-hand retained action examples", fontsize=15)
    return _save(fig, output_path)


def plot_single_hand_diagnostics(
    history_path: pathlib.Path,
    result: EvaluationResult,
    *,
    output_dir: pathlib.Path,
    pairs: Any | None = None,
    min_rank_satisfied: float = 0.8,
    min_low_safe: float = 0.9,
) -> tuple[pathlib.Path, ...]:
    """Attempt every dashboard independently so one rendering issue cannot stop training."""

    output_dir = pathlib.Path(output_dir)
    plotters = (
        (
            "training overview",
            lambda: plot_single_hand_training_overview(
                history_path,
                output_path=output_dir / "training_overview.png",
                min_rank_satisfied=min_rank_satisfied,
                min_low_safe=min_low_safe,
            ),
        ),
        (
            "behavior",
            lambda: plot_single_hand_behavior(
                result, output_path=output_dir / "single_hand_behavior.png"
            ),
        ),
        (
            "Gate diagnostics",
            lambda: plot_single_hand_gate_diagnostics(
                history_path,
                result,
                output_path=output_dir / "gate_diagnostics.png",
            ),
        ),
        (
            "action examples",
            lambda: plot_single_hand_action_examples(
                result,
                pairs,
                output_path=output_dir / "single_hand_action_examples.png",
            ),
        ),
    )
    paths: list[pathlib.Path] = []
    for label, plot in plotters:
        try:
            paths.append(plot())
        except Exception as exc:
            warnings.warn(
                f"could not render single-hand {label}: {exc}",
                RuntimeWarning,
                stacklevel=2,
            )
    return tuple(paths)
