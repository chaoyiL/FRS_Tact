"""History-only Matplotlib dashboards for the bimanual FRS objective."""

from __future__ import annotations

import csv
import math
import os
import pathlib
from typing import Any

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from train_smolvla_frs.utils.bimanual_metrics import BIMANUAL_QUADRANTS, BIMANUAL_WRISTS
from train_smolvla_frs.utils.history_plot import _finite_series
from train_smolvla_frs.utils.metrics import EvaluationResult
from utils.cache import CachedPairs, MultiCachedPairs


_QUADRANT_METRICS = (
    "mse_gt",
    "mse_vla",
    "mse_vla_gt",
    "gt_gain",
    "relative_gt_error",
    "vla_preserve_ratio",
    "rank_satisfied_frac",
)
_REQUIRED_BIMANUAL_FIELDS = frozenset(
    {
        "train_gate_w_left",
        "train_gate_w_right",
        *(f"val_quadrant_{quadrant}_n" for quadrant in BIMANUAL_QUADRANTS),
        *(
            f"val_quadrant_{quadrant}_{metric}_{wrist}"
            for quadrant in BIMANUAL_QUADRANTS
            for wrist in BIMANUAL_WRISTS
            for metric in _QUADRANT_METRICS
        ),
    }
)
_OVERVIEW_REQUIRED_FIELDS = frozenset(
    f"val_{metric}_{wrist}"
    for wrist in BIMANUAL_WRISTS
    for metric in ("low_safe_frac", "rank_satisfied_high_frac")
)
_HIGH_WRISTS_BY_QUADRANT = {
    "low_low": frozenset(),
    "high_low": frozenset({"left"}),
    "low_high": frozenset({"right"}),
    "high_high": frozenset(BIMANUAL_WRISTS),
}


def _read_bimanual_rows(
    history_path: pathlib.Path,
    *,
    require_overview_fields: bool = False,
) -> list[dict[str, Any]]:
    """Return parsed history after checking the CSV was written in bimanual mode."""

    by_epoch: dict[int, dict[str, Any]] = {}
    with history_path.open(encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        required_fields = _REQUIRED_BIMANUAL_FIELDS
        if require_overview_fields:
            required_fields |= _OVERVIEW_REQUIRED_FIELDS
        if reader.fieldnames is None or not required_fields.issubset(reader.fieldnames):
            raise ValueError("bimanual history fields are absent")
        for raw in reader:
            epoch_text = (raw.get("epoch") or "").strip()
            if not epoch_text:
                continue
            epoch = int(epoch_text)
            parsed: dict[str, Any] = {"epoch": epoch}
            for field, value in raw.items():
                if field in (None, "epoch"):
                    continue
                try:
                    parsed[field] = float((value or "").strip())
                except ValueError:
                    parsed[field] = math.nan
            by_epoch[epoch] = parsed
    if not by_epoch:
        raise ValueError(f"No training history rows found in {history_path}.")
    return [by_epoch[epoch] for epoch in sorted(by_epoch)]


def _save_figure(fig: Any, output_path: pathlib.Path) -> pathlib.Path:
    """Write a complete image before atomically replacing the prior dashboard."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(f"{output_path.suffix}.tmp")
    try:
        fig.savefig(temporary, format=output_path.suffix.lstrip("."), dpi=150)
        os.replace(temporary, output_path)
    finally:
        if temporary.exists():
            temporary.unlink()
        plt.close(fig)
    return output_path


def _plot_series(
    axis: Any,
    rows: list[dict[str, Any]],
    field: str,
    *,
    label: str,
    color: str,
    alpha: float = 1.0,
    linestyle: str = "-",
) -> bool:
    epochs, values = _finite_series(rows, field)
    if not epochs:
        return False
    axis.plot(
        epochs,
        values,
        label=label,
        color=color,
        alpha=alpha,
        linewidth=2.0 if alpha >= 0.9 else 1.7,
        marker="o",
        markersize=3.5,
        linestyle=linestyle,
    )
    return True


def _finish_axis(axis: Any, *, title: str, ylabel: str) -> None:
    axis.set_title(title, loc="left", fontsize=10, pad=6)
    axis.set_ylabel(ylabel)
    axis.grid(True, alpha=0.28)
    handles, _ = axis.get_legend_handles_labels()
    if handles:
        axis.legend(loc="best", fontsize=8, framealpha=0.9)


def plot_bimanual_training_overview(
    history_path: pathlib.Path,
    *,
    output_path: pathlib.Path,
    min_rank_satisfied: float = 0.8,
    min_low_safe: float = 0.9,
) -> pathlib.Path:
    """Render a six-panel convergence dashboard for a bimanual training run."""

    rows = _read_bimanual_rows(history_path, require_overview_fields=True)
    fig, axes = plt.subplots(6, 1, figsize=(11, 20), sharex=True)
    fig.subplots_adjust(left=0.10, right=0.97, top=0.97, bottom=0.06, hspace=0.45)

    loss_axis = axes[0]
    for field, label, color in (
        ("train_loss_total", "train total", "#4C72B0"),
        ("train_loss_composite_fm", "train composite FM", "#8172B2"),
        ("val_composite_fm", "validation composite FM", "#C44E52"),
    ):
        _plot_series(loss_axis, rows, field, label=label, color=color)
    _finish_axis(loss_axis, title="Bimanual objective convergence", ylabel="loss")

    gate_axis = axes[1]
    _plot_series(gate_axis, rows, "train_gate_w_left", label="left wrist", color="#4C72B0")
    _plot_series(gate_axis, rows, "train_gate_w_right", label="right wrist", color="#DD8452")
    gate_axis.set_ylim(-0.05, 1.05)
    _finish_axis(gate_axis, title="Training Gate weights", ylabel="weight")

    low_safe_axis = axes[2]
    for wrist, color in zip(BIMANUAL_WRISTS, ("#4C72B0", "#DD8452"), strict=True):
        _plot_series(
            low_safe_axis,
            rows,
            f"val_low_safe_frac_{wrist}",
            label=f"{wrist} low safe",
            color=color,
        )
    low_safe_axis.axhline(min_low_safe, color="#555555", linestyle="--", linewidth=1.2, label="minimum safe")
    low_safe_axis.set_ylim(-0.05, 1.05)
    _finish_axis(low_safe_axis, title="Low-Gate safety by wrist", ylabel="safe fraction")

    rank_axis = axes[3]
    for wrist, color in zip(BIMANUAL_WRISTS, ("#4C72B0", "#DD8452"), strict=True):
        _plot_series(
            rank_axis,
            rows,
            f"val_rank_satisfied_high_frac_{wrist}",
            label=f"{wrist} high-rank satisfied",
            color=color,
        )
    rank_axis.axhline(min_rank_satisfied, color="#555555", linestyle="--", linewidth=1.2, label="minimum rank")
    rank_axis.set_ylim(-0.05, 1.05)
    _finish_axis(rank_axis, title="High-Gate rank feasibility by wrist", ylabel="satisfied fraction")

    count_axis = axes[4]
    colors = {"low_low": "#55A868", "high_low": "#C44E52", "low_high": "#8172B2", "high_high": "#4C72B0"}
    for quadrant in BIMANUAL_QUADRANTS:
        _plot_series(
            count_axis,
            rows,
            f"val_quadrant_{quadrant}_n",
            label=quadrant.replace("_", "/"),
            color=colors[quadrant],
        )
    _finish_axis(count_axis, title="Validation samples by Gate quadrant", ylabel="samples")

    status_axis = axes[5]
    for wrist, color in zip(BIMANUAL_WRISTS, ("#4C72B0", "#DD8452"), strict=True):
        _plot_series(
            status_axis,
            rows,
            f"val_quadrant_high_high_relative_gt_error_{wrist}",
            label=f"{wrist} high/high relative GT error",
            color=color,
        )
    status_axis.axhline(1.0, color="#555555", linestyle=":", linewidth=1.2, label="VLA-to-GT baseline")
    _finish_axis(status_axis, title="Both-high target error by wrist", ylabel="relative error")
    status_axis.set_xlabel("epoch")

    fig.suptitle("Bimanual FRS training overview", fontsize=15)
    return _save_figure(fig, output_path)


def _latest_value(rows: list[dict[str, Any]], field: str) -> float:
    for row in reversed(rows):
        value = float(row.get(field, math.nan))
        if math.isfinite(value):
            return value
    return math.nan


def plot_bimanual_behavior(
    history_path: pathlib.Path,
    *,
    output_path: pathlib.Path,
    min_reliable_samples: int = 20,
) -> pathlib.Path:
    """Render independent left/right behavior in all four Gate quadrants."""

    rows = _read_bimanual_rows(history_path)
    fig, axes = plt.subplots(4, 2, figsize=(14, 17), sharex=True)
    fig.subplots_adjust(left=0.08, right=0.98, top=0.94, bottom=0.06, hspace=0.42, wspace=0.20)

    for row_index, quadrant in enumerate(BIMANUAL_QUADRANTS):
        for column_index, wrist in enumerate(BIMANUAL_WRISTS):
            axis = axes[row_index, column_index]
            relative_field = f"val_quadrant_{quadrant}_relative_gt_error_{wrist}"
            preserve_field = f"val_quadrant_{quadrant}_vla_preserve_ratio_{wrist}"
            axis.axhline(1.0, color="#555555", linestyle=":", linewidth=1.2, label="GT baseline")
            sample_count = _latest_value(rows, f"val_quadrant_{quadrant}_n")
            if not math.isfinite(sample_count) or sample_count <= 0:
                axis.text(
                    0.5,
                    0.5,
                    "No validation samples",
                    transform=axis.transAxes,
                    va="center",
                    ha="center",
                    fontsize=10,
                    color="#666666",
                )
            else:
                high_wrist = wrist in _HIGH_WRISTS_BY_QUADRANT[quadrant]
                expected_field, expected_label, expected_color = (
                    (relative_field, "FRS vs GT (expected)", "#C44E52")
                    if high_wrist
                    else (preserve_field, "VLA preserved (expected)", "#4C72B0")
                )
                reference_field, reference_label, reference_color = (
                    (preserve_field, "VLA preserved (reference)", "#4C72B0")
                    if high_wrist
                    else (relative_field, "FRS vs GT (reference)", "#C44E52")
                )
                _plot_series(
                    axis,
                    rows,
                    expected_field,
                    label=expected_label,
                    color=expected_color,
                )
                _plot_series(
                    axis,
                    rows,
                    reference_field,
                    label=reference_label,
                    color=reference_color,
                    alpha=0.38,
                    linestyle="--",
                )
                mse_gt = _latest_value(rows, f"val_quadrant_{quadrant}_mse_gt_{wrist}")
                mse_vla = _latest_value(rows, f"val_quadrant_{quadrant}_mse_vla_{wrist}")
                mse_vla_gt = _latest_value(rows, f"val_quadrant_{quadrant}_mse_vla_gt_{wrist}")
                gain = _latest_value(rows, f"val_quadrant_{quadrant}_gt_gain_{wrist}")
                rank = _latest_value(rows, f"val_quadrant_{quadrant}_rank_satisfied_frac_{wrist}")
                annotation = (
                    f"latest n={sample_count:.0f}\n"
                    f"MSE(FRS,GT)={mse_gt:.3g}  MSE(FRS,VLA)={mse_vla:.3g}\n"
                    f"MSE(VLA,GT)={mse_vla_gt:.3g}  gain={gain:.3g}\n"
                    f"rank satisfied={rank:.1%}"
                )
                axis.text(
                    0.02,
                    0.97,
                    annotation,
                    transform=axis.transAxes,
                    va="top",
                    ha="left",
                    fontsize=8,
                    bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "alpha": 0.82, "edgecolor": "#cccccc"},
                )
            if math.isfinite(sample_count) and 0 < sample_count < min_reliable_samples:
                axis.text(
                    0.98,
                    0.97,
                    f"Insufficient samples: n={sample_count:.0f} < {min_reliable_samples}",
                    transform=axis.transAxes,
                    va="top",
                    ha="right",
                    fontsize=8,
                    color="#B22222",
                    fontweight="bold",
                )
            _finish_axis(
                axis,
                title=f"{quadrant.replace('_', '/')} — {wrist} wrist",
                ylabel="normalized error",
            )
            if row_index == len(BIMANUAL_QUADRANTS) - 1:
                axis.set_xlabel("epoch")

    fig.suptitle("Bimanual FRS behavior by Gate quadrant", fontsize=15)
    return _save_figure(fig, output_path)


def _sample_percentiles(values: np.ndarray | None) -> np.ndarray:
    if values is None or len(values) == 0:
        return np.full(5, math.nan)
    return np.quantile(np.asarray(values, dtype=np.float64), (0.1, 0.25, 0.5, 0.75, 0.9))


def plot_gate_diagnostics(
    history_path: pathlib.Path,
    *,
    result: EvaluationResult,
    output_path: pathlib.Path,
) -> pathlib.Path:
    """Render latest per-wrist Gate distributions and the 3×3 joint region map."""

    if history_path.exists():
        _read_bimanual_rows(history_path)
    left_gate = result.sample_gate_w_left
    right_gate = result.sample_gate_w_right
    if left_gate is None or right_gate is None or result.bimanual_gate_region_counts is None:
        raise ValueError("bimanual Gate diagnostics require per-wrist retained Gate values")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.subplots_adjust(left=0.08, right=0.96, top=0.91, bottom=0.09, hspace=0.38, wspace=0.28)
    percentile_labels = ("p10", "p25", "p50", "p75", "p90")
    positions = np.arange(len(percentile_labels))
    wrist_values = (("left", left_gate, "#4C72B0"), ("right", right_gate, "#DD8452"))

    gate_axis = axes[0, 0]
    for wrist, values, color in wrist_values:
        gate_axis.plot(positions, _sample_percentiles(values), marker="o", label=wrist, color=color)
    gate_axis.set_xticks(positions, percentile_labels)
    gate_axis.set_ylim(-0.05, 1.05)
    _finish_axis(gate_axis, title="Gate percentiles", ylabel="Gate weight")

    tactile_axis = axes[0, 1]
    for wrist, values, color in (
        ("left", result.sample_tactile_change_left, "#4C72B0"),
        ("right", result.sample_tactile_change_right, "#DD8452"),
    ):
        tactile_axis.plot(positions, _sample_percentiles(values), marker="o", label=wrist, color=color)
    tactile_axis.set_xticks(positions, percentile_labels)
    _finish_axis(tactile_axis, title="Tactile-change percentiles", ylabel="change")

    count_axis = axes[1, 0]
    region_labels = ("low", "mid", "high")
    offsets = (-0.18, 0.18)
    for offset, (wrist, values, color) in zip(offsets, wrist_values, strict=True):
        values = np.asarray(values, dtype=np.float64)
        counts = np.asarray(
            (
                np.sum(values <= 0.3),
                np.sum((values > 0.3) & (values < 0.7)),
                np.sum(values >= 0.7),
            )
        )
        count_axis.bar(np.arange(3) + offset, counts, width=0.36, label=wrist, color=color)
    count_axis.set_xticks(np.arange(3), region_labels)
    _finish_axis(count_axis, title="Per-wrist Gate region counts", ylabel="samples")

    heatmap_axis = axes[1, 1]
    counts = np.asarray(result.bimanual_gate_region_counts, dtype=np.int64)
    if counts.shape != (3, 3):
        raise ValueError(f"bimanual Gate region counts must have shape (3, 3), got {counts.shape}")
    image = heatmap_axis.imshow(counts, cmap="Blues")
    total = int(np.sum(counts))
    for row, column in np.ndindex(counts.shape):
        percentage = 0.0 if total == 0 else 100.0 * float(counts[row, column]) / total
        heatmap_axis.text(column, row, f"{counts[row, column]}\n{percentage:.1f}%", ha="center", va="center")
    heatmap_axis.set_xticks(np.arange(3), ("right low", "right mid", "right high"))
    heatmap_axis.set_yticks(np.arange(3), ("left low", "left mid", "left high"))
    heatmap_axis.set_title("Latest Gate-region samples", loc="left", fontsize=10, pad=6)
    fig.colorbar(image, ax=heatmap_axis, fraction=0.046, pad=0.04, label="samples")
    fig.suptitle("Bimanual FRS Gate diagnostics", fontsize=15)
    return _save_figure(fig, output_path)


def _mixed_quadrant_examples(
    result: EvaluationResult,
    quadrant: str,
) -> tuple[tuple[str, int] | None, tuple[str, int] | None]:
    """Return median and worst low-wrist VLA-preservation examples for a quadrant."""

    left_gate = result.sample_gate_w_left
    right_gate = result.sample_gate_w_right
    if left_gate is None or right_gate is None:
        return None, None
    if quadrant == "high_low":
        mask = (left_gate >= 0.7) & (right_gate <= 0.3)
        preservation = result.sample_mse_vla_right
    elif quadrant == "low_high":
        mask = (left_gate <= 0.3) & (right_gate >= 0.7)
        preservation = result.sample_mse_vla_left
    else:
        raise ValueError(f"unsupported mixed quadrant {quadrant!r}")
    if preservation is None:
        raise ValueError("bimanual action examples require per-wrist VLA errors")
    positions = np.flatnonzero(mask)
    if len(positions) == 0:
        return None, None
    values = np.asarray(preservation, dtype=np.float64)[positions]
    median_position = positions[int(np.argmin(np.abs(values - np.median(values))))]
    worst_position = positions[int(np.argmax(values))]
    return ("median", int(median_position)), ("worst", int(worst_position))


def plot_bimanual_action_examples(
    result: EvaluationResult,
    pairs: CachedPairs | MultiCachedPairs,
    *,
    output_path: pathlib.Path,
) -> pathlib.Path:
    """Plot retained FRS/VLA/GT actions for representative mixed-Gate examples."""

    if result.predictions is None or result.gt_actions is None or result.vla_actions is None:
        raise ValueError("bimanual action examples require retained predictions, GT actions, and VLA actions")
    prediction = np.asarray(result.predictions)
    gt_action = np.asarray(result.gt_actions)
    vla_action = np.asarray(result.vla_actions)
    expected_shape = (
        int(pairs.manifest["action_horizon"]),
        int(pairs.manifest["action_dim"]),
    )
    if prediction.ndim != 3 or prediction.shape[1:] != expected_shape:
        raise ValueError(
            "retained prediction shape must be "
            f"(samples, {expected_shape[0]}, {expected_shape[1]}), got {prediction.shape}"
        )
    if gt_action.shape != prediction.shape or vla_action.shape != prediction.shape:
        raise ValueError("retained bimanual actions must share the prediction shape")

    selected = [
        (quadrant, choice)
        for quadrant in ("high_low", "low_high")
        for choice in _mixed_quadrant_examples(result, quadrant)
    ]
    fig, axes = plt.subplots(4, 4, figsize=(18, 16), squeeze=False)
    fig.subplots_adjust(left=0.06, right=0.98, top=0.93, bottom=0.06, hspace=0.52, wspace=0.30)
    steps = np.arange(expected_shape[0])
    for row, (quadrant, choice) in enumerate(selected):
        row_axes = axes[row]
        if choice is None:
            for axis in row_axes:
                axis.set_axis_off()
                axis.text(
                    0.5,
                    0.5,
                    f"No {quadrant.replace('_', '/')} examples",
                    ha="center",
                    va="center",
                    transform=axis.transAxes,
                )
            continue
        selection, position = choice
        cache_index = int(result.cache_indices[position])
        display_quadrant = quadrant.replace("_", "/")
        for axis, action_slice, wrist in zip(
            row_axes[:2],
            (slice(0, 10), slice(10, 20)),
            ("left", "right"),
            strict=True,
        ):
            axis.plot(
                steps,
                np.linalg.norm(
                    prediction[position, :, action_slice] - gt_action[position, :, action_slice],
                    axis=1,
                ),
                marker="o",
                label="FRS−GT",
            )
            axis.plot(
                steps,
                np.linalg.norm(
                    vla_action[position, :, action_slice] - gt_action[position, :, action_slice],
                    axis=1,
                ),
                marker="o",
                linestyle="--",
                label="VLA−GT",
            )
            _finish_axis(
                axis,
                title=f"{display_quadrant} {selection} cache={cache_index} — {wrist}",
                ylabel="per-step distance",
            )
            axis.set_xlabel("horizon step")
        heatmap_axis = row_axes[2]
        image = heatmap_axis.imshow(
            prediction[position] - vla_action[position],
            aspect="auto",
            cmap="coolwarm",
        )
        heatmap_axis.set_title(f"FRS−VLA ({display_quadrant} {selection})", loc="left", fontsize=10, pad=6)
        heatmap_axis.set_xlabel("action dimension")
        heatmap_axis.set_ylabel("horizon step")
        fig.colorbar(image, ax=heatmap_axis, fraction=0.046, pad=0.04)
        gripper_axis = row_axes[3]
        for action_index, color in ((9, "#4C72B0"), (19, "#DD8452")):
            gripper_axis.plot(
                steps,
                prediction[position, :, action_index],
                color=color,
                label=f"FRS gripper {action_index}",
            )
            gripper_axis.plot(
                steps,
                gt_action[position, :, action_index],
                color=color,
                linestyle="--",
                label=f"GT gripper {action_index}",
            )
            gripper_axis.plot(
                steps,
                vla_action[position, :, action_index],
                color=color,
                linestyle=":",
                label=f"VLA gripper {action_index}",
            )
        _finish_axis(gripper_axis, title=f"Grippers — {display_quadrant} {selection}", ylabel="action")
        gripper_axis.set_xlabel("horizon step")
    fig.suptitle("Bimanual retained action examples", fontsize=15)
    return _save_figure(fig, output_path)
