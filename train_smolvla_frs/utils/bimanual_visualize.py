"""History-only Matplotlib dashboards for the bimanual FRS objective."""

from __future__ import annotations

import csv
import math
import os
import pathlib
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from train_smolvla_frs.utils.bimanual_metrics import BIMANUAL_QUADRANTS, BIMANUAL_WRISTS
from train_smolvla_frs.utils.history_plot import _finite_series


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
