"""Matplotlib dashboards for Pi0.5's bimanual FRS objective.

All action diagnostics deliberately render only the 20 physical action
dimensions.  Pi0.5 retains its padded action tail so checkpoints can preserve
their native action width, but that tail is not a physical robot command.
"""

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

from train_pi05_frs.utils.bimanual_metrics import BIMANUAL_QUADRANTS, BIMANUAL_WRISTS
from train_pi05_frs.utils.bimanual_schema import LEFT_ACTION_SLICE, RIGHT_ACTION_SLICE, STEERED_ACTION_DIM
from train_pi05_frs.utils.history_plot import _finite_series
from train_pi05_frs.utils.metrics import EvaluationResult


_QUADRANT_METRICS = (
    "mse_gt", "mse_vla", "mse_vla_gt", "gt_gain", "relative_gt_error",
    "vla_preserve_ratio", "rank_satisfied_frac",
)
_REQUIRED_BIMANUAL_FIELDS = frozenset({
    "train_gate_w_left", "train_gate_w_right",
    *(f"val_quadrant_{quadrant}_n" for quadrant in BIMANUAL_QUADRANTS),
    *(f"val_quadrant_{quadrant}_{metric}_{wrist}" for quadrant in BIMANUAL_QUADRANTS
      for wrist in BIMANUAL_WRISTS for metric in _QUADRANT_METRICS),
})
_OVERVIEW_REQUIRED_FIELDS = frozenset({
    "train_loss_total", "train_loss_composite_fm", "train_loss_decode",
    "train_loss_rank", "train_loss_low_safety", "train_loss_repair",
    "val_composite_fm", "val_mse_gt", "val_mse_pred", "val_mse_vla_gt",
    "val_gt_gain", "val_relative_gt_error", "checkpoint_selection_feasible",
    *(f"val_{metric}_{wrist}" for wrist in BIMANUAL_WRISTS for metric in (
        "low_safe_frac", "rank_satisfied_high_frac", "gate_w", "gate_w_p10",
        "gate_w_p50", "gate_w_p90", "n_low_w", "n_mid_w", "n_high_w",
    )),
})
_HIGH_WRISTS_BY_QUADRANT = {
    "low_low": frozenset(), "high_low": frozenset({"left"}),
    "low_high": frozenset({"right"}), "high_high": frozenset(BIMANUAL_WRISTS),
}


def _read_bimanual_rows(
    history_path: pathlib.Path, *, require_overview_fields: bool = False
) -> list[dict[str, Any]]:
    """Read bimanual history while rejecting legacy CSVs for bimanual panels."""

    by_epoch: dict[int, dict[str, Any]] = {}
    with history_path.open(encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        required = _REQUIRED_BIMANUAL_FIELDS | (
            _OVERVIEW_REQUIRED_FIELDS if require_overview_fields else frozenset()
        )
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError("bimanual history fields are absent")
        for raw in reader:
            epoch_text = (raw.get("epoch") or "").strip()
            if not epoch_text:
                continue
            row: dict[str, Any] = {"epoch": int(epoch_text)}
            for field, value in raw.items():
                if field not in (None, "epoch"):
                    try:
                        row[field] = float((value or "").strip())
                    except ValueError:
                        row[field] = math.nan
            by_epoch[row["epoch"]] = row
    if not by_epoch:
        raise ValueError(f"No training history rows found in {history_path}.")
    return [by_epoch[epoch] for epoch in sorted(by_epoch)]


def _save_figure(fig: Any, output_path: pathlib.Path) -> pathlib.Path:
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


def _plot_series(axis: Any, rows: list[dict[str, Any]], field: str, *, label: str,
                 color: str, alpha: float = 1.0, linestyle: str = "-") -> bool:
    epochs, values = _finite_series(rows, field)
    if not epochs:
        return False
    axis.plot(epochs, values, label=label, color=color, alpha=alpha,
              linewidth=2.0 if alpha >= .9 else 1.7, marker="o", markersize=3.5,
              linestyle=linestyle)
    return True


def _finish_axis(axis: Any, *, title: str, ylabel: str) -> None:
    axis.set_title(title, loc="left", fontsize=10, pad=6)
    axis.set_ylabel(ylabel)
    axis.grid(True, alpha=.28)
    if axis.get_legend_handles_labels()[0]:
        axis.legend(loc="best", fontsize=8, framealpha=.9)


def plot_bimanual_training_overview(history_path: pathlib.Path, *, output_path: pathlib.Path,
                                    min_rank_satisfied: float = .8,
                                    min_low_safe: float = .9) -> pathlib.Path:
    rows = _read_bimanual_rows(history_path, require_overview_fields=True)
    fig, axes = plt.subplots(6, 1, figsize=(11, 20), sharex=True)
    fig.subplots_adjust(left=.10, right=.91, top=.97, bottom=.06, hspace=.45)
    for field, label, color in (
        ("train_loss_total", "train total", "#4C72B0"),
        ("train_loss_composite_fm", "train composite FM", "#8172B2"),
        ("train_loss_decode", "train decode", "#C44E52"),
        ("train_loss_rank", "train rank", "#DD8452"),
        ("train_loss_low_safety", "train low safety", "#55A868"),
        ("train_loss_repair", "train repair", "#937860"),
    ):
        _plot_series(axes[0], rows, field, label=label, color=color)
    _finish_axis(axes[0], title="Training objective components", ylabel="loss")
    for field, label, color in (("train_loss_composite_fm", "train composite FM", "#8172B2"),
                                ("val_composite_fm", "validation composite FM", "#C44E52")):
        _plot_series(axes[1], rows, field, label=label, color=color)
    _finish_axis(axes[1], title="Train/validation composite FM", ylabel="loss")
    for field, label, color in (("val_mse_gt", "MSE(FRS, GT)", "#C44E52"),
                                ("val_mse_pred", "MSE(FRS, VLA)", "#4C72B0"),
                                ("val_mse_vla_gt", "MSE(VLA, GT) frozen baseline", "#555555")):
        _plot_series(axes[2], rows, field, label=label, color=color)
    _finish_axis(axes[2], title="Full-20D validation decode errors", ylabel="MSE")
    for field, label, color in (("val_gt_gain", "GT gain", "#55A868"),
                                ("val_relative_gt_error", "relative GT error", "#8172B2")):
        _plot_series(axes[3], rows, field, label=label, color=color)
    axes[3].axhline(0, color="#555555", linestyle="--", linewidth=1.2, label="zero gain")
    axes[3].axhline(1, color="#999999", linestyle=":", linewidth=1.2, label="VLA baseline")
    _finish_axis(axes[3], title="Validation improvement over frozen VLA", ylabel="value")
    for wrist, color in zip(BIMANUAL_WRISTS, ("#4C72B0", "#DD8452"), strict=True):
        _plot_series(axes[4], rows, f"val_rank_satisfied_high_frac_{wrist}",
                     label=f"{wrist} high-rank satisfied", color=color)
        _plot_series(axes[4], rows, f"val_low_safe_frac_{wrist}",
                     label=f"{wrist} low safe", color=color, linestyle="--")
    _plot_series(axes[4], rows, "checkpoint_selection_feasible", label="checkpoint feasible",
                 color="#222222", linestyle=":")
    axes[4].axhline(min_rank_satisfied, color="#555555", linestyle="--", linewidth=1.2, label="minimum rank")
    axes[4].axhline(min_low_safe, color="#999999", linestyle=":", linewidth=1.2, label="minimum safe")
    axes[4].set_ylim(-.05, 1.05)
    _finish_axis(axes[4], title="Per-wrist constraints and checkpoint feasibility", ylabel="fraction / status")
    gate_axis, count_axis = axes[5], axes[5].twinx()
    for wrist, color in zip(BIMANUAL_WRISTS, ("#4C72B0", "#DD8452"), strict=True):
        for suffix, linestyle in (("", "-"), ("_p10", "--"), ("_p50", ":"), ("_p90", "-.")):
            _plot_series(gate_axis, rows, f"val_gate_w{suffix}_{wrist}",
                         label=f"{wrist} Gate {'mean' if not suffix else suffix[1:]}", color=color, linestyle=linestyle)
        for region, linestyle in (("low", "--"), ("mid", ":"), ("high", "-.")):
            _plot_series(count_axis, rows, f"val_n_{region}_w_{wrist}", label=f"{wrist} {region} samples",
                         color=color, alpha=.72, linestyle=linestyle)
    gate_axis.set_ylim(-.05, 1.05)
    _finish_axis(gate_axis, title="Validation Gate distribution and region counts", ylabel="Gate weight")
    count_axis.set_ylabel("samples"); count_axis.set_ylim(bottom=0); count_axis.grid(False)
    handles, labels = gate_axis.get_legend_handles_labels(); count_handles, count_labels = count_axis.get_legend_handles_labels()
    gate_axis.legend(handles + count_handles, labels + count_labels, loc="upper center", ncol=4, fontsize=6.5, framealpha=.9)
    gate_axis.set_xlabel("epoch")
    fig.suptitle("Bimanual FRS training overview", fontsize=15)
    return _save_figure(fig, output_path)


def _latest_value(rows: list[dict[str, Any]], field: str) -> float:
    for row in reversed(rows):
        value = float(row.get(field, math.nan))
        if math.isfinite(value):
            return value
    return math.nan


def plot_bimanual_behavior(history_path: pathlib.Path, *, output_path: pathlib.Path,
                           min_reliable_samples: int = 20) -> pathlib.Path:
    rows = _read_bimanual_rows(history_path)
    fig, axes = plt.subplots(4, 2, figsize=(14, 17), sharex=True)
    fig.subplots_adjust(left=.08, right=.98, top=.94, bottom=.06, hspace=.42, wspace=.20)
    for row_index, quadrant in enumerate(BIMANUAL_QUADRANTS):
        for column_index, wrist in enumerate(BIMANUAL_WRISTS):
            axis = axes[row_index, column_index]
            axis.axhline(1, color="#555555", linestyle=":", linewidth=1.2,
                         label="RGE=1: frozen VLA baseline; VLA preserve ratio=1: unit/baseline-scale reference")
            count = _latest_value(rows, f"val_quadrant_{quadrant}_n")
            if not math.isfinite(count) or count <= 0:
                axis.text(.5, .5, "No validation samples", transform=axis.transAxes, va="center", ha="center", color="#666666")
            else:
                high = wrist in _HIGH_WRISTS_BY_QUADRANT[quadrant]
                expected = "relative_gt_error" if high else "vla_preserve_ratio"
                reference = "vla_preserve_ratio" if high else "relative_gt_error"
                _plot_series(axis, rows, f"val_quadrant_{quadrant}_{expected}_{wrist}",
                             label="FRS vs GT (expected)" if high else "VLA preserved (expected)",
                             color="#C44E52" if high else "#4C72B0")
                _plot_series(axis, rows, f"val_quadrant_{quadrant}_{reference}_{wrist}",
                             label="VLA preserved (reference)" if high else "FRS vs GT (reference)",
                             color="#4C72B0" if high else "#C44E52", alpha=.38, linestyle="--")
                annotation = "latest n={:.0f}\nMSE(FRS,GT)={:.3g}  MSE(FRS,VLA)={:.3g}\nMSE(VLA,GT)={:.3g}  gain={:.3g}\nrank satisfied={:.1%}".format(
                    count, _latest_value(rows, f"val_quadrant_{quadrant}_mse_gt_{wrist}"),
                    _latest_value(rows, f"val_quadrant_{quadrant}_mse_vla_{wrist}"),
                    _latest_value(rows, f"val_quadrant_{quadrant}_mse_vla_gt_{wrist}"),
                    _latest_value(rows, f"val_quadrant_{quadrant}_gt_gain_{wrist}"),
                    _latest_value(rows, f"val_quadrant_{quadrant}_rank_satisfied_frac_{wrist}"))
                axis.text(.02, .97, annotation, transform=axis.transAxes, va="top", fontsize=8,
                          bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "alpha": .82, "edgecolor": "#cccccc"})
            if math.isfinite(count) and 0 < count < min_reliable_samples:
                axis.text(.98, .97, f"Insufficient samples: n={count:.0f} < {min_reliable_samples}", transform=axis.transAxes,
                          va="top", ha="right", fontsize=8, color="#B22222", fontweight="bold")
            _finish_axis(axis, title=(f"{quadrant.replace('_', '/')} — {wrist} wrist — "
                                     f"{'High Gate: approach GT' if wrist in _HIGH_WRISTS_BY_QUADRANT[quadrant] else 'Low Gate: preserve VLA'}"),
                         ylabel="normalized error")
            if row_index == 3: axis.set_xlabel("epoch")
    fig.suptitle("Bimanual FRS behavior by Gate quadrant", fontsize=15)
    return _save_figure(fig, output_path)


def _latest_gate_diagnostic_row(result: EvaluationResult) -> dict[str, float | int]:
    row: dict[str, float | int] = {"epoch": 0}
    for wrist in BIMANUAL_WRISTS:
        for prefix in ("gate_w", "tactile_change"):
            for quantile in ("p10", "p50", "p90"):
                value = getattr(result, f"{prefix}_{quantile}_{wrist}")
                row[f"val_{prefix}_{quantile}_{wrist}"] = math.nan if value is None else float(value)
        for region in ("low", "mid", "high"):
            value = getattr(result, f"n_{region}_w_{wrist}")
            row[f"val_n_{region}_w_{wrist}"] = math.nan if value is None else int(value)
    return row


def _plot_percentile_history(axis: Any, rows: list[dict[str, Any]], *, field_prefix: str,
                             title: str, ylabel: str) -> None:
    for wrist, color in zip(BIMANUAL_WRISTS, ("#4C72B0", "#DD8452"), strict=True):
        values = [(int(row["epoch"]), *(float(row.get(f"{field_prefix}_{quantile}_{wrist}", math.nan)) for quantile in ("p10", "p50", "p90"))) for row in rows]
        values = [item for item in values if all(math.isfinite(value) for value in item[1:])]
        if values:
            axis.fill_between([v[0] for v in values], [v[1] for v in values], [v[3] for v in values], color=color, alpha=.18)
            axis.plot([v[0] for v in values], [v[2] for v in values], marker="o", linewidth=2, color=color, label=f"{wrist} median")
    _finish_axis(axis, title=title, ylabel=ylabel)


def plot_bimanual_gate_diagnostics(history_path: pathlib.Path, *, result: EvaluationResult,
                                   output_path: pathlib.Path) -> pathlib.Path:
    """Render Gate histories and the latest 3×3 joint Gate-region map."""
    rows = _read_bimanual_rows(history_path) if history_path.exists() else [_latest_gate_diagnostic_row(result)]
    if result.sample_gate_w_left is None or result.sample_gate_w_right is None or result.bimanual_gate_region_counts is None:
        raise ValueError("bimanual Gate diagnostics require per-wrist retained Gate values")
    counts = np.asarray(result.bimanual_gate_region_counts, dtype=np.int64)
    if counts.shape != (3, 3):
        raise ValueError(f"bimanual Gate region counts must have shape (3, 3), got {counts.shape}")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.subplots_adjust(left=.08, right=.96, top=.91, bottom=.09, hspace=.38, wspace=.28)
    _plot_percentile_history(axes[0, 0], rows, field_prefix="val_gate_w", title="Gate median and p10-p90 history", ylabel="Gate weight")
    axes[0, 0].set_ylim(-.05, 1.05)
    _plot_percentile_history(axes[0, 1], rows, field_prefix="val_tactile_change", title="Tactile-change median and p10-p90 history", ylabel="change")
    for wrist, color in zip(BIMANUAL_WRISTS, ("#4C72B0", "#DD8452"), strict=True):
        for region, linestyle in (("low", "-"), ("mid", "--"), ("high", ":")):
            _plot_series(axes[1, 0], rows, f"val_n_{region}_w_{wrist}", label=f"{wrist} {region}", color=color, linestyle=linestyle)
    _finish_axis(axes[1, 0], title="Per-wrist Gate region count history", ylabel="samples")
    axes[1, 0].set_xlabel("epoch")
    image = axes[1, 1].imshow(counts, cmap="Blues"); total = int(counts.sum())
    for row, column in np.ndindex(counts.shape):
        percentage = 0 if total == 0 else 100 * counts[row, column] / total
        axes[1, 1].text(column, row, f"{counts[row, column]}\n{percentage:.1f}%", ha="center", va="center")
    axes[1, 1].set_xticks(np.arange(3), ("right low", "right mid", "right high"))
    axes[1, 1].set_yticks(np.arange(3), ("left low", "left mid", "left high"))
    axes[1, 1].set_title("Latest Gate-region samples", loc="left", fontsize=10, pad=6)
    fig.colorbar(image, ax=axes[1, 1], fraction=.046, pad=.04, label="samples")
    fig.suptitle("Bimanual FRS Gate diagnostics", fontsize=15)
    return _save_figure(fig, output_path)


# Source-compatible alias retained for existing scripts.
plot_gate_diagnostics = plot_bimanual_gate_diagnostics


def _mixed_quadrant_examples(result: EvaluationResult, quadrant: str) -> tuple[tuple[str, int] | None, tuple[str, int] | None]:
    left, right = result.sample_gate_w_left, result.sample_gate_w_right
    if left is None or right is None: return None, None
    if quadrant == "high_low":
        mask, preservation = (left >= result.gate_high_threshold) & (right <= result.gate_low_threshold), result.sample_mse_vla_right
    elif quadrant == "low_high":
        mask, preservation = (left <= result.gate_low_threshold) & (right >= result.gate_high_threshold), result.sample_mse_vla_left
    else:
        raise ValueError(f"unsupported mixed quadrant {quadrant!r}")
    if preservation is None: raise ValueError("bimanual action examples require per-wrist VLA errors")
    positions = np.flatnonzero(mask)
    if not len(positions): return None, None
    values = np.asarray(preservation, dtype=float)[positions]
    return (("median", int(positions[np.argmin(np.abs(values - np.median(values)))])),
            ("worst", int(positions[np.argmax(values)])))


def _action_metrics(result: EvaluationResult) -> dict[str, np.ndarray]:
    metrics: dict[str, np.ndarray] = {}
    for name in ("sample_mse_gt_left", "sample_mse_vla_left", "sample_mse_vla_gt_left",
                 "sample_mse_gt_right", "sample_mse_vla_right", "sample_mse_vla_gt_right"):
        value = getattr(result, name)
        if value is None: raise ValueError(f"bimanual action examples require {name}")
        metrics[name] = np.asarray(value, dtype=float)
    return metrics


def _action_example_identity(pairs: Any | None, cache_index: int) -> str:
    """Resolve stable source/local/episode metadata when cache metadata is available."""
    if pairs is None:
        return f"global_cache={cache_index}"
    if hasattr(pairs, "source_and_local_indices") and hasattr(pairs, "metadata_values"):
        source_indices, local_indices = pairs.source_and_local_indices([cache_index])
        episodes = pairs.metadata_values([cache_index], "episode_index")
        source_names = getattr(pairs, "source_names")
        source = int(source_indices[0])
        return (
            f"source={source_names[source]} global_cache={cache_index} "
            f"local_cache={int(local_indices[0])} episode={int(episodes[0])}"
        )
    arrays = getattr(pairs, "arrays", None)
    if isinstance(arrays, dict) and "episode_index" in arrays:
        episodes = np.asarray(arrays["episode_index"])
        if 0 <= cache_index < len(episodes):
            return (
                f"source=single global_cache={cache_index} "
                f"local_cache={cache_index} episode={int(episodes[cache_index])}"
            )
    raise ValueError("episode metadata is unavailable")


def plot_bimanual_action_examples(result: EvaluationResult, pairs: Any | None = None, *,
                                  output_path: pathlib.Path) -> pathlib.Path:
    """Plot median/worst mixed-Gate retained examples over physical dimensions 0:20."""
    if result.predictions is None or result.gt_actions is None or result.vla_actions is None:
        raise ValueError("bimanual action examples require retained predictions, GT actions, and VLA actions")
    prediction, gt_action, vla_action = (np.asarray(value) for value in (result.predictions, result.gt_actions, result.vla_actions))
    if prediction.ndim != 3 or prediction.shape[-1] < STEERED_ACTION_DIM:
        raise ValueError("retained bimanual actions must have shape (samples, horizon, >=20)")
    if gt_action.shape != prediction.shape or vla_action.shape != prediction.shape:
        raise ValueError("retained bimanual actions must share the prediction shape")
    # Do not display the retained padded tail (e.g. action width 32).
    prediction, gt_action, vla_action = (value[..., :STEERED_ACTION_DIM] for value in (prediction, gt_action, vla_action))
    metrics = _action_metrics(result)
    selected = [(quadrant, choice) for quadrant in ("high_low", "low_high")
                for choice in _mixed_quadrant_examples(result, quadrant)]
    fig, axes = plt.subplots(4, 4, figsize=(18, 16), squeeze=False)
    fig.subplots_adjust(left=.06, right=.98, top=.93, bottom=.06, hspace=.52, wspace=.30)
    steps = np.arange(prediction.shape[1])
    for row, (quadrant, choice) in enumerate(selected):
        row_axes = axes[row]
        if choice is None:
            for axis in row_axes:
                axis.set_axis_off(); axis.text(.5, .5, f"No {quadrant.replace('_', '/')} examples", ha="center", va="center", transform=axis.transAxes)
            continue
        selection, position = choice; cache_index = int(result.cache_indices[position])
        try:
            identity = _action_example_identity(pairs, cache_index)
        except (AttributeError, IndexError, KeyError, TypeError, ValueError) as exc:
            warnings.warn(
                f"cannot safely map action example cache index {cache_index}; skipping row: {exc}",
                RuntimeWarning,
                stacklevel=2,
            )
            for axis in row_axes:
                axis.set_axis_off()
                axis.text(.5, .5, "Metadata unavailable; example skipped", ha="center", va="center", transform=axis.transAxes)
            continue
        metadata = (f"{identity} w_left={float(result.sample_gate_w_left[position]):.3f} w_right={float(result.sample_gate_w_right[position]):.3f}\n"
                    f"left MSE(FRS,GT)={metrics['sample_mse_gt_left'][position]:.3g} left MSE(FRS,VLA)={metrics['sample_mse_vla_left'][position]:.3g} left MSE(VLA,GT)={metrics['sample_mse_vla_gt_left'][position]:.3g}\n"
                    f"right MSE(FRS,GT)={metrics['sample_mse_gt_right'][position]:.3g} right MSE(FRS,VLA)={metrics['sample_mse_vla_right'][position]:.3g} right MSE(VLA,GT)={metrics['sample_mse_vla_gt_right'][position]:.3g}")
        for axis, action_slice, wrist in zip(row_axes[:2], (LEFT_ACTION_SLICE, RIGHT_ACTION_SLICE), BIMANUAL_WRISTS, strict=True):
            for left_value, right_value, label, linestyle in ((prediction, gt_action, "FRS−GT", "-"), (vla_action, gt_action, "VLA−GT", "--"), (prediction, vla_action, "FRS−VLA", ":")):
                axis.plot(steps, np.linalg.norm(left_value[position, :, action_slice] - right_value[position, :, action_slice], axis=1), marker="o", label=label, linestyle=linestyle)
            _finish_axis(
                axis,
                title=(
                    f"{quadrant.replace('_', '/')} {selection} cache={cache_index} "
                    f"— {wrist}\n{metadata}"
                ),
                ylabel="per-step distance",
            )
            axis.set_xlabel("horizon step")
        image = row_axes[2].imshow(prediction[position] - vla_action[position], aspect="auto", cmap="coolwarm")
        row_axes[2].set_title(f"FRS−VLA ({quadrant.replace('_', '/')} {selection})", loc="left", fontsize=10, pad=6)
        row_axes[2].set_xlabel("physical action dimension (0:20)"); row_axes[2].set_ylabel("horizon step")
        for action_index, label, color in ((9, "left gripper 9", "#4C72B0"), (19, "right gripper 19", "#DD8452")):
            row_axes[2].axvline(action_index, color=color, linestyle="--", linewidth=1.5)
            row_axes[2].text(action_index, 1.02, label, transform=row_axes[2].get_xaxis_transform(), ha="center", va="bottom", fontsize=8, color=color)
        fig.colorbar(image, ax=row_axes[2], fraction=.046, pad=.04)
        for index, color in ((9, "#4C72B0"), (19, "#DD8452")):
            for values, label, linestyle in ((prediction, "FRS", "-"), (gt_action, "GT", "--"), (vla_action, "VLA", ":")):
                row_axes[3].plot(steps, values[position, :, index], color=color, label=f"{label} gripper {index}", linestyle=linestyle)
        _finish_axis(row_axes[3], title=f"Grippers — {quadrant.replace('_', '/')} {selection}", ylabel="action")
        row_axes[3].set_xlabel("horizon step")
    fig.suptitle("Bimanual retained action examples", fontsize=15)
    return _save_figure(fig, output_path)


def plot_bimanual_diagnostics(
    history_path: pathlib.Path,
    result: EvaluationResult,
    *,
    output_dir: pathlib.Path,
    min_rank_satisfied: float = 0.8,
    min_low_safe: float = 0.9,
) -> tuple[pathlib.Path, ...]:
    """Attempt every stable-name dashboard, preserving valid sibling plots."""

    output_dir = pathlib.Path(output_dir)
    plotters = (
        (
            "training overview",
            lambda: plot_bimanual_training_overview(
                history_path,
                output_path=output_dir / "training_overview.png",
                min_rank_satisfied=min_rank_satisfied,
                min_low_safe=min_low_safe,
            ),
        ),
        (
            "behavior",
            lambda: plot_bimanual_behavior(
                history_path,
                output_path=output_dir / "bimanual_behavior.png",
            ),
        ),
        (
            "Gate diagnostics",
            lambda: plot_bimanual_gate_diagnostics(
                history_path,
                result=result,
                output_path=output_dir / "gate_diagnostics.png",
            ),
        ),
        (
            "action examples",
            lambda: plot_bimanual_action_examples(
                result,
                output_path=output_dir / "bimanual_action_examples.png",
            ),
        ),
    )
    paths: list[pathlib.Path] = []
    for label, plot in plotters:
        try:
            paths.append(plot())
        except Exception as exc:
            warnings.warn(
                f"could not render bimanual {label}: {exc}",
                RuntimeWarning,
                stacklevel=2,
            )
    return tuple(paths)
