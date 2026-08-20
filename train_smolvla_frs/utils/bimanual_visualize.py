"""History-only Matplotlib dashboards for the bimanual FRS objective."""

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
        "checkpoint_selection_feasible",
        *(
            f"val_{metric}_{wrist}"
            for wrist in BIMANUAL_WRISTS
            for metric in (
                "low_safe_frac",
                "rank_satisfied_high_frac",
                "gate_w",
                "gate_w_p10",
                "gate_w_p50",
                "gate_w_p90",
                "n_low_w",
                "n_mid_w",
                "n_high_w",
            )
        ),
    }
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
    fig.subplots_adjust(left=0.10, right=0.91, top=0.97, bottom=0.06, hspace=0.45)

    loss_axis = axes[0]
    for field, label, color in (
        ("train_loss_total", "train total", "#4C72B0"),
        ("train_loss_composite_fm", "train composite FM", "#8172B2"),
        ("train_loss_decode", "train decode", "#C44E52"),
        ("train_loss_rank", "train rank", "#DD8452"),
        ("train_loss_low_safety", "train low safety", "#55A868"),
        ("train_loss_repair", "train repair", "#937860"),
    ):
        _plot_series(loss_axis, rows, field, label=label, color=color)
    _finish_axis(loss_axis, title="Training objective components", ylabel="loss")

    composite_axis = axes[1]
    _plot_series(
        composite_axis,
        rows,
        "train_loss_composite_fm",
        label="train composite FM",
        color="#8172B2",
    )
    _plot_series(
        composite_axis,
        rows,
        "val_composite_fm",
        label="validation composite FM",
        color="#C44E52",
    )
    _finish_axis(composite_axis, title="Train/validation composite FM", ylabel="loss")

    mse_axis = axes[2]
    for field, label, color in (
        ("val_mse_gt", "MSE(FRS, GT)", "#C44E52"),
        ("val_mse_pred", "MSE(FRS, VLA)", "#4C72B0"),
        ("val_mse_vla_gt", "MSE(VLA, GT) frozen baseline", "#555555"),
    ):
        _plot_series(mse_axis, rows, field, label=label, color=color)
    _finish_axis(mse_axis, title="Full-20D validation decode errors", ylabel="MSE")

    gain_axis = axes[3]
    _plot_series(gain_axis, rows, "val_gt_gain", label="GT gain", color="#55A868")
    _plot_series(
        gain_axis,
        rows,
        "val_relative_gt_error",
        label="relative GT error",
        color="#8172B2",
    )
    gain_axis.axhline(0.0, color="#555555", linestyle="--", linewidth=1.2, label="zero gain")
    gain_axis.axhline(1.0, color="#999999", linestyle=":", linewidth=1.2, label="VLA baseline")
    _finish_axis(gain_axis, title="Validation improvement over frozen VLA", ylabel="value")

    feasibility_axis = axes[4]
    for wrist, color in zip(BIMANUAL_WRISTS, ("#4C72B0", "#DD8452"), strict=True):
        _plot_series(
            feasibility_axis,
            rows,
            f"val_rank_satisfied_high_frac_{wrist}",
            label=f"{wrist} high-rank satisfied",
            color=color,
        )
        _plot_series(
            feasibility_axis,
            rows,
            f"val_low_safe_frac_{wrist}",
            label=f"{wrist} low safe",
            color=color,
            linestyle="--",
        )
    _plot_series(
        feasibility_axis,
        rows,
        "checkpoint_selection_feasible",
        label="checkpoint feasible",
        color="#222222",
        linestyle=":",
    )
    feasibility_axis.axhline(
        min_rank_satisfied,
        color="#555555",
        linestyle="--",
        linewidth=1.2,
        label="minimum rank",
    )
    feasibility_axis.axhline(
        min_low_safe,
        color="#999999",
        linestyle=":",
        linewidth=1.2,
        label="minimum safe",
    )
    feasibility_axis.set_ylim(-0.05, 1.05)
    _finish_axis(
        feasibility_axis,
        title="Per-wrist constraints and checkpoint feasibility",
        ylabel="fraction / status",
    )

    gate_axis = axes[5]
    count_axis = gate_axis.twinx()
    for wrist, color in zip(BIMANUAL_WRISTS, ("#4C72B0", "#DD8452"), strict=True):
        for statistic, linestyle in (
            ("", "-"),
            ("_p10", "--"),
            ("_p50", ":"),
            ("_p90", "-."),
        ):
            label_statistic = "mean" if not statistic else statistic.removeprefix("_")
            _plot_series(
                gate_axis,
                rows,
                f"val_gate_w{statistic}_{wrist}",
                label=f"{wrist} Gate {label_statistic}",
                color=color,
                linestyle=linestyle,
            )
        for region, linestyle in (("low", "--"), ("mid", ":"), ("high", "-.")):
            _plot_series(
                count_axis,
                rows,
                f"val_n_{region}_w_{wrist}",
                label=f"{wrist} {region} samples",
                color=color,
                alpha=0.72,
                linestyle=linestyle,
            )
    gate_axis.set_ylim(-0.05, 1.05)
    _finish_axis(
        gate_axis,
        title="Validation Gate distribution and region counts",
        ylabel="Gate weight",
    )
    count_axis.set_ylabel("samples")
    count_axis.set_ylim(bottom=0.0)
    count_axis.grid(False)
    gate_handles, gate_labels = gate_axis.get_legend_handles_labels()
    count_handles, count_labels = count_axis.get_legend_handles_labels()
    gate_axis.legend(
        gate_handles + count_handles,
        gate_labels + count_labels,
        loc="upper center",
        ncol=4,
        fontsize=6.5,
        framealpha=0.9,
    )
    gate_axis.set_xlabel("epoch")

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
            axis.axhline(
                1.0,
                color="#555555",
                linestyle=":",
                linewidth=1.2,
                label=(
                    "RGE=1: frozen VLA baseline; VLA preserve ratio=1: "
                    "unit/baseline-scale reference"
                ),
            )
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
                title=(
                    f"{quadrant.replace('_', '/')} — {wrist} wrist — "
                    f"{'High Gate: approach GT' if wrist in _HIGH_WRISTS_BY_QUADRANT[quadrant] else 'Low Gate: preserve VLA'}"
                ),
                ylabel="normalized error",
            )
            if row_index == len(BIMANUAL_QUADRANTS) - 1:
                axis.set_xlabel("epoch")

    fig.suptitle("Bimanual FRS behavior by Gate quadrant", fontsize=15)
    return _save_figure(fig, output_path)


def _latest_gate_diagnostic_row(result: EvaluationResult) -> dict[str, float | int]:
    """Build a one-point fallback when standalone evaluation has no history CSV."""

    row: dict[str, float | int] = {"epoch": 0}
    for wrist in BIMANUAL_WRISTS:
        for prefix in ("gate_w", "tactile_change"):
            for quantile in ("p10", "p50", "p90"):
                value = getattr(result, f"{prefix}_{quantile}_{wrist}")
                row[f"val_{prefix}_{quantile}_{wrist}"] = (
                    math.nan if value is None else float(value)
                )
        for region in ("low", "mid", "high"):
            value = getattr(result, f"n_{region}_w_{wrist}")
            row[f"val_n_{region}_w_{wrist}"] = (
                math.nan if value is None else int(value)
            )
    return row


def _plot_percentile_history(
    axis: Any,
    rows: list[dict[str, Any]],
    *,
    field_prefix: str,
    title: str,
    ylabel: str,
) -> None:
    """Plot per-wrist median lines and p10-p90 bands at validation epochs."""

    for wrist, color in zip(BIMANUAL_WRISTS, ("#4C72B0", "#DD8452"), strict=True):
        band_rows = []
        for row in rows:
            p10 = float(row.get(f"{field_prefix}_p10_{wrist}", math.nan))
            p50 = float(row.get(f"{field_prefix}_p50_{wrist}", math.nan))
            p90 = float(row.get(f"{field_prefix}_p90_{wrist}", math.nan))
            if all(math.isfinite(value) for value in (p10, p50, p90)):
                band_rows.append((int(row["epoch"]), p10, p50, p90))
        if not band_rows:
            continue
        epochs = [item[0] for item in band_rows]
        axis.fill_between(
            epochs,
            [item[1] for item in band_rows],
            [item[3] for item in band_rows],
            color=color,
            alpha=0.18,
        )
        axis.plot(
            epochs,
            [item[2] for item in band_rows],
            marker="o",
            linewidth=2.0,
            color=color,
            label=f"{wrist} median",
        )
    _finish_axis(axis, title=title, ylabel=ylabel)


def plot_gate_diagnostics(
    history_path: pathlib.Path,
    *,
    result: EvaluationResult,
    output_path: pathlib.Path,
) -> pathlib.Path:
    """Render validation-history Gate diagnostics and the latest 3×3 joint map."""

    if history_path.exists():
        rows = _read_bimanual_rows(history_path)
    else:
        rows = [_latest_gate_diagnostic_row(result)]
    left_gate = result.sample_gate_w_left
    right_gate = result.sample_gate_w_right
    if left_gate is None or right_gate is None or result.bimanual_gate_region_counts is None:
        raise ValueError("bimanual Gate diagnostics require per-wrist retained Gate values")
    counts = np.asarray(result.bimanual_gate_region_counts, dtype=np.int64)
    if counts.shape != (3, 3):
        raise ValueError(f"bimanual Gate region counts must have shape (3, 3), got {counts.shape}")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.subplots_adjust(left=0.08, right=0.96, top=0.91, bottom=0.09, hspace=0.38, wspace=0.28)

    gate_axis = axes[0, 0]
    _plot_percentile_history(
        gate_axis,
        rows,
        field_prefix="val_gate_w",
        title="Gate median and p10-p90 history",
        ylabel="Gate weight",
    )
    gate_axis.set_ylim(-0.05, 1.05)

    tactile_axis = axes[0, 1]
    _plot_percentile_history(
        tactile_axis,
        rows,
        field_prefix="val_tactile_change",
        title="Tactile-change median and p10-p90 history",
        ylabel="change",
    )

    count_axis = axes[1, 0]
    for wrist, color in zip(BIMANUAL_WRISTS, ("#4C72B0", "#DD8452"), strict=True):
        for region, linestyle in (("low", "-"), ("mid", "--"), ("high", ":")):
            _plot_series(
                count_axis,
                rows,
                f"val_n_{region}_w_{wrist}",
                label=f"{wrist} {region}",
                color=color,
                linestyle=linestyle,
            )
    _finish_axis(count_axis, title="Per-wrist Gate region count history", ylabel="samples")
    count_axis.set_xlabel("epoch")

    heatmap_axis = axes[1, 1]
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
        mask = (left_gate >= result.gate_high_threshold) & (
            right_gate <= result.gate_low_threshold
        )
        preservation = result.sample_mse_vla_right
    elif quadrant == "low_high":
        mask = (left_gate <= result.gate_low_threshold) & (
            right_gate >= result.gate_high_threshold
        )
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


def _action_example_identity(
    pairs: CachedPairs | MultiCachedPairs,
    cache_index: int,
) -> str:
    """Resolve stable source/local/episode metadata for an action row."""

    if isinstance(pairs, MultiCachedPairs):
        source_indices, local_indices = pairs.source_and_local_indices([cache_index])
        episode_indices = pairs.metadata_values([cache_index], "episode_index")
        source_index = int(source_indices[0])
        source_name = pairs.source_names[source_index]
        local_index = int(local_indices[0])
        episode_index = int(episode_indices[0])
    else:
        arrays = getattr(pairs, "arrays", None)
        if not isinstance(arrays, dict) or "episode_index" not in arrays:
            raise ValueError("single-cache episode_index metadata is unavailable")
        episode_values = np.asarray(arrays["episode_index"])
        if cache_index < 0 or cache_index >= len(episode_values):
            raise IndexError(f"cache index {cache_index} is outside episode metadata")
        source_name = "single"
        local_index = cache_index
        episode_index = int(episode_values[cache_index])
    return (
        f"source={source_name} global_cache={cache_index} "
        f"local_cache={local_index} episode={episode_index}"
    )


def _required_action_example_metrics(
    result: EvaluationResult,
) -> dict[str, np.ndarray]:
    names = (
        "sample_mse_gt_left",
        "sample_mse_vla_left",
        "sample_mse_vla_gt_left",
        "sample_mse_gt_right",
        "sample_mse_vla_right",
        "sample_mse_vla_gt_right",
    )
    metrics: dict[str, np.ndarray] = {}
    for name in names:
        value = getattr(result, name)
        if value is None:
            raise ValueError(f"bimanual action examples require {name}")
        metrics[name] = np.asarray(value, dtype=np.float64)
    return metrics


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
    metrics = _required_action_example_metrics(result)

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
        try:
            identity = _action_example_identity(pairs, cache_index)
        except (AttributeError, IndexError, KeyError, TypeError, ValueError) as exc:
            warnings.warn(
                f"cannot safely map action example cache index {cache_index}; "
                f"skipping row: {exc}",
                RuntimeWarning,
                stacklevel=2,
            )
            for axis in row_axes:
                axis.set_axis_off()
                axis.text(
                    0.5,
                    0.5,
                    "Metadata unavailable; example skipped",
                    ha="center",
                    va="center",
                    transform=axis.transAxes,
                )
            continue
        gate_metadata = (
            f"w_left={float(result.sample_gate_w_left[position]):.3f} "
            f"w_right={float(result.sample_gate_w_right[position]):.3f}"
        )
        mse_metadata = (
            f"left MSE(FRS,GT)={metrics['sample_mse_gt_left'][position]:.3g} "
            f"left MSE(FRS,VLA)={metrics['sample_mse_vla_left'][position]:.3g} "
            f"left MSE(VLA,GT)={metrics['sample_mse_vla_gt_left'][position]:.3g}\n"
            f"right MSE(FRS,GT)={metrics['sample_mse_gt_right'][position]:.3g} "
            f"right MSE(FRS,VLA)={metrics['sample_mse_vla_right'][position]:.3g} "
            f"right MSE(VLA,GT)={metrics['sample_mse_vla_gt_right'][position]:.3g}"
        )
        row_metadata = f"{identity} {gate_metadata}\n{mse_metadata}"
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
            axis.plot(
                steps,
                np.linalg.norm(
                    prediction[position, :, action_slice]
                    - vla_action[position, :, action_slice],
                    axis=1,
                ),
                marker="o",
                linestyle=":",
                label="FRS−VLA",
            )
            _finish_axis(
                axis,
                title=(
                    f"{display_quadrant} {selection} cache={cache_index} — {wrist}\n"
                    f"{row_metadata}"
                ),
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
        for action_index, label, color in (
            (9, "left gripper 9", "#4C72B0"),
            (19, "right gripper 19", "#DD8452"),
        ):
            heatmap_axis.axvline(
                action_index,
                color=color,
                linestyle="--",
                linewidth=1.5,
            )
            heatmap_axis.text(
                action_index,
                1.02,
                label,
                transform=heatmap_axis.get_xaxis_transform(),
                ha="center",
                va="bottom",
                fontsize=8,
                color=color,
            )
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
