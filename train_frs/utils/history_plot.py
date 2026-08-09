"""Plot tactile flow steering training curves from history.csv."""

from __future__ import annotations

import csv
import math
import pathlib
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HISTORY_FIELDS = (
    "epoch",
    "train_loss_total",
    "train_loss_gt_fm",
    "train_loss_vla_fm",
    "train_loss_decode",
    "train_loss_rank",
    "train_loss_repair",
    "train_flow_loss",
    "val_flow_loss",
    "val_mse",
    "val_rmse",
    "val_mae",
    "val_flow_loss_gt",
    "val_mse_gt",
    "val_rmse_gt",
    "val_mae_gt",
    "val_flow_loss_pred",
    "val_mse_pred",
    "val_rmse_pred",
    "val_mae_pred",
    "val_mse_gt_high_w",
    "val_mse_gt_low_w",
    "val_mse_pred_high_w",
    "val_mse_pred_low_w",
    "val_mse_vla_gt",
    "val_gt_gain",
    "val_relative_gt_error",
    "val_mse_vla_gt_high_w",
    "val_mse_vla_gt_low_w",
    "val_gt_gain_high_w",
    "val_gt_gain_low_w",
    "val_relative_gt_error_high_w",
    "val_relative_gt_error_low_w",
    "val_rank_penalty_high_w",
    "val_rank_penalty_low_w",
    "val_rank_satisfied_high_frac",
    "val_rank_satisfied_low_frac",
    "val_repair_penalty_high_w",
    "val_repair_satisfied_high_frac",
    "val_gate_w",
    "val_gate_active_frac",
    "val_gate_w_high_mean",
    "val_gate_w_low_mean",
    "val_gate_w_p10",
    "val_gate_w_p25",
    "val_gate_w_p50",
    "val_gate_w_p75",
    "val_gate_w_p90",
    "val_tactile_change",
    "val_tactile_change_high_mean",
    "val_tactile_change_low_mean",
    "val_tactile_change_p10",
    "val_tactile_change_p25",
    "val_tactile_change_p50",
    "val_tactile_change_p75",
    "val_tactile_change_p90",
    "val_n_high_w",
    "val_n_low_w",
    "val_worst_dataset_mse_pred_low_w",
    "val_min_dataset_gt_gain_high_w",
    "checkpoint_selection_key",
    "checkpoint_selection_feasible",
)

# Composite checkpoint keys are serialized as comma-separated text, for example
# ``0,0.08,-0.01,0.12``.  They are useful metadata but are not numeric plot series.
HISTORY_TEXT_FIELDS = frozenset({"checkpoint_selection_key"})


def _parse_float(value: str | None) -> float:
    if value is None:
        return math.nan
    text = value.strip()
    if not text or text.lower() == "nan":
        return math.nan
    return float(text)


def _read_history_rows(history_path: pathlib.Path) -> list[dict[str, Any]]:
    if not history_path.exists():
        raise FileNotFoundError(f"Training history not found: {history_path}")

    by_epoch: dict[int, dict[str, Any]] = {}
    with history_path.open(encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        if reader.fieldnames is None:
            raise ValueError(f"No header row found in {history_path}.")
        for raw in reader:
            epoch_text = (raw.get("epoch") or "").strip()
            if not epoch_text:
                continue
            epoch = int(epoch_text)
            by_epoch[epoch] = {
                "epoch": epoch,
                **{
                    field: (
                        (raw.get(field) or "").strip()
                        if field in HISTORY_TEXT_FIELDS
                        else _parse_float(raw.get(field))
                    )
                    for field in HISTORY_FIELDS
                    if field != "epoch"
                },
            }
    if not by_epoch:
        raise ValueError(f"No training history rows found in {history_path}.")
    return [by_epoch[epoch] for epoch in sorted(by_epoch)]


def _finite_series(
    rows: list[dict[str, Any]],
    field: str,
) -> tuple[list[int], list[float]]:
    epochs: list[int] = []
    values: list[float] = []
    for row in rows:
        value = float(row.get(field, math.nan))
        if math.isnan(value):
            continue
        epochs.append(int(row["epoch"]))
        values.append(value)
    return epochs, values


def plot_training_history(
    history_path: pathlib.Path,
    *,
    output_path: pathlib.Path | None = None,
) -> pathlib.Path:
    """Plot flow loss, gate-stratified val MSE, and high/low-w sample counts."""

    rows = _read_history_rows(history_path)
    train_epochs, train_total_loss = _finite_series(rows, "train_loss_total")
    if not train_epochs:
        train_epochs, train_total_loss = _finite_series(rows, "train_flow_loss")
    train_component_series = {
        name: _finite_series(rows, f"train_loss_{name}")
        for name in ("gt_fm", "vla_fm", "decode", "rank", "repair")
    }
    val_loss_epochs, val_flow_loss = _finite_series(rows, "val_flow_loss")
    high_gt_epochs, mse_gt_high = _finite_series(rows, "val_mse_gt_high_w")
    low_gt_epochs, mse_gt_low = _finite_series(rows, "val_mse_gt_low_w")
    high_pred_epochs, mse_pred_high = _finite_series(rows, "val_mse_pred_high_w")
    low_pred_epochs, mse_pred_low = _finite_series(rows, "val_mse_pred_low_w")
    high_vla_epochs, mse_vla_high = _finite_series(rows, "val_mse_vla_gt_high_w")
    low_vla_epochs, mse_vla_low = _finite_series(rows, "val_mse_vla_gt_low_w")
    high_gain_epochs, gt_gain_high = _finite_series(rows, "val_gt_gain_high_w")
    low_gain_epochs, gt_gain_low = _finite_series(rows, "val_gt_gain_low_w")
    high_relative_epochs, relative_high = _finite_series(
        rows, "val_relative_gt_error_high_w"
    )
    low_relative_epochs, relative_low = _finite_series(
        rows, "val_relative_gt_error_low_w"
    )
    rank_high_epochs, rank_satisfied_high = _finite_series(
        rows, "val_rank_satisfied_high_frac"
    )
    rank_low_epochs, rank_satisfied_low = _finite_series(
        rows, "val_rank_satisfied_low_frac"
    )
    repair_high_epochs, repair_satisfied_high = _finite_series(
        rows, "val_repair_satisfied_high_frac"
    )
    gate_p10_epochs, gate_p10 = _finite_series(rows, "val_gate_w_p10")
    gate_p50_epochs, gate_p50 = _finite_series(rows, "val_gate_w_p50")
    gate_p90_epochs, gate_p90 = _finite_series(rows, "val_gate_w_p90")
    change_p10_epochs, change_p10 = _finite_series(rows, "val_tactile_change_p10")
    change_p50_epochs, change_p50 = _finite_series(rows, "val_tactile_change_p50")
    change_p90_epochs, change_p90 = _finite_series(rows, "val_tactile_change_p90")
    # Fallback for older histories without stratified columns.
    mse_gt_epochs, val_mse_gt = _finite_series(rows, "val_mse_gt")
    mse_pred_epochs, val_mse_pred = _finite_series(rows, "val_mse_pred")
    n_high_epochs, n_high_w = _finite_series(rows, "val_n_high_w")
    n_low_epochs, n_low_w = _finite_series(rows, "val_n_low_w")

    has_stratified = bool(high_gt_epochs or low_gt_epochs or high_pred_epochs or low_pred_epochs)
    has_overall_mse = bool(mse_gt_epochs or mse_pred_epochs)
    # Keep the pre-refactor five-panel dashboard stable from epoch 1 onward.
    # Validation values are still plotted only after evaluation has actually run;
    # before then their four panels remain visible as pending placeholders.
    has_val_mse = True
    has_counts = True
    has_repair = True
    has_rank_stats = False
    has_gate_stats = True

    destination = output_path or history_path.with_name("training_curves.png")
    destination.parent.mkdir(parents=True, exist_ok=True)

    n_rows = (
        1
        + int(has_val_mse)
        + int(has_repair)
        + int(has_rank_stats)
        + int(has_gate_stats)
        + int(has_counts)
    )
    fig, axes = plt.subplots(
        n_rows,
        1,
        figsize=(10, 4.2 + 3.1 * n_rows),
        sharex=True,
    )
    fig.subplots_adjust(left=0.09, right=0.97, top=0.90, bottom=0.08, hspace=0.32)
    if n_rows == 1:
        axes = [axes]

    def _mark_pending(axis: Any) -> None:
        axis.text(
            0.5,
            0.5,
            "Waiting for the first validation epoch",
            transform=axis.transAxes,
            ha="center",
            va="center",
            color="#777777",
            fontsize=9,
        )

    row = 0
    axes[row].plot(
        train_epochs,
        train_total_loss,
        label="train_loss_total",
        linewidth=2.0,
        color="#4C72B0",
    )
    component_colors = {
        "gt_fm": "#8172B2",
        "vla_fm": "#CCB974",
        "decode": "#64B5CD",
        "rank": "#C44E52",
        "repair": "#937860",
    }
    for name, (epochs_component, values_component) in train_component_series.items():
        if epochs_component:
            axes[row].plot(
                epochs_component,
                values_component,
                label=f"train_loss_{name}",
                linewidth=1.3,
                alpha=0.85,
                color=component_colors[name],
            )
    if val_loss_epochs:
        axes[row].plot(
            val_loss_epochs,
            val_flow_loss,
            label="val_flow_loss",
            linewidth=2.0,
            color="#55A868",
            marker="o",
            markersize=5,
        )
    axes[row].set_ylabel("loss")
    axes[row].set_title("Flow matching loss", pad=8)
    axes[row].grid(True, alpha=0.3)
    axes[row].legend(loc="upper right", fontsize=9, framealpha=0.92)
    row += 1

    if has_val_mse:
        if has_stratified:
            if high_gt_epochs:
                axes[row].plot(
                    high_gt_epochs,
                    mse_gt_high,
                    label="val_mse_gt (w>0.5)",
                    linewidth=2.0,
                    color="#C44E52",
                    marker="o",
                    markersize=5,
                )
            if low_gt_epochs:
                axes[row].plot(
                    low_gt_epochs,
                    mse_gt_low,
                    label="val_mse_gt (w≤0.5)",
                    linewidth=2.0,
                    color="#DD8452",
                    marker="^",
                    markersize=5,
                )
            if high_pred_epochs:
                axes[row].plot(
                    high_pred_epochs,
                    mse_pred_high,
                    label="val_mse_pred (w>0.5)",
                    linewidth=2.0,
                    color="#4C72B0",
                    marker="s",
                    markersize=5,
                )
            if low_pred_epochs:
                axes[row].plot(
                    low_pred_epochs,
                    mse_pred_low,
                    label="val_mse_pred (w≤0.5)",
                    linewidth=2.0,
                    color="#64B5CD",
                    marker="D",
                    markersize=5,
                )
            if high_vla_epochs:
                axes[row].plot(
                    high_vla_epochs,
                    mse_vla_high,
                    label="VLA→GT baseline (w>0.5)",
                    linewidth=1.8,
                    color="#8C2D2D",
                    linestyle="--",
                )
            if low_vla_epochs:
                axes[row].plot(
                    low_vla_epochs,
                    mse_vla_low,
                    label="VLA→GT baseline (w≤0.5)",
                    linewidth=1.8,
                    color="#A65F2A",
                    linestyle="--",
                )
            axes[row].set_title("Validation decode MSE by gate weight", pad=8)
        else:
            if mse_gt_epochs:
                axes[row].plot(
                    mse_gt_epochs,
                    val_mse_gt,
                    label="val_mse_gt",
                    linewidth=2.0,
                    color="#C44E52",
                    marker="o",
                    markersize=5,
                )
            if mse_pred_epochs:
                axes[row].plot(
                    mse_pred_epochs,
                    val_mse_pred,
                    label="val_mse_pred",
                    linewidth=2.0,
                    color="#4C72B0",
                    marker="s",
                    markersize=5,
                )
            axes[row].set_title(
                "Validation decode MSE (gt vs predicted)"
                if has_overall_mse
                else "Validation decode MSE by gate weight",
                pad=8,
            )
        axes[row].set_ylabel("action MSE")
        axes[row].grid(True, alpha=0.3)
        handles, labels = axes[row].get_legend_handles_labels()
        if handles:
            axes[row].legend(loc="best", fontsize=8, framealpha=0.92)
        else:
            _mark_pending(axes[row])
        row += 1

    if has_repair:
        repair_axis = axes[row]
        if high_gain_epochs:
            repair_axis.plot(
                high_gain_epochs,
                gt_gain_high,
                label="GT gain (w>0.5)",
                color="#55A868",
                marker="o",
                linewidth=2.0,
            )
        if low_gain_epochs:
            repair_axis.plot(
                low_gain_epochs,
                gt_gain_low,
                label="GT gain (w≤0.5)",
                color="#C7A24B",
                marker="^",
                linewidth=2.0,
            )
        repair_axis.axhline(0.0, color="#555555", linestyle=":", linewidth=1.2)
        repair_axis.set_ylabel("VLA MSE − FRS MSE")
        repair_axis.set_title("FRS repair gain and normalized GT error", pad=8)
        repair_axis.grid(True, alpha=0.3)
        relative_axis = repair_axis.twinx()
        if high_relative_epochs:
            relative_axis.plot(
                high_relative_epochs,
                relative_high,
                label="relative GT error (w>0.5)",
                color="#4C72B0",
                linestyle="--",
                linewidth=1.8,
            )
        if low_relative_epochs:
            relative_axis.plot(
                low_relative_epochs,
                relative_low,
                label="relative GT error (w≤0.5)",
                color="#8172B2",
                linestyle="--",
                linewidth=1.8,
            )
        relative_axis.axhline(1.0, color="#888888", linestyle="--", linewidth=1.0)
        relative_axis.set_ylabel("FRS→GT / VLA→GT")
        handles, labels = repair_axis.get_legend_handles_labels()
        rel_handles, rel_labels = relative_axis.get_legend_handles_labels()
        if handles or rel_handles:
            repair_axis.legend(
                handles + rel_handles,
                labels + rel_labels,
                loc="best",
                fontsize=8,
            )
        else:
            _mark_pending(repair_axis)
        row += 1

    if has_rank_stats:
        if rank_high_epochs:
            axes[row].plot(
                rank_high_epochs,
                rank_satisfied_high,
                label="preference satisfied (w>0.5)",
                color="#55A868",
                marker="o",
                linewidth=2.0,
            )
        if rank_low_epochs:
            axes[row].plot(
                rank_low_epochs,
                rank_satisfied_low,
                label="preference satisfied (w≤0.5)",
                color="#4C72B0",
                marker="s",
                linewidth=2.0,
            )
        if repair_high_epochs:
            axes[row].plot(
                repair_high_epochs,
                repair_satisfied_high,
                label="absolute repair satisfied (w>0.5)",
                color="#C44E52",
                marker="^",
                linewidth=2.0,
            )
        axes[row].axhline(1.0, color="#888888", linestyle="--", linewidth=1.0)
        axes[row].set_ylim(-0.02, 1.02)
        axes[row].set_ylabel("satisfied fraction")
        axes[row].set_title("Gate preference and absolute-repair satisfaction", pad=8)
        axes[row].grid(True, alpha=0.3)
        axes[row].legend(loc="best", fontsize=8)
        row += 1

    if has_gate_stats:
        gate_axis = axes[row]
        for epochs, values, label, style in (
            (gate_p10_epochs, gate_p10, "gate w p10", ":"),
            (gate_p50_epochs, gate_p50, "gate w p50", "-"),
            (gate_p90_epochs, gate_p90, "gate w p90", "--"),
        ):
            if epochs:
                gate_axis.plot(epochs, values, label=label, linestyle=style, linewidth=1.8)
        gate_axis.set_ylabel("gate w")
        gate_axis.set_ylim(-0.02, 1.02)
        gate_axis.set_title("Gate and tactile-change validation quantiles", pad=8)
        gate_axis.grid(True, alpha=0.3)
        change_axis = gate_axis.twinx()
        for epochs, values, label, style in (
            (change_p10_epochs, change_p10, "change p10", ":"),
            (change_p50_epochs, change_p50, "change p50", "-"),
            (change_p90_epochs, change_p90, "change p90", "--"),
        ):
            if epochs:
                change_axis.plot(
                    epochs,
                    values,
                    label=label,
                    linestyle=style,
                    linewidth=1.6,
                    alpha=0.75,
                )
        change_axis.set_ylabel("tactile change")
        handles, labels = gate_axis.get_legend_handles_labels()
        change_handles, change_labels = change_axis.get_legend_handles_labels()
        if handles or change_handles:
            gate_axis.legend(
                handles + change_handles,
                labels + change_labels,
                loc="best",
                fontsize=8,
                ncol=2,
            )
        else:
            _mark_pending(gate_axis)
        row += 1

    if has_counts:
        if n_high_epochs:
            axes[row].plot(
                n_high_epochs,
                n_high_w,
                label="val count (w>0.5)",
                linewidth=2.0,
                color="#55A868",
                marker="o",
                markersize=5,
            )
        if n_low_epochs:
            axes[row].plot(
                n_low_epochs,
                n_low_w,
                label="val count (w≤0.5)",
                linewidth=2.0,
                color="#8172B2",
                marker="s",
                markersize=5,
            )
        axes[row].set_ylabel("# val samples")
        axes[row].set_title("Validation sample counts by gate weight", pad=8)
        axes[row].grid(True, alpha=0.3)
        handles, labels = axes[row].get_legend_handles_labels()
        if handles:
            axes[row].legend(loc="best", fontsize=8, framealpha=0.92)
        else:
            _mark_pending(axes[row])
        row += 1

    axes[-1].set_xlabel("epoch")
    fig.suptitle(
        f"Training history: {history_path.parent.name}/{history_path.name}",
        fontsize=12,
        y=0.97,
    )
    temporary = destination.with_name(destination.name + ".tmp.png")
    fig.savefig(temporary, dpi=150)
    plt.close(fig)
    temporary.replace(destination)
    return destination
