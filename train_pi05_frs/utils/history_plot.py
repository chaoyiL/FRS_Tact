"""Plot tactile flow steering training curves from history.csv."""

from __future__ import annotations

import csv
import math
import pathlib
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from train_pi05_frs.utils.bimanual_metrics import BIMANUAL_QUADRANTS, BIMANUAL_WRISTS


_BIMANUAL_QUADRANT_HISTORY_METRICS = (
    "mse_gt",
    "mse_vla",
    "mse_vla_gt",
    "gt_gain",
    "relative_gt_error",
    "vla_preserve_ratio",
    "rank_satisfied_frac",
)
_BIMANUAL_DISTRIBUTION_HISTORY_METRICS = (
    "gate_w",
    "gate_w_p10",
    "gate_w_p25",
    "gate_w_p50",
    "gate_w_p75",
    "gate_w_p90",
    "tactile_change",
    "tactile_change_p10",
    "tactile_change_p25",
    "tactile_change_p50",
    "tactile_change_p75",
    "tactile_change_p90",
    "n_low_w",
    "n_mid_w",
    "n_high_w",
    "low_safe_frac",
    "rank_satisfied_high_frac",
)

HISTORY_FIELDS = (
    "epoch",
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
    "val_n_high_w",
    "val_n_low_w",
) + (
    "train_loss_total",
    "train_loss_composite_fm",
    "train_loss_low_safety",
    "train_loss_decode",
    "train_loss_rank",
    "train_loss_repair",
    "train_gate_w_left",
    "train_gate_w_right",
    "val_composite_fm",
    "val_mse_vla_gt",
    "val_gt_gain",
    "val_relative_gt_error",
    "checkpoint_selection_feasible",
    "train_loss_gate_classification",
    "train_loss_residual",
    "train_loss_execute",
    "train_loss_preserve",
    "val_gate_classification_accuracy",
    "val_gate_safe_accuracy",
    "val_gate_repair_accuracy",
    "val_execute_arm9_mse",
    "val_preserve_arm9_mse",
    "val_residual_tail_p95",
    "val_residual_tail_max",
) + tuple(
    f"val_{metric}_{wrist}"
    for wrist in BIMANUAL_WRISTS
    for metric in _BIMANUAL_DISTRIBUTION_HISTORY_METRICS
) + tuple(
    f"val_quadrant_{quadrant}_n" for quadrant in BIMANUAL_QUADRANTS
) + tuple(
    f"val_quadrant_{quadrant}_{metric}_{wrist}"
    for quadrant in BIMANUAL_QUADRANTS
    for wrist in BIMANUAL_WRISTS
    for metric in _BIMANUAL_QUADRANT_HISTORY_METRICS
)


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
                    field: _parse_float(raw.get(field))
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
    train_epochs, train_flow_loss = _finite_series(rows, "train_flow_loss")
    val_loss_epochs, val_flow_loss = _finite_series(rows, "val_flow_loss")
    high_gt_epochs, mse_gt_high = _finite_series(rows, "val_mse_gt_high_w")
    low_gt_epochs, mse_gt_low = _finite_series(rows, "val_mse_gt_low_w")
    high_pred_epochs, mse_pred_high = _finite_series(rows, "val_mse_pred_high_w")
    low_pred_epochs, mse_pred_low = _finite_series(rows, "val_mse_pred_low_w")
    # Fallback for older histories without stratified columns.
    mse_gt_epochs, val_mse_gt = _finite_series(rows, "val_mse_gt")
    mse_pred_epochs, val_mse_pred = _finite_series(rows, "val_mse_pred")
    n_high_epochs, n_high_w = _finite_series(rows, "val_n_high_w")
    n_low_epochs, n_low_w = _finite_series(rows, "val_n_low_w")
    v3_repair_epochs, v3_repair_accuracy = _finite_series(
        rows, "val_gate_repair_accuracy"
    )
    v3_safe_epochs, v3_safe_accuracy = _finite_series(
        rows, "val_gate_safe_accuracy"
    )
    v3_execute_epochs, v3_execute_mse = _finite_series(
        rows, "val_execute_arm9_mse"
    )
    v3_preserve_epochs, v3_preserve_mse = _finite_series(
        rows, "val_preserve_arm9_mse"
    )
    v3_tail_epochs, v3_tail_p95 = _finite_series(
        rows, "val_residual_tail_p95"
    )

    has_stratified = bool(high_gt_epochs or low_gt_epochs or high_pred_epochs or low_pred_epochs)
    has_overall_mse = bool(mse_gt_epochs or mse_pred_epochs)
    has_val_mse = has_stratified or has_overall_mse
    has_counts = bool(n_high_epochs or n_low_epochs)
    has_v3_gate = bool(v3_repair_epochs or v3_safe_epochs)
    has_v3_action = bool(
        v3_execute_epochs or v3_preserve_epochs or v3_tail_epochs
    )

    destination = output_path or history_path.with_name("training_curves.png")
    destination.parent.mkdir(parents=True, exist_ok=True)

    n_rows = (
        1
        + int(has_val_mse)
        + int(has_counts)
        + int(has_v3_gate)
        + int(has_v3_action)
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

    row = 0
    axes[row].plot(
        train_epochs,
        train_flow_loss,
        label="train_flow_loss",
        linewidth=2.0,
        color="#4C72B0",
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
    axes[row].set_ylabel("flow loss")
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
            axes[row].set_title("Validation decode MSE (gt vs predicted)", pad=8)
        axes[row].set_ylabel("action MSE")
        axes[row].grid(True, alpha=0.3)
        axes[row].legend(loc="best", fontsize=8, framealpha=0.92)
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
        axes[row].legend(loc="best", fontsize=8, framealpha=0.92)
        row += 1

    if has_v3_gate:
        if v3_repair_epochs:
            axes[row].plot(
                v3_repair_epochs,
                v3_repair_accuracy,
                label="repair Gate accuracy",
                color="#C44E52",
                marker="o",
            )
        if v3_safe_epochs:
            axes[row].plot(
                v3_safe_epochs,
                v3_safe_accuracy,
                label="safe Gate accuracy",
                color="#55A868",
                marker="s",
            )
        axes[row].set_ylim(-0.05, 1.05)
        axes[row].set_ylabel("accuracy")
        axes[row].set_title("Learned residual Gate classification", pad=8)
        axes[row].grid(True, alpha=0.3)
        axes[row].legend(loc="best", fontsize=8, framealpha=0.92)
        row += 1

    if has_v3_action:
        for epochs_, values, label, color in (
            (v3_execute_epochs, v3_execute_mse, "execute arm9 MSE", "#C44E52"),
            (v3_preserve_epochs, v3_preserve_mse, "preserve arm9 MSE", "#4C72B0"),
            (v3_tail_epochs, v3_tail_p95, "residual |tail| p95", "#8172B2"),
        ):
            if epochs_:
                axes[row].plot(
                    epochs_, values, label=label, color=color, marker="o"
                )
        axes[row].set_ylabel("value")
        axes[row].set_title("Learned residual execution and tail", pad=8)
        axes[row].grid(True, alpha=0.3)
        axes[row].legend(loc="best", fontsize=8, framealpha=0.92)
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
