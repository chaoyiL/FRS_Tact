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
    "train_flow_loss",
    "val_flow_loss",
    "val_mse",
    "val_rmse",
    "val_mae",
    "train_tactile_sim",
    "train_tactile_change",
    "train_gate_w",
    "train_gate_active_frac",
    "val_tactile_sim",
    "val_tactile_change",
    "val_gate_w",
    "val_gate_active_frac",
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
    """Plot flow loss, val decode error, and tactile similarity / gate curves."""

    rows = _read_history_rows(history_path)
    train_epochs, train_flow_loss = _finite_series(rows, "train_flow_loss")
    val_loss_epochs, val_flow_loss = _finite_series(rows, "val_flow_loss")
    val_mse_epochs, val_mse = _finite_series(rows, "val_mse")
    _, val_rmse = _finite_series(rows, "val_rmse")
    _, val_mae = _finite_series(rows, "val_mae")
    sim_epochs, train_tactile_sim = _finite_series(rows, "train_tactile_sim")
    _, train_tactile_change = _finite_series(rows, "train_tactile_change")
    _, train_gate_w = _finite_series(rows, "train_gate_w")
    _, train_gate_active = _finite_series(rows, "train_gate_active_frac")
    val_sim_epochs, val_tactile_sim = _finite_series(rows, "val_tactile_sim")
    _, val_gate_w = _finite_series(rows, "val_gate_w")
    has_val = bool(val_loss_epochs or val_mse_epochs)
    has_tactile = bool(sim_epochs or val_sim_epochs)

    destination = output_path or history_path.with_name("training_curves.png")
    destination.parent.mkdir(parents=True, exist_ok=True)

    n_rows = 1 + int(has_val) + int(has_tactile)
    fig, axes = plt.subplots(
        n_rows,
        1,
        figsize=(10, 4 + 3 * n_rows),
        constrained_layout=True,
        sharex=True,
    )
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
    axes[row].set_title("Flow matching loss")
    axes[row].grid(True, alpha=0.3)
    axes[row].legend(loc="upper right")
    row += 1

    if has_val:
        if val_mse_epochs:
            axes[row].plot(
                val_mse_epochs,
                val_mse,
                label="val_mse",
                linewidth=2.0,
                color="#C44E52",
                marker="o",
                markersize=5,
            )
            axes[row].plot(
                val_mse_epochs,
                val_rmse,
                label="val_rmse",
                linewidth=2.0,
                color="#8172B2",
                marker="s",
                markersize=5,
            )
            axes[row].plot(
                val_mse_epochs,
                val_mae,
                label="val_mae",
                linewidth=2.0,
                color="#DD8452",
                marker="^",
                markersize=5,
            )
            axes[row].legend(loc="upper right")
        axes[row].set_ylabel("action error")
        axes[row].set_title("Validation decode error vs GT")
        axes[row].grid(True, alpha=0.3)
        row += 1

    if has_tactile:
        ax = axes[row]
        if sim_epochs:
            ax.plot(
                sim_epochs,
                train_tactile_sim,
                label="train_tactile_sim (mean cos)",
                linewidth=2.0,
                color="#4C72B0",
            )
            ax.plot(
                sim_epochs,
                train_tactile_change,
                label="train_tactile_change s=1-cos",
                linewidth=2.0,
                color="#C44E52",
            )
            ax.plot(
                sim_epochs,
                train_gate_w,
                label="train_gate_w",
                linewidth=2.0,
                color="#55A868",
            )
            ax.plot(
                sim_epochs,
                train_gate_active,
                label="train_gate_active_frac (w>0.5)",
                linewidth=1.8,
                color="#8172B2",
                linestyle="--",
            )
        if val_sim_epochs:
            ax.plot(
                val_sim_epochs,
                val_tactile_sim,
                label="val_tactile_sim",
                linewidth=2.0,
                color="#64B5CD",
                marker="o",
                markersize=5,
            )
            ax.plot(
                val_sim_epochs,
                val_gate_w,
                label="val_gate_w",
                linewidth=2.0,
                color="#DD8452",
                marker="s",
                markersize=5,
            )
        ax.set_ylabel("sim / change / gate")
        ax.set_title("Tactile similarity vs episode baseline (gated)")
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=8)
        row += 1

    axes[-1].set_xlabel("epoch")
    fig.suptitle(f"Training history: {history_path.parent.name}/{history_path.name}", fontsize=12)
    fig.savefig(destination, dpi=150)
    plt.close(fig)
    return destination
