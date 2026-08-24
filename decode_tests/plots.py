"""Training curves and GT-vs-decoded sample plots for decode_tests."""

from __future__ import annotations

import csv
import math
import pathlib
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HISTORY_FIELDS = (
    "epoch",
    "train_flow_loss",
    "val_flow_loss",
    "val_mse_target",
    "val_rmse_target",
    "val_mae_target",
    "val_mse_gt",
    "val_rmse_gt",
    "val_mae_gt",
)

_DIM_COLORS = ("#4C72B0", "#55A868", "#C44E52")


def _parse_float(value: str | None) -> float:
    if value is None:
        return math.nan
    text = value.strip()
    if not text or text.lower() == "nan":
        return math.nan
    return float(text)


def read_history_rows(history_path: pathlib.Path) -> list[dict[str, Any]]:
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


def _atomic_savefig(fig, destination: pathlib.Path) -> pathlib.Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp.png")
    fig.savefig(temporary, dpi=150)
    plt.close(fig)
    temporary.replace(destination)
    return destination


def plot_training_curves(
    history_path: pathlib.Path,
    *,
    output_dir: pathlib.Path | None = None,
) -> tuple[pathlib.Path, pathlib.Path]:
    """Rewrite loss and val-target-MSE curve PNGs from history.csv."""
    rows = read_history_rows(history_path)
    destination_dir = output_dir or history_path.parent
    train_epochs, train_loss = _finite_series(rows, "train_flow_loss")
    val_epochs, val_loss = _finite_series(rows, "val_flow_loss")
    mse_epochs, val_mse_target = _finite_series(rows, "val_mse_target")

    loss_fig, loss_ax = plt.subplots(figsize=(10, 4.4), constrained_layout=True)
    if train_epochs:
        loss_ax.plot(
            train_epochs,
            train_loss,
            label="train_flow_loss",
            linewidth=2.0,
            color="#4C72B0",
        )
    if val_epochs:
        loss_ax.plot(
            val_epochs,
            val_loss,
            label="val_flow_loss",
            linewidth=2.0,
            color="#55A868",
            marker="o",
            markersize=5,
        )
    loss_ax.set_xlabel("epoch")
    loss_ax.set_ylabel("flow loss")
    loss_ax.set_title("Flow matching loss (train / val)")
    loss_ax.grid(True, alpha=0.3)
    if train_epochs or val_epochs:
        loss_ax.legend(loc="upper right", fontsize=9, framealpha=0.92)
    loss_path = _atomic_savefig(loss_fig, destination_dir / "training_curves_loss.png")

    mse_fig, mse_ax = plt.subplots(figsize=(10, 4.4), constrained_layout=True)
    if mse_epochs:
        mse_ax.plot(
            mse_epochs,
            val_mse_target,
            label="val_mse (decoded vs VLA target)",
            linewidth=2.0,
            color="#C44E52",
            marker="o",
            markersize=5,
        )
    mse_ax.set_xlabel("epoch")
    mse_ax.set_ylabel("action MSE")
    mse_ax.set_title("Validation decode MSE vs VLA predicted actions")
    mse_ax.grid(True, alpha=0.3)
    if mse_epochs:
        mse_ax.legend(loc="upper right", fontsize=9, framealpha=0.92)
    mse_path = _atomic_savefig(mse_fig, destination_dir / "training_curves_mse.png")
    return loss_path, mse_path


def select_high_mid_low_positions(sample_mse: np.ndarray) -> list[tuple[str, int]]:
    """Return unique (label, position) for high / median / low MSE."""
    values = np.asarray(sample_mse)
    if values.size == 0:
        return []
    order = np.argsort(values)
    high = int(order[-1])
    mid = int(order[len(order) // 2])
    low = int(order[0])
    selected: list[tuple[str, int]] = []
    seen: set[int] = set()
    for label, position in (("high", high), ("mid", mid), ("low", low)):
        if position in seen:
            continue
        seen.add(position)
        selected.append((label, position))
    return selected


def plot_gt_vs_pred_samples(
    path: pathlib.Path,
    *,
    cache_indices: np.ndarray,
    sample_mse_gt: np.ndarray,
    gt_actions: np.ndarray,
    predictions: np.ndarray,
    episode_indices: np.ndarray,
    dataset_indices: np.ndarray,
) -> pathlib.Path:
    """Plot GT vs decoded actions for high / mid / low vs-GT MSE samples."""
    if gt_actions.shape != predictions.shape:
        raise ValueError(
            f"gt/pred shape mismatch: {gt_actions.shape} vs {predictions.shape}"
        )
    if gt_actions.ndim != 3:
        raise ValueError(f"Expected actions [N, T, A], got {gt_actions.shape}")

    picks = select_high_mid_low_positions(sample_mse_gt)
    if not picks:
        raise ValueError("No samples available to plot GT vs predicted actions.")

    action_horizon = gt_actions.shape[1]
    action_dim = gt_actions.shape[2]
    dims_to_plot = min(3, action_dim)
    timesteps = np.arange(action_horizon)

    fig, axes = plt.subplots(
        len(picks),
        1,
        figsize=(10, 3.2 * len(picks)),
        constrained_layout=True,
        squeeze=False,
    )
    for row, (label, position) in enumerate(picks):
        axis = axes[row, 0]
        cache_index = int(cache_indices[position])
        for dim in range(dims_to_plot):
            color = _DIM_COLORS[dim % len(_DIM_COLORS)]
            axis.plot(
                timesteps,
                gt_actions[position, :, dim],
                linestyle="-",
                linewidth=1.8,
                color=color,
                label=f"GT dim {dim}",
            )
            axis.plot(
                timesteps,
                predictions[position, :, dim],
                linestyle="--",
                linewidth=1.8,
                color=color,
                label=f"decoded dim {dim}",
            )
        axis.set_xlabel("action horizon step")
        axis.set_ylabel("normalized action")
        axis.set_title(
            f"{label} MSE  cache={cache_index} "
            f"episode={int(episode_indices[cache_index])} "
            f"dataset={int(dataset_indices[cache_index])} "
            f"mse_gt={float(sample_mse_gt[position]):.4f}"
        )
        axis.legend(loc="upper right", fontsize=8, ncol=2)
        axis.grid(True, alpha=0.25)

    fig.suptitle(
        f"Ground truth vs decoded actions (high / mid / low vs-GT MSE, first {dims_to_plot} dims)",
        fontsize=13,
    )
    return _atomic_savefig(fig, path)
