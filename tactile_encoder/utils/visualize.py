"""Plot tactile CLIP pretraining curves from history.csv."""

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
    "train_loss",
    "train_batch_recall@1",
    "train_batch_recall@5",
    "train_batch_mean_rank",
    "train_bank_filled_frac",
    "train_bank_hard_neg_logit_mean",
    "train_batch_vs_positive_gap",
    "val_recall@1",
    "val_recall@5",
    "val_mean_rank",
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

    rows: list[dict[str, Any]] = []
    with history_path.open(encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        if reader.fieldnames is None:
            raise ValueError(f"No header row found in {history_path}.")
        for raw in reader:
            row = {
                key.strip(): (value.strip() if value is not None else "")
                for key, value in raw.items()
                if key is not None
            }
            epoch_text = row.get("epoch", "")
            if not epoch_text:
                continue
            rows.append(
                {
                    "epoch": int(epoch_text),
                    **{
                        field: _parse_float(row.get(field))
                        for field in HISTORY_FIELDS
                        if field != "epoch"
                    },
                }
            )
    if not rows:
        raise ValueError(f"No training history rows found in {history_path}.")
    rows.sort(key=lambda row: int(row["epoch"]))
    return rows


def _dedupe_epochs(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep the last row per epoch when history.csv was appended out of order."""

    by_epoch: dict[int, dict[str, Any]] = {}
    for row in rows:
        by_epoch[int(row["epoch"])] = row
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
    """Plot train loss, validation retrieval, and hard-negative train metrics."""

    rows = _dedupe_epochs(_read_history_rows(history_path))
    epochs = [int(row["epoch"]) for row in rows]
    train_loss = [float(row["train_loss"]) for row in rows]

    val_epochs, val_recall1 = _finite_series(rows, "val_recall@1")
    _, val_recall5 = _finite_series(rows, "val_recall@5")
    _, val_mean_rank = _finite_series(rows, "val_mean_rank")

    hard_epochs, hard_neg_logit = _finite_series(rows, "train_bank_hard_neg_logit_mean")
    gap_epochs, pos_gap = _finite_series(rows, "train_batch_vs_positive_gap")
    filled_epochs, bank_filled = _finite_series(rows, "train_bank_filled_frac")
    has_hard_metrics = bool(hard_epochs or gap_epochs or filled_epochs)

    destination = output_path or history_path.with_name("training_curves.png")
    destination.parent.mkdir(parents=True, exist_ok=True)

    n_rows = 4 if has_hard_metrics else 3
    fig, axes = plt.subplots(
        n_rows,
        1,
        figsize=(10, 12 if has_hard_metrics else 10),
        constrained_layout=True,
        sharex=True,
    )

    axes[0].plot(epochs, train_loss, label="train_loss", linewidth=2.0, color="#4C72B0")
    axes[0].set_ylabel("loss")
    axes[0].set_title("Train loss")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc="upper right")

    if val_epochs:
        axes[1].plot(
            val_epochs,
            val_recall1,
            label="val_recall@1",
            linewidth=2.0,
            color="#55A868",
            marker="o",
            markersize=5,
        )
        axes[1].plot(
            val_epochs,
            val_recall5,
            label="val_recall@5",
            linewidth=2.0,
            color="#C44E52",
            marker="s",
            markersize=5,
        )
        axes[1].set_ylim(bottom=0.0)
    axes[1].set_ylabel("recall")
    axes[1].set_title("Validation recall")
    axes[1].grid(True, alpha=0.3)
    if val_epochs:
        axes[1].legend(loc="upper right")

    if val_epochs:
        axes[2].plot(
            val_epochs,
            val_mean_rank,
            label="val_mean_rank",
            linewidth=2.0,
            color="#8172B2",
            marker="o",
            markersize=5,
        )
        axes[2].legend(loc="upper right")
    axes[2].set_ylabel("mean rank")
    axes[2].set_title("Validation mean rank (lower is better)")
    axes[2].grid(True, alpha=0.3)

    if has_hard_metrics:
        ax_hard = axes[3]
        if hard_epochs:
            ax_hard.plot(
                hard_epochs,
                hard_neg_logit,
                label="train_bank_hard_neg_logit_mean",
                linewidth=2.0,
                color="#DD8452",
            )
        if gap_epochs:
            ax_hard.plot(
                gap_epochs,
                pos_gap,
                label="train_batch_vs_positive_gap",
                linewidth=2.0,
                color="#64B5CD",
            )
        ax_hard.set_ylabel("logit / gap")
        ax_hard.set_title("Hard-negative diagnostics")
        ax_hard.grid(True, alpha=0.3)
        if filled_epochs:
            ax_filled = ax_hard.twinx()
            ax_filled.plot(
                filled_epochs,
                bank_filled,
                label="train_bank_filled_frac",
                linewidth=1.8,
                color="#937860",
                linestyle="--",
            )
            ax_filled.set_ylabel("bank filled frac")
            ax_filled.set_ylim(0.0, 1.05)
            lines_left, labels_left = ax_hard.get_legend_handles_labels()
            lines_right, labels_right = ax_filled.get_legend_handles_labels()
            ax_hard.legend(lines_left + lines_right, labels_left + labels_right, loc="upper right")
        else:
            ax_hard.legend(loc="upper right")
        ax_hard.set_xlabel("epoch")
    else:
        axes[2].set_xlabel("epoch")

    fig.suptitle(f"Training history: {history_path.parent.name}/{history_path.name}", fontsize=12)
    fig.savefig(destination, dpi=150)
    plt.close(fig)
    return destination
