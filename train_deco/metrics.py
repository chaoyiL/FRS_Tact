import csv
import json
import os
from pathlib import Path


EPOCH_FIELDS = (
    "epoch",
    "global_step",
    "train_loss",
    "val_loss",
    "val_train_loss",
    "val_unseen_loss",
    "train_velocity_mae",
    "val_velocity_mae",
    "val_train_velocity_mae",
    "val_unseen_velocity_mae",
    "lr",
    "backbone_lr",
    "backbone_frozen",
    "val_loss_std",
    "val_train_loss_std",
    "val_unseen_loss_std",
    "val_velocity_mae_std",
    "val_train_velocity_mae_std",
    "val_unseen_velocity_mae_std",
    "val_element_count",
    "val_train_element_count",
    "val_unseen_element_count",
    "validation_noise_seeds",
    "elapsed_seconds",
)


class MetricsLogger:
    """Durable JSONL step metrics and CSV epoch summaries on the output PVC."""

    def __init__(self, output_dir: Path):
        output_dir.mkdir(parents=True, exist_ok=True)
        self.jsonl_path = output_dir / "metrics.jsonl"
        self.csv_path = output_dir / "metrics.csv"
        self.jsonl_path.touch(exist_ok=True)
        if not self.csv_path.exists() or self.csv_path.stat().st_size == 0:
            with self.csv_path.open("w", encoding="utf-8", newline="") as handle:
                csv.DictWriter(handle, fieldnames=EPOCH_FIELDS).writeheader()
                self._sync(handle)

    @staticmethod
    def _sync(handle) -> None:
        handle.flush()
        os.fsync(handle.fileno())

    def log(self, record: dict) -> None:
        with self.jsonl_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
            self._sync(handle)

    def log_epoch(self, record: dict) -> None:
        self.log(record)
        val_unseen = record.get("val_unseen", record["val"])
        val_train = record.get("val_train")
        row = {
            "epoch": record["epoch"],
            "global_step": record["global_step"],
            "train_loss": record["train"]["loss"],
            "val_loss": val_unseen["loss"],
            "val_train_loss": val_train["loss"] if val_train else "",
            "val_unseen_loss": val_unseen["loss"],
            "train_velocity_mae": record["train"]["velocity_mae"],
            "val_velocity_mae": val_unseen["velocity_mae"],
            "val_train_velocity_mae": (
                val_train["velocity_mae"] if val_train else ""
            ),
            "val_unseen_velocity_mae": val_unseen["velocity_mae"],
            "lr": record["lr"],
            "backbone_lr": record.get("backbone_lr", ""),
            "backbone_frozen": record.get("backbone_frozen", ""),
            "val_loss_std": val_unseen.get("loss_std", ""),
            "val_train_loss_std": val_train.get("loss_std", "") if val_train else "",
            "val_unseen_loss_std": val_unseen.get("loss_std", ""),
            "val_velocity_mae_std": val_unseen.get("velocity_mae_std", ""),
            "val_train_velocity_mae_std": (
                val_train.get("velocity_mae_std", "") if val_train else ""
            ),
            "val_unseen_velocity_mae_std": val_unseen.get("velocity_mae_std", ""),
            "val_element_count": val_unseen.get("element_count", ""),
            "val_train_element_count": (
                val_train.get("element_count", "") if val_train else ""
            ),
            "val_unseen_element_count": val_unseen.get("element_count", ""),
            "validation_noise_seeds": val_unseen.get("noise_seeds", ""),
            "elapsed_seconds": record["elapsed_seconds"],
        }
        with self.csv_path.open("a", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=EPOCH_FIELDS)
            writer.writerow(row)
            self._sync(handle)
