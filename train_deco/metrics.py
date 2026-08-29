import csv
import json
import os
from pathlib import Path
from typing import Any

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


def wandb_payload(record: dict[str, Any]) -> dict[str, Any]:
    """Flatten a DECO metric record into stable W&B metric names."""

    event = str(record.get("event", "metrics"))
    payload: dict[str, Any] = {"event": event}
    control_fields = {
        "epoch", "batch", "batches_in_epoch", "global_step",
        "elapsed_seconds", "lr", "backbone_lr", "backbone_frozen",
    }

    def add_value(prefix: str, value: Any) -> None:
        if isinstance(value, dict):
            for key, nested in value.items():
                add_value(f"{prefix}/{key}" if prefix else str(key), nested)
        elif isinstance(value, (bool, int, float, str)) or value is None:
            payload[prefix] = value

    for key, value in record.items():
        if key == "event":
            continue
        if event == "train_step" and key not in control_fields:
            add_value(f"train/{key}", value)
        else:
            add_value(key, value)
    return payload


class WandbMetricsLogger:
    """Optional rank-zero W&B logger; importing wandb stays opt-in."""

    def __init__(
        self,
        *,
        project: str,
        entity: str | None,
        name: str,
        run_id: str,
        group: str | None,
        tags: list[str],
        mode: str,
        output_dir: Path,
        config: dict[str, Any],
        resume: str | None = None,
    ) -> None:
        try:
            import wandb
        except ImportError as exc:
            raise RuntimeError(
                "W&B logging is enabled but wandb is not installed; "
                "rerun train_deco/setup_environment.sh"
            ) from exc

        init_options = {
            "project": project,
            "entity": entity or None,
            "name": name,
            "id": run_id,
            "group": group or None,
            "tags": tags,
            "mode": mode,
            "dir": str(output_dir),
            "config": config,
        }
        if resume is not None:
            init_options["resume"] = resume
        self._run = wandb.init(**init_options)
        self._run.define_metric("global_step")
        self._run.define_metric("*", step_metric="global_step")

    @property
    def url(self) -> str | None:
        return getattr(self._run, "url", None)

    def log(self, record: dict[str, Any]) -> None:
        self._run.log(wandb_payload(record))
