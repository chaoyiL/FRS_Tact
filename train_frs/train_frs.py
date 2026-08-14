#!/usr/bin/env python
"""Train multi-dataset tactile FRS from a YAML config."""

# ruff: noqa: E402, I001

from __future__ import annotations

import argparse
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from train_frs.train import train_decoder

DEFAULT_CONFIG = Path(__file__).resolve().parent / "configs" / "train_frs.yaml"
RUN_CONFIG_NAME = "train_config.yaml"


def source_cache_dir(cache_root: str | Path, repo_id: str) -> Path:
    parts = [part for part in str(repo_id).split("/") if part not in ("", ".", "..")]
    if not parts:
        raise ValueError(f"invalid repo id: {repo_id!r}")
    return Path(cache_root).expanduser().joinpath(*parts)


def load_config(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as file:
        value = yaml.safe_load(file) or {}
    if not isinstance(value, dict):
        raise ValueError(f"config root must be a mapping: {path}")
    return value


def _positive_int(config: Mapping[str, Any], key: str, default: int) -> int:
    value = int(config.get(key, default))
    if value <= 0:
        raise ValueError(f"{key} must be positive, got {value}")
    return value


def resolve_resume_mode(value: Any, *, output_dir: Path) -> bool:
    """Resolve false/true/auto without treating non-empty strings as true."""

    if isinstance(value, bool):
        return value
    mode = str(value if value is not None else "false").strip().lower()
    if mode == "auto":
        return (output_dir / "last" / "checkpoint.json").is_file()
    if mode in {"true", "1", "yes", "on"}:
        return True
    if mode in {"false", "0", "no", "off", ""}:
        return False
    raise ValueError("frs_training.resume must be false, true, or auto")


def save_run_config(config: Mapping[str, Any], *, output_dir: Path) -> Path:
    """Persist the effective YAML config once and reject mixed-config runs."""

    output_dir.mkdir(parents=True, exist_ok=True)
    destination = output_dir / RUN_CONFIG_NAME
    config_dict = dict(config)
    if destination.exists():
        existing = load_config(destination)
        if existing != config_dict:
            raise ValueError(
                f"FRS output directory already contains a different {RUN_CONFIG_NAME}: "
                f"{destination}. Use a new frs_training.output directory for the new "
                "parameter set, or restore the original config before resuming."
            )
        print(f"run_config={destination} (existing, matched)", flush=True)
        return destination

    serialized = yaml.safe_dump(
        config_dict,
        allow_unicode=True,
        sort_keys=False,
    )
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(serialized, encoding="utf-8")
    temporary.replace(destination)
    print(f"run_config={destination}", flush=True)
    return destination


def train_from_config(config: Mapping[str, Any]) -> None:
    datasets = config.get("datasets") or []
    if not isinstance(datasets, list) or not datasets:
        raise ValueError("config.datasets must be a non-empty list")
    action_cache = config.get("action_cache") or {}
    tactile_cache = config.get("tactile_embedding_cache") or {}
    model = config.get("model") or {}
    training = config.get("frs_training") or {}
    for name, value in (
        ("action_cache", action_cache),
        ("tactile_embedding_cache", tactile_cache),
        ("model", model),
        ("frs_training", training),
    ):
        if not isinstance(value, Mapping):
            raise ValueError(f"config.{name} must be a mapping")
    if not action_cache.get("root") or not tactile_cache.get("root"):
        raise ValueError("action_cache.root and tactile_embedding_cache.root are required")
    encoder_dir = Path(str(model["tactile_encoder_path"])).expanduser()
    if not encoder_dir.is_dir():
        raise FileNotFoundError(f"tactile encoder does not exist: {encoder_dir}")
    cache_dirs = [source_cache_dir(action_cache["root"], str(source["repo_id"])) for source in datasets]
    missing = [path for path in cache_dirs if not (path / "manifest.json").is_file()]
    if missing:
        raise FileNotFoundError(
            f"action caches are missing: {missing}. Run python -m train_frs.prepare_frs_caches first."
        )

    output_dir = Path(str(training["output"])).expanduser()
    rank_source_weights = training.get("high_gate_rank_source_weights") or {}
    if not isinstance(rank_source_weights, Mapping):
        raise ValueError("frs_training.high_gate_rank_source_weights must be a mapping")
    rank_source_weights = {
        str(repo_id): float(weight) for repo_id, weight in rank_source_weights.items()
    }
    tactile_keys = tuple(str(key) for key in model["tactile_keys"])
    tactile_num_tokens = _positive_int(model, "tactile_num_tokens", len(tactile_keys))
    if tactile_num_tokens != len(tactile_keys):
        raise ValueError(
            "model.tactile_num_tokens must match model.tactile_keys length: "
            f"{tactile_num_tokens} != {len(tactile_keys)}"
        )
    save_run_config(config, output_dir=output_dir)
    train_decoder(
        cache_dir=None,
        tactile_encoder_dir=encoder_dir,
        output_dir=output_dir,
        dataset_repo_id=None,
        dataset_root=None,
        tactile_window_divisor=_positive_int(training, "tactile_window_divisor", 1),
        history_stride=_positive_int(training, "history_stride", 3),
        loss_mode=str(training.get("loss_mode", "gated")),  # type: ignore[arg-type]
        gate_tau=float(training.get("gate_tau", 0.5)),
        gate_temperature=float(training.get("gate_temperature", 0.1)),
        gate_lambda=float(training.get("gate_lambda", 1.0)),
        aux_decode_weight=float(training.get("aux_decode_weight", 1.0)),
        aux_decode_steps=_positive_int(training, "aux_decode_steps", 10),
        low_gate_safety_weight=float(training.get("low_gate_safety_weight", 0.0)),
        low_gate_safety_margin=float(training.get("low_gate_safety_margin", 0.03)),
        rank_weight=float(training.get("rank_weight", 0.0)),
        rank_margin=float(training.get("rank_margin", 0.0)),
        repair_weight=float(training.get("repair_weight", 0.0)),
        repair_margin=float(training.get("repair_margin", 0.0)),
        rank_low_gate_threshold=float(training.get("rank_low_gate_threshold", 0.3)),
        rank_high_gate_threshold=float(training.get("rank_high_gate_threshold", 0.7)),
        state_conditioning=bool(model.get("state_conditioning", False)),
        state_dropout_rate=float(model.get("state_dropout_rate", 0.0)),
        model_dim=_positive_int(training, "model_dim", 256),
        depth=_positive_int(training, "depth", 6),
        num_heads=_positive_int(training, "num_heads", 4),
        mlp_ratio=_positive_int(training, "mlp_ratio", 4),
        learning_rate=float(training.get("learning_rate", 3e-4)),
        weight_decay=float(training.get("weight_decay", 1e-4)),
        grad_clip_norm=float(training.get("grad_clip_norm", 1.0)),
        warmup_epochs=int(training.get("warmup_epochs", 5)),
        lr_reference_dim=int(training.get("lr_reference_dim", 256)),
        min_learning_rate_ratio=float(training.get("min_lr_ratio", 0.1)),
        cosine_decay=str(training.get("lr_schedule", "cosine")) == "cosine",
        batch_size=_positive_int(training, "batch_size", 64),
        epochs=_positive_int(training, "epochs", 300),
        validation_steps=_positive_int(training, "validation_steps", 10),
        eval_every=_positive_int(training, "eval_every", 5),
        seed=int(training.get("seed", 42)),
        write_plots=bool(training.get("write_plots", True)),
        num_workers=0,
        prefetch_batches=1,
        load_threads=1,
        pipeline_prefetch=1,
        image_cache_size=0,
        encode_batch_size=1,
        resume=resolve_resume_mode(training.get("resume", False), output_dir=output_dir),
        resume_from=(
            None if training.get("resume_from") in (None, "") else Path(str(training["resume_from"])).expanduser()
        ),
        init_from=(
            None if training.get("init_from") in (None, "") else Path(str(training["init_from"])).expanduser()
        ),
        cache_dirs=cache_dirs,
        dataset_sources=datasets,
        tactile_embedding_cache_root=Path(str(tactile_cache["root"])).expanduser(),
        tactile_keys=tactile_keys,
        tactile_embedding_dim=int(model.get("tactile_embedding_dim", 512)),
        tactile_image_size=int(model.get("tactile_image_size", 224)),
        tactile_num_tokens=tactile_num_tokens,
        best_max_low_gate_unsafe_frac=float(
            training.get("best_max_low_gate_unsafe_frac", 0.1)
        ),
        best_min_high_gate_gain=float(training.get("best_min_high_gate_gain", 0.0)),
        best_min_high_gate_rank_satisfied=float(
            training.get("best_min_high_gate_rank_satisfied", 0.8)
        ),
        dataset_balanced_sampling=bool(training.get("dataset_balanced_sampling", False)),
        dataset_balanced_loss=bool(training.get("dataset_balanced_loss", False)),
        early_stop_patience=int(training.get("early_stop_patience", 0)),
        early_stop_min_evals=int(training.get("early_stop_min_evals", 0)),
        high_gate_rank_aggregation=str(
            training.get("high_gate_rank_aggregation", "balanced_mean")
        ),
        high_gate_rank_hard_fraction=float(
            training.get("high_gate_rank_hard_fraction", 0.3)
        ),
        high_gate_rank_worst_beta=float(
            training.get("high_gate_rank_worst_beta", 20.0)
        ),
        high_gate_rank_source_weights=rank_source_weights,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    train_from_config(load_config(args.config))


if __name__ == "__main__":
    main()
