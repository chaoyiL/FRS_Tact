#!/usr/bin/env python
"""Compare single-right-arm FRS checkpoints with deployment-aligned metrics."""

from __future__ import annotations

import argparse
import csv
import gc
import json
from pathlib import Path
from typing import Any

import numpy as np

from train_pi05_frs.utils.deployment_metrics import (
    deployment_aligned_single_hand_metrics,
)


DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "train_pi05_frs_right.yaml"
EXPECTED_PROFILE = "single-right-arm-7x10"
EXPECTED_ACTION_DIM = 10
GRIPPER_INDEX = 9


def _metadata(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def discover_validation_checkpoints(
    run_dir: Path,
    *,
    epochs: tuple[int, ...] | None = None,
) -> dict[int, Path]:
    """Resolve one immutable, evaluated checkpoint generation per epoch."""

    generation_root = Path(run_dir) / ".checkpoint-generations"
    if not generation_root.is_dir():
        raise FileNotFoundError(f"checkpoint generation directory not found: {generation_root}")
    requested = None if epochs is None else {int(epoch) for epoch in epochs}
    selected: dict[int, Path] = {}
    for generation in sorted(generation_root.iterdir()):
        if not generation.is_dir() or generation.name.startswith("."):
            continue
        metadata = _metadata(generation / "checkpoint.json")
        if metadata is None or "epoch" not in metadata:
            continue
        epoch = int(metadata["epoch"])
        if requested is not None and epoch not in requested:
            continue
        metrics = metadata.get("metrics")
        if not isinstance(metrics, dict) or "val_mse_gt" not in metrics:
            continue
        selected.setdefault(epoch, generation)
    if requested is not None:
        missing = sorted(requested - selected.keys())
        if missing:
            raise ValueError(f"missing validation checkpoints for epochs: {missing}")
    if not selected:
        raise ValueError(f"no evaluated checkpoint generations found in {generation_root}")
    return dict(sorted(selected.items()))


def _flatten_metrics(
    *,
    epoch: int,
    checkpoint: Path,
    summaries: dict[str, dict[str, float | int]],
    max_low_gate_unsafe_frac: float,
    min_high_gate_gain: float,
    min_high_gate_repair_satisfied_frac: float,
    max_high_gate_harm_p95: float,
    max_low_gate_regression_frac: float,
) -> dict[str, float | int | str]:
    row: dict[str, float | int | str] = {
        "epoch": int(epoch),
        "checkpoint": str(checkpoint.resolve()),
    }
    for scope, values in summaries.items():
        row.update({f"{scope}_{name}": value for name, value in values.items()})
        row[f"{scope}_checkpoint_feasible"] = int(
            float(values["low_unsafe_frac"]) <= max_low_gate_unsafe_frac
            and float(values["high_gain"]) >= min_high_gate_gain
            and float(values["high_repair_satisfied_frac"])
            >= min_high_gate_repair_satisfied_frac
            and float(values["high_gate_harm_p95"]) <= max_high_gate_harm_p95
            and float(values["low_gate_regression_frac"])
            <= max_low_gate_regression_frac
        )
    return row


def evaluate_checkpoints(
    *,
    config_path: Path,
    epochs: tuple[int, ...] | None,
    run_dir: Path | None,
    output_dir: Path | None,
    batch_size: int | None,
    num_steps: int | None,
    solver: str | None,
) -> Path:
    from train_pi05_frs.pi05_cache.cache import MultiCachedPairs
    from train_pi05_frs.tools.train_frs import (
        load_config,
        resolve_local_path,
        resolved_dataset_sources,
        source_cache_dir,
        validate_config,
    )
    from train_pi05_frs.utils.checkpoint import load_checkpoint
    from train_pi05_frs.utils.data import (
        CachedTactileEmbeddingBatches,
        gate_weights_from_change,
        resolve_tactile_window,
    )
    from train_pi05_frs.utils.model import decode_actions
    from train_pi05_frs.utils.objective_schema import COMPOSITE_GATED_LOSS_MODE
    import jax
    import jax.numpy as jnp

    config = load_config(Path(config_path))
    # Training path validation rejects an already-populated non-resume output.
    # This tool intentionally reads that completed/in-progress run, so validate
    # the schema here and let the cache/checkpoint readers validate their assets.
    validate_config(config, check_paths=False)
    datasets = config["datasets"]
    action_cache = config["action_cache"]
    tactile_cache = config["tactile_embedding_cache"]
    model_config = config["model"]
    training = config["frs_training"]
    if model_config.get("state_action_profile") != EXPECTED_PROFILE:
        raise ValueError(
            f"deployment comparison requires profile {EXPECTED_PROFILE!r}"
        )
    if int(model_config.get("action_dim", -1)) != EXPECTED_ACTION_DIM:
        raise ValueError("deployment comparison requires a 10D single-right-arm action")
    if str(training.get("loss_mode")) != COMPOSITE_GATED_LOSS_MODE:
        raise ValueError("deployment comparison requires composite_gated checkpoints")
    if not bool(tactile_cache.get("enabled")):
        raise ValueError("deployment comparison requires tactile_embedding_cache.enabled=true")

    resolved_run_dir = (
        resolve_local_path(str(training["output"]))
        if run_dir is None
        else Path(run_dir).expanduser().resolve()
    )
    resolved_output_dir = (
        resolved_run_dir / "evaluation_deployment_aligned"
        if output_dir is None
        else Path(output_dir).expanduser().resolve()
    )
    checkpoints = discover_validation_checkpoints(resolved_run_dir, epochs=epochs)
    sources = resolved_dataset_sources(datasets)
    cache_dirs = [
        source_cache_dir(action_cache["root"], str(source["repo_id"]))
        for source in datasets
    ]
    pairs = MultiCachedPairs(
        cache_dirs,
        source_names=[str(source["repo_id"]) for source in sources],
    )
    if int(pairs.manifest["action_dim"]) != EXPECTED_ACTION_DIM:
        raise ValueError("action cache is not 10D single-right-arm data")

    first_checkpoint = next(iter(checkpoints.values()))
    first_metadata = _metadata(first_checkpoint / "checkpoint.json")
    if first_metadata is None:
        raise ValueError(f"invalid checkpoint metadata: {first_checkpoint}")
    first_extra = first_metadata.get("extra_metadata")
    if not isinstance(first_extra, dict):
        raise ValueError("checkpoint is missing extra_metadata")
    tactile_window = resolve_tactile_window(
        action_horizon=int(pairs.manifest["action_horizon"]),
        window_divisor=int(first_extra["tactile_window_divisor"]),
    )
    tactile_keys = tuple(str(key) for key in model_config["tactile_keys"])
    conditioner = CachedTactileEmbeddingBatches(
        pairs,
        sources=sources,
        tactile_cache_root=resolve_local_path(str(tactile_cache["root"])),
        tactile_encoder_dir=resolve_local_path(str(model_config["tactile_encoder_path"])),
        tactile_keys=tactile_keys,
        tactile_window=tactile_window,
        history_stride=int(training.get("history_stride", 3)),
        embedding_dim=int(model_config.get("tactile_embedding_dim", 512)),
        image_size=int(model_config.get("tactile_image_size", 224)),
        build_episode_baselines=True,
    )

    resolved_batch_size = int(batch_size or training.get("batch_size", 128))
    resolved_num_steps = int(num_steps or training.get("validation_steps", 10))
    resolved_solver = str(solver or training.get("aux_decode_solver", "fireflow"))
    rows: list[dict[str, float | int | str]] = []
    try:
        for epoch, checkpoint in checkpoints.items():
            decoder, metadata = load_checkpoint(checkpoint)
            extra = metadata.get("extra_metadata")
            if not isinstance(extra, dict):
                raise ValueError(f"checkpoint epoch {epoch} is missing extra_metadata")
            if str(extra.get("loss_mode")) != COMPOSITE_GATED_LOSS_MODE:
                raise ValueError(f"checkpoint epoch {epoch} is not composite_gated")
            if int(decoder.config.action_dim) != EXPECTED_ACTION_DIM:
                raise ValueError(f"checkpoint epoch {epoch} is not 10D")
            expected_digest = extra.get("cache_records_sha256")
            if expected_digest and expected_digest != pairs.manifest["records_sha256"]:
                raise ValueError(f"checkpoint epoch {epoch} was trained from another cache")

            low_threshold = float(extra.get("low_gate_threshold", 0.3))
            high_threshold = float(extra.get("high_gate_threshold", 0.7))
            prediction_parts: list[np.ndarray] = []
            gt_parts: list[np.ndarray] = []
            vla_parts: list[np.ndarray] = []
            gate_parts: list[np.ndarray] = []
            for (
                indices,
                x_base,
                vla_actions,
                gt_actions,
                state,
                tactile_seq,
            ) in conditioner.batches(
                "val", batch_size=resolved_batch_size, shuffle=False, seed=0
            ):
                prediction = decode_actions(
                    decoder,
                    jnp.asarray(x_base),
                    jnp.asarray(tactile_seq),
                    num_steps=resolved_num_steps,
                    solver=resolved_solver,
                    state=jnp.asarray(state),
                )
                prediction_parts.append(
                    np.asarray(jax.device_get(prediction), dtype=np.float32)
                )
                gt_parts.append(np.asarray(gt_actions, dtype=np.float32))
                vla_parts.append(np.asarray(vla_actions, dtype=np.float32))
                current_tokens = np.asarray(tactile_seq[:, -1, :, :], dtype=np.float32)
                tactile_change = conditioner.tactile_change_for_cache_indices(
                    indices, current_tokens
                )
                gate_parts.append(
                    gate_weights_from_change(
                        tactile_change,
                        tau=float(extra["gate_tau"]),
                        temperature=float(extra["gate_temperature"]),
                    )
                )
            summaries = deployment_aligned_single_hand_metrics(
                np.concatenate(prediction_parts),
                np.concatenate(gt_parts),
                np.concatenate(vla_parts),
                np.concatenate(gate_parts),
                gripper_index=GRIPPER_INDEX,
                low_gate_threshold=low_threshold,
                high_gate_threshold=high_threshold,
                low_gate_safety_margin=float(extra.get("low_gate_safety_margin", 0.03)),
                low_gate_regression_margin=float(
                    extra.get("low_gate_regression_margin", 0.005)
                ),
                rank_margin=float(extra.get("rank_margin", 0.0)),
                repair_margin=float(extra.get("repair_margin", 0.0)),
            )
            row = _flatten_metrics(
                epoch=epoch,
                checkpoint=checkpoint,
                summaries=dict(summaries),
                max_low_gate_unsafe_frac=float(
                    extra.get("best_max_low_gate_unsafe_frac", 0.1)
                ),
                min_high_gate_gain=float(extra.get("best_min_high_gate_gain", 0.0)),
                min_high_gate_repair_satisfied_frac=float(
                    extra.get("best_min_high_gate_repair_satisfied_frac", 0.8)
                ),
                max_high_gate_harm_p95=float(
                    extra.get("best_max_high_gate_harm_p95", 0.03)
                ),
                max_low_gate_regression_frac=float(
                    extra.get("best_max_low_gate_regression_frac", 0.05)
                ),
            )
            rows.append(row)
            print(json.dumps(row, ensure_ascii=False), flush=True)
            del prediction_parts, gt_parts, vla_parts, gate_parts, decoder
            gc.collect()
    finally:
        conditioner.close()

    resolved_output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = resolved_output_dir / "checkpoint_comparison.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary_path = resolved_output_dir / "checkpoint_comparison.json"
    summary_path.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"comparison_csv={csv_path}", flush=True)
    print(f"comparison_json={summary_path}", flush=True)
    return csv_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare single-right-arm checkpoints using arm9 and runtime10 metrics."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--epochs", type=int, nargs="+", default=None)
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-steps", type=int, default=None)
    parser.add_argument("--solver", choices=("euler", "fireflow"), default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    evaluate_checkpoints(
        config_path=args.config,
        epochs=None if args.epochs is None else tuple(args.epochs),
        run_dir=args.run_dir,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        num_steps=args.num_steps,
        solver=args.solver,
    )


if __name__ == "__main__":
    main()
