#!/usr/bin/env python
"""Verify no-GRU FRS: GT Flow Matching vs direct VLA-action regression."""

from __future__ import annotations

import argparse
import math
import pathlib
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Literal

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

RunName = Literal["tactile", "zero_tactile_tokens", "vla_direct", "vla_tactile_direct"]
RUN_NAMES: tuple[RunName, ...] = (
    "tactile",
    "zero_tactile_tokens",
    "vla_direct",
    "vla_tactile_direct",
)
RUN_SPECS: dict[RunName, dict[str, bool]] = {
    "tactile": {"use_flow_matching": True, "zero_tactile_tokens": False},
    "zero_tactile_tokens": {"use_flow_matching": True, "zero_tactile_tokens": True},
    "vla_direct": {"use_flow_matching": False, "zero_tactile_tokens": True},
    "vla_tactile_direct": {"use_flow_matching": False, "zero_tactile_tokens": False},
}
DEFAULT_CONFIG = Path(__file__).resolve().parent / "configs" / "verify_gt_fm.yaml"
HISTORY_FIELDS = (
    "epoch",
    "train_loss",
    "val_mse_frs_gt",
    "val_mse_vla_gt",
    "val_relative_reduction",
)


def _read_relative_reduction_history(history_path: Path) -> tuple[list[int], list[float]]:
    import csv as csv_module

    epochs: list[int] = []
    reductions: list[float] = []
    with history_path.open(encoding="utf-8", newline="") as file:
        for raw in csv_module.DictReader(file):
            epoch_text = (raw.get("epoch") or "").strip()
            reduction_text = (raw.get("val_relative_reduction") or "").strip()
            if not epoch_text or not reduction_text:
                continue
            epochs.append(int(epoch_text))
            reductions.append(float(reduction_text))
    return epochs, reductions


def collect_verify_histories(output_dir: Path, run_name: RunName) -> dict[str, Path]:
    histories = {run_name: output_dir / "history.csv"}
    parent = output_dir.parent
    for name in RUN_NAMES:
        path = parent / name / "history.csv"
        if name != run_name and path.is_file():
            histories[name] = path
    return {name: path for name, path in histories.items() if path.is_file()}


def plot_relative_reduction(
    histories: Mapping[str, Path],
    output_path: Path,
) -> Path:
    """Plot per-epoch validation relative reduction for one or more verify runs."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    series = []
    for name, history_path in histories.items():
        epochs, reductions = _read_relative_reduction_history(history_path)
        if epochs:
            series.append((name, epochs, reductions))
    if not series:
        raise ValueError(f"no relative-reduction rows found in {sorted(histories)}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(8, 4.5))
    for name, epochs, reductions in series:
        axis.plot(
            epochs,
            [100.0 * value for value in reductions],
            marker="o",
            label=name,
        )
    axis.axhline(0.0, color="0.5", linewidth=1.0, linestyle="--")
    axis.set_xlabel("epoch")
    axis.set_ylabel("val relative reduction (%)")
    axis.set_title("verify_gt_fm: (MSE_VLA - MSE_pred) / MSE_VLA")
    if len(series) > 1:
        axis.legend()
    axis.grid(True, alpha=0.3)
    figure.tight_layout()
    figure.savefig(output_path, dpi=120)
    plt.close(figure)
    return output_path


def write_relative_reduction_plots(output_dir: Path, run_name: RunName) -> list[Path]:
    histories = collect_verify_histories(output_dir, run_name)
    written = [plot_relative_reduction({run_name: output_dir / "history.csv"}, output_dir / "relative_reduction.png")]
    if len(histories) > 1:
        written.append(plot_relative_reduction(histories, output_dir.parent / "relative_reduction.png"))
    return written


def relative_reduction(mse_frs_gt: float, mse_vla_gt: float) -> float:
    """Return ``(MSE_VLA - MSE_FRS) / MSE_VLA``; 0 when the VLA baseline is 0."""

    baseline = float(mse_vla_gt)
    if baseline == 0.0:
        return 0.0
    return (baseline - float(mse_frs_gt)) / baseline


def summarize_comparison(run_metrics: Mapping[str, Mapping[str, float]]) -> dict[str, Any]:
    """Combine run metrics and tactile-vs-control reduction gaps."""

    summary: dict[str, Any] = {name: dict(metrics) for name, metrics in run_metrics.items()}
    if "tactile" in run_metrics and "zero_tactile_tokens" in run_metrics:
        gap = float(run_metrics["tactile"]["relative_reduction"]) - float(
            run_metrics["zero_tactile_tokens"]["relative_reduction"]
        )
        summary["reduction_gap"] = gap
        summary["reduction_gap_fm"] = gap
    if "vla_tactile_direct" in run_metrics and "vla_direct" in run_metrics:
        summary["reduction_gap_direct"] = float(
            run_metrics["vla_tactile_direct"]["relative_reduction"]
        ) - float(run_metrics["vla_direct"]["relative_reduction"])
    return summary


def format_run_line(name: str, metrics: Mapping[str, float]) -> str:
    return (
        f"{name}: MSE_FRS={metrics['mse_frs_gt']:.8f}  "
        f"MSE_VLA={metrics['mse_vla_gt']:.8f}  "
        f"reduction={100.0 * float(metrics['relative_reduction']):.2f}%"
    )


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


def _require_mapping(config: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = config.get(key) or {}
    if not isinstance(value, Mapping):
        raise ValueError(f"config.{key} must be a mapping")
    return value


def evaluate_verify_split(
    model: Any,
    conditioner: Any,
    *,
    split: Literal["train", "val"],
    batch_size: int,
    num_steps: int,
    solver: str,
    use_flow_matching: bool,
) -> dict[str, float]:
    import jax
    import jax.numpy as jnp
    import numpy as np

    from train_frs.utils.model import decode_actions

    mse_frs_parts: list[np.ndarray] = []
    mse_vla_parts: list[np.ndarray] = []
    for (
        _indices,
        x_base_np,
        predicted_np,
        gt_action_np,
        _state_np,
        tactile_input,
    ) in conditioner.batches(split, batch_size=batch_size, shuffle=False, seed=0):
        predicted = jnp.asarray(predicted_np)
        tactile = jnp.asarray(tactile_input)
        if use_flow_matching:
            decoded = decode_actions(
                model,
                jnp.asarray(x_base_np),
                tactile,
                num_steps=num_steps,
                solver=solver,  # type: ignore[arg-type]
            )
        else:
            dummy_t = jnp.zeros((predicted.shape[0],), dtype=jnp.float32)
            decoded = model(predicted, dummy_t, tactile)
        gt_action = jnp.asarray(gt_action_np)
        mse_frs_parts.append(
            np.asarray(jax.device_get(jnp.mean(jnp.square(decoded - gt_action), axis=(1, 2))))
        )
        mse_vla_parts.append(
            np.asarray(jax.device_get(jnp.mean(jnp.square(predicted - gt_action), axis=(1, 2))))
        )
    if not mse_frs_parts:
        raise ValueError(f"No samples found for split {split!r}.")
    mse_frs = float(np.mean(np.concatenate(mse_frs_parts)))
    mse_vla = float(np.mean(np.concatenate(mse_vla_parts)))
    return {
        "mse_frs_gt": mse_frs,
        "mse_vla_gt": mse_vla,
        "relative_reduction": relative_reduction(mse_frs, mse_vla),
    }


def _fm_train_step(model, optimizer, x_base, gt_action, tactile_seq, key):
    import jax
    import jax.numpy as jnp
    from flax import nnx

    from train_frs.utils.model import flow_matching_loss_per_sample

    t = jax.random.uniform(key, (x_base.shape[0],), minval=0.0, maxval=1.0)

    def loss_fn(candidate):
        return jnp.mean(
            flow_matching_loss_per_sample(candidate, x_base, gt_action, t, tactile_seq)
        )

    loss, gradients = nnx.value_and_grad(loss_fn)(model)
    optimizer.update(model, gradients)
    return loss


def _direct_train_step(model, optimizer, vla_action, gt_action, tactile_seq, key):
    import jax.numpy as jnp
    from flax import nnx

    del key
    dummy_t = jnp.zeros((vla_action.shape[0],), dtype=jnp.float32)

    def loss_fn(candidate):
        predicted = candidate(vla_action, dummy_t, tactile_seq)
        return jnp.mean(jnp.square(predicted - gt_action))

    loss, gradients = nnx.value_and_grad(loss_fn)(model)
    optimizer.update(model, gradients)
    return loss


def train_one_run(
    *,
    run_name: RunName,
    pairs: Any,
    conditioner: Any,
    output_dir: Path,
    seed: int,
    model_dim: int,
    depth: int,
    num_heads: int,
    mlp_ratio: int,
    tactile_num_tokens: int,
    learning_rate: float,
    weight_decay: float,
    grad_clip_norm: float,
    warmup_epochs: int,
    lr_reference_dim: int,
    min_lr_ratio: float,
    cosine_decay: bool,
    batch_size: int,
    epochs: int,
    decode_steps: int,
    decode_solver: str,
    extra_metadata: Mapping[str, Any],
) -> dict[str, float]:
    import csv as csv_module

    import jax
    import jax.numpy as jnp
    import numpy as np
    from flax import nnx

    from train_frs.utils.checkpoint import save_checkpoint
    from train_frs.utils.model import (
        DEFAULT_GRU_HIDDEN_DIM,
        DecoderConfig,
        TactileConditionedFlowDecoder,
        make_optimizer,
        resolve_peak_learning_rate,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    spec = RUN_SPECS[run_name]
    action_horizon = int(pairs.manifest["action_horizon"])
    decoder_config = DecoderConfig(
        action_dim=int(pairs.manifest["action_dim"]),
        action_horizon=action_horizon,
        tactile_window=1,
        gru_hidden_dim=DEFAULT_GRU_HIDDEN_DIM,
        resnet_embedding_dim=int(conditioner.resnet_embedding_dim),
        model_dim=model_dim,
        depth=depth,
        num_heads=num_heads,
        mlp_ratio=mlp_ratio,
        num_tactile_tokens=tactile_num_tokens,
        state_dim=0,
        state_conditioning=False,
        tactile_encoder_trainable=False,
        use_gru=False,
        zero_tactile_tokens=spec["zero_tactile_tokens"],
        use_flow_matching=spec["use_flow_matching"],
    )
    model = TactileConditionedFlowDecoder(decoder_config, rngs=nnx.Rngs(seed))
    train_samples = len(pairs.indices("train"))
    steps_per_epoch = max(1, math.ceil(train_samples / batch_size))
    total_steps = max(1, steps_per_epoch * epochs)
    peak_lr = resolve_peak_learning_rate(
        learning_rate,
        model_dim=model_dim,
        lr_reference_dim=lr_reference_dim,
    )
    optimizer = make_optimizer(
        model,
        learning_rate=peak_lr,
        weight_decay=weight_decay,
        grad_clip_norm=grad_clip_norm,
        warmup_steps=steps_per_epoch * max(0, warmup_epochs),
        total_steps=total_steps,
        min_learning_rate_ratio=min_lr_ratio,
        cosine_decay=cosine_decay,
    )
    train_step = nnx.jit(_fm_train_step if spec["use_flow_matching"] else _direct_train_step)
    history_path = output_dir / "history.csv"
    best_metrics: dict[str, float] | None = None
    print(
        f"run={run_name} samples_train={train_samples} "
        f"samples_val={len(pairs.indices('val'))} "
        f"zero_tactile_tokens={decoder_config.zero_tactile_tokens} "
        f"use_gru={decoder_config.use_gru} "
        f"use_flow_matching={decoder_config.use_flow_matching}",
        flush=True,
    )
    with history_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv_module.DictWriter(file, fieldnames=list(HISTORY_FIELDS))
        writer.writeheader()
        file.flush()
        for epoch in range(1, epochs + 1):
            losses: list[float] = []
            weights: list[int] = []
            for batch_number, (
                _indices,
                x_base_np,
                predicted_np,
                gt_action_np,
                _state_np,
                tactile_input,
            ) in enumerate(
                conditioner.batches("train", batch_size=batch_size, shuffle=True, seed=seed + epoch)
            ):
                action_input = x_base_np if spec["use_flow_matching"] else predicted_np
                loss = train_step(
                    model,
                    optimizer,
                    jnp.asarray(action_input),
                    jnp.asarray(gt_action_np),
                    jnp.asarray(tactile_input),
                    jax.random.fold_in(jax.random.key(seed), epoch * 1_000_000 + batch_number),
                )
                losses.append(float(jax.device_get(loss)))
                weights.append(len(x_base_np))
                if batch_number == 0 or (batch_number + 1) % 20 == 0:
                    print(
                        f"run={run_name} epoch={epoch}/{epochs} "
                        f"batch={batch_number + 1}/{steps_per_epoch} "
                        f"train_loss={losses[-1]:.6f}",
                        flush=True,
                    )
            train_loss = float(np.average(losses, weights=weights))
            val_metrics = evaluate_verify_split(
                model,
                conditioner,
                split="val",
                batch_size=batch_size,
                num_steps=decode_steps,
                solver=decode_solver,
                use_flow_matching=spec["use_flow_matching"],
            )
            row = {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_mse_frs_gt": val_metrics["mse_frs_gt"],
                "val_mse_vla_gt": val_metrics["mse_vla_gt"],
                "val_relative_reduction": val_metrics["relative_reduction"],
            }
            writer.writerow(row)
            file.flush()
            plot_paths = write_relative_reduction_plots(output_dir, run_name)
            print(
                f"run={run_name} epoch={epoch}/{epochs} train_loss={train_loss:.6f} "
                f"{format_run_line('val', val_metrics)} "
                f"plot={plot_paths[0]}",
                flush=True,
            )
            checkpoint_metrics = {
                "train_loss": train_loss,
                **{f"val_{key}": value for key, value in val_metrics.items()},
            }
            run_metadata = {
                **dict(extra_metadata),
                "run_name": run_name,
                "loss": "fm_gt" if spec["use_flow_matching"] else "direct_mse",
                "use_flow_matching": spec["use_flow_matching"],
                "zero_tactile_tokens": spec["zero_tactile_tokens"],
            }
            save_checkpoint(
                output_dir / "last",
                model,
                epoch=epoch,
                metrics=checkpoint_metrics,
                extra_metadata=run_metadata,
            )
            if best_metrics is None or val_metrics["mse_frs_gt"] < best_metrics["mse_frs_gt"]:
                best_metrics = dict(val_metrics)
                save_checkpoint(
                    output_dir / "best",
                    model,
                    epoch=epoch,
                    metrics=checkpoint_metrics,
                    extra_metadata=run_metadata,
                )
    assert best_metrics is not None
    return best_metrics


def verify_from_config(config: Mapping[str, Any], *, runs: Sequence[RunName] = RUN_NAMES) -> dict[str, Any]:
    from train_frs.train_frs import resolve_decode_solver
    from train_frs.utils.data import CachedTactileEmbeddingBatches
    from utils.cache import MultiCachedPairs, atomic_write_json

    datasets = config.get("datasets") or []
    if not isinstance(datasets, list) or not datasets:
        raise ValueError("config.datasets must be a non-empty list")
    action_cache = _require_mapping(config, "action_cache")
    tactile_cache = _require_mapping(config, "tactile_embedding_cache")
    model_cfg = _require_mapping(config, "model")
    training = _require_mapping(config, "verify_training")
    if not action_cache.get("root") or not tactile_cache.get("root"):
        raise ValueError("action_cache.root and tactile_embedding_cache.root are required")
    from train_frs.prepare_frs_caches import prepare_tactile_embeddings_from_config

    prepare_tactile_embeddings_from_config(config)
    encoder_dir = Path(str(model_cfg["tactile_encoder_path"])).expanduser()
    if not encoder_dir.is_dir():
        raise FileNotFoundError(f"tactile encoder does not exist: {encoder_dir}")
    tactile_keys = tuple(str(key) for key in model_cfg["tactile_keys"])
    tactile_num_tokens = _positive_int(model_cfg, "tactile_num_tokens", len(tactile_keys))
    if tactile_num_tokens != len(tactile_keys):
        raise ValueError(
            "model.tactile_num_tokens must match model.tactile_keys length: "
            f"{tactile_num_tokens} != {len(tactile_keys)}"
        )
    cache_dirs = [source_cache_dir(action_cache["root"], str(source["repo_id"])) for source in datasets]
    missing = [path for path in cache_dirs if not (path / "manifest.json").is_file()]
    if missing:
        raise FileNotFoundError(
            f"action caches are missing: {missing}. Run python -m train_frs.prepare_frs_caches first."
        )
    output_root = Path(str(training["output"])).expanduser()
    output_root.mkdir(parents=True, exist_ok=True)
    source_names = [str(source["repo_id"]) for source in datasets]
    pairs = MultiCachedPairs(cache_dirs, source_names=source_names)
    conditioner = CachedTactileEmbeddingBatches(
        pairs,
        sources=datasets,
        tactile_cache_root=Path(str(tactile_cache["root"])).expanduser(),
        tactile_encoder_dir=encoder_dir,
        tactile_keys=tactile_keys,
        tactile_window=1,
        history_stride=1,
        embedding_dim=int(model_cfg.get("tactile_embedding_dim", 512)),
        image_size=int(model_cfg.get("tactile_image_size", 224)),
        build_episode_baselines=False,
        return_raw_images=False,
        num_workers=0,
    )
    decode_solver = resolve_decode_solver(training.get("decode_solver", "fireflow"))
    shared = {
        "pairs": pairs,
        "conditioner": conditioner,
        "seed": int(training.get("seed", 42)),
        "model_dim": _positive_int(training, "model_dim", 256),
        "depth": _positive_int(training, "depth", 6),
        "num_heads": _positive_int(training, "num_heads", 4),
        "mlp_ratio": _positive_int(training, "mlp_ratio", 4),
        "tactile_num_tokens": tactile_num_tokens,
        "learning_rate": float(training.get("learning_rate", 1.0e-4)),
        "weight_decay": float(training.get("weight_decay", 1.0e-4)),
        "grad_clip_norm": float(training.get("grad_clip_norm", 1.0)),
        "warmup_epochs": int(training.get("warmup_epochs", 1)),
        "lr_reference_dim": int(training.get("lr_reference_dim", 256)),
        "min_lr_ratio": float(training.get("min_lr_ratio", 0.1)),
        "cosine_decay": str(training.get("lr_schedule", "cosine")) == "cosine",
        "batch_size": _positive_int(training, "batch_size", 128),
        "epochs": _positive_int(training, "epochs", 30),
        "decode_steps": _positive_int(training, "decode_steps", 10),
        "decode_solver": decode_solver,
        "extra_metadata": {
            "experiment": "verify_gt_fm",
            "loss": "fm_gt",
            "use_gru": False,
            "state_conditioning": False,
            "tactile_window": 1,
            "cache_records_sha256": pairs.manifest["records_sha256"],
        },
    }
    run_metrics: dict[str, dict[str, float]] = {}
    try:
        for run_name in runs:
            if run_name not in RUN_NAMES:
                raise ValueError(f"unknown run {run_name!r}; expected one of {RUN_NAMES}")
            run_metrics[run_name] = train_one_run(
                run_name=run_name,
                output_dir=output_root / run_name,
                **shared,
            )
    finally:
        conditioner.close()

    summary = summarize_comparison(run_metrics)
    atomic_write_json(output_root / "metrics.json", summary)
    for name, metrics in run_metrics.items():
        print(format_run_line(name, metrics), flush=True)
    if "reduction_gap" in summary:
        print(f"reduction_gap_fm={100.0 * float(summary['reduction_gap']):.2f}%", flush=True)
    if "reduction_gap_direct" in summary:
        print(
            f"reduction_gap_direct={100.0 * float(summary['reduction_gap_direct']):.2f}%",
            flush=True,
        )
    print(f"metrics={output_root / 'metrics.json'}", flush=True)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Train no-GRU current-frame FRS controls: GT Flow Matching "
            "(tactile / zero tokens) and direct VLA-action regression "
            "(VLA only / VLA+tactile). Report MSE reduction vs VLA."
        )
    )
    parser.add_argument("--config", type=pathlib.Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--run",
        choices=RUN_NAMES,
        action="append",
        dest="runs",
        help="Train only this run. Repeat to select a subset. Default: all four runs.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    runs: tuple[RunName, ...] = tuple(args.runs) if args.runs else RUN_NAMES
    verify_from_config(load_config(args.config), runs=runs)


if __name__ == "__main__":
    main()
