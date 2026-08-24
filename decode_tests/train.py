"""Train the unconditional self-attention flow decoder on cache (x_base, VLA) pairs."""

from __future__ import annotations

import argparse
import csv
import dataclasses
import math
import pathlib
import sys
from collections.abc import Iterator, Sequence
from typing import Any

_PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from decode_tests.plots import HISTORY_FIELDS
from decode_tests.plots import plot_gt_vs_pred_samples
from decode_tests.plots import plot_training_curves
from decode_tests.split import EpisodeSplit
from decode_tests.split import build_episode_split
from decode_tests.split import write_split_json
from utils.cache import CachedPairs
from utils.cache import atomic_write_json
from utils.checkpoint import CHECKPOINT_NAME
from utils.checkpoint import save_checkpoint
from utils.model import DecoderConfig
from utils.model import FlowSolver
from utils.model import SelfAttentionFlowDecoder
from utils.model import decode_actions
from utils.model import flow_matching_loss_per_sample
from utils.model import make_optimizer
from utils.model import train_step


@dataclasses.dataclass(frozen=True)
class SplitEval:
    flow_loss: float
    mse_target: float
    rmse_target: float
    mae_target: float
    mse_gt: float
    rmse_gt: float
    mae_gt: float
    cache_indices: np.ndarray
    sample_flow_loss: np.ndarray
    sample_mse_target: np.ndarray
    sample_mse_gt: np.ndarray
    predictions: np.ndarray | None


def iter_index_batches(
    pairs: CachedPairs,
    indices: np.ndarray,
    *,
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> Iterator[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}.")
    order = np.asarray(indices, dtype=np.int64)
    if shuffle:
        order = np.random.default_rng(seed).permutation(order)
    for start in range(0, len(order), batch_size):
        batch_indices = order[start : start + batch_size]
        yield (
            batch_indices,
            np.asarray(pairs.arrays["x_base"][batch_indices], dtype=np.float32),
            np.asarray(pairs.arrays["target"][batch_indices], dtype=np.float32),
            np.asarray(pairs.arrays["gt_action"][batch_indices], dtype=np.float32),
        )


def evaluate_indices(
    model: SelfAttentionFlowDecoder,
    pairs: CachedPairs,
    indices: np.ndarray,
    *,
    batch_size: int,
    num_steps: int,
    solver: FlowSolver,
    keep_predictions: bool,
) -> SplitEval:
    if len(indices) == 0:
        raise ValueError("Cannot evaluate an empty index set.")

    cache_indices: list[np.ndarray] = []
    flow_losses: list[np.ndarray] = []
    mse_targets: list[np.ndarray] = []
    mae_targets: list[np.ndarray] = []
    mse_gts: list[np.ndarray] = []
    mae_gts: list[np.ndarray] = []
    predictions: list[np.ndarray] = []

    for batch_indices, x_base_np, target_np, gt_np in iter_index_batches(
        pairs, indices, batch_size=batch_size, shuffle=False, seed=0
    ):
        x_base = jnp.asarray(x_base_np)
        target = jnp.asarray(target_np)
        gt_action = jnp.asarray(gt_np)
        t = jnp.full((len(batch_indices),), 0.5, dtype=jnp.float32)
        flow_loss = flow_matching_loss_per_sample(model, x_base, target, t)
        prediction = decode_actions(model, x_base, num_steps=num_steps, solver=solver)
        diff_target = prediction - target
        diff_gt = prediction - gt_action
        mse_target = jnp.mean(jnp.square(diff_target), axis=(1, 2))
        mae_target = jnp.mean(jnp.abs(diff_target), axis=(1, 2))
        mse_gt = jnp.mean(jnp.square(diff_gt), axis=(1, 2))
        mae_gt = jnp.mean(jnp.abs(diff_gt), axis=(1, 2))

        cache_indices.append(batch_indices)
        flow_losses.append(np.asarray(jax.device_get(flow_loss)))
        mse_targets.append(np.asarray(jax.device_get(mse_target)))
        mae_targets.append(np.asarray(jax.device_get(mae_target)))
        mse_gts.append(np.asarray(jax.device_get(mse_gt)))
        mae_gts.append(np.asarray(jax.device_get(mae_gt)))
        if keep_predictions:
            predictions.append(np.asarray(jax.device_get(prediction), dtype=np.float32))

    all_indices = np.concatenate(cache_indices)
    all_flow = np.concatenate(flow_losses)
    all_mse_target = np.concatenate(mse_targets)
    all_mae_target = np.concatenate(mae_targets)
    all_mse_gt = np.concatenate(mse_gts)
    all_mae_gt = np.concatenate(mae_gts)
    return SplitEval(
        flow_loss=float(np.mean(all_flow)),
        mse_target=float(np.mean(all_mse_target)),
        rmse_target=float(np.sqrt(np.mean(all_mse_target))),
        mae_target=float(np.mean(all_mae_target)),
        mse_gt=float(np.mean(all_mse_gt)),
        rmse_gt=float(np.sqrt(np.mean(all_mse_gt))),
        mae_gt=float(np.mean(all_mae_gt)),
        cache_indices=all_indices,
        sample_flow_loss=all_flow,
        sample_mse_target=all_mse_target,
        sample_mse_gt=all_mse_gt,
        predictions=np.concatenate(predictions) if keep_predictions else None,
    )


def _blank_history_row(epoch: int, **filled: float) -> dict[str, float | int | str]:
    row: dict[str, float | int | str] = {field: "" for field in HISTORY_FIELDS}
    row["epoch"] = epoch
    row.update(filled)
    return row


def _float_metrics(metrics: dict[str, float | int | str]) -> dict[str, float]:
    return {key: float(value) for key, value in metrics.items() if isinstance(value, (int, float))}


def train_decoder(
    *,
    cache_dir: pathlib.Path,
    output_dir: pathlib.Path,
    model_dim: int,
    depth: int,
    num_heads: int,
    mlp_ratio: int,
    learning_rate: float,
    weight_decay: float,
    grad_clip_norm: float | None,
    warmup_epochs: int,
    min_learning_rate_ratio: float,
    cosine_decay: bool,
    batch_size: int,
    epochs: int,
    validation_steps: int,
    eval_every: int,
    solver: FlowSolver,
    seed: int,
    write_plots: bool,
) -> None:
    if epochs <= 0 or batch_size <= 0:
        raise ValueError("epochs and batch_size must be positive.")
    if warmup_epochs < 0:
        raise ValueError("warmup_epochs must be non-negative.")
    if not 0.0 <= min_learning_rate_ratio <= 1.0:
        raise ValueError("min_learning_rate_ratio must be in [0, 1].")
    if eval_every <= 0:
        raise ValueError(f"eval_every must be positive, got {eval_every}.")
    if validation_steps <= 0:
        raise ValueError(f"validation_steps must be positive, got {validation_steps}.")
    if solver not in ("euler", "fireflow"):
        raise ValueError(f"solver must be 'euler' or 'fireflow', got {solver!r}.")

    print(f"jax_devices={jax.devices()}", flush=True)
    if not any(device.platform == "gpu" for device in jax.devices()):
        print(
            "WARNING: no JAX GPU device visible; training will run on CPU.",
            flush=True,
        )

    pairs = CachedPairs(cache_dir)
    split = build_episode_split(pairs, seed=seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_split_json(
        output_dir / "split.json",
        split,
        cache_dir=cache_dir,
        records_sha256=str(pairs.manifest["records_sha256"]),
    )
    counts = split.counts()
    print(
        f"split=episode-disjoint 8:1:1 seed={seed} "
        f"episodes train/val/test="
        f"{counts['train_episodes']}/{counts['val_episodes']}/{counts['test_episodes']} "
        f"samples train/val/test="
        f"{counts['train_samples']}/{counts['val_samples']}/{counts['test_samples']}",
        flush=True,
    )

    decoder_config = DecoderConfig(
        action_dim=int(pairs.manifest["action_dim"]),
        action_horizon=int(pairs.manifest["action_horizon"]),
        model_dim=model_dim,
        depth=depth,
        num_heads=num_heads,
        mlp_ratio=mlp_ratio,
    )
    model = SelfAttentionFlowDecoder(decoder_config, rngs=nnx.Rngs(seed))
    steps_per_epoch = max(1, math.ceil(len(split.train_indices) / batch_size))
    warmup_steps = min(warmup_epochs, epochs) * steps_per_epoch
    total_steps = epochs * steps_per_epoch
    optimizer = make_optimizer(
        model,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        grad_clip_norm=grad_clip_norm,
        warmup_steps=warmup_steps,
        total_steps=total_steps,
        min_learning_rate_ratio=min_learning_rate_ratio,
        cosine_decay=cosine_decay,
    )
    print(
        f"decoder=SelfAttentionFlowDecoder dim={model_dim} depth={depth} "
        f"heads={num_heads} mlp_ratio={mlp_ratio} "
        f"action=[{decoder_config.action_horizon}, {decoder_config.action_dim}]",
        flush=True,
    )
    schedule_name = "warmup+cosine" if cosine_decay else "warmup+constant"
    print(
        f"optim=adamw lr={learning_rate:g} wd={weight_decay:g} "
        f"clip={grad_clip_norm} schedule={schedule_name} "
        f"warmup_epochs={warmup_epochs} steps_per_epoch={steps_per_epoch} "
        f"eval_every={eval_every} solver={solver} decode_steps={validation_steps}",
        flush=True,
    )

    history_path = output_dir / "history.csv"
    best_mse_target = float("inf")
    base_key = jax.random.key(seed)
    checkpoint_extra: dict[str, Any] = {
        "cache_records_sha256": pairs.manifest["records_sha256"],
        "cache_configuration": pairs.manifest["configuration"],
        "split_seed": seed,
        "solver": solver,
        "validation_steps": validation_steps,
        "eval_every": eval_every,
        **counts,
    }

    def _refresh_plots(*, announce: bool = False) -> None:
        if not write_plots:
            return
        try:
            loss_path, mse_path = plot_training_curves(history_path, output_dir=output_dir)
        except (FileNotFoundError, ValueError) as exc:
            print(f"warning: could not refresh training plots: {exc}", flush=True)
            return
        if announce:
            print(f"plots={loss_path} {mse_path}", flush=True)

    with history_path.open("w", newline="", encoding="utf-8") as history_file:
        writer = csv.DictWriter(history_file, fieldnames=HISTORY_FIELDS)
        writer.writeheader()

        for epoch in range(1, epochs + 1):
            losses: list[float] = []
            weights: list[int] = []
            for batch_number, (_, x_base_np, target_np, _) in enumerate(
                iter_index_batches(
                    pairs,
                    split.train_indices,
                    batch_size=batch_size,
                    shuffle=True,
                    seed=seed + epoch,
                )
            ):
                step_key = jax.random.fold_in(base_key, epoch * 1_000_000 + batch_number)
                loss = train_step(
                    model,
                    optimizer,
                    jnp.asarray(x_base_np),
                    jnp.asarray(target_np),
                    step_key,
                )
                losses.append(float(jax.device_get(loss)))
                weights.append(len(x_base_np))
                if batch_number == 0 or (batch_number + 1) % 20 == 0:
                    print(
                        f"epoch={epoch}/{epochs} batch={batch_number + 1}/{steps_per_epoch} "
                        f"flow_loss={losses[-1]:.6f}",
                        flush=True,
                    )

            train_loss = float(np.average(losses, weights=weights))
            run_eval = (epoch % eval_every == 0) or (epoch == epochs)
            if run_eval:
                validation = evaluate_indices(
                    model,
                    pairs,
                    split.val_indices,
                    batch_size=batch_size,
                    num_steps=validation_steps,
                    solver=solver,
                    keep_predictions=False,
                )
                metrics: dict[str, float] = {
                    "train_flow_loss": train_loss,
                    "val_flow_loss": validation.flow_loss,
                    "val_mse_target": validation.mse_target,
                    "val_rmse_target": validation.rmse_target,
                    "val_mae_target": validation.mae_target,
                    "val_mse_gt": validation.mse_gt,
                    "val_rmse_gt": validation.rmse_gt,
                    "val_mae_gt": validation.mae_gt,
                }
                writer.writerow(_blank_history_row(epoch, **metrics))
                history_file.flush()
                _refresh_plots()
                save_checkpoint(
                    output_dir / "last",
                    model,
                    epoch=epoch,
                    metrics=_float_metrics(metrics),
                    extra_metadata=checkpoint_extra,
                )
                if validation.mse_target < best_mse_target:
                    best_mse_target = validation.mse_target
                    save_checkpoint(
                        output_dir / "best",
                        model,
                        epoch=epoch,
                        metrics=_float_metrics(metrics),
                        extra_metadata=checkpoint_extra,
                    )
                print(
                    f"epoch={epoch}/{epochs} train_flow_loss={train_loss:.8f} "
                    f"val_flow_loss={validation.flow_loss:.8f} "
                    f"val_mse_target={validation.mse_target:.8f} "
                    f"val_mse_gt={validation.mse_gt:.8f}",
                    flush=True,
                )
            else:
                metrics = {"train_flow_loss": train_loss}
                writer.writerow(_blank_history_row(epoch, **metrics))
                history_file.flush()
                _refresh_plots()
                save_checkpoint(
                    output_dir / "last",
                    model,
                    epoch=epoch,
                    metrics=_float_metrics(metrics),
                    extra_metadata=checkpoint_extra,
                )
                print(
                    f"epoch={epoch}/{epochs} train_flow_loss={train_loss:.8f} (skip val)",
                    flush=True,
                )

    _refresh_plots(announce=True)
    _write_final_reports(
        model,
        pairs,
        split,
        output_dir=output_dir,
        batch_size=batch_size,
        num_steps=validation_steps,
        solver=solver,
        best_mse_target=best_mse_target,
    )


def _summarize_eval(prefix: str, result: SplitEval) -> dict[str, float]:
    return {
        f"{prefix}_flow_loss": result.flow_loss,
        f"{prefix}_mse_target": result.mse_target,
        f"{prefix}_rmse_target": result.rmse_target,
        f"{prefix}_mae_target": result.mae_target,
        f"{prefix}_mse_gt": result.mse_gt,
        f"{prefix}_rmse_gt": result.rmse_gt,
        f"{prefix}_mae_gt": result.mae_gt,
    }


def _write_final_reports(
    model: SelfAttentionFlowDecoder,
    pairs: CachedPairs,
    split: EpisodeSplit,
    *,
    output_dir: pathlib.Path,
    batch_size: int,
    num_steps: int,
    solver: FlowSolver,
    best_mse_target: float,
) -> None:
    validation = evaluate_indices(
        model,
        pairs,
        split.val_indices,
        batch_size=batch_size,
        num_steps=num_steps,
        solver=solver,
        keep_predictions=True,
    )
    test = evaluate_indices(
        model,
        pairs,
        split.test_indices,
        batch_size=batch_size,
        num_steps=num_steps,
        solver=solver,
        keep_predictions=False,
    )
    if validation.predictions is None:
        raise RuntimeError("Validation predictions missing after final eval.")

    gt_actions = np.asarray(pairs.arrays["gt_action"][validation.cache_indices], dtype=np.float32)
    sample_path = output_dir / "val_gt_vs_pred_samples.png"
    plot_gt_vs_pred_samples(
        sample_path,
        cache_indices=validation.cache_indices,
        sample_mse_gt=validation.sample_mse_gt,
        gt_actions=gt_actions,
        predictions=validation.predictions,
        episode_indices=np.asarray(pairs.arrays["episode_index"]),
        dataset_indices=np.asarray(pairs.arrays["dataset_index"]),
    )

    final_metrics = {
        "best_val_mse_target": float(best_mse_target),
        **_summarize_eval("val", validation),
        **_summarize_eval("test", test),
        "sample_plot": str(sample_path),
    }
    atomic_write_json(output_dir / "final_metrics.json", final_metrics)
    print(
        f"val_mse_gt={validation.mse_gt:.8f} "
        f"val_rmse_gt={validation.rmse_gt:.8f} "
        f"val_mae_gt={validation.mae_gt:.8f}",
        flush=True,
    )
    print(
        f"test_mse_gt={test.mse_gt:.8f} "
        f"test_mse_target={test.mse_target:.8f} "
        f"best_val_mse_target={best_mse_target:.8f}",
        flush=True,
    )
    print(f"sample_plot={sample_path}", flush=True)
    print(f"final_metrics={output_dir / 'final_metrics.json'}", flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Train SelfAttentionFlowDecoder on cache x_base noise "
            "to match VLA predicted actions (episode-disjoint 8:1:1)."
        )
    )
    parser.add_argument("--cache-dir", type=pathlib.Path, required=True)
    parser.add_argument("--output-dir", type=pathlib.Path, required=True)
    parser.add_argument("--model-dim", type=int, default=128)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--mlp-ratio", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--warmup-epochs", type=int, default=10)
    parser.add_argument("--min-lr-ratio", type=float, default=0.1)
    parser.add_argument("--lr-schedule", choices=("cosine", "constant"), default="cosine")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--validation-steps", type=int, default=10)
    parser.add_argument(
        "--eval-every",
        type=int,
        default=1,
        help="Run validation every N epochs (also always on the final epoch). Default 1.",
    )
    parser.add_argument("--solver", choices=("euler", "fireflow"), default="fireflow")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-plots", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    train_decoder(
        cache_dir=args.cache_dir,
        output_dir=args.output_dir,
        model_dim=args.model_dim,
        depth=args.depth,
        num_heads=args.num_heads,
        mlp_ratio=args.mlp_ratio,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        grad_clip_norm=args.grad_clip_norm if args.grad_clip_norm > 0 else None,
        warmup_epochs=args.warmup_epochs,
        min_learning_rate_ratio=args.min_lr_ratio,
        cosine_decay=args.lr_schedule == "cosine",
        batch_size=args.batch_size,
        epochs=args.epochs,
        validation_steps=args.validation_steps,
        eval_every=args.eval_every,
        solver=args.solver,
        seed=args.seed,
        write_plots=not args.no_plots,
    )


if __name__ == "__main__":
    main()
