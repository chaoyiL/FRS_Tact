"""Train tactile-conditioned flow decoder.

IMPORTANT: keep module-level imports free of JAX/Flax/data loaders. Mp spawn workers
re-import this file as ``__main__`` under ``CUDA_VISIBLE_DEVICES=""``; eager ``import jax``
there causes ``CUDA_ERROR_NO_DEVICE`` spam and fails the light-import guard.
"""

from __future__ import annotations

import argparse
import pathlib
from collections.abc import Sequence
from typing import Literal

LossMode = Literal["gt", "gated"]


def _resolve_resume_dir(
    *,
    output_dir: pathlib.Path,
    resume: bool,
    resume_from: pathlib.Path | None,
) -> pathlib.Path | None:
    if resume_from is not None:
        return resume_from
    if resume:
        return output_dir / "last"
    return None


def train_decoder(
    *,
    cache_dir: pathlib.Path,
    tactile_encoder_dir: pathlib.Path,
    output_dir: pathlib.Path,
    dataset_repo_id: str | None,
    dataset_root: pathlib.Path | None,
    tactile_window_divisor: int,
    history_stride: int,
    loss_mode: LossMode,
    gate_tau: float,
    gate_temperature: float,
    gate_lambda: float,
    model_dim: int,
    depth: int,
    num_heads: int,
    mlp_ratio: int,
    learning_rate: float,
    weight_decay: float,
    grad_clip_norm: float | None,
    warmup_epochs: int,
    lr_reference_dim: int | None,
    min_learning_rate_ratio: float,
    cosine_decay: bool,
    batch_size: int,
    epochs: int,
    validation_steps: int,
    eval_every: int,
    seed: int,
    write_plots: bool,
    num_workers: int,
    prefetch_batches: int,
    load_threads: int,
    pipeline_prefetch: int,
    image_cache_size: int,
    encode_batch_size: int,
    resume: bool = False,
    resume_from: pathlib.Path | None = None,
) -> None:
    import csv
    import json

    import jax
    import jax.numpy as jnp
    import numpy as np
    from flax import nnx

    from tactile_flow_steering.utils.checkpoint import CHECKPOINT_NAME
    from tactile_flow_steering.utils.checkpoint import load_checkpoint
    from tactile_flow_steering.utils.checkpoint import load_optimizer_state
    from tactile_flow_steering.utils.checkpoint import restore_optimizer_state
    from tactile_flow_steering.utils.checkpoint import save_checkpoint
    from tactile_flow_steering.utils.data import TactileConditionedBatches
    from tactile_flow_steering.utils.data import gate_weights_from_change
    from tactile_flow_steering.utils.data import resolve_tactile_window
    from tactile_flow_steering.utils.metrics import evaluate_split
    from tactile_flow_steering.utils.model import DEFAULT_GRU_HIDDEN_DIM
    from tactile_flow_steering.utils.model import DecoderConfig
    from tactile_flow_steering.utils.model import TactileConditionedFlowDecoder
    from tactile_flow_steering.utils.model import make_optimizer
    from tactile_flow_steering.utils.model import resolve_peak_learning_rate
    from tactile_flow_steering.utils.model import train_step
    from tactile_flow_steering.utils.history_plot import plot_training_history
    from utils.cache import CachedPairs

    history_fields = [
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
    ]

    def _blank_history_row(epoch: int, **filled: float | str) -> dict[str, float | int | str]:
        row: dict[str, float | int | str] = {field: "" for field in history_fields}
        row["epoch"] = epoch
        row.update(filled)
        return row

    def _weighted_mean(values: list[float], counts: list[int]) -> float:
        return float(np.average(np.asarray(values, dtype=np.float64), weights=counts))

    if epochs <= 0 or batch_size <= 0:
        raise ValueError("epochs and batch_size must be positive.")
    if warmup_epochs < 0:
        raise ValueError("warmup_epochs must be non-negative.")
    if not 0.0 <= min_learning_rate_ratio <= 1.0:
        raise ValueError("min_learning_rate_ratio must be in [0, 1].")
    if loss_mode not in ("gt", "gated"):
        raise ValueError(f"loss_mode must be 'gt' or 'gated', got {loss_mode!r}.")
    if gate_temperature <= 0:
        raise ValueError(f"gate_temperature must be positive, got {gate_temperature}.")
    if gate_lambda < 0:
        raise ValueError(f"gate_lambda must be non-negative, got {gate_lambda}.")
    if eval_every <= 0:
        raise ValueError(f"eval_every must be positive, got {eval_every}.")

    resume_dir = _resolve_resume_dir(output_dir=output_dir, resume=resume, resume_from=resume_from)
    start_epoch = 1
    resume_metadata: dict | None = None
    resumed_opt_state = None
    resumed_opt_step: int | None = None
    if resume_dir is not None:
        if not (resume_dir / CHECKPOINT_NAME).exists():
            raise FileNotFoundError(f"Resume checkpoint not found: {resume_dir}")
        model, resume_metadata = load_checkpoint(resume_dir)
        resumed_opt_state, resumed_opt_step = load_optimizer_state(resume_dir)
        start_epoch = int(resume_metadata["epoch"]) + 1
        print(
            f"resuming from {resume_dir} epoch={resume_metadata['epoch']} "
            f"next_epoch={start_epoch} has_opt_state={resumed_opt_state is not None}",
            flush=True,
        )
        if start_epoch > epochs:
            print(
                f"already finished: last epoch {resume_metadata['epoch']} >= --epochs {epochs}",
                flush=True,
            )
            return

    print(f"jax_devices={jax.devices()}", flush=True)
    if not any(d.platform == "gpu" for d in jax.devices()):
        print(
            "WARNING: no JAX GPU device visible; ResNet encode + training will run on CPU "
            "(very slow). Check nvidia-smi / CUDA_VISIBLE_DEVICES.",
            flush=True,
        )

    pairs = CachedPairs(cache_dir)
    action_horizon = int(pairs.manifest["action_horizon"])
    tactile_window = resolve_tactile_window(
        action_horizon=action_horizon,
        window_divisor=tactile_window_divisor,
    )
    conditioner = TactileConditionedBatches(
        pairs,
        tactile_encoder_dir=tactile_encoder_dir,
        tactile_window=tactile_window,
        dataset_repo_id=dataset_repo_id,
        dataset_root=dataset_root,
        history_stride=history_stride,
        build_episode_baselines=(loss_mode == "gated"),
        num_workers=num_workers,
        prefetch_batches=prefetch_batches,
        load_threads=load_threads,
        pipeline_prefetch=pipeline_prefetch,
        image_cache_size=image_cache_size,
        encode_batch_size=encode_batch_size,
    )
    decoder_config = DecoderConfig(
        action_dim=int(pairs.manifest["action_dim"]),
        action_horizon=action_horizon,
        tactile_window=tactile_window,
        gru_hidden_dim=DEFAULT_GRU_HIDDEN_DIM,
        resnet_embedding_dim=conditioner.resnet_embedding_dim,
        model_dim=model_dim,
        depth=depth,
        num_heads=num_heads,
        mlp_ratio=mlp_ratio,
    )
    if resume_metadata is None:
        model = TactileConditionedFlowDecoder(decoder_config, rngs=nnx.Rngs(seed))
    else:
        ckpt_config = DecoderConfig(**resume_metadata["decoder_config"])
        if dataclasses_asdict_mismatch := _config_diff(ckpt_config, decoder_config):
            print(
                "warning: CLI decoder config differs from resume checkpoint; "
                f"keeping checkpoint weights. diffs={dataclasses_asdict_mismatch}",
                flush=True,
            )
        # ``model`` already loaded above.
    train_samples = len(pairs.indices("train"))
    steps_per_epoch = max(1, (train_samples + batch_size - 1) // batch_size)
    warmup_steps = min(warmup_epochs, epochs) * steps_per_epoch
    total_steps = epochs * steps_per_epoch
    peak_learning_rate = resolve_peak_learning_rate(
        learning_rate,
        model_dim=int(model.config.model_dim),
        lr_reference_dim=lr_reference_dim,
    )
    optimizer = make_optimizer(
        model,
        learning_rate=peak_learning_rate,
        weight_decay=weight_decay,
        grad_clip_norm=grad_clip_norm,
        warmup_steps=warmup_steps,
        total_steps=total_steps,
        min_learning_rate_ratio=min_learning_rate_ratio,
        cosine_decay=cosine_decay,
    )
    if resumed_opt_state is not None:
        restore_optimizer_state(optimizer, opt_state=resumed_opt_state, step=resumed_opt_step)
    elif resume_dir is not None:
        print(
            "warning: optimizer state missing in checkpoint; reinitialized Adam state.",
            flush=True,
        )
    if lr_reference_dim is not None:
        print(
            f"learning_rate={learning_rate:g} scaled by sqrt({lr_reference_dim}/{model.config.model_dim}) "
            f"-> peak={peak_learning_rate:g}"
        )
    else:
        print(f"learning_rate peak={peak_learning_rate:g}")
    print(
        f"tactile_window={tactile_window} "
        f"(action_horizon={action_horizon} / divisor={tactile_window_divisor}) "
        f"gru_hidden_dim={DEFAULT_GRU_HIDDEN_DIM} resnet_dim={conditioner.resnet_embedding_dim} "
        f"(frozen ResNet + trainable shared GRU)"
    )
    print(
        f"dataloader=num_workers={num_workers} prefetch_batches={prefetch_batches} "
        f"load_threads={load_threads} pipeline_prefetch={pipeline_prefetch} "
        f"image_cache_size={image_cache_size} encode_batch_size={encode_batch_size} "
        f"eval_every={eval_every} start_epoch={start_epoch} epochs={epochs}"
    )
    if loss_mode == "gt":
        print("loss_mode=gt (target=gt_action)")
    else:
        print(
            f"loss_mode=gated L=w*L*+lambda*(1-w)*L_stop "
            f"tau={gate_tau:g} T={gate_temperature:g} lambda={gate_lambda:g}"
        )
    if cosine_decay:
        print(
            f"lr_schedule=warmup({warmup_steps} steps)+cosine "
            f"min_ratio={min_learning_rate_ratio:g} total_steps={total_steps}"
        )
    elif warmup_steps > 0:
        print(f"lr_schedule=warmup({warmup_steps} steps)+constant total_steps={total_steps}")

    output_dir.mkdir(parents=True, exist_ok=True)
    history_path = output_dir / "history.csv"
    best_mse = float("inf")
    best_path = output_dir / "best" / CHECKPOINT_NAME
    if best_path.exists():
        with best_path.open(encoding="utf-8") as file:
            best_meta = json.load(file)
        best_mse = float(best_meta.get("metrics", {}).get("val_mse", best_mse))
    base_key = jax.random.key(seed)
    history_exists = history_path.exists() and history_path.stat().st_size > 0
    history_mode = "a" if resume_dir is not None and history_exists else "w"

    try:
        with history_path.open(history_mode, newline="", encoding="utf-8") as history_file:
            writer = csv.DictWriter(history_file, fieldnames=history_fields)
            if history_mode == "w":
                writer.writeheader()

            for epoch in range(start_epoch, epochs + 1):
                losses: list[float] = []
                weights: list[int] = []
                tactile_sims: list[float] = []
                tactile_changes: list[float] = []
                gate_ws: list[float] = []
                gate_actives: list[float] = []
                for batch_number, (indices, x_base_np, predicted_np, gt_action_np, tactile_seq) in enumerate(
                    conditioner.batches("train", batch_size=batch_size, shuffle=True, seed=seed + epoch)
                ):
                    step_key = jax.random.fold_in(base_key, epoch * 1_000_000 + batch_number)
                    batch_n = len(x_base_np)
                    if loss_mode == "gated":
                        current_tokens = np.asarray(tactile_seq[:, -1, :, :], dtype=np.float32)
                        change = conditioner.tactile_change_for_cache_indices(indices, current_tokens)
                        gate_w = gate_weights_from_change(
                            change, tau=gate_tau, temperature=gate_temperature
                        )
                        tactile_sims.append(float(np.mean(1.0 - change)))
                        tactile_changes.append(float(np.mean(change)))
                        gate_ws.append(float(np.mean(gate_w)))
                        gate_actives.append(float(np.mean(gate_w > 0.5)))
                    else:
                        gate_w = np.ones((batch_n,), dtype=np.float32)
                    loss = train_step(
                        model,
                        optimizer,
                        jnp.asarray(x_base_np),
                        jnp.asarray(gt_action_np),
                        jnp.asarray(predicted_np),
                        tactile_seq,
                        jnp.asarray(gate_w),
                        step_key,
                        loss_mode=loss_mode,
                        gate_lambda=gate_lambda,
                    )
                    losses.append(float(jax.device_get(loss)))
                    weights.append(batch_n)
                    if batch_number == 0 or (batch_number + 1) % 20 == 0:
                        extra = ""
                        if loss_mode == "gated":
                            extra = (
                                f" tactile_sim={tactile_sims[-1]:.4f}"
                                f" gate_w={gate_ws[-1]:.4f}"
                            )
                        print(
                            f"epoch={epoch}/{epochs} batch={batch_number + 1}/{steps_per_epoch} "
                            f"flow_loss={losses[-1]:.6f}{extra}",
                            flush=True,
                        )
                train_loss = float(np.average(losses, weights=weights))
                train_tactile_metrics: dict[str, float] = {}
                if loss_mode == "gated" and tactile_sims:
                    train_tactile_metrics = {
                        "train_tactile_sim": _weighted_mean(tactile_sims, weights),
                        "train_tactile_change": _weighted_mean(tactile_changes, weights),
                        "train_gate_w": _weighted_mean(gate_ws, weights),
                        "train_gate_active_frac": _weighted_mean(gate_actives, weights),
                    }
                run_eval = (epoch % eval_every == 0) or (epoch == epochs)
                checkpoint_extra = {
                    "cache_records_sha256": pairs.manifest["records_sha256"],
                    "cache_configuration": pairs.manifest["configuration"],
                    "tactile_encoder_dir": str(tactile_encoder_dir.resolve()),
                    "tactile_window_divisor": tactile_window_divisor,
                    "tactile_window": tactile_window,
                    "gru_hidden_dim": DEFAULT_GRU_HIDDEN_DIM,
                    "history_stride": history_stride,
                    "loss_mode": loss_mode,
                    "gate_tau": gate_tau,
                    "gate_temperature": gate_temperature,
                    "gate_lambda": gate_lambda,
                    "eval_every": eval_every,
                }
                if run_eval:
                    validation = evaluate_split(
                        model,
                        conditioner,
                        split="val",
                        batch_size=batch_size,
                        num_steps=validation_steps,
                        keep_predictions=False,
                        gate_tau=gate_tau if loss_mode == "gated" else None,
                        gate_temperature=gate_temperature if loss_mode == "gated" else None,
                    )
                    metrics: dict[str, float] = {
                        "train_flow_loss": train_loss,
                        "val_flow_loss": validation.flow_loss,
                        "val_mse": validation.mse,
                        "val_rmse": validation.rmse,
                        "val_mae": validation.mae,
                        **train_tactile_metrics,
                    }
                    if validation.tactile_sim is not None:
                        metrics.update(
                            {
                                "val_tactile_sim": validation.tactile_sim,
                                "val_tactile_change": float(validation.tactile_change),
                                "val_gate_w": float(validation.gate_w),
                                "val_gate_active_frac": float(validation.gate_active_frac),
                            }
                        )
                    writer.writerow(_blank_history_row(epoch, **metrics))
                    history_file.flush()
                    save_checkpoint(
                        output_dir / "last",
                        model,
                        epoch=epoch,
                        metrics=metrics,
                        extra_metadata=checkpoint_extra,
                        optimizer=optimizer,
                    )
                    if validation.mse < best_mse:
                        best_mse = validation.mse
                        save_checkpoint(
                            output_dir / "best",
                            model,
                            epoch=epoch,
                            metrics=metrics,
                            extra_metadata=checkpoint_extra,
                            optimizer=optimizer,
                        )
                    tactile_msg = ""
                    if train_tactile_metrics:
                        tactile_msg = (
                            f" train_tactile_sim={train_tactile_metrics['train_tactile_sim']:.4f}"
                            f" train_gate_w={train_tactile_metrics['train_gate_w']:.4f}"
                        )
                        if validation.tactile_sim is not None:
                            tactile_msg += (
                                f" val_tactile_sim={validation.tactile_sim:.4f}"
                                f" val_gate_w={validation.gate_w:.4f}"
                            )
                    print(
                        f"epoch={epoch}/{epochs} train_flow_loss={train_loss:.8f} "
                        f"val_flow_loss={validation.flow_loss:.8f} val_mse={validation.mse:.8f} "
                        f"val_rmse={validation.rmse:.8f} val_mae={validation.mae:.8f}"
                        f"{tactile_msg}",
                        flush=True,
                    )
                else:
                    metrics = {"train_flow_loss": train_loss, **train_tactile_metrics}
                    writer.writerow(_blank_history_row(epoch, **metrics))
                    history_file.flush()
                    save_checkpoint(
                        output_dir / "last",
                        model,
                        epoch=epoch,
                        metrics=metrics,
                        extra_metadata=checkpoint_extra,
                        optimizer=optimizer,
                    )
                    tactile_msg = ""
                    if train_tactile_metrics:
                        tactile_msg = (
                            f" train_tactile_sim={train_tactile_metrics['train_tactile_sim']:.4f}"
                            f" train_gate_w={train_tactile_metrics['train_gate_w']:.4f}"
                        )
                    print(
                        f"epoch={epoch}/{epochs} train_flow_loss={train_loss:.8f}"
                        f"{tactile_msg} (skip val)",
                        flush=True,
                    )

        print(f"best_val_mse={best_mse:.8f}")
        print(f"checkpoints={output_dir}")
        if write_plots:
            plot_path = plot_training_history(history_path, output_path=output_dir / "training_curves.png")
            print(f"plot={plot_path}")
    finally:
        conditioner.close()


def _config_diff(left: object, right: object) -> dict[str, tuple[object, object]]:
    import dataclasses

    diffs: dict[str, tuple[object, object]] = {}
    left_dict = dataclasses.asdict(left)  # type: ignore[arg-type]
    right_dict = dataclasses.asdict(right)  # type: ignore[arg-type]
    for key, left_value in left_dict.items():
        right_value = right_dict.get(key)
        if left_value != right_value:
            diffs[key] = (left_value, right_value)
    return diffs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Train tactile GRU + cross-attn flow decoder "
            "(frozen ResNet features; loss-mode gt or gated hybrid)."
        )
    )
    parser.add_argument("--cache-dir", type=pathlib.Path, required=True)
    parser.add_argument("--tactile-encoder-dir", type=pathlib.Path, required=True)
    parser.add_argument("--output-dir", type=pathlib.Path, required=True)
    parser.add_argument(
        "--dataset-repo-id",
        type=str,
        default=None,
        help="Override LeRobot dataset repo id (default: cache manifest configuration).",
    )
    parser.add_argument(
        "--dataset-root",
        type=pathlib.Path,
        default=None,
        help="Optional local dataset root hint (currently unused by image loader; reserved).",
    )
    parser.add_argument(
        "--tactile-window-divisor",
        type=int,
        default=1,
        help="tactile_window = action_horizon // divisor (must divide evenly). Default 1.",
    )
    parser.add_argument(
        "--history-stride",
        type=int,
        default=1,
        help="Frame stride when looking back for the tactile window (default 1 = contiguous).",
    )
    parser.add_argument(
        "--loss-mode",
        choices=("gt", "gated"),
        default="gt",
        help="gt: FM vs GT only. gated: L=w*L*+lambda*(1-w)*L_stop.",
    )
    parser.add_argument(
        "--gate-tau",
        type=float,
        default=0.5,
        help="Soft-gate midpoint tau for w=sigmoid((s-tau)/T). Default 0.5.",
    )
    parser.add_argument(
        "--gate-temperature",
        type=float,
        default=0.1,
        help="Soft-gate temperature T. Default 0.1.",
    )
    parser.add_argument(
        "--gate-lambda",
        type=float,
        default=1.0,
        help="Weight on (1-w)*L_stop in gated mode. Default 1.0.",
    )

    parser.add_argument("--model-dim", type=int, default=256)
    parser.add_argument("--depth", type=int, default=6)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--mlp-ratio", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)

    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--warmup-epochs", type=int, default=10)
    parser.add_argument("--lr-reference-dim", type=int, default=256)
    parser.add_argument("--min-lr-ratio", type=float, default=0.1)
    parser.add_argument("--lr-schedule", choices=("cosine", "constant"), default="cosine")

    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--validation-steps", type=int, default=10)
    parser.add_argument(
        "--eval-every",
        type=int,
        default=5,
        help="Run full validation every N epochs (also always on the final epoch). Default 5.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from output-dir/last (params + optimizer state if present).",
    )
    parser.add_argument(
        "--resume-from",
        type=pathlib.Path,
        help="Resume from an explicit checkpoint directory (overrides --resume).",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=8,
        help="Spawn process workers for video/parquet decode (0/1 = in-process threads only).",
    )
    parser.add_argument(
        "--prefetch-batches",
        type=int,
        default=8,
        help="In-flight mp decode batches queued ahead of the trainer.",
    )
    parser.add_argument(
        "--load-threads",
        type=int,
        default=16,
        help="Per-process threads for unique-frame decode within a batch.",
    )
    parser.add_argument(
        "--pipeline-prefetch",
        type=int,
        default=4,
        help="Decoded image batches buffered while parent runs ResNet/train step.",
    )
    parser.add_argument(
        "--image-cache-size",
        type=int,
        default=8192,
        help="Total LRU decoded-frame budget (split across mp workers).",
    )
    parser.add_argument(
        "--encode-batch-size",
        type=int,
        default=256,
        help="Frozen ResNet microbatch size on the parent process/GPU.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    train_decoder(
        cache_dir=args.cache_dir,
        tactile_encoder_dir=args.tactile_encoder_dir,
        output_dir=args.output_dir,
        dataset_repo_id=args.dataset_repo_id,
        dataset_root=args.dataset_root,
        tactile_window_divisor=args.tactile_window_divisor,
        history_stride=args.history_stride,
        loss_mode=args.loss_mode,
        gate_tau=args.gate_tau,
        gate_temperature=args.gate_temperature,
        gate_lambda=args.gate_lambda,
        model_dim=args.model_dim,
        depth=args.depth,
        num_heads=args.num_heads,
        mlp_ratio=args.mlp_ratio,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        grad_clip_norm=args.grad_clip_norm if args.grad_clip_norm > 0 else None,
        warmup_epochs=args.warmup_epochs,
        lr_reference_dim=args.lr_reference_dim if args.lr_reference_dim > 0 else None,
        min_learning_rate_ratio=args.min_lr_ratio,
        cosine_decay=args.lr_schedule == "cosine",
        batch_size=args.batch_size,
        epochs=args.epochs,
        validation_steps=args.validation_steps,
        eval_every=args.eval_every,
        seed=args.seed,
        write_plots=not args.no_plots,
        num_workers=args.num_workers,
        prefetch_batches=args.prefetch_batches,
        load_threads=args.load_threads,
        pipeline_prefetch=args.pipeline_prefetch,
        image_cache_size=args.image_cache_size,
        encode_batch_size=args.encode_batch_size,
        resume=args.resume,
        resume_from=args.resume_from,
    )


if __name__ == "__main__":
    main()
