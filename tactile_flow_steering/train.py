from __future__ import annotations

import argparse
import csv
import pathlib
from collections.abc import Sequence

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from tactile_flow_steering.utils.checkpoint import save_checkpoint
from tactile_flow_steering.utils.data import LossMode
from tactile_flow_steering.utils.data import TactileConditionedBatches
from tactile_flow_steering.utils.data import resolve_tactile_window
from tactile_flow_steering.utils.metrics import evaluate_split
from tactile_flow_steering.utils.model import DEFAULT_GRU_HIDDEN_DIM
from tactile_flow_steering.utils.model import DecoderConfig
from tactile_flow_steering.utils.model import TactileConditionedFlowDecoder
from tactile_flow_steering.utils.model import make_optimizer
from tactile_flow_steering.utils.model import resolve_peak_learning_rate
from tactile_flow_steering.utils.model import train_step
from tactile_flow_steering.utils.visualize import plot_training_history
from utils.cache import CachedPairs


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
    seed: int,
    write_plots: bool,
    num_workers: int,
    prefetch_batches: int,
    load_threads: int,
    pipeline_prefetch: int,
    image_cache_size: int,
) -> None:
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
    model = TactileConditionedFlowDecoder(decoder_config, rngs=nnx.Rngs(seed))
    train_samples = len(pairs.indices("train"))
    steps_per_epoch = max(1, (train_samples + batch_size - 1) // batch_size)
    warmup_steps = min(warmup_epochs, epochs) * steps_per_epoch
    total_steps = epochs * steps_per_epoch
    peak_learning_rate = resolve_peak_learning_rate(
        learning_rate,
        model_dim=model_dim,
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
    if lr_reference_dim is not None:
        print(
            f"learning_rate={learning_rate:g} scaled by sqrt({lr_reference_dim}/{model_dim}) "
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
        f"image_cache_size={image_cache_size}"
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
    base_key = jax.random.key(seed)

    try:
        with history_path.open("w", newline="", encoding="utf-8") as history_file:
            writer = csv.DictWriter(
                history_file,
                fieldnames=["epoch", "train_flow_loss", "val_flow_loss", "val_mse", "val_rmse", "val_mae"],
            )
            writer.writeheader()

            for epoch in range(1, epochs + 1):
                losses: list[float] = []
                weights: list[int] = []
                for batch_number, (indices, x_base_np, predicted_np, gt_action_np, tactile_seq) in enumerate(
                    conditioner.batches("train", batch_size=batch_size, shuffle=True, seed=seed + epoch)
                ):
                    step_key = jax.random.fold_in(base_key, epoch * 1_000_000 + batch_number)
                    if loss_mode == "gated":
                        current_tokens = np.asarray(tactile_seq[:, -1, :, :], dtype=np.float32)
                        gate_w = conditioner.gate_weights_for_cache_indices(
                            indices,
                            current_tokens,
                            tau=gate_tau,
                            temperature=gate_temperature,
                        )
                    else:
                        gate_w = np.ones((len(indices),), dtype=np.float32)
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
                    weights.append(len(x_base_np))

                train_loss = float(np.average(losses, weights=weights))
                validation = evaluate_split(
                    model,
                    conditioner,
                    split="val",
                    batch_size=batch_size,
                    num_steps=validation_steps,
                    keep_predictions=False,
                )
                metrics = {
                    "train_flow_loss": train_loss,
                    "val_flow_loss": validation.flow_loss,
                    "val_mse": validation.mse,
                    "val_rmse": validation.rmse,
                    "val_mae": validation.mae,
                }
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
                }
                writer.writerow({"epoch": epoch, **metrics})
                history_file.flush()
                save_checkpoint(
                    output_dir / "last",
                    model,
                    epoch=epoch,
                    metrics=metrics,
                    extra_metadata=checkpoint_extra,
                )
                if validation.mse < best_mse:
                    best_mse = validation.mse
                    save_checkpoint(
                        output_dir / "best",
                        model,
                        epoch=epoch,
                        metrics=metrics,
                        extra_metadata=checkpoint_extra,
                    )
                print(
                    f"epoch={epoch}/{epochs} train_flow_loss={train_loss:.8f} "
                    f"val_flow_loss={validation.flow_loss:.8f} val_mse={validation.mse:.8f} "
                    f"val_rmse={validation.rmse:.8f} val_mae={validation.mae:.8f}"
                )

        print(f"best_val_mse={best_mse:.8f}")
        print(f"checkpoints={output_dir}")
        if write_plots:
            plot_path = plot_training_history(history_path, output_path=output_dir / "training_curves.png")
            print(f"plot={plot_path}")
    finally:
        conditioner.close()


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
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Spawn process workers for video decode (0/1 = in-process threads only).",
    )
    parser.add_argument(
        "--prefetch-batches",
        type=int,
        default=4,
        help="In-flight mp decode batches queued ahead of the trainer.",
    )
    parser.add_argument(
        "--load-threads",
        type=int,
        default=8,
        help="Per-process threads for unique-frame video decode within a batch.",
    )
    parser.add_argument(
        "--pipeline-prefetch",
        type=int,
        default=2,
        help="Decoded image batches buffered while parent runs ResNet/train step.",
    )
    parser.add_argument(
        "--image-cache-size",
        type=int,
        default=4096,
        help="Total LRU decoded-frame budget (split across mp workers).",
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
        seed=args.seed,
        write_plots=not args.no_plots,
        num_workers=args.num_workers,
        prefetch_batches=args.prefetch_batches,
        load_threads=args.load_threads,
        pipeline_prefetch=args.pipeline_prefetch,
        image_cache_size=args.image_cache_size,
    )


if __name__ == "__main__":
    main()
