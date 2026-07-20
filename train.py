from __future__ import annotations

import argparse
import csv
import pathlib
from collections.abc import Sequence

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from utils.cache import CachedPairs
from utils.checkpoint import save_checkpoint
from utils.metrics import evaluate_split
from utils.model import DecoderConfig
from utils.model import SelfAttentionFlowDecoder
from utils.model import make_optimizer
from utils.model import resolve_peak_learning_rate
from utils.model import train_step
from utils.visualize import plot_training_history


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
    lr_reference_dim: int | None,
    min_learning_rate_ratio: float,
    cosine_decay: bool,
    batch_size: int,
    epochs: int,
    validation_steps: int,
    seed: int,
    write_plots: bool,
) -> None:
    if epochs <= 0 or batch_size <= 0:
        raise ValueError("epochs and batch_size must be positive.")
    if warmup_epochs < 0:
        raise ValueError("warmup_epochs must be non-negative.")
    if not 0.0 <= min_learning_rate_ratio <= 1.0:
        raise ValueError("min_learning_rate_ratio must be in [0, 1].")
    pairs = CachedPairs(cache_dir)
    decoder_config = DecoderConfig(
        action_dim=int(pairs.manifest["action_dim"]),
        action_horizon=int(pairs.manifest["action_horizon"]),
        model_dim=model_dim,
        depth=depth,
        num_heads=num_heads,
        mlp_ratio=mlp_ratio,
    )
    model = SelfAttentionFlowDecoder(decoder_config, rngs=nnx.Rngs(seed))
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

    with history_path.open("w", newline="", encoding="utf-8") as history_file:
        writer = csv.DictWriter(
            history_file,
            fieldnames=["epoch", "train_flow_loss", "val_flow_loss", "val_mse", "val_rmse", "val_mae"],
        )
        writer.writeheader()

        for epoch in range(1, epochs + 1):
            losses: list[float] = []
            weights: list[int] = []
            for batch_number, (_, x_base_np, target_np) in enumerate(
                pairs.batches("train", batch_size=batch_size, shuffle=True, seed=seed + epoch)
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

            train_loss = float(np.average(losses, weights=weights))
            validation = evaluate_split(
                model,
                pairs,
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the unconditional self-attention flow decoder.")
    parser.add_argument("--cache-dir", type=pathlib.Path, required=True)
    parser.add_argument("--output-dir", type=pathlib.Path, required=True)

    # Model parameters
    parser.add_argument("--model-dim", type=int, default=256)
    parser.add_argument("--depth", type=int, default=6)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--mlp-ratio", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)

    # Optimization parameters
    parser.add_argument(
        "--grad-clip-norm",
        type=float,
        default=1.0,
        help="Global gradient norm clip (<=0 disables clipping).",
    )
    parser.add_argument(
        "--warmup-epochs",
        type=int,
        default=10,
        help="Linear LR warmup length in epochs (0 disables).",
    )
    parser.add_argument(
        "--lr-reference-dim",
        type=int,
        default=256,
        help="Scale --learning-rate by sqrt(reference_dim / model_dim); <=0 disables scaling.",
    )
    parser.add_argument(
        "--min-lr-ratio",
        type=float,
        default=0.1,
        help="Final LR as a fraction of peak when cosine decay is enabled.",
    )
    parser.add_argument(
        "--lr-schedule",
        choices=("cosine", "constant"),
        default="cosine",
        help="LR schedule after warmup: cosine decay or constant peak.",
    )

    # Training parameters
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--validation-steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-plots", action="store_true", help="Skip training_curves.png after training.")
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
        lr_reference_dim=args.lr_reference_dim if args.lr_reference_dim > 0 else None,
        min_learning_rate_ratio=args.min_lr_ratio,
        cosine_decay=args.lr_schedule == "cosine",
        batch_size=args.batch_size,
        epochs=args.epochs,
        validation_steps=args.validation_steps,
        seed=args.seed,
        write_plots=not args.no_plots,
    )


if __name__ == "__main__":
    main()
