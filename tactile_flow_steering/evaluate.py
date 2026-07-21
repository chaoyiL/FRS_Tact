from __future__ import annotations

import argparse
import csv
import pathlib
from collections.abc import Sequence

import numpy as np

from tactile_flow_steering.utils.checkpoint import load_checkpoint
from tactile_flow_steering.utils.data import TactileConditionedBatches
from tactile_flow_steering.utils.data import resolve_tactile_window
from tactile_flow_steering.utils.metrics import evaluate_split
from tactile_flow_steering.utils.model import FlowSolver
from tactile_flow_steering.utils.visualize import write_evaluation_plots
from utils.cache import CachedPairs
from utils.cache import atomic_write_json


def evaluate_decoder(
    *,
    cache_dir: pathlib.Path,
    tactile_encoder_dir: pathlib.Path,
    checkpoint_dir: pathlib.Path,
    output_dir: pathlib.Path,
    dataset_repo_id: str | None,
    dataset_root: pathlib.Path | None,
    tactile_window_divisor: int | None,
    history_stride: int | None,
    batch_size: int,
    num_steps: int,
    solver: FlowSolver,
    save_predictions: bool,
    write_plots: bool,
    num_trajectory_samples: int,
    num_episode_strips: int,
    num_workers: int,
    prefetch_batches: int,
    load_threads: int,
    pipeline_prefetch: int,
    image_cache_size: int,
) -> dict[str, float | int | str]:
    pairs = CachedPairs(cache_dir)
    model, checkpoint_metadata = load_checkpoint(checkpoint_dir)
    checkpoint_cache_digest = checkpoint_metadata.get("extra_metadata", {}).get("cache_records_sha256")
    if checkpoint_cache_digest is not None and checkpoint_cache_digest != pairs.manifest["records_sha256"]:
        raise ValueError("Checkpoint was trained from a different cache sample set.")
    expected_shape = (int(pairs.manifest["action_horizon"]), int(pairs.manifest["action_dim"]))
    actual_shape = (model.config.action_horizon, model.config.action_dim)
    if actual_shape != expected_shape:
        raise ValueError(f"Checkpoint/cache action shape mismatch: {actual_shape} != {expected_shape}.")

    extra = checkpoint_metadata.get("extra_metadata") or {}
    if tactile_window_divisor is None:
        tactile_window_divisor = int(extra.get("tactile_window_divisor", 1))
    if history_stride is None:
        history_stride = int(extra.get("history_stride", 1))
    action_horizon = int(pairs.manifest["action_horizon"])
    tactile_window = resolve_tactile_window(
        action_horizon=action_horizon,
        window_divisor=tactile_window_divisor,
    )
    if tactile_window != model.config.tactile_window:
        raise ValueError(
            f"Resolved tactile_window={tactile_window} does not match "
            f"checkpoint tactile_window={model.config.tactile_window}."
        )

    conditioner = TactileConditionedBatches(
        pairs,
        tactile_encoder_dir=tactile_encoder_dir,
        tactile_window=tactile_window,
        dataset_repo_id=dataset_repo_id,
        dataset_root=dataset_root,
        history_stride=history_stride,
        num_workers=num_workers,
        prefetch_batches=prefetch_batches,
        load_threads=load_threads,
        pipeline_prefetch=pipeline_prefetch,
        image_cache_size=image_cache_size,
    )
    try:
        if conditioner.resnet_embedding_dim != model.config.resnet_embedding_dim:
            raise ValueError(
                f"Encoder resnet_embedding_dim={conditioner.resnet_embedding_dim} does not match "
                f"checkpoint resnet_embedding_dim={model.config.resnet_embedding_dim}."
            )

        result = evaluate_split(
            model,
            conditioner,
            split="val",
            batch_size=batch_size,
            num_steps=num_steps,
            solver=solver,
            keep_predictions=save_predictions,
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        metrics: dict[str, float | int | str] = {
            "checkpoint": str(checkpoint_dir.resolve()),
            "checkpoint_epoch": int(checkpoint_metadata["epoch"]),
            "sample_count": len(result.cache_indices),
            "decoder_steps": num_steps,
            "decoder_solver": solver,
            "tactile_window": tactile_window,
            "tactile_window_divisor": tactile_window_divisor,
            "flow_loss": result.flow_loss,
            "mse": result.mse,
            "rmse": result.rmse,
            "mae": result.mae,
            "target": "gt_action",
        }
        atomic_write_json(output_dir / "metrics.json", metrics)

        arrays = pairs.arrays
        with (output_dir / "per_sample.csv").open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(
                file,
                fieldnames=[
                    "cache_index",
                    "dataset_index",
                    "episode_index",
                    "flow_loss",
                    "mse",
                    "rmse",
                    "mae",
                ],
            )
            writer.writeheader()
            for position, cache_index in enumerate(result.cache_indices):
                writer.writerow(
                    {
                        "cache_index": int(cache_index),
                        "dataset_index": int(arrays["dataset_index"][cache_index]),
                        "episode_index": int(arrays["episode_index"][cache_index]),
                        "flow_loss": float(result.sample_flow_loss[position]),
                        "mse": float(result.sample_mse[position]),
                        "rmse": float(result.sample_rmse[position]),
                        "mae": float(result.sample_mae[position]),
                    }
                )
        if result.predictions is not None:
            np.savez(
                output_dir / "predictions.npz",
                cache_indices=result.cache_indices,
                predicted_actions=result.predictions,
            )

        if write_plots:
            plot_paths = write_evaluation_plots(
                output_dir=output_dir,
                result=result,
                pairs=pairs,
                model=model,
                conditioner=conditioner,
                num_steps=num_steps,
                solver=solver,
                num_trajectory_samples=num_trajectory_samples,
                num_episode_strips=num_episode_strips,
            )
            for plot_path in plot_paths:
                print(f"plot={plot_path}")

        print(
            f"validation_samples={len(result.cache_indices)} solver={solver} flow_loss={result.flow_loss:.8f} "
            f"mse={result.mse:.8f} rmse={result.rmse:.8f} mae={result.mae:.8f}"
        )
        print(f"evaluation={output_dir}")
        return metrics
    finally:
        conditioner.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate tactile GRU-conditioned flow decoder against GT actions."
    )
    parser.add_argument("--cache-dir", type=pathlib.Path, required=True)
    parser.add_argument("--tactile-encoder-dir", type=pathlib.Path, required=True)
    parser.add_argument("--checkpoint-dir", type=pathlib.Path, required=True)
    parser.add_argument("--output-dir", type=pathlib.Path, required=True)
    parser.add_argument("--dataset-repo-id", type=str, default=None)
    parser.add_argument("--dataset-root", type=pathlib.Path, default=None)
    parser.add_argument(
        "--tactile-window-divisor",
        type=int,
        default=None,
        help="Override window divisor (default: value stored in checkpoint metadata).",
    )
    parser.add_argument(
        "--history-stride",
        type=int,
        default=None,
        help="Override history stride (default: value stored in checkpoint metadata).",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-steps", type=int, default=5)
    parser.add_argument(
        "--solver",
        "--decoder-solver",
        choices=("euler", "fireflow"),
        default="fireflow",
    )
    parser.add_argument("--save-predictions", action="store_true")
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--num-trajectory-samples", type=int, default=6)
    parser.add_argument("--num-episode-strips", type=int, default=6)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--prefetch-batches", type=int, default=4)
    parser.add_argument("--load-threads", type=int, default=8)
    parser.add_argument("--pipeline-prefetch", type=int, default=2)
    parser.add_argument("--image-cache-size", type=int, default=4096)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    evaluate_decoder(
        cache_dir=args.cache_dir,
        tactile_encoder_dir=args.tactile_encoder_dir,
        checkpoint_dir=args.checkpoint_dir,
        output_dir=args.output_dir,
        dataset_repo_id=args.dataset_repo_id,
        dataset_root=args.dataset_root,
        tactile_window_divisor=args.tactile_window_divisor,
        history_stride=args.history_stride,
        batch_size=args.batch_size,
        num_steps=args.num_steps,
        solver=args.solver,
        save_predictions=args.save_predictions,
        write_plots=not args.no_plots,
        num_trajectory_samples=args.num_trajectory_samples,
        num_episode_strips=args.num_episode_strips,
        num_workers=args.num_workers,
        prefetch_batches=args.prefetch_batches,
        load_threads=args.load_threads,
        pipeline_prefetch=args.pipeline_prefetch,
        image_cache_size=args.image_cache_size,
    )


if __name__ == "__main__":
    main()
