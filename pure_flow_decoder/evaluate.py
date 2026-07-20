from __future__ import annotations

import argparse
import csv
import pathlib
from collections.abc import Sequence

import numpy as np

from pure_flow_decoder.utils.cache import CachedPairs
from pure_flow_decoder.utils.cache import atomic_write_json
from pure_flow_decoder.utils.checkpoint import load_checkpoint
from pure_flow_decoder.utils.metrics import evaluate_split
from pure_flow_decoder.utils.model import FlowSolver
from pure_flow_decoder.utils.visualize import write_evaluation_plots


def evaluate_decoder(
    *,
    cache_dir: pathlib.Path,
    checkpoint_dir: pathlib.Path,
    output_dir: pathlib.Path,
    batch_size: int,
    num_steps: int,
    solver: FlowSolver,
    save_predictions: bool,
    write_plots: bool,
    num_trajectory_samples: int,
    num_episode_strips: int,
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

    result = evaluate_split(
        model,
        pairs,
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
        "flow_loss": result.flow_loss,
        "mse": result.mse,
        "rmse": result.rmse,
        "mae": result.mae,
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate a flow decoder on held-out episodes.")
    parser.add_argument("--cache-dir", type=pathlib.Path, required=True)
    parser.add_argument("--checkpoint-dir", type=pathlib.Path, required=True)
    parser.add_argument("--output-dir", type=pathlib.Path, required=True)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-steps", type=int, default=5)
    parser.add_argument(
        "--solver",
        "--decoder-solver",
        choices=("euler", "fireflow"),
        default="fireflow",
        help="Numerical integrator for decoder flow matching denoising.",
    )
    parser.add_argument("--save-predictions", action="store_true")
    parser.add_argument("--no-plots", action="store_true", help="Skip PNG visualization outputs.")
    parser.add_argument(
        "--num-trajectory-samples",
        type=int,
        default=6,
        help="Number of validation samples to plot in action_trajectories.png (0 to disable).",
    )
    parser.add_argument(
        "--num-episode-strips",
        type=int,
        default=6,
        help="Number of validation episodes to plot in episode_action_strips.png (0 to disable).",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    evaluate_decoder(
        cache_dir=args.cache_dir,
        checkpoint_dir=args.checkpoint_dir,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        num_steps=args.num_steps,
        solver=args.solver,
        save_predictions=args.save_predictions,
        write_plots=not args.no_plots,
        num_trajectory_samples=args.num_trajectory_samples,
        num_episode_strips=args.num_episode_strips,
    )


if __name__ == "__main__":
    main()
