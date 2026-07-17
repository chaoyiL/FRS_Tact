from __future__ import annotations

import argparse
import csv
import os
import pathlib
import sys
from collections.abc import Sequence
from dataclasses import dataclass

import jax
import jax.numpy as jnp

EVAL_DIR = pathlib.Path(__file__).resolve().parent
if str(EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(EVAL_DIR))

from utils import (  # noqa: E402
    EvalObservation,
    SmolVLAEvalModel,
    _batch_actions,
    _batch_observation,
    _scalar,
    ablate_modality_observation,
    add_eval_data_arguments,
    load_episode,
    load_model_from_args,
)


@dataclass(frozen=True)
class ActionErrorResult:
    actions: jax.Array
    mse: jax.Array
    rmse: jax.Array
    mae: jax.Array


def _prediction_error(predicted_actions: jax.Array, reference_actions: jax.Array) -> ActionErrorResult:
    predicted_actions = jnp.asarray(predicted_actions, dtype=jnp.float32)
    reference_actions = jnp.asarray(reference_actions, dtype=jnp.float32)
    difference = predicted_actions - reference_actions
    event_axes = tuple(range(1, difference.ndim))
    mse = jnp.mean(jnp.square(difference), axis=event_axes)
    mae = jnp.mean(jnp.abs(difference), axis=event_axes)
    return ActionErrorResult(
        actions=predicted_actions,
        mse=mse,
        rmse=jnp.sqrt(mse),
        mae=mae,
    )


def evaluate_modality_error_change(
    model: SmolVLAEvalModel,
    observation: EvalObservation,
    reference_actions: jax.Array,
    *,
    modality: str,
    num_steps: int,
    rng: jax.Array,
) -> tuple[ActionErrorResult, ActionErrorResult, jax.Array]:
    original = _batch_observation(observation)
    ablated = _batch_observation(ablate_modality_observation(observation, modality=modality))
    reference_actions = _batch_actions(reference_actions).astype(jnp.float32)
    noise = jax.random.normal(
        rng,
        (1, model.config.chunk_size, model.config.max_action_dim),
        dtype=jnp.float32,
    )
    original_actions = model.sample_actions(
        rng,
        original,
        num_steps=num_steps,
        noise=noise,
    )
    ablated_actions = model.sample_actions(
        rng,
        ablated,
        num_steps=num_steps,
        noise=noise,
    )
    original_error = _prediction_error(original_actions, reference_actions)
    ablated_error = _prediction_error(ablated_actions, reference_actions)
    return original_error, ablated_error, ablated_error.mse - original_error.mse


def save_error_curve(
    rows: Sequence[dict[str, float | int]],
    *,
    output_dir: pathlib.Path,
    modality: str,
    episode_index: str,
) -> tuple[pathlib.Path, pathlib.Path | None]:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    os.environ.setdefault("XDG_CACHE_HOME", "/tmp")
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"{modality}_action_error_episode_{episode_index}.csv"
    fields = (
        "frame",
        "dataset_index",
        "original_mse",
        "ablated_mse",
        "delta_mse",
        "relative_delta_mse",
        "original_rmse",
        "ablated_rmse",
        "original_mae",
        "ablated_mae",
    )
    with csv_path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    try:
        import matplotlib

        matplotlib.use("Agg")
        from matplotlib import pyplot as plt
    except ImportError:
        return csv_path, None

    plot_path = output_dir / f"{modality}_action_error_episode_{episode_index}.png"
    frames = [row["frame"] for row in rows]
    figure, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    axes[0].plot(frames, [row["original_mse"] for row in rows], marker="o", label="original")
    axes[0].plot(frames, [row["ablated_mse"] for row in rows], marker="o", label="ablated")
    axes[0].set_ylabel("action MSE")
    axes[0].legend()
    axes[0].grid(visible=True, alpha=0.3)
    axes[1].plot(frames, [row["delta_mse"] for row in rows], marker="o")
    axes[1].axhline(0.0, color="black", linewidth=1.0, alpha=0.5)
    axes[1].set_xlabel("episode frame")
    axes[1].set_ylabel("delta MSE")
    axes[1].grid(visible=True, alpha=0.3)
    figure.suptitle(f"{modality} action error change over episode {episode_index}")
    figure.tight_layout()
    figure.savefig(plot_path, dpi=160)
    plt.close(figure)
    return csv_path, plot_path


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Evaluate SmolVLA modality contribution via action error")
    add_eval_data_arguments(parser)
    parser.add_argument("--episode-index", type=int, default=0)
    parser.add_argument("--frame", type=int, default=0)
    parser.add_argument("--sample-interval", type=int)
    parser.add_argument("--num-steps", "-k", type=int, default=10)
    parser.add_argument(
        "--remove-modality",
        choices=("vision", "tactile", "state", "language_prompt"),
        default="vision",
    )
    parser.add_argument("--max-frames", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", type=pathlib.Path, default=pathlib.Path("eval_outputs/action_error"))
    args = parser.parse_args(argv)

    model = load_model_from_args(args)
    episode = load_episode(
        model,
        args.episode_index,
        start_frame=args.frame,
        sample_interval=args.sample_interval,
        max_frames=args.max_frames,
        frame_indices=(args.frame,) if args.sample_interval is None else None,
    )
    print(f"loaded episode={args.episode_index} frames={len(episode.frames)} indices={episode.indices[:5]}")
    print(f"prompt={episode.prompts[0]!r}")
    print(f"ablated_modality={args.remove_modality} num_steps={args.num_steps}")

    rows = []
    base_rng = jax.random.key(args.seed)
    for frame, dataset_index, observation, reference_actions in zip(
        episode.frames,
        episode.indices,
        episode.observations,
        episode.actions,
        strict=True,
    ):
        rng = jax.random.fold_in(base_rng, dataset_index)
        original, ablated, delta = evaluate_modality_error_change(
            model,
            observation,
            reference_actions,
            modality=args.remove_modality,
            num_steps=args.num_steps,
            rng=rng,
        )
        original_mse = _scalar(original.mse)
        delta_mse = _scalar(delta)
        row = {
            "frame": frame,
            "dataset_index": dataset_index,
            "original_mse": original_mse,
            "ablated_mse": _scalar(ablated.mse),
            "delta_mse": delta_mse,
            "relative_delta_mse": delta_mse / original_mse if original_mse else float("nan"),
            "original_rmse": _scalar(original.rmse),
            "ablated_rmse": _scalar(ablated.rmse),
            "original_mae": _scalar(original.mae),
            "ablated_mae": _scalar(ablated.mae),
        }
        rows.append(row)
        print(
            f"frame={frame} original_mse={row['original_mse']:.6f} "
            f"ablated_mse={row['ablated_mse']:.6f} delta_mse={row['delta_mse']:.6f}"
        )

    csv_path, plot_path = save_error_curve(
        rows,
        output_dir=args.output_dir,
        modality=args.remove_modality,
        episode_index=str(args.episode_index),
    )
    print(f"curve_csv={csv_path}")
    if plot_path is not None:
        print(f"curve_plot={plot_path}")


if __name__ == "__main__":
    main()
