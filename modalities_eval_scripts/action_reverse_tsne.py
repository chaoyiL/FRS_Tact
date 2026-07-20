from __future__ import annotations

# Imports below the path setup and optional plotting imports are intentional.
# ruff: noqa: E402, PLC0415, SLF001
import argparse
import csv
import os
import pathlib
import sys
from collections.abc import Sequence
from typing import Any

ROOT = pathlib.Path(__file__).resolve().parents[1]
EVAL_DIR = pathlib.Path(__file__).resolve().parent
if str(EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(EVAL_DIR))

import jax
import jax.numpy as jnp
import numpy as np
from utils import (
    EpisodeData,
    EvalObservation,
    SmolVLAEvalModel,
    VelocityContext,
    _stack_observations,
    add_eval_data_arguments,
    create_velocity_context,
    load_episode,
    load_model_from_args,
    predict_velocity_with_context,
)

from lerobot.datasets import LeRobotDataset

_REVERSE_SCAN_CACHE: dict[tuple[int, int, int], Any] = {}


def select_frames(
    frame_count: int,
    *,
    frames: Sequence[int] | None,
    num_frames: int,
    seed: int,
) -> tuple[int, ...]:
    """Select explicit episode-relative frames or sample them without replacement."""

    if frame_count <= 0:
        raise ValueError("The selected episode has no frames.")

    if frames is not None:
        selected = tuple(int(frame) for frame in frames)
        if not selected:
            raise ValueError("--frames cannot be empty.")
        if len(set(selected)) != len(selected):
            raise ValueError(f"--frames contains duplicates: {selected}")
        for frame in selected:
            if frame < 0 or frame >= frame_count:
                raise ValueError(
                    f"Frame {frame} is out of range; available episode-relative frames are 0..{frame_count - 1}."
                )
        return selected

    if num_frames <= 0:
        raise ValueError(f"--num-frames must be positive, got {num_frames}.")
    if num_frames > frame_count:
        raise ValueError(
            f"Cannot sample {num_frames} frames without replacement; the episode contains {frame_count} frames."
        )
    rng = np.random.default_rng(seed)
    return tuple(sorted(int(frame) for frame in rng.choice(frame_count, size=num_frames, replace=False)))


def load_selected_episode(
    model: SmolVLAEvalModel,
    episode_index: int,
    *,
    frames: Sequence[int] | None,
    num_frames: int,
    seed: int,
    max_frames: int | None,
) -> EpisodeData:
    """Create the data pipeline once and materialize only the selected frames."""

    dataset = LeRobotDataset(
        model.dataset_repo_id,
        root=model.dataset_root,
        revision=model.dataset_revision,
        episodes=[episode_index],
    )
    frame_count = len(dataset)
    if max_frames is not None:
        if max_frames <= 0:
            raise ValueError(f"--max-frames must be positive, got {max_frames}.")
        frame_count = min(frame_count, max_frames)

    selected_frames = select_frames(
        frame_count,
        frames=frames,
        num_frames=num_frames,
        seed=seed,
    )
    return load_episode(
        model,
        episode_index,
        frame_indices=selected_frames,
    )


def stack_observations(observations: Sequence[EvalObservation]) -> EvalObservation:
    if not observations:
        raise ValueError("Cannot stack an empty observation sequence.")
    return _stack_observations(*observations)


def _get_reverse_scan(model: SmolVLAEvalModel, *, batch_size: int, num_steps: int):
    cache_key = (id(model), batch_size, num_steps)
    if cache_key in _REVERSE_SCAN_CACHE:
        return _REVERSE_SCAN_CACHE[cache_key]

    dt = jnp.asarray(1.0 / num_steps, dtype=jnp.float32)
    step_indices = jnp.arange(num_steps, dtype=jnp.int32)

    @jax.jit
    def run_scan(context: VelocityContext, x: jax.Array) -> jax.Array:
        def scan_body(carry: tuple[jax.Array, jax.Array], _: jax.Array):
            x_t, t = carry
            velocity = predict_velocity_with_context(model, context, x_t, t).astype(jnp.float32)
            return (x_t + dt * velocity, t + dt), None

        t0 = jnp.zeros((batch_size,), dtype=jnp.float32)
        (x_base, _), _ = jax.lax.scan(scan_body, (x, t0), step_indices)
        return x_base

    _REVERSE_SCAN_CACHE[cache_key] = run_scan
    return run_scan


def reverse_integrate_actions(
    model: SmolVLAEvalModel,
    observations: Sequence[EvalObservation],
    actions: np.ndarray,
    *,
    num_steps: int,
    batch_size: int,
) -> np.ndarray:
    """Euler-integrate model-space actions from data time t=0 to base time t=1."""

    if num_steps <= 0:
        raise ValueError(f"--num-steps must be positive, got {num_steps}.")
    if batch_size <= 0:
        raise ValueError(f"--batch-size must be positive, got {batch_size}.")
    if len(observations) != len(actions):
        raise ValueError(f"Observation/action count mismatch: {len(observations)} != {len(actions)}")

    integrated_batches: list[np.ndarray] = []
    for start in range(0, len(observations), batch_size):
        stop = min(start + batch_size, len(observations))
        observation_batch = stack_observations(observations[start:stop])
        action_batch = jnp.asarray(actions[start:stop], dtype=jnp.float32)
        context = create_velocity_context(model, observation_batch)
        run_scan = _get_reverse_scan(model, batch_size=stop - start, num_steps=num_steps)
        x_base = run_scan(context, action_batch)
        integrated_batches.append(np.asarray(jax.device_get(x_base), dtype=np.float32))
        print(f"reverse_integration_batch={start}:{stop}/{len(observations)}")

    return np.concatenate(integrated_batches, axis=0)


def paired_tsne(
    action_truth: np.ndarray,
    x_base: np.ndarray,
    *,
    perplexity: float | None,
    seed: int,
    standardize: bool,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Fit one t-SNE embedding jointly, preserving truth/base point pairing by index."""

    try:
        from sklearn.manifold import TSNE
    except ImportError as exc:
        raise ImportError(
            "scikit-learn is required for t-SNE. Install the project's dev dependencies."
        ) from exc

    if action_truth.shape != x_base.shape:
        raise ValueError(f"Action/base shapes must match, got {action_truth.shape} and {x_base.shape}.")
    if action_truth.ndim < 2:
        raise ValueError(f"Expected batched action vectors, got shape {action_truth.shape}.")

    num_pairs = action_truth.shape[0]
    features = np.concatenate(
        [action_truth.reshape(num_pairs, -1), x_base.reshape(num_pairs, -1)],
        axis=0,
    ).astype(np.float64, copy=False)
    if not np.all(np.isfinite(features)):
        raise ValueError("Non-finite values found in action truth or reverse-integrated vectors.")

    if standardize:
        mean = features.mean(axis=0, keepdims=True)
        std = features.std(axis=0, keepdims=True)
        features = (features - mean) / np.where(std > 1e-12, std, 1.0)

    num_points = features.shape[0]
    if perplexity is None:
        # A conservative small-sample default; sklearn requires perplexity < n_samples.
        perplexity = min(30.0, max(1.0, (num_points - 1) / 3.0))
    if not 0.0 < perplexity < num_points:
        raise ValueError(f"t-SNE perplexity must satisfy 0 < perplexity < {num_points}, got {perplexity}.")

    embedding = TSNE(
        n_components=2,
        perplexity=perplexity,
        init="random",
        learning_rate="auto",
        random_state=seed,
    ).fit_transform(features)
    return embedding[:num_pairs], embedding[num_pairs:], float(perplexity)


def save_outputs(
    *,
    output_dir: pathlib.Path,
    episode_index: int,
    frames: Sequence[int],
    dataset_indices: Sequence[int],
    action_truth: np.ndarray,
    raw_action_truth: np.ndarray | None,
    x_base: np.ndarray,
    truth_xy: np.ndarray,
    base_xy: np.ndarray,
    perplexity: float,
    num_steps: int,
    seed: int,
    tsne_seed: int,
    standardize: bool,
    annotate: bool,
) -> tuple[pathlib.Path, pathlib.Path, pathlib.Path]:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    os.environ.setdefault("XDG_CACHE_HOME", "/tmp")
    try:
        import matplotlib as mpl

        mpl.use("Agg")
        from matplotlib import pyplot as plt
    except ImportError as exc:
        raise ImportError(
            "matplotlib is required to save the t-SNE plot. Install the project's viz dependencies."
        ) from exc

    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"episode_{episode_index}_action_reverse_tsne"
    npz_path = output_dir / f"{prefix}.npz"
    csv_path = output_dir / f"{prefix}.csv"
    plot_path = output_dir / f"{prefix}.png"

    arrays: dict[str, np.ndarray] = {
        "frames": np.asarray(frames, dtype=np.int64),
        "dataset_indices": np.asarray(dataset_indices, dtype=np.int64),
        "action_truth_model_space": np.asarray(action_truth, dtype=np.float32),
        "reverse_integrated_x_base": np.asarray(x_base, dtype=np.float32),
        "action_truth_tsne": np.asarray(truth_xy, dtype=np.float32),
        "reverse_integrated_tsne": np.asarray(base_xy, dtype=np.float32),
        "perplexity": np.asarray(perplexity, dtype=np.float32),
        "num_steps": np.asarray(num_steps, dtype=np.int64),
        "frame_sampling_seed": np.asarray(seed, dtype=np.int64),
        "tsne_seed": np.asarray(tsne_seed, dtype=np.int64),
        "standardized_before_tsne": np.asarray(standardize, dtype=np.bool_),
    }
    if raw_action_truth is not None:
        arrays["action_truth_raw_dataset_space"] = np.asarray(raw_action_truth, dtype=np.float32)
    np.savez_compressed(npz_path, **arrays)

    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "frame",
                "dataset_index",
                "action_truth_x",
                "action_truth_y",
                "reverse_integrated_x",
                "reverse_integrated_y",
                "pair_distance_2d",
            ],
        )
        writer.writeheader()
        for frame, dataset_index, action_point, base_point in zip(
            frames, dataset_indices, truth_xy, base_xy, strict=True
        ):
            writer.writerow(
                {
                    "frame": int(frame),
                    "dataset_index": int(dataset_index),
                    "action_truth_x": float(action_point[0]),
                    "action_truth_y": float(action_point[1]),
                    "reverse_integrated_x": float(base_point[0]),
                    "reverse_integrated_y": float(base_point[1]),
                    "pair_distance_2d": float(np.linalg.norm(action_point - base_point)),
                }
            )

    fig, ax = plt.subplots(figsize=(10, 8))
    for action_point, base_point in zip(truth_xy, base_xy, strict=True):
        ax.plot(
            [action_point[0], base_point[0]],
            [action_point[1], base_point[1]],
            color="0.35",
            linewidth=0.9,
            alpha=0.75,
            solid_capstyle="round",
            zorder=2,
        )
    ax.scatter(
        truth_xy[:, 0],
        truth_xy[:, 1],
        marker="o",
        facecolors="none",
        edgecolors="#1f77b4",
        linewidths=1.7,
        s=78,
        label="Action truth (model space)",
        zorder=3,
    )
    ax.scatter(
        base_xy[:, 0],
        base_xy[:, 1],
        marker="x",
        c="#ff7f0e",
        linewidths=1.8,
        s=62,
        label="Reverse-integrated x_base",
        zorder=4,
    )
    if annotate:
        for frame, action_point, base_point in zip(frames, truth_xy, base_xy, strict=True):
            ax.annotate(
                str(frame),
                action_point,
                xytext=(4, 4),
                textcoords="offset points",
                fontsize=8,
                color="#1f77b4",
            )
            ax.annotate(
                str(frame),
                base_point,
                xytext=(4, -11),
                textcoords="offset points",
                fontsize=8,
                color="#ff7f0e",
            )

    ax.set_title(f"Episode {episode_index}: action truth vs. reverse integration (t-SNE)")
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.grid(visible=True, alpha=0.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(plot_path, dpi=180)
    plt.close(fig)
    return npz_path, csv_path, plot_path


def _extract_raw_actions(
    raw_samples: Sequence[dict[str, Any]],
    expected_shape: tuple[int, ...],
    *,
    action_key: str,
) -> np.ndarray | None:
    raw_actions = []
    for sample in raw_samples:
        if action_key not in sample:
            return None
        action = np.asarray(sample[action_key], dtype=np.float32)
        if action.shape != expected_shape:
            return None
        raw_actions.append(action)
    return np.stack(raw_actions, axis=0)


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Sample frames from one episode, reverse-integrate their ground-truth actions to x_base, "
            "and jointly visualize both sets with paired t-SNE points."
        )
    )
    add_eval_data_arguments(parser)
    parser.add_argument("--episode-index", type=int, default=0)
    parser.add_argument(
        "--frames",
        nargs="+",
        type=int,
        help="Explicit episode-relative frame indices. Overrides --num-frames.",
    )
    parser.add_argument("--num-frames", type=int, default=100, help="Number of random frames to sample.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed used for frame sampling.")
    parser.add_argument(
        "--max-frames",
        type=int,
        help="Optionally restrict selection to the first N episode frames.",
    )
    parser.add_argument("--num-steps", "-k", type=int, default=120, help="Euler steps from t=0 to t=1.")
    parser.add_argument(
        "--batch-size", type=int, default=4, help="Frames per reverse-integration model batch."
    )
    parser.add_argument(
        "--perplexity",
        type=float,
        help="t-SNE perplexity. By default it is selected from the number of plotted points.",
    )
    parser.add_argument("--tsne-seed", type=int, help="t-SNE random seed; defaults to --seed.")
    parser.add_argument(
        "--standardize",
        action="store_true",
        help="Standardize every flattened action coordinate jointly before t-SNE.",
    )
    parser.add_argument(
        "--annotate", action="store_true", help="Label both endpoints with episode frame indices."
    )
    parser.add_argument(
        "--output-dir",
        type=pathlib.Path,
        default=pathlib.Path("eval_outputs/action_reverse_tsne"),
    )
    args = parser.parse_args(argv)

    model = load_model_from_args(args)
    episode = load_selected_episode(
        model,
        args.episode_index,
        frames=args.frames,
        num_frames=args.num_frames,
        seed=args.seed,
        max_frames=args.max_frames,
    )
    action_truth = np.stack(episode.actions, axis=0).astype(np.float32)

    print(f"episode={args.episode_index}")
    print(f"selected_frames={episode.frames}")
    print(f"dataset_indices={episode.indices}")
    print(f"action_shape={action_truth.shape}")
    print(f"num_steps={args.num_steps}")

    x_base = reverse_integrate_actions(
        model,
        episode.observations,
        action_truth,
        num_steps=args.num_steps,
        batch_size=args.batch_size,
    )

    tsne_seed = args.seed if args.tsne_seed is None else args.tsne_seed
    truth_xy, base_xy, perplexity = paired_tsne(
        action_truth,
        x_base,
        perplexity=args.perplexity,
        seed=tsne_seed,
        standardize=args.standardize,
    )
    raw_action_truth = _extract_raw_actions(
        episode.raw_samples,
        action_truth.shape[1:],
        action_key=model.action_key,
    )
    npz_path, csv_path, plot_path = save_outputs(
        output_dir=args.output_dir,
        episode_index=args.episode_index,
        frames=episode.frames,
        dataset_indices=episode.indices,
        action_truth=action_truth,
        raw_action_truth=raw_action_truth,
        x_base=x_base,
        truth_xy=truth_xy,
        base_xy=base_xy,
        perplexity=perplexity,
        num_steps=args.num_steps,
        seed=args.seed,
        tsne_seed=tsne_seed,
        standardize=args.standardize,
        annotate=args.annotate,
    )
    print(f"tsne_perplexity={perplexity}")
    print(f"arrays={npz_path}")
    print(f"coordinates={csv_path}")
    print(f"plot={plot_path}")


if __name__ == "__main__":
    main()
