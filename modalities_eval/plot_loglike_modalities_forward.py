from __future__ import annotations

# Shared-noise forward (noise→data) log-likelihood modality ablations.
# Estimates log p(a_gen | o) along a generative path from a fixed Gaussian seed,
# not log p(a_GT | o) from the reverse data→base path in plot_loglike_modalities.py.
# ruff: noqa: E402
import argparse
import csv
import os
import pathlib
import sys
from collections.abc import Sequence

import jax

ROOT = pathlib.Path(__file__).resolve().parents[1]
EVAL_SCRIPTS = pathlib.Path(__file__).resolve().parent
for path in (EVAL_SCRIPTS,):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from loglike_evaluate import (
    DEFAULT_HUTCHINSON_SAMPLES,
    DEFAULT_HUTCHINSON_SEED,
    DEFAULT_NOISE_SEED,
    ODE_SOLVER_FIREFLOW,
    ODE_SOLVERS,
    compute_episode_modality_contributions_from_noise,
    load_episode,
    save_contribution_curve,
)
from plot_loglike_config import DEFAULT_CONFIG, add_config_argument, parse_args_with_config
from plot_loglike_modalities import (
    MODALITIES,
    _default_output_path,
    _infer_modality,
    plot_modalities,
)
from utils import SmolVLAEvalModel, add_eval_data_arguments, load_model_from_args

DEFAULT_CHECKPOINT_DIR = pathlib.Path("/home/typhon/models/tactile_test_05_1.5w")
DEFAULT_DATASET_REPO_ID = "chaoyi/tactile_test_03"
DEFAULT_MODALITIES = ("vision", "state", "language_prompt")
DEFAULT_OUTPUT_DIR = pathlib.Path("eval_outputs/loglike_forward")


def evaluate_modalities_from_noise(
    *,
    model: SmolVLAEvalModel,
    episode_index: int | str,
    frame: int,
    max_frames: int,
    sample_interval: int | None,
    num_steps: int,
    ode_solver: str,
    eval_batch_size: int,
    hutchinson_samples: int,
    hutchinson_seed: int,
    noise_seed: int,
    modalities: Sequence[str],
    output_dir: pathlib.Path,
) -> list[pathlib.Path]:
    """Run shared-noise forward log-likelihood ablation for each modality."""

    if hutchinson_samples <= 0:
        raise ValueError(f"--hutchinson-samples must be positive, got {hutchinson_samples}.")
    if eval_batch_size <= 0:
        raise ValueError(f"--eval-batch-size must be positive, got {eval_batch_size}.")
    if sample_interval is not None and sample_interval <= 0:
        raise ValueError(f"--sample-interval must be positive, got {sample_interval}.")

    if sample_interval is None:
        episode = load_episode(
            model,
            episode_index,
            max_frames=max_frames,
            frame_indices=(frame,),
        )
    else:
        episode = load_episode(
            model,
            episode_index,
            start_frame=frame,
            sample_interval=sample_interval,
            max_frames=max_frames,
        )
    print(
        f"loaded episode={episode_index} frames={len(episode.indices)} dataset_indices={episode.indices[:5]}"
    )
    print(f"prompt={episode.prompts[0]!r}")
    print("ablation_method=input_mask_or_zero")
    print("likelihood_direction=from_noise")
    print("divergence_method=hutchinson_rademacher_jvp")
    print(f"hutchinson_samples={hutchinson_samples}")
    print(f"hutchinson_seed={hutchinson_seed}")
    print(f"noise_seed={noise_seed}")
    print(f"eval_batch_size={eval_batch_size}")
    print(f"ode_solver={ode_solver}")
    print(f"model_dtype={jax.tree.leaves(model.params)[0].dtype}")

    csv_paths: list[pathlib.Path] = []
    for modality in modalities:
        print(f"ablated_modality={modality}")
        rows = compute_episode_modality_contributions_from_noise(
            model,
            episode.frames,
            episode.indices,
            episode.observations,
            episode.prompts,
            modality=modality,
            num_steps=num_steps,
            prompt_tokenizer=None,
            state_in_prompt=False,
            hutchinson_samples=hutchinson_samples,
            hutchinson_seed=hutchinson_seed,
            noise_seed=noise_seed,
            ode_solver=ode_solver,
            eval_batch_size=eval_batch_size,
        )
        csv_path, component_plot_path = save_contribution_curve(
            rows,
            output_dir=output_dir,
            modality=modality,
            episode_index=str(episode_index),
        )
        csv_paths.append(csv_path)
        print(f"curve_csv={csv_path}")
        if component_plot_path is not None:
            print(f"curve_plot={component_plot_path}")

    return csv_paths


def _read_contribution_rows(csv_path: pathlib.Path) -> dict[int, dict[str, float]]:
    rows: dict[int, dict[str, float]] = {}
    with csv_path.open(newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"{csv_path} is empty.")
        required = {
            "frame",
            "contribution",
            "original_log_likelihood",
            "ablated_log_likelihood",
        }
        missing = required.difference(reader.fieldnames)
        if missing:
            raise ValueError(f"{csv_path} is missing required column(s): {', '.join(sorted(missing))}")
        for row in reader:
            frame = int(row["frame"])
            rows[frame] = {
                "contribution": float(row["contribution"]),
                "original_log_likelihood": float(row["original_log_likelihood"]),
                "ablated_log_likelihood": float(row["ablated_log_likelihood"]),
            }
    if not rows:
        raise ValueError(f"{csv_path} has no data rows.")
    return rows


def compare_forward_to_reverse(
    forward_csv_paths: Sequence[pathlib.Path],
    *,
    reverse_dir: pathlib.Path,
    episode_index: int | str,
    output_dir: pathlib.Path,
) -> list[pathlib.Path]:
    """Write per-modality forward-vs-reverse contribution delta CSV/PNG."""

    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    os.environ.setdefault("XDG_CACHE_HOME", "/tmp")
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        import matplotlib

        matplotlib.use("Agg")
        from matplotlib import pyplot as plt
    except ImportError:
        plt = None

    written: list[pathlib.Path] = []
    for forward_csv in forward_csv_paths:
        modality = _infer_modality(forward_csv)
        reverse_csv = reverse_dir / f"{modality}_contribution_episode_{episode_index}.csv"
        if not reverse_csv.is_file():
            raise FileNotFoundError(
                f"Missing reverse CSV for modality={modality!r}: {reverse_csv}"
            )

        forward_rows = _read_contribution_rows(forward_csv)
        reverse_rows = _read_contribution_rows(reverse_csv)
        shared_frames = sorted(set(forward_rows).intersection(reverse_rows))
        if not shared_frames:
            raise ValueError(
                f"No overlapping frames between {forward_csv} and {reverse_csv}."
            )

        compare_rows = []
        for frame in shared_frames:
            fwd = forward_rows[frame]
            rev = reverse_rows[frame]
            compare_rows.append(
                {
                    "frame": frame,
                    "contribution_forward": fwd["contribution"],
                    "contribution_reverse": rev["contribution"],
                    "delta_contribution": fwd["contribution"] - rev["contribution"],
                    "delta_original_log_likelihood": (
                        fwd["original_log_likelihood"] - rev["original_log_likelihood"]
                    ),
                    "delta_ablated_log_likelihood": (
                        fwd["ablated_log_likelihood"] - rev["ablated_log_likelihood"]
                    ),
                }
            )

        csv_path = output_dir / f"{modality}_forward_vs_reverse_episode_{episode_index}.csv"
        with csv_path.open("w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "frame",
                    "contribution_forward",
                    "contribution_reverse",
                    "delta_contribution",
                    "delta_original_log_likelihood",
                    "delta_ablated_log_likelihood",
                ],
            )
            writer.writeheader()
            writer.writerows(compare_rows)
        written.append(csv_path)
        print(f"compare_csv={csv_path}")

        if plt is None:
            continue

        plot_path = output_dir / f"{modality}_forward_vs_reverse_episode_{episode_index}.png"
        frames = [row["frame"] for row in compare_rows]
        fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
        fig.suptitle(f"{modality}: forward vs reverse contribution (episode {episode_index})")
        axes[0].plot(frames, [row["contribution_forward"] for row in compare_rows], marker="o", label="forward")
        axes[0].plot(frames, [row["contribution_reverse"] for row in compare_rows], marker="o", label="reverse")
        axes[0].set_ylabel("contribution")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        axes[1].plot(frames, [row["delta_contribution"] for row in compare_rows], marker="o", color="C3")
        axes[1].set_ylabel("fwd - rev")
        axes[1].grid(True, alpha=0.3)
        axes[2].plot(
            frames,
            [row["delta_original_log_likelihood"] for row in compare_rows],
            marker="o",
            label="delta original logp",
        )
        axes[2].plot(
            frames,
            [row["delta_ablated_log_likelihood"] for row in compare_rows],
            marker="o",
            label="delta ablated logp",
        )
        axes[2].set_ylabel("logp deltas")
        axes[2].set_xlabel("Episode frame")
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(plot_path, dpi=160)
        plt.close(fig)
        written.append(plot_path)
        print(f"compare_plot={plot_path}")

    return written


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run shared-noise forward log-likelihood modality ablations and plot "
            "contribution curves. Optional comparison against reverse (data→base) CSVs. "
            f"Defaults load from --config (default: {DEFAULT_CONFIG})."
        )
    )
    add_config_argument(parser)
    add_eval_data_arguments(parser, required=False)
    parser.set_defaults(
        checkpoint_dir=DEFAULT_CHECKPOINT_DIR,
        dataset_repo_id=DEFAULT_DATASET_REPO_ID,
    )
    parser.add_argument("--episode-index", type=int, default=0)
    parser.add_argument("--frame", type=int, default=0)
    parser.add_argument("--max-frames", type=int, default=1000)
    parser.add_argument(
        "--sample-interval",
        type=int,
        default=10,
        help="Frame stride for episode evaluation. Use --single-frame to evaluate only --frame.",
    )
    parser.add_argument(
        "--single-frame",
        action="store_true",
        help="Evaluate only --frame instead of sampling an episode curve.",
    )
    parser.add_argument("--num-steps", "-k", type=int, default=15)
    parser.add_argument(
        "--ode-solver",
        choices=ODE_SOLVERS,
        default=ODE_SOLVER_FIREFLOW,
        help="ODE solver for noise-to-data likelihood integration (euler, fireflow, slerpflow).",
    )
    parser.add_argument(
        "--eval-batch-size",
        type=int,
        default=4,
        help="Number of episode frames to integrate per batch. Actual model batch is twice this value.",
    )
    parser.add_argument(
        "--hutchinson-samples",
        type=int,
        default=DEFAULT_HUTCHINSON_SAMPLES,
        help="Number of Hutchinson probes per trace evaluation.",
    )
    parser.add_argument(
        "--hutchinson-seed",
        type=int,
        default=DEFAULT_HUTCHINSON_SEED,
        help="Random seed for Hutchinson probes.",
    )
    parser.add_argument(
        "--noise-seed",
        type=int,
        default=DEFAULT_NOISE_SEED,
        help="Seed for shared Gaussian sampling noise (folded with dataset_index per frame).",
    )
    parser.add_argument(
        "--modalities",
        nargs="+",
        default=list(DEFAULT_MODALITIES),
        choices=MODALITIES,
        help="Modalities to ablate and plot together.",
    )
    parser.add_argument("--output-dir", type=pathlib.Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--y-field",
        default="contribution",
        help="CSV column to plot. Defaults to the modality contribution.",
    )
    parser.add_argument("--output-path", type=pathlib.Path)
    parser.add_argument(
        "--compare-reverse-dir",
        type=pathlib.Path,
        help=(
            "Optional directory of reverse-path contribution CSVs "
            "(same episode/modality naming) to compare against."
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args_with_config(_build_parser, script="forward", argv=argv)

    sample_interval = None if args.single_frame else args.sample_interval
    model = load_model_from_args(args)
    csv_paths = evaluate_modalities_from_noise(
        model=model,
        episode_index=args.episode_index,
        frame=args.frame,
        max_frames=args.max_frames,
        sample_interval=sample_interval,
        num_steps=args.num_steps,
        ode_solver=args.ode_solver,
        eval_batch_size=args.eval_batch_size,
        hutchinson_samples=args.hutchinson_samples,
        hutchinson_seed=args.hutchinson_seed,
        noise_seed=args.noise_seed,
        modalities=args.modalities,
        output_dir=args.output_dir,
    )

    output_path = args.output_path
    if output_path is None:
        output_path = _default_output_path(
            output_dir=args.output_dir,
            y_field=args.y_field,
            episode_index=args.episode_index,
            csv_paths=csv_paths,
            modalities=args.modalities,
        )

    output_path = plot_modalities(
        csv_paths,
        y_field=args.y_field,
        output_path=output_path,
    )
    print(f"plot={output_path}")

    if args.compare_reverse_dir is not None:
        compare_forward_to_reverse(
            csv_paths,
            reverse_dir=args.compare_reverse_dir,
            episode_index=args.episode_index,
            output_dir=args.output_dir,
        )


if __name__ == "__main__":
    main()
