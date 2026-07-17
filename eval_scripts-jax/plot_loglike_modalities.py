from __future__ import annotations

# The eval directory is inserted below so this script also works when invoked by path.
# ruff: noqa: E402
import argparse
import csv
import os
import pathlib
import re
import sys
from collections.abc import Sequence

import jax
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
EVAL_SCRIPTS = pathlib.Path(__file__).resolve().parent
for path in (EVAL_SCRIPTS,):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from loglike_evaluate import (
    DEFAULT_HUTCHINSON_SAMPLES,
    DEFAULT_HUTCHINSON_SEED,
    ODE_SOLVER_EULER,
    ODE_SOLVERS,
    compute_episode_modality_contributions,
    load_episode,
    save_contribution_curve,
)
from utils import SmolVLAEvalModel, add_eval_data_arguments, load_model_from_args

MODALITIES = ("vision", "tactile", "state", "language_prompt")
CSV_PATTERN = re.compile(r"(?P<modality>.+)_contribution_episode_(?P<episode>.+)\.csv$")


def _read_curve(csv_path: pathlib.Path, y_field: str) -> tuple[np.ndarray, np.ndarray]:
    frames = []
    values = []
    with csv_path.open(newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"{csv_path} is empty.")
        required = {"frame", y_field}
        missing = required.difference(reader.fieldnames)
        if missing:
            raise ValueError(f"{csv_path} is missing required column(s): {', '.join(sorted(missing))}")

        for row in reader:
            frames.append(int(row["frame"]))
            values.append(float(row[y_field]))

    if not frames:
        raise ValueError(f"{csv_path} has no data rows.")
    return np.asarray(frames), np.asarray(values)


def _infer_modality(csv_path: pathlib.Path) -> str:
    match = CSV_PATTERN.match(csv_path.name)
    if match is None:
        return csv_path.stem
    return match.group("modality")


def _auto_csv_paths(
    input_dir: pathlib.Path, episode_index: str, modalities: Sequence[str]
) -> list[pathlib.Path]:
    paths = []
    missing = []
    for modality in modalities:
        path = input_dir / f"{modality}_contribution_episode_{episode_index}.csv"
        if path.exists():
            paths.append(path)
        else:
            missing.append(path.name)

    if not paths:
        raise FileNotFoundError(
            f"No modality CSV files found in {input_dir} for episode {episode_index}. "
            f"Looked for: {', '.join(missing)}"
        )
    return paths


def _default_output_path(
    *,
    output_dir: pathlib.Path,
    y_field: str,
    episode_index: int | str,
    csv_paths: Sequence[pathlib.Path],
    modalities: Sequence[str] | None,
) -> pathlib.Path:
    if modalities is None:
        modalities = tuple(_infer_modality(csv_path) for csv_path in csv_paths)

    modality_slug = "-".join(modalities)
    if tuple(modalities) == MODALITIES:
        return output_dir / f"{y_field}_episode_{episode_index}.png"
    return output_dir / f"{y_field}_{modality_slug}_episode_{episode_index}.png"


def evaluate_modalities(
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
    modalities: Sequence[str],
    output_dir: pathlib.Path,
) -> list[pathlib.Path]:
    """Run the same log-likelihood ablation as loglike_evaluate.py for each modality."""

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
    print("divergence_method=hutchinson_rademacher_jvp")
    print(f"hutchinson_samples={hutchinson_samples}")
    print(f"hutchinson_seed={hutchinson_seed}")
    print(f"eval_batch_size={eval_batch_size}")
    print(f"ode_solver={ode_solver}")
    print(f"model_dtype={jax.tree.leaves(model.params)[0].dtype}")

    csv_paths: list[pathlib.Path] = []
    for modality in modalities:
        print(f"ablated_modality={modality}")
        rows = compute_episode_modality_contributions(
            model,
            episode.frames,
            episode.indices,
            episode.observations,
            episode.actions,
            episode.prompts,
            modality=modality,
            num_steps=num_steps,
            prompt_tokenizer=None,
            state_in_prompt=False,
            hutchinson_samples=hutchinson_samples,
            hutchinson_seed=hutchinson_seed,
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


def plot_modalities(
    csv_paths: Sequence[pathlib.Path],
    *,
    y_field: str,
    output_path: pathlib.Path,
) -> pathlib.Path:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(11, 6))

    for csv_path in csv_paths:
        modality = _infer_modality(csv_path)
        frames, values = _read_curve(csv_path, y_field)
        ax.plot(frames, values, marker="o", markersize=3, linewidth=1.7, label=modality)

    ax.set_xlabel("Frame")
    ax.set_ylabel(y_field.replace("_", " "))
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Run log-likelihood modality ablations and plot modality contribution curves."
    )
    parser.add_argument(
        "csv_paths",
        nargs="*",
        type=pathlib.Path,
        help="Existing CSV files to plot. If omitted, --modalities are evaluated first.",
    )
    add_eval_data_arguments(parser, required=False)
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
    parser.add_argument("--num-steps", "-k", type=int, default=120)
    parser.add_argument(
        "--ode-solver",
        choices=ODE_SOLVERS,
        default=ODE_SOLVER_EULER,
        help="ODE solver for data-to-base likelihood integration.",
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
        "--modalities",
        nargs="+",
        default=None,
        choices=MODALITIES,
        help="Modalities to ablate and plot together. Required when CSV paths are omitted.",
    )
    parser.add_argument("--input-dir", type=pathlib.Path, default=pathlib.Path("eval_outputs/loglike"))
    parser.add_argument("--output-dir", type=pathlib.Path, default=None)
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help="Skip evaluation and plot CSVs from positional paths or --input-dir.",
    )
    parser.add_argument(
        "--y-field",
        default="contribution",
        help="CSV column to plot. Defaults to the modality contribution.",
    )
    parser.add_argument("--output-path", type=pathlib.Path)
    args = parser.parse_args(argv)

    csv_paths = list(args.csv_paths)
    if not csv_paths:
        if args.modalities is None:
            parser.error("--modalities is required when CSV paths are omitted.")
        output_dir = args.output_dir or args.input_dir
        if args.plot_only:
            csv_paths = _auto_csv_paths(args.input_dir, str(args.episode_index), args.modalities)
        else:
            if args.checkpoint_dir is None or args.dataset_repo_id is None:
                parser.error("--checkpoint-dir and --dataset-repo-id are required for evaluation.")
            sample_interval = None if args.single_frame else args.sample_interval
            model = load_model_from_args(args)
            csv_paths = evaluate_modalities(
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
                modalities=args.modalities,
                output_dir=output_dir,
            )

    output_path = args.output_path
    if output_path is None:
        output_dir = args.output_dir or args.input_dir
        output_path = _default_output_path(
            output_dir=output_dir,
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


if __name__ == "__main__":
    main()
