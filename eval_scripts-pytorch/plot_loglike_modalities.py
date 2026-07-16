#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import os
import re
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from loglike_evaluate import (
    DEFAULT_HUTCHINSON_SAMPLES,
    DEFAULT_HUTCHINSON_SEED,
    MODALITIES,
    ODE_SOLVER_EULER,
    ODE_SOLVER_FIREFLOW,
    ODE_SOLVERS,
    compute_episode_modality_contributions,
    save_contribution_curve,
)
from utils import DEFAULT_MODEL, load_episode, load_policy, parse_json_map, parse_key_list


CSV_PATTERN = re.compile(r"(?P<modality>.+)_contribution_episode_(?P<episode>.+)\.csv$")


def _read_curve(csv_path: Path, y_field: str) -> tuple[np.ndarray, np.ndarray]:
    frames: list[int] = []
    values: list[float] = []
    with csv_path.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        if reader.fieldnames is None:
            raise ValueError(f"{csv_path} is empty")
        required = {"frame", y_field}
        missing = required.difference(reader.fieldnames)
        if missing:
            raise ValueError(
                f"{csv_path} is missing required columns: {', '.join(sorted(missing))}"
            )
        for row in reader:
            frames.append(int(row["frame"]))
            values.append(float(row[y_field]))
    if not frames:
        raise ValueError(f"{csv_path} has no data rows")
    return np.asarray(frames), np.asarray(values)


def _infer_modality(csv_path: Path) -> str:
    match = CSV_PATTERN.match(csv_path.name)
    return match.group("modality") if match is not None else csv_path.stem


def _auto_csv_paths(
    input_dir: Path,
    episode_index: str,
    modalities: Sequence[str],
) -> list[Path]:
    paths = [
        input_dir / f"{modality}_contribution_episode_{episode_index}.csv"
        for modality in modalities
    ]
    existing = [path for path in paths if path.exists()]
    if not existing:
        raise FileNotFoundError(
            f"No modality CSV files found in {input_dir}; looked for "
            f"{', '.join(path.name for path in paths)}"
        )
    return existing


def _default_output_path(
    *,
    output_dir: Path,
    y_field: str,
    episode_index: int | str,
    csv_paths: Sequence[Path],
    modalities: Sequence[str] | None,
) -> Path:
    resolved_modalities = (
        tuple(modalities)
        if modalities is not None
        else tuple(_infer_modality(path) for path in csv_paths)
    )
    if resolved_modalities == MODALITIES:
        return output_dir / f"{y_field}_episode_{episode_index}.png"
    slug = "-".join(resolved_modalities)
    return output_dir / f"{y_field}_{slug}_episode_{episode_index}.png"


def evaluate_modalities(
    *,
    dataset_path: Path,
    model: str,
    device: str | None,
    episode_index: int,
    frame: int,
    max_frames: int | None,
    sample_interval: int | None,
    action_key: str,
    task: str | None,
    rename_map: dict[str, str],
    vision_keys: Sequence[str] | None,
    tactile_keys: Sequence[str] | None,
    num_steps: int,
    ode_solver: str,
    eval_batch_size: int,
    hutchinson_samples: int,
    hutchinson_seed: int,
    modalities: Sequence[str],
    output_dir: Path,
) -> list[Path]:
    if eval_batch_size <= 0:
        raise ValueError(f"--eval-batch-size must be positive, got {eval_batch_size}")
    if hutchinson_samples <= 0:
        raise ValueError(
            f"--hutchinson-samples must be positive, got {hutchinson_samples}"
        )

    loaded_policy = load_policy(model, device=device, rename_map=rename_map)
    episode, meta = load_episode(
        loaded_policy,
        dataset_path,
        episode_index,
        action_key=action_key,
        start_frame=frame,
        sample_interval=sample_interval,
        max_frames=max_frames,
        frame_indices=(frame,) if sample_interval is None else None,
        task_override=task,
    )
    policy = loaded_policy.policy
    resolved_vision_keys = parse_key_list(vision_keys)
    resolved_tactile_keys = parse_key_list(tactile_keys)

    print(
        f"loaded episode={episode_index} frames={len(episode.frames)} "
        f"dataset_indices={episode.dataset_indices[:5]}"
    )
    print(f"dataset_path={Path(dataset_path).expanduser().resolve()}")
    print(f"prompt={episode.prompts[0]!r}")
    print(f"robot_type={meta.info.robot_type}")
    print(f"model={model}")
    print(f"device={policy.config.device}")
    print(f"action_key={action_key}")
    print(f"action_dim={policy.config.action_feature.shape[0]}")
    print(f"action_horizon={policy.config.chunk_size}")
    print("ablation_method=attention_mask")
    print("divergence_method=hutchinson_rademacher_autograd")
    print(f"hutchinson_samples={hutchinson_samples}")
    print(f"hutchinson_seed={hutchinson_seed}")
    print(f"eval_batch_size={eval_batch_size}")
    print(f"ode_solver={ode_solver}")
    print(f"num_steps={num_steps}")

    csv_paths: list[Path] = []
    for modality in modalities:
        print(f"ablated_modality={modality}")
        rows = compute_episode_modality_contributions(
            policy,
            episode,
            modality=modality,
            num_steps=num_steps,
            hutchinson_samples=hutchinson_samples,
            hutchinson_seed=hutchinson_seed,
            ode_solver=ode_solver,
            eval_batch_size=eval_batch_size,
            vision_keys=resolved_vision_keys,
            tactile_keys=resolved_tactile_keys,
        )
        csv_path, component_plot = save_contribution_curve(
            rows,
            output_dir=output_dir,
            modality=modality,
            episode_index=str(episode_index),
        )
        csv_paths.append(csv_path)
        print(f"curve_csv={csv_path}")
        print(f"curve_plot={component_plot}")
    return csv_paths


def plot_modalities(
    csv_paths: Sequence[Path],
    *,
    y_field: str,
    output_path: Path,
) -> Path:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    os.environ.setdefault("XDG_CACHE_HOME", "/tmp")
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(11, 6))
    for csv_path in csv_paths:
        frames, values = _read_curve(csv_path, y_field)
        axis.plot(
            frames,
            values,
            marker="o",
            markersize=3,
            linewidth=1.7,
            label=_infer_modality(csv_path),
        )
    axis.set_xlabel("Frame")
    axis.set_ylabel(y_field.replace("_", " "))
    axis.grid(True, alpha=0.3)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)
    return output_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Estimate SmolVLA modality contributions with probability-flow likelihood "
            "and plot their episode curves."
        )
    )
    parser.add_argument(
        "csv_paths",
        nargs="*",
        type=Path,
        help="Existing contribution CSVs. Supplying these skips model evaluation.",
    )
    parser.add_argument("--dataset-path", type=Path, help="Local LeRobot dataset root")
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"SmolVLA checkpoint path or Hub id (default: {DEFAULT_MODEL})",
    )
    parser.add_argument("--device", default="cuda", help="cpu | cuda | cuda:0 | mps")
    parser.add_argument("--episode-index", type=int, default=0)
    parser.add_argument("--frame", type=int, default=0)
    parser.add_argument("--max-frames", type=int, default=1000)
    parser.add_argument("--sample-interval", type=int, default=10)
    parser.add_argument(
        "--single-frame",
        action="store_true",
        help="Evaluate only --frame instead of an episode curve",
    )
    parser.add_argument("--action-key", default="actions")
    parser.add_argument("--task", default=None, help="Override dataset task text")
    parser.add_argument(
        "--rename-map",
        default=None,
        help="JSON map from dataset observation keys to checkpoint keys",
    )
    parser.add_argument(
        "--vision-keys",
        nargs="+",
        default=None,
        help="Post-rename image keys classified as vision; comma-separated values are accepted",
    )
    parser.add_argument(
        "--tactile-keys",
        nargs="+",
        default=None,
        help="Post-rename image keys classified as tactile; defaults to keys containing 'tactile'",
    )
    parser.add_argument("--num-steps", "-k", type=int, default=120)
    parser.add_argument("--ode-solver", choices=ODE_SOLVERS, default=ODE_SOLVER_FIREFLOW)
    parser.add_argument("--eval-batch-size", type=int, default=4)
    parser.add_argument(
        "--hutchinson-samples",
        type=int,
        default=DEFAULT_HUTCHINSON_SAMPLES,
    )
    parser.add_argument("--hutchinson-seed", type=int, default=DEFAULT_HUTCHINSON_SEED)
    parser.add_argument("--modalities", nargs="+", choices=MODALITIES, default=None)
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("eval_outputs/loglike"),
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--plot-only", action="store_true")
    parser.add_argument("--y-field", default="contribution")
    parser.add_argument("--output-path", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    csv_paths = list(args.csv_paths)
    output_dir = args.output_dir or args.input_dir

    if not csv_paths:
        if args.modalities is None:
            parser.error("--modalities is required when positional CSV paths are omitted")
        if args.plot_only:
            csv_paths = _auto_csv_paths(
                args.input_dir,
                str(args.episode_index),
                args.modalities,
            )
        else:
            if args.dataset_path is None:
                parser.error("--dataset-path is required for evaluation")
            sample_interval = None if args.single_frame else args.sample_interval
            csv_paths = evaluate_modalities(
                dataset_path=args.dataset_path,
                model=args.model,
                device=args.device,
                episode_index=args.episode_index,
                frame=args.frame,
                max_frames=args.max_frames,
                sample_interval=sample_interval,
                action_key=args.action_key,
                task=args.task,
                rename_map=parse_json_map(args.rename_map),
                vision_keys=args.vision_keys,
                tactile_keys=args.tactile_keys,
                num_steps=args.num_steps,
                ode_solver=args.ode_solver,
                eval_batch_size=args.eval_batch_size,
                hutchinson_samples=args.hutchinson_samples,
                hutchinson_seed=args.hutchinson_seed,
                modalities=args.modalities,
                output_dir=output_dir,
            )

    output_path = args.output_path or _default_output_path(
        output_dir=output_dir,
        y_field=args.y_field,
        episode_index=args.episode_index,
        csv_paths=csv_paths,
        modalities=args.modalities,
    )
    plotted = plot_modalities(csv_paths, y_field=args.y_field, output_path=output_path)
    print(f"plot={plotted}")


if __name__ == "__main__":
    main()

