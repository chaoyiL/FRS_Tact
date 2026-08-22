from __future__ import annotations

# Numerical experiment: SlerpFlow logp vs k, then full-obs logp for Euler/FireFlow/SlerpFlow at k*.
# ruff: noqa: E402
import argparse
import csv
import dataclasses
import gc
import json
import os
import pathlib
import sys
from collections.abc import Mapping, Sequence
from typing import Any

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")

ROOT = pathlib.Path(__file__).resolve().parents[1]
EVAL_DIR = pathlib.Path(__file__).resolve().parent
if str(EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(EVAL_DIR))

import jax
import yaml
from loglike_evaluate import (
    DEFAULT_HUTCHINSON_SAMPLES,
    DEFAULT_HUTCHINSON_SEED,
    ODE_SOLVER_EULER,
    ODE_SOLVER_FIREFLOW,
    ODE_SOLVER_SLERPFLOW,
    _add_batch_dim,
    _scalar,
    clear_likelihood_scan_cache,
    create_velocity_context,
    integrate_to_base_log_likelihood_with_context,
    load_episode,
)
from utils import add_eval_data_arguments, load_model_from_args

DEFAULT_CONFIG = ROOT / "configs" / "solver_logp_sweep.yaml"
DEFAULT_K_VALUES = (10, 20, 30, 40, 50, 80, 100, 150, 200)
DEFAULT_OUTPUT_DIR = pathlib.Path("eval_outputs/loglike/solver_logp_sweep")
COMPARE_SOLVERS = (ODE_SOLVER_EULER, ODE_SOLVER_FIREFLOW, ODE_SOLVER_SLERPFLOW)


@dataclasses.dataclass(frozen=True)
class LogpRow:
    k: int
    log_likelihood: float
    log_p_base: float
    r_tot: float


@dataclasses.dataclass(frozen=True)
class ConvergenceResult:
    k_star: int
    converged: bool
    atol: float
    patience: int


def load_yaml_config(path: pathlib.Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"config not found: {path}")
    with path.open(encoding="utf-8") as file:
        data = yaml.safe_load(file) or {}
    if not isinstance(data, dict):
        raise ValueError(f"config root must be a mapping: {path}")
    return data


def _section(cfg: Mapping[str, Any], name: str) -> dict[str, Any]:
    value = cfg.get(name)
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError(f"config section {name!r} must be a mapping")
    return dict(value)


def _coerce_path(value: Any) -> pathlib.Path | None:
    if value in (None, ""):
        return None
    return pathlib.Path(value)


def _coerce_rename_map(value: Any) -> str | None:
    if value in (None, ""):
        return None
    if isinstance(value, str):
        return value
    if isinstance(value, Mapping):
        return json.dumps(dict(value))
    raise ValueError("rename_map must be null, a JSON string, or a mapping")


def flatten_sweep_defaults(cfg: Mapping[str, Any]) -> dict[str, Any]:
    defaults: dict[str, Any] = {}
    path_keys = {"checkpoint_dir", "dataset_root", "output_dir"}
    for section_name in ("data", "experiment"):
        for key, value in _section(cfg, section_name).items():
            if key in path_keys:
                defaults[key] = _coerce_path(value)
            elif key == "rename_map":
                defaults[key] = _coerce_rename_map(value)
            else:
                defaults[key] = value
    return defaults


def parse_k_values(values: Sequence[Any]) -> tuple[int, ...]:
    k_values = tuple(int(value) for value in values)
    if not k_values:
        raise ValueError("At least one k value is required.")
    if any(value <= 0 for value in k_values):
        raise ValueError(f"All k values must be positive, got {k_values}.")
    return tuple(sorted(set(k_values)))


def detect_convergence(
    curve: Sequence[LogpRow],
    *,
    atol: float,
    patience: int,
) -> ConvergenceResult:
    """Return k* when |Δlogp| < atol for ``patience`` consecutive intervals."""

    if atol < 0:
        raise ValueError(f"atol must be non-negative, got {atol}")
    if patience <= 0:
        raise ValueError(f"patience must be positive, got {patience}")
    if not curve:
        raise ValueError("curve must be non-empty")

    streak = 0
    for index in range(1, len(curve)):
        delta = abs(curve[index].log_likelihood - curve[index - 1].log_likelihood)
        if delta < atol:
            streak += 1
            if streak >= patience:
                return ConvergenceResult(
                    k_star=curve[index].k,
                    converged=True,
                    atol=atol,
                    patience=patience,
                )
        else:
            streak = 0

    return ConvergenceResult(
        k_star=curve[-1].k,
        converged=False,
        atol=atol,
        patience=patience,
    )


def _release_jax_memory() -> None:
    clear_likelihood_scan_cache()
    gc.collect()


def sweep_slerpflow_logp_over_k(
    model,
    observation,
    reference_actions,
    *,
    k_values: Sequence[int],
    hutchinson_samples: int,
    hutchinson_seed: int,
    clear_cache_between_runs: bool,
) -> list[LogpRow]:
    context = create_velocity_context(model, _add_batch_dim(observation))
    rows: list[LogpRow] = []
    for k in k_values:
        result = integrate_to_base_log_likelihood_with_context(
            model,
            context,
            reference_actions,
            num_steps=k,
            hutchinson_samples=hutchinson_samples,
            hutchinson_seed=hutchinson_seed,
            ode_solver=ODE_SOLVER_SLERPFLOW,
        )
        row = LogpRow(
            k=int(k),
            log_likelihood=_scalar(result.log_likelihood),
            log_p_base=_scalar(result.log_p_base),
            r_tot=_scalar(result.r_tot),
        )
        rows.append(row)
        print(
            f"[slerpflow] k={row.k} log_likelihood={row.log_likelihood:.6f} "
            f"log_p_base={row.log_p_base:.6f} r_tot={row.r_tot:.6f}"
        )
        if clear_cache_between_runs:
            _release_jax_memory()
    return rows


def compare_solvers_at_k(
    model,
    observation,
    reference_actions,
    *,
    k: int,
    hutchinson_samples: int,
    hutchinson_seed: int,
    clear_cache_between_runs: bool,
) -> list[dict[str, float | int | str]]:
    context = create_velocity_context(model, _add_batch_dim(observation))
    by_solver: dict[str, LogpRow] = {}
    for solver in COMPARE_SOLVERS:
        result = integrate_to_base_log_likelihood_with_context(
            model,
            context,
            reference_actions,
            num_steps=k,
            hutchinson_samples=hutchinson_samples,
            hutchinson_seed=hutchinson_seed,
            ode_solver=solver,
        )
        by_solver[solver] = LogpRow(
            k=int(k),
            log_likelihood=_scalar(result.log_likelihood),
            log_p_base=_scalar(result.log_p_base),
            r_tot=_scalar(result.r_tot),
        )
        print(
            f"[compare] solver={solver} k={k} "
            f"log_likelihood={by_solver[solver].log_likelihood:.6f}"
        )
        if clear_cache_between_runs:
            _release_jax_memory()

    baseline = by_solver[ODE_SOLVER_SLERPFLOW]
    rows: list[dict[str, float | int | str]] = []
    for solver in COMPARE_SOLVERS:
        row = by_solver[solver]
        rows.append(
            {
                "solver": solver,
                "k": row.k,
                "log_likelihood": row.log_likelihood,
                "log_p_base": row.log_p_base,
                "r_tot": row.r_tot,
                "delta_logp_vs_slerpflow": row.log_likelihood - baseline.log_likelihood,
                "delta_log_p_base_vs_slerpflow": row.log_p_base - baseline.log_p_base,
                "delta_r_tot_vs_slerpflow": row.r_tot - baseline.r_tot,
            }
        )
    return rows


def save_logp_vs_k_csv(rows: Sequence[LogpRow], path: pathlib.Path) -> pathlib.Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["k", "log_likelihood", "log_p_base", "r_tot"],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(dataclasses.asdict(row))
    return path


def save_dict_rows_csv(
    rows: Sequence[Mapping[str, Any]],
    path: pathlib.Path,
    fieldnames: Sequence[str],
) -> pathlib.Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(fieldnames))
        writer.writeheader()
        writer.writerows(rows)
    return path


def plot_logp_vs_k(
    rows: Sequence[LogpRow],
    *,
    convergence: ConvergenceResult,
    output_path: pathlib.Path,
    episode_index: int | str,
    frame: int,
) -> pathlib.Path:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    os.environ.setdefault("XDG_CACHE_HOME", "/tmp")
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    output_path.parent.mkdir(parents=True, exist_ok=True)
    ks = [row.k for row in rows]
    logps = [row.log_likelihood for row in rows]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(ks, logps, marker="o", linewidth=1.8, label="slerpflow logp")
    ax.axvline(
        convergence.k_star,
        color="C3",
        linestyle="--",
        linewidth=1.4,
        label=f"k*={convergence.k_star} (converged={convergence.converged})",
    )
    ax.set_title(
        f"SlerpFlow logp vs k (episode={episode_index}, frame={frame}, "
        f"atol={convergence.atol}, patience={convergence.patience})"
    )
    ax.set_xlabel("k (integration steps)")
    ax.set_ylabel("log likelihood")
    ax.set_xticks(ks)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path


def plot_solver_compare(
    solver_rows: Sequence[Mapping[str, Any]],
    *,
    k_star: int,
    output_path: pathlib.Path,
    episode_index: int | str,
    frame: int,
) -> pathlib.Path:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    os.environ.setdefault("XDG_CACHE_HOME", "/tmp")
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    output_path.parent.mkdir(parents=True, exist_ok=True)
    solvers = [str(row["solver"]) for row in solver_rows]
    logps = [float(row["log_likelihood"]) for row in solver_rows]
    deltas = [float(row["delta_logp_vs_slerpflow"]) for row in solver_rows]

    fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    fig.suptitle(
        f"Full-observation logp by solver at k*={k_star} "
        f"(episode={episode_index}, frame={frame})"
    )

    axes[0].bar(solvers, logps, color=["C0", "C1", "C2"][: len(solvers)])
    axes[0].set_ylabel("log likelihood")
    axes[0].set_title("logp(a_GT | o)")
    axes[0].grid(True, axis="y", alpha=0.3)

    axes[1].bar(solvers, deltas, color=["C0", "C1", "C2"][: len(solvers)])
    axes[1].axhline(0.0, color="black", linewidth=0.8)
    axes[1].set_ylabel("Δlogp vs slerpflow")
    axes[1].set_title("Difference relative to SlerpFlow")
    axes[1].grid(True, axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Sweep SlerpFlow logp vs integration steps k, detect convergence, "
            "then compare Euler / FireFlow / SlerpFlow at k*."
        )
    )
    parser.add_argument(
        "--config",
        type=pathlib.Path,
        default=DEFAULT_CONFIG,
        help=f"YAML config path (default: {DEFAULT_CONFIG})",
    )
    add_eval_data_arguments(parser, required=False)
    parser.set_defaults(
        checkpoint_dir=pathlib.Path("/home/typhon/models/tactile_test_05_1.5w"),
        dataset_repo_id="chaoyi/tactile_test_03",
    )
    parser.add_argument("--episode-index", type=int, default=0)
    parser.add_argument("--frame", type=int, default=0)
    parser.add_argument("--max-frames", type=int, default=2000)
    parser.add_argument(
        "--k-values",
        nargs="+",
        default=None,
        help="Integration step counts for the SlerpFlow sweep.",
    )
    parser.add_argument("--atol", type=float, default=1.0)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--hutchinson-samples", type=int, default=DEFAULT_HUTCHINSON_SAMPLES)
    parser.add_argument("--hutchinson-seed", type=int, default=DEFAULT_HUTCHINSON_SEED)
    parser.add_argument("--output-dir", type=pathlib.Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--keep-jax-cache-between-runs",
        action="store_true",
        help="Keep compiled scans between k/solver runs (faster, more memory).",
    )
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    argv_list = list(argv) if argv is not None else None
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", type=pathlib.Path, default=DEFAULT_CONFIG)
    pre_args, _ = pre.parse_known_args(argv_list)

    parser = _build_parser()
    cfg = load_yaml_config(pre_args.config)
    parser.set_defaults(**flatten_sweep_defaults(cfg))
    args = parser.parse_args(argv_list)

    if args.k_values is None:
        args.k_values = list(_section(cfg, "experiment").get("k_values", DEFAULT_K_VALUES))

    clear_default = _section(cfg, "experiment").get("clear_cache_between_runs", True)
    args.clear_cache_between_runs = bool(clear_default) and not args.keep_jax_cache_between_runs
    return args


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if args.hutchinson_samples <= 0:
        raise ValueError(f"--hutchinson-samples must be positive, got {args.hutchinson_samples}")
    if args.patience <= 0:
        raise ValueError(f"--patience must be positive, got {args.patience}")
    if args.atol < 0:
        raise ValueError(f"--atol must be non-negative, got {args.atol}")

    k_values = parse_k_values(args.k_values)
    output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model = load_model_from_args(args)
    episode = load_episode(
        model,
        args.episode_index,
        max_frames=args.max_frames,
        frame_indices=(args.frame,),
    )
    observation = episode.observations[0]
    reference_actions = episode.actions[0]

    print(f"episode={args.episode_index} frame={args.frame} dataset_index={episode.indices[0]}")
    print(f"model_dtype={jax.tree.leaves(model.params)[0].dtype}")
    print(f"k_values={k_values}")
    print(f"atol={args.atol} patience={args.patience}")
    print(f"hutchinson_samples={args.hutchinson_samples} hutchinson_seed={args.hutchinson_seed}")
    print(f"clear_cache_between_runs={args.clear_cache_between_runs}")
    print(f"output_dir={output_dir}")

    print("=== Exp1: SlerpFlow logp vs k ===")
    curve = sweep_slerpflow_logp_over_k(
        model,
        observation,
        reference_actions,
        k_values=k_values,
        hutchinson_samples=args.hutchinson_samples,
        hutchinson_seed=args.hutchinson_seed,
        clear_cache_between_runs=args.clear_cache_between_runs,
    )
    convergence = detect_convergence(curve, atol=args.atol, patience=args.patience)
    print(
        f"convergence: k_star={convergence.k_star} converged={convergence.converged} "
        f"atol={convergence.atol} patience={convergence.patience}"
    )

    curve_csv = save_logp_vs_k_csv(curve, output_dir / "slerpflow_logp_vs_k.csv")
    curve_plot = plot_logp_vs_k(
        curve,
        convergence=convergence,
        output_path=output_dir / "slerpflow_logp_vs_k.png",
        episode_index=args.episode_index,
        frame=args.frame,
    )
    print(f"curve_csv={curve_csv}")
    print(f"curve_plot={curve_plot}")

    print(f"=== Exp2: full-observation logp by solver at k*={convergence.k_star} ===")
    solver_rows = compare_solvers_at_k(
        model,
        observation,
        reference_actions,
        k=convergence.k_star,
        hutchinson_samples=args.hutchinson_samples,
        hutchinson_seed=args.hutchinson_seed,
        clear_cache_between_runs=args.clear_cache_between_runs,
    )

    solver_csv = save_dict_rows_csv(
        solver_rows,
        output_dir / "solver_compare_at_kstar.csv",
        fieldnames=[
            "solver",
            "k",
            "log_likelihood",
            "log_p_base",
            "r_tot",
            "delta_logp_vs_slerpflow",
            "delta_log_p_base_vs_slerpflow",
            "delta_r_tot_vs_slerpflow",
        ],
    )
    compare_plot = plot_solver_compare(
        solver_rows,
        k_star=convergence.k_star,
        output_path=output_dir / "solver_compare_at_kstar.png",
        episode_index=args.episode_index,
        frame=args.frame,
    )
    print(f"solver_csv={solver_csv}")
    print(f"compare_plot={compare_plot}")

    summary_path = output_dir / "summary.txt"
    with summary_path.open("w", encoding="utf-8") as file:
        file.write(
            f"episode={args.episode_index}\n"
            f"frame={args.frame}\n"
            f"k_star={convergence.k_star}\n"
            f"converged={convergence.converged}\n"
            f"atol={convergence.atol}\n"
            f"patience={convergence.patience}\n"
        )
        for row in solver_rows:
            file.write(
                f"solver={row['solver']} logp={row['log_likelihood']:.6f} "
                f"delta_vs_slerpflow={row['delta_logp_vs_slerpflow']:.6f}\n"
            )
    print(f"summary={summary_path}")


if __name__ == "__main__":
    main()
