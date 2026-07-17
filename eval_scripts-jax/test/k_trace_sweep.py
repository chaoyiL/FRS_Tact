from __future__ import annotations

# The parent eval directory is inserted below for direct script execution.
# ruff: noqa: E402
import argparse
import gc
import os
import pathlib
import sys
from collections.abc import Sequence

# Reduce JAX GPU memory retention before the backend initializes.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")

ROOT = pathlib.Path(__file__).resolve().parents[2]
EVAL_SCRIPTS = ROOT / "eval_scripts-jax"
for path in (EVAL_SCRIPTS,):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from loglike_evaluate import (
    DEFAULT_HUTCHINSON_SAMPLES,
    DEFAULT_HUTCHINSON_SEED,
    ODE_SOLVER_FIREFLOW,
    ODE_SOLVERS,
    _add_batch_dim,
    _scalar,
    clear_likelihood_scan_cache,
    create_velocity_context,
    integrate_to_base_log_likelihood_with_context,
    load_episode,
)
from utils import add_eval_data_arguments, load_model_from_args


def _parse_k_values(values: Sequence[str]) -> tuple[int, ...]:
    k_values = tuple(int(value) for value in values)
    if not k_values:
        raise ValueError("At least one k value is required.")
    if any(value <= 0 for value in k_values):
        raise ValueError(f"All k values must be positive, got {k_values}.")
    return tuple(sorted(k_values))


def _release_jax_memory() -> None:
    clear_likelihood_scan_cache()
    gc.collect()


def sweep_log_likelihood_over_k(
    model,
    observation,
    reference_actions,
    *,
    k_values: Sequence[int],
    ode_solver: str,
    hutchinson_samples: int,
    hutchinson_seed: int,
    clear_cache_between_k: bool,
) -> list[tuple[int, float]]:
    """Return (k, log_likelihood) pairs in ascending k order."""

    context = create_velocity_context(model, _add_batch_dim(observation))
    rows: list[tuple[int, float]] = []
    for k in k_values:
        result = integrate_to_base_log_likelihood_with_context(
            model,
            context,
            reference_actions,
            num_steps=k,
            hutchinson_samples=hutchinson_samples,
            hutchinson_seed=hutchinson_seed,
            ode_solver=ode_solver,
        )
        log_likelihood = _scalar(result.log_likelihood)
        rows.append((int(k), log_likelihood))
        print(f"k={k} log_likelihood={log_likelihood:.6f}")
        if clear_cache_between_k:
            _release_jax_memory()
    return rows


def save_log_likelihood_plot(
    curve: Sequence[tuple[int, float]],
    *,
    output_path: pathlib.Path,
    episode_index: int | str,
    frame: int,
    ode_solver: str,
) -> pathlib.Path:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    os.environ.setdefault("XDG_CACHE_HOME", "/tmp")
    try:
        import matplotlib

        matplotlib.use("Agg")
        from matplotlib import pyplot as plt
    except ImportError as exc:
        raise ImportError("matplotlib is required to save the k sweep plot.") from exc

    output_path.parent.mkdir(parents=True, exist_ok=True)
    k_values = [k for k, _ in curve]
    log_likelihoods = [value for _, value in curve]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(k_values, log_likelihoods, marker="o", linewidth=1.8, label="log_likelihood")
    ax.set_title(f"log-likelihood vs k (episode={episode_index}, frame={frame}, solver={ode_solver})")
    ax.set_xlabel("k (integration steps)")
    ax.set_ylabel("log likelihood")
    ax.set_xticks(k_values)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Sweep integration step count k and plot log-likelihood convergence."
    )
    add_eval_data_arguments(parser)
    parser.add_argument("--episode-index", type=int, default=0)
    parser.add_argument("--frame", type=int, default=0, help="Episode-relative frame to evaluate.")
    parser.add_argument("--max-frames", type=int, default=2000)
    parser.add_argument(
        "--ode-solver",
        choices=ODE_SOLVERS,
        default=ODE_SOLVER_FIREFLOW,
        help="ODE solver for data-to-base likelihood integration.",
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
        "--k-values",
        nargs="+",
        default=("10", "20", "30", "40", "50"),
        help="Integration step counts to test.",
    )
    parser.add_argument(
        "--keep-jax-cache-between-k",
        action="store_true",
        help="Keep compiled likelihood scans between k values. Faster but uses more memory.",
    )
    parser.add_argument(
        "--output-path",
        type=pathlib.Path,
        default=pathlib.Path("eval_outputs/loglike/k_trace_sweep/log_likelihood_vs_k.png"),
        help="Output plot path.",
    )
    args = parser.parse_args(argv)

    if args.hutchinson_samples <= 0:
        raise ValueError(f"--hutchinson-samples must be positive, got {args.hutchinson_samples}.")

    k_values = _parse_k_values(args.k_values)
    model = load_model_from_args(args)
    episode = load_episode(
        model,
        args.episode_index,
        max_frames=args.max_frames,
        frame_indices=(args.frame,),
    )
    print(f"episode={args.episode_index} frame={args.frame} dataset_index={episode.indices[0]}")
    print(f"ode_solver={args.ode_solver}")
    print(f"hutchinson_samples={args.hutchinson_samples}")
    print(f"hutchinson_seed={args.hutchinson_seed}")
    print(f"k_values={k_values}")
    print(f"clear_cache_between_k={not args.keep_jax_cache_between_k}")

    curve = sweep_log_likelihood_over_k(
        model,
        episode.observations[0],
        episode.actions[0],
        k_values=k_values,
        ode_solver=args.ode_solver,
        hutchinson_samples=args.hutchinson_samples,
        hutchinson_seed=args.hutchinson_seed,
        clear_cache_between_k=not args.keep_jax_cache_between_k,
    )
    plot_path = save_log_likelihood_plot(
        curve,
        output_path=args.output_path,
        episode_index=args.episode_index,
        frame=args.frame,
        ode_solver=args.ode_solver,
    )
    print(f"plot={plot_path}")


if __name__ == "__main__":
    main()
