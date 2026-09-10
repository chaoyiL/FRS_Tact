from __future__ import annotations

"""Compare reverse-reused and independently recomputed forward FireFlow fields.

SmolVLA uses model time ``t=1`` for Gaussian noise and ``t=0`` for data.  This
module calls the data-to-base solve "reverse" (model time 0 -> 1), matching the
terminology used by the modality likelihood evaluation.  For the generative
base-to-data direction we use ``s = 1 - t`` and therefore

    u(x, s) = -v_theta(x, 1 - s).

Method A reuses the velocities seen by the reverse solve, reverses their order,
and negates them.  Method B starts at the inferred base point and independently
runs FireFlow back to data, evaluating a fresh forward field on that trajectory.
"""

# ruff: noqa: E402
import argparse
import csv
import gc
import json
import os
import pathlib
import subprocess
import sys
import tempfile
from collections.abc import Callable, Mapping, Sequence
from typing import Any, NamedTuple

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")

ROOT = pathlib.Path(__file__).resolve().parents[1]
EVAL_DIR = pathlib.Path(__file__).resolve().parent
if str(EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(EVAL_DIR))

import jax
import jax.numpy as jnp
import numpy as np
import yaml
from loglike_evaluate import (
    DEFAULT_HUTCHINSON_SAMPLES,
    DEFAULT_HUTCHINSON_SEED,
    standard_normal_log_prob,
    velocity_and_hutchinson_trace,
)
from utils import (
    EvalObservation,
    SmolVLAEvalModel,
    VelocityContext,
    _add_batch_dim,
    add_eval_data_arguments,
    create_velocity_context,
    load_episode,
    load_model_from_args,
    predict_velocity_with_context,
)

DEFAULT_CONFIG = ROOT / "configs" / "reverse_forward_logp_compare.yaml"
DEFAULT_OUTPUT_DIR = pathlib.Path("eval_outputs/loglike/reverse_forward_compare")

VelocityFn = Callable[[jax.Array, jax.Array], jax.Array]
VelocityTraceFn = Callable[[jax.Array, jax.Array, jax.Array], tuple[jax.Array, jax.Array]]


class FireFlowTrajectory(NamedTuple):
    """One FireFlow solve, including every model-field midpoint evaluation."""

    states: jax.Array
    midpoint_states: jax.Array
    model_velocities: jax.Array
    model_divergences: jax.Array
    model_divergence_integral: jax.Array


class ReverseForwardComparison(NamedTuple):
    """Aligned Method-A/Method-B fields and their likelihood comparison."""

    x_base: jax.Array
    x_reconstructed: jax.Array
    reverse_states: jax.Array
    forward_states: jax.Array
    reverse_midpoints_aligned: jax.Array
    forward_midpoints: jax.Array
    reverse_reused_fields: jax.Array
    forward_recomputed_fields: jax.Array
    reverse_reused_field_divergences: jax.Array
    forward_recomputed_field_divergences: jax.Array
    log_p_base: jax.Array
    reverse_reused_field_divergence_integral: jax.Array
    forward_recomputed_field_divergence_integral: jax.Array
    reverse_reused_log_p: jax.Array
    forward_recomputed_log_p: jax.Array
    reconstruction_rmse: jax.Array
    trajectory_rmse: jax.Array
    trajectory_max_abs_error: jax.Array
    field_rmse: jax.Array
    field_mae: jax.Array
    field_max_abs_error: jax.Array
    field_relative_l2: jax.Array
    field_cosine_similarity: jax.Array
    divergence_rmse: jax.Array
    divergence_relative_l2: jax.Array
    log_p_abs_error: jax.Array
    log_p_relative_error: jax.Array


def _fireflow_trajectory(
    *,
    x: jax.Array,
    start_time: float,
    dt: jax.Array,
    interval_indices: jax.Array,
    rng_key: jax.Array,
    velocity_fn: VelocityFn,
    velocity_trace_fn: VelocityTraceFn,
) -> FireFlowTrajectory:
    """Run FireFlow modified midpoint while retaining midpoint field values."""

    batch_size = x.shape[0]
    t = jnp.full((batch_size,), start_time, dtype=jnp.float32)
    first_interval = interval_indices[0]

    v0 = velocity_fn(x, t)
    first_midpoint = x + 0.5 * dt * v0
    first_velocity, first_divergence = velocity_trace_fn(
        first_midpoint,
        t + 0.5 * dt,
        jax.random.fold_in(rng_key, first_interval),
    )
    first_state = x + dt * first_velocity
    first_integral = dt * first_divergence

    def body(carry, interval_index):
        x_t, t_t, previous_mid_velocity, divergence_integral = carry
        midpoint = x_t + 0.5 * dt * previous_mid_velocity
        velocity, divergence = velocity_trace_fn(
            midpoint,
            t_t + 0.5 * dt,
            jax.random.fold_in(rng_key, interval_index),
        )
        next_x = x_t + dt * velocity
        return (
            next_x,
            t_t + dt,
            velocity,
            divergence_integral + dt * divergence,
        ), (next_x, midpoint, velocity, divergence)

    (_, _, _, divergence_integral), history = jax.lax.scan(
        body,
        (first_state, t + dt, first_velocity, first_integral),
        interval_indices[1:],
    )
    remaining_states, remaining_midpoints, remaining_velocities, remaining_divergences = history
    return FireFlowTrajectory(
        states=jnp.concatenate((x[None, ...], first_state[None, ...], remaining_states), axis=0),
        midpoint_states=jnp.concatenate((first_midpoint[None, ...], remaining_midpoints), axis=0),
        model_velocities=jnp.concatenate((first_velocity[None, ...], remaining_velocities), axis=0),
        model_divergences=jnp.concatenate(
            (first_divergence[None, ...], remaining_divergences), axis=0
        ),
        model_divergence_integral=divergence_integral,
    )


def _relative_l2(error: jax.Array, reference: jax.Array, axes: tuple[int, ...]) -> jax.Array:
    numerator = jnp.sum(jnp.square(error), axis=axes)
    denominator = jnp.sum(jnp.square(reference), axis=axes)
    return jnp.sqrt(numerator / jnp.maximum(denominator, jnp.asarray(1e-12, dtype=jnp.float32)))


def compare_reverse_reuse_to_forward_recompute(
    velocity_fn: VelocityFn,
    velocity_trace_fn: VelocityTraceFn,
    x_data: jax.Array,
    *,
    num_steps: int,
    rng_key: jax.Array,
) -> ReverseForwardComparison:
    """Evaluate the two likelihood fields using paired FireFlow intervals.

    Hutchinson probes are keyed by the physical interval index.  The forward
    solve visits intervals in reverse order, so corresponding Method-A and
    Method-B divergence evaluations use the same probe.
    """

    if num_steps <= 0:
        raise ValueError(f"num_steps must be positive, got {num_steps}")

    x_data = jnp.asarray(x_data, dtype=jnp.float32)
    if x_data.ndim < 2:
        raise ValueError(f"x_data must include batch and event dimensions, got {x_data.shape}")

    step_size = jnp.asarray(1.0 / num_steps, dtype=jnp.float32)
    reverse_intervals = jnp.arange(num_steps, dtype=jnp.int32)
    forward_intervals = reverse_intervals[::-1]

    # GT/data -> inferred base x0*: model time 0 -> 1.
    reverse = _fireflow_trajectory(
        x=x_data,
        start_time=0.0,
        dt=step_size,
        interval_indices=reverse_intervals,
        rng_key=rng_key,
        velocity_fn=velocity_fn,
        velocity_trace_fn=velocity_trace_fn,
    )
    x_base = reverse.states[-1]

    # Inferred base x0* -> reconstructed data: model time 1 -> 0.
    forward = _fireflow_trajectory(
        x=x_base,
        start_time=1.0,
        dt=-step_size,
        interval_indices=forward_intervals,
        rng_key=rng_key,
        velocity_fn=velocity_fn,
        velocity_trace_fn=velocity_trace_fn,
    )

    # Both arrays below are ordered in increasing generative time s=1-t.  The
    # minus sign converts the model-time field v_theta into the generative field u.
    reverse_reused_fields = -reverse.model_velocities[::-1]
    forward_recomputed_fields = -forward.model_velocities
    reverse_reused_divergences = -reverse.model_divergences[::-1]
    forward_recomputed_divergences = -forward.model_divergences

    reverse_field_integral = step_size * jnp.sum(reverse_reused_divergences, axis=0)
    forward_field_integral = step_size * jnp.sum(forward_recomputed_divergences, axis=0)
    log_p_base = standard_normal_log_prob(x_base)
    reverse_log_p = log_p_base - reverse_field_integral
    forward_log_p = log_p_base - forward_field_integral

    aligned_reverse_states = reverse.states[::-1]
    aligned_reverse_midpoints = reverse.midpoint_states[::-1]
    state_difference = forward.states - aligned_reverse_states
    field_difference = forward_recomputed_fields - reverse_reused_fields
    divergence_difference = forward_recomputed_divergences - reverse_reused_divergences

    event_axes = tuple(range(1, x_data.ndim))
    history_event_axes = (0, *tuple(range(2, forward.states.ndim)))
    field_axes = (0, *tuple(range(2, forward_recomputed_fields.ndim)))
    divergence_axes = (0,)

    reconstruction_rmse = jnp.sqrt(jnp.mean(jnp.square(forward.states[-1] - x_data), axis=event_axes))
    trajectory_rmse = jnp.sqrt(jnp.mean(jnp.square(state_difference), axis=history_event_axes))
    trajectory_max_abs_error = jnp.max(jnp.abs(state_difference), axis=history_event_axes)
    field_rmse = jnp.sqrt(jnp.mean(jnp.square(field_difference), axis=field_axes))
    field_mae = jnp.mean(jnp.abs(field_difference), axis=field_axes)
    field_max_abs_error = jnp.max(jnp.abs(field_difference), axis=field_axes)
    field_relative_l2 = _relative_l2(field_difference, reverse_reused_fields, field_axes)
    field_dot = jnp.sum(forward_recomputed_fields * reverse_reused_fields, axis=field_axes)
    field_norm_product = jnp.sqrt(
        jnp.sum(jnp.square(forward_recomputed_fields), axis=field_axes)
        * jnp.sum(jnp.square(reverse_reused_fields), axis=field_axes)
    )
    field_cosine_similarity = field_dot / jnp.maximum(
        field_norm_product, jnp.asarray(1e-12, dtype=jnp.float32)
    )
    divergence_rmse = jnp.sqrt(jnp.mean(jnp.square(divergence_difference), axis=divergence_axes))
    divergence_relative_l2 = _relative_l2(
        divergence_difference,
        reverse_reused_divergences,
        divergence_axes,
    )
    log_p_abs_error = jnp.abs(forward_log_p - reverse_log_p)
    log_p_relative_error = log_p_abs_error / jnp.maximum(
        jnp.abs(reverse_log_p), jnp.asarray(1e-12, dtype=jnp.float32)
    )

    return ReverseForwardComparison(
        x_base=x_base,
        x_reconstructed=forward.states[-1],
        reverse_states=reverse.states,
        forward_states=forward.states,
        reverse_midpoints_aligned=aligned_reverse_midpoints,
        forward_midpoints=forward.midpoint_states,
        reverse_reused_fields=reverse_reused_fields,
        forward_recomputed_fields=forward_recomputed_fields,
        reverse_reused_field_divergences=reverse_reused_divergences,
        forward_recomputed_field_divergences=forward_recomputed_divergences,
        log_p_base=log_p_base,
        reverse_reused_field_divergence_integral=reverse_field_integral,
        forward_recomputed_field_divergence_integral=forward_field_integral,
        reverse_reused_log_p=reverse_log_p,
        forward_recomputed_log_p=forward_log_p,
        reconstruction_rmse=reconstruction_rmse,
        trajectory_rmse=trajectory_rmse,
        trajectory_max_abs_error=trajectory_max_abs_error,
        field_rmse=field_rmse,
        field_mae=field_mae,
        field_max_abs_error=field_max_abs_error,
        field_relative_l2=field_relative_l2,
        field_cosine_similarity=field_cosine_similarity,
        divergence_rmse=divergence_rmse,
        divergence_relative_l2=divergence_relative_l2,
        log_p_abs_error=log_p_abs_error,
        log_p_relative_error=log_p_relative_error,
    )


_MODEL_RUN_CACHE: dict[tuple[int, int, int, int], Callable] = {}


def _get_model_runner(
    model: SmolVLAEvalModel,
    *,
    num_steps: int,
    hutchinson_samples: int,
    hutchinson_seed: int,
):
    cache_key = (id(model), num_steps, hutchinson_samples, hutchinson_seed)
    if cache_key in _MODEL_RUN_CACHE:
        return _MODEL_RUN_CACHE[cache_key]

    rng_key = jax.random.PRNGKey(hutchinson_seed)

    @jax.jit
    def run(context: VelocityContext, x_data: jax.Array) -> ReverseForwardComparison:
        def velocity_fn(x: jax.Array, t: jax.Array) -> jax.Array:
            return predict_velocity_with_context(model, context, x, t).astype(jnp.float32)

        def velocity_trace_fn(
            x: jax.Array,
            t: jax.Array,
            step_rng_key: jax.Array,
        ) -> tuple[jax.Array, jax.Array]:
            return velocity_and_hutchinson_trace(
                model,
                context,
                x,
                t,
                step_rng_key,
                num_samples=hutchinson_samples,
            )

        return compare_reverse_reuse_to_forward_recompute(
            velocity_fn,
            velocity_trace_fn,
            x_data,
            num_steps=num_steps,
            rng_key=rng_key,
        )

    _MODEL_RUN_CACHE[cache_key] = run
    return run


def compare_model_reverse_forward(
    model: SmolVLAEvalModel,
    observation: EvalObservation,
    reference_actions: jax.Array,
    *,
    num_steps: int,
    hutchinson_samples: int = DEFAULT_HUTCHINSON_SAMPLES,
    hutchinson_seed: int = DEFAULT_HUTCHINSON_SEED,
) -> ReverseForwardComparison:
    """Run the paired FireFlow comparison for one normalized GT action chunk."""

    if hutchinson_samples <= 0:
        raise ValueError(f"hutchinson_samples must be positive, got {hutchinson_samples}")
    context = create_velocity_context(model, _add_batch_dim(observation))
    actions = jnp.asarray(reference_actions, dtype=jnp.float32)
    if actions.ndim == 2:
        actions = actions[None, ...]
    runner = _get_model_runner(
        model,
        num_steps=num_steps,
        hutchinson_samples=hutchinson_samples,
        hutchinson_seed=hutchinson_seed,
    )
    return runner(context, actions)


SUMMARY_FIELDS = (
    "steps",
    "event_size",
    "log_p_base",
    "reverse_reused_field_divergence_integral",
    "forward_recomputed_field_divergence_integral",
    "reverse_reused_log_p",
    "forward_recomputed_log_p",
    "log_p_abs_error",
    "log_p_abs_error_per_dim",
    "log_p_relative_error",
    "reconstruction_rmse",
    "trajectory_rmse",
    "trajectory_max_abs_error",
    "field_rmse",
    "field_mae",
    "field_max_abs_error",
    "field_relative_l2",
    "field_cosine_similarity",
    "divergence_rmse",
    "divergence_relative_l2",
)

PROFILE_FIELDS = (
    "steps",
    "forward_step",
    "generative_s_mid",
    "model_t_mid",
    "midpoint_state_rmse",
    "field_rmse",
    "field_relative_l2",
    "field_cosine_similarity",
    "reverse_reused_field_divergence",
    "forward_recomputed_field_divergence",
    "field_divergence_difference",
)


def _first_scalar(value: Any) -> float:
    return float(np.asarray(jax.device_get(value)).reshape(-1)[0])


def result_row(result: ReverseForwardComparison, *, num_steps: int) -> dict[str, float | int]:
    event_size = int(np.prod(result.x_base.shape[1:]))
    log_p_abs_error = _first_scalar(result.log_p_abs_error)
    return {
        "steps": int(num_steps),
        "event_size": event_size,
        "log_p_base": _first_scalar(result.log_p_base),
        "reverse_reused_field_divergence_integral": _first_scalar(
            result.reverse_reused_field_divergence_integral
        ),
        "forward_recomputed_field_divergence_integral": _first_scalar(
            result.forward_recomputed_field_divergence_integral
        ),
        "reverse_reused_log_p": _first_scalar(result.reverse_reused_log_p),
        "forward_recomputed_log_p": _first_scalar(result.forward_recomputed_log_p),
        "log_p_abs_error": log_p_abs_error,
        "log_p_abs_error_per_dim": log_p_abs_error / event_size,
        "log_p_relative_error": _first_scalar(result.log_p_relative_error),
        "reconstruction_rmse": _first_scalar(result.reconstruction_rmse),
        "trajectory_rmse": _first_scalar(result.trajectory_rmse),
        "trajectory_max_abs_error": _first_scalar(result.trajectory_max_abs_error),
        "field_rmse": _first_scalar(result.field_rmse),
        "field_mae": _first_scalar(result.field_mae),
        "field_max_abs_error": _first_scalar(result.field_max_abs_error),
        "field_relative_l2": _first_scalar(result.field_relative_l2),
        "field_cosine_similarity": _first_scalar(result.field_cosine_similarity),
        "divergence_rmse": _first_scalar(result.divergence_rmse),
        "divergence_relative_l2": _first_scalar(result.divergence_relative_l2),
    }


def profile_rows(
    result: ReverseForwardComparison,
    *,
    num_steps: int,
    batch_index: int = 0,
) -> list[dict[str, float | int]]:
    """Return per-midpoint diagnostics in increasing generative time."""

    reverse_fields = np.asarray(jax.device_get(result.reverse_reused_fields[:, batch_index]))
    forward_fields = np.asarray(jax.device_get(result.forward_recomputed_fields[:, batch_index]))
    reverse_midpoints = np.asarray(jax.device_get(result.reverse_midpoints_aligned[:, batch_index]))
    forward_midpoints = np.asarray(jax.device_get(result.forward_midpoints[:, batch_index]))
    reverse_divergences = np.asarray(
        jax.device_get(result.reverse_reused_field_divergences[:, batch_index])
    )
    forward_divergences = np.asarray(
        jax.device_get(result.forward_recomputed_field_divergences[:, batch_index])
    )

    rows: list[dict[str, float | int]] = []
    for step in range(num_steps):
        reverse_field = reverse_fields[step].reshape(-1)
        forward_field = forward_fields[step].reshape(-1)
        difference = forward_field - reverse_field
        reference_norm = np.linalg.norm(reverse_field)
        norm_product = np.linalg.norm(reverse_field) * np.linalg.norm(forward_field)
        s_mid = (step + 0.5) / num_steps
        rows.append(
            {
                "steps": int(num_steps),
                "forward_step": step,
                "generative_s_mid": s_mid,
                "model_t_mid": 1.0 - s_mid,
                "midpoint_state_rmse": float(
                    np.sqrt(np.mean(np.square(forward_midpoints[step] - reverse_midpoints[step])))
                ),
                "field_rmse": float(np.sqrt(np.mean(np.square(difference)))),
                "field_relative_l2": float(
                    np.linalg.norm(difference) / max(float(reference_norm), 1e-12)
                ),
                "field_cosine_similarity": float(
                    np.dot(reverse_field, forward_field) / max(float(norm_product), 1e-12)
                ),
                "reverse_reused_field_divergence": float(reverse_divergences[step]),
                "forward_recomputed_field_divergence": float(forward_divergences[step]),
                "field_divergence_difference": float(
                    forward_divergences[step] - reverse_divergences[step]
                ),
            }
        )
    return rows


def _write_csv(
    rows: Sequence[Mapping[str, Any]],
    path: pathlib.Path,
    *,
    fieldnames: Sequence[str],
) -> pathlib.Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(fieldnames))
        writer.writeheader()
        writer.writerows(rows)
    return path


def save_comparison_plot(rows: Sequence[Mapping[str, Any]], path: pathlib.Path) -> pathlib.Path:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    os.environ.setdefault("XDG_CACHE_HOME", "/tmp")
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    steps = [int(row["steps"]) for row in rows]
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    axes[0, 0].plot(steps, [float(row["field_relative_l2"]) for row in rows], marker="o")
    axes[0, 0].set_ylabel("field relative L2")
    axes[0, 1].plot(
        steps,
        [float(row["field_cosine_similarity"]) for row in rows],
        marker="o",
    )
    axes[0, 1].set_ylabel("field cosine similarity")
    axes[1, 0].plot(steps, [float(row["log_p_abs_error"]) for row in rows], marker="o")
    axes[1, 0].set_ylabel("absolute logp gap")
    axes[1, 1].plot(steps, [float(row["reconstruction_rmse"]) for row in rows], marker="o")
    axes[1, 1].set_ylabel("round-trip action RMSE")
    for axis in axes.flat:
        axis.set_xlabel("FireFlow steps")
        axis.grid(True, alpha=0.3)
        if len(steps) > 1 and all(step > 0 for step in steps):
            axis.set_xscale("log")
    fig.suptitle("Reverse-reused vs forward-recomputed FireFlow likelihood field")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def _load_config(path: pathlib.Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"config not found: {path}")
    with path.open(encoding="utf-8") as file:
        config = yaml.safe_load(file) or {}
    if not isinstance(config, dict):
        raise ValueError(f"config root must be a mapping: {path}")
    return config


def _flatten_config(config: Mapping[str, Any]) -> dict[str, Any]:
    defaults: dict[str, Any] = {}
    for section_name in ("data", "experiment"):
        section = config.get(section_name, {})
        if not isinstance(section, Mapping):
            raise ValueError(f"config section {section_name!r} must be a mapping")
        for key, value in section.items():
            if key in {"checkpoint_dir", "dataset_root", "output_dir"} and value is not None:
                defaults[key] = pathlib.Path(value)
            elif key == "rename_map" and isinstance(value, Mapping):
                defaults[key] = json.dumps(dict(value))
            else:
                defaults[key] = value
    return defaults


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compare sign-flipped reverse FireFlow velocities with independently recomputed "
            "base-to-data velocities and compare their CNF log-likelihoods."
        )
    )
    parser.add_argument("--config", type=pathlib.Path, default=DEFAULT_CONFIG)
    add_eval_data_arguments(parser, required=False)
    parser.set_defaults(
        checkpoint_dir=ROOT / "checkpoints" / "tactile_test_L32_4k",
        dataset_repo_id="chaoyi/tactile_test_05",
    )
    parser.add_argument("--episode-index", type=int, default=100)
    parser.add_argument("--frame", type=int, default=0)
    parser.add_argument("--steps", nargs="+", type=int, default=(10, 20, 50, 100, 200))
    parser.add_argument("--hutchinson-samples", type=int, default=DEFAULT_HUTCHINSON_SAMPLES)
    parser.add_argument("--hutchinson-seed", type=int, default=DEFAULT_HUTCHINSON_SEED)
    parser.add_argument("--output-dir", type=pathlib.Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--keep-jax-cache-between-runs",
        action="store_true",
        help="Keep all step-specific compiled runners (faster, but uses more memory).",
    )
    parser.add_argument("--single-process", action="store_true", help=argparse.SUPPRESS)
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    argv_list = list(argv) if argv is not None else None
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", type=pathlib.Path, default=DEFAULT_CONFIG)
    pre_args, _ = pre_parser.parse_known_args(argv_list)
    parser = _build_parser()
    parser.set_defaults(**_flatten_config(_load_config(pre_args.config)))
    return parser.parse_args(argv_list)


def _append_option(command: list[str], name: str, value: Any | None) -> None:
    if value is not None:
        command.extend((name, str(value)))


def _isolated_worker_command(
    args: argparse.Namespace,
    *,
    num_steps: int,
    output_dir: pathlib.Path,
) -> list[str]:
    """Build a fully resolved single-k command, independent of YAML defaults."""

    command = [
        sys.executable,
        "-u",
        "-m",
        "modalities_eval.reverse_forward_logp_compare",
        "--config",
        str(args.config),
        "--checkpoint-dir",
        str(args.checkpoint_dir),
        "--dataset-repo-id",
        str(args.dataset_repo_id),
        "--episode-index",
        str(args.episode_index),
        "--frame",
        str(args.frame),
        "--steps",
        str(num_steps),
        "--hutchinson-samples",
        str(args.hutchinson_samples),
        "--hutchinson-seed",
        str(args.hutchinson_seed),
        "--output-dir",
        str(output_dir),
        "--single-process",
    ]
    _append_option(command, "--dataset-root", args.dataset_root)
    _append_option(command, "--dataset-revision", args.dataset_revision)
    _append_option(command, "--action-key", args.action_key)
    _append_option(command, "--rename-map", args.rename_map)
    if args.allow_download:
        command.append("--allow-download")
    return command


def _read_csv(path: pathlib.Path) -> list[dict[str, str]]:
    with path.open(newline="") as file:
        return list(csv.DictReader(file))


def _run_isolated_steps(
    args: argparse.Namespace,
    *,
    steps: Sequence[int],
    output_dir: pathlib.Path,
) -> None:
    """Run each static JAX scan in a fresh process, then merge its CSV output."""

    summary_rows: list[dict[str, str]] = []
    all_profile_rows: list[dict[str, str]] = []
    with tempfile.TemporaryDirectory(prefix="reverse_forward_logp_") as temporary_dir:
        temporary_root = pathlib.Path(temporary_dir)
        for num_steps in steps:
            worker_dir = temporary_root / f"k{num_steps}"
            subprocess.run(
                _isolated_worker_command(
                    args,
                    num_steps=num_steps,
                    output_dir=worker_dir,
                ),
                check=True,
            )
            summary_rows.extend(_read_csv(worker_dir / "summary.csv"))
            all_profile_rows.extend(_read_csv(worker_dir / "field_profile.csv"))

    summary_path = _write_csv(
        summary_rows,
        output_dir / "summary.csv",
        fieldnames=SUMMARY_FIELDS,
    )
    profile_path = _write_csv(
        all_profile_rows,
        output_dir / "field_profile.csv",
        fieldnames=PROFILE_FIELDS,
    )
    plot_path = save_comparison_plot(summary_rows, output_dir / "comparison.png")
    print(f"combined_summary_csv={summary_path}")
    print(f"combined_field_profile_csv={profile_path}")
    print(f"combined_plot={plot_path}")


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if not args.steps or any(step <= 0 for step in args.steps):
        raise ValueError(f"all --steps values must be positive, got {args.steps}")
    if args.hutchinson_samples <= 0:
        raise ValueError(
            f"--hutchinson-samples must be positive, got {args.hutchinson_samples}"
        )

    steps = tuple(sorted(set(int(step) for step in args.steps)))
    output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if (
        len(steps) > 1
        and not args.single_process
        and not args.keep_jax_cache_between_runs
    ):
        _run_isolated_steps(args, steps=steps, output_dir=output_dir)
        return

    model = load_model_from_args(args)
    episode = load_episode(model, args.episode_index, frame_indices=(args.frame,))
    observation = episode.observations[0]
    reference_actions = episode.actions[0]

    print(
        f"episode={args.episode_index} frame={episode.frames[0]} "
        f"dataset_index={episode.indices[0]} solver=fireflow"
    )
    print(
        f"steps={steps} hutchinson_samples={args.hutchinson_samples} "
        f"hutchinson_seed={args.hutchinson_seed}"
    )
    print(
        "steps,field_relative_l2,field_cosine_similarity,reconstruction_rmse,"
        "reverse_reused_log_p,forward_recomputed_log_p,log_p_abs_error"
    )

    summary_rows: list[dict[str, float | int]] = []
    all_profile_rows: list[dict[str, float | int]] = []
    for num_steps in steps:
        result = compare_model_reverse_forward(
            model,
            observation,
            reference_actions,
            num_steps=num_steps,
            hutchinson_samples=args.hutchinson_samples,
            hutchinson_seed=args.hutchinson_seed,
        )
        row = result_row(result, num_steps=num_steps)
        summary_rows.append(row)
        all_profile_rows.extend(profile_rows(result, num_steps=num_steps))
        print(
            f"{num_steps},{row['field_relative_l2']:.9g},"
            f"{row['field_cosine_similarity']:.9g},{row['reconstruction_rmse']:.9g},"
            f"{row['reverse_reused_log_p']:.9g},{row['forward_recomputed_log_p']:.9g},"
            f"{row['log_p_abs_error']:.9g}"
        )
        if not args.keep_jax_cache_between_runs:
            _MODEL_RUN_CACHE.clear()
            jax.clear_caches()
            gc.collect()

    summary_path = _write_csv(
        summary_rows,
        output_dir / "summary.csv",
        fieldnames=SUMMARY_FIELDS,
    )
    profile_path = _write_csv(
        all_profile_rows,
        output_dir / "field_profile.csv",
        fieldnames=PROFILE_FIELDS,
    )
    plot_path = save_comparison_plot(summary_rows, output_dir / "comparison.png")
    print(f"summary_csv={summary_path}")
    print(f"field_profile_csv={profile_path}")
    print(f"plot={plot_path}")


if __name__ == "__main__":
    main()
