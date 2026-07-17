from __future__ import annotations

# The eval directory is inserted below so these scripts also work when invoked by path.
# ruff: noqa: E402
import argparse
import csv
import dataclasses
import os
import pathlib
import sys
from collections.abc import Callable, Sequence
from typing import Any

ROOT = pathlib.Path(__file__).resolve().parents[1]
EVAL_DIR = pathlib.Path(__file__).resolve().parent
if str(EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(EVAL_DIR))

import jax
import jax.numpy as jnp
import numpy as np
from utils import (  # noqa: E402
    EvalObservation,
    SmolVLAEvalModel,
    VelocityContext,
    _add_batch_dim,
    _scalar,
    _stack_observations,
    ablate_modality_observation,
    add_eval_data_arguments,
    create_velocity_context,
    load_episode,
    load_model_from_args,
    predict_velocity_with_context,
)

DEFAULT_HUTCHINSON_SAMPLES = 1
DEFAULT_HUTCHINSON_SEED = 0
ODE_SOLVER_EULER = "euler"
ODE_SOLVER_FIREFLOW = "fireflow"
ODE_SOLVERS = (ODE_SOLVER_EULER, ODE_SOLVER_FIREFLOW)


@dataclasses.dataclass(frozen=True)
class LikelihoodIntegrationResult:
    """Result of integrating data actions to the base distribution."""

    x_base: jax.Array
    r_tot: jax.Array
    log_p_base: jax.Array
    log_likelihood: jax.Array


VelocityFn = Callable[[jax.Array, jax.Array], jax.Array]
VelocityTraceFn = Callable[[jax.Array, jax.Array, jax.Array], tuple[jax.Array, jax.Array]]

_RUN_SCAN_CACHE: dict[tuple[int, int, int, int, int, str], Any] = {}


def _validate_ode_solver(ode_solver: str) -> str:
    if ode_solver not in ODE_SOLVERS:
        raise ValueError(f"ode_solver must be one of {ODE_SOLVERS}, got {ode_solver!r}")
    return ode_solver


def _run_euler_likelihood_scan(
    *,
    x: jax.Array,
    r_tot: jax.Array,
    t: jax.Array,
    step_indices: jax.Array,
    dt: jax.Array,
    rng_key: jax.Array,
    velocity_trace_fn: VelocityTraceFn,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Integrate likelihood ODE with the existing Euler update."""

    def scan_body(carry, step_index):
        x_t, r_tot_t, t_t, nfe = carry
        step_rng_key = jax.random.fold_in(rng_key, step_index)
        velocity, divergence = velocity_trace_fn(x_t, t_t, step_rng_key)
        return (x_t + velocity * dt, r_tot_t + divergence * dt, t_t + dt, nfe + 1), None

    nfe0 = jnp.asarray(0, dtype=jnp.int32)
    (x, r_tot, _, nfe), _ = jax.lax.scan(scan_body, (x, r_tot, t, nfe0), step_indices)
    return x, r_tot, nfe


def _run_fireflow_likelihood_scan(
    *,
    x: jax.Array,
    r_tot: jax.Array,
    t: jax.Array,
    step_indices: jax.Array,
    dt: jax.Array,
    rng_key: jax.Array,
    velocity_fn: VelocityFn,
    velocity_trace_fn: VelocityTraceFn,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """FireFlow-style modified midpoint scan on the same data-to-base time grid."""

    first_step_index = step_indices[0]
    v0 = velocity_fn(x, t)
    nfe = jnp.asarray(1, dtype=jnp.int32)

    x_mid = x + 0.5 * dt * v0
    t_mid = t + 0.5 * dt
    v_mid_prev, divergence_mid = velocity_trace_fn(
        x_mid,
        t_mid,
        jax.random.fold_in(rng_key, first_step_index),
    )
    nfe = nfe + 1
    x = x + dt * v_mid_prev
    r_tot = r_tot + dt * divergence_mid
    t = t + dt

    def scan_body(carry, step_index):
        x_t, r_tot_t, t_t, v_mid_prev_t, nfe_t = carry
        x_mid_t = x_t + 0.5 * dt * v_mid_prev_t
        t_mid_t = t_t + 0.5 * dt
        v_mid, divergence_mid_t = velocity_trace_fn(
            x_mid_t,
            t_mid_t,
            jax.random.fold_in(rng_key, step_index),
        )
        return (
            x_t + dt * v_mid,
            r_tot_t + dt * divergence_mid_t,
            t_t + dt,
            v_mid,
            nfe_t + 1,
        ), None

    (x, r_tot, _, _, nfe), _ = jax.lax.scan(
        scan_body,
        (x, r_tot, t, v_mid_prev, nfe),
        step_indices[1:],
    )
    return x, r_tot, nfe


def _get_likelihood_scan(
    model: SmolVLAEvalModel,
    *,
    batch_size: int,
    num_steps: int,
    hutchinson_samples: int,
    hutchinson_seed: int,
    ode_solver: str,
):
    """Return a cached compiled scan so observations are not captured as constants."""

    ode_solver = _validate_ode_solver(ode_solver)
    cache_key = (id(model), batch_size, num_steps, hutchinson_samples, hutchinson_seed, ode_solver)
    if cache_key in _RUN_SCAN_CACHE:
        return _RUN_SCAN_CACHE[cache_key]

    dt = jnp.asarray(1.0 / num_steps, dtype=jnp.float32)
    rng_key = jax.random.PRNGKey(hutchinson_seed)

    @jax.jit
    def run_scan(context: VelocityContext, x: jax.Array, step_indices: jax.Array):
        def velocity_fn(x_arg: jax.Array, t_arg: jax.Array) -> jax.Array:
            return predict_velocity_with_context(model, context, x_arg, t_arg).astype(jnp.float32)

        def velocity_trace_fn(
            x_arg: jax.Array,
            t_arg: jax.Array,
            step_rng_key: jax.Array,
        ) -> tuple[jax.Array, jax.Array]:
            return velocity_and_hutchinson_trace(
                model,
                context,
                x_arg,
                t_arg,
                step_rng_key,
                num_samples=hutchinson_samples,
            )

        t = jnp.zeros((batch_size,), dtype=jnp.float32)
        r_tot = jnp.zeros((batch_size,), dtype=jnp.float32)
        if ode_solver == ODE_SOLVER_EULER:
            x, r_tot, _ = _run_euler_likelihood_scan(
                x=x,
                r_tot=r_tot,
                t=t,
                step_indices=step_indices,
                dt=dt,
                rng_key=rng_key,
                velocity_trace_fn=velocity_trace_fn,
            )
        else:
            x, r_tot, _ = _run_fireflow_likelihood_scan(
                x=x,
                r_tot=r_tot,
                t=t,
                step_indices=step_indices,
                dt=dt,
                rng_key=rng_key,
                velocity_fn=velocity_fn,
                velocity_trace_fn=velocity_trace_fn,
            )
        return x, r_tot

    _RUN_SCAN_CACHE[cache_key] = run_scan
    return run_scan


def velocity_and_hutchinson_trace(
    model: SmolVLAEvalModel,
    context: VelocityContext,
    x: jax.Array,
    t: jax.Array,
    rng_key: jax.Array,
    *,
    num_samples: int,
) -> tuple[jax.Array, jax.Array]:
    """Estimate div v(x,t,o) with Rademacher Hutchinson probes."""

    if num_samples <= 0:
        raise ValueError(f"num_samples must be positive, got {num_samples}")

    x = jnp.asarray(x, dtype=jnp.float32)

    def velocity_fn(x_arg: jax.Array) -> jax.Array:
        return predict_velocity_with_context(model, context, x_arg, t).astype(jnp.float32)

    event_axes = tuple(range(1, x.ndim))
    sample_keys = jax.random.split(rng_key, num_samples)

    def scan_body(carry, key: jax.Array):
        trace, velocity = carry
        probe = jax.random.rademacher(key, (1, *x.shape[1:]), dtype=jnp.float32)
        probe = jnp.broadcast_to(probe, x.shape)
        velocity, tangent_out = jax.jvp(velocity_fn, (x,), (probe,))
        trace_estimate = jnp.sum(probe * tangent_out, axis=event_axes)
        return (trace + trace_estimate, velocity), None

    trace0 = jnp.zeros((x.shape[0],), dtype=jnp.float32)
    velocity0 = jnp.zeros_like(x)
    (trace, velocity), _ = jax.lax.scan(scan_body, (trace0, velocity0), sample_keys)
    trace = trace / jnp.asarray(num_samples, dtype=jnp.float32)
    return velocity, trace


def standard_normal_log_prob(x: jax.Array) -> jax.Array:
    """Compute log p0(x) for a standard Gaussian base distribution."""

    event_dims = tuple(range(1, x.ndim))
    event_size = int(np.prod(x.shape[1:]))
    return -0.5 * (jnp.sum(jnp.square(x), axis=event_dims) + event_size * jnp.log(2.0 * jnp.pi))


def _select_batch_result(result: LikelihoodIntegrationResult, index: int) -> LikelihoodIntegrationResult:
    return LikelihoodIntegrationResult(
        x_base=result.x_base[index : index + 1],
        r_tot=result.r_tot[index : index + 1],
        log_p_base=result.log_p_base[index : index + 1],
        log_likelihood=result.log_likelihood[index : index + 1],
    )


def clear_likelihood_scan_cache() -> None:
    """Drop cached likelihood scan executables to reduce peak memory during k sweeps."""

    _RUN_SCAN_CACHE.clear()
    jax.clear_caches()


def integrate_batched_to_base_log_likelihood_with_context(
    model: SmolVLAEvalModel,
    context: VelocityContext,
    reference_actions: jax.Array,
    *,
    num_steps: int,
    hutchinson_samples: int = DEFAULT_HUTCHINSON_SAMPLES,
    hutchinson_seed: int = DEFAULT_HUTCHINSON_SEED,
    ode_solver: str = ODE_SOLVER_EULER,
) -> LikelihoodIntegrationResult:
    """Integrate actions to base noise while reusing a prebuilt prefix/KV context."""

    if num_steps <= 0:
        raise ValueError(f"num_steps must be positive, got {num_steps}")
    if hutchinson_samples <= 0:
        raise ValueError(f"hutchinson_samples must be positive, got {hutchinson_samples}")
    ode_solver = _validate_ode_solver(ode_solver)

    x = jnp.asarray(reference_actions, dtype=jnp.float32)
    if x.ndim == 2:
        x = x[None, ...]

    batch_size = x.shape[0]
    step_indices = jnp.arange(num_steps, dtype=jnp.int32)
    run_scan = _get_likelihood_scan(
        model,
        batch_size=batch_size,
        num_steps=num_steps,
        hutchinson_samples=hutchinson_samples,
        hutchinson_seed=hutchinson_seed,
        ode_solver=ode_solver,
    )
    x, r_tot = run_scan(context, x, step_indices)

    log_p_base = standard_normal_log_prob(x)
    log_likelihood = log_p_base + r_tot
    return LikelihoodIntegrationResult(
        x_base=x,
        r_tot=r_tot,
        log_p_base=log_p_base,
        log_likelihood=log_likelihood,
    )


def integrate_batched_to_base_log_likelihood(
    model: SmolVLAEvalModel,
    observation: EvalObservation,
    reference_actions: jax.Array,
    *,
    num_steps: int,
    hutchinson_samples: int = DEFAULT_HUTCHINSON_SAMPLES,
    hutchinson_seed: int = DEFAULT_HUTCHINSON_SEED,
    ode_solver: str = ODE_SOLVER_EULER,
) -> LikelihoodIntegrationResult:
    """Integrate SmolVLA actions at t=0 to standard-Gaussian noise at t=1."""

    if num_steps <= 0:
        raise ValueError(f"num_steps must be positive, got {num_steps}")
    if hutchinson_samples <= 0:
        raise ValueError(f"hutchinson_samples must be positive, got {hutchinson_samples}")
    ode_solver = _validate_ode_solver(ode_solver)

    x = jnp.asarray(reference_actions, dtype=jnp.float32)
    if x.ndim == 2:
        x = x[None, ...]

    context = create_velocity_context(model, observation)
    return integrate_batched_to_base_log_likelihood_with_context(
        model,
        context,
        x,
        num_steps=num_steps,
        hutchinson_samples=hutchinson_samples,
        hutchinson_seed=hutchinson_seed,
        ode_solver=ode_solver,
    )


def integrate_to_base_log_likelihood(
    model: SmolVLAEvalModel,
    observation: EvalObservation,
    reference_actions: jax.Array,
    *,
    num_steps: int,
    hutchinson_samples: int = DEFAULT_HUTCHINSON_SAMPLES,
    hutchinson_seed: int = DEFAULT_HUTCHINSON_SEED,
    ode_solver: str = ODE_SOLVER_EULER,
) -> LikelihoodIntegrationResult:
    return integrate_batched_to_base_log_likelihood(
        model,
        _add_batch_dim(observation),
        reference_actions,
        num_steps=num_steps,
        hutchinson_samples=hutchinson_samples,
        hutchinson_seed=hutchinson_seed,
        ode_solver=ode_solver,
    )


def integrate_to_base_log_likelihood_with_context(
    model: SmolVLAEvalModel,
    context: VelocityContext,
    reference_actions: jax.Array,
    *,
    num_steps: int,
    hutchinson_samples: int = DEFAULT_HUTCHINSON_SAMPLES,
    hutchinson_seed: int = DEFAULT_HUTCHINSON_SEED,
    ode_solver: str = ODE_SOLVER_EULER,
) -> LikelihoodIntegrationResult:
    return integrate_batched_to_base_log_likelihood_with_context(
        model,
        context,
        reference_actions,
        num_steps=num_steps,
        hutchinson_samples=hutchinson_samples,
        hutchinson_seed=hutchinson_seed,
        ode_solver=ode_solver,
    )


def compute_modality_contribution(
    model: SmolVLAEvalModel,
    observation: EvalObservation,
    reference_actions: jax.Array,
    *,
    modality: str,
    num_steps: int,
    hutchinson_samples: int = DEFAULT_HUTCHINSON_SAMPLES,
    hutchinson_seed: int = DEFAULT_HUTCHINSON_SEED,
    ode_solver: str = ODE_SOLVER_EULER,
    prompt: str | None = None,
    prompt_tokenizer: Any | None = None,
    state_in_prompt: bool = False,
) -> tuple[LikelihoodIntegrationResult, LikelihoodIntegrationResult, jax.Array]:
    ablated_observation = ablate_modality_observation(
        observation,
        modality=modality,
        prompt=prompt,
        prompt_tokenizer=prompt_tokenizer,
        state_in_prompt=state_in_prompt,
    )
    batched_observation = _stack_observations(observation, ablated_observation)
    batched_actions = jnp.stack(
        [
            jnp.asarray(reference_actions, dtype=jnp.float32),
            jnp.asarray(reference_actions, dtype=jnp.float32),
        ],
        axis=0,
    )
    batched_result = integrate_batched_to_base_log_likelihood(
        model,
        batched_observation,
        batched_actions,
        num_steps=num_steps,
        hutchinson_samples=hutchinson_samples,
        hutchinson_seed=hutchinson_seed,
        ode_solver=ode_solver,
    )
    original_result = _select_batch_result(batched_result, 0)
    ablated_result = _select_batch_result(batched_result, 1)
    contribution = original_result.log_likelihood - ablated_result.log_likelihood
    return original_result, ablated_result, contribution


def compute_episode_modality_contributions(
    model: SmolVLAEvalModel,
    frames: Sequence[int],
    dataset_indices: Sequence[int],
    observations: Sequence[EvalObservation],
    reference_actions: Sequence[jax.Array],
    prompts: Sequence[str | None],
    *,
    modality: str,
    num_steps: int,
    hutchinson_samples: int = DEFAULT_HUTCHINSON_SAMPLES,
    hutchinson_seed: int = DEFAULT_HUTCHINSON_SEED,
    ode_solver: str = ODE_SOLVER_EULER,
    prompt_tokenizer: Any | None = None,
    state_in_prompt: bool = False,
    eval_batch_size: int = 4,
) -> list[dict[str, float | int]]:
    """Compute contribution rows in frame chunks to reduce repeated compilation/dispatch."""

    if eval_batch_size <= 0:
        raise ValueError(f"eval_batch_size must be positive, got {eval_batch_size}")
    ode_solver = _validate_ode_solver(ode_solver)

    rows: list[dict[str, float | int]] = []
    total_frames = len(frames)
    for start in range(0, total_frames, eval_batch_size):
        stop = min(start + eval_batch_size, total_frames)
        chunk_observations = observations[start:stop]
        chunk_actions = reference_actions[start:stop]
        chunk_prompts = prompts[start:stop]

        batched_observations = []
        batched_actions = []
        for observation, actions, prompt in zip(
            chunk_observations, chunk_actions, chunk_prompts, strict=True
        ):
            ablated_observation = ablate_modality_observation(
                observation,
                modality=modality,
                prompt=prompt,
                prompt_tokenizer=prompt_tokenizer,
                state_in_prompt=state_in_prompt,
            )
            batched_observations.extend((observation, ablated_observation))
            action = jnp.asarray(actions, dtype=jnp.float32)
            batched_actions.extend((action, action))

        batched_result = integrate_batched_to_base_log_likelihood(
            model,
            _stack_observations(*batched_observations),
            jnp.stack(batched_actions, axis=0),
            num_steps=num_steps,
            hutchinson_samples=hutchinson_samples,
            hutchinson_seed=hutchinson_seed,
            ode_solver=ode_solver,
        )

        for chunk_offset, (frame, dataset_index) in enumerate(
            zip(frames[start:stop], dataset_indices[start:stop], strict=True)
        ):
            original_index = 2 * chunk_offset
            ablated_index = original_index + 1
            row = {
                "frame": int(frame),
                "dataset_index": int(dataset_index),
                "original_log_likelihood": _scalar(batched_result.log_likelihood[original_index]),
                "ablated_log_likelihood": _scalar(batched_result.log_likelihood[ablated_index]),
                "original_r_tot": _scalar(batched_result.r_tot[original_index]),
                "ablated_r_tot": _scalar(batched_result.r_tot[ablated_index]),
                "delta_logp": _scalar(
                    batched_result.log_p_base[original_index] - batched_result.log_p_base[ablated_index]
                ),
                "delta_r_tot": _scalar(
                    batched_result.r_tot[original_index] - batched_result.r_tot[ablated_index]
                ),
                "contribution": _scalar(
                    batched_result.log_likelihood[original_index]
                    - batched_result.log_likelihood[ablated_index]
                ),
            }
            rows.append(row)
            print(
                f"frame={row['frame']} dataset_index={row['dataset_index']} "
                f"original_log_likelihood={row['original_log_likelihood']:.6f} "
                f"ablated_log_likelihood={row['ablated_log_likelihood']:.6f} "
                f"delta_logp(x_base)={row['delta_logp']:.6f} "
                f"delta_r_tot={row['delta_r_tot']:.6f}"
            )

    return rows


def save_contribution_curve(
    rows: Sequence[dict[str, float | int]],
    *,
    output_dir: pathlib.Path,
    modality: str,
    episode_index: str,
) -> tuple[pathlib.Path, pathlib.Path | None]:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    os.environ.setdefault("XDG_CACHE_HOME", "/tmp")
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"{modality}_contribution_episode_{episode_index}.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "frame",
                "dataset_index",
                "original_log_likelihood",
                "ablated_log_likelihood",
                "original_r_tot",
                "ablated_r_tot",
                "delta_logp",
                "delta_r_tot",
                "contribution",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    try:
        import matplotlib

        matplotlib.use("Agg")
        from matplotlib import pyplot as plt
    except ImportError:
        return csv_path, None

    plot_path = output_dir / f"{modality}_contribution_components_episode_{episode_index}.png"
    frames = [row["frame"] for row in rows]
    curves = (
        ("contribution", f"{modality} contribution"),
        ("delta_logp", "delta_logp(x_base)"),
        ("delta_r_tot", "delta_r_tot"),
    )
    fig, axes = plt.subplots(len(curves), 1, figsize=(10, 9), sharex=True)
    fig.suptitle(f"{modality} contribution components over episode {episode_index}")
    for ax, (field, ylabel) in zip(axes, curves, strict=True):
        ax.plot(frames, [row[field] for row in rows], marker="o", linewidth=1.5)
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel("Episode frame")
    fig.tight_layout()
    fig.savefig(plot_path, dpi=160)
    plt.close(fig)
    return csv_path, plot_path


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Estimate SmolVLA JAX modality contribution by ablation.")
    add_eval_data_arguments(parser)
    parser.add_argument("--episode-index", type=int, default=0)
    parser.add_argument("--frame", type=int, default=0)
    parser.add_argument("--max-frames", type=int, default=1000)
    parser.add_argument("--sample-interval", type=int)
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
        "--remove-modality", choices=("vision", "tactile", "state", "language_prompt"), default="vision"
    )
    parser.add_argument("--output-dir", type=pathlib.Path, default=pathlib.Path("eval_outputs/loglike"))
    args = parser.parse_args(argv)

    if args.hutchinson_samples <= 0:
        raise ValueError(f"--hutchinson-samples must be positive, got {args.hutchinson_samples}.")
    if args.eval_batch_size <= 0:
        raise ValueError(f"--eval-batch-size must be positive, got {args.eval_batch_size}.")

    model = load_model_from_args(args)
    if args.sample_interval is None:
        episode = load_episode(
            model,
            args.episode_index,
            max_frames=args.max_frames,
            frame_indices=(args.frame,),
        )
    else:
        episode = load_episode(
            model,
            args.episode_index,
            start_frame=args.frame,
            sample_interval=args.sample_interval,
            max_frames=args.max_frames,
        )

    print(
        f"loaded episode={args.episode_index} frames={len(episode.indices)} dataset_indices={episode.indices[:5]}"
    )
    print(f"prompt={episode.prompts[0]!r}")
    print(f"ablated_modality={args.remove_modality}")
    print("ablation_method=input_mask_or_zero")
    print("divergence_method=hutchinson_rademacher_jvp")
    print(f"hutchinson_samples={args.hutchinson_samples}")
    print(f"hutchinson_seed={args.hutchinson_seed}")
    print(f"eval_batch_size={args.eval_batch_size}")
    print(f"ode_solver={args.ode_solver}")
    print(f"model_dtype={jax.tree.leaves(model.params)[0].dtype}")

    rows = compute_episode_modality_contributions(
        model,
        episode.frames,
        episode.indices,
        episode.observations,
        episode.actions,
        episode.prompts,
        modality=args.remove_modality,
        num_steps=args.num_steps,
        prompt_tokenizer=None,
        state_in_prompt=False,
        hutchinson_samples=args.hutchinson_samples,
        hutchinson_seed=args.hutchinson_seed,
        ode_solver=args.ode_solver,
        eval_batch_size=args.eval_batch_size,
    )

    csv_path, plot_path = save_contribution_curve(
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
