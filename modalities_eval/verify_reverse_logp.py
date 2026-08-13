from __future__ import annotations

import argparse
import csv
import pathlib
from collections.abc import Callable, Sequence
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from modalities_eval.loglike_evaluate import (
    DEFAULT_HUTCHINSON_SAMPLES,
    DEFAULT_HUTCHINSON_SEED,
    ODE_SOLVER_EULER,
    ODE_SOLVER_FIREFLOW,
    ODE_SOLVERS,
    standard_normal_log_prob,
    velocity_and_hutchinson_trace,
)
from modalities_eval.utils import (
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

VelocityFn = Callable[[jax.Array, jax.Array], jax.Array]
VelocityTraceFn = Callable[[jax.Array, jax.Array, jax.Array], tuple[jax.Array, jax.Array]]


class RoundTripResult(NamedTuple):
    """Independent data-to-base and base-to-data integration results."""

    x_base: jax.Array
    x_reconstructed: jax.Array
    forward_states: jax.Array
    reverse_states: jax.Array
    forward_divergences: jax.Array
    reverse_divergences: jax.Array
    log_p_base: jax.Array
    forward_divergence_integral: jax.Array
    inferred_return_divergence_integral: jax.Array
    actual_return_divergence_integral: jax.Array
    direct_log_p: jax.Array
    roundtrip_log_p: jax.Array
    reconstruction_mse: jax.Array
    trajectory_max_abs_error: jax.Array
    pointwise_divergence_max_abs_error: jax.Array
    divergence_closure_abs_error: jax.Array
    log_p_abs_error: jax.Array


def _euler_trajectory(
    *,
    x: jax.Array,
    start_time: float,
    dt: jax.Array,
    interval_indices: jax.Array,
    rng_key: jax.Array,
    velocity_trace_fn: VelocityTraceFn,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Euler integration that retains states and independently evaluated divergences."""

    batch_size = x.shape[0]
    t = jnp.full((batch_size,), start_time, dtype=jnp.float32)

    def body(carry, interval_index):
        x_t, t_t, divergence_integral = carry
        velocity, divergence = velocity_trace_fn(
            x_t,
            t_t,
            jax.random.fold_in(rng_key, interval_index),
        )
        next_x = x_t + dt * velocity
        return (
            next_x,
            t_t + dt,
            divergence_integral + dt * divergence,
        ), (next_x, divergence)

    integral0 = jnp.zeros((batch_size,), dtype=jnp.float32)
    (_, _, divergence_integral), (updated_states, divergences) = jax.lax.scan(
        body,
        (x, t, integral0),
        interval_indices,
    )
    states = jnp.concatenate((x[None, ...], updated_states), axis=0)
    return states, divergences, divergence_integral


def _fireflow_trajectory(
    *,
    x: jax.Array,
    start_time: float,
    dt: jax.Array,
    interval_indices: jax.Array,
    rng_key: jax.Array,
    velocity_fn: VelocityFn,
    velocity_trace_fn: VelocityTraceFn,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """FireFlow modified-midpoint integration with a complete state history."""

    batch_size = x.shape[0]
    t = jnp.full((batch_size,), start_time, dtype=jnp.float32)
    first_interval = interval_indices[0]

    v0 = velocity_fn(x, t)
    x_mid = x + 0.5 * dt * v0
    v_mid, divergence_mid = velocity_trace_fn(
        x_mid,
        t + 0.5 * dt,
        jax.random.fold_in(rng_key, first_interval),
    )
    first_state = x + dt * v_mid
    first_integral = dt * divergence_mid

    def body(carry, interval_index):
        x_t, t_t, previous_mid_velocity, divergence_integral = carry
        x_mid_t = x_t + 0.5 * dt * previous_mid_velocity
        v_mid_t, divergence_mid_t = velocity_trace_fn(
            x_mid_t,
            t_t + 0.5 * dt,
            jax.random.fold_in(rng_key, interval_index),
        )
        next_x = x_t + dt * v_mid_t
        return (
            next_x,
            t_t + dt,
            v_mid_t,
            divergence_integral + dt * divergence_mid_t,
        ), (next_x, divergence_mid_t)

    (_, _, _, divergence_integral), (remaining_states, remaining_divergences) = jax.lax.scan(
        body,
        (first_state, t + dt, v_mid, first_integral),
        interval_indices[1:],
    )
    states = jnp.concatenate(
        (x[None, ...], first_state[None, ...], remaining_states),
        axis=0,
    )
    divergences = jnp.concatenate(
        (divergence_mid[None, ...], remaining_divergences),
        axis=0,
    )
    return states, divergences, divergence_integral


def integrate_round_trip(
    velocity_fn: VelocityFn,
    velocity_trace_fn: VelocityTraceFn,
    x_data: jax.Array,
    *,
    num_steps: int,
    rng_key: jax.Array,
    ode_solver: str = ODE_SOLVER_EULER,
) -> RoundTripResult:
    """Compare direct likelihood transport with an independently evaluated return trip.

    The forward pass integrates model time from 0 to 1.  The return pass starts at
    the resulting base point and independently re-evaluates both velocity and
    divergence while model time runs from 1 to 0.  The same Hutchinson probe is
    assigned to the corresponding time interval in both directions.
    """

    if num_steps <= 0:
        raise ValueError(f"num_steps must be positive, got {num_steps}")
    if ode_solver not in ODE_SOLVERS:
        raise ValueError(f"ode_solver must be one of {ODE_SOLVERS}, got {ode_solver!r}")

    x_data = jnp.asarray(x_data, dtype=jnp.float32)
    if x_data.ndim < 2:
        raise ValueError(f"x_data must include batch and event dimensions, got {x_data.shape}")

    dt = jnp.asarray(1.0 / num_steps, dtype=jnp.float32)
    forward_intervals = jnp.arange(num_steps, dtype=jnp.int32)
    reverse_intervals = forward_intervals[::-1]

    if ode_solver == ODE_SOLVER_EULER:
        forward_states, forward_divergences, forward_integral = _euler_trajectory(
            x=x_data,
            start_time=0.0,
            dt=dt,
            interval_indices=forward_intervals,
            rng_key=rng_key,
            velocity_trace_fn=velocity_trace_fn,
        )
        reverse_states, reverse_divergences, actual_return_integral = _euler_trajectory(
            x=forward_states[-1],
            start_time=1.0,
            dt=-dt,
            interval_indices=reverse_intervals,
            rng_key=rng_key,
            velocity_trace_fn=velocity_trace_fn,
        )
    else:
        forward_states, forward_divergences, forward_integral = _fireflow_trajectory(
            x=x_data,
            start_time=0.0,
            dt=dt,
            interval_indices=forward_intervals,
            rng_key=rng_key,
            velocity_fn=velocity_fn,
            velocity_trace_fn=velocity_trace_fn,
        )
        reverse_states, reverse_divergences, actual_return_integral = _fireflow_trajectory(
            x=forward_states[-1],
            start_time=1.0,
            dt=-dt,
            interval_indices=reverse_intervals,
            rng_key=rng_key,
            velocity_fn=velocity_fn,
            velocity_trace_fn=velocity_trace_fn,
        )

    x_base = forward_states[-1]
    x_reconstructed = reverse_states[-1]
    log_p_base = standard_normal_log_prob(x_base)

    # Reparameterizing the return trip with increasing time gives w(x,s)=-v(x,1-s),
    # so its inferred divergence integral is the negative of the forward integral.
    inferred_return_integral = -forward_integral
    direct_log_p = log_p_base + forward_integral
    roundtrip_log_p = log_p_base - actual_return_integral

    event_axes = tuple(range(1, x_data.ndim))
    history_event_axes = tuple(range(2, forward_states.ndim))
    reconstruction_mse = jnp.mean(jnp.square(x_reconstructed - x_data), axis=event_axes)
    trajectory_error = jnp.max(
        jnp.abs(reverse_states - forward_states[::-1]),
        axis=(0, *history_event_axes),
    )
    # Both arrays below contain div(v).  Reverse order aligns corresponding intervals;
    # negating both to express div(w) would leave the same absolute difference.
    pointwise_divergence_error = jnp.max(
        jnp.abs(reverse_divergences - forward_divergences[::-1]),
        axis=0,
    )
    divergence_closure_error = jnp.abs(forward_integral + actual_return_integral)
    log_p_error = jnp.abs(roundtrip_log_p - direct_log_p)

    return RoundTripResult(
        x_base=x_base,
        x_reconstructed=x_reconstructed,
        forward_states=forward_states,
        reverse_states=reverse_states,
        forward_divergences=forward_divergences,
        reverse_divergences=reverse_divergences,
        log_p_base=log_p_base,
        forward_divergence_integral=forward_integral,
        inferred_return_divergence_integral=inferred_return_integral,
        actual_return_divergence_integral=actual_return_integral,
        direct_log_p=direct_log_p,
        roundtrip_log_p=roundtrip_log_p,
        reconstruction_mse=reconstruction_mse,
        trajectory_max_abs_error=trajectory_error,
        pointwise_divergence_max_abs_error=pointwise_divergence_error,
        divergence_closure_abs_error=divergence_closure_error,
        log_p_abs_error=log_p_error,
    )


_MODEL_RUN_CACHE: dict[tuple[int, int, int, int, str], Callable] = {}


def _get_model_roundtrip_runner(
    model: SmolVLAEvalModel,
    *,
    num_steps: int,
    hutchinson_samples: int,
    hutchinson_seed: int,
    ode_solver: str,
):
    cache_key = (id(model), num_steps, hutchinson_samples, hutchinson_seed, ode_solver)
    cached = _MODEL_RUN_CACHE.get(cache_key)
    if cached is not None:
        return cached

    rng_key = jax.random.PRNGKey(hutchinson_seed)

    @jax.jit
    def run(context: VelocityContext, x_data: jax.Array) -> RoundTripResult:
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

        return integrate_round_trip(
            velocity_fn,
            velocity_trace_fn,
            x_data,
            num_steps=num_steps,
            rng_key=rng_key,
            ode_solver=ode_solver,
        )

    _MODEL_RUN_CACHE[cache_key] = run
    return run


def verify_model_round_trip(
    model: SmolVLAEvalModel,
    observation: EvalObservation,
    reference_actions: jax.Array,
    *,
    num_steps: int,
    hutchinson_samples: int = DEFAULT_HUTCHINSON_SAMPLES,
    hutchinson_seed: int = DEFAULT_HUTCHINSON_SEED,
    ode_solver: str = ODE_SOLVER_EULER,
) -> RoundTripResult:
    """Run one checkpoint-backed round-trip verification for a normalized GT action."""

    if hutchinson_samples <= 0:
        raise ValueError(f"hutchinson_samples must be positive, got {hutchinson_samples}")
    batched_observation = _add_batch_dim(observation)
    context = create_velocity_context(model, batched_observation)
    actions = jnp.asarray(reference_actions, dtype=jnp.float32)
    if actions.ndim == 2:
        actions = actions[None, ...]
    runner = _get_model_roundtrip_runner(
        model,
        num_steps=num_steps,
        hutchinson_samples=hutchinson_samples,
        hutchinson_seed=hutchinson_seed,
        ode_solver=ode_solver,
    )
    return runner(context, actions)


CSV_FIELDS = (
    "steps",
    "log_p_base",
    "forward_divergence_integral",
    "inferred_return_divergence_integral",
    "actual_return_divergence_integral",
    "direct_log_p",
    "roundtrip_log_p",
    "reconstruction_rmse",
    "trajectory_max_abs_error",
    "pointwise_divergence_max_abs_error",
    "divergence_closure_abs_error",
    "log_p_abs_error",
)


def _first_scalar(value: jax.Array) -> float:
    return float(np.asarray(jax.device_get(value)).reshape(-1)[0])


def result_row(result: RoundTripResult, *, num_steps: int) -> dict[str, float | int]:
    return {
        "steps": num_steps,
        "log_p_base": _first_scalar(result.log_p_base),
        "forward_divergence_integral": _first_scalar(result.forward_divergence_integral),
        "inferred_return_divergence_integral": _first_scalar(
            result.inferred_return_divergence_integral
        ),
        "actual_return_divergence_integral": _first_scalar(
            result.actual_return_divergence_integral
        ),
        "direct_log_p": _first_scalar(result.direct_log_p),
        "roundtrip_log_p": _first_scalar(result.roundtrip_log_p),
        "reconstruction_rmse": _first_scalar(jnp.sqrt(result.reconstruction_mse)),
        "trajectory_max_abs_error": _first_scalar(result.trajectory_max_abs_error),
        "pointwise_divergence_max_abs_error": _first_scalar(
            result.pointwise_divergence_max_abs_error
        ),
        "divergence_closure_abs_error": _first_scalar(result.divergence_closure_abs_error),
        "log_p_abs_error": _first_scalar(result.log_p_abs_error),
    }


def save_rows(rows: Sequence[dict[str, float | int]], output_path: pathlib.Path) -> pathlib.Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return output_path


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Validate direct GT-to-base logP against an independent base-to-data return integration."
        )
    )
    add_eval_data_arguments(parser)
    parser.add_argument("--episode-index", type=int, default=0)
    parser.add_argument("--frame", type=int, default=0)
    parser.add_argument(
        "--steps",
        type=int,
        nargs="+",
        default=(10, 20, 50, 100),
        help="Integration step counts used for the convergence sweep.",
    )
    parser.add_argument(
        "--ode-solver",
        choices=ODE_SOLVERS,
        default=ODE_SOLVER_FIREFLOW,
    )
    parser.add_argument(
        "--hutchinson-samples",
        type=int,
        default=DEFAULT_HUTCHINSON_SAMPLES,
        help="Use 16 or more probes for a lower-noise final check if memory permits.",
    )
    parser.add_argument("--hutchinson-seed", type=int, default=DEFAULT_HUTCHINSON_SEED)
    parser.add_argument("--output", type=pathlib.Path)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)
    if any(steps <= 0 for steps in args.steps):
        raise ValueError(f"all --steps values must be positive, got {args.steps}")
    if args.hutchinson_samples <= 0:
        raise ValueError(
            f"--hutchinson-samples must be positive, got {args.hutchinson_samples}"
        )

    model = load_model_from_args(args)
    episode = load_episode(
        model,
        args.episode_index,
        frame_indices=(args.frame,),
    )
    observation = episode.observations[0]
    reference_actions = episode.actions[0]

    print(
        f"episode={args.episode_index} frame={episode.frames[0]} "
        f"dataset_index={episode.indices[0]} solver={args.ode_solver}"
    )
    print(
        f"hutchinson_samples={args.hutchinson_samples} "
        f"hutchinson_seed={args.hutchinson_seed}"
    )
    print(
        "steps,reconstruction_rmse,trajectory_max_abs_error,"
        "divergence_closure_abs_error,log_p_abs_error"
    )

    rows = []
    for num_steps in args.steps:
        result = verify_model_round_trip(
            model,
            observation,
            reference_actions,
            num_steps=num_steps,
            hutchinson_samples=args.hutchinson_samples,
            hutchinson_seed=args.hutchinson_seed,
            ode_solver=args.ode_solver,
        )
        row = result_row(result, num_steps=num_steps)
        rows.append(row)
        print(
            f"{num_steps},{row['reconstruction_rmse']:.9g},"
            f"{row['trajectory_max_abs_error']:.9g},"
            f"{row['divergence_closure_abs_error']:.9g},"
            f"{row['log_p_abs_error']:.9g}"
        )

    output_path = args.output or pathlib.Path(
        f"eval_outputs/loglike/reverse_logp_validation_episode_{args.episode_index}_"
        f"frame_{args.frame}.csv"
    )
    save_rows(rows, output_path)
    print(f"results_csv={output_path}")
    print(
        "Interpretation: inferred_return_divergence_integral is the sign-flipped method; "
        "actual_return_divergence_integral comes from an independent return solve."
    )


if __name__ == "__main__":
    main()
