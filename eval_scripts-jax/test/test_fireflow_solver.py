from __future__ import annotations

import os
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
EVAL_SCRIPTS = ROOT / "eval_scripts"
POLICY_SRC = ROOT / "policy" / "src"
for path in (EVAL_SCRIPTS, POLICY_SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from loglike_evaluate import ODE_SOLVER_EULER
from loglike_evaluate import ODE_SOLVER_FIREFLOW
from loglike_evaluate import _add_batch_dim
from loglike_evaluate import _run_euler_likelihood_scan
from loglike_evaluate import _run_fireflow_likelihood_scan
from loglike_evaluate import create_velocity_context
from loglike_evaluate import integrate_to_base_log_likelihood
from loglike_evaluate import load_episode
from loglike_evaluate import load_model
from loglike_evaluate import predict_velocity_with_context
from openpi.training import config as _config


def _expand_time(t: jax.Array, x_ndim: int) -> jax.Array:
    return t.reshape((t.shape[0],) + (1,) * (x_ndim - 1))


def _toy_velocity(x: jax.Array, t: jax.Array) -> jax.Array:
    return 0.25 * x + _expand_time(t, x.ndim)


def _toy_velocity_trace(
    x: jax.Array,
    t: jax.Array,
    rng_key: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    del rng_key
    event_size = int(np.prod(x.shape[1:]))
    divergence = jnp.full((x.shape[0],), 0.25 * event_size, dtype=jnp.float32)
    return _toy_velocity(x, t), divergence


def test_fireflow_preserves_input_shape() -> None:
    x_start = jnp.ones((2, 3, 4), dtype=jnp.float32)
    r_tot = jnp.zeros((x_start.shape[0],), dtype=jnp.float32)
    t = jnp.zeros((x_start.shape[0],), dtype=jnp.float32)
    num_steps = 5

    x_out, r_out, _ = _run_fireflow_likelihood_scan(
        x=x_start,
        r_tot=r_tot,
        t=t,
        step_indices=jnp.arange(num_steps, dtype=jnp.int32),
        dt=jnp.asarray(1.0 / num_steps, dtype=jnp.float32),
        rng_key=jax.random.PRNGKey(0),
        velocity_fn=_toy_velocity,
        velocity_trace_fn=_toy_velocity_trace,
    )

    assert x_out.shape == x_start.shape
    assert r_out.shape == r_tot.shape


@pytest.mark.parametrize("num_steps", [1, 2, 7])
def test_solver_nfe_counts(num_steps: int) -> None:
    x_start = jnp.ones((2, 3, 4), dtype=jnp.float32)
    r_tot = jnp.zeros((x_start.shape[0],), dtype=jnp.float32)
    t = jnp.zeros((x_start.shape[0],), dtype=jnp.float32)
    step_indices = jnp.arange(num_steps, dtype=jnp.int32)
    dt = jnp.asarray(1.0 / num_steps, dtype=jnp.float32)
    rng_key = jax.random.PRNGKey(0)

    _, _, euler_nfe = _run_euler_likelihood_scan(
        x=x_start,
        r_tot=r_tot,
        t=t,
        step_indices=step_indices,
        dt=dt,
        rng_key=rng_key,
        velocity_trace_fn=_toy_velocity_trace,
    )
    _, _, fireflow_nfe = _run_fireflow_likelihood_scan(
        x=x_start,
        r_tot=r_tot,
        t=t,
        step_indices=step_indices,
        dt=dt,
        rng_key=rng_key,
        velocity_fn=_toy_velocity,
        velocity_trace_fn=_toy_velocity_trace,
    )

    assert int(euler_nfe) == num_steps
    assert int(fireflow_nfe) == num_steps + 1


def _denoise_with_solver(model, observation, x_base: jax.Array, *, num_steps: int, ode_solver: str) -> jax.Array:
    x = jnp.asarray(x_base, dtype=jnp.float32)
    context = create_velocity_context(model, _add_batch_dim(observation))
    t = jnp.ones((x.shape[0],), dtype=jnp.float32)
    r_tot = jnp.zeros((x.shape[0],), dtype=jnp.float32)
    dt = jnp.asarray(-1.0 / num_steps, dtype=jnp.float32)
    step_indices = jnp.arange(num_steps, dtype=jnp.int32)

    def velocity_fn(x_arg: jax.Array, t_arg: jax.Array) -> jax.Array:
        return predict_velocity_with_context(model, context, x_arg, t_arg).astype(jnp.float32)

    def velocity_trace_fn(
        x_arg: jax.Array,
        t_arg: jax.Array,
        rng_key: jax.Array,
    ) -> tuple[jax.Array, jax.Array]:
        del rng_key
        return velocity_fn(x_arg, t_arg), jnp.zeros((x_arg.shape[0],), dtype=jnp.float32)

    if ode_solver == ODE_SOLVER_EULER:
        x_recon, _, _ = _run_euler_likelihood_scan(
            x=x,
            r_tot=r_tot,
            t=t,
            step_indices=step_indices,
            dt=dt,
            rng_key=jax.random.PRNGKey(0),
            velocity_trace_fn=velocity_trace_fn,
        )
        return x_recon

    x_recon, _, _ = _run_fireflow_likelihood_scan(
        x=x,
        r_tot=r_tot,
        t=t,
        step_indices=step_indices,
        dt=dt,
        rng_key=jax.random.PRNGKey(0),
        velocity_fn=velocity_fn,
        velocity_trace_fn=velocity_trace_fn,
    )
    return x_recon


@pytest.mark.manual
def test_fireflow_reconstruction_sanity_with_model() -> None:
    if os.environ.get("FIREFLOW_RECONSTRUCTION_TEST") != "1":
        pytest.skip("Set FIREFLOW_RECONSTRUCTION_TEST=1 to run checkpoint-backed reconstruction sanity.")

    config_name = os.environ.get("FIREFLOW_CONFIG_NAME", "pi05_bi_vitac")
    checkpoint_dir = os.environ.get("FIREFLOW_CHECKPOINT_DIR", str(ROOT / "checkpoints" / "11999"))
    episode_index = os.environ.get("FIREFLOW_EPISODE_INDEX", "10")
    frame = int(os.environ.get("FIREFLOW_FRAME", "0"))
    num_steps = int(os.environ.get("FIREFLOW_NUM_STEPS", "20"))

    train_config = _config.get_config(config_name)
    episode = load_episode(
        train_config,
        checkpoint_dir,
        episode_index,
        frame_indices=(frame,),
    )
    model = load_model(train_config, checkpoint_dir)
    observation = episode.observations[0]
    x_data = jnp.asarray(episode.actions[0], dtype=jnp.float32)

    euler_result = integrate_to_base_log_likelihood(
        model,
        observation,
        x_data,
        num_steps=num_steps,
        ode_solver=ODE_SOLVER_EULER,
    )
    fireflow_result = integrate_to_base_log_likelihood(
        model,
        observation,
        x_data,
        num_steps=num_steps,
        ode_solver=ODE_SOLVER_FIREFLOW,
    )

    euler_recon = _denoise_with_solver(
        model,
        observation,
        euler_result.x_base,
        num_steps=num_steps,
        ode_solver=ODE_SOLVER_EULER,
    )
    fireflow_recon = _denoise_with_solver(
        model,
        observation,
        fireflow_result.x_base,
        num_steps=num_steps,
        ode_solver=ODE_SOLVER_FIREFLOW,
    )

    x_data_batched = x_data[None, ...]
    euler_mse = jnp.mean(jnp.square(euler_recon - x_data_batched))
    fireflow_mse = jnp.mean(jnp.square(fireflow_recon - x_data_batched))

    assert bool(jnp.isfinite(euler_mse))
    assert bool(jnp.isfinite(fireflow_mse))
    assert float(fireflow_mse) <= max(float(euler_mse) * 5.0, 1e-4)
