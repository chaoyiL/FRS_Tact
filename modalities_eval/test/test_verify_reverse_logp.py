from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from modalities_eval.loglike_evaluate import ODE_SOLVER_EULER, ODE_SOLVER_FIREFLOW
from modalities_eval.verify_reverse_logp import integrate_round_trip, result_row


def _expand_time(t: jax.Array, ndim: int) -> jax.Array:
    return t.reshape((t.shape[0],) + (1,) * (ndim - 1))


def _nonlinear_velocity(x: jax.Array, t: jax.Array) -> jax.Array:
    return 0.18 * jnp.square(x) + 0.07 * _expand_time(t, x.ndim)


def _nonlinear_velocity_trace(
    x: jax.Array,
    t: jax.Array,
    rng_key: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    del rng_key
    event_axes = tuple(range(1, x.ndim))
    return _nonlinear_velocity(x, t), jnp.sum(0.36 * x, axis=event_axes)


@pytest.mark.parametrize("ode_solver", [ODE_SOLVER_EULER, ODE_SOLVER_FIREFLOW])
def test_round_trip_records_complete_independent_histories(ode_solver: str) -> None:
    x_data = jnp.asarray([[[0.7, -0.2], [0.1, 0.4]]], dtype=jnp.float32)
    result = integrate_round_trip(
        _nonlinear_velocity,
        _nonlinear_velocity_trace,
        x_data,
        num_steps=8,
        rng_key=jax.random.PRNGKey(0),
        ode_solver=ode_solver,
    )

    assert result.forward_states.shape == (9, *x_data.shape)
    assert result.reverse_states.shape == (9, *x_data.shape)
    assert result.forward_divergences.shape == (8, x_data.shape[0])
    assert result.reverse_divergences.shape == (8, x_data.shape[0])
    np.testing.assert_allclose(
        result.inferred_return_divergence_integral,
        -result.forward_divergence_integral,
        rtol=0,
        atol=0,
    )
    assert float(result.divergence_closure_abs_error[0]) > 0.0


def test_euler_errors_converge_for_nonlinear_flow() -> None:
    x_data = jnp.asarray([[[0.7, -0.2], [0.1, 0.4]]], dtype=jnp.float32)

    def run(num_steps: int):
        return integrate_round_trip(
            _nonlinear_velocity,
            _nonlinear_velocity_trace,
            x_data,
            num_steps=num_steps,
            rng_key=jax.random.PRNGKey(3),
            ode_solver=ODE_SOLVER_EULER,
        )

    coarse = run(10)
    fine = run(200)

    assert float(fine.reconstruction_mse[0]) < float(coarse.reconstruction_mse[0])
    assert float(fine.trajectory_max_abs_error[0]) < float(coarse.trajectory_max_abs_error[0])
    assert float(fine.divergence_closure_abs_error[0]) < float(
        coarse.divergence_closure_abs_error[0]
    )
    assert float(fine.log_p_abs_error[0]) < float(coarse.log_p_abs_error[0])


def test_log_p_gap_equals_divergence_closure_gap() -> None:
    x_data = jnp.asarray([[[0.3, 0.6]]], dtype=jnp.float32)
    result = integrate_round_trip(
        _nonlinear_velocity,
        _nonlinear_velocity_trace,
        x_data,
        num_steps=30,
        rng_key=jax.random.PRNGKey(5),
        ode_solver=ODE_SOLVER_FIREFLOW,
    )

    np.testing.assert_allclose(
        result.log_p_abs_error,
        result.divergence_closure_abs_error,
        rtol=1e-5,
        atol=1e-6,
    )
    row = result_row(result, num_steps=30)
    assert row["steps"] == 30
    assert row["reconstruction_rmse"] >= 0.0
    assert row["log_p_abs_error"] >= 0.0
