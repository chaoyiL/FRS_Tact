from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from modalities_eval.loglike_evaluate import _run_fireflow_likelihood_scan, standard_normal_log_prob
from modalities_eval.reverse_forward_logp_compare import (
    compare_reverse_reuse_to_forward_recompute,
    profile_rows,
    result_row,
)


def _expand_time(t: jax.Array, ndim: int) -> jax.Array:
    return t.reshape((t.shape[0],) + (1,) * (ndim - 1))


def _nonlinear_velocity(x: jax.Array, t: jax.Array) -> jax.Array:
    return 0.12 * jnp.square(x) + 0.07 * _expand_time(t, x.ndim)


def _nonlinear_velocity_trace(
    x: jax.Array,
    t: jax.Array,
    rng_key: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    del rng_key
    event_axes = tuple(range(1, x.ndim))
    return _nonlinear_velocity(x, t), jnp.sum(0.24 * x, axis=event_axes)


def test_reverse_method_matches_existing_fireflow_likelihood() -> None:
    x_data = jnp.asarray([[[0.7, -0.2], [0.1, 0.4]]], dtype=jnp.float32)
    num_steps = 8
    result = compare_reverse_reuse_to_forward_recompute(
        _nonlinear_velocity,
        _nonlinear_velocity_trace,
        x_data,
        num_steps=num_steps,
        rng_key=jax.random.PRNGKey(0),
    )
    x_base, reverse_divergence_integral, _ = _run_fireflow_likelihood_scan(
        x=x_data,
        r_tot=jnp.zeros((x_data.shape[0],), dtype=jnp.float32),
        t=jnp.zeros((x_data.shape[0],), dtype=jnp.float32),
        step_indices=jnp.arange(num_steps, dtype=jnp.int32),
        dt=jnp.asarray(1.0 / num_steps, dtype=jnp.float32),
        rng_key=jax.random.PRNGKey(0),
        velocity_fn=_nonlinear_velocity,
        velocity_trace_fn=_nonlinear_velocity_trace,
    )

    np.testing.assert_allclose(result.x_base, x_base, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(
        result.reverse_reused_log_p,
        standard_normal_log_prob(x_base) + reverse_divergence_integral,
        rtol=1e-6,
        atol=1e-6,
    )


def test_constant_field_round_trip_is_exact() -> None:
    x_data = jnp.ones((2, 3, 4), dtype=jnp.float32)

    def velocity(x: jax.Array, t: jax.Array) -> jax.Array:
        del t
        return jnp.full_like(x, 0.25)

    def velocity_trace(
        x: jax.Array,
        t: jax.Array,
        rng_key: jax.Array,
    ) -> tuple[jax.Array, jax.Array]:
        del t, rng_key
        return jnp.full_like(x, 0.25), jnp.zeros((x.shape[0],), dtype=jnp.float32)

    result = compare_reverse_reuse_to_forward_recompute(
        velocity,
        velocity_trace,
        x_data,
        num_steps=7,
        rng_key=jax.random.PRNGKey(0),
    )
    np.testing.assert_allclose(result.x_reconstructed, x_data, atol=1e-6)
    np.testing.assert_allclose(result.reverse_reused_fields, result.forward_recomputed_fields)
    np.testing.assert_allclose(result.reverse_reused_log_p, result.forward_recomputed_log_p)
    np.testing.assert_allclose(result.field_cosine_similarity, 1.0, atol=1e-6)


def test_nonlinear_field_and_logp_gaps_converge() -> None:
    x_data = jnp.asarray([[[0.7, -0.2], [0.1, 0.4]]], dtype=jnp.float32)

    def run(num_steps: int):
        return compare_reverse_reuse_to_forward_recompute(
            _nonlinear_velocity,
            _nonlinear_velocity_trace,
            x_data,
            num_steps=num_steps,
            rng_key=jax.random.PRNGKey(3),
        )

    coarse = run(8)
    fine = run(200)
    assert float(fine.reconstruction_rmse[0]) < float(coarse.reconstruction_rmse[0])
    assert float(fine.field_relative_l2[0]) < float(coarse.field_relative_l2[0])
    assert float(fine.log_p_abs_error[0]) < float(coarse.log_p_abs_error[0])
    np.testing.assert_allclose(
        fine.log_p_abs_error,
        jnp.abs(
            fine.forward_recomputed_field_divergence_integral
            - fine.reverse_reused_field_divergence_integral
        ),
        rtol=1e-5,
        atol=1e-6,
    )


def test_result_and_profile_rows_have_expected_sizes() -> None:
    x_data = jnp.asarray([[[0.3, 0.6]]], dtype=jnp.float32)
    result = compare_reverse_reuse_to_forward_recompute(
        _nonlinear_velocity,
        _nonlinear_velocity_trace,
        x_data,
        num_steps=5,
        rng_key=jax.random.PRNGKey(5),
    )
    summary = result_row(result, num_steps=5)
    profile = profile_rows(result, num_steps=5)
    assert summary["steps"] == 5
    assert summary["event_size"] == 2
    assert len(profile) == 5
    assert profile[0]["generative_s_mid"] == 0.1
    assert np.isclose(profile[-1]["model_t_mid"], 0.1)
