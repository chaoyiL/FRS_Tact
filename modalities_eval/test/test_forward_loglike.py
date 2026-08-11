from __future__ import annotations

# ruff: noqa: E402
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
EVAL_SCRIPTS = ROOT / "modalities_eval"
if str(EVAL_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(EVAL_SCRIPTS))

from collections.abc import Callable

import jax
import jax.numpy as jnp
import numpy as np
from loglike_evaluate import (
    ODE_SOLVER_SLERPFLOW,
    ODE_SOLVERS,
    _run_euler_likelihood_scan,
    sample_shared_action_noise,
    standard_normal_log_prob,
)


def _constant_div_trace(
    divergence: float,
) -> Callable[[jax.Array, jax.Array, jax.Array], tuple[jax.Array, jax.Array]]:
    def velocity_trace_fn(
        x: jax.Array,
        t: jax.Array,
        rng_key: jax.Array,
    ) -> tuple[jax.Array, jax.Array]:
        del t, rng_key
        return jnp.zeros_like(x), jnp.full((x.shape[0],), divergence, dtype=jnp.float32)

    return velocity_trace_fn


def test_ode_solvers_include_slerpflow() -> None:
    assert ODE_SOLVER_SLERPFLOW in ODE_SOLVERS
    assert ODE_SOLVERS == ("euler", "fireflow", "slerpflow")


def test_forward_reverse_logp_formula_duality_euler() -> None:
    """With v=0 and constant div, forward and reverse logp formulas agree."""

    z = jax.random.normal(jax.random.PRNGKey(0), (2, 3, 4), dtype=jnp.float32)
    num_steps = 5
    divergence = 1.25
    step_indices = jnp.arange(num_steps, dtype=jnp.int32)
    r0 = jnp.zeros((z.shape[0],), dtype=jnp.float32)
    trace_fn = _constant_div_trace(divergence)

    x_fwd, r_fwd, _ = _run_euler_likelihood_scan(
        x=z,
        r_tot=r0,
        t=jnp.ones((z.shape[0],), dtype=jnp.float32),
        step_indices=step_indices,
        dt=jnp.asarray(-1.0 / num_steps, dtype=jnp.float32),
        rng_key=jax.random.PRNGKey(0),
        velocity_trace_fn=trace_fn,
    )
    logp_fwd = standard_normal_log_prob(z) - r_fwd

    x_rev, r_rev, _ = _run_euler_likelihood_scan(
        x=z,
        r_tot=r0,
        t=jnp.zeros((z.shape[0],), dtype=jnp.float32),
        step_indices=step_indices,
        dt=jnp.asarray(1.0 / num_steps, dtype=jnp.float32),
        rng_key=jax.random.PRNGKey(0),
        velocity_trace_fn=trace_fn,
    )
    logp_rev = standard_normal_log_prob(x_rev) + r_rev

    np.testing.assert_allclose(np.asarray(x_fwd), np.asarray(z), atol=1e-6)
    np.testing.assert_allclose(np.asarray(r_fwd), -divergence, atol=1e-5)
    np.testing.assert_allclose(np.asarray(r_rev), divergence, atol=1e-5)
    np.testing.assert_allclose(np.asarray(logp_fwd), np.asarray(logp_rev), atol=1e-5)


def test_sample_shared_action_noise_is_deterministic_per_index() -> None:
    class _FakeConfig:
        chunk_size = 5
        max_action_dim = 7

    class _FakeModel:
        config = _FakeConfig()

    model = _FakeModel()
    a = sample_shared_action_noise(model, batch_size=1, noise_seed=3, dataset_index=11)
    b = sample_shared_action_noise(model, batch_size=1, noise_seed=3, dataset_index=11)
    c = sample_shared_action_noise(model, batch_size=1, noise_seed=3, dataset_index=12)
    assert a.shape == (1, 5, 7)
    np.testing.assert_array_equal(np.asarray(a), np.asarray(b))
    assert not np.array_equal(np.asarray(a), np.asarray(c))
