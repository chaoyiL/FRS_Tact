from __future__ import annotations

import numpy as np
import pytest
import jax.numpy as jnp
from flax import nnx

from decode_tests.condition_swap import classify_outcome
from decode_tests.condition_swap import decoder_directional_derivatives
from decode_tests.condition_swap import jacobian_null_ratios
from decode_tests.condition_swap import select_experiment_indices
from utils.model import DecoderConfig
from utils.model import SelfAttentionFlowDecoder
from utils.model import decode_actions


def test_selection_keeps_original_condition_first_and_prefers_distinct_episodes():
    candidates = np.arange(12, dtype=np.int64)
    episodes = np.repeat(np.arange(6, dtype=np.int64), 2)

    anchors, conditions = select_experiment_indices(
        candidates,
        episodes,
        num_actions=3,
        num_conditions=4,
        seed=7,
    )

    np.testing.assert_array_equal(conditions[:, 0], anchors)
    assert conditions.shape == (3, 4)
    for row in conditions:
        assert len(set(row.tolist())) == len(row)
        assert len(set(episodes[row].tolist())) == len(row)


@pytest.mark.parametrize(
    ("delta", "degradation", "expected"),
    [
        (1e-4, 1.0, "condition_ignored"),
        (0.2, 0.005, "decoder_redundant_or_null_direction"),
        (0.2, 0.02, "action_code_damaged"),
    ],
)
def test_classify_outcome_matches_three_experimental_hypotheses(
    delta: float,
    degradation: float,
    expected: str,
):
    label, explanation = classify_outcome(
        mean_latent_delta_rms=delta,
        reconstruction_degradation_mse=degradation,
        latent_rms_epsilon=1e-3,
        reconstruction_degradation_epsilon=1e-2,
    )

    assert label == expected
    assert explanation


def test_jacobian_null_ratio_uses_l2_norms_per_sample():
    latent_directions = jnp.asarray([[[3.0, 4.0]], [[0.0, 2.0]]])
    decoder_jvps = jnp.asarray([[[0.3, 0.4]], [[0.0, 1.0]]])

    ratios = jacobian_null_ratios(decoder_jvps, latent_directions)

    np.testing.assert_allclose(np.asarray(ratios), np.asarray([0.1, 0.5]), rtol=1e-6)


@pytest.mark.parametrize("solver", ["euler", "fireflow"])
def test_decoder_jvp_matches_centered_finite_difference(solver: str):
    model = SelfAttentionFlowDecoder(
        DecoderConfig(
            action_dim=2,
            action_horizon=3,
            model_dim=8,
            depth=1,
            num_heads=2,
            mlp_ratio=2,
        ),
        rngs=nnx.Rngs(5),
    )
    base = jnp.asarray(
        [
            [[0.1, -0.2], [0.3, 0.4], [-0.1, 0.2]],
            [[-0.4, 0.1], [0.2, -0.3], [0.5, 0.2]],
        ],
        dtype=jnp.float32,
    )
    direction = jnp.asarray(
        [
            [[0.2, 0.1], [-0.1, 0.3], [0.4, -0.2]],
            [[-0.1, 0.4], [0.3, 0.2], [-0.2, 0.1]],
        ],
        dtype=jnp.float32,
    )

    jvp = decoder_directional_derivatives(
        model,
        base,
        direction,
        num_steps=3,
        solver=solver,
    )
    # A moderate epsilon is required for a stable float32 finite difference;
    # smaller values are dominated by cancellation through repeated decoder steps.
    epsilon = 1e-1
    plus = decode_actions(model, base + epsilon * direction, num_steps=3, solver=solver)
    minus = decode_actions(model, base - epsilon * direction, num_steps=3, solver=solver)
    finite_difference = (plus - minus) / (2.0 * epsilon)

    np.testing.assert_allclose(
        np.asarray(jvp),
        np.asarray(finite_difference),
        rtol=3e-2,
        atol=4e-3,
    )
