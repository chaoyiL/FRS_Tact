from __future__ import annotations

from types import SimpleNamespace

import jax
import numpy as np
import pytest

from deploy_pi05.policy import Pi05RemotePolicy


def _policy_with_recorded_rngs(recorded_rngs: list[np.ndarray]) -> Pi05RemotePolicy:
    policy = Pi05RemotePolicy.__new__(Pi05RemotePolicy)
    policy.config = SimpleNamespace(action_horizon=2, action_dim=3)
    policy._rng = None
    policy._rng_seed = None
    policy.prepare_observation = lambda observation, task: object()

    def sample(rng, observation, *, num_steps):
        recorded_rngs.append(np.asarray(jax.random.key_data(rng)))
        return np.zeros((1, 2, 3), dtype=np.float32)

    policy._sample_actions = sample
    return policy


def test_predict_action_chunk_advances_maniskill_rng_stream() -> None:
    recorded_rngs: list[np.ndarray] = []
    policy = _policy_with_recorded_rngs(recorded_rngs)

    policy.predict_action_chunk({}, "pick", seed=0, num_steps=10)
    policy.predict_action_chunk({}, "pick", seed=0, num_steps=10)

    rng = jax.random.key(0)
    rng, first = jax.random.split(rng)
    rng, second = jax.random.split(rng)
    np.testing.assert_array_equal(recorded_rngs[0], np.asarray(jax.random.key_data(first)))
    np.testing.assert_array_equal(recorded_rngs[1], np.asarray(jax.random.key_data(second)))


def test_predict_action_chunk_rejects_mid_session_reseed() -> None:
    policy = _policy_with_recorded_rngs([])
    policy.predict_action_chunk({}, "pick", seed=0, num_steps=10)

    with pytest.raises(ValueError, match="seed"):
        policy.predict_action_chunk({}, "pick", seed=1, num_steps=10)


def test_failed_preprocessing_does_not_advance_rng_stream() -> None:
    recorded_rngs: list[np.ndarray] = []
    policy = _policy_with_recorded_rngs(recorded_rngs)
    attempts = 0

    def prepare(observation, task):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise ValueError("bad observation")
        return object()

    policy.prepare_observation = prepare

    with pytest.raises(ValueError, match="bad observation"):
        policy.predict_action_chunk({}, "pick", seed=0, num_steps=10)
    policy.predict_action_chunk({}, "pick", seed=0, num_steps=10)

    _, first = jax.random.split(jax.random.key(0))
    np.testing.assert_array_equal(recorded_rngs[0], np.asarray(jax.random.key_data(first)))
