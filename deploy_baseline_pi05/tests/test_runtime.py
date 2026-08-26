from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from deploy_baseline_pi05.deployment import TACTILE_KEYS
from deploy_baseline_pi05.runtime import DirectDecoderRuntime
from deploy_baseline_pi05.tactile_encoder import _rms_normalize_tokens


def _observation(value: int) -> dict[str, np.ndarray]:
    return {
        key: np.full((3, 4, 3), value + index, dtype=np.uint8)
        for index, key in enumerate(TACTILE_KEYS)
    }


@dataclass
class _FakePolicy:
    calls: int = 0

    def predict_action_chunk(self, observation, task, *, seed: int, num_steps: int):
        del observation, task
        assert (seed, num_steps) == (0, 10)
        self.calls += 1
        return np.arange(1_000, dtype=np.float32).reshape(1, 50, 20)

    def unnormalize_actions(self, actions):
        return np.asarray(actions, dtype=np.float32) * 2.0


class _FakeEncoder:
    def __init__(self) -> None:
        self.observations: list[dict[str, np.ndarray]] = []

    def encode(self, observation):
        self.observations.append(observation)
        value = float(observation[TACTILE_KEYS[0]][0, 0, 0])
        return np.full((1, 4, 512), value, dtype=np.float32)


class _FakeDecoder:
    def __init__(self) -> None:
        self.calls = 0
        self.output = np.arange(1_000, dtype=np.float32).reshape(1, 50, 20)

    def decode(self, coarse, tactile):
        assert coarse.shape == (1, 50, 20)
        assert tactile.shape == (1, 4, 512)
        self.calls += 1
        return np.asarray(self.output, dtype=np.float32) + tactile[0, 0, 0]


@pytest.fixture
def fakes():
    return _FakePolicy(), _FakeEncoder(), _FakeDecoder()


@pytest.fixture
def runtime(fakes):
    policy, encoder, decoder = fakes
    return DirectDecoderRuntime(
        policy=policy,
        tactile_encoder=encoder,
        decoder=decoder,
        max_normalized_action_abs=2_000.0,
        max_normalized_delta_rms=2_000.0,
    )


def test_one_pi_sample_and_latest_tactile_per_unique_action(runtime, fakes):
    policy, encoder, decoder = fakes
    ready = runtime.begin_chunk(7, _observation(1), "pick", seed=0, num_steps=10)
    first = runtime.steer_action(7, 10, _observation(20), 0)
    second = runtime.steer_action(7, 11, _observation(30), 1)

    assert ready.action_vla_normalized.shape == (1, 50, 20)
    assert policy.calls == 1
    assert [item[TACTILE_KEYS[0]][0, 0, 0] for item in encoder.observations] == [20, 30]
    assert decoder.calls == 2
    np.testing.assert_array_equal(first.selected_normalized, decoder.output[0, 0] + 20)
    np.testing.assert_array_equal(second.selected_normalized, decoder.output[0, 1] + 30)
    np.testing.assert_array_equal(first.selected_action, first.selected_normalized * 2.0)


def test_duplicate_request_is_idempotent_only_for_the_same_tactile_payload(runtime, fakes):
    _, _, decoder = fakes
    runtime.begin_chunk(7, _observation(1), "pick", seed=0, num_steps=10)
    first = runtime.steer_action(7, 10, _observation(20), 0)
    duplicate = runtime.steer_action(7, 10, _observation(20), 0)

    assert duplicate is first
    assert decoder.calls == 1
    with pytest.raises(ValueError, match="conflicting duplicate"):
        runtime.steer_action(7, 10, _observation(21), 0)
    with pytest.raises(ValueError, match="conflicting duplicate"):
        runtime.steer_action(7, 10, _observation(20), 1)


def test_state_machine_enforces_single_chunk_matching_ids_and_increasing_indices(runtime):
    with pytest.raises(RuntimeError, match="no active"):
        runtime.steer_action(7, 10, _observation(1), 0)
    runtime.begin_chunk(7, _observation(1), "pick", seed=0, num_steps=10)
    with pytest.raises(RuntimeError, match="active"):
        runtime.begin_chunk(8, _observation(1), "pick", seed=0, num_steps=10)
    with pytest.raises(ValueError, match="does not match"):
        runtime.steer_action(8, 10, _observation(1), 0)
    for index in (True, -1, 50, 1.5):
        with pytest.raises(ValueError, match="action index"):
            runtime.steer_action(7, 10, _observation(1), index)
    runtime.steer_action(7, 11, _observation(1), 1)
    with pytest.raises(ValueError, match="strictly increasing"):
        runtime.steer_action(7, 12, _observation(2), 1)
    runtime.end_chunk(7)
    with pytest.raises(RuntimeError, match="no active"):
        runtime.end_chunk(7)


@pytest.mark.parametrize("kind", ("nonfinite", "exception", "magnitude", "delta"))
def test_refinement_failures_are_fail_stop_without_coarse_fallback(runtime, fakes, kind):
    _, _, decoder = fakes
    runtime.begin_chunk(7, _observation(1), "pick", seed=0, num_steps=10)
    if kind == "nonfinite":
        decoder.output = np.full((1, 50, 20), np.nan, dtype=np.float32)
        message = "finite"
    elif kind == "exception":
        def raise_decode(coarse, tactile):
            raise RuntimeError("decoder exploded")
        decoder.decode = raise_decode
        message = "decoder exploded"
    elif kind == "magnitude":
        decoder.output = np.full((1, 50, 20), 3_000.0, dtype=np.float32)
        message = "magnitude"
    else:
        decoder.output = np.full((1, 50, 20), 1_000.0, dtype=np.float32)
        runtime.max_normalized_delta_rms = 1.0
        message = "delta"

    with pytest.raises((RuntimeError, ValueError), match=message):
        runtime.steer_action(7, 10, _observation(1), 0)
    assert runtime.cached_result(10) is None


def test_result_arrays_are_immutable_copies(runtime, fakes):
    _, _, decoder = fakes
    ready = runtime.begin_chunk(7, _observation(1), "pick", seed=0, num_steps=10)
    result = runtime.steer_action(7, 10, _observation(20), 0)
    decoder.output[:] = -1

    np.testing.assert_array_equal(ready.action_vla_normalized, np.arange(1_000, dtype=np.float32).reshape(1, 50, 20))
    with pytest.raises(ValueError):
        result.selected_action[0] = 0.0


def test_tactile_encoder_rejects_zero_embedding_tokens() -> None:
    with pytest.raises(ValueError, match="zero-RMS"):
        _rms_normalize_tokens(np.zeros((4, 512), dtype=np.float32))
