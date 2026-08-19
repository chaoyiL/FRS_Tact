from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
import pytest

from deploy_smolvla import remote_client
from deploy_smolvla import frs_runtime as frs_runtime_module
from deploy_smolvla.bridge_client import RobotBridgeClient
from deploy_smolvla.frs_protocol import FRSSteerRequest
from deploy_smolvla.frs_runtime import (
    FRSChunkReady,
    FRSDiagnostics,
    FRSRuntime,
    FRSSteerResult,
    TactileHistory,
)

ROOT = Path(__file__).resolve().parents[2]
FRS_CONFIG = ROOT / "deploy_smolvla" / "configs" / "deploy_frs.yaml"


class LifecycleSource:
    def __init__(self) -> None:
        self.config = SimpleNamespace(chunk_size=3, action_dim=2)
        self.preprocessor = SimpleNamespace(
            unnormalize_actions=lambda actions: np.asarray(actions) * 10.0
        )
        self.predict_calls = 0
        self.reverse_calls = 0
        self.predict_kwargs: dict[str, object] = {}
        self.reverse_kwargs: dict[str, object] = {}

    def predict_action_chunk(self, observation, task, **kwargs):
        del observation, task
        self.predict_calls += 1
        self.predict_kwargs = kwargs
        return jnp.arange(6, dtype=jnp.float32).reshape(1, 3, 2)

    def reverse_action_chunk(self, observation, task, normalized_actions, **kwargs):
        del observation, task
        self.reverse_calls += 1
        self.reverse_kwargs = kwargs
        return normalized_actions + 100.0


@pytest.fixture
def lifecycle_source() -> LifecycleSource:
    return LifecycleSource()


@pytest.fixture
def lifecycle_runtime(
    lifecycle_source: LifecycleSource,
    monkeypatch: pytest.MonkeyPatch,
):
    runtime = object.__new__(FRSRuntime)
    runtime.policy = lifecycle_source
    runtime.config = SimpleNamespace(
        reverse_steps=5,
        reverse_solver="slerpflow",
    )
    runtime.history = TactileHistory(window=2, stride=1, token_shape=(1, 2))
    runtime.baseline = None
    runtime.last_diagnostics = None
    runtime.last_vla_normalized = None
    runtime.last_frs_normalized = None
    runtime._episode_baseline = None
    runtime._active_chunk_id = None
    runtime._action_vla_normalized = None
    runtime._action_vla = None
    runtime._x_base = None
    runtime._tactile_sequence = []
    runtime._request_results = {}
    runtime.decode_calls = 0

    def record_decode(*args, **kwargs):
        del args, kwargs
        runtime.decode_calls += 1
        raise AssertionError("begin_chunk must not decode")

    monkeypatch.setattr(frs_runtime_module, "decode_actions", record_decode)
    runtime._encode_observation = lambda observation: np.asarray(
        observation["encoded"], dtype=np.float32
    )
    return runtime


def initial_observation(value: float = 1.0) -> dict[str, np.ndarray]:
    return {"encoded": np.full((1, 2), value, dtype=np.float32)}


def test_frs_runtime_is_a_true_compatibility_alias() -> None:
    assert frs_runtime_module.FRSRuntime is frs_runtime_module.FRSSteeringPolicy


def test_begin_chunk_predicts_and_reverses_exactly_once_without_decoding(
    lifecycle_runtime,
    lifecycle_source: LifecycleSource,
) -> None:
    lifecycle_runtime.reset_episode(initial_observation())
    lifecycle_runtime._tactile_sequence.append(np.ones((1, 2), dtype=np.float32))
    lifecycle_runtime._request_results[6] = object()

    ready = lifecycle_runtime.begin_chunk(
        7,
        initial_observation(),
        "pick the tube",
        seed=3,
        jit=True,
        num_steps=None,
    )

    assert lifecycle_source.predict_calls == 1
    assert lifecycle_source.reverse_calls == 1
    assert lifecycle_runtime.decode_calls == 0
    assert lifecycle_source.predict_kwargs == {
        "seed": 3,
        "jit": True,
        "num_steps": None,
        "normalized": True,
    }
    assert lifecycle_source.reverse_kwargs == {"num_steps": 5, "solver": "slerpflow"}
    assert ready.chunk_id == 7
    assert ready.action_vla_normalized.shape == ready.x_base.shape == (1, 3, 2)
    np.testing.assert_array_equal(ready.action_vla, ready.action_vla_normalized * 10.0)
    assert lifecycle_runtime._tactile_sequence == []
    assert lifecycle_runtime._request_results == {}
    assert ready.prediction_started_at <= ready.prediction_finished_at


def test_begin_chunk_requires_an_episode_baseline(lifecycle_runtime) -> None:
    with pytest.raises(RuntimeError, match="reset_episode"):
        lifecycle_runtime.begin_chunk(
            1,
            initial_observation(),
            "pick the tube",
            seed=0,
            jit=False,
            num_steps=4,
        )


def test_begin_chunk_rejects_nested_active_chunks(lifecycle_runtime) -> None:
    lifecycle_runtime.reset_episode(initial_observation())
    lifecycle_runtime.begin_chunk(
        1,
        initial_observation(),
        "pick the tube",
        seed=0,
        jit=False,
        num_steps=4,
    )

    with pytest.raises(RuntimeError, match="active chunk"):
        lifecycle_runtime.begin_chunk(
            2,
            initial_observation(),
            "pick the tube",
            seed=1,
            jit=False,
            num_steps=4,
        )


def test_end_chunk_rejects_the_wrong_active_chunk_id(lifecycle_runtime) -> None:
    lifecycle_runtime.reset_episode(initial_observation())
    lifecycle_runtime.begin_chunk(
        4,
        initial_observation(),
        "pick the tube",
        seed=0,
        jit=False,
        num_steps=4,
    )

    with pytest.raises(ValueError, match="active chunk 4"):
        lifecycle_runtime.end_chunk(5)

    assert lifecycle_runtime._active_chunk_id == 4


def test_chunk_lifecycle_clears_local_state_and_preserves_episode_baseline(
    lifecycle_runtime,
) -> None:
    lifecycle_runtime.reset_episode(initial_observation(2.0))
    baseline = lifecycle_runtime._episode_baseline
    ready = lifecycle_runtime.begin_chunk(
        4,
        initial_observation(),
        "pick the tube",
        seed=0,
        jit=False,
        num_steps=4,
    )
    lifecycle_runtime._tactile_sequence.append(np.ones((1, 2), dtype=np.float32))
    lifecycle_runtime._request_results[9] = object()

    lifecycle_runtime.end_chunk(4)

    assert lifecycle_runtime._episode_baseline is baseline
    assert lifecycle_runtime._active_chunk_id is None
    assert lifecycle_runtime._action_vla_normalized is None
    assert lifecycle_runtime._action_vla is None
    assert lifecycle_runtime._x_base is None
    assert lifecycle_runtime._tactile_sequence == []
    assert lifecycle_runtime._request_results == {}
    assert ready.action_vla_normalized.flags.writeable is False
    assert ready.action_vla.flags.writeable is False
    assert ready.x_base.flags.writeable is False


def test_reset_episode_replaces_baseline_and_clears_an_active_chunk(lifecycle_runtime) -> None:
    lifecycle_runtime.reset_episode(initial_observation(1.0))
    lifecycle_runtime.begin_chunk(
        3,
        initial_observation(),
        "pick the tube",
        seed=0,
        jit=False,
        num_steps=4,
    )

    lifecycle_runtime.reset_episode(initial_observation(8.0))

    np.testing.assert_array_equal(
        lifecycle_runtime._episode_baseline,
        np.full((1, 2), 8.0, dtype=np.float32),
    )
    assert lifecycle_runtime._episode_baseline.flags.writeable is False
    assert lifecycle_runtime._active_chunk_id is None
    assert lifecycle_runtime._tactile_sequence == []
    assert lifecycle_runtime._request_results == {}


@pytest.mark.parametrize(
    "invalid_tokens",
    [
        pytest.param(np.ones((2,), dtype=np.float32), id="malformed"),
        pytest.param(np.asarray([[np.nan, 0.0]], dtype=np.float32), id="nonfinite"),
    ],
)
def test_reset_episode_validation_failure_preserves_the_complete_live_state(
    lifecycle_runtime,
    invalid_tokens: np.ndarray,
) -> None:
    lifecycle_runtime.reset_episode(initial_observation(2.0))
    lifecycle_runtime.begin_chunk(
        3,
        initial_observation(),
        "pick the tube",
        seed=0,
        jit=False,
        num_steps=4,
    )
    tactile = np.full((1, 2), 6.0, dtype=np.float32)
    lifecycle_runtime._tactile_sequence.append(tactile)
    result = object()
    lifecycle_runtime._request_results[8] = result
    old_internal_baseline = lifecycle_runtime._episode_baseline
    old_public_baseline = lifecycle_runtime.baseline
    old_history = lifecycle_runtime.history
    old_history_tokens = old_history.window_tokens().copy()
    old_normalized = lifecycle_runtime._action_vla_normalized
    old_action = lifecycle_runtime._action_vla
    old_x_base = lifecycle_runtime._x_base

    with pytest.raises(ValueError, match="expected tactile tokens|NaN or Inf"):
        lifecycle_runtime.reset_episode({"encoded": invalid_tokens})

    assert lifecycle_runtime._episode_baseline is old_internal_baseline
    assert lifecycle_runtime.baseline is old_public_baseline
    assert lifecycle_runtime.history is old_history
    np.testing.assert_array_equal(lifecycle_runtime.history.window_tokens(), old_history_tokens)
    assert lifecycle_runtime._active_chunk_id == 3
    assert lifecycle_runtime._action_vla_normalized is old_normalized
    assert lifecycle_runtime._action_vla is old_action
    assert lifecycle_runtime._x_base is old_x_base
    assert lifecycle_runtime._tactile_sequence == [tactile]
    assert lifecycle_runtime._request_results == {8: result}


def test_tactile_history_reset_is_transactional_on_invalid_tokens() -> None:
    history = TactileHistory(window=2, stride=1, token_shape=(1, 2))
    history.reset(np.asarray([[1.0, 2.0]], dtype=np.float32))
    history.append(np.asarray([[3.0, 4.0]], dtype=np.float32))
    before = history.window_tokens().copy()

    with pytest.raises(ValueError, match="expected tactile tokens"):
        history.reset(np.ones((2,), dtype=np.float32))

    np.testing.assert_array_equal(history.window_tokens(), before)


def test_public_episode_baseline_is_isolated_from_internal_baseline(lifecycle_runtime) -> None:
    lifecycle_runtime.reset_episode(initial_observation(3.0))
    public = lifecycle_runtime.baseline
    internal = lifecycle_runtime._episode_baseline
    assert public is not None
    assert internal is not None
    assert not np.shares_memory(public, internal)

    public.setflags(write=True)
    public[...] = -99.0

    np.testing.assert_array_equal(internal, np.full((1, 2), 3.0, dtype=np.float32))
    np.testing.assert_array_equal(
        lifecycle_runtime.history.window_tokens()[-1],
        np.full((1, 2), 3.0, dtype=np.float32),
    )


def test_legacy_steer_uses_internal_episode_baseline_after_public_mutation(
    lifecycle_runtime,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lifecycle_runtime.reset_episode(initial_observation(3.0))
    lifecycle_runtime.baseline.setflags(write=True)
    lifecycle_runtime.baseline[...] = -99.0
    captured: list[np.ndarray] = []

    def capture_baseline(current, baseline):
        del current
        captured.append(np.asarray(baseline).copy())
        raise RuntimeError("baseline captured")

    monkeypatch.setattr(
        frs_runtime_module,
        "tactile_change_from_tokens",
        capture_baseline,
    )

    with pytest.raises(RuntimeError, match="baseline captured"):
        lifecycle_runtime.steer(
            object(),
            initial_observation(4.0),
            "pick the tube",
            jnp.zeros((1, 3, 2), dtype=jnp.float32),
            update_history=False,
        )

    np.testing.assert_array_equal(
        captured[0],
        np.full((1, 1, 2), 3.0, dtype=np.float32),
    )


def test_chunk_ready_arrays_are_isolated_from_internal_chunk_state(lifecycle_runtime) -> None:
    lifecycle_runtime.reset_episode(initial_observation())
    ready = lifecycle_runtime.begin_chunk(
        7,
        initial_observation(),
        "pick the tube",
        seed=0,
        jit=False,
        num_steps=4,
    )
    internal_normalized = lifecycle_runtime._action_vla_normalized.copy()
    internal_action = lifecycle_runtime._action_vla.copy()
    internal_x_base = lifecycle_runtime._x_base.copy()

    for array in (ready.action_vla_normalized, ready.action_vla, ready.x_base):
        array.setflags(write=True)
        array[...] = -123.0

    np.testing.assert_array_equal(
        lifecycle_runtime._action_vla_normalized,
        internal_normalized,
    )
    np.testing.assert_array_equal(lifecycle_runtime._action_vla, internal_action)
    np.testing.assert_array_equal(lifecycle_runtime._x_base, internal_x_base)


@pytest.mark.parametrize(
    ("source_result", "reverse_result"),
    [
        pytest.param(np.zeros((1, 2, 2), dtype=np.float32), None, id="source-shape"),
        pytest.param(
            np.full((1, 3, 2), np.nan, dtype=np.float32),
            None,
            id="source-nonfinite",
        ),
        pytest.param(
            None,
            np.zeros((1, 2, 2), dtype=np.float32),
            id="reverse-shape",
        ),
        pytest.param(
            None,
            np.full((1, 3, 2), np.inf, dtype=np.float32),
            id="reverse-nonfinite",
        ),
    ],
)
def test_begin_chunk_rejects_invalid_source_chunks_without_partial_activation(
    lifecycle_runtime,
    lifecycle_source: LifecycleSource,
    source_result: np.ndarray | None,
    reverse_result: np.ndarray | None,
) -> None:
    if source_result is not None:
        lifecycle_source.predict_action_chunk = lambda *args, **kwargs: source_result
    if reverse_result is not None:
        lifecycle_source.reverse_action_chunk = lambda *args, **kwargs: reverse_result
    lifecycle_runtime.reset_episode(initial_observation())

    with pytest.raises(ValueError, match="normalized VLA|reverse-flow base"):
        lifecycle_runtime.begin_chunk(
            4,
            initial_observation(),
            "pick the tube",
            seed=0,
            jit=False,
            num_steps=4,
        )

    assert lifecycle_runtime._active_chunk_id is None
    assert lifecycle_runtime._action_vla_normalized is None
    assert lifecycle_runtime._action_vla is None
    assert lifecycle_runtime._x_base is None
    assert lifecycle_runtime._tactile_sequence == []
    assert lifecycle_runtime._request_results == {}


def test_activate_chunk_conversion_failure_preserves_existing_local_state(
    lifecycle_runtime,
    lifecycle_source: LifecycleSource,
) -> None:
    lifecycle_runtime.reset_episode(initial_observation())
    tactile = np.ones((1, 2), dtype=np.float32)
    cached = object()
    lifecycle_runtime._tactile_sequence.append(tactile)
    lifecycle_runtime._request_results[9] = cached

    def fail_unnormalize(actions):
        del actions
        raise RuntimeError("unnormalization failed")

    lifecycle_source.preprocessor.unnormalize_actions = fail_unnormalize
    normalized = np.zeros((1, 3, 2), dtype=np.float32)

    with pytest.raises(RuntimeError, match="unnormalization failed"):
        lifecycle_runtime._activate_chunk(4, normalized, normalized)

    assert lifecycle_runtime._active_chunk_id is None
    assert lifecycle_runtime._action_vla_normalized is None
    assert lifecycle_runtime._action_vla is None
    assert lifecycle_runtime._x_base is None
    assert lifecycle_runtime._tactile_sequence == [tactile]
    assert lifecycle_runtime._request_results == {9: cached}


@pytest.mark.parametrize(
    "robot_actions",
    [
        pytest.param(np.zeros((1, 2, 2), dtype=np.float32), id="malformed"),
        pytest.param(
            np.full((1, 3, 2), np.nan, dtype=np.float32),
            id="nonfinite",
        ),
    ],
)
def test_activate_chunk_rejects_invalid_robot_actions_without_partial_state(
    lifecycle_runtime,
    lifecycle_source: LifecycleSource,
    robot_actions: np.ndarray,
) -> None:
    lifecycle_runtime.reset_episode(initial_observation())
    lifecycle_source.preprocessor.unnormalize_actions = lambda actions: robot_actions
    normalized = np.zeros((1, 3, 2), dtype=np.float32)

    with pytest.raises(ValueError, match="robot-space VLA"):
        lifecycle_runtime._activate_chunk(4, normalized, normalized)

    assert lifecycle_runtime._active_chunk_id is None
    assert lifecycle_runtime._action_vla_normalized is None
    assert lifecycle_runtime._action_vla is None
    assert lifecycle_runtime._x_base is None
    assert lifecycle_runtime._tactile_sequence == []
    assert lifecycle_runtime._request_results == {}


class PerActionSource:
    def __init__(self) -> None:
        self.config = SimpleNamespace(chunk_size=4, action_dim=2)
        self.unnormalize_inputs: list[np.ndarray] = []
        self.predicted_normalized = jnp.arange(8, dtype=jnp.float32).reshape(1, 4, 2) / 20.0
        self.reverse_result = self.predicted_normalized + 77.0
        self.reverse_inputs: list[np.ndarray] = []
        self.preprocessor = SimpleNamespace(unnormalize_actions=self._unnormalize)

    def _unnormalize(self, actions):
        array = np.asarray(actions, dtype=np.float32)
        self.unnormalize_inputs.append(array.copy())
        return array * 10.0

    def predict_action_chunk(self, observation, task, **kwargs):
        del observation, task, kwargs
        return self.predicted_normalized

    def reverse_action_chunk(self, observation, task, normalized_actions, **kwargs):
        del observation, task, kwargs
        self.reverse_inputs.append(np.asarray(normalized_actions).copy())
        return self.reverse_result


@pytest.fixture
def per_action_runtime(monkeypatch: pytest.MonkeyPatch):
    runtime = object.__new__(FRSRuntime)
    runtime.policy = PerActionSource()
    runtime.config = SimpleNamespace(
        tactile_keys=("left", "right"),
        tactile_window_divisor=1,
        history_stride=1,
        decode_steps=3,
        decode_solver="euler",
        max_normalized_action_abs=8.0,
        max_normalized_delta_rms=4.0,
    )
    runtime.model = SimpleNamespace(config=SimpleNamespace(tactile_window=2))
    runtime.history = TactileHistory(window=2, stride=1, token_shape=(2, 3))
    runtime._episode_baseline = runtime._readonly_array(np.zeros((2, 3), dtype=np.float32))
    runtime.baseline = runtime._readonly_array(runtime._episode_baseline)
    runtime.history.reset(runtime._episode_baseline)
    runtime._clear_chunk_state()
    runtime.encode_calls = 0
    runtime.decode_calls = 0
    runtime.decode_input_shapes = []
    runtime.decode_x_bases = []
    runtime.decode_tactiles = []
    runtime.decode_result = np.arange(8, dtype=np.float32).reshape(1, 4, 2) / 10.0

    def encode_observation(observation):
        runtime.encode_calls += 1
        value = float(np.asarray(observation["left"]).reshape(-1)[0])
        return np.full((2, 3), value, dtype=np.float32)

    def decode(model, x_base, tactile, *, num_steps, solver, **kwargs):
        del model, kwargs
        assert num_steps == 3
        assert solver == "euler"
        runtime.decode_calls += 1
        runtime.decode_input_shapes.append(tuple(tactile.shape))
        runtime.decode_x_bases.append(np.asarray(x_base).copy())
        runtime.decode_tactiles.append(np.asarray(tactile).copy())
        return jnp.asarray(runtime.decode_result)

    runtime._encode_observation = encode_observation
    monkeypatch.setattr(frs_runtime_module, "decode_actions", decode)
    monkeypatch.setattr(
        frs_runtime_module,
        "tactile_change_from_tokens",
        lambda current, baseline: jnp.asarray(
            [float(np.mean(np.asarray(current) - np.asarray(baseline)))],
            dtype=jnp.float32,
        ),
    )
    return runtime


def tactile_observation(
    value: float,
    *,
    dtype: np.dtype = np.dtype(np.float32),
    metadata: float = 0.0,
) -> dict[str, np.ndarray | float]:
    return {
        "left": np.full((2, 2, 3), value, dtype=dtype),
        "right": np.full((2, 2, 3), value + 0.5, dtype=dtype),
        "visual_metadata": metadata,
    }


def start_per_action_chunk(runtime: FRSRuntime, *, chunk_id: int = 4) -> None:
    normalized = np.arange(8, dtype=np.float32).reshape(1, 4, 2) / 20.0
    x_base = normalized + 100.0
    runtime._activate_chunk(chunk_id, normalized, x_base)
    runtime.policy.unnormalize_inputs.clear()


def test_steer_action_result_has_the_exact_frozen_contract(per_action_runtime) -> None:
    start_per_action_chunk(per_action_runtime)

    result = per_action_runtime.steer_action(4, 10, tactile_observation(1.0), 2)

    assert tuple(result.__dataclass_fields__) == (
        "chunk_id",
        "request_id",
        "action_index",
        "action_vla_normalized",
        "x_base",
        "decoded_normalized",
        "selected_normalized",
        "selected_action",
        "tactile_sequence_length",
        "diagnostics",
        "encode_started_at",
        "encode_finished_at",
        "decode_started_at",
        "decode_finished_at",
    )
    assert result.chunk_id == 4
    assert result.request_id == 10
    assert result.action_index == 2
    assert result.action_vla_normalized.shape == result.x_base.shape == (1, 4, 2)
    assert result.decoded_normalized.shape == (1, 4, 2)
    assert result.selected_normalized.shape == result.selected_action.shape == (2,)
    assert all(
        array.dtype == np.dtype(np.float32)
        for array in (
            result.action_vla_normalized,
            result.x_base,
            result.decoded_normalized,
            result.selected_normalized,
            result.selected_action,
        )
    )
    np.testing.assert_array_equal(result.selected_normalized, result.decoded_normalized[0, 2])
    np.testing.assert_array_equal(result.selected_action, result.selected_normalized * 10.0)
    assert result.encode_started_at <= result.encode_finished_at <= result.decode_started_at
    assert result.decode_started_at <= result.decode_finished_at

    with pytest.raises(AttributeError):
        result.request_id = 11


def test_steer_action_temporal_ensemble_weights_newer_predictions_more(
    per_action_runtime,
) -> None:
    per_action_runtime.config.temporal_ensemble_coeff = float(np.log(2.0))
    start_per_action_chunk(per_action_runtime)
    first_chunk = np.zeros((1, 4, 2), dtype=np.float32)
    first_chunk[0, 1] = (1.0, 2.0)
    per_action_runtime.decode_result = first_chunk

    per_action_runtime.steer_action(4, 10, tactile_observation(1.0), 0)
    second_chunk = np.zeros((1, 4, 2), dtype=np.float32)
    second_chunk[0, 1] = (4.0, 5.0)
    per_action_runtime.decode_result = second_chunk
    result = per_action_runtime.steer_action(4, 11, tactile_observation(2.0), 1)

    np.testing.assert_allclose(result.selected_normalized, (3.0, 4.0), rtol=1e-6)
    np.testing.assert_allclose(result.selected_action, (30.0, 40.0), rtol=1e-6)
    np.testing.assert_array_equal(result.decoded_normalized, second_chunk)


def test_steer_action_temporal_ensemble_is_stable_for_large_finite_coeff(
    per_action_runtime,
) -> None:
    per_action_runtime.config.temporal_ensemble_coeff = 1e100
    start_per_action_chunk(per_action_runtime)
    per_action_runtime.decode_result = np.ones((1, 4, 2), dtype=np.float32)
    per_action_runtime.steer_action(4, 10, tactile_observation(1.0), 0)
    newest_chunk = np.full((1, 4, 2), 2.0, dtype=np.float32)
    per_action_runtime.decode_result = newest_chunk

    result = per_action_runtime.steer_action(4, 11, tactile_observation(2.0), 1)

    np.testing.assert_array_equal(result.selected_normalized, newest_chunk[0, 1])


@pytest.mark.parametrize(
    ("left_xyz_m", "right_xyz_m", "protected_slices"),
    [
        ((0.0001, -0.0002, 0.000249), (0.0001, 0.0003, 0.0), (slice(0, 3),)),
        ((0.0003, 0.0, 0.0), (-0.000249, 0.0001, 0.0), (slice(10, 13),)),
        (
            ((-0.0001, 0.0001, 0.0)),
            ((0.0002, -0.0002, 0.0002)),
            (slice(0, 3), slice(10, 13)),
        ),
    ],
)
def test_steer_action_protects_only_inactive_arm_xyz_in_robot_space(
    per_action_runtime,
    left_xyz_m: tuple[float, float, float],
    right_xyz_m: tuple[float, float, float],
    protected_slices: tuple[slice, ...],
) -> None:
    per_action_runtime.policy.config.action_dim = 20
    per_action_runtime.config.inactive_arm_xyz_threshold_m = 0.00025
    normalized_vla = np.zeros((1, 4, 20), dtype=np.float32)
    normalized_vla[0, 0, 0:3] = np.asarray(left_xyz_m) / 10.0
    normalized_vla[0, 0, 10:13] = np.asarray(right_xyz_m) / 10.0
    per_action_runtime._activate_chunk(4, normalized_vla, normalized_vla)
    decoded = np.arange(80, dtype=np.float32).reshape(1, 4, 20) / 20.0
    per_action_runtime.decode_result = decoded

    result = per_action_runtime.steer_action(4, 10, tactile_observation(1.0), 0)

    expected_normalized = decoded[0, 0].copy()
    for protected_slice in protected_slices:
        expected_normalized[protected_slice] = normalized_vla[0, 0, protected_slice]
    np.testing.assert_array_equal(result.decoded_normalized, decoded)
    np.testing.assert_array_equal(result.selected_normalized, expected_normalized)
    np.testing.assert_array_equal(result.selected_action, expected_normalized * 10.0)
    np.testing.assert_array_equal(result.selected_normalized[3:10], decoded[0, 0, 3:10])
    np.testing.assert_array_equal(result.selected_normalized[13:20], decoded[0, 0, 13:20])


def test_inactive_arm_xyz_protection_uses_strict_float32_threshold() -> None:
    threshold = np.float32(0.00025)
    immediately_below = np.nextafter(threshold, np.float32(-np.inf))
    selected = np.arange(20, dtype=np.float32)
    vla_normalized = np.zeros(20, dtype=np.float32)
    vla_action = np.full(20, 1.0, dtype=np.float32)
    vla_action[0:3] = (immediately_below, 0.0, 0.0)

    protected = frs_runtime_module._protect_inactive_arm_xyz(
        selected,
        vla_normalized,
        vla_action,
        float(threshold),
    )
    np.testing.assert_array_equal(protected[0:3], vla_normalized[0:3])

    vla_action[0] = threshold
    unprotected = frs_runtime_module._protect_inactive_arm_xyz(
        selected,
        vla_normalized,
        vla_action,
        float(threshold),
    )
    np.testing.assert_array_equal(unprotected, selected)


def test_steer_action_multiplies_robot_space_grippers_below_threshold(
    per_action_runtime,
) -> None:
    per_action_runtime.policy.config.action_dim = 20
    per_action_runtime.config.gripper_gain = (0.1, 1.5, 2.0)
    normalized = np.zeros((1, 4, 20), dtype=np.float32)
    per_action_runtime._activate_chunk(4, normalized, normalized)
    decoded = normalized.copy()
    decoded[0, 0, 8:11] = (0.008, 0.008, 0.010)
    decoded[0, 0, 18:20] = (0.018, 0.015)
    per_action_runtime.decode_result = decoded

    result = per_action_runtime.steer_action(4, 10, tactile_observation(1.0), 0)

    expected = decoded[0, 0] * 10.0
    expected[[9, 19]] = (0.12, 0.30)
    np.testing.assert_array_equal(result.selected_normalized, decoded[0, 0])
    np.testing.assert_allclose(result.selected_action, expected, atol=1e-7)


def test_steer_result_arrays_cannot_be_mutated_or_corrupt_cached_replay(
    per_action_runtime,
) -> None:
    start_per_action_chunk(per_action_runtime)
    observation = tactile_observation(1.0)
    result = per_action_runtime.steer_action(4, 10, observation, 1)
    public_arrays = (
        result.action_vla_normalized,
        result.x_base,
        result.decoded_normalized,
        result.selected_normalized,
        result.selected_action,
    )
    expected = tuple(array.copy() for array in public_arrays)
    internal_arrays = (
        per_action_runtime._action_vla_normalized,
        per_action_runtime._x_base,
        *per_action_runtime._tactile_sequence,
    )

    for public in public_arrays:
        assert public.flags.writeable is False
        assert all(not np.shares_memory(public, internal) for internal in internal_arrays)
        with pytest.raises(ValueError):
            public.setflags(write=True)
        with pytest.raises(ValueError):
            public.flat[0] = -999.0

    replay = per_action_runtime.steer_action(4, 10, observation, 1)
    assert replay is result
    for actual, original in zip(public_arrays, expected, strict=True):
        np.testing.assert_array_equal(actual, original)
    np.testing.assert_array_equal(
        per_action_runtime._action_vla_normalized,
        np.arange(8, dtype=np.float32).reshape(1, 4, 2) / 20.0,
    )
    np.testing.assert_array_equal(
        per_action_runtime._x_base,
        np.arange(8, dtype=np.float32).reshape(1, 4, 2) / 20.0 + 100.0,
    )


def test_unique_steer_requests_grow_true_unpadded_sequence_one_at_a_time(
    per_action_runtime,
) -> None:
    start_per_action_chunk(per_action_runtime)

    first = per_action_runtime.steer_action(4, 10, tactile_observation(1.0), 1)
    second = per_action_runtime.steer_action(4, 11, tactile_observation(2.0), 3)

    assert first.tactile_sequence_length == 1
    assert second.tactile_sequence_length == 2
    assert per_action_runtime.decode_input_shapes == [(1, 1, 2, 3), (1, 2, 2, 3)]


def test_online_tactile_sequence_content_and_order_is_exact_for_every_length(
    per_action_runtime,
) -> None:
    start_per_action_chunk(per_action_runtime)

    for index, value in enumerate((1.0, 2.0, 3.0, 4.0)):
        result = per_action_runtime.steer_action(
            4,
            10 + index,
            tactile_observation(value),
            index,
        )
        expected = np.stack(
            [np.full((2, 3), prior, dtype=np.float32) for prior in range(1, index + 2)]
        )[None, ...]
        assert result.tactile_sequence_length == index + 1
        np.testing.assert_array_equal(per_action_runtime.decode_tactiles[index], expected)


def test_duplicate_identical_request_is_cached_without_side_effects(per_action_runtime) -> None:
    start_per_action_chunk(per_action_runtime)
    observation = tactile_observation(1.0, metadata=1.0)
    first = per_action_runtime.steer_action(4, 10, observation, 1)

    replay = tactile_observation(1.0, metadata=999.0)
    second = per_action_runtime.steer_action(4, 10, replay, 1)

    assert second is first
    assert per_action_runtime.encode_calls == 1
    assert per_action_runtime.decode_calls == 1
    assert len(per_action_runtime._tactile_sequence) == 1


@pytest.mark.parametrize(
    ("chunk_id", "action_index", "observation"),
    [
        pytest.param(5, 1, tactile_observation(1.0), id="chunk"),
        pytest.param(4, 2, tactile_observation(1.0), id="action-index"),
        pytest.param(4, 1, tactile_observation(2.0), id="bytes"),
        pytest.param(4, 1, tactile_observation(1.0, dtype=np.float64), id="dtype"),
        pytest.param(
            4,
            1,
            {"left": np.ones((4, 3), dtype=np.float32), "right": np.ones((2, 2, 3), dtype=np.float32)},
            id="shape",
        ),
    ],
)
def test_conflicting_duplicate_request_fails_without_mutation(
    per_action_runtime,
    chunk_id: int,
    action_index: int,
    observation: dict[str, np.ndarray],
) -> None:
    start_per_action_chunk(per_action_runtime)
    first = per_action_runtime.steer_action(4, 10, tactile_observation(1.0), 1)
    tactile_before = tuple(per_action_runtime._tactile_sequence)

    with pytest.raises(ValueError, match="conflicting duplicate|active chunk"):
        per_action_runtime.steer_action(chunk_id, 10, observation, action_index)

    assert per_action_runtime._request_results[10][-1] is first
    assert tuple(per_action_runtime._tactile_sequence) == tactile_before
    assert per_action_runtime.encode_calls == 1
    assert per_action_runtime.decode_calls == 1


def test_tactile_hash_uses_configured_key_order_not_mapping_order(per_action_runtime) -> None:
    start_per_action_chunk(per_action_runtime)
    observation = tactile_observation(1.0)
    first = per_action_runtime.steer_action(4, 10, observation, 1)
    reordered = {
        "right": observation["right"],
        "visual_metadata": -1.0,
        "left": observation["left"],
    }

    assert per_action_runtime.steer_action(4, 10, reordered, 1) is first
    assert per_action_runtime.encode_calls == 1


def test_noncontiguous_tactile_replay_is_content_equivalent(per_action_runtime) -> None:
    start_per_action_chunk(per_action_runtime)
    observation = tactile_observation(1.0)
    first = per_action_runtime.steer_action(4, 10, observation, 1)
    replay = {
        key: np.asfortranarray(value)
        for key, value in observation.items()
        if key in ("left", "right")
    }
    assert replay["left"].flags.c_contiguous is False

    assert per_action_runtime.steer_action(4, 10, replay, 1) is first
    assert per_action_runtime.encode_calls == 1


@pytest.mark.parametrize(
    "invalid_dtype",
    [
        pytest.param(np.dtype(object), id="object"),
        pytest.param(np.dtype([("value", np.float32)]), id="structured"),
    ],
)
def test_tactile_hash_rejects_object_and_structured_dtypes_before_encoding(
    per_action_runtime,
    invalid_dtype: np.dtype,
) -> None:
    start_per_action_chunk(per_action_runtime)
    observation = tactile_observation(1.0)
    observation["left"] = np.ones((2, 2, 3), dtype=invalid_dtype)

    with pytest.raises(ValueError, match="numeric non-structured dtype"):
        per_action_runtime.steer_action(4, 10, observation, 0)

    assert per_action_runtime.encode_calls == 0
    assert per_action_runtime.decode_calls == 0
    assert per_action_runtime._tactile_sequence == []
    assert per_action_runtime._request_results == {}


@pytest.mark.parametrize("indices", [(1, 1), (2, 0), (0, 4), (0, -1)])
def test_new_steer_requests_require_strictly_increasing_in_range_indices(
    per_action_runtime,
    indices: tuple[int, int],
) -> None:
    start_per_action_chunk(per_action_runtime)
    first_index, second_index = indices
    if 0 <= first_index < 4:
        per_action_runtime.steer_action(4, 10, tactile_observation(1.0), first_index)
        expected_encodes = 1
    else:
        expected_encodes = 0

    with pytest.raises(ValueError, match="action_index"):
        per_action_runtime.steer_action(4, 11, tactile_observation(2.0), second_index)

    assert per_action_runtime.encode_calls == expected_encodes


def test_steer_action_rejects_more_requests_than_the_action_horizon(per_action_runtime) -> None:
    start_per_action_chunk(per_action_runtime)
    for index in range(4):
        per_action_runtime.steer_action(4, 10 + index, tactile_observation(index + 1.0), index)

    with pytest.raises(ValueError, match="action horizon|tactile sequence"):
        per_action_runtime.steer_action(4, 20, tactile_observation(9.0), 4)

    assert len(per_action_runtime._tactile_sequence) == 4
    assert per_action_runtime.encode_calls == 4
    assert per_action_runtime.decode_calls == 4


def test_steer_action_uses_fixed_base_and_unnormalizes_only_selected_row(
    per_action_runtime,
) -> None:
    start_per_action_chunk(per_action_runtime)
    fixed_base = per_action_runtime._x_base.copy()

    first = per_action_runtime.steer_action(4, 10, tactile_observation(1.0), 0)
    second = per_action_runtime.steer_action(4, 11, tactile_observation(2.0), 3)

    np.testing.assert_array_equal(per_action_runtime.decode_x_bases[0], fixed_base)
    np.testing.assert_array_equal(per_action_runtime.decode_x_bases[1], fixed_base)
    assert per_action_runtime.policy.unnormalize_inputs[0].shape == (2,)
    assert per_action_runtime.policy.unnormalize_inputs[1].shape == (2,)
    np.testing.assert_array_equal(first.selected_action, first.selected_normalized * 10.0)
    np.testing.assert_array_equal(second.selected_action, second.selected_normalized * 10.0)


def test_begin_chunk_reverse_receives_normalized_source_and_decoder_reuses_exact_base(
    per_action_runtime,
) -> None:
    per_action_runtime.config.reverse_steps = 5
    per_action_runtime.config.reverse_solver = "slerpflow"

    ready = per_action_runtime.begin_chunk(
        4,
        tactile_observation(0.0),
        "pick the tube",
        seed=3,
        jit=True,
        num_steps=7,
    )
    per_action_runtime.steer_action(4, 10, tactile_observation(1.0), 0)

    np.testing.assert_array_equal(
        per_action_runtime.policy.reverse_inputs[0],
        per_action_runtime.policy.predicted_normalized,
    )
    np.testing.assert_array_equal(ready.x_base, per_action_runtime.policy.reverse_result)
    np.testing.assert_array_equal(
        per_action_runtime.decode_x_bases[0],
        per_action_runtime.policy.reverse_result,
    )


def test_full_chunk_rms_rejects_safe_selected_row_when_another_row_is_unsafe(
    per_action_runtime,
) -> None:
    start_per_action_chunk(per_action_runtime)
    decoded = per_action_runtime._action_vla_normalized.copy()
    decoded[0, 1:, :] = 7.0
    per_action_runtime.decode_result = decoded

    with pytest.raises(ValueError, match="delta safety limit"):
        per_action_runtime.steer_action(4, 10, tactile_observation(1.0), 0)

    assert np.max(np.abs(decoded[0, 0])) < per_action_runtime.config.max_normalized_action_abs
    assert per_action_runtime.policy.unnormalize_inputs == []
    assert per_action_runtime._tactile_sequence == []


@pytest.mark.parametrize(
    ("decoded", "match"),
    [
        pytest.param(np.zeros((1, 3, 2), dtype=np.float32), "shape", id="shape"),
        pytest.param(
            np.asarray([[[0.0, 0.0], [0.0, 0.0], [np.nan, 0.0], [0.0, 0.0]]]),
            "NaN or Inf",
            id="nonfinite-unselected-row",
        ),
        pytest.param(
            np.asarray([[[0.0, 0.0], [0.0, 0.0], [9.0, 0.0], [0.0, 0.0]]]),
            "action safety limit",
            id="max-unselected-row",
        ),
        pytest.param(np.full((1, 4, 2), 5.0, dtype=np.float32), "delta safety limit", id="rms"),
    ],
)
def test_full_chunk_safety_runs_before_selection_without_partial_mutation(
    per_action_runtime,
    decoded: np.ndarray,
    match: str,
) -> None:
    start_per_action_chunk(per_action_runtime)
    per_action_runtime.decode_result = decoded

    with pytest.raises(ValueError, match=match):
        per_action_runtime.steer_action(4, 10, tactile_observation(1.0), 0)

    assert per_action_runtime._tactile_sequence == []
    assert per_action_runtime._request_results == {}
    assert per_action_runtime.policy.unnormalize_inputs == []


def test_nonfinite_tactile_encoding_fails_without_partial_mutation(
    per_action_runtime,
) -> None:
    start_per_action_chunk(per_action_runtime)
    per_action_runtime._encode_observation = lambda observation: np.full(
        (2, 3), np.nan, dtype=np.float32
    )

    with pytest.raises(ValueError, match="tactile encoder produced NaN or Inf"):
        per_action_runtime.steer_action(4, 10, tactile_observation(1.0), 0)

    assert per_action_runtime._tactile_sequence == []
    assert per_action_runtime._request_results == {}
    assert per_action_runtime.decode_calls == 0


def test_chunk_boundary_clears_sequence_cache_and_index_state(per_action_runtime) -> None:
    start_per_action_chunk(per_action_runtime, chunk_id=4)
    old = per_action_runtime.steer_action(4, 10, tactile_observation(1.0), 3)
    per_action_runtime.end_chunk(4)
    start_per_action_chunk(per_action_runtime, chunk_id=5)

    new = per_action_runtime.steer_action(5, 10, tactile_observation(1.0), 0)

    assert new is not old
    assert new.tactile_sequence_length == 1
    assert per_action_runtime.decode_input_shapes[-1] == (1, 1, 2, 3)


def test_tactile_change_uses_newest_current_token_and_protected_episode_baseline(
    per_action_runtime,
    monkeypatch,
) -> None:
    start_per_action_chunk(per_action_runtime)
    per_action_runtime.baseline.setflags(write=True)
    per_action_runtime.baseline[...] = -999.0
    captured = []

    def capture(current, baseline):
        captured.append((np.asarray(current).copy(), np.asarray(baseline).copy()))
        return jnp.asarray([0.0], dtype=jnp.float32)

    monkeypatch.setattr(frs_runtime_module, "tactile_change_from_tokens", capture)
    per_action_runtime.steer_action(4, 10, tactile_observation(1.0), 0)
    per_action_runtime.steer_action(4, 11, tactile_observation(2.0), 1)

    np.testing.assert_array_equal(captured[0][0], np.ones((1, 2, 3), dtype=np.float32))
    np.testing.assert_array_equal(captured[0][1], np.zeros((1, 2, 3), dtype=np.float32))
    np.testing.assert_array_equal(captured[1][0], np.full((1, 2, 3), 2.0, dtype=np.float32))
    np.testing.assert_array_equal(captured[1][1], np.zeros((1, 2, 3), dtype=np.float32))


def _array_state(array: np.ndarray | None) -> tuple[object, ...] | None:
    if array is None:
        return None
    return (
        id(array),
        array.dtype.str,
        array.shape,
        array.flags.writeable,
        array.tobytes(order="C"),
    )


def _result_state(result) -> tuple[object, ...]:
    return (
        id(result),
        result.chunk_id,
        result.request_id,
        result.action_index,
        _array_state(result.action_vla_normalized),
        _array_state(result.x_base),
        _array_state(result.decoded_normalized),
        _array_state(result.selected_normalized),
        _array_state(result.selected_action),
        result.tactile_sequence_length,
        result.diagnostics,
        result.encode_started_at,
        result.encode_finished_at,
        result.decode_started_at,
        result.decode_finished_at,
    )


def _deep_live_state(runtime: FRSRuntime) -> tuple[object, ...]:
    requests = tuple(
        (
            request_id,
            chunk_id,
            action_index,
            tactile_hash,
            _result_state(result),
        )
        for request_id, (chunk_id, action_index, tactile_hash, result) in sorted(
            runtime._request_results.items()
        )
    )
    return (
        _array_state(runtime._episode_baseline),
        _array_state(runtime.baseline),
        id(runtime.history),
        id(runtime.history._frames),
        runtime.history._frames.maxlen,
        tuple(_array_state(frame) for frame in runtime.history._frames),
        runtime._active_chunk_id,
        _array_state(runtime._action_vla_normalized),
        _array_state(runtime._action_vla),
        _array_state(runtime._x_base),
        id(runtime._tactile_sequence),
        tuple(_array_state(token) for token in runtime._tactile_sequence),
        id(runtime._request_results),
        requests,
        runtime._last_action_index,
        runtime.last_diagnostics,
        _array_state(runtime.last_vla_normalized),
        _array_state(runtime.last_frs_normalized),
    )


@pytest.mark.parametrize("failure", [False, True], ids=("success", "decode-exception"))
def test_warmup_preserves_deep_active_state_content_and_identity(
    per_action_runtime,
    monkeypatch,
    failure: bool,
) -> None:
    start_per_action_chunk(per_action_runtime)
    per_action_runtime.steer_action(4, 10, tactile_observation(1.0), 1)
    before = _deep_live_state(per_action_runtime)
    calls = 0

    def warmup_decode(*args, **kwargs):
        nonlocal calls
        del args, kwargs
        calls += 1
        if failure and calls == 2:
            raise RuntimeError("warmup decode failed")
        return jnp.zeros((1, 4, 2), dtype=jnp.float32)

    monkeypatch.setattr(frs_runtime_module, "decode_actions", warmup_decode)

    if failure:
        with pytest.raises(RuntimeError, match="warmup decode failed"):
            per_action_runtime.warmup_all_tactile_lengths()
    else:
        per_action_runtime.warmup_all_tactile_lengths()

    assert _deep_live_state(per_action_runtime) == before


@pytest.mark.parametrize(
    "failure_kind",
    ("decoder", "unnormalize-exception", "unnormalize-shape", "unnormalize-nonfinite"),
)
def test_unique_steer_late_failures_leave_all_live_state_unchanged(
    per_action_runtime,
    monkeypatch,
    failure_kind: str,
) -> None:
    start_per_action_chunk(per_action_runtime)
    sentinel_diagnostics = frs_runtime_module.FRSDiagnostics(0.1, 0.3, 0.4)
    per_action_runtime.last_diagnostics = sentinel_diagnostics
    per_action_runtime.last_vla_normalized = per_action_runtime._readonly_array(
        np.full((1, 4, 2), 11.0, dtype=np.float32)
    )
    per_action_runtime.last_frs_normalized = per_action_runtime._readonly_array(
        np.full((1, 4, 2), 12.0, dtype=np.float32)
    )

    if failure_kind == "decoder":
        monkeypatch.setattr(
            frs_runtime_module,
            "decode_actions",
            lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("decode failed")),
        )
        match = "decode failed"
    elif failure_kind == "unnormalize-exception":
        per_action_runtime.policy.preprocessor.unnormalize_actions = (
            lambda actions: (_ for _ in ()).throw(RuntimeError("unnormalize failed"))
        )
        match = "unnormalize failed"
    elif failure_kind == "unnormalize-shape":
        per_action_runtime.policy.preprocessor.unnormalize_actions = lambda actions: np.zeros(
            (1, 2), dtype=np.float32
        )
        match = "selected action must have shape"
    else:
        per_action_runtime.policy.preprocessor.unnormalize_actions = lambda actions: np.asarray(
            [np.nan, 0.0], dtype=np.float32
        )
        match = "selected action must be finite"
    before = _deep_live_state(per_action_runtime)

    with pytest.raises((RuntimeError, ValueError), match=match):
        per_action_runtime.steer_action(4, 10, tactile_observation(1.0), 0)

    assert _deep_live_state(per_action_runtime) == before


def _live_state_identities(runtime: FRSRuntime) -> tuple[object, ...]:
    return (
        runtime._episode_baseline,
        runtime.baseline,
        runtime.history,
        runtime._active_chunk_id,
        runtime._action_vla_normalized,
        runtime._action_vla,
        runtime._x_base,
        runtime._tactile_sequence,
        runtime._request_results,
        runtime._last_action_index,
        runtime.last_diagnostics,
        runtime.last_vla_normalized,
        runtime.last_frs_normalized,
    )


def test_warmup_compiles_every_true_tactile_length_and_preserves_inactive_state(
    per_action_runtime,
    monkeypatch,
    caplog,
) -> None:
    blocked = []
    monkeypatch.setattr(frs_runtime_module.jax, "block_until_ready", lambda value: blocked.append(value))
    before = _live_state_identities(per_action_runtime)

    per_action_runtime.warmup_all_tactile_lengths()

    assert per_action_runtime.decode_input_shapes == [
        (1, 1, 2, 3),
        (1, 2, 2, 3),
        (1, 3, 2, 3),
        (1, 4, 2, 3),
    ]
    assert len(blocked) == 4
    assert "warming 4 concrete lengths" in caplog.text
    assert _live_state_identities(per_action_runtime) == before
    assert per_action_runtime.encode_calls == 0


def test_steer_action_keeps_only_the_resolved_tactile_window(per_action_runtime) -> None:
    per_action_runtime.config.tactile_window_divisor = 2
    start_per_action_chunk(per_action_runtime)

    lengths = []
    for index, value in enumerate((1.0, 2.0, 3.0, 4.0)):
        result = per_action_runtime.steer_action(
            4,
            10 + index,
            tactile_observation(value),
            index,
        )
        lengths.append(result.tactile_sequence_length)

    assert lengths == [1, 2, 2, 2]
    assert per_action_runtime.decode_input_shapes == [
        (1, 1, 2, 3),
        (1, 2, 2, 3),
        (1, 2, 2, 3),
        (1, 2, 2, 3),
    ]
    np.testing.assert_array_equal(
        per_action_runtime.decode_tactiles[-1],
        np.stack(
            [np.full((2, 3), value, dtype=np.float32) for value in (3.0, 4.0)]
        )[None, ...],
    )


def test_warmup_compiles_only_the_resolved_tactile_window(
    per_action_runtime,
    monkeypatch,
) -> None:
    per_action_runtime.config.tactile_window_divisor = 2
    blocked = []
    monkeypatch.setattr(frs_runtime_module.jax, "block_until_ready", lambda value: blocked.append(value))

    per_action_runtime.warmup_all_tactile_lengths()

    assert per_action_runtime.decode_input_shapes == [
        (1, 1, 2, 3),
        (1, 2, 2, 3),
    ]
    assert len(blocked) == 2


def test_frs_config_requires_tactile_window_divisor() -> None:
    config = {
        "frs": {
            "enabled": True,
            "checkpoint": "unused",
            "tactile_encoder_checkpoint": "unused",
            "tactile_keys": ["left"],
            "reverse_steps": 1,
            "reverse_solver": "euler",
            "decode_steps": 1,
            "decode_solver": "euler",
        },
        "observation": {"data_type": "vitac"},
        "control": {"steps_per_inference": 2, "action_horizon": 2},
    }

    with pytest.raises(ValueError, match="tactile_window_divisor"):
        frs_runtime_module.validate_frs_config_section(config)


def test_frs_config_rejects_window_divisor_that_does_not_divide_horizon() -> None:
    config = {
        "frs": {
            "enabled": True,
            "checkpoint": "unused",
            "tactile_encoder_checkpoint": "unused",
            "tactile_keys": ["left"],
            "tactile_window_divisor": 3,
            "reverse_steps": 1,
            "reverse_solver": "euler",
            "decode_steps": 1,
            "decode_solver": "euler",
        },
        "observation": {"data_type": "vitac"},
        "control": {"steps_per_inference": 20, "action_horizon": 20},
    }

    with pytest.raises(ValueError, match="divisible"):
        frs_runtime_module.validate_frs_config_section(config)


def test_warmup_restores_active_state_even_when_decode_fails(
    per_action_runtime,
    monkeypatch,
) -> None:
    start_per_action_chunk(per_action_runtime)
    cached = per_action_runtime.steer_action(4, 10, tactile_observation(1.0), 1)
    sequence_bytes = [item.tobytes() for item in per_action_runtime._tactile_sequence]
    before = _live_state_identities(per_action_runtime)
    calls = 0

    def fail_second(*args, **kwargs):
        nonlocal calls
        del args, kwargs
        calls += 1
        if calls == 2:
            raise RuntimeError("warmup decode failed")
        return jnp.zeros((1, 4, 2), dtype=jnp.float32)

    monkeypatch.setattr(frs_runtime_module, "decode_actions", fail_second)

    with pytest.raises(RuntimeError, match="warmup decode failed"):
        per_action_runtime.warmup_all_tactile_lengths()

    assert _live_state_identities(per_action_runtime) == before
    assert per_action_runtime._request_results[10][-1] is cached
    assert [item.tobytes() for item in per_action_runtime._tactile_sequence] == sequence_bytes
    assert per_action_runtime.encode_calls == 1


def test_deploy_frs_config_uses_project_local_downloads() -> None:
    config = remote_client.load_config(FRS_CONFIG)
    root = ROOT / "checkpoints"
    assert Path(config["checkpoint"]) == root / "model/pick_tube_01_jax"
    assert Path(config["frs"]["checkpoint"]) == root / "frs/frs_lambda_05/best"
    assert Path(config["frs"]["tactile_encoder_checkpoint"]) == (
        root / "encoder/encoder_ckpt_0809"
    )


def test_deploy_frs_config_preserves_training_time_scale() -> None:
    config = remote_client.load_config(FRS_CONFIG)

    assert config["observation"]["data_type"] == "vitac"
    assert config["control"]["control_frequency"] == 10.0
    assert config["control"]["steps_per_inference"] == 20
    assert config["control"]["steps_per_inference"] == config["control"]["action_horizon"]
    assert config["frs"]["steering_protection_interval_s"] is None
    assert config["frs"]["history_stride"] == 1
    assert config["frs"]["tactile_window_divisor"] == 4
    assert config["frs"]["reverse_solver"] == "slerpflow"
    assert config["frs"]["decode_solver"] == "fireflow"
    assert "gate_tau" not in config["frs"]
    assert "gate_temperature" not in config["frs"]


def test_frs_config_defaults_history_stride_to_one() -> None:
    config = {
        "frs": {
            "enabled": True,
            "checkpoint": "unused",
            "tactile_encoder_checkpoint": "unused",
            "tactile_keys": ["left"],
            "tactile_window_divisor": 1,
            "reverse_steps": 1,
            "reverse_solver": "euler",
            "decode_steps": 1,
            "decode_solver": "euler",
        },
        "observation": {"data_type": "vitac"},
        "control": {"steps_per_inference": 2, "action_horizon": 2},
    }

    frs_runtime_module.validate_frs_config_section(config)


@pytest.mark.parametrize("value", [0, -1])
def test_frs_config_rejects_nonpositive_history_stride(value: int) -> None:
    config = {
        "frs": {
            "enabled": True,
            "checkpoint": "unused",
            "tactile_encoder_checkpoint": "unused",
            "tactile_keys": ["left"],
            "history_stride": value,
            "tactile_window_divisor": 1,
            "reverse_steps": 1,
            "reverse_solver": "euler",
            "decode_steps": 1,
            "decode_solver": "euler",
        },
        "observation": {"data_type": "vitac"},
        "control": {"steps_per_inference": 2, "action_horizon": 2},
    }

    with pytest.raises(ValueError, match="history_stride"):
        frs_runtime_module.validate_frs_config_section(config)


def test_steer_action_samples_unpadded_history_with_stride(per_action_runtime) -> None:
    per_action_runtime.config.tactile_window_divisor = 2
    per_action_runtime.config.history_stride = 2
    start_per_action_chunk(per_action_runtime)

    lengths = []
    for index, value in enumerate((1.0, 2.0, 3.0, 4.0)):
        result = per_action_runtime.steer_action(
            4,
            10 + index,
            tactile_observation(value),
            index,
        )
        lengths.append(result.tactile_sequence_length)

    assert lengths == [1, 1, 2, 2]
    np.testing.assert_array_equal(
        per_action_runtime.decode_tactiles[-1],
        np.stack(
            [np.full((2, 3), value, dtype=np.float32) for value in (2.0, 4.0)]
        )[None, ...],
    )


def test_frs_server_config_advertises_explicit_v1_fields() -> None:
    observation = {
        "data_type": "vitac",
        "language_prompt": "test",
        "single_arm_mode": False,
        "no_state_obs_mode": False,
    }
    control = {
        "control_frequency": 30.0,
        "controller_frequency": 80.0,
        "steps_per_inference": 10,
        "action_horizon": 10,
    }
    frs_policy = SimpleNamespace(
        config=SimpleNamespace(steering_protection_interval_s=None),
        tactile_keys=("left", "right"),
    )

    config = remote_client._build_server_config(
        observation,
        control,
        frs_policy=frs_policy,
    )

    assert config == {
        "data_type": "vitac",
        "language_prompt": "test",
        "control_frequency": 30.0,
        "controller_frequency": 80.0,
        "single_arm_mode": False,
        "no_state_obs_mode": False,
        "steps_per_inference": 10,
        "action_horizon": 10,
        "execution_protocol": "frs_steering_v1",
        "steering_protection_interval_s": None,
        "frs_tactile_keys": ["left", "right"],
    }


def test_legacy_server_config_omits_all_frs_protocol_fields() -> None:
    observation = {
        "data_type": "vision",
        "language_prompt": "test",
        "single_arm_mode": False,
        "no_state_obs_mode": False,
    }
    control = {
        "control_frequency": 20.0,
        "controller_frequency": 80.0,
        "steps_per_inference": 5,
        "action_horizon": 10,
    }

    config = remote_client._build_server_config(observation, control, frs_policy=None)

    assert config == {
        "data_type": "vision",
        "language_prompt": "test",
        "control_frequency": 20.0,
        "controller_frequency": 80.0,
        "single_arm_mode": False,
        "no_state_obs_mode": False,
        "steps_per_inference": 5,
        "action_horizon": 10,
    }
    assert "execution_protocol" not in config
    assert "steering_protection_interval_s" not in config
    assert "frs_tactile_keys" not in config


@pytest.mark.parametrize(
    "value",
    [None, 0, 0.0, 0.25, np.float32(0.5)],
)
def test_frs_config_accepts_null_or_nonnegative_finite_protection_interval(
    value: object,
) -> None:
    config = {
        "frs": {
            "enabled": True,
            "checkpoint": "unused",
            "tactile_encoder_checkpoint": "unused",
            "tactile_keys": ["left"],
            "tactile_window_divisor": 1,
            "reverse_steps": 1,
            "reverse_solver": "euler",
            "decode_steps": 1,
            "decode_solver": "euler",
            "steering_protection_interval_s": value,
        },
        "observation": {"data_type": "vitac"},
        "control": {"steps_per_inference": 2, "action_horizon": 2},
    }

    frs_runtime_module.validate_frs_config_section(config)


def test_parse_frs_config_stores_validated_protection_interval(tmp_path: Path) -> None:
    frs_checkpoint = tmp_path / "frs"
    encoder_checkpoint = tmp_path / "encoder"
    frs_checkpoint.mkdir()
    encoder_checkpoint.mkdir()
    raw = {
        "checkpoint": str(frs_checkpoint),
        "tactile_encoder_checkpoint": str(encoder_checkpoint),
        "tactile_keys": ["left"],
        "tactile_window_divisor": 1,
        "reverse_steps": 1,
        "reverse_solver": "euler",
        "decode_steps": 1,
        "decode_solver": "euler",
        "steering_protection_interval_s": 0.25,
    }

    parsed = frs_runtime_module.parse_frs_config(raw, config_path=tmp_path / "deploy.yaml")

    assert parsed.steering_protection_interval_s == 0.25
    assert parsed.tactile_window_divisor == 1
    assert parsed.history_stride == 1
    assert not hasattr(parsed, "gate_tau")
    assert not hasattr(parsed, "gate_temperature")

    raw["gate_tau"] = 0.4
    with pytest.raises(ValueError, match="Deprecated"):
        frs_runtime_module.parse_frs_config(raw, config_path=tmp_path / "deploy.yaml")


def test_parse_frs_config_defaults_and_stores_temporal_ensemble_coeff(
    tmp_path: Path,
) -> None:
    frs_checkpoint = tmp_path / "frs"
    encoder_checkpoint = tmp_path / "encoder"
    frs_checkpoint.mkdir()
    encoder_checkpoint.mkdir()
    raw = {
        "checkpoint": str(frs_checkpoint),
        "tactile_encoder_checkpoint": str(encoder_checkpoint),
        "tactile_keys": ["left"],
        "tactile_window_divisor": 1,
        "reverse_steps": 1,
        "reverse_solver": "euler",
        "decode_steps": 1,
        "decode_solver": "euler",
    }

    disabled = frs_runtime_module.parse_frs_config(
        raw,
        config_path=tmp_path / "deploy.yaml",
    )
    raw["temporal_ensemble_coeff"] = 0.1
    enabled = frs_runtime_module.parse_frs_config(
        raw,
        config_path=tmp_path / "deploy.yaml",
    )

    assert disabled.temporal_ensemble_coeff is None
    assert enabled.temporal_ensemble_coeff == pytest.approx(0.1)


def test_parse_frs_config_defaults_and_stores_inactive_arm_xyz_threshold(
    tmp_path: Path,
) -> None:
    frs_checkpoint = tmp_path / "frs"
    encoder_checkpoint = tmp_path / "encoder"
    frs_checkpoint.mkdir()
    encoder_checkpoint.mkdir()
    raw = {
        "checkpoint": str(frs_checkpoint),
        "tactile_encoder_checkpoint": str(encoder_checkpoint),
        "tactile_keys": ["left"],
        "tactile_window_divisor": 1,
        "reverse_steps": 1,
        "reverse_solver": "euler",
        "decode_steps": 1,
        "decode_solver": "euler",
    }

    disabled = frs_runtime_module.parse_frs_config(
        raw,
        config_path=tmp_path / "deploy.yaml",
    )
    raw["inactive_arm_xyz_threshold_m"] = 0.00025
    enabled = frs_runtime_module.parse_frs_config(
        raw,
        config_path=tmp_path / "deploy.yaml",
    )

    assert disabled.inactive_arm_xyz_threshold_m is None
    assert enabled.inactive_arm_xyz_threshold_m == pytest.approx(0.00025)


@pytest.mark.parametrize(
    "value",
    [True, False, 0, -0.1, float("nan"), float("inf"), float("-inf"), "0.1"],
)
def test_frs_config_rejects_invalid_inactive_arm_xyz_threshold(
    tmp_path: Path,
    value: object,
) -> None:
    frs_checkpoint = tmp_path / "frs"
    encoder_checkpoint = tmp_path / "encoder"
    frs_checkpoint.mkdir()
    encoder_checkpoint.mkdir()
    raw = {
        "checkpoint": str(frs_checkpoint),
        "tactile_encoder_checkpoint": str(encoder_checkpoint),
        "tactile_keys": ["left"],
        "tactile_window_divisor": 1,
        "reverse_steps": 1,
        "reverse_solver": "euler",
        "decode_steps": 1,
        "decode_solver": "euler",
        "inactive_arm_xyz_threshold_m": value,
    }

    with pytest.raises(ValueError, match="inactive_arm_xyz_threshold_m"):
        frs_runtime_module.parse_frs_config(raw, config_path=tmp_path / "deploy.yaml")
    with pytest.raises(ValueError, match="inactive_arm_xyz_threshold_m"):
        frs_runtime_module.validate_frs_config_section(
            {
                "frs": {**raw, "checkpoint": "unused", "tactile_encoder_checkpoint": "unused"},
                "observation": {"data_type": "vitac"},
                "control": {"steps_per_inference": 2, "action_horizon": 2},
            }
        )


def test_parse_frs_config_defaults_and_stores_gripper_gain(
    tmp_path: Path,
) -> None:
    frs_checkpoint = tmp_path / "frs"
    encoder_checkpoint = tmp_path / "encoder"
    frs_checkpoint.mkdir()
    encoder_checkpoint.mkdir()
    raw = {
        "checkpoint": str(frs_checkpoint),
        "tactile_encoder_checkpoint": str(encoder_checkpoint),
        "tactile_keys": ["left"],
        "tactile_window_divisor": 1,
        "reverse_steps": 1,
        "reverse_solver": "euler",
        "decode_steps": 1,
        "decode_solver": "euler",
    }

    disabled = frs_runtime_module.parse_frs_config(
        raw,
        config_path=tmp_path / "deploy.yaml",
    )
    raw["gripper_gain"] = {
        "threshold": 0.1,
        "multiplier": 1.5,
        "above_multiplier": 2.0,
    }
    enabled = frs_runtime_module.parse_frs_config(
        raw,
        config_path=tmp_path / "deploy.yaml",
    )

    assert disabled.gripper_gain is None
    assert enabled.gripper_gain == pytest.approx((0.1, 1.5, 2.0))


@pytest.mark.parametrize(
    "value",
    [True, False, -0.1, float("nan"), float("inf"), float("-inf"), "0.1"],
)
def test_frs_config_rejects_invalid_temporal_ensemble_coeff(
    tmp_path: Path,
    value: object,
) -> None:
    frs_checkpoint = tmp_path / "frs"
    encoder_checkpoint = tmp_path / "encoder"
    frs_checkpoint.mkdir()
    encoder_checkpoint.mkdir()
    raw = {
        "checkpoint": str(frs_checkpoint),
        "tactile_encoder_checkpoint": str(encoder_checkpoint),
        "tactile_keys": ["left"],
        "tactile_window_divisor": 1,
        "reverse_steps": 1,
        "reverse_solver": "euler",
        "decode_steps": 1,
        "decode_solver": "euler",
        "temporal_ensemble_coeff": value,
    }

    with pytest.raises(ValueError, match="temporal_ensemble_coeff"):
        frs_runtime_module.parse_frs_config(raw, config_path=tmp_path / "deploy.yaml")
    with pytest.raises(ValueError, match="temporal_ensemble_coeff"):
        frs_runtime_module.validate_frs_config_section(
            {
                "frs": {**raw, "checkpoint": "unused", "tactile_encoder_checkpoint": "unused"},
                "observation": {"data_type": "vitac"},
                "control": {"steps_per_inference": 2, "action_horizon": 2},
            }
        )


@pytest.mark.parametrize(
    "value",
    [True, False, -0.1, float("nan"), float("inf"), float("-inf"), "0.1"],
)
def test_frs_config_rejects_invalid_protection_interval(value: object) -> None:
    config = {
        "frs": {
            "enabled": True,
            "checkpoint": "unused",
            "tactile_encoder_checkpoint": "unused",
            "tactile_keys": ["left"],
            "tactile_window_divisor": 1,
            "reverse_steps": 1,
            "reverse_solver": "euler",
            "decode_steps": 1,
            "decode_solver": "euler",
            "steering_protection_interval_s": value,
        },
        "observation": {"data_type": "vitac"},
        "control": {"steps_per_inference": 2, "action_horizon": 2},
    }

    with pytest.raises(ValueError, match="steering_protection_interval_s"):
        frs_runtime_module.validate_frs_config_section(config)


def test_frs_config_requires_full_horizon_steps_per_inference() -> None:
    config = {
        "frs": {
            "enabled": True,
            "checkpoint": "unused",
            "tactile_encoder_checkpoint": "unused",
            "tactile_keys": ["left"],
            "tactile_window_divisor": 1,
            "reverse_steps": 1,
            "reverse_solver": "euler",
            "decode_steps": 1,
            "decode_solver": "euler",
        },
        "observation": {"data_type": "vitac"},
        "control": {"steps_per_inference": 1, "action_horizon": 2},
    }

    with pytest.raises(ValueError, match="steps_per_inference.*action_horizon"):
        frs_runtime_module.validate_frs_config_section(config)


def test_frs_config_rejects_deprecated_gate_values() -> None:
    config = {
        "frs": {
            "enabled": True,
            "checkpoint": "unused",
            "tactile_encoder_checkpoint": "unused",
            "tactile_keys": ["left"],
            "tactile_window_divisor": 1,
            "reverse_steps": 1,
            "reverse_solver": "euler",
            "decode_steps": 1,
            "decode_solver": "euler",
            "gate_tau": 0.4,
        },
        "observation": {"data_type": "vitac"},
        "control": {"steps_per_inference": 2, "action_horizon": 2},
    }

    with pytest.raises(ValueError, match="Deprecated"):
        frs_runtime_module.validate_frs_config_section(config)


@pytest.mark.parametrize("key", ("gate_tau", "gate_temperature"))
def test_disabled_frs_config_rejects_deprecated_gate_values(key: str) -> None:
    with pytest.raises(ValueError, match="Deprecated"):
        frs_runtime_module.validate_frs_config_section(
            {"frs": {"enabled": False, key: 0.4}}
        )


def test_trace_v2_chunk_copies_and_freezes_source_base_and_timing() -> None:
    normalized = np.asarray([[[1.0], [2.0]]], dtype=np.float32)
    action = np.asarray([[[10.0], [20.0]]], dtype=np.float32)
    x_base = np.asarray([[[3.0], [4.0]]], dtype=np.float32)
    ready = FRSChunkReady(7, normalized, action, x_base, 100.25, 100.75)

    trace = remote_client._build_frs_chunk_trace(ready)

    assert trace["version"] == 2
    assert trace["kind"] == "frs_chunk"
    assert trace["chunk_id"] == 7
    np.testing.assert_array_equal(trace["action_vla_normalized"], normalized)
    np.testing.assert_array_equal(trace["action_vla"], action)
    np.testing.assert_array_equal(trace["x_base"], x_base)
    assert trace["prediction_started_at"] == 100.25
    assert trace["prediction_finished_at"] == 100.75
    for key in ("action_vla_normalized", "action_vla", "x_base"):
        assert trace[key].flags.writeable is False
        assert not np.shares_memory(trace[key], getattr(ready, key))

    normalized[...] = -1.0
    np.testing.assert_array_equal(trace["action_vla_normalized"], [[[1.0], [2.0]]])


def test_trace_v2_steer_records_request_decode_selection_timing_and_diagnostics() -> None:
    result = FRSSteerResult(
        chunk_id=7,
        request_id=11,
        action_index=1,
        action_vla_normalized=np.asarray([[[1.0], [2.0]]], dtype=np.float32),
        x_base=np.asarray([[[3.0], [4.0]]], dtype=np.float32),
        decoded_normalized=np.asarray([[[5.0], [6.0]]], dtype=np.float32),
        selected_normalized=np.asarray([6.0], dtype=np.float32),
        selected_action=np.asarray([60.0], dtype=np.float32),
        tactile_sequence_length=2,
        diagnostics=FRSDiagnostics(0.25, 2.0, 6.0),
        encode_started_at=101.0,
        encode_finished_at=101.1,
        decode_started_at=101.2,
        decode_finished_at=101.4,
    )
    request = FRSSteerRequest(
        chunk_id=7,
        request_id=11,
        action_index=1,
        target_timestamp=102.0,
        protection_applied=True,
        observation={},
    )

    trace = remote_client._build_frs_steer_trace(result, request)

    assert trace == {
        "version": 2,
        "kind": "frs_steer",
        "chunk_id": 7,
        "request_id": 11,
        "action_index": 1,
        "target_timestamp": 102.0,
        "protection_applied": True,
        "decoded_normalized": trace["decoded_normalized"],
        "selected_normalized": trace["selected_normalized"],
        "selected_action": trace["selected_action"],
        "tactile_sequence_length": 2,
        "encode_started_at": 101.0,
        "encode_finished_at": 101.1,
        "decode_started_at": 101.2,
        "decode_finished_at": 101.4,
        "frs_diagnostics": {
            "tactile_change": 0.25,
            "delta_rms": 2.0,
            "max_normalized_action_abs": 6.0,
        },
    }
    assert "status" not in trace
    np.testing.assert_array_equal(trace["decoded_normalized"], result.decoded_normalized)
    np.testing.assert_array_equal(trace["selected_normalized"], result.selected_normalized)
    np.testing.assert_array_equal(trace["selected_action"], result.selected_action)
    for key in ("decoded_normalized", "selected_normalized", "selected_action"):
        assert trace[key].flags.writeable is False
        assert not np.shares_memory(trace[key], getattr(result, key))
    assert not hasattr(result.diagnostics, "gate_weight")
    assert "gate_weight" not in trace["frs_diagnostics"]


def test_trace_v2_builder_exception_is_fail_open(caplog: pytest.LogCaptureFixture) -> None:
    def fail_builder() -> dict[str, object]:
        raise RuntimeError("trace serialization failed")

    assert remote_client._build_trace_or_none(fail_builder) is None
    assert "Omitting FRS trace after serialization failure" in caplog.text


def test_tactile_history_matches_clamped_training_indices() -> None:
    history = TactileHistory(window=4, stride=2, token_shape=(1, 1))
    history.reset(np.asarray([[0.0]], dtype=np.float32))
    for value in range(1, 7):
        history.append(np.asarray([[value]], dtype=np.float32))

    # Current=6 with offsets [6, 4, 2, 0], returned oldest -> newest.
    np.testing.assert_array_equal(
        history.window_tokens()[:, 0, 0],
        np.asarray([0.0, 2.0, 4.0, 6.0], dtype=np.float32),
    )


def test_tactile_history_clamps_short_episode_to_first_frame() -> None:
    history = TactileHistory(window=4, stride=3, token_shape=(1, 1))
    history.reset(np.asarray([[10.0]], dtype=np.float32))
    history.append(np.asarray([[11.0]], dtype=np.float32))

    np.testing.assert_array_equal(
        history.window_tokens()[:, 0, 0],
        np.asarray([10.0, 10.0, 10.0, 11.0], dtype=np.float32),
    )


def test_predict_chunk_unnormalizes_the_legacy_source_chunk_without_frs() -> None:
    class Preprocessor:
        @staticmethod
        def unnormalize_actions(actions):
            return actions * 10.0

    class Policy:
        config = SimpleNamespace(chunk_size=2, action_dim=1)
        preprocessor = Preprocessor()

        @staticmethod
        def predict_action_chunk(*args, **kwargs):
            del args, kwargs
            return jnp.ones((1, 2, 1), dtype=jnp.float32)

    action, normalized = remote_client._predict_chunk(
        Policy(),
        {},
        "task",
        seed=0,
        jit=False,
        num_steps=10,
        previous_chunk=None,
        inference_delay=None,
        execution_horizon=None,
    )

    np.testing.assert_array_equal(normalized, np.ones((2, 1), dtype=np.float32))
    np.testing.assert_array_equal(action, np.full((2, 1), 10.0, dtype=np.float32))


def test_frs_runtime_retains_vla_and_refined_normalized_chunks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = object.__new__(FRSRuntime)
    runtime.config = SimpleNamespace(
        tactile_keys=("left",),
        reverse_steps=2,
        reverse_solver="euler",
        decode_steps=2,
        decode_solver="euler",
        max_normalized_action_abs=8.0,
        max_normalized_delta_rms=4.0,
    )
    runtime._episode_baseline = np.zeros((1, 1), dtype=np.float32)
    runtime.baseline = np.array(runtime._episode_baseline, copy=True)
    runtime.history = SimpleNamespace(
        append=lambda tokens: None,
        window_tokens=lambda: np.zeros((1, 1, 1), dtype=np.float32),
    )
    runtime._encode_observation = lambda observation: np.ones((1, 1), dtype=np.float32)
    eval_observation = SimpleNamespace(state=jnp.asarray([[0.5]], dtype=jnp.float32))
    runtime._eval_observation = lambda policy, observation, task: eval_observation
    runtime.model = SimpleNamespace(config=SimpleNamespace(state_conditioning=True))

    monkeypatch.setattr(
        "deploy_smolvla.frs_runtime.tactile_change_from_tokens",
        lambda current, baseline: jnp.asarray([0.25], dtype=jnp.float32),
    )
    monkeypatch.setattr(
        "deploy_smolvla.frs_runtime.reverse_integrate_actions",
        lambda *args, **kwargs: args[2],
    )
    def decode(
        model,
        x_base,
        tactile,
        *,
        num_steps,
        solver,
        state=None,
    ):
        assert model is runtime.model
        assert num_steps == 2
        assert solver == "euler"
        assert state is eval_observation.state
        return x_base + 1.0

    monkeypatch.setattr("deploy_smolvla.frs_runtime.decode_actions", decode)

    vla = jnp.asarray([[[1.0], [2.0]]], dtype=jnp.float32)
    refined = runtime.steer(object(), {}, "task", vla)

    np.testing.assert_array_equal(runtime.last_vla_normalized, np.asarray(vla))
    np.testing.assert_array_equal(runtime.last_frs_normalized, np.asarray(refined))


def test_action_trace_contains_complete_vla_frs_chunks_timestamps_and_diagnostics() -> None:
    class Preprocessor:
        @staticmethod
        def unnormalize_actions(actions):
            return np.asarray(actions) * 10.0

    policy = SimpleNamespace(preprocessor=Preprocessor())
    frs_runtime = SimpleNamespace(
        last_vla_normalized=np.asarray([[[1.0], [2.0]]], dtype=np.float32),
        last_frs_normalized=np.asarray([[[3.0], [4.0]]], dtype=np.float32),
        last_diagnostics=SimpleNamespace(
            tactile_change=0.25,
            delta_rms=2.0,
            max_normalized_action_abs=4.0,
        ),
    )

    trace = remote_client._build_action_trace(
        policy,
        frs_runtime,
        inference_wall_start_s=100.25,
        inference_wall_end_s=100.75,
    )

    assert trace["version"] == 1
    np.testing.assert_array_equal(trace["vla_normalized"], [[1.0], [2.0]])
    np.testing.assert_array_equal(trace["vla_action"], [[10.0], [20.0]])
    np.testing.assert_array_equal(trace["frs_normalized"], [[3.0], [4.0]])
    np.testing.assert_array_equal(trace["frs_action"], [[30.0], [40.0]])
    assert trace["inference_started_at"] == 100.25
    assert trace["inference_finished_at"] == 100.75
    assert trace["frs_diagnostics"] == {
        "tactile_change": 0.25,
        "delta_rms": 2.0,
        "max_normalized_action_abs": 4.0,
    }


def test_action_trace_failure_is_omitted_without_raising() -> None:
    class Preprocessor:
        @staticmethod
        def unnormalize_actions(actions):
            del actions
            raise RuntimeError("trace-only unnormalization failed")

    policy = SimpleNamespace(preprocessor=Preprocessor())
    frs_runtime = SimpleNamespace(
        last_vla_normalized=np.asarray([[[1.0]]], dtype=np.float32),
        last_frs_normalized=np.asarray([[[2.0]]], dtype=np.float32),
        last_diagnostics=SimpleNamespace(
            tactile_change=0.25,
            delta_rms=1.0,
            max_normalized_action_abs=2.0,
        ),
    )

    assert (
        remote_client._build_action_trace_or_none(
            policy,
            frs_runtime,
            inference_wall_start_s=100.25,
            inference_wall_end_s=100.75,
        )
        is None
    )


def test_action_trace_contains_complete_direct_decoder_chunks_and_diagnostics() -> None:
    class Preprocessor:
        @staticmethod
        def unnormalize_actions(actions):
            return np.asarray(actions) * 10.0

    policy = SimpleNamespace(preprocessor=Preprocessor())
    base = np.asarray([[[1.0], [2.0]]], dtype=np.float32)
    direct = np.asarray([[[3.0], [4.0]]], dtype=np.float32)
    direct_runtime = SimpleNamespace(
        last_vla_normalized=base,
        last_direct_normalized=direct,
    )

    trace = remote_client._build_action_trace(
        policy,
        None,
        direct_decoder=direct_runtime,
        inference_wall_start_s=100.25,
        inference_wall_end_s=100.75,
    )

    assert trace["version"] == 1
    assert trace["prediction_source"] == "direct_decode"
    np.testing.assert_array_equal(trace["vla_normalized"], base[0])
    np.testing.assert_array_equal(trace["frs_normalized"], direct[0])
    np.testing.assert_array_equal(trace["vla_action"], base[0] * 10.0)
    np.testing.assert_array_equal(trace["frs_action"], direct[0] * 10.0)
    assert trace["frs_diagnostics"] == {
        "delta_rms": float(np.sqrt(np.mean(np.square(direct - base)))),
        "max_normalized_action_abs": float(np.max(np.abs(direct))),
    }


@pytest.mark.parametrize(
    "direct_runtime",
    [
        SimpleNamespace(last_vla_normalized=np.ones((1, 2, 1), dtype=np.float32)),
        SimpleNamespace(
            last_vla_normalized=np.ones((1,), dtype=np.float32),
            last_direct_normalized=np.ones((1, 2, 1), dtype=np.float32),
        ),
        SimpleNamespace(
            last_vla_normalized=np.ones((1, 2, 1), dtype=np.float32),
            last_direct_normalized=np.ones((1, 1, 1), dtype=np.float32),
        ),
    ],
)
def test_direct_action_trace_missing_or_malformed_snapshots_are_omitted(
    direct_runtime: SimpleNamespace,
) -> None:
    policy = SimpleNamespace(
        preprocessor=SimpleNamespace(unnormalize_actions=lambda actions: actions)
    )

    assert (
        remote_client._build_action_trace_or_none(
            policy,
            None,
            direct_decoder=direct_runtime,
            inference_wall_start_s=100.25,
            inference_wall_end_s=100.75,
        )
        is None
    )


def test_bridge_send_action_keeps_legacy_payload_without_trace_and_adds_keyword_trace() -> None:
    bridge = object.__new__(RobotBridgeClient)
    messages: list[dict[str, object]] = []
    bridge._send = messages.append
    action = np.asarray([[1.0, 2.0]], dtype=np.float32)

    bridge.send_action(action, 7)
    bridge.send_action(action, 8, trace={"version": 1})

    assert messages[0] == {"type": "action", "obs_seq": 7, "action": action}
    assert messages[1] == {
        "type": "action",
        "obs_seq": 8,
        "action": action,
        "trace": {"version": 1},
    }


def _contract_runtime() -> tuple[FRSRuntime, SimpleNamespace]:
    runtime = object.__new__(FRSRuntime)
    runtime.config = SimpleNamespace(
        tactile_keys=("left", "right", "left_1", "right_1"),
        decode_steps=10,
        decode_solver="euler",
        reverse_steps=20,
        reverse_solver="slerpflow",
        verify_source_checkpoint_fingerprint=False,
    )
    runtime.embedding_dim = 512
    runtime.model = SimpleNamespace(
        config=SimpleNamespace(
            action_dim=20,
            action_horizon=10,
            num_tactile_tokens=4,
            resnet_embedding_dim=512,
            tactile_window=10,
        )
    )
    runtime.metadata = {
        "extra_metadata": {
            "loss_mode": "gated",
            "decoder_input_version": 2,
            "history_stride": 3,
            "tactile_window": 10,
            "validation_steps": 10,
            "validation_solver": "euler",
            "cache_configuration": {
                "model_sample_steps": 10,
                "reverse_steps": 20,
                "reverse_solver": "slerpflow",
                "normalization_source": "checkpoint",
                "reverse_integration_version": 1,
            },
        }
    }
    policy = SimpleNamespace(
        config=SimpleNamespace(action_dim=20, chunk_size=10),
        checkpoint=Path("unused"),
    )
    return runtime, policy


@pytest.mark.parametrize(
    "objective_metadata",
    [
        {"loss_mode": "gated"},
        {
            "loss_mode": "bimanual_gated",
            "loss_objective_version": 2,
            "action_slices": {"left": [0, 10], "right": [10, 20]},
            "wrist_token_indices": {"left": [0, 1], "right": [2, 3]},
        },
    ],
    ids=("legacy-gated", "bimanual-gated"),
)
def test_frs_contract_accepts_matching_training_metadata(
    objective_metadata: dict[str, object],
) -> None:
    runtime, policy = _contract_runtime()
    runtime.metadata["extra_metadata"].update(objective_metadata)

    runtime._validate_contract(policy, source_sample_steps=10)


@pytest.mark.parametrize(
    ("field_name", "invalid_value", "expected_error"),
    [
        (
            "action_slices",
            {"left": [0, 9], "right": [10, 20]},
            r"action_slices\.left",
        ),
        (
            "wrist_token_indices",
            {"left": [0, 1], "right": [1, 3]},
            r"wrist_token_indices\.right",
        ),
    ],
)
def test_frs_contract_rejects_invalid_bimanual_objective_metadata(
    field_name: str,
    invalid_value: object,
    expected_error: str,
) -> None:
    runtime, policy = _contract_runtime()
    runtime.metadata["extra_metadata"].update(
        {
            "loss_mode": "bimanual_gated",
            "loss_objective_version": 2,
            "action_slices": {"left": [0, 10], "right": [10, 20]},
            "wrist_token_indices": {"left": [0, 1], "right": [2, 3]},
        }
    )
    runtime.metadata["extra_metadata"][field_name] = invalid_value

    with pytest.raises(ValueError, match=expected_error):
        runtime._validate_contract(policy, source_sample_steps=10)


def test_frs_contract_allows_deployment_solver_to_differ_from_validation_solver() -> None:
    runtime, policy = _contract_runtime()
    runtime.config.decode_solver = "fireflow"
    runtime.metadata["extra_metadata"]["validation_solver"] = "euler"

    runtime._validate_contract(policy, source_sample_steps=10)


def test_frs_contract_allows_decode_steps_to_differ_from_validation_steps() -> None:
    runtime, policy = _contract_runtime()
    runtime.config.decode_steps = 5
    runtime.metadata["extra_metadata"]["validation_steps"] = 10

    runtime._validate_contract(policy, source_sample_steps=10)


@pytest.mark.parametrize("version", [0, 1, 3, "2", 2.0, True, None])
def test_frs_contract_rejects_invalid_decoder_input_versions(version: object) -> None:
    runtime, policy = _contract_runtime()
    runtime.metadata["extra_metadata"]["decoder_input_version"] = version

    with pytest.raises(ValueError, match="decoder_input_version"):
        runtime._validate_contract(policy, source_sample_steps=10)


def test_frs_contract_accepts_matching_state_conditioning() -> None:
    runtime, policy = _contract_runtime()
    runtime.model.config.state_conditioning = True
    runtime.model.config.state_dim = 20
    runtime.metadata["extra_metadata"]["state_conditioning"] = True
    policy.config.state_dim = 20

    runtime._validate_contract(policy, source_sample_steps=10)


def test_frs_contract_rejects_different_source_sampling_steps() -> None:
    runtime, policy = _contract_runtime()

    with pytest.raises(ValueError, match="sample_steps"):
        runtime._validate_contract(policy, source_sample_steps=8)
