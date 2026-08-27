from types import SimpleNamespace

import numpy as np
import deploy_pi05.frs_runtime as frs_runtime_module

from deploy_pi05.frs_runtime import FRSRuntime
from deploy_pi05.frs_config import GripperHysteresisConfig, Task1MotionGainConfig


def _runtime() -> FRSRuntime:
    runtime = object.__new__(FRSRuntime)
    runtime.config = SimpleNamespace(
        max_normalized_action_abs=8.0,
        max_normalized_delta_rms=4.0,
    )
    runtime.policy = SimpleNamespace(
        config=SimpleNamespace(action_horizon=3, action_dim=20),
    )
    runtime.metadata = {"extra_metadata": {"loss_mode": "bimanual_gated"}}
    runtime.task1_motion_gain = Task1MotionGainConfig(
        approach_translation_gain=1.0,
        translation_gain=1.0,
        rotation_gain=1.0,
    )
    runtime._action_vla_normalized = np.zeros((1, 3, 20), dtype=np.float32)
    runtime._action_vla_normalized[0, :, 9] = (0.11, 0.22, 0.33)
    runtime._action_vla_normalized[0, :, 19] = (0.44, 0.55, 0.66)
    return runtime


def test_validated_decoded_restores_pi05_vla_grippers_before_safety_checks() -> None:
    runtime = _runtime()
    decoded = np.full((1, 3, 20), 0.5, dtype=np.float32)
    decoded[..., 9] = 100.0
    decoded[..., 19] = -100.0

    validated, _, max_abs = runtime._validated_decoded(decoded)

    np.testing.assert_array_equal(
        validated[..., (9, 19)],
        runtime._action_vla_normalized[..., (9, 19)],
    )
    np.testing.assert_array_equal(validated[..., :9], decoded[..., :9])
    np.testing.assert_array_equal(validated[..., 10:19], decoded[..., 10:19])
    assert max_abs == np.float32(0.66)


class _History:
    window = 10

    def append(self, _tokens: np.ndarray) -> None:
        pass

    def window_tokens(self) -> np.ndarray:
        return np.zeros((10, 4, 1), dtype=np.float32)


def test_steer_action_keeps_robot_space_vla_grippers_and_ignores_legacy_gain() -> None:
    runtime = _runtime()
    runtime.task = 1
    runtime.config.temporal_ensemble_coeff = 0.1
    runtime.config.gripper_gain = (10.0, 0.0)
    runtime.config.decode_steps = 1
    runtime.config.decode_solver = "euler"
    runtime.policy.config.robot_action_dim = 20
    runtime.policy.unnormalize_actions = lambda action: np.asarray(action) * 10.0
    runtime._active_chunk_id = 4
    runtime._action_vla = runtime._action_vla_normalized * 10.0
    runtime._x_base = np.zeros((1, 3, 20), dtype=np.float32)
    runtime._x_base_device = runtime._x_base
    runtime._episode_baseline = np.zeros((4, 1), dtype=np.float32)
    runtime.history = _History()
    runtime._request_results = {}
    runtime._last_action_index = None
    runtime.last_diagnostics = None
    runtime.last_vla_normalized = None
    runtime.last_frs_normalized = None
    runtime.gripper_hysteresis = GripperHysteresisConfig(
        left_close_threshold=0.08,
        left_reopen_threshold=0.10,
        left_closed_command=0.01,
        right_close_threshold=0.09,
        right_reopen_threshold=0.10,
        right_closed_command=0.01,
    )
    runtime._vla_translation_latched = (True, False)
    decoded = np.full((1, 3, 20), 0.5, dtype=np.float32)
    decoded[..., 9] = -0.8
    decoded[..., 19] = -0.9
    runtime._payload_hash = lambda _observation: b"tactile"
    runtime._encode_observation = lambda _observation: np.zeros((4, 1), dtype=np.float32)
    runtime._normalized_state = lambda _observation: None
    runtime._decode_action_chunk = lambda *_args, **_kwargs: decoded

    result = runtime.steer_action(4, 10, {}, 1)

    np.testing.assert_array_equal(
        result.selected_normalized[[9, 19]],
        runtime._action_vla_normalized[0, 1, [9, 19]],
    )
    np.testing.assert_array_equal(
        result.selected_action[[9, 19]],
        runtime._action_vla[0, 1, [9, 19]],
    )
    assert result.selected_action[0] == 5.0


def test_steer_action_uses_vla_translation_for_only_the_latched_arm() -> None:
    runtime = _runtime()
    runtime.task = 1
    runtime.config.temporal_ensemble_coeff = None
    runtime.config.decode_steps = 1
    runtime.config.decode_solver = "euler"
    runtime.policy.config.robot_action_dim = 20
    runtime.policy.unnormalize_actions = lambda action: np.asarray(action) * 10.0
    runtime._active_chunk_id = 4
    runtime._action_vla_normalized[0, 1, 0:3] = (0.1, 0.2, 0.3)
    runtime._action_vla_normalized[0, 1, 10:13] = (0.4, 0.5, 0.6)
    runtime._action_vla = runtime._action_vla_normalized * 10.0
    runtime._action_vla[0, 1, [9, 19]] = (0.09, 0.11)
    runtime._x_base = np.zeros((1, 3, 20), dtype=np.float32)
    runtime._x_base_device = runtime._x_base
    runtime._episode_baseline = np.zeros((4, 1), dtype=np.float32)
    runtime.history = _History()
    runtime._request_results = {}
    runtime._last_action_index = None
    runtime.last_diagnostics = None
    runtime.last_vla_normalized = None
    runtime.last_frs_normalized = None
    runtime.gripper_hysteresis = GripperHysteresisConfig(
        left_close_threshold=0.08,
        left_reopen_threshold=0.10,
        left_closed_command=0.01,
        right_close_threshold=0.09,
        right_reopen_threshold=0.10,
        right_closed_command=0.01,
    )
    runtime._vla_translation_latched = (True, False)
    decoded = np.full((1, 3, 20), 0.5, dtype=np.float32)
    runtime._payload_hash = lambda _observation: b"tactile"
    runtime._encode_observation = lambda _observation: np.zeros((4, 1), dtype=np.float32)
    runtime._normalized_state = lambda _observation: None
    runtime._decode_action_chunk = lambda *_args, **_kwargs: decoded

    result = runtime.steer_action(4, 10, {}, 1)

    np.testing.assert_array_equal(
        result.selected_normalized[0:3], runtime._action_vla_normalized[0, 1, 0:3]
    )
    np.testing.assert_array_equal(result.selected_normalized[3:9], decoded[0, 1, 3:9])
    np.testing.assert_array_equal(result.selected_normalized[10:19], decoded[0, 1, 10:19])
    assert runtime._vla_translation_latched == (True, False)


def test_steer_action_applies_task1_translation_gain_after_unnormalization() -> None:
    runtime = _runtime()
    runtime.task = 1
    runtime.task1_motion_gain = Task1MotionGainConfig(
        approach_translation_gain=1.2,
        translation_gain=1.5,
        rotation_gain=1.0,
    )
    runtime.config.temporal_ensemble_coeff = None
    runtime.config.decode_steps = 1
    runtime.config.decode_solver = "euler"
    runtime.policy.config.robot_action_dim = 20
    runtime.policy.unnormalize_actions = lambda action: np.asarray(action) * 10.0 + 1.0
    runtime._active_chunk_id = 4
    runtime._action_vla_normalized[0, 1, 0:3] = (0.1, 0.2, 0.3)
    runtime._action_vla = runtime._action_vla_normalized * 10.0
    runtime._action_vla[0, 1, [9, 19]] = (0.07, 0.11)
    runtime._x_base = np.zeros((1, 3, 20), dtype=np.float32)
    runtime._x_base_device = runtime._x_base
    runtime._episode_baseline = np.zeros((4, 1), dtype=np.float32)
    runtime.history = _History()
    runtime._request_results = {}
    runtime._last_action_index = None
    runtime.last_diagnostics = None
    runtime.last_vla_normalized = None
    runtime.last_frs_normalized = None
    runtime.gripper_hysteresis = GripperHysteresisConfig(
        left_close_threshold=0.08,
        left_reopen_threshold=0.10,
        left_closed_command=0.01,
        right_close_threshold=0.09,
        right_reopen_threshold=0.10,
        right_closed_command=0.01,
    )
    runtime._vla_translation_latched = (False, False)
    decoded = np.full((1, 3, 20), 0.5, dtype=np.float32)
    runtime._payload_hash = lambda _observation: b"tactile"
    runtime._encode_observation = lambda _observation: np.zeros((4, 1), dtype=np.float32)
    runtime._normalized_state = lambda _observation: None
    runtime._decode_action_chunk = lambda *_args, **_kwargs: decoded

    result = runtime.steer_action(4, 10, {}, 1)

    np.testing.assert_allclose(result.selected_action[0:3], [3.0, 4.5, 6.0])
    np.testing.assert_allclose(result.selected_action[10:13], [7.2, 7.2, 7.2])
    np.testing.assert_array_equal(
        result.selected_action[[9, 19]], runtime._action_vla[0, 1, [9, 19]]
    )


def test_task0_bypasses_task1_motion_gain_in_robot_space(monkeypatch) -> None:
    runtime = _runtime()
    runtime.task = 0
    runtime.task1_motion_gain = Task1MotionGainConfig(
        approach_translation_gain=1.2,
        translation_gain=1.5,
        rotation_gain=1.25,
    )
    runtime.config.temporal_ensemble_coeff = None
    runtime.config.decode_steps = 1
    runtime.config.decode_solver = "euler"
    runtime.policy.config.robot_action_dim = 20
    runtime.policy.unnormalize_actions = lambda action: np.asarray(action)
    runtime._active_chunk_id = 4
    runtime._action_vla_normalized[0, 1, 0:3] = (0.1, 0.2, 0.3)
    runtime._action_vla = runtime._action_vla_normalized.copy()
    runtime._action_vla[0, 1, [9, 19]] = (0.07, 0.11)
    runtime._x_base = np.zeros((1, 3, 20), dtype=np.float32)
    runtime._x_base_device = runtime._x_base
    runtime._episode_baseline = np.zeros((4, 1), dtype=np.float32)
    runtime.history = _History()
    runtime._request_results = {}
    runtime._last_action_index = None
    runtime.last_diagnostics = None
    runtime.last_vla_normalized = None
    runtime.last_frs_normalized = None
    runtime.gripper_hysteresis = GripperHysteresisConfig(
        left_close_threshold=0.08,
        left_reopen_threshold=0.10,
        left_closed_command=0.01,
        right_close_threshold=0.09,
        right_reopen_threshold=0.10,
        right_closed_command=0.01,
    )
    runtime._vla_translation_latched = (True, True)
    decoded = np.full((1, 3, 20), 0.5, dtype=np.float32)
    runtime._payload_hash = lambda _observation: b"tactile"
    runtime._encode_observation = lambda _observation: np.zeros((4, 1), dtype=np.float32)
    runtime._normalized_state = lambda _observation: None
    runtime._decode_action_chunk = lambda *_args, **_kwargs: decoded

    def task1_gain_must_not_run(*_args, **_kwargs):
        raise AssertionError("Task 0 must bypass Task 1 motion gains")

    monkeypatch.setattr(
        frs_runtime_module, "_apply_task1_motion_gain", task1_gain_must_not_run
    )

    result = runtime.steer_action(4, 10, {}, 1)

    expected = decoded[0, 1].copy()
    expected[[9, 19]] = runtime._action_vla[0, 1, [9, 19]]
    np.testing.assert_array_equal(result.selected_action, expected)
    np.testing.assert_array_equal(result.selected_action[0:9], decoded[0, 1, 0:9])
    np.testing.assert_array_equal(result.selected_action[10:19], decoded[0, 1, 10:19])
    np.testing.assert_array_equal(
        result.selected_action[[9, 19]], runtime._action_vla[0, 1, [9, 19]]
    )
    assert runtime._vla_translation_latched == (False, False)
