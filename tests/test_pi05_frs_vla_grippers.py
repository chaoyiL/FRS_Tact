from pathlib import Path
from types import SimpleNamespace

import numpy as np
import deploy_pi05.frs_runtime as frs_runtime_module
import pytest

from deploy_pi05.frs_runtime import FRSRuntime, _validate_loss_contract
from train_pi05_frs.utils.objective_schema import composite_gated_objective_metadata
from deploy_pi05.frs_config import GripperHysteresisConfig, Task1MotionGainConfig


def _runtime() -> FRSRuntime:
    runtime = object.__new__(FRSRuntime)
    runtime.config = SimpleNamespace(
        max_normalized_action_abs=8.0,
        max_normalized_delta_rms=4.0,
    )
    runtime.policy = SimpleNamespace(
        config=SimpleNamespace(
            action_horizon=3,
            action_dim=20,
            robot_action_dim=20,
        ),
    )
    runtime._gripper_action_indices = (9, 19)
    runtime.metadata = {"extra_metadata": {"loss_mode": "bimanual_gated"}}
    runtime.task1_motion_gain = Task1MotionGainConfig(
        approach_translation_gain=1.0,
        right_approach_translation_gain=1.0,
        translation_gain=1.0,
        rotation_gain=1.0,
    )
    runtime._action_vla_normalized = np.zeros((1, 3, 20), dtype=np.float32)
    runtime._action_vla_normalized[0, :, 9] = (0.11, 0.22, 0.33)
    runtime._action_vla_normalized[0, :, 19] = (0.44, 0.55, 0.66)
    return runtime


def test_pi05_runtime_accepts_arm9_vla_gripper_objective() -> None:
    _validate_loss_contract(
        {
            "loss_mode": "composite_gated",
            **composite_gated_objective_metadata(),
        },
        action_dim=10,
        tactile_keys=(
            "observation.images.tactile_left_1",
            "observation.images.tactile_right_1",
        ),
    )


def test_validated_decoded_restores_pi05_vla_grippers_before_safety_checks() -> None:
    runtime = _runtime()
    decoded = np.full((1, 3, 20), 0.5, dtype=np.float32)
    decoded[..., 9] = 100.0
    decoded[..., 19] = -100.0

    validated, _, max_abs, _ = runtime._validated_decoded(decoded)

    np.testing.assert_array_equal(
        validated[..., (9, 19)],
        runtime._action_vla_normalized[..., (9, 19)],
    )
    np.testing.assert_array_equal(validated[..., :9], decoded[..., :9])
    np.testing.assert_array_equal(validated[..., 10:19], decoded[..., 10:19])
    assert max_abs == np.float32(0.66)


def test_parse_frs_config_defaults_and_accepts_deployment_guards(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint"
    encoder = tmp_path / "encoder"
    checkpoint.mkdir()
    encoder.mkdir()
    raw = {
        "checkpoint": str(checkpoint),
        "tactile_encoder_checkpoint": str(encoder),
        "tactile_keys": ["left", "right"],
        "tactile_window_divisor": 5,
        "reverse_steps": 50,
        "reverse_solver": "slerpflow",
        "decode_steps": 10,
        "decode_solver": "fireflow",
    }

    defaults = frs_runtime_module.parse_frs_config(
        raw, config_path=tmp_path / "deploy.yaml"
    )
    enabled = frs_runtime_module.parse_frs_config(
        {
            **raw,
            "hard_low_gate_bypass": True,
            "max_normalized_residual_abs": 0.5,
        },
        config_path=tmp_path / "deploy.yaml",
    )

    assert defaults.hard_low_gate_bypass is False
    assert defaults.max_normalized_residual_abs is None
    assert enabled.hard_low_gate_bypass is True
    assert enabled.max_normalized_residual_abs == pytest.approx(0.5)


def test_validated_decoded_clamps_only_arm9_residual_and_preserves_gripper() -> None:
    runtime = _runtime()
    runtime.policy.config.action_dim = 10
    runtime.policy.config.robot_action_dim = 10
    runtime._gripper_action_indices = (9,)
    runtime.metadata = {
        "extra_metadata": {
            "loss_mode": "composite_gated",
            "loss_objective_version": 2,
            "steered_action_dim": 9,
        }
    }
    runtime.config.max_normalized_residual_abs = 0.5
    runtime._action_vla_normalized = np.zeros((1, 3, 10), dtype=np.float32)
    runtime._action_vla_normalized[..., 9] = 0.25
    decoded = np.full((1, 3, 10), 2.0, dtype=np.float32)
    decoded[..., 1] = -3.0
    decoded[..., 9] = 100.0

    validated, delta, max_abs, clamped = runtime._validated_decoded(decoded)

    np.testing.assert_array_equal(validated[..., 0], 0.5)
    np.testing.assert_array_equal(validated[..., 1], -0.5)
    np.testing.assert_array_equal(validated[..., 2:9], 0.5)
    np.testing.assert_array_equal(validated[..., 9], 0.25)
    assert delta == pytest.approx(np.sqrt((9 * 0.25) / 10))
    assert max_abs == pytest.approx(0.5)
    assert clamped is True


class _Arm9History:
    window = 10

    def __init__(self) -> None:
        self.append_calls = 0

    def append(self, _tokens: np.ndarray) -> None:
        self.append_calls += 1

    def window_tokens(self) -> np.ndarray:
        return np.zeros((10, 2, 1), dtype=np.float32)


def test_low_gate_bypass_uses_checkpoint_gate_and_skips_decode_and_ensemble(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = _runtime()
    runtime.task = 0
    runtime.policy.config.action_dim = 10
    runtime.policy.config.robot_action_dim = 10
    runtime.policy.unnormalize_actions = lambda action: np.asarray(action) * 10.0
    runtime._gripper_action_indices = (9,)
    runtime.config.hard_low_gate_bypass = True
    runtime.config.max_normalized_residual_abs = 0.5
    runtime.config.temporal_ensemble_coeff = 0.1
    runtime.config.decode_steps = 10
    runtime.config.decode_solver = "fireflow"
    runtime.metadata = {
        "extra_metadata": {
            "loss_mode": "composite_gated",
            "loss_objective_version": 2,
            "gate_tau": 0.4,
            "gate_temperature": 0.1,
            "low_gate_threshold": 0.3,
            "steered_action_dim": 9,
        }
    }
    vla = np.arange(30, dtype=np.float32).reshape(1, 3, 10) / 100.0
    runtime._action_vla_normalized = vla
    runtime._action_vla = vla * 10.0
    runtime._action_vla[0, 1, 9] = 0.123
    runtime._active_chunk_id = 4
    runtime._x_base = np.zeros_like(vla)
    runtime._x_base_device = runtime._x_base
    runtime._episode_baseline = np.zeros((2, 1), dtype=np.float32)
    runtime.history = _Arm9History()
    runtime._request_results = {
        10: (
            4,
            0,
            b"old",
            SimpleNamespace(decoded_normalized=np.full_like(vla, 99.0)),
        )
    }
    runtime._last_action_index = 0
    runtime.last_diagnostics = None
    runtime.last_vla_normalized = None
    runtime.last_frs_normalized = None
    runtime._payload_hash = lambda _observation: b"current"
    runtime._encode_observation = lambda _observation: np.zeros(
        (2, 1), dtype=np.float32
    )
    runtime._normalized_state = lambda _observation: (_ for _ in ()).throw(
        AssertionError("low-Gate bypass must run before state/decode")
    )
    runtime._decode_action_chunk = lambda *_args, **_kwargs: (_ for _ in ()).throw(
        AssertionError("low-Gate bypass must skip decode")
    )
    monkeypatch.setattr(
        frs_runtime_module,
        "tactile_change_from_tokens",
        lambda *_args: np.asarray([0.0], dtype=np.float32),
    )

    result = runtime.steer_action(4, 11, {}, 1)

    np.testing.assert_array_equal(result.decoded_normalized, vla)
    np.testing.assert_array_equal(result.selected_normalized, vla[0, 1])
    expected_robot = vla[0, 1] * 10.0
    expected_robot[9] = 0.123
    np.testing.assert_array_equal(result.selected_action, expected_robot)
    assert runtime.history.append_calls == 1
    assert result.diagnostics.gate_weight == pytest.approx(
        1.0 / (1.0 + np.exp(4.0))
    )
    assert result.diagnostics.bypassed is True
    assert result.diagnostics.clamped is False
    assert result.diagnostics.delta_rms == pytest.approx(0.0)


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
        right_approach_translation_gain=1.5,
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
    np.testing.assert_allclose(result.selected_action[10:13], [9.0, 9.0, 9.0])
    np.testing.assert_array_equal(
        result.selected_action[[9, 19]], runtime._action_vla[0, 1, [9, 19]]
    )


def test_task0_bypasses_task1_motion_gain_in_robot_space(monkeypatch) -> None:
    runtime = _runtime()
    runtime.task = 0
    runtime.task1_motion_gain = Task1MotionGainConfig(
        approach_translation_gain=1.2,
        right_approach_translation_gain=1.5,
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
