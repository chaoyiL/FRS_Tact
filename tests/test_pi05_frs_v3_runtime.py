from __future__ import annotations

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx

from deploy_pi05 import frs_runtime as runtime_module
from deploy_pi05.frs_inference import decoder as decoder_module


def _v3_metadata() -> dict[str, object]:
    return {
        "loss_mode": "learned_residual_gated",
        "loss_objective_version": 3,
        "model_architecture": "single_arm_vla_residual_gate_v1",
        "action_dim": 10,
        "steered_action_dim": 9,
        "gripper_index": 9,
        "gripper_policy": "vla_runtime_preserved",
        "gate_granularity": "chunk",
        "residual_parameterization": "bounded_normalized_vla_additive",
        "gate_label_policy": "arm9_chunk_mse_two_threshold",
        "oracle_safe_mse_threshold": 0.01,
        "oracle_repair_mse_threshold": 0.04,
        "residual_bound": 0.5,
    }


def test_v3_checkpoint_contract_is_accepted_and_strict() -> None:
    metadata = _v3_metadata()

    runtime_module._validate_loss_contract(
        metadata,
        action_dim=10,
        tactile_keys=("tactile_left_1", "tactile_right_1"),
    )

    invalid = {**metadata, "gate_granularity": "step"}
    with pytest.raises(ValueError, match="gate_granularity"):
        runtime_module._validate_loss_contract(
            invalid,
            action_dim=10,
            tactile_keys=("tactile_left_1", "tactile_right_1"),
        )


def test_v3_inference_helper_applies_gate_to_arm9_and_preserves_gripper() -> None:
    config = decoder_module.DecoderConfig(
        action_dim=10,
        action_horizon=3,
        tactile_window=2,
        gru_hidden_dim=8,
        resnet_embedding_dim=3,
        model_dim=8,
        depth=1,
        num_heads=2,
        mlp_ratio=2,
        num_tactile_tokens=2,
        state_conditioning=True,
        state_dim=7,
        output_mode="learned_residual_gate",
        residual_bound=0.5,
    )
    model = decoder_module.TactileConditionedFlowDecoder(config, rngs=nnx.Rngs(0))
    vla = jnp.arange(30, dtype=jnp.float32).reshape(1, 3, 10) / 10.0
    executed, residual, gate_logits = decoder_module.infer_learned_residual_gate(
        model,
        vla,
        jnp.zeros((1, 2, 2, 3), dtype=jnp.float32),
        jnp.zeros((1, 2, 3), dtype=jnp.float32),
        state=jnp.zeros((1, 7), dtype=jnp.float32),
    )

    expected_arm = vla[..., :9] + jax.nn.sigmoid(gate_logits)[:, None, :] * residual
    np.testing.assert_allclose(executed[..., :9], expected_arm, atol=1e-6)
    np.testing.assert_array_equal(executed[..., 9], vla[..., 9])
    assert bool(jnp.all(jnp.abs(residual) <= 0.5 + 1e-6))


def test_v3_decoder_exposes_checkpoint_residual_gate_configuration() -> None:
    config = decoder_module.DecoderConfig(
        action_dim=10,
        action_horizon=3,
        tactile_window=2,
        gru_hidden_dim=8,
        resnet_embedding_dim=3,
        model_dim=8,
        depth=1,
        num_heads=2,
        mlp_ratio=2,
        num_tactile_tokens=2,
        state_conditioning=True,
        state_dim=7,
        output_mode="learned_residual_gate",
        residual_bound=0.5,
    )
    assert config.output_mode == "learned_residual_gate"
    assert config.residual_bound == pytest.approx(0.5)
    assert hasattr(decoder_module.TactileConditionedFlowDecoder, "predict_residual_gate")


def test_runtime_routes_v3_checkpoint_to_learned_residual_inference(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = object.__new__(runtime_module.FRSRuntime)
    runtime.model = SimpleNamespace(config=SimpleNamespace())
    runtime.metadata = {"extra_metadata": _v3_metadata()}
    runtime._episode_baseline = np.full((2, 3), 2.0, dtype=np.float32)
    vla = np.arange(30, dtype=np.float32).reshape(1, 3, 10) / 10.0
    tactile = jnp.ones((1, 2, 2, 3), dtype=jnp.float32)
    state = jnp.ones((1, 7), dtype=jnp.float32)
    expected = jnp.asarray(vla + 0.1)
    calls: list[tuple[object, ...]] = []

    def infer(model, vla_action, tactile_seq, episode_baseline, *, state):
        calls.append((model, vla_action, tactile_seq, episode_baseline, state))
        return (
            expected,
            jnp.full((1, 3, 9), 0.2, dtype=jnp.float32),
            jnp.zeros((1, 1), dtype=jnp.float32),
        )

    monkeypatch.setattr(
        runtime_module,
        "infer_learned_residual_gate",
        infer,
        raising=False,
    )

    decoded, residual, gate_logits = runtime._infer_action_chunk(
        jnp.zeros_like(vla),
        tactile,
        frozen_endpoint=vla,
        state=state,
    )

    np.testing.assert_array_equal(decoded, expected)
    assert residual is not None and residual.shape == (1, 3, 9)
    assert gate_logits is not None and gate_logits.shape == (1, 1)
    assert len(calls) == 1
    np.testing.assert_array_equal(calls[0][1], vla)
    np.testing.assert_array_equal(calls[0][3], runtime._episode_baseline[None, ...])


def test_runtime_chunk_reset_clears_v3_debug_outputs() -> None:
    runtime = object.__new__(runtime_module.FRSRuntime)
    runtime.last_gate_probability = 0.75
    runtime.last_bounded_residual = np.ones((1, 3, 9), dtype=np.float32)

    runtime._clear_chunk_state()

    assert runtime.last_gate_probability is None
    assert runtime.last_bounded_residual is None


def test_v3_begin_chunk_skips_reverse_and_uses_vla_as_x_base() -> None:
    normalized = np.arange(30, dtype=np.float32).reshape(1, 3, 10) / 10.0

    class Policy:
        config = SimpleNamespace(
            action_horizon=3,
            action_dim=10,
            robot_action_dim=10,
        )

        def predict_action_chunk(self, *_args, **_kwargs):
            return normalized

        def reverse_action_chunk(self, *_args, **_kwargs):
            raise AssertionError("objective-v3 must not run reverse flow")

        def unnormalize_actions(self, actions):
            return np.asarray(actions) * 10.0

    runtime = object.__new__(runtime_module.FRSRuntime)
    runtime.policy = Policy()
    runtime.metadata = {"extra_metadata": _v3_metadata()}
    runtime.config = SimpleNamespace(reverse_steps=50, reverse_solver="slerpflow")
    runtime._episode_baseline = np.zeros((2, 3), dtype=np.float32)
    runtime._active_chunk_id = None

    ready = runtime.begin_chunk(7, {}, "insert", seed=0, num_steps=10)

    np.testing.assert_array_equal(ready.action_vla_normalized, normalized)
    np.testing.assert_array_equal(ready.x_base, normalized)
    np.testing.assert_array_equal(runtime._x_base, normalized)
