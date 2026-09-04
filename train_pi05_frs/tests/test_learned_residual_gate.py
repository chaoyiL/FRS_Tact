from __future__ import annotations

import copy

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx

from train_pi05_frs.utils import model as model_module
from train_pi05_frs.utils import objective_schema


def _learned_config():
    return model_module.DecoderConfig(
        action_dim=10,
        action_horizon=3,
        tactile_window=2,
        gru_hidden_dim=4,
        resnet_embedding_dim=4,
        model_dim=8,
        depth=1,
        num_heads=2,
        mlp_ratio=2,
        num_tactile_tokens=2,
        state_conditioning=True,
        state_dim=2,
        output_mode="learned_residual_gate",
        residual_bound=0.25,
    )


def test_learned_residual_gate_objective_v3_contract_is_exact() -> None:
    factory = getattr(
        objective_schema, "learned_residual_gated_objective_metadata", None
    )
    validator = getattr(
        objective_schema, "validate_learned_residual_gated_objective_metadata", None
    )
    assert factory is not None
    assert validator is not None

    metadata = {
        "loss_mode": "learned_residual_gated",
        **factory(
            oracle_safe_mse_threshold=0.01,
            oracle_repair_mse_threshold=0.04,
            residual_bound=0.5,
        ),
    }
    validator(metadata)
    assert metadata["loss_objective_version"] == 3
    assert metadata["gate_granularity"] == "chunk"
    assert metadata["steered_action_dim"] == 9
    assert metadata["gripper_index"] == 9

    mismatched = copy.deepcopy(metadata)
    mismatched["residual_bound"] = 0.4
    with pytest.raises(ValueError, match="residual_bound"):
        validator(mismatched, residual_bound=0.5)

    missing = copy.deepcopy(metadata)
    missing.pop("oracle_safe_mse_threshold")
    with pytest.raises(ValueError, match="oracle_safe_mse_threshold"):
        validator(missing)

    with pytest.raises(ValueError, match="residual_bound"):
        factory(
            oracle_safe_mse_threshold=0.01,
            oracle_repair_mse_threshold=0.04,
            residual_bound=float("inf"),
        )


def test_oracle_gate_labels_use_arm9_chunk_mse_and_ignore_middle_band() -> None:
    labels_fn = getattr(model_module, "learned_residual_gate_labels", None)
    assert labels_fn is not None

    vla = jnp.zeros((3, 2, 10), dtype=jnp.float32)
    gt = vla.at[1, :, :9].set(0.2).at[2, :, :9].set(1.0)
    # A wildly different gripper must not affect the arm-only oracle label.
    gt = gt.at[..., 9].set(100.0)
    labels, mask, error = labels_fn(
        vla,
        gt,
        oracle_safe_mse_threshold=0.01,
        oracle_repair_mse_threshold=0.09,
    )

    np.testing.assert_allclose(error, [0.0, 0.04, 1.0], atol=1e-6)
    np.testing.assert_allclose(labels, [0.0, 0.0, 1.0])
    np.testing.assert_array_equal(mask, [True, False, True])


def test_learned_model_returns_bounded_arm9_residual_and_chunk_gate() -> None:
    model = model_module.TactileConditionedFlowDecoder(
        _learned_config(), rngs=nnx.Rngs(0)
    )
    vla = jnp.zeros((2, 3, 10), dtype=jnp.float32)
    tactile = jnp.ones((2, 2, 2, 4), dtype=jnp.float32)
    baseline = jnp.zeros((2, 2, 4), dtype=jnp.float32)
    state = jnp.ones((2, 2), dtype=jnp.float32)

    residual, gate_logits = model.predict_residual_gate(
        vla, tactile, baseline, state=state
    )

    assert residual.shape == (2, 3, 9)
    assert gate_logits.shape == (2, 1)
    assert bool(jnp.all(jnp.abs(residual) <= 0.25 + 1e-6))


def test_inference_applies_sigmoid_gate_and_preserves_vla_gripper() -> None:
    apply_fn = getattr(model_module, "apply_learned_residual_gate", None)
    assert apply_fn is not None

    vla = jnp.zeros((2, 2, 10), dtype=jnp.float32).at[..., 9].set(0.7)
    residual = jnp.ones((2, 2, 9), dtype=jnp.float32)
    logits = jnp.asarray([[0.0], [jnp.log(3.0)]], dtype=jnp.float32)
    executed = apply_fn(vla, residual, logits)

    np.testing.assert_allclose(executed[0, ..., :9], 0.5, atol=1e-6)
    np.testing.assert_allclose(executed[1, ..., :9], 0.75, atol=1e-6)
    np.testing.assert_allclose(executed[..., 9], 0.7, atol=1e-6)


def test_inference_helper_returns_executed_residual_and_logits() -> None:
    infer_fn = getattr(model_module, "infer_learned_residual_gate", None)
    assert infer_fn is not None
    model = model_module.TactileConditionedFlowDecoder(
        _learned_config(), rngs=nnx.Rngs(1)
    )
    vla = jnp.zeros((1, 3, 10), dtype=jnp.float32).at[..., 9].set(0.4)
    tactile = jnp.ones((1, 2, 2, 4), dtype=jnp.float32)
    baseline = jnp.zeros((1, 2, 4), dtype=jnp.float32)
    state = jnp.ones((1, 2), dtype=jnp.float32)

    executed, residual, logits = infer_fn(
        model, vla, tactile, baseline, state=state
    )

    assert executed.shape == (1, 3, 10)
    assert residual.shape == (1, 3, 9)
    assert logits.shape == (1, 1)
    np.testing.assert_allclose(executed[..., 9], 0.4, atol=1e-6)


def test_decode_wrapper_returns_gate_probability_vector() -> None:
    decode_fn = getattr(model_module, "decode_learned_residual_actions", None)
    assert decode_fn is not None
    model = model_module.TactileConditionedFlowDecoder(
        _learned_config(), rngs=nnx.Rngs(2)
    )
    vla = jnp.zeros((2, 3, 10), dtype=jnp.float32).at[..., 9].set(0.6)
    tactile = jnp.ones((2, 2, 2, 4), dtype=jnp.float32)
    baseline = jnp.zeros((2, 2, 4), dtype=jnp.float32)
    state = jnp.ones((2, 2), dtype=jnp.float32)

    executed, gate_probability, residual = decode_fn(
        model, vla, tactile, baseline, state=state
    )

    assert executed.shape == (2, 3, 10)
    assert gate_probability.shape == (2,)
    assert residual.shape == (2, 3, 9)
    assert bool(jnp.all((gate_probability >= 0.0) & (gate_probability <= 1.0)))
    np.testing.assert_allclose(executed[..., 9], 0.6, atol=1e-6)


def test_v3_loss_supervises_all_residuals_but_masks_middle_gate_bce() -> None:
    loss_fn = getattr(
        model_module, "learned_residual_gate_loss_components_per_sample", None
    )
    assert loss_fn is not None

    vla = jnp.zeros((3, 2, 10), dtype=jnp.float32)
    gt = vla.at[1, :, :9].set(0.2).at[2, :, :9].set(1.0)
    residual = jnp.zeros((3, 2, 9), dtype=jnp.float32)
    logits = jnp.zeros((3, 1), dtype=jnp.float32)
    components = loss_fn(
        residual,
        logits,
        vla,
        gt,
        oracle_safe_mse_threshold=0.01,
        oracle_repair_mse_threshold=0.09,
    )

    assert set(components) == {
        "gate_classification",
        "residual",
        "execute",
        "preserve",
    }
    for value in components.values():
        assert value.shape == (3,)
    # Middle-band gate classification is ignored, while its residual target is not.
    assert float(components["gate_classification"][1]) == pytest.approx(0.0)
    assert float(components["residual"][1]) > 0.0
    assert float(components["execute"][1]) == pytest.approx(0.0)
    assert float(components["preserve"][1]) == pytest.approx(0.0)


def test_v3_gate_bce_balances_safe_and_repair_groups() -> None:
    vla = jnp.zeros((4, 1, 10), dtype=jnp.float32)
    # Three safe chunks and one repair chunk, all with the same zero logit/BCE.
    gt = vla.at[3, :, :9].set(1.0)
    components = model_module.learned_residual_gate_loss_components_per_sample(
        jnp.zeros((4, 1, 9), dtype=jnp.float32),
        jnp.zeros((4, 1), dtype=jnp.float32),
        vla,
        gt,
        oracle_safe_mse_threshold=0.01,
        oracle_repair_mse_threshold=0.09,
    )

    safe_total = float(jnp.sum(components["gate_classification"][:3]))
    repair_total = float(components["gate_classification"][3])
    assert safe_total == pytest.approx(repair_total, rel=1e-6)


def test_v3_execute_and_preserve_losses_are_group_normalized() -> None:
    vla = jnp.zeros((4, 1, 10), dtype=jnp.float32)
    gt = vla.at[3, :, :9].set(1.0)
    residual = jnp.ones((4, 1, 9), dtype=jnp.float32)
    logits = jnp.zeros((4, 1), dtype=jnp.float32)

    components = model_module.learned_residual_gate_loss_components_per_sample(
        residual,
        logits,
        vla,
        gt,
        oracle_safe_mse_threshold=0.01,
        oracle_repair_mse_threshold=0.09,
    )

    # Three safe samples and one repair sample must each contribute one group
    # mean after the training loop averages across the four-item batch.
    assert float(jnp.mean(components["preserve"])) == pytest.approx(0.25)
    assert float(jnp.mean(components["execute"])) == pytest.approx(0.25)


def test_v3_residual_supervision_uses_huber_for_outliers() -> None:
    vla = jnp.zeros((2, 1, 10), dtype=jnp.float32)
    gt = vla.at[0, :, :9].set(1.0).at[1, :, :9].set(10.0)
    components = model_module.learned_residual_gate_loss_components_per_sample(
        jnp.zeros((2, 1, 9), dtype=jnp.float32),
        jnp.zeros((2, 1), dtype=jnp.float32),
        vla,
        gt,
        oracle_safe_mse_threshold=0.01,
        oracle_repair_mse_threshold=0.09,
    )

    # Huber is linear outside delta=1, so a 10x error stays far below 100x loss.
    assert float(components["residual"][1]) < 20.0 * float(
        components["residual"][0]
    )


def test_train_step_dispatches_learned_residual_gated_objective() -> None:
    model = model_module.TactileConditionedFlowDecoder(
        _learned_config(), rngs=nnx.Rngs(3)
    )
    optimizer = model_module.make_optimizer(
        model, learning_rate=1e-3, weight_decay=0.0
    )
    batch = 2
    zeros = jnp.zeros((batch, 3, 10), dtype=jnp.float32)
    loss, components = model_module.train_step(
        model,
        optimizer,
        zeros,
        zeros.at[1, :, :9].set(0.5),
        zeros,
        jnp.ones((batch, 2, 2, 4), dtype=jnp.float32),
        jnp.zeros((batch,), dtype=jnp.float32),
        jax.random.key(4),
        state=jnp.ones((batch, 2), dtype=jnp.float32),
        loss_mode="learned_residual_gated",
        baseline_tokens=jnp.zeros((batch, 2, 4), dtype=jnp.float32),
        oracle_safe_mse_threshold=0.01,
        oracle_repair_mse_threshold=0.09,
        gate_classification_weight=1.0,
        residual_loss_weight=2.0,
        execute_loss_weight=3.0,
        preserve_loss_weight=4.0,
        residual_bound=0.25,
    )

    assert bool(jnp.isfinite(loss))
    assert set(components) == {
        "gate_classification",
        "residual",
        "execute",
        "preserve",
    }


def test_default_flow_model_keeps_legacy_parameter_structure() -> None:
    config = model_module.DecoderConfig(
        action_dim=3,
        action_horizon=2,
        tactile_window=2,
        gru_hidden_dim=4,
        resnet_embedding_dim=4,
        model_dim=8,
        depth=1,
        num_heads=2,
        mlp_ratio=2,
        num_tactile_tokens=2,
    )
    model = model_module.TactileConditionedFlowDecoder(config, rngs=nnx.Rngs(0))
    assert config.output_mode == "flow"
    assert not hasattr(model, "gate_out")
    assert not hasattr(model, "baseline_proj")
