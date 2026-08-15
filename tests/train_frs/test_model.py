from __future__ import annotations

import dataclasses
import inspect
import json
import pathlib
import tempfile
import unittest

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx
from flax import traverse_util

from train_encoder.utils.resnet import init_resnet18_params
from train_frs.utils.checkpoint import load_checkpoint, save_checkpoint
from train_frs.utils.metrics import (
    evaluate_split,
    gate_binned_decode_metrics,
    gate_stratified_decode_metrics,
)
from train_frs.utils.integration import euler_integrate_velocity, fireflow_integrate_velocity
from train_frs.utils.model import (
    DecoderConfig,
    TactileConditionedFlowDecoder,
    decode_actions,
    decode_mse_per_sample,
    flow_matching_loss_per_sample,
    gate_preference_ranking_loss_per_sample,
    gated_flow_matching_loss_per_sample,
    gated_loss_components_per_sample,
    gt_supervised_loss_per_sample,
    high_gate_repair_loss_per_sample,
    high_gate_worst_source_cvar_loss,
    low_gate_safety_loss_per_sample,
    make_optimizer,
    source_balanced_mean,
    three_region_effective_gate_weights,
    train_step,
)


def test_high_gate_worst_source_cvar_downweights_easy_source_prior() -> None:
    penalty = jnp.asarray([1.0, 0.5, 0.1, 0.8, 0.4, 0.2], dtype=jnp.float32)
    strength = jnp.ones_like(penalty)
    sources = jnp.asarray([0, 0, 0, 1, 1, 1], dtype=jnp.int32)
    equal, active = high_gate_worst_source_cvar_loss(
        penalty,
        strength,
        sources,
        jnp.asarray([1.0, 1.0], dtype=jnp.float32),
        num_sources=2,
        hard_fraction=0.5,
        worst_beta=20.0,
    )
    easy_downweighted, active_weighted = high_gate_worst_source_cvar_loss(
        penalty,
        strength,
        sources,
        jnp.asarray([1.0, 0.25], dtype=jnp.float32),
        num_sources=2,
        hard_fraction=0.5,
        worst_beta=20.0,
    )
    assert bool(active)
    assert bool(active_weighted)
    assert float(easy_downweighted) < 0.75
    assert float(easy_downweighted) > float(equal)


@pytest.fixture
def decoder() -> TactileConditionedFlowDecoder:
    return TactileConditionedFlowDecoder(
        DecoderConfig(
            action_dim=3,
            action_horizon=6,
            tactile_window=3,
            gru_hidden_dim=8,
            resnet_embedding_dim=8,
            model_dim=16,
            depth=2,
            num_heads=4,
            num_tactile_tokens=2,
        ),
        rngs=nnx.Rngs(0),
    )


@pytest.fixture
def decode_inputs():
    batch_size = 2
    return (
        jax.random.normal(jax.random.key(100), (batch_size, 6, 3)),
        jax.random.normal(jax.random.key(101), (batch_size, 3, 2, 8)),
    )


def integrate_decode_reference(velocity, x_base, *, num_steps: int, solver: str):
    if solver == "euler":
        return euler_integrate_velocity(velocity, x_base, num_steps=num_steps)
    if solver == "fireflow":
        return fireflow_integrate_velocity(velocity, x_base, num_steps=num_steps)
    raise AssertionError(f"Unexpected solver: {solver}")


@pytest.mark.parametrize("solver", ["euler", "fireflow"])
def test_cached_condition_decode_matches_recomputed_condition(decoder, decode_inputs, solver):
    x_base, tactile = decode_inputs
    expected = integrate_decode_reference(
        lambda x_t, t: decoder(x_t, t, tactile),
        x_base,
        num_steps=4,
        solver=solver,
    )
    actual = decode_actions(decoder, x_base, tactile, num_steps=4, solver=solver)
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("solver", ["euler", "fireflow"])
def test_decode_encodes_tactile_condition_once(decoder, decode_inputs, solver, monkeypatch):
    x_base, tactile = decode_inputs
    original_encode = decoder.encode_tactile_condition
    call_count = 0

    def count_encode(tactile_seq):
        nonlocal call_count
        call_count += 1
        return original_encode(tactile_seq)

    monkeypatch.setattr(decoder, "encode_tactile_condition", count_encode)
    decoded = decode_actions(decoder, x_base, tactile, num_steps=4, solver=solver)

    assert decoded.shape == x_base.shape
    assert call_count == 1

def test_decode_validates_solver_before_encoding_tactile_condition(decoder, decode_inputs, monkeypatch):
    x_base, tactile = decode_inputs
    original_encode = decoder.encode_tactile_condition
    call_count = 0

    def count_encode(tactile_seq):
        nonlocal call_count
        call_count += 1
        return original_encode(tactile_seq)

    monkeypatch.setattr(decoder, "encode_tactile_condition", count_encode)
    with pytest.raises(ValueError, match="solver must be 'euler' or 'fireflow'"):
        decode_actions(decoder, x_base, tactile, num_steps=4, solver="invalid")

    assert call_count == 0


def test_decoder_has_no_gate_input(decoder):
    assert "gate_conditioning" not in dataclasses.asdict(decoder.config)
    assert not hasattr(decoder, "gate_mlp")
    assert "gate_weights" not in inspect.signature(decoder.__call__).parameters
    assert "gate_weights" not in inspect.signature(decode_actions).parameters


def test_legacy_gate_conditioned_checkpoint_is_rejected(tmp_path, decoder):
    save_checkpoint(tmp_path, decoder, epoch=1, metrics={"val_mse": 0.5})
    metadata_path = tmp_path / "checkpoint.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["decoder_config"]["gate_conditioning"] = True
    metadata_path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="(?i)gate-conditioned.*retrain"):
        load_checkpoint(tmp_path)


def test_gated_training_checkpoint_metadata_declares_gate_training_only(tmp_path, monkeypatch):
    import train_frs.train_frs as train_module
    import train_frs.utils.data as data_module
    import train_frs.utils.metrics as metrics_module
    import train_frs.utils.model as model_module
    import utils.cache as cache_module

    class FakePairs:
        manifest = {
            "action_horizon": 2,
            "action_dim": 1,
            "state_dim": 0,
            "records_sha256": "test-digest",
            "configuration": {"dataset_repo_id": "owner/data"},
        }

        def __init__(self, cache_dir):
            del cache_dir

        def indices(self, split):
            assert split == "train"
            return np.asarray([0], dtype=np.int64)

    class FakeConditioner:
        resnet_embedding_dim = 4

        def __init__(self, pairs, **kwargs):
            del pairs
            assert kwargs["build_episode_baselines"] is True

        def batches(self, split, *, batch_size, shuffle, seed):
            del batch_size, shuffle, seed
            assert split == "train"
            yield (
                np.asarray([0], dtype=np.int64),
                np.zeros((1, 2, 1), dtype=np.float32),
                np.zeros((1, 2, 1), dtype=np.float32),
                np.ones((1, 2, 1), dtype=np.float32),
                np.zeros((1, 0), dtype=np.float32),
                jnp.ones((1, 2, 1, 4), dtype=jnp.float32),
            )

        def tactile_change_for_cache_indices(self, indices, current_tokens):
            del indices, current_tokens
            return np.asarray([0.9], dtype=np.float32)

        def gate_current_tokens(self, indices, tactile_input):
            del indices
            return np.asarray(tactile_input[:, -1, :, :], dtype=np.float32)

        def close(self):
            return None

    validation = type(
        "Validation",
        (),
        {
            "target": "gt",
            "flow_loss": 0.0,
            "mse": 0.0,
            "rmse": 0.0,
            "mae": 0.0,
            "flow_loss_gt": 0.0,
            "mse_gt": 0.0,
            "rmse_gt": 0.0,
            "mae_gt": 0.0,
            "flow_loss_pred": 0.0,
            "mse_pred": 0.0,
            "rmse_pred": 0.0,
            "mae_pred": 0.0,
            "mse_vla_gt": 0.0,
            "gt_gain": 0.0,
            "relative_gt_error": 0.0,
            "n_high_w": None,
            "gate_bin_metrics": None,
        },
    )()

    monkeypatch.setattr(cache_module, "CachedPairs", FakePairs)
    monkeypatch.setattr(data_module, "TactileConditionedBatches", FakeConditioner)
    monkeypatch.setattr(
        model_module,
        "train_step",
        lambda *args, **kwargs: (
            jnp.asarray(0.0),
            {
                name: jnp.asarray(0.0)
                for name in ("gt_fm", "vla_fm", "low_safety", "decode", "rank", "repair")
            },
        ),
    )
    monkeypatch.setattr(metrics_module, "evaluate_split", lambda *args, **kwargs: validation)

    train_module.train_decoder(
        cache_dir=tmp_path / "cache",
        tactile_encoder_dir=tmp_path,
        output_dir=tmp_path / "output",
        dataset_repo_id="owner/data",
        dataset_root=None,
        tactile_window_divisor=1,
        history_stride=1,
        loss_mode="gated",
        gate_tau=0.5,
        gate_temperature=0.1,
        gate_lambda=1.0,
        aux_decode_weight=0.0,
        aux_decode_steps=1,
        aux_decode_solver="euler",
        low_gate_safety_weight=0.0,
        low_gate_safety_margin=0.03,
        rank_weight=0.0,
        rank_margin=0.0,
        repair_weight=0.0,
        repair_margin=0.0,
        rank_low_gate_threshold=0.3,
        rank_high_gate_threshold=0.7,
        state_conditioning=False,
        state_dropout_rate=0.0,
        model_dim=4,
        depth=1,
        num_heads=1,
        mlp_ratio=1,
        learning_rate=1.0e-3,
        weight_decay=0.0,
        grad_clip_norm=None,
        warmup_epochs=0,
        lr_reference_dim=None,
        min_learning_rate_ratio=0.1,
        cosine_decay=False,
        batch_size=1,
        epochs=1,
        validation_steps=1,
        eval_every=1,
        seed=0,
        write_plots=False,
        num_workers=0,
        prefetch_batches=1,
        load_threads=1,
        pipeline_prefetch=1,
        image_cache_size=0,
        encode_batch_size=1,
        tactile_num_tokens=1,
    )

    metadata = json.loads((tmp_path / "output" / "last" / "checkpoint.json").read_text())
    extra_metadata = metadata["extra_metadata"]
    assert extra_metadata["decoder_input_version"] == 2
    assert extra_metadata["loss_weighting_version"] == 7
    assert "gate_conditioning" not in extra_metadata


@pytest.mark.parametrize("sequence_length", [1, 3, 6])
def test_tactile_encoder_accepts_sequence_lengths_one_training_window_and_horizon(
    decoder,
    sequence_length,
):
    tactile = jnp.ones((2, sequence_length, 2, 8), dtype=jnp.float32)
    encoded = decoder.encode_tactile_tokens(tactile)
    assert encoded.shape == (2, 2, decoder.config.gru_hidden_dim)
    assert bool(jnp.isfinite(encoded).all())


def test_tactile_encoder_rejects_empty_sequence(decoder):
    tactile = jnp.ones((2, 0, 2, 8), dtype=jnp.float32)
    with pytest.raises(ValueError, match="at least one time step"):
        decoder.encode_tactile_tokens(tactile)


def test_raw_tactile_and_state_encoders_receive_gradients_and_checkpoint(
    tmp_path,
):
    config = DecoderConfig(
        action_dim=2,
        action_horizon=2,
        tactile_window=1,
        gru_hidden_dim=4,
        resnet_embedding_dim=4,
        model_dim=8,
        depth=1,
        num_heads=2,
        num_tactile_tokens=2,
        state_dim=3,
        state_conditioning=True,
        tactile_encoder_trainable=True,
        tactile_image_size=16,
        tactile_encode_microbatch_size=2,
    )
    resnet_variables = init_resnet18_params(
        jax.random.key(200),
        image_size=16,
        embedding_dim=4,
    )
    model = TactileConditionedFlowDecoder(
        config,
        rngs=nnx.Rngs(201),
        tactile_resnet_variables=resnet_variables,
    )
    images = jax.random.uniform(
        jax.random.key(202),
        (1, 1, 2, 16, 16, 3),
    )
    state = jax.random.normal(jax.random.key(203), (1, 3))
    x_t = jax.random.normal(jax.random.key(204), (1, 2, 2))
    t = jnp.asarray([0.5], dtype=jnp.float32)

    def loss_fn(candidate):
        return jnp.mean(jnp.square(candidate(x_t, t, images, state=state)))

    _, gradients = nnx.value_and_grad(loss_fn)(model)
    flat_gradients = traverse_util.flatten_dict(gradients.to_pure_dict())
    tactile_gradients = [
        value
        for path, value in flat_gradients.items()
        if "tactile_resnet_params" in "/".join(str(part) for part in path)
    ]
    state_gradients = [
        value
        for path, value in flat_gradients.items()
        if "state_fc" in "/".join(str(part) for part in path)
    ]
    assert tactile_gradients
    assert state_gradients
    assert any(bool(jnp.any(jnp.abs(value) > 0)) for value in tactile_gradients)
    assert any(bool(jnp.any(jnp.abs(value) > 0)) for value in state_gradients)

    expected = model.encode_tactile_images(images)
    save_checkpoint(tmp_path, model, epoch=1, metrics={"val_mse": 0.0})
    restored, metadata = load_checkpoint(tmp_path)
    actual = restored.encode_tactile_images(images)
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)
    assert metadata["decoder_config"]["tactile_encoder_trainable"] is True
    assert any("tactile_resnet_batch_stats" in path for path in metadata["parameter_paths"])


class ConditionedDecoderModelTest(unittest.TestCase):
    def make_model(
        self,
        *,
        tactile_window: int = 3,
        state_conditioning: bool = False,
    ) -> TactileConditionedFlowDecoder:
        return TactileConditionedFlowDecoder(
            DecoderConfig(
                action_dim=3,
                action_horizon=6,
                tactile_window=tactile_window,
                gru_hidden_dim=8,
                resnet_embedding_dim=4,
                model_dim=16,
                depth=2,
                num_heads=4,
                state_dim=5 if state_conditioning else 0,
                state_conditioning=state_conditioning,
            ),
            rngs=nnx.Rngs(0),
        )

    def _tactile_seq(self, key, batch: int, window: int = 3):
        return jax.random.normal(key, (batch, window, 4, 4))

    def test_state_token_conditions_decode_and_can_be_fully_masked(self):
        model = self.make_model(state_conditioning=True)
        x_base = jax.random.normal(jax.random.key(80), (2, 6, 3))
        tactile = self._tactile_seq(jax.random.key(81), 2)
        state_a = jnp.tile(jnp.arange(5, dtype=jnp.float32)[None, :], (2, 1))
        state_b = jnp.flip(state_a, axis=1)

        decoded_a = decode_actions(model, x_base, tactile, num_steps=2, state=state_a)
        decoded_b = decode_actions(model, x_base, tactile, num_steps=2, state=state_b)
        self.assertGreater(float(jnp.max(jnp.abs(decoded_a - decoded_b))), 1e-6)

        dropped_a = decode_actions(
            model,
            x_base,
            tactile,
            num_steps=2,
            state=state_a,
            state_keep_mask=jnp.zeros((2,), dtype=jnp.float32),
        )
        dropped_b = decode_actions(
            model,
            x_base,
            tactile,
            num_steps=2,
            state=state_b,
            state_keep_mask=jnp.zeros((2,), dtype=jnp.float32),
        )
        self.assertTrue(bool(jnp.allclose(dropped_a, dropped_b, atol=1e-6)))

    def test_source_balanced_mean_weights_sources_equally(self):
        values = jnp.asarray([1.0, 3.0, 10.0], dtype=jnp.float32)
        sources = jnp.asarray([0, 0, 1], dtype=jnp.int32)
        balanced = source_balanced_mean(values, sources, num_sources=2)
        self.assertAlmostEqual(float(balanced), 6.0)

    def test_single_source_cvar_does_not_require_balanced_loss(self):
        model = self.make_model()
        batch_size = 10
        x_base = jax.random.normal(jax.random.key(70), (batch_size, 6, 3))
        gt = x_base + 1.0
        predicted = x_base + 0.1
        tactile = self._tactile_seq(jax.random.key(71), batch_size)
        gate = jnp.asarray(
            [0.95, 0.9, 0.85, 0.8, 0.75, 0.2, 0.15, 0.1, 0.5, 0.6],
            dtype=jnp.float32,
        )
        source_indices = jnp.zeros((batch_size,), dtype=jnp.int32)
        optimizer = make_optimizer(model, learning_rate=2.5e-5, weight_decay=0.0)
        loss, components = train_step(
            model,
            optimizer,
            x_base,
            gt,
            predicted,
            tactile,
            gate,
            jax.random.key(72),
            source_indices,
            jnp.ones((1,), dtype=jnp.float32),
            loss_mode="gated",
            gate_lambda=1.5,
            aux_decode_weight=2.0,
            aux_decode_steps=2,
            rank_weight=5.0,
            rank_margin=0.01,
            repair_weight=2.0,
            repair_margin=0.01,
            rank_low_gate_threshold=0.3,
            rank_high_gate_threshold=0.7,
            source_balanced_loss=False,
            num_sources=1,
            high_gate_rank_aggregation="worst_source_cvar",
            high_gate_rank_hard_fraction=0.3,
            high_gate_rank_worst_beta=20.0,
        )
        self.assertTrue(bool(jnp.isfinite(loss)))
        self.assertGreater(float(components["rank"]), 0.0)

    def test_shape_finite_gradient_and_decode(self):
        model = self.make_model()
        x_base = jax.random.normal(jax.random.key(1), (4, 6, 3))
        gt = x_base + 0.25
        predicted = x_base + 0.1
        tactile = self._tactile_seq(jax.random.key(3), 4)
        t = jnp.linspace(0.1, 0.9, 4)
        loss = flow_matching_loss_per_sample(model, x_base, gt, t, tactile)
        tokens = model.encode_tactile_tokens(tactile)
        self.assertEqual(tokens.shape, (4, 4, 8))
        decoded = decode_actions(model, x_base, tactile, num_steps=4, solver="euler")
        decoded_fireflow = decode_actions(model, x_base, tactile, num_steps=4, solver="fireflow")
        self.assertEqual(loss.shape, (4,))
        self.assertEqual(decoded.shape, gt.shape)
        self.assertEqual(decoded_fireflow.shape, gt.shape)
        self.assertTrue(bool(jnp.all(jnp.isfinite(loss))))
        self.assertTrue(bool(jnp.all(jnp.isfinite(decoded_fireflow))))
        optimizer = make_optimizer(model, learning_rate=1e-3, weight_decay=0.0)
        gate = jnp.ones((4,), dtype=jnp.float32)
        step_loss, step_components = train_step(
            model,
            optimizer,
            x_base,
            gt,
            predicted,
            tactile,
            gate,
            jax.random.key(2),
            loss_mode="gt",
            gate_lambda=1.0,
            aux_decode_weight=1.0,
            aux_decode_steps=4,
        )
        self.assertTrue(bool(jnp.isfinite(step_loss)))
        self.assertTrue(bool(jnp.allclose(step_loss, sum(step_components.values()))))
        pred_step_loss, pred_components = train_step(
            model,
            optimizer,
            x_base,
            gt,
            predicted,
            tactile,
            gate,
            jax.random.key(3),
            loss_mode="predicted",
            gate_lambda=1.0,
            aux_decode_weight=1.0,
            aux_decode_steps=4,
        )
        self.assertTrue(bool(jnp.allclose(pred_step_loss, sum(pred_components.values()))))
        self.assertTrue(bool(jnp.isfinite(pred_step_loss)))

    def test_gate_stratified_decode_metrics(self):
        out = gate_stratified_decode_metrics(
            np.asarray([1.0, 2.0, 3.0, 4.0]),
            np.asarray([0.1, 0.2, 0.3, 0.4]),
            np.asarray([2.0, 4.0, 2.0, 8.0]),
            np.asarray([0.9, 0.8, 0.1, 0.2]),
            np.asarray([0.8, 0.9, 0.1, 0.2]),
            ranking_margin=0.01,
            repair_margin=0.01,
            low_safety_margin=0.35,
        )
        self.assertEqual(out["n_high_w"], 2)
        self.assertEqual(out["n_low_w"], 2)
        self.assertEqual(out["n_mid_w"], 0)
        self.assertAlmostEqual(float(out["mse_gt_high_w"]), 1.5)
        self.assertAlmostEqual(float(out["mse_gt_low_w"]), 3.5)
        self.assertAlmostEqual(float(out["mse_pred_high_w"]), 0.15)
        self.assertAlmostEqual(float(out["mse_pred_low_w"]), 0.35)
        self.assertAlmostEqual(float(out["mse_vla_gt_high_w"]), 3.0)
        self.assertAlmostEqual(float(out["mse_vla_gt_low_w"]), 5.0)
        self.assertAlmostEqual(float(out["gt_gain_high_w"]), 1.5)
        self.assertAlmostEqual(float(out["relative_gt_error_high_w"]), 0.5)
        self.assertAlmostEqual(float(out["relative_gt_error_low_w"]), 0.7)
        self.assertAlmostEqual(float(out["gate_w_high_mean"]), 0.85)
        self.assertAlmostEqual(float(out["gate_w_low_mean"]), 0.15)
        self.assertAlmostEqual(float(out["rank_penalty_high_w"]), 1.36)
        self.assertAlmostEqual(float(out["rank_penalty_low_w"]), 0.0)
        self.assertAlmostEqual(float(out["rank_satisfied_high_frac"]), 0.0)
        self.assertAlmostEqual(float(out["rank_satisfied_low_frac"]), 1.0)
        self.assertAlmostEqual(float(out["repair_penalty_high_w"]), 0.0)
        self.assertAlmostEqual(float(out["repair_satisfied_high_frac"]), 1.0)
        self.assertAlmostEqual(float(out["low_nearest_endpoint_mse"]), 0.35)
        self.assertAlmostEqual(float(out["low_safety_penalty"]), 0.025)
        self.assertAlmostEqual(float(out["low_safe_frac"]), 0.5)
        self.assertAlmostEqual(float(out["low_unsafe_frac"]), 0.5)

    def test_gate_preference_ranking_loss_only_constrains_high_gate(self):
        gt = jnp.asarray([[[0.0]], [[0.0]]], dtype=jnp.float32)
        predicted = jnp.asarray([[[1.0]], [[1.0]]], dtype=jnp.float32)
        wrongly_ordered = jnp.asarray([[[0.8]], [[0.2]]], dtype=jnp.float32)
        gate = jnp.asarray([0.9, 0.1], dtype=jnp.float32)
        penalty = gate_preference_ranking_loss_per_sample(
            wrongly_ordered,
            gt,
            predicted,
            gate,
            margin=0.01,
        )
        np.testing.assert_allclose(penalty, np.asarray([1.22, 0.0]), atol=1e-6)

        correctly_ordered = jnp.asarray([[[0.2]], [[0.8]]], dtype=jnp.float32)
        zero_penalty = gate_preference_ranking_loss_per_sample(
            correctly_ordered,
            gt,
            predicted,
            gate,
            margin=0.01,
        )
        np.testing.assert_allclose(zero_penalty, np.asarray([0.0, 0.0]), atol=1e-6)

        transition_penalty = gate_preference_ranking_loss_per_sample(
            wrongly_ordered[:1],
            gt[:1],
            predicted[:1],
            jnp.asarray([0.5], dtype=jnp.float32),
            margin=0.01,
        )
        np.testing.assert_allclose(transition_penalty, np.asarray([0.0]), atol=1e-6)

        padded_penalty = gate_preference_ranking_loss_per_sample(
            jnp.concatenate([wrongly_ordered, jnp.full((8, 1, 1), 0.5)], axis=0),
            jnp.concatenate([gt, jnp.zeros((8, 1, 1))], axis=0),
            jnp.concatenate([predicted, jnp.ones((8, 1, 1))], axis=0),
            jnp.asarray([0.9, 0.1] + [0.5] * 8, dtype=jnp.float32),
            margin=0.01,
        )
        self.assertAlmostEqual(float(jnp.mean(padded_penalty)), float(jnp.mean(penalty)), places=6)

    def test_low_gate_safety_accepts_either_endpoint_and_rejects_neither(self):
        gt = jnp.asarray([[[0.0]], [[0.0]], [[0.0]]], dtype=jnp.float32)
        predicted = jnp.asarray([[[1.0]], [[1.0]], [[1.0]]], dtype=jnp.float32)
        decoded = jnp.asarray([[[0.2]], [[2.0]], [[2.0]]], dtype=jnp.float32)
        gate = jnp.asarray([0.1, 0.2, 0.9], dtype=jnp.float32)
        penalty = low_gate_safety_loss_per_sample(
            decoded,
            gt,
            predicted,
            gate,
            tolerance=0.1,
        )
        self.assertAlmostEqual(float(penalty[0]), 0.0)
        self.assertGreater(float(penalty[1]), 0.0)
        self.assertAlmostEqual(float(penalty[2]), 0.0)
        self.assertAlmostEqual(float(jnp.mean(penalty)), 0.72 / 1.7, places=6)

    def test_three_region_effective_gate_saturates_confident_regions(self):
        effective = three_region_effective_gate_weights(
            jnp.asarray([0.0, 0.3, 0.5, 0.7, 1.0], dtype=jnp.float32),
            low_gate_threshold=0.3,
            high_gate_threshold=0.7,
        )
        np.testing.assert_allclose(effective, np.asarray([0.0, 0.0, 0.5, 1.0, 1.0]), atol=1e-6)

    def test_high_gate_repair_loss_requires_absolute_gt_gain(self):
        gt = jnp.asarray([[[0.0]], [[0.0]]], dtype=jnp.float32)
        predicted = jnp.asarray([[[1.0]], [[1.0]]], dtype=jnp.float32)
        decoded = jnp.asarray([[[1.1]], [[2.0]]], dtype=jnp.float32)
        gate = jnp.asarray([0.9, 0.1], dtype=jnp.float32)
        penalty = high_gate_repair_loss_per_sample(
            decoded,
            gt,
            predicted,
            gate,
            margin=0.01,
        )
        np.testing.assert_allclose(penalty, np.asarray([0.44, 0.0]), atol=1e-6)

        improved = high_gate_repair_loss_per_sample(
            jnp.asarray([[[0.8]], [[2.0]]], dtype=jnp.float32),
            gt,
            predicted,
            gate,
            margin=0.01,
        )
        np.testing.assert_allclose(improved, np.asarray([0.0, 0.0]), atol=1e-6)

    def test_gate_binned_decode_metrics_reports_all_six_bins(self):
        weights = np.asarray([0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0])
        mse_gt = np.asarray([1.0, 2.0, 3.0, 0.4, 0.2, 0.1, 0.0])
        mse_pred = np.asarray([0.0, 0.1, 0.2, 0.5, 0.6, 0.8, 1.0])
        baseline = np.ones_like(weights)
        bins = gate_binned_decode_metrics(
            mse_gt,
            mse_pred,
            baseline,
            weights,
            ranking_margin=0.01,
        )
        self.assertEqual(list(bins), ["00_01", "01_03", "03_05", "05_07", "07_09", "09_10"])
        self.assertEqual([int(values["n"]) for values in bins.values()], [1, 1, 1, 1, 1, 2])
        self.assertAlmostEqual(float(bins["09_10"]["mse_gt"]), 0.05)
        self.assertAlmostEqual(float(bins["09_10"]["rank_satisfied_frac"]), 1.0)

    def test_gated_loss_components_sum_to_total(self):
        model = self.make_model()
        x_base = jax.random.normal(jax.random.key(50), (3, 6, 3))
        gt = x_base + 1.0
        predicted = x_base + 0.1
        tactile = self._tactile_seq(jax.random.key(51), 3)
        t = jnp.asarray([0.2, 0.5, 0.8], dtype=jnp.float32)
        gate = jnp.asarray([0.1, 0.5, 0.9], dtype=jnp.float32)
        components = gated_loss_components_per_sample(
            model,
            x_base,
            gt,
            predicted,
            t,
            tactile,
            gate,
            gate_lambda=1.2,
            aux_decode_weight=0.7,
            aux_decode_steps=3,
            low_gate_safety_weight=0.2,
            low_gate_safety_margin=0.05,
            rank_weight=0.5,
            rank_margin=0.01,
            repair_weight=0.75,
            repair_margin=0.01,
        )
        total = gated_flow_matching_loss_per_sample(
            model,
            x_base,
            gt,
            predicted,
            t,
            tactile,
            gate,
            gate_lambda=1.2,
            aux_decode_weight=0.7,
            aux_decode_steps=3,
            low_gate_safety_weight=0.2,
            low_gate_safety_margin=0.05,
            rank_weight=0.5,
            rank_margin=0.01,
            repair_weight=0.75,
            repair_margin=0.01,
        )
        decoded = decode_actions(
            model,
            x_base,
            tactile,
            num_steps=3,
            solver="euler",
        )
        mse_gt = jnp.mean(jnp.square(decoded - gt), axis=(1, 2))
        expected_decode = jnp.zeros_like(mse_gt).at[2].set(0.7 * 3.0 * mse_gt[2])
        expected_safety = 0.2 * low_gate_safety_loss_per_sample(
            decoded,
            gt,
            predicted,
            gate,
            tolerance=0.05,
        )
        self.assertEqual(
            set(components),
            {"gt_fm", "vla_fm", "low_safety", "decode", "rank", "repair"},
        )
        self.assertTrue(bool(jnp.allclose(components["decode"], expected_decode, atol=1e-6)))
        self.assertTrue(bool(jnp.allclose(components["low_safety"], expected_safety, atol=1e-6)))
        self.assertTrue(bool(jnp.allclose(total, sum(components.values()), atol=1e-6)))

    def test_gate_supervision_evaluation_reports_vla_baseline_and_gain(self):
        model = self.make_model()

        class FakeConditioner:
            episode_baselines = {0: np.zeros((4, 4), dtype=np.float32)}

            def batches(self, split, *, batch_size, shuffle, seed):
                del split, batch_size, shuffle, seed
                yield (
                    np.asarray([0, 1], dtype=np.int64),
                    np.zeros((2, 6, 3), dtype=np.float32),
                    np.zeros((2, 6, 3), dtype=np.float32),
                    np.ones((2, 6, 3), dtype=np.float32),
                    np.ones((2, 5), dtype=np.float32),
                    jnp.ones((2, 3, 4, 4), dtype=jnp.float32),
                )

            def tactile_change_for_cache_indices(self, indices, current_tokens):
                del indices, current_tokens
                return np.asarray([0.1, 0.9], dtype=np.float32)

        result = evaluate_split(
            model,
            FakeConditioner(),  # type: ignore[arg-type]
            split="val",
            batch_size=2,
            num_steps=2,
            keep_predictions=True,
            target="gt",
            gate_tau=0.5,
            gate_temperature=0.1,
        )
        self.assertAlmostEqual(result.mse_vla_gt, 1.0, places=6)
        self.assertAlmostEqual(result.gt_gain, 1.0 - result.mse_gt, places=6)
        self.assertAlmostEqual(result.relative_gt_error, result.mse_gt, places=6)
        self.assertEqual(result.n_high_w, 1)
        self.assertEqual(result.n_low_w, 1)
        self.assertIsNotNone(result.sample_gate_w)
        self.assertIsNotNone(result.sample_tactile_change)
        np.testing.assert_allclose(result.sample_mse_vla_gt, np.ones((2,)), atol=1e-6)

    def test_gt_supervised_adds_aux_decode_mse(self):
        model = self.make_model()
        x_base = jax.random.normal(jax.random.key(20), (3, 6, 3))
        gt = x_base + 1.0
        tactile = self._tactile_seq(jax.random.key(21), 3)
        t = jnp.full((3,), 0.5, dtype=jnp.float32)
        flow = flow_matching_loss_per_sample(model, x_base, gt, t, tactile)
        decode_mse = decode_mse_per_sample(model, x_base, gt, tactile, num_steps=4, solver="euler")
        combined = gt_supervised_loss_per_sample(
            model,
            x_base,
            gt,
            t,
            tactile,
            aux_decode_weight=1.0,
            aux_decode_steps=4,
        )
        flow_only = gt_supervised_loss_per_sample(
            model,
            x_base,
            gt,
            t,
            tactile,
            aux_decode_weight=0.0,
            aux_decode_steps=4,
        )
        self.assertTrue(bool(jnp.allclose(flow_only, flow, atol=1e-5)))
        self.assertTrue(bool(jnp.allclose(combined, flow + decode_mse, atol=1e-5)))

    def test_gated_loss_respects_weights(self):
        model = self.make_model()
        x_base = jax.random.normal(jax.random.key(10), (3, 6, 3))
        gt = x_base + 1.0
        predicted = x_base + 0.1
        tactile = self._tactile_seq(jax.random.key(11), 3)
        t = jnp.full((3,), 0.5, dtype=jnp.float32)
        ones = jnp.ones((3,), dtype=jnp.float32)
        zeros = jnp.zeros((3,), dtype=jnp.float32)
        loss_star = gt_supervised_loss_per_sample(
            model,
            x_base,
            gt,
            t,
            tactile,
            aux_decode_weight=1.0,
            aux_decode_steps=4,
        )
        loss_stop = flow_matching_loss_per_sample(model, x_base, predicted, t, tactile)
        gated_w1 = gated_flow_matching_loss_per_sample(
            model,
            x_base,
            gt,
            predicted,
            t,
            tactile,
            ones,
            gate_lambda=1.0,
            aux_decode_weight=1.0,
            aux_decode_steps=4,
        )
        gated_w0 = gated_flow_matching_loss_per_sample(
            model,
            x_base,
            gt,
            predicted,
            t,
            tactile,
            zeros,
            gate_lambda=1.0,
            aux_decode_weight=1.0,
            aux_decode_steps=4,
        )
        self.assertTrue(bool(jnp.allclose(gated_w1, loss_star, atol=1e-5)))
        self.assertTrue(bool(jnp.allclose(gated_w0, loss_stop, atol=1e-5)))
        gated_half = gated_flow_matching_loss_per_sample(
            model,
            x_base,
            gt,
            predicted,
            t,
            tactile,
            jnp.full((3,), 0.5, dtype=jnp.float32),
            gate_lambda=2.0,
            aux_decode_weight=1.0,
            aux_decode_steps=4,
        )
        flow_star = flow_matching_loss_per_sample(model, x_base, gt, t, tactile)
        expected = 0.5 * flow_star + 2.0 * 0.5 * loss_stop
        self.assertTrue(bool(jnp.allclose(gated_half, expected, atol=1e-5)))

    def test_gated_loss_adds_weighted_preference_and_repair_constraints(self):
        model = self.make_model()
        x_base = jax.random.normal(jax.random.key(40), (2, 6, 3))
        gt = x_base + 1.0
        predicted = x_base + 0.1
        tactile = self._tactile_seq(jax.random.key(41), 2)
        t = jnp.full((2,), 0.5, dtype=jnp.float32)
        gate = jnp.asarray([0.9, 0.1], dtype=jnp.float32)
        base = gated_flow_matching_loss_per_sample(
            model,
            x_base,
            gt,
            predicted,
            t,
            tactile,
            gate,
            gate_lambda=1.0,
            aux_decode_weight=1.0,
            aux_decode_steps=3,
        )
        constrained = gated_flow_matching_loss_per_sample(
            model,
            x_base,
            gt,
            predicted,
            t,
            tactile,
            gate,
            gate_lambda=1.0,
            aux_decode_weight=1.0,
            aux_decode_steps=3,
            rank_weight=0.5,
            rank_margin=0.01,
            repair_weight=0.75,
            repair_margin=0.01,
        )
        decoded = decode_actions(model, x_base, tactile, num_steps=3)
        rank_penalty = gate_preference_ranking_loss_per_sample(
            decoded,
            gt,
            predicted,
            gate,
            margin=0.01,
        )
        repair_penalty = high_gate_repair_loss_per_sample(
            decoded,
            gt,
            predicted,
            gate,
            margin=0.01,
        )
        expected = base + 0.5 * rank_penalty + 0.75 * repair_penalty
        self.assertTrue(bool(jnp.allclose(constrained, expected, atol=1e-5)))

    def test_tactile_seq_changes_output(self):
        model = self.make_model()
        x_t = jax.random.normal(jax.random.key(4), (2, 6, 3))
        t = jnp.asarray([0.3, 0.7], dtype=jnp.float32)
        tactile_a = self._tactile_seq(jax.random.key(5), 2)
        tactile_b = tactile_a + 5.0
        velocity_a = model(x_t, t, tactile_a)
        velocity_b = model(x_t, t, tactile_b)
        self.assertGreater(float(jnp.max(jnp.abs(velocity_a - velocity_b))), 1e-4)

        x_base = jax.random.normal(jax.random.key(6), (2, 6, 3))
        decoded_a = decode_actions(model, x_base, tactile_a, num_steps=3, solver="euler")
        decoded_b = decode_actions(model, x_base, tactile_b, num_steps=3, solver="euler")
        self.assertGreater(float(jnp.max(jnp.abs(decoded_a - decoded_b))), 1e-4)

    def test_checkpoint_round_trip(self):
        model = self.make_model()
        x = jnp.ones((2, 6, 3), dtype=jnp.float32)
        t = jnp.asarray([0.25, 0.75], dtype=jnp.float32)
        tactile = jnp.ones((2, 3, 4, 4), dtype=jnp.float32)
        expected = model(x, t, tactile)
        with tempfile.TemporaryDirectory() as directory:
            checkpoint_dir = pathlib.Path(directory)
            save_checkpoint(checkpoint_dir, model, epoch=3, metrics={"val_mse": 0.5})
            restored, metadata = load_checkpoint(checkpoint_dir)
            self.assertTrue(jnp.array_equal(expected, restored(x, t, tactile)))
            self.assertEqual(metadata["epoch"], 3)
            self.assertEqual(metadata["decoder_config"]["gru_hidden_dim"], 8)
            self.assertEqual(metadata["decoder_config"]["tactile_window"], 3)
            self.assertEqual(metadata["decoder_config"]["num_tactile_tokens"], 4)
            self.assertTrue((checkpoint_dir / metadata["params_file"]).is_file())

    def test_optimizer_state_round_trip(self):
        from train_frs.utils.checkpoint import (
            load_optimizer_state,
            restore_optimizer_state,
        )
        from train_frs.utils.model import make_optimizer

        model = self.make_model()
        optimizer = make_optimizer(model, learning_rate=1e-3, weight_decay=0.0, total_steps=10)
        optimizer.step[...] = jnp.asarray(4, dtype=jnp.uint32)
        with tempfile.TemporaryDirectory() as directory:
            checkpoint_dir = pathlib.Path(directory)
            save_checkpoint(
                checkpoint_dir,
                model,
                epoch=2,
                metrics={"train_flow_loss": 1.0},
                optimizer=optimizer,
            )
            restored_model, metadata = load_checkpoint(checkpoint_dir)
            self.assertTrue(metadata["has_opt_state"])
            self.assertTrue((checkpoint_dir / metadata["opt_state_file"]).is_file())
            opt_state, step = load_optimizer_state(checkpoint_dir)
            self.assertIsNotNone(opt_state)
            self.assertEqual(step, 4)
            restored_opt = make_optimizer(restored_model, learning_rate=1e-3, weight_decay=0.0, total_steps=10)
            restore_optimizer_state(restored_opt, opt_state=opt_state, step=step)
            self.assertEqual(int(restored_opt.step[...]), 4)


if __name__ == "__main__":
    unittest.main()
