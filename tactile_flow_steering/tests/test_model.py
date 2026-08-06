from __future__ import annotations

import pathlib
import tempfile
import unittest

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from tactile_flow_steering.utils.checkpoint import load_checkpoint
from tactile_flow_steering.utils.checkpoint import save_checkpoint
from tactile_flow_steering.utils.metrics import evaluate_split
from tactile_flow_steering.utils.metrics import gate_stratified_decode_metrics
from tactile_flow_steering.utils.model import DecoderConfig
from tactile_flow_steering.utils.model import TactileConditionedFlowDecoder
from tactile_flow_steering.utils.model import decode_actions
from tactile_flow_steering.utils.model import decode_euler
from tactile_flow_steering.utils.model import decode_mse_per_sample
from tactile_flow_steering.utils.model import flow_matching_loss_per_sample
from tactile_flow_steering.utils.model import gate_preference_ranking_loss_per_sample
from tactile_flow_steering.utils.model import gated_flow_matching_loss_per_sample
from tactile_flow_steering.utils.model import gt_supervised_loss_per_sample
from tactile_flow_steering.utils.model import high_gate_repair_loss_per_sample
from tactile_flow_steering.utils.model import make_optimizer
from tactile_flow_steering.utils.model import train_step


class ConditionedDecoderModelTest(unittest.TestCase):
    def make_model(
        self, *, tactile_window: int = 3, gate_conditioning: bool = False
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
                gate_conditioning=gate_conditioning,
            ),
            rngs=nnx.Rngs(0),
        )

    def _tactile_seq(self, key, batch: int, window: int = 3):
        return jax.random.normal(key, (batch, window, 4, 4))

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
        decoded = decode_euler(model, x_base, tactile, num_steps=4)
        decoded_fireflow = decode_actions(
            model, x_base, tactile, num_steps=4, solver="fireflow"
        )
        self.assertEqual(loss.shape, (4,))
        self.assertEqual(decoded.shape, gt.shape)
        self.assertEqual(decoded_fireflow.shape, gt.shape)
        self.assertTrue(bool(jnp.all(jnp.isfinite(loss))))
        self.assertTrue(bool(jnp.all(jnp.isfinite(decoded_fireflow))))
        optimizer = make_optimizer(model, learning_rate=1e-3, weight_decay=0.0)
        gate = jnp.ones((4,), dtype=jnp.float32)
        step_loss = train_step(
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
        pred_step_loss = train_step(
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
        )
        self.assertEqual(out["n_high_w"], 2)
        self.assertEqual(out["n_low_w"], 2)
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

    def test_gate_preference_ranking_loss_selects_endpoint_by_gate(self):
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
        np.testing.assert_allclose(penalty, np.asarray([0.61, 0.61]), atol=1e-6)

        correctly_ordered = jnp.asarray([[[0.2]], [[0.8]]], dtype=jnp.float32)
        zero_penalty = gate_preference_ranking_loss_per_sample(
            correctly_ordered,
            gt,
            predicted,
            gate,
            margin=0.01,
        )
        np.testing.assert_allclose(zero_penalty, np.zeros((2,)), atol=1e-6)

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
        np.testing.assert_allclose(penalty, np.asarray([0.22, 0.0]), atol=1e-6)

        improved = high_gate_repair_loss_per_sample(
            jnp.asarray([[[0.8]], [[2.0]]], dtype=jnp.float32),
            gt,
            predicted,
            gate,
            margin=0.01,
        )
        np.testing.assert_allclose(improved, np.zeros((2,)), atol=1e-6)

    def test_explicit_gate_condition_changes_output(self):
        model = self.make_model(gate_conditioning=True)
        x_t = jax.random.normal(jax.random.key(30), (2, 6, 3))
        t = jnp.asarray([0.4, 0.4], dtype=jnp.float32)
        tactile = self._tactile_seq(jax.random.key(31), 2)
        with self.assertRaisesRegex(ValueError, "gate_weights are required"):
            model(x_t, t, tactile)
        low = model(x_t, t, tactile, jnp.zeros((2,), dtype=jnp.float32))
        high = model(x_t, t, tactile, jnp.ones((2,), dtype=jnp.float32))
        self.assertGreater(float(jnp.max(jnp.abs(low - high))), 1e-4)

        decoded = decode_euler(
            model,
            x_t,
            tactile,
            jnp.asarray([0.2, 0.8], dtype=jnp.float32),
            num_steps=3,
        )
        self.assertEqual(decoded.shape, x_t.shape)

    def test_gate_conditioned_evaluation_reports_vla_baseline_and_gain(self):
        model = self.make_model(gate_conditioning=True)

        class FakeConditioner:
            episode_baselines = {0: np.zeros((4, 4), dtype=np.float32)}

            def batches(self, split, *, batch_size, shuffle, seed):
                del split, batch_size, shuffle, seed
                yield (
                    np.asarray([0, 1], dtype=np.int64),
                    np.zeros((2, 6, 3), dtype=np.float32),
                    np.zeros((2, 6, 3), dtype=np.float32),
                    np.ones((2, 6, 3), dtype=np.float32),
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
        decode_mse = decode_mse_per_sample(
            model, x_base, gt, tactile, num_steps=4, solver="euler"
        )
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
        expected = 0.5 * loss_star + 2.0 * 0.5 * loss_stop
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
        decoded = decode_actions(model, x_base, tactile, gate, num_steps=3)
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
        decoded_a = decode_euler(model, x_base, tactile_a, num_steps=3)
        decoded_b = decode_euler(model, x_base, tactile_b, num_steps=3)
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

    def test_gate_conditioned_checkpoint_round_trip(self):
        model = self.make_model(gate_conditioning=True)
        x = jnp.ones((2, 6, 3), dtype=jnp.float32)
        t = jnp.asarray([0.25, 0.75], dtype=jnp.float32)
        tactile = jnp.ones((2, 3, 4, 4), dtype=jnp.float32)
        gate = jnp.asarray([0.1, 0.9], dtype=jnp.float32)
        expected = model(x, t, tactile, gate)
        with tempfile.TemporaryDirectory() as directory:
            checkpoint_dir = pathlib.Path(directory)
            save_checkpoint(checkpoint_dir, model, epoch=1, metrics={"val_mse": 0.5})
            restored, metadata = load_checkpoint(checkpoint_dir)
            self.assertTrue(jnp.array_equal(expected, restored(x, t, tactile, gate)))
            self.assertTrue(metadata["decoder_config"]["gate_conditioning"])

    def test_optimizer_state_round_trip(self):
        from tactile_flow_steering.utils.checkpoint import load_optimizer_state
        from tactile_flow_steering.utils.checkpoint import restore_optimizer_state
        from tactile_flow_steering.utils.model import make_optimizer

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
            restored_opt = make_optimizer(
                restored_model, learning_rate=1e-3, weight_decay=0.0, total_steps=10
            )
            restore_optimizer_state(restored_opt, opt_state=opt_state, step=step)
            self.assertEqual(int(restored_opt.step[...]), 4)


if __name__ == "__main__":
    unittest.main()
