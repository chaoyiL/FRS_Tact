from __future__ import annotations

import hashlib
import inspect
import json
import pathlib
import shutil
import tempfile
import unittest
from unittest import mock

import jax
import jax.numpy as jnp
from flax import nnx

from train_pi05_frs.utils.checkpoint import load_checkpoint
from train_pi05_frs.utils.checkpoint import resolve_checkpoint_snapshot
from train_pi05_frs.utils.checkpoint import save_checkpoint
from train_pi05_frs.utils.model import DecoderConfig
from train_pi05_frs.utils.model import TactileConditionedFlowDecoder
from train_pi05_frs.utils.model import decode_actions
from train_pi05_frs.utils.model import decode_euler
from train_pi05_frs.utils.metrics import gate_stratified_decode_metrics
from train_pi05_frs.utils.model import decode_mse_per_sample
from train_pi05_frs.utils.model import flow_matching_loss_per_sample
from train_pi05_frs.utils.model import gated_flow_matching_loss_per_sample
from train_pi05_frs.utils.model import gt_supervised_loss_per_sample
from train_pi05_frs.utils.model import make_optimizer
from train_pi05_frs.utils.model import train_step
from train_pi05_frs.train import (
    _validate_resume_cache_provenance,
    _validate_training_path_boundaries,
    train_decoder,
)
import numpy as np


class ConditionedDecoderModelTest(unittest.TestCase):
    def test_direct_decoder_rejects_output_overlapping_read_only_inputs(self):
        with tempfile.TemporaryDirectory() as temporary:
            shared = pathlib.Path(temporary) / "shared"
            shared.mkdir()
            for label, arguments in (
                ("encoder", {"tactile_encoder_dir": shared}),
                ("action cache", {"cache_dir": shared}),
                ("dataset", {"dataset_root": shared}),
                ("resume", {"resume_from": shared}),
            ):
                values = {
                    "cache_dir": None,
                    "cache_dirs": None,
                    "tactile_encoder_dir": pathlib.Path(temporary) / "encoder",
                    "output_dir": shared,
                    "dataset_root": None,
                    "dataset_sources": None,
                    "tactile_embedding_cache_root": None,
                    "resume": False,
                    "resume_from": None,
                }
                values.update(arguments)
                with self.subTest(label=label), self.assertRaisesRegex(
                    ValueError, "read-only.*overlap"
                ):
                    _validate_training_path_boundaries(**values)

    def test_direct_decoder_rejects_nonempty_fresh_output(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            output = root / "output"
            output.mkdir()
            (output / "stale").write_text("stale", encoding="utf-8")
            with self.assertRaisesRegex(FileExistsError, "output directory is not empty"):
                _validate_training_path_boundaries(
                    cache_dir=root / "cache",
                    cache_dirs=None,
                    tactile_encoder_dir=root / "encoder",
                    output_dir=output,
                    dataset_root=root / "dataset",
                    dataset_sources=None,
                    tactile_embedding_cache_root=None,
                    resume=False,
                    resume_from=None,
                )

    def test_direct_decoder_preserves_transactional_implicit_resume(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            output = root / "output"
            (output / "last").mkdir(parents=True)
            _validate_training_path_boundaries(
                cache_dir=root / "cache",
                cache_dirs=None,
                tactile_encoder_dir=root / "encoder",
                output_dir=output,
                dataset_root=root / "dataset",
                dataset_sources=None,
                tactile_embedding_cache_root=None,
                resume=True,
                resume_from=None,
            )

    def test_implicit_resume_pins_generation_before_saving_next_generation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            output = root / "output"
            last = output / "last"
            model = self.make_model()
            save_checkpoint(last, model, epoch=1, metrics={"val_mse": 1.0})

            _validate_training_path_boundaries(
                cache_dir=root / "cache",
                cache_dirs=None,
                tactile_encoder_dir=root / "encoder",
                output_dir=output,
                dataset_root=root / "dataset",
                dataset_sources=None,
                tactile_embedding_cache_root=None,
                resume=True,
                resume_from=None,
            )
            pinned = resolve_checkpoint_snapshot(last)
            _, pinned_metadata = load_checkpoint(pinned)
            save_checkpoint(last, model, epoch=2, metrics={"val_mse": 0.5})
            _, next_metadata = load_checkpoint(last)

            self.assertNotEqual(pinned, last.resolve(strict=True))
            self.assertEqual(pinned_metadata["epoch"], 1)
            self.assertEqual(next_metadata["epoch"], 2)

    def make_model(self, *, tactile_window: int = 3) -> TactileConditionedFlowDecoder:
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
            ),
            rngs=nnx.Rngs(0),
        )

    def test_resume_rejects_same_shape_checkpoint_from_a_different_cache(self):
        for changed_field in ("cache_records_sha256", "cache_configuration"):
            checkpoint_metadata = {
                "extra_metadata": {
                    "cache_records_sha256": "same-records",
                    "cache_configuration": {"dataset_repo_id": "org/cache-a"},
                }
            }
            current_cache_manifest = {
                "action_horizon": 6,
                "action_dim": 3,
                "state_dim": 5,
                "records_sha256": "same-records",
                "configuration": {"dataset_repo_id": "org/cache-a"},
            }
            if changed_field == "cache_records_sha256":
                current_cache_manifest["records_sha256"] = "other-records"
            else:
                current_cache_manifest["configuration"] = {
                    "dataset_repo_id": "org/cache-b"
                }

            with self.subTest(changed_field=changed_field), self.assertRaisesRegex(
                ValueError, f"{changed_field}.*fine-tun"
            ):
                _validate_resume_cache_provenance(
                    checkpoint_metadata, current_cache_manifest
                )

    def test_resume_rejects_checkpoint_without_cache_provenance(self):
        with self.assertRaisesRegex(ValueError, "cache_records_sha256.*cache_configuration"):
            _validate_resume_cache_provenance(
                {"extra_metadata": {}},
                {"records_sha256": "current", "configuration": {}},
            )

    def test_resume_checks_cache_before_optimizer_restore_or_history_write(self):
        source = inspect.getsource(train_decoder)
        validate_at = source.index("_validate_resume_cache_provenance(")
        restore_at = source.index("load_optimizer_state(resume_dir)")
        history_at = source.index("output_dir.mkdir(parents=True, exist_ok=True)")
        pin_at = source.index("resolve_checkpoint_snapshot(resume_dir)")

        self.assertLess(pin_at, validate_at)
        self.assertLess(validate_at, restore_at)
        self.assertLess(validate_at, history_at)

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
        step_loss, components = train_step(
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
        self.assertEqual(
            set(components), {"gt_fm", "vla_fm", "low_safety", "decode", "rank", "repair"}
        )
        pred_step_loss, _ = train_step(
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
            np.asarray([0.9, 0.8, 0.1, 0.2]),
        )
        self.assertEqual(out["n_high_w"], 2)
        self.assertEqual(out["n_low_w"], 2)
        self.assertAlmostEqual(float(out["mse_gt_high_w"]), 1.5)
        self.assertAlmostEqual(float(out["mse_gt_low_w"]), 3.5)
        self.assertAlmostEqual(float(out["mse_pred_high_w"]), 0.15)
        self.assertAlmostEqual(float(out["mse_pred_low_w"]), 0.35)

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
        # Mid-Gate mixes FM targets; decode supervision is reserved for high Gate.
        flow_star = flow_matching_loss_per_sample(model, x_base, gt, t, tactile)
        expected = 0.5 * flow_star + 2.0 * 0.5 * loss_stop
        self.assertTrue(bool(jnp.allclose(gated_half, expected, atol=1e-5)))

    def test_state_conditioning_adds_one_condition_token(self):
        model = TactileConditionedFlowDecoder(
            DecoderConfig(
                action_dim=3,
                action_horizon=6,
                tactile_window=3,
                gru_hidden_dim=8,
                resnet_embedding_dim=4,
                model_dim=16,
                depth=1,
                num_heads=4,
                state_conditioning=True,
                state_dim=5,
            ),
            rngs=nnx.Rngs(0),
        )
        tactile = self._tactile_seq(jax.random.key(30), 2)
        state = jnp.ones((2, 5), dtype=jnp.float32)
        condition = model.encode_condition(tactile, state)
        self.assertEqual(condition.shape, (2, 5, 16))

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
            checkpoint_dir = pathlib.Path(directory) / "output" / "last"
            save_checkpoint(checkpoint_dir, model, epoch=3, metrics={"val_mse": 0.5})
            restored, metadata = load_checkpoint(checkpoint_dir)
            self.assertTrue(jnp.array_equal(expected, restored(x, t, tactile)))
            self.assertEqual(metadata["epoch"], 3)
            self.assertEqual(metadata["decoder_config"]["gru_hidden_dim"], 8)
            self.assertEqual(metadata["decoder_config"]["tactile_window"], 3)
            self.assertEqual(metadata["decoder_config"]["num_tactile_tokens"], 4)

    def test_optimizer_state_round_trip(self):
        from train_pi05_frs.utils.checkpoint import load_optimizer_state
        from train_pi05_frs.utils.checkpoint import restore_optimizer_state
        from train_pi05_frs.utils.model import make_optimizer

        model = self.make_model()
        optimizer = make_optimizer(model, learning_rate=1e-3, weight_decay=0.0, total_steps=10)
        batch_size = 2
        x_base = jnp.zeros((batch_size, 6, 3), dtype=jnp.float32)
        tactile = jnp.ones((batch_size, 3, 4, 4), dtype=jnp.float32)
        train_step(
            model,
            optimizer,
            x_base,
            jnp.ones_like(x_base),
            jnp.zeros_like(x_base),
            tactile,
            jnp.ones((batch_size,), dtype=jnp.float32),
            jax.random.key(99),
            loss_mode="gt",
            gate_lambda=1.0,
            aux_decode_weight=0.0,
            aux_decode_steps=1,
        )
        optimizer.step.value = jnp.asarray(4, dtype=jnp.uint32)
        with tempfile.TemporaryDirectory() as directory:
            checkpoint_dir = pathlib.Path(directory) / "output" / "last"
            save_checkpoint(
                checkpoint_dir,
                model,
                epoch=2,
                metrics={"train_flow_loss": 1.0},
                optimizer=optimizer,
            )
            restored_model, metadata = load_checkpoint(checkpoint_dir)
            self.assertTrue(metadata["has_opt_state"])
            opt_state, step = load_optimizer_state(checkpoint_dir)
            self.assertIsNotNone(opt_state)
            self.assertEqual(step, 4)
            restored_opt = make_optimizer(
                restored_model, learning_rate=1e-3, weight_decay=0.0, total_steps=10
            )
            restore_optimizer_state(restored_opt, opt_state=opt_state, step=step)
            self.assertEqual(int(restored_opt.step[...]), 4)
            expected_leaves = jax.tree_util.tree_leaves(
                nnx.state(optimizer, type(optimizer.step))["opt_state"].to_pure_dict()
            )
            restored_leaves = jax.tree_util.tree_leaves(
                nnx.state(restored_opt, type(restored_opt.step))["opt_state"].to_pure_dict()
            )
            self.assertEqual(len(expected_leaves), len(restored_leaves))
            for expected_leaf, restored_leaf in zip(expected_leaves, restored_leaves, strict=True):
                self.assertTrue(jnp.array_equal(expected_leaf, restored_leaf))

    def test_checkpoint_publishes_immutable_checksummed_generation(self):
        from train_pi05_frs.utils.checkpoint import load_optimizer_state

        model = self.make_model()
        optimizer = make_optimizer(
            model, learning_rate=1e-3, weight_decay=0.0, total_steps=10
        )
        optimizer.step.value = jnp.asarray(7, dtype=jnp.uint32)
        with tempfile.TemporaryDirectory() as directory:
            checkpoint_dir = pathlib.Path(directory) / "output" / "last"
            save_checkpoint(
                checkpoint_dir,
                model,
                epoch=4,
                metrics={"val_mse": 0.25},
                optimizer=optimizer,
            )

            self.assertTrue(checkpoint_dir.is_symlink())
            snapshot = checkpoint_dir.resolve(strict=True)
            self.assertEqual(snapshot.parent.name, ".checkpoint-generations")
            metadata = json.loads(
                (snapshot / "checkpoint.json").read_text(encoding="utf-8")
            )
            self.assertEqual(metadata["version"], 3)
            self.assertEqual(metadata["generation"], snapshot.name)
            self.assertEqual(
                set(metadata["files"]),
                {
                    "checkpoint.json",
                    "params.npz",
                    "opt_state.npz",
                    "opt_state.treedef.pkl",
                },
            )
            for name, record in metadata["files"].items():
                payload = (snapshot / name).read_bytes()
                if name == "checkpoint.json":
                    # The metadata record covers the canonical metadata payload with
                    # its own checksum field blanked to avoid recursive hashing.
                    canonical = json.loads(payload)
                    canonical["files"][name]["sha256"] = ""
                    payload = (json.dumps(canonical, sort_keys=True) + "\n").encode()
                self.assertEqual(record["size"], len(payload))
                self.assertEqual(record["sha256"], hashlib.sha256(payload).hexdigest())
                self.assertEqual(record["generation"], snapshot.name)

            _, loaded_metadata = load_checkpoint(checkpoint_dir)
            _, step = load_optimizer_state(checkpoint_dir)
            self.assertEqual(loaded_metadata["generation"], snapshot.name)
            self.assertEqual(step, 7)

    def test_checkpoint_faults_never_expose_a_mixed_generation(self):
        from train_pi05_frs.utils import checkpoint as checkpoint_module

        stages = (
            "after_params_fsync",
            "after_opt_state_fsync",
            "after_opt_treedef_fsync",
            "after_metadata_fsync",
            "after_snapshot_validation",
            "after_snapshot_dir_fsync",
            "after_generation_publish",
            "after_pointer_prepare",
            "after_pointer_publish",
            "after_pointer_parent_fsync",
        )
        for stage in stages:
            with self.subTest(stage=stage), tempfile.TemporaryDirectory() as directory:
                checkpoint_dir = pathlib.Path(directory) / "output" / "last"
                model_a = self.make_model()
                optimizer_a = make_optimizer(
                    model_a, learning_rate=1e-3, weight_decay=0.0, total_steps=10
                )
                optimizer_a.step.value = jnp.asarray(1, dtype=jnp.uint32)
                save_checkpoint(
                    checkpoint_dir,
                    model_a,
                    epoch=1,
                    metrics={"val_mse": 1.0},
                    optimizer=optimizer_a,
                )
                generation_a = checkpoint_dir.resolve(strict=True).name
                model_b = TactileConditionedFlowDecoder(
                    model_a.config, rngs=nnx.Rngs(99)
                )
                optimizer_b = make_optimizer(
                    model_b, learning_rate=1e-3, weight_decay=0.0, total_steps=10
                )
                optimizer_b.step.value = jnp.asarray(2, dtype=jnp.uint32)

                def fail_at(actual: str) -> None:
                    if actual == stage:
                        raise OSError(f"injected failure at {stage}")

                with mock.patch.object(
                    checkpoint_module, "_checkpoint_fault", side_effect=fail_at
                ), self.assertRaisesRegex(OSError, stage):
                    save_checkpoint(
                        checkpoint_dir,
                        model_b,
                        epoch=2,
                        metrics={"val_mse": 0.5},
                        optimizer=optimizer_b,
                    )

                _, metadata = load_checkpoint(checkpoint_dir)
                opt_state, step = checkpoint_module.load_optimizer_state(checkpoint_dir)
                pointer_was_published = stage in {
                    "after_pointer_publish",
                    "after_pointer_parent_fsync",
                }
                self.assertEqual(metadata["epoch"], 2 if pointer_was_published else 1)
                self.assertEqual(step, 2 if pointer_was_published else 1)
                self.assertIsNotNone(opt_state)
                self.assertEqual(
                    checkpoint_dir.resolve(strict=True).name == generation_a,
                    not pointer_was_published,
                )

    def test_v3_checkpoint_checksum_corruption_fails_closed(self):
        from train_pi05_frs.utils.checkpoint import load_optimizer_state

        model = self.make_model()
        optimizer = make_optimizer(
            model, learning_rate=1e-3, weight_decay=0.0, total_steps=10
        )
        with tempfile.TemporaryDirectory() as directory:
            checkpoint_dir = pathlib.Path(directory) / "output" / "last"
            save_checkpoint(
                checkpoint_dir,
                model,
                epoch=1,
                metrics={},
                optimizer=optimizer,
            )
            with (checkpoint_dir / "opt_state.npz").open("ab") as file:
                file.write(b"corrupt")
            with self.assertRaisesRegex(ValueError, "checksum|size"):
                load_optimizer_state(checkpoint_dir)

    def test_loader_keeps_legacy_v2_directory_compatibility(self):
        model = self.make_model()
        with tempfile.TemporaryDirectory() as directory:
            checkpoint_dir = pathlib.Path(directory) / "output" / "last"
            save_checkpoint(checkpoint_dir, model, epoch=3, metrics={})
            snapshot = checkpoint_dir.resolve(strict=True)
            legacy = pathlib.Path(directory) / "legacy"
            shutil.copytree(snapshot, legacy)
            metadata_path = legacy / "checkpoint.json"
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            metadata["version"] = 2
            metadata.pop("generation")
            metadata.pop("files")
            metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

            _, restored_metadata = load_checkpoint(legacy)
            self.assertEqual(restored_metadata["version"], 2)

    def test_v3_snapshot_can_be_dereferenced_into_a_named_handoff_directory(self):
        model = self.make_model()
        with tempfile.TemporaryDirectory() as directory:
            checkpoint_dir = pathlib.Path(directory) / "output" / "best"
            save_checkpoint(checkpoint_dir, model, epoch=3, metrics={})
            handoff = pathlib.Path(directory) / "chosen-checkpoint"
            shutil.copytree(checkpoint_dir.resolve(strict=True), handoff)

            restored, metadata = load_checkpoint(handoff)

            self.assertEqual(metadata["epoch"], 3)
            self.assertEqual(restored.config, model.config)

    def test_v3_snapshot_inside_canonical_generation_root_must_match_its_name(self):
        model = self.make_model()
        with tempfile.TemporaryDirectory() as directory:
            checkpoint_dir = pathlib.Path(directory) / "output" / "best"
            save_checkpoint(checkpoint_dir, model, epoch=3, metrics={})
            mismatched = checkpoint_dir.resolve(strict=True).parent / "wrong-generation"
            shutil.copytree(checkpoint_dir.resolve(strict=True), mismatched)

            with self.assertRaisesRegex(ValueError, "generation mismatch"):
                load_checkpoint(mismatched)

    def test_save_atomically_upgrades_an_existing_legacy_directory(self):
        model = self.make_model()
        with tempfile.TemporaryDirectory() as directory:
            output = pathlib.Path(directory) / "output"
            seed = output / "seed"
            save_checkpoint(seed, model, epoch=1, metrics={})
            legacy = output / "last"
            shutil.copytree(seed.resolve(strict=True), legacy)
            metadata_path = legacy / "checkpoint.json"
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            metadata["version"] = 2
            metadata.pop("generation")
            metadata.pop("files")
            metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

            save_checkpoint(legacy, model, epoch=2, metrics={})

            self.assertTrue(legacy.is_symlink())
            _, restored = load_checkpoint(legacy)
            self.assertEqual(restored["epoch"], 2)
            retired = list((output / ".checkpoint-generations").glob("legacy-*"))
            self.assertEqual(len(retired), 1)
            retired_metadata = json.loads(
                (retired[0] / "checkpoint.json").read_text(encoding="utf-8")
            )
            self.assertEqual(retired_metadata["epoch"], 1)


if __name__ == "__main__":
    unittest.main()
