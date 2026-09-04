from __future__ import annotations

import copy
import csv
import hashlib
import inspect
import json
import pathlib
import shutil
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

import jax
import jax.numpy as jnp
from flax import nnx
import pytest

import train_pi05_frs.evaluate as evaluate_module
import train_pi05_frs.utils.metrics as metrics_module
import train_pi05_frs.utils.model as model_module
from train_pi05_frs.utils.checkpoint import load_checkpoint
from train_pi05_frs.utils.checkpoint import resolve_checkpoint_snapshot
from train_pi05_frs.utils.checkpoint import save_checkpoint
from train_pi05_frs.utils.bimanual_schema import bimanual_objective_metadata
from train_pi05_frs.utils.objective_schema import composite_gated_objective_metadata
from train_pi05_frs.utils.model import DecoderConfig
from train_pi05_frs.utils.model import TactileConditionedFlowDecoder
from train_pi05_frs.utils.model import bimanual_composite_endpoint
from train_pi05_frs.utils.model import bimanual_loss_components_per_sample
from train_pi05_frs.utils.model import bimanual_mse_per_sample
from train_pi05_frs.utils.model import composite_endpoint
from train_pi05_frs.utils.model import decode_actions
from train_pi05_frs.utils.model import decode_euler
from train_pi05_frs.utils.metrics import bimanual_source_decode_metrics
from train_pi05_frs.utils.metrics import EvaluationResult
from train_pi05_frs.utils.metrics import evaluate_split
from train_pi05_frs.utils.metrics import gate_stratified_decode_metrics
from train_pi05_frs.utils.bimanual_visualize import plot_bimanual_training_overview
from train_pi05_frs.utils.model import decode_mse_per_sample
from train_pi05_frs.utils.model import flow_matching_loss_per_sample
from train_pi05_frs.utils.model import gated_flow_matching_loss_per_sample
from train_pi05_frs.utils.model import gt_supervised_loss_per_sample
from train_pi05_frs.utils.model import make_optimizer
from train_pi05_frs.utils.model import masked_flow_matching_loss_per_sample
from train_pi05_frs.utils.model import train_step
from train_pi05_frs.train import (
    _validate_resume_loss_objective,
    _validate_resume_cache_provenance,
    _validate_training_path_boundaries,
    checkpoint_selection_key,
    train_decoder,
)
import numpy as np


@pytest.mark.parametrize(
    ("write_plots", "expected_keep_predictions"),
    ((False, False), (True, True)),
)
def test_bimanual_evaluate_decoder_emits_source_local_and_wrist_outputs(
    tmp_path, monkeypatch, write_plots, expected_keep_predictions
):
    class FakePairs:
        source_names = ("dataset-a", "dataset-b")
        manifest = {
            "action_horizon": 1,
            "action_dim": 32,
            "state_dim": 0,
            "records_sha256": "records",
        }

        def __init__(self, cache_dirs, *, source_names):
            assert len(cache_dirs) == 2
            assert tuple(source_names) == self.source_names

        def source_and_local_indices(self, indices):
            np.testing.assert_array_equal(indices, [0, 1, 2, 3])
            return np.asarray([0, 0, 1, 1]), np.asarray([0, 1, 0, 1])

        def metadata_values(self, indices, key):
            del indices
            return {
                "dataset_index": np.asarray([10, 11, 20, 21]),
                "episode_index": np.asarray([1, 1, 2, 2]),
            }[key]

    class FakeConditioner:
        resnet_embedding_dim = 4

        def __init__(self, pairs, **kwargs):
            del pairs
            assert kwargs["build_episode_baselines"] is True

        def close(self):
            return None

    model = SimpleNamespace(
        config=SimpleNamespace(
            action_horizon=1,
            action_dim=32,
            state_conditioning=False,
            tactile_window=1,
            resnet_embedding_dim=4,
        )
    )
    gate_left = np.asarray([1.0, 0.0, 1.0, 0.0])
    gate_right = np.asarray([0.0, 1.0, 0.0, 1.0])
    sample_gt_left = np.asarray([0.0, 0.0, 4.0, 0.0])
    sample_gt_right = np.zeros((4,))
    sample_vla_left = np.asarray([1.0, 0.0, 1.0, 0.0])
    sample_vla_right = np.asarray([0.0, 1.0, 0.0, 1.0])
    result = EvaluationResult(
        target="gt",
        flow_loss=1.0,
        mse=1.0,
        rmse=1.0,
        mae=1.0,
        flow_loss_gt=1.0,
        mse_gt=1.0,
        rmse_gt=1.0,
        mae_gt=1.0,
        flow_loss_pred=2.0,
        mse_pred=2.0,
        rmse_pred=np.sqrt(2.0),
        mae_pred=2.0,
        cache_indices=np.arange(4),
        sample_flow_loss=np.ones((4,)),
        sample_mse=np.ones((4,)),
        sample_rmse=np.ones((4,)),
        sample_mae=np.ones((4,)),
        sample_mse_gt=np.ones((4,)),
        sample_mae_gt=np.ones((4,)),
        sample_mse_pred=np.full((4,), 2.0),
        sample_mae_pred=np.full((4,), 2.0),
        predictions=np.zeros((4, 1, 32), dtype=np.float32),
        mse_vla_gt=1.0,
        gt_gain=0.0,
        relative_gt_error=1.0,
        sample_mse_vla_gt=np.ones((4,)),
        sample_gt_gain=np.zeros((4,)),
        sample_relative_gt_error=np.ones((4,)),
        sample_gate_w_left=gate_left,
        sample_gate_w_right=gate_right,
        sample_tactile_change_left=gate_left,
        sample_tactile_change_right=gate_right,
        sample_composite_fm=np.ones((4,)),
        composite_fm=1.0,
        sample_mse_gt_left=sample_gt_left,
        sample_mse_gt_right=sample_gt_right,
        sample_mse_vla_left=sample_vla_left,
        sample_mse_vla_right=sample_vla_right,
        sample_mse_vla_gt_left=np.ones((4,)),
        sample_mse_vla_gt_right=np.ones((4,)),
        n_high_w_left=2,
        n_low_w_left=2,
        n_mid_w_left=0,
        n_high_w_right=2,
        n_low_w_right=2,
        n_mid_w_right=0,
        gate_w_left=0.5,
        gate_w_right=0.5,
        tactile_change_left=0.5,
        tactile_change_right=0.5,
        bimanual_quadrants={},
        bimanual_gate_region_counts=np.zeros((3, 3), dtype=np.int64),
        gt_actions=np.zeros((4, 1, 32), dtype=np.float32),
        vla_actions=np.zeros((4, 1, 32), dtype=np.float32),
    )
    monkeypatch.setattr(evaluate_module, "MultiCachedPairs", FakePairs, raising=False)
    monkeypatch.setattr(
        evaluate_module, "CachedTactileEmbeddingBatches", FakeConditioner, raising=False
    )
    monkeypatch.setattr(
        evaluate_module,
        "load_checkpoint",
        lambda path: (
            model,
            {
                "epoch": 3,
                "extra_metadata": {
                    **bimanual_objective_metadata(action_dim=32),
                    "cache_records_sha256": "records",
                    "gate_tau": 0.4,
                    "gate_temperature": 0.1,
                    "low_gate_threshold": 0.3,
                    "high_gate_threshold": 0.7,
                },
            },
        ),
    )
    captured = {}

    def fake_evaluate_split(*args, **kwargs):
        captured.update(kwargs)
        return result

    monkeypatch.setattr(evaluate_module, "evaluate_split", fake_evaluate_split)
    plot_calls = []
    monkeypatch.setattr(
        evaluate_module,
        "plot_bimanual_diagnostics",
        lambda *args, **kwargs: plot_calls.append((args, kwargs))
        or tuple(
            tmp_path / "output" / name
            for name in (
                "training_overview.png",
                "bimanual_behavior.png",
                "gate_diagnostics.png",
                "bimanual_action_examples.png",
            )
        ),
    )

    run_dir = tmp_path / "run"
    checkpoint_dir = run_dir / "best"
    run_dir.mkdir()
    (run_dir / "history.csv").write_text("epoch\n1\n", encoding="utf-8")

    metrics = evaluate_module.evaluate_decoder(
        cache_dir=None,
        cache_dirs=[tmp_path / "a", tmp_path / "b"],
        dataset_sources=[{"repo_id": "dataset-a"}, {"repo_id": "dataset-b"}],
        tactile_embedding_cache_root=tmp_path / "tactile-cache",
        tactile_keys=[
            "observation.images.tactile_left_0",
            "observation.images.tactile_right_0",
            "observation.images.tactile_left_1",
            "observation.images.tactile_right_1",
        ],
        tactile_embedding_dim=4,
        tactile_image_size=8,
        tactile_encoder_dir=tmp_path / "encoder",
        checkpoint_dir=checkpoint_dir,
        output_dir=tmp_path / "output",
        dataset_repo_id=None,
        dataset_root=None,
        tactile_window_divisor=None,
        history_stride=None,
        batch_size=4,
        num_steps=1,
        solver="euler",
        target=None,
        save_predictions=False,
        write_plots=write_plots,
        num_trajectory_samples=0,
        num_episode_strips=0,
        num_workers=0,
        prefetch_batches=1,
        load_threads=1,
        pipeline_prefetch=1,
        image_cache_size=0,
    )

    assert captured["loss_mode"] == "bimanual_gated"
    assert captured["keep_predictions"] is expected_keep_predictions
    assert bool(plot_calls) is write_plots
    if write_plots:
        assert plot_calls[0][0][0] == run_dir / "history.csv"
        assert plot_calls[0][0][1].predictions is not None
        assert plot_calls[0][1]["pairs"] is not None
    assert not (tmp_path / "output" / "predictions.npz").exists()
    assert metrics["gate_w_mean_left"] == pytest.approx(0.5)
    assert metrics["per_dataset"]["dataset-b"]["mse_gt"] == pytest.approx(1.0)
    assert metrics["per_dataset"]["dataset-b"]["gt_gain_high_w_left"] == pytest.approx(-3.0)
    assert metrics["min_dataset_gt_gain_high_w_left"] == pytest.approx(-3.0)
    with (tmp_path / "output" / "per_sample.csv").open(encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    assert rows[2]["source"] == "dataset-b"
    assert rows[2]["source_cache_index"] == "0"
    assert rows[2]["mse_gt_left"] == "4.0"


def test_bimanual_evaluation_uses_first_20_dims_and_keeps_wrists_separate(
    monkeypatch,
):
    gt_action = np.zeros((2, 1, 32), dtype=np.float32)
    predicted_action = np.ones((2, 1, 32), dtype=np.float32)
    predicted_action[..., 20:] = 9.0
    prediction = np.ones((2, 1, 32), dtype=np.float32)
    prediction[0, :, :10] = 0.0
    prediction[1, :, 10:20] = 2.0
    prediction[..., 20:] = 100.0

    class FakeConditioner:
        episode_baselines = {0: np.zeros((4, 4), dtype=np.float32)}

        def batches(self, split, *, batch_size, shuffle, seed):
            del batch_size, shuffle, seed
            assert split == "val"
            yield (
                np.asarray([0, 1], dtype=np.int64),
                np.zeros_like(gt_action),
                predicted_action,
                gt_action,
                np.zeros((2, 0), dtype=np.float32),
                np.zeros((2, 1, 4, 4), dtype=np.float32),
            )

        def tactile_change_per_wrist_for_cache_indices(self, indices, current_tokens):
            del indices, current_tokens
            return np.asarray([[1.0, 0.0], [0.0, 0.8]], dtype=np.float32)

    def fake_flow(model, x_base, target, t, tactile_input, state):
        del model, x_base, t, tactile_input, state
        return np.mean(np.square(np.asarray(target)), axis=(1, 2))

    def fake_masked_flow(model, x_base, target, t, tactile_input, state):
        del model, x_base, t, tactile_input, state
        return np.mean(np.square(np.asarray(target)[..., :20]), axis=(1, 2))

    monkeypatch.setattr(metrics_module, "flow_matching_loss_per_sample", fake_flow)
    monkeypatch.setattr(
        metrics_module,
        "masked_flow_matching_loss_per_sample",
        fake_masked_flow,
        raising=False,
    )
    monkeypatch.setattr(
        metrics_module,
        "decode_bimanual_actions",
        lambda *args, **kwargs: prediction,
    )

    result = evaluate_split(
        object(),  # type: ignore[arg-type]
        FakeConditioner(),  # type: ignore[arg-type]
        split="val",
        batch_size=2,
        num_steps=1,
        keep_predictions=True,
        loss_mode="bimanual_gated",
        gate_tau=0.5,
        gate_temperature=0.1,
        rank_low_gate_threshold=0.2,
        rank_high_gate_threshold=0.8,
    )

    np.testing.assert_allclose(result.sample_composite_fm, [0.5, 0.5])
    assert result.composite_fm == pytest.approx(0.5)
    np.testing.assert_allclose(result.sample_mse_gt_left, [0.0, 1.0])
    np.testing.assert_allclose(result.sample_mse_gt_right, [1.0, 4.0])
    np.testing.assert_allclose(result.sample_mse_vla_left, [1.0, 0.0])
    np.testing.assert_allclose(result.sample_mse_vla_right, [0.0, 1.0])
    assert result.bimanual_quadrants["high_low"]["n"] == 1
    assert result.bimanual_gate_region_counts.shape == (3, 3)
    assert result.predictions.shape[-1] == 32
    np.testing.assert_allclose(result.gt_actions, gt_action)
    np.testing.assert_allclose(result.vla_actions, predicted_action)


def test_single_hand_composite_evaluation_retains_plot_inputs(monkeypatch) -> None:
    gt_action = np.zeros((2, 2, 10), dtype=np.float32)
    vla_action = np.ones_like(gt_action)
    prediction = np.stack((gt_action[0], vla_action[1]), axis=0)

    class FakeConditioner:
        episode_baselines = {0: np.zeros((2, 4), dtype=np.float32)}

        def batches(self, split, *, batch_size, shuffle, seed):
            del batch_size, shuffle, seed
            assert split == "val"
            yield (
                np.asarray([0, 1], dtype=np.int64),
                np.zeros_like(gt_action),
                vla_action,
                gt_action,
                np.zeros((2, 0), dtype=np.float32),
                np.zeros((2, 1, 2, 4), dtype=np.float32),
            )

        def tactile_change_for_cache_indices(self, indices, current_tokens):
            del indices, current_tokens
            return np.asarray([1.0, 0.0], dtype=np.float32)

    def fake_flow(model, x_base, target, t, tactile_input, state):
        del model, x_base, t, tactile_input, state
        return np.mean(np.square(np.asarray(target)), axis=(1, 2))

    monkeypatch.setattr(metrics_module, "flow_matching_loss_per_sample", fake_flow)
    monkeypatch.setattr(
        metrics_module,
        "decode_actions",
        lambda *args, **kwargs: prediction,
    )

    result = evaluate_split(
        object(),  # type: ignore[arg-type]
        FakeConditioner(),  # type: ignore[arg-type]
        split="val",
        batch_size=2,
        num_steps=1,
        keep_predictions=True,
        loss_mode="composite_gated",
        gate_tau=0.5,
        gate_temperature=0.1,
        low_gate_threshold=0.3,
        high_gate_threshold=0.7,
    )

    np.testing.assert_allclose(result.sample_composite_fm, [0.1, 1.0])
    assert result.composite_fm == pytest.approx(0.55)
    np.testing.assert_allclose(result.sample_mse_vla_gt, [1.0, 1.0])
    np.testing.assert_allclose(result.sample_gt_gain, [1.0, 0.0])
    assert result.sample_gate_w.shape == (2,)
    assert result.sample_tactile_change.shape == (2,)
    np.testing.assert_allclose(result.predictions, prediction)
    np.testing.assert_allclose(result.gt_actions, gt_action)
    np.testing.assert_allclose(result.vla_actions, vla_action)


def test_single_hand_composite_evaluation_ignores_gripper_metrics(monkeypatch) -> None:
    gt_action = np.zeros((2, 1, 10), dtype=np.float32)
    vla_action = np.ones_like(gt_action)
    vla_action[..., 9] = -75.0
    prediction = np.ones_like(gt_action)
    prediction[0, :, :9] = 0.0
    prediction[..., 9] = 100.0

    class FakeConditioner:
        episode_baselines = {0: np.zeros((2, 4), dtype=np.float32)}

        def batches(self, split, *, batch_size, shuffle, seed):
            del batch_size, shuffle, seed
            assert split == "val"
            yield (
                np.asarray([0, 1], dtype=np.int64),
                np.zeros_like(gt_action),
                vla_action,
                gt_action,
                np.zeros((2, 0), dtype=np.float32),
                np.zeros((2, 1, 2, 4), dtype=np.float32),
            )

        def tactile_change_for_cache_indices(self, indices, current_tokens):
            del indices, current_tokens
            return np.asarray([1.0, 0.0], dtype=np.float32)

    monkeypatch.setattr(
        metrics_module,
        "flow_matching_loss_per_sample",
        lambda *args, **kwargs: np.zeros((2,), dtype=np.float32),
    )
    monkeypatch.setattr(metrics_module, "decode_actions", lambda *args, **kwargs: prediction)

    result = evaluate_split(
        object(),  # type: ignore[arg-type]
        FakeConditioner(),  # type: ignore[arg-type]
        split="val",
        batch_size=2,
        num_steps=1,
        keep_predictions=True,
        loss_mode="composite_gated",
        gate_tau=0.5,
        gate_temperature=0.1,
        low_gate_threshold=0.3,
        high_gate_threshold=0.7,
    )

    np.testing.assert_allclose(result.sample_mse_gt, [0.0, 1.0])
    np.testing.assert_allclose(result.sample_mse_pred, [1.0, 0.0])
    np.testing.assert_allclose(result.sample_mse_vla_gt, [1.0, 1.0])
    assert result.mse_gt == pytest.approx(0.5)
    assert result.mse_pred == pytest.approx(0.5)
    assert result.mse_vla_gt == pytest.approx(1.0)
    assert result.high_gate_gain == pytest.approx(1.0)
    assert result.high_gate_rank_satisfied_frac == pytest.approx(1.0)
    assert result.high_gate_repair_satisfied_frac == pytest.approx(1.0)
    assert result.low_gate_unsafe_frac == pytest.approx(0.0)
    assert result.low_gate_regression_frac == pytest.approx(0.0)
    assert result.high_gate_harm_p95 == pytest.approx(0.0)
    assert result.predictions.shape[-1] == 10
    np.testing.assert_allclose(result.predictions[..., 9], 100.0)
    np.testing.assert_allclose(result.vla_actions[..., 9], -75.0)


def test_single_hand_low_gate_safety_preserves_vla_not_nearest_endpoint() -> None:
    decoded = jnp.zeros((1, 1, 1), dtype=jnp.float32)
    gt = jnp.zeros_like(decoded)
    vla = jnp.ones_like(decoded)

    loss = model_module.low_gate_safety_loss_per_sample(
        decoded,
        gt,
        vla,
        jnp.asarray([0.0], dtype=jnp.float32),
        tolerance=0.03,
        low_gate_threshold=0.3,
    )

    assert float(loss[0]) == pytest.approx(0.97)


def test_bimanual_aggregate_metrics_and_selection_ignore_padding_tail(
    monkeypatch,
) -> None:
    gt_action = np.zeros((2, 1, 32), dtype=np.float32)
    physical_vla = np.ones((2, 1, 20), dtype=np.float32)
    physical_prediction = np.ones((2, 1, 20), dtype=np.float32)
    physical_prediction[0, :, :10] = 0.0
    physical_prediction[1, :, 10:20] = 2.0
    current: dict[str, np.ndarray] = {}

    class FakeConditioner:
        episode_baselines = {0: np.zeros((4, 4), dtype=np.float32)}

        def batches(self, split, *, batch_size, shuffle, seed):
            del batch_size, shuffle, seed
            assert split == "val"
            yield (
                np.asarray([0, 1], dtype=np.int64),
                np.zeros_like(gt_action),
                current["vla"],
                gt_action,
                np.zeros((2, 0), dtype=np.float32),
                np.zeros((2, 1, 4, 4), dtype=np.float32),
            )

        def tactile_change_per_wrist_for_cache_indices(self, indices, tokens):
            del indices, tokens
            return np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)

    def fake_full_width_flow(model, x_base, target, t, tactile_input, state):
        del model, x_base, t, tactile_input, state
        return np.mean(np.square(np.asarray(target)), axis=(1, 2))

    def fake_physical_flow(model, x_base, target, t, tactile_input, state):
        del model, x_base, t, tactile_input, state
        return np.mean(np.square(np.asarray(target)[..., :20]), axis=(1, 2))

    monkeypatch.setattr(
        metrics_module, "flow_matching_loss_per_sample", fake_full_width_flow
    )
    monkeypatch.setattr(
        metrics_module,
        "masked_flow_matching_loss_per_sample",
        fake_physical_flow,
    )
    monkeypatch.setattr(
        metrics_module,
        "decode_bimanual_actions",
        lambda *args, **kwargs: current["prediction"],
    )

    def evaluate_with_tail(*, prediction_tail: float, vla_tail: float):
        current["prediction"] = np.concatenate(
            (
                physical_prediction,
                np.full((2, 1, 12), prediction_tail, dtype=np.float32),
            ),
            axis=-1,
        )
        current["vla"] = np.concatenate(
            (
                physical_vla,
                np.full((2, 1, 12), vla_tail, dtype=np.float32),
            ),
            axis=-1,
        )
        result = evaluate_split(
            object(),  # type: ignore[arg-type]
            FakeConditioner(),  # type: ignore[arg-type]
            split="val",
            batch_size=2,
            num_steps=1,
            keep_predictions=False,
            loss_mode="bimanual_gated",
            gate_tau=0.5,
            gate_temperature=0.1,
        )
        selection = checkpoint_selection_key(
            {
                "val_mse_gt": result.mse_gt,
                "val_low_unsafe_frac_left": 0.0,
                "val_low_unsafe_frac_right": 0.0,
                "val_gt_gain_high_w_left": 0.5,
                "val_gt_gain_high_w_right": 0.5,
                "val_rank_satisfied_high_frac_left": 1.0,
                "val_rank_satisfied_high_frac_right": 1.0,
            },
            loss_mode="bimanual_gated",
            max_low_gate_unsafe_frac=0.1,
            min_high_gate_gain=0.0,
            min_high_gate_rank_satisfied_frac=0.8,
        )
        return result, selection

    baseline, baseline_key = evaluate_with_tail(prediction_tail=0.0, vla_tail=0.0)
    perturbed, perturbed_key = evaluate_with_tail(
        prediction_tail=100.0, vla_tail=-75.0
    )

    for field in (
        "flow_loss",
        "mse",
        "rmse",
        "mae",
        "flow_loss_gt",
        "mse_gt",
        "rmse_gt",
        "mae_gt",
        "flow_loss_pred",
        "mse_pred",
        "rmse_pred",
        "mae_pred",
        "mse_vla_gt",
        "gt_gain",
        "relative_gt_error",
        "composite_fm",
    ):
        assert getattr(perturbed, field) == pytest.approx(getattr(baseline, field))
    for field in (
        "sample_flow_loss",
        "sample_mse",
        "sample_rmse",
        "sample_mae",
        "sample_mse_gt",
        "sample_mae_gt",
        "sample_mse_pred",
        "sample_mae_pred",
        "sample_mse_vla_gt",
        "sample_gt_gain",
        "sample_relative_gt_error",
    ):
        np.testing.assert_allclose(getattr(perturbed, field), getattr(baseline, field))
    assert perturbed_key == baseline_key


def test_bimanual_evaluation_uses_physical_decode_path_and_restores_vla_tail() -> None:
    gt_action = np.zeros((1, 2, 32), dtype=np.float32)
    vla_action = np.zeros_like(gt_action)
    vla_action[..., 20:] = 0.25

    class FakeConditioner:
        episode_baselines = {0: np.zeros((4, 1), dtype=np.float32)}

        def __init__(self, x_base_tail: float):
            self.x_base_tail = x_base_tail

        def batches(self, split, *, batch_size, shuffle, seed):
            del batch_size, shuffle, seed
            assert split == "val"
            x_base = np.zeros_like(gt_action)
            x_base[..., 20:] = self.x_base_tail
            yield (
                np.asarray([0], dtype=np.int64),
                x_base,
                vla_action,
                gt_action,
                np.zeros((1, 0), dtype=np.float32),
                np.zeros((1, 1, 4, 1), dtype=np.float32),
            )

        def tactile_change_per_wrist_for_cache_indices(self, indices, tokens):
            del indices, tokens
            return np.ones((1, 2), dtype=np.float32)

    def evaluate(x_base_tail: float):
        return evaluate_split(
            _TailCoupledVelocity(),
            FakeConditioner(x_base_tail),  # type: ignore[arg-type]
            split="val",
            batch_size=1,
            num_steps=2,
            keep_predictions=True,
            loss_mode="bimanual_gated",
            gate_tau=0.5,
            gate_temperature=0.1,
        )

    baseline = evaluate(0.0)
    perturbed = evaluate(5.0)

    assert baseline.mse_gt == pytest.approx(0.0)
    assert perturbed.mse_gt == pytest.approx(baseline.mse_gt)
    np.testing.assert_allclose(baseline.predictions[..., 20:], 0.25)
    np.testing.assert_allclose(perturbed.predictions, baseline.predictions)


@pytest.mark.parametrize("action_dim", (19, 21, 24, 33))
def test_bimanual_model_core_rejects_unsupported_action_widths(
    action_dim: int,
) -> None:
    actions = jnp.zeros((1, 2, action_dim), dtype=jnp.float32)
    with pytest.raises(ValueError, match="action_dim"):
        bimanual_composite_endpoint(
            actions,
            actions,
            jnp.ones((1, 2), dtype=jnp.float32),
        )


def test_bimanual_training_and_evaluation_boundaries_reject_tactile_permutation() -> None:
    metadata = bimanual_objective_metadata(action_dim=32)
    permutation = (
        "observation.images.tactile_right_0",
        "observation.images.tactile_left_0",
        "observation.images.tactile_left_1",
        "observation.images.tactile_right_1",
    )

    import train_pi05_frs.train as train_module

    with pytest.raises(ValueError, match="fixed bimanual tactile key order"):
        train_module._validate_bimanual_training_contract(
            loss_mode="bimanual_gated",
            action_dim=32,
            tactile_keys=permutation,
        )
    with pytest.raises(ValueError, match="fixed bimanual tactile key order"):
        evaluate_module._validate_bimanual_evaluation_contract(
            metadata,
            action_dim=32,
            tactile_keys=permutation,
        )


def test_direct_cli_accepts_bimanual_without_applying_gate_lambda(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import train_pi05_frs.train as train_module

    captured: dict[str, object] = {}
    monkeypatch.setattr(
        train_module, "train_decoder", lambda **kwargs: captured.update(kwargs)
    )

    train_module.main(
        [
            "--cache-dir",
            str(tmp_path / "cache"),
            "--tactile-encoder-dir",
            str(tmp_path / "encoder"),
            "--output-dir",
            str(tmp_path / "output"),
            "--loss-mode",
            "bimanual_gated",
        ]
    )

    assert captured["loss_mode"] == "bimanual_gated"
    assert captured["gate_lambda"] == 0.0


def test_resume_history_header_must_match_exactly_before_append(
    tmp_path: pathlib.Path,
) -> None:
    import train_pi05_frs.train as train_module

    history = tmp_path / "history.csv"
    history.write_text("epoch,val_mse,train_loss\n1,0.1,0.2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="history.csv header"):
        train_module._validate_history_csv_header(
            history, ("epoch", "train_loss", "val_mse")
        )

    history.write_text("epoch,train_loss,val_mse\n1,0.2,0.1\n", encoding="utf-8")
    train_module._validate_history_csv_header(
        history, ("epoch", "train_loss", "val_mse")
    )


def test_bimanual_source_metrics_aggregate_before_worst_rollups():
    per_source, rollups = bimanual_source_decode_metrics(
        sample_mse_gt_left=np.asarray([0.0, 0.0, 4.0, 0.0]),
        sample_mse_gt_right=np.asarray([0.0, 0.0, 0.0, 0.0]),
        sample_mse_vla_left=np.asarray([1.0, 0.0, 1.0, 0.0]),
        sample_mse_vla_right=np.asarray([0.0, 1.0, 0.0, 1.0]),
        sample_mse_vla_gt_left=np.ones((4,)),
        sample_mse_vla_gt_right=np.ones((4,)),
        sample_gate_w_left=np.asarray([1.0, 0.0, 1.0, 0.0]),
        sample_gate_w_right=np.asarray([0.0, 1.0, 0.0, 1.0]),
        source_indices=np.asarray([0, 0, 1, 1]),
        num_sources=2,
        low_w_threshold=0.3,
        high_w_threshold=0.7,
        ranking_margin=0.0,
        repair_margin=0.0,
        low_safety_margin=0.0,
    )

    assert per_source[0]["gt_gain_high_w_left"] == pytest.approx(1.0)
    assert per_source[1]["gt_gain_high_w_left"] == pytest.approx(-3.0)
    assert rollups["min_dataset_gt_gain_high_w_left"] == pytest.approx(-3.0)
    assert rollups["min_dataset_rank_satisfied_high_frac_left"] == pytest.approx(0.0)
    assert rollups["worst_dataset_low_unsafe_frac_left"] == pytest.approx(0.0)


@pytest.mark.parametrize(
    ("write_plots", "expected_keep_predictions"),
    ((False, False), (True, True)),
)
def test_bimanual_trainer_validation_writes_finite_source_wrist_selection(
    tmp_path, monkeypatch, write_plots, expected_keep_predictions
):
    import train_pi05_frs.pi05_cache.cache as cache_module
    import train_pi05_frs.utils.checkpoint as checkpoint_module
    import train_pi05_frs.utils.data as data_module
    import train_pi05_frs.utils.model as model_module

    source_calls = []

    class FakeMultiPairs:
        source_names = ("dataset-a", "dataset-b")
        sources = (object(), object())
        manifest = {
            "sample_count": 4,
            "train_sample_count": 4,
            "val_sample_count": 4,
            "action_horizon": 1,
            "action_dim": 32,
            "state_dim": 0,
            "records_sha256": "records",
            "configuration": {"sources": ["dataset-a", "dataset-b"]},
        }

        def __init__(self, cache_dirs, *, source_names):
            assert len(cache_dirs) == 2
            assert tuple(source_names) == self.source_names

        def indices(self, split):
            assert split in ("train", "val")
            return np.arange(4, dtype=np.int64)

        def source_and_local_indices(self, indices):
            indices = np.asarray(indices)
            source_calls.append(indices.copy())
            np.testing.assert_array_equal(indices, np.arange(4))
            return np.asarray([0, 0, 1, 1]), np.asarray([0, 1, 0, 1])

    class FakeConditioner:
        resnet_embedding_dim = 4
        episode_baselines = {(0, 0): np.zeros((4, 4), dtype=np.float32)}

        def __init__(self, pairs, **kwargs):
            del pairs
            assert kwargs["build_episode_baselines"] is True

        def batches(self, split, *, batch_size, shuffle, seed):
            del batch_size, shuffle, seed
            assert split == "train"
            shape = (4, 1, 32)
            yield (
                np.arange(4, dtype=np.int64),
                np.zeros(shape, dtype=np.float32),
                np.ones(shape, dtype=np.float32),
                np.zeros(shape, dtype=np.float32),
                np.zeros((4, 0), dtype=np.float32),
                jnp.zeros((4, 1, 4, 4), dtype=jnp.float32),
            )

        def tactile_change_per_wrist_for_cache_indices(self, indices, tokens):
            del indices, tokens
            return np.asarray(
                [[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0]],
                dtype=np.float32,
            )

        def close(self):
            return None

    gate_left = np.asarray([1.0, 0.0, 1.0, 0.0])
    gate_right = np.asarray([0.0, 1.0, 0.0, 1.0])
    wrist_metrics = {
        "mse_gt_high_w": 0.1,
        "mse_vla_high_w": 0.2,
        "mse_vla_gt_high_w": 1.0,
        "gt_gain_high_w": 0.9,
        "rank_penalty_high_w": 0.0,
        "rank_satisfied_high_frac": 1.0,
        "repair_penalty_high_w": 0.0,
        "repair_satisfied_high_frac": 1.0,
        "low_nearest_endpoint_mse": 0.0,
        "low_safety_penalty": 0.0,
        "low_safe_frac": 1.0,
        "low_unsafe_frac": 0.0,
        "n_high_w": 2,
        "n_low_w": 2,
        "n_mid_w": 0,
    }
    validation_values = {
        "target": "gt",
        "flow_loss": 1.0,
        "mse": 1.0,
        "rmse": 1.0,
        "mae": 1.0,
        "flow_loss_gt": 1.0,
        "mse_gt": 1.0,
        "rmse_gt": 1.0,
        "mae_gt": 1.0,
        "flow_loss_pred": 2.0,
        "mse_pred": 2.0,
        "rmse_pred": np.sqrt(2.0),
        "mae_pred": 2.0,
        "mse_vla_gt": 1.0,
        "gt_gain": 0.0,
        "relative_gt_error": 1.0,
        "cache_indices": np.arange(4),
        "n_high_w": None,
        "composite_fm": 0.375,
        "sample_mse_gt_left": np.asarray([0.0, 0.0, 4.0, 0.0]),
        "sample_mse_gt_right": np.zeros((4,)),
        "sample_mse_vla_left": np.asarray([1.0, 0.0, 1.0, 0.0]),
        "sample_mse_vla_right": np.asarray([0.0, 1.0, 0.0, 1.0]),
        "sample_mse_vla_gt_left": np.ones((4,)),
        "sample_mse_vla_gt_right": np.ones((4,)),
        "sample_gate_w_left": gate_left,
        "sample_gate_w_right": gate_right,
        "bimanual_quadrants": {
            quadrant: {
                "n": 1,
                **{
                    wrist: {
                        "mse_gt": 0.1,
                        "mse_vla": 0.2,
                        "mse_vla_gt": 0.3,
                        "gt_gain": 0.2,
                        "relative_gt_error": 1 / 3,
                        "vla_preserve_ratio": 2 / 3,
                        "rank_satisfied_frac": 1.0,
                    }
                    for wrist in ("left", "right")
                },
            }
            for quadrant in ("low_low", "high_low", "low_high", "high_high")
        },
    }
    for wrist, gate, change in (
        ("left", 0.5, 0.6),
        ("right", 0.5, 0.4),
    ):
        validation_values[f"gate_w_{wrist}"] = gate
        validation_values[f"tactile_change_{wrist}"] = change
        validation_values.update(
            {f"{name}_{wrist}": value for name, value in wrist_metrics.items()}
        )
        for quantile in ("p10", "p25", "p50", "p75", "p90"):
            validation_values[f"gate_w_{quantile}_{wrist}"] = gate
            validation_values[f"tactile_change_{quantile}_{wrist}"] = change
    validation = SimpleNamespace(**validation_values)

    monkeypatch.setattr(cache_module, "MultiCachedPairs", FakeMultiPairs)
    monkeypatch.setattr(data_module, "CachedTactileEmbeddingBatches", FakeConditioner)
    monkeypatch.setattr(
        model_module,
        "train_step",
        lambda *args, **kwargs: (
            jnp.asarray(0.0),
            {
                name: jnp.asarray(0.0)
                for name in (
                    "gt_fm",
                    "vla_fm",
                    "composite_fm",
                    "low_safety",
                    "decode",
                    "rank",
                    "repair",
                )
            },
        ),
    )
    evaluation_kwargs = {}

    def fake_evaluate_split(*args, **kwargs):
        del args
        evaluation_kwargs.update(kwargs)
        return validation

    monkeypatch.setattr(metrics_module, "evaluate_split", fake_evaluate_split)
    checkpoint_metrics = []

    def fake_save_checkpoint(path, model, *, epoch, metrics, **kwargs):
        del path, model, epoch, kwargs
        checkpoint_metrics.append(dict(metrics))

    monkeypatch.setattr(checkpoint_module, "save_checkpoint", fake_save_checkpoint)
    import train_pi05_frs.utils.bimanual_visualize as visualize_module

    rendered_diagnostics = []
    monkeypatch.setattr(
        visualize_module,
        "plot_bimanual_diagnostics",
        lambda *args, **kwargs: rendered_diagnostics.append((args, kwargs)) or (),
    )

    train_decoder(
        cache_dir=None,
        cache_dirs=[tmp_path / "cache-a", tmp_path / "cache-b"],
        dataset_sources=[{"repo_id": "dataset-a"}, {"repo_id": "dataset-b"}],
        tactile_embedding_cache_root=tmp_path / "tactile-cache",
        tactile_keys=[
            "observation.images.tactile_left_0",
            "observation.images.tactile_right_0",
            "observation.images.tactile_left_1",
            "observation.images.tactile_right_1",
        ],
        tactile_embedding_dim=4,
        tactile_image_size=8,
        tactile_encoder_dir=tmp_path / "encoder",
        output_dir=tmp_path / "output",
        dataset_repo_id=None,
        dataset_root=None,
        tactile_window_divisor=1,
        history_stride=1,
        loss_mode="bimanual_gated",
        gate_tau=0.4,
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
        low_gate_threshold=0.3,
        high_gate_threshold=0.7,
        state_conditioning=False,
        state_dropout_rate=0.0,
        best_max_low_gate_unsafe_frac=0.1,
        best_min_high_gate_gain=0.0,
        best_min_high_gate_rank_satisfied_frac=0.8,
        model_dim=4,
        depth=1,
        num_heads=1,
        mlp_ratio=1,
        learning_rate=1e-3,
        weight_decay=0.0,
        grad_clip_norm=None,
        warmup_epochs=0,
        lr_reference_dim=None,
        min_learning_rate_ratio=0.1,
        cosine_decay=False,
        batch_size=4,
        epochs=1,
        validation_steps=1,
        eval_every=1,
        seed=0,
        write_plots=write_plots,
        num_workers=0,
        prefetch_batches=1,
        load_threads=1,
        pipeline_prefetch=1,
        image_cache_size=0,
        encode_batch_size=1,
    )

    assert evaluation_kwargs["loss_mode"] == "bimanual_gated"
    assert evaluation_kwargs["gate_tau"] == pytest.approx(0.4)
    assert evaluation_kwargs["gate_temperature"] == pytest.approx(0.1)
    assert evaluation_kwargs["low_gate_threshold"] == pytest.approx(0.3)
    assert evaluation_kwargs["high_gate_threshold"] == pytest.approx(0.7)
    assert evaluation_kwargs["low_gate_safety_margin"] == pytest.approx(0.03)
    assert evaluation_kwargs["low_gate_regression_margin"] == pytest.approx(0.005)
    assert evaluation_kwargs["rank_margin"] == pytest.approx(0.0)
    assert evaluation_kwargs["repair_margin"] == pytest.approx(0.0)
    assert evaluation_kwargs["keep_predictions"] is expected_keep_predictions
    assert bool(rendered_diagnostics) is write_plots
    assert len(source_calls) == 2
    with (tmp_path / "output" / "history.csv").open(
        newline="", encoding="utf-8"
    ) as file:
        history_row = next(csv.DictReader(file))
    assert float(history_row["val_composite_fm"]) == pytest.approx(0.375)
    assert float(history_row["val_gate_w_left"]) == pytest.approx(0.5)
    assert int(history_row["checkpoint_selection_feasible"]) == 0
    assert int(history_row["val_quadrant_high_low_n"]) == 1
    assert float(history_row["val_quadrant_high_low_mse_gt_left"]) == pytest.approx(0.1)
    overview = plot_bimanual_training_overview(
        tmp_path / "output" / "history.csv",
        output_path=tmp_path / "output" / "training_overview.png",
    )
    assert overview.is_file()
    assert float(history_row["val_worst_dataset_low_unsafe_frac_left"]) == 0.0
    assert float(history_row["val_min_dataset_gt_gain_high_w_left"]) == -3.0
    assert float(history_row["val_source_1_gt_gain_high_w_left"]) == -3.0
    assert len(checkpoint_metrics) == 2
    for metrics in checkpoint_metrics:
        key = checkpoint_selection_key(
            metrics,
            loss_mode="bimanual_gated",
            max_low_gate_unsafe_frac=0.1,
            min_high_gate_gain=0.0,
            min_high_gate_rank_satisfied_frac=0.8,
        )
        assert all(np.isfinite(key))
        assert metrics["val_min_dataset_gt_gain_high_w_left"] == -3.0


def test_bimanual_resume_requires_exact_objective_metadata():
    valid = {"extra_metadata": bimanual_objective_metadata(action_dim=32)}
    _validate_resume_loss_objective(
        valid, loss_mode="bimanual_gated", action_dim=32
    )
    invalid = copy.deepcopy(valid)
    invalid["extra_metadata"]["steered_action_dim"] = 19
    with pytest.raises(ValueError, match="steered_action_dim"):
        _validate_resume_loss_objective(
            invalid, loss_mode="bimanual_gated", action_dim=32
        )


def test_composite_resume_requires_exact_objective_metadata():
    valid = {
        "extra_metadata": {
            "loss_mode": "composite_gated",
            **composite_gated_objective_metadata(),
        }
    }
    _validate_resume_loss_objective(
        valid, loss_mode="composite_gated", action_dim=10
    )
    invalid = copy.deepcopy(valid)
    invalid["extra_metadata"]["endpoint_policy"] = "dual_flow"

    with pytest.raises(ValueError, match="endpoint_policy"):
        _validate_resume_loss_objective(
            invalid, loss_mode="composite_gated", action_dim=10
        )


def test_bimanual_checkpoint_selection_uses_worst_source_wrist_metrics():
    metrics = {
        "val_mse_gt": 0.25,
        "val_worst_dataset_low_unsafe_frac_left": 0.15,
        "val_worst_dataset_low_unsafe_frac_right": 0.05,
        "val_min_dataset_gt_gain_high_w_left": -0.1,
        "val_min_dataset_gt_gain_high_w_right": 0.2,
        "val_min_dataset_rank_satisfied_high_frac_left": 0.7,
        "val_min_dataset_rank_satisfied_high_frac_right": 0.9,
    }

    key = checkpoint_selection_key(
        metrics,
        loss_mode="bimanual_gated",
        max_low_gate_unsafe_frac=0.1,
        min_high_gate_gain=0.0,
        min_high_gate_rank_satisfied_frac=0.8,
    )

    assert key == pytest.approx((0.25, 0.1, -0.7, 0.15, 0.25))


def test_scalar_checkpoint_selection_enforces_new_safety_constraints_and_order():
    common = {
        "val_mse": 0.02,
        "val_low_gate_unsafe_frac": 0.04,
        "val_high_gate_gain": 0.10,
        "val_high_gate_rank_satisfied_frac": 0.90,
        "val_high_gate_repair_satisfied_frac": 0.90,
        "val_high_gate_harm_p95": 0.02,
        "val_low_gate_regression_frac": 0.04,
    }
    kwargs = {
        "loss_mode": "composite_gated",
        "max_low_gate_unsafe_frac": 0.05,
        "min_high_gate_gain": 0.0,
        "min_high_gate_rank_satisfied_frac": 0.8,
        "min_high_gate_repair_satisfied_frac": 0.8,
        "max_high_gate_harm_p95": 0.03,
        "max_low_gate_regression_frac": 0.05,
    }

    feasible = checkpoint_selection_key(common, **kwargs)
    assert feasible == pytest.approx((0.0, -0.90, 0.02, 0.04, -0.10, 0.02))

    low_rank_composite = checkpoint_selection_key(
        {**common, "val_high_gate_rank_satisfied_frac": 0.0}, **kwargs
    )
    assert low_rank_composite[0] == pytest.approx(0.0)

    low_rank_legacy = checkpoint_selection_key(
        {**common, "val_high_gate_rank_satisfied_frac": 0.0},
        **{**kwargs, "loss_mode": "gated"},
    )
    assert low_rank_legacy[0] > 0.0

    lower_repair = checkpoint_selection_key(
        {
            **common,
            "val_high_gate_repair_satisfied_frac": 0.85,
            "val_high_gate_harm_p95": 0.0,
            "val_mse": 0.0,
        },
        **kwargs,
    )
    assert feasible < lower_repair

    infeasible = checkpoint_selection_key(
        {
            **common,
            "val_high_gate_repair_satisfied_frac": 0.70,
            "val_high_gate_harm_p95": 0.04,
            "val_low_gate_regression_frac": 0.10,
        },
        **kwargs,
    )
    assert infeasible[0] > 0.0


def test_32d_composite_steers_first_20_and_preserves_vla_tail():
    gt = jnp.ones((2, 3, 32))
    vla = jnp.full((2, 3, 32), 2.0)
    gates = jnp.asarray([[1.0, 0.0], [0.0, 1.0]])
    target, effective = bimanual_composite_endpoint(gt, vla, gates)
    np.testing.assert_allclose(target[0, :, :10], 1.0)
    np.testing.assert_allclose(target[0, :, 10:20], 2.0)
    np.testing.assert_allclose(target[..., 20:], 2.0)
    assert effective.shape == (2, 2)


def test_scalar_composite_endpoint_uses_three_gate_regions():
    gt = jnp.ones((3, 2, 4), dtype=jnp.float32)
    vla = jnp.zeros_like(gt)
    target, effective = composite_endpoint(
        gt,
        vla,
        jnp.asarray([0.2, 0.5, 0.8], dtype=jnp.float32),
        low_gate_threshold=0.3,
        high_gate_threshold=0.7,
    )

    np.testing.assert_allclose(effective, [0.0, 0.5, 1.0])
    np.testing.assert_allclose(target[:, 0, 0], [0.0, 0.5, 1.0])


def test_single_hand_composite_training_preserves_and_ignores_gripper(monkeypatch):
    gt = jnp.zeros((2, 1, 10), dtype=jnp.float32)
    vla = jnp.ones_like(gt)
    decoded = vla.at[1, :, :9].set(0.0).at[..., 9].set(100.0)
    gates = jnp.asarray([0.1, 0.9], dtype=jnp.float32)
    captured = {}

    def fake_flow(model, x_base, target, t, tactile_seq, **kwargs):
        del model, x_base, t, tactile_seq, kwargs
        captured["target"] = np.asarray(target)
        return jnp.zeros((2,), dtype=jnp.float32)

    monkeypatch.setattr(model_module, "flow_matching_loss_per_sample", fake_flow)
    monkeypatch.setattr(model_module, "decode_actions", lambda *args, **kwargs: decoded)

    components = model_module.gated_loss_components_per_sample(
        object(),  # type: ignore[arg-type]
        jnp.zeros_like(gt),
        gt,
        vla,
        jnp.full((2,), 0.5),
        jnp.zeros((2, 1, 2, 4)),
        gates,
        gate_lambda=0.0,
        aux_decode_weight=1.0,
        low_gate_safety_weight=1.0,
        low_gate_safety_margin=0.03,
        rank_weight=1.0,
        rank_margin=0.0,
        repair_weight=1.0,
        repair_margin=0.0,
        use_composite_endpoint=True,
    )

    np.testing.assert_allclose(captured["target"][..., 9], 1.0)
    np.testing.assert_allclose(components["decode"], [0.0, 0.0])
    np.testing.assert_allclose(components["low_safety"], [0.0, 0.0])
    np.testing.assert_allclose(components["rank"], [0.0, 0.0])
    np.testing.assert_allclose(components["repair"], [0.0, 0.0])


def test_scalar_composite_decode_matches_bimanual_endpoint_weighting(monkeypatch):
    gt = jnp.zeros((3, 2, 4), dtype=jnp.float32)
    vla = jnp.ones_like(gt)
    decoded = jnp.full_like(gt, 0.25)
    gates = jnp.asarray([0.2, 0.5, 0.8], dtype=jnp.float32)

    monkeypatch.setattr(
        model_module,
        "flow_matching_loss_per_sample",
        lambda *args, **kwargs: jnp.zeros((3,), dtype=jnp.float32),
    )
    monkeypatch.setattr(
        model_module,
        "decode_actions",
        lambda *args, **kwargs: decoded,
    )
    components = model_module.gated_loss_components_per_sample(
        object(),  # type: ignore[arg-type]
        jnp.zeros_like(gt),
        gt,
        vla,
        jnp.full((3,), 0.5),
        jnp.zeros((3, 1, 2, 4)),
        gates,
        gate_lambda=0.0,
        aux_decode_weight=2.0,
        low_gate_safety_weight=0.0,
        rank_weight=0.0,
        repair_weight=0.0,
        use_composite_endpoint=True,
    )

    np.testing.assert_allclose(components["decode"], [1.125, 0.625, 0.125])


def test_bimanual_mse_ignores_32d_padding_tail():
    left = jnp.zeros((1, 2, 32))
    right = left.at[..., 20:].set(100.0)
    np.testing.assert_allclose(bimanual_mse_per_sample(left, right), [[0.0, 0.0]])


def test_bimanual_mse_rejects_width_below_physical_action():
    with pytest.raises(ValueError, match="action_dim"):
        bimanual_mse_per_sample(
            jnp.zeros((1, 2, 19)), jnp.zeros((1, 2, 19))
        )


def test_bimanual_composite_rejects_width_below_physical_action():
    with pytest.raises(ValueError, match="action_dim"):
        bimanual_composite_endpoint(
            jnp.zeros((1, 2, 19)), jnp.zeros((1, 2, 19)), jnp.ones((1, 2))
        )


class _ConstantVelocity:
    def __init__(self, velocity):
        self.velocity = velocity

    def __call__(self, x_t, t, tactile_seq, *, state=None, state_keep_mask=None):
        del t, tactile_seq, state, state_keep_mask
        return jnp.broadcast_to(self.velocity, x_t.shape)


class _TailCoupledVelocity(nnx.Module):
    """Expose any padded decoder input through the physical velocity output."""

    def __call__(self, x_t, t, tactile_seq, *, state=None, state_keep_mask=None):
        del t, tactile_seq, state, state_keep_mask
        return self._velocity(x_t)

    def encode_condition(self, tactile_seq, state, state_keep_mask):
        del state, state_keep_mask
        return tactile_seq

    def velocity_from_condition(self, x_t, t, condition):
        del t, condition
        return self._velocity(x_t)

    @staticmethod
    def _velocity(x_t):
        physical = x_t[..., :20]
        if x_t.shape[-1] == 20:
            return jnp.zeros_like(physical)
        tail_signal = jnp.sum(x_t[..., 20:], axis=-1, keepdims=True)
        coupled_physical = jnp.broadcast_to(tail_signal, physical.shape)
        return jnp.concatenate(
            (coupled_physical, jnp.full_like(x_t[..., 20:], 17.0)), axis=-1
        )


@pytest.mark.parametrize("perturbed_input", ("x_base", "target"))
def test_masked_flow_matching_projects_32d_tail_before_real_model_call(
    perturbed_input: str,
) -> None:
    model = _TailCoupledVelocity()
    x_base = jnp.zeros((1, 2, 32), dtype=jnp.float32)
    target = jnp.concatenate(
        (
            jnp.ones((1, 2, 20), dtype=jnp.float32),
            jnp.zeros((1, 2, 12), dtype=jnp.float32),
        ),
        axis=-1,
    )
    arguments = {
        "x_base": x_base,
        "target": target,
    }
    baseline = masked_flow_matching_loss_per_sample(
        model,
        arguments["x_base"],
        arguments["target"],
        jnp.asarray([0.25], dtype=jnp.float32),
        jnp.zeros((1, 1, 1), dtype=jnp.float32),
    )
    arguments[perturbed_input] = arguments[perturbed_input].at[..., 20:].set(5.0)
    perturbed = masked_flow_matching_loss_per_sample(
        model,
        arguments["x_base"],
        arguments["target"],
        jnp.asarray([0.25], dtype=jnp.float32),
        jnp.zeros((1, 1, 1), dtype=jnp.float32),
    )

    np.testing.assert_allclose(baseline, [1.0])
    np.testing.assert_allclose(perturbed, baseline)


@pytest.mark.parametrize("solver", ("euler", "fireflow"))
@pytest.mark.parametrize("action_dim", (20, 32))
def test_bimanual_decode_projects_every_velocity_input_and_restores_vla_tail(
    solver: str,
    action_dim: int,
) -> None:
    decode_bimanual_actions = getattr(
        model_module, "decode_bimanual_actions", None
    )
    assert callable(decode_bimanual_actions)
    x_base = jnp.zeros((1, 2, action_dim), dtype=jnp.float32)
    frozen_vla = jnp.zeros_like(x_base)
    if action_dim == 32:
        x_base = x_base.at[..., 20:].set(5.0)
        frozen_vla = frozen_vla.at[..., 20:].set(0.25)

    decoded = decode_bimanual_actions(
        _TailCoupledVelocity(),
        x_base,
        jnp.zeros((1, 1, 1), dtype=jnp.float32),
        frozen_endpoint=frozen_vla,
        num_steps=2,
        solver=solver,
    )

    np.testing.assert_allclose(decoded[..., :20], 0.0)
    if action_dim == 20:
        legacy = decode_actions(
            _TailCoupledVelocity(),
            x_base,
            jnp.zeros((1, 1, 1), dtype=jnp.float32),
            num_steps=2,
            solver=solver,
        )
        np.testing.assert_allclose(decoded, legacy)
    else:
        np.testing.assert_allclose(decoded[..., 20:], 0.25)


def test_bimanual_aux_decode_is_invariant_to_32d_base_tail() -> None:
    model = _TailCoupledVelocity()
    gt = jnp.zeros((1, 2, 32), dtype=jnp.float32)
    frozen_vla = gt.at[..., 20:].set(0.25)

    def decode_component(x_base):
        return bimanual_loss_components_per_sample(
            model,
            x_base,
            gt,
            frozen_vla,
            jnp.asarray([0.5], dtype=jnp.float32),
            jnp.zeros((1, 1, 1), dtype=jnp.float32),
            jnp.ones((1, 2), dtype=jnp.float32),
            aux_decode_weight=1.0,
            aux_decode_steps=2,
            low_gate_safety_weight=0.0,
            rank_weight=0.0,
            repair_weight=0.0,
        )["decode"]

    baseline = decode_component(jnp.zeros_like(gt))
    perturbed = decode_component(jnp.zeros_like(gt).at[..., 20:].set(5.0))

    np.testing.assert_allclose(baseline, [0.0])
    np.testing.assert_allclose(perturbed, baseline)


def test_masked_flow_matching_ignores_32d_tail_residual():
    tail_only = jnp.concatenate([jnp.zeros((20,)), jnp.full((12,), 100.0)])
    loss = masked_flow_matching_loss_per_sample(
        _ConstantVelocity(tail_only),
        jnp.zeros((1, 2, 32)),
        jnp.zeros((1, 2, 32)),
        jnp.asarray([0.5]),
        jnp.zeros((1, 1, 1)),
    )
    np.testing.assert_allclose(loss, [0.0])


def test_bimanual_composite_flow_gives_low_and_high_wrists_independent_gradients():
    x_base = jnp.zeros((1, 1, 32))
    gt = jnp.ones_like(x_base)
    vla = -jnp.ones_like(x_base)
    gate = jnp.asarray([[1.0, 0.0]])

    def objective(velocity):
        components = bimanual_loss_components_per_sample(
            _ConstantVelocity(velocity),
            x_base,
            gt,
            vla,
            jnp.asarray([0.5]),
            jnp.zeros((1, 1, 1)),
            gate,
            aux_decode_weight=0.0,
            low_gate_safety_weight=0.0,
            rank_weight=0.0,
            repair_weight=0.0,
        )
        return jnp.mean(components["composite_fm"])

    gradient = jax.grad(objective)(jnp.zeros((32,)))
    assert bool(jnp.all(gradient[:10] < 0.0))
    assert bool(jnp.all(gradient[10:20] > 0.0))
    np.testing.assert_allclose(gradient[20:], 0.0)


def test_bimanual_single_active_wrist_rank_is_not_diluted():
    gt = jnp.zeros((2, 1, 20))
    vla = jnp.full_like(gt, 2.0)
    decoded = vla.at[..., :10].set(2.5)
    with mock.patch(
        "train_pi05_frs.utils.model.decode_bimanual_actions", return_value=decoded
    ) as decode:
        components = bimanual_loss_components_per_sample(
            _ConstantVelocity(jnp.zeros((20,))),
            jnp.zeros_like(gt),
            gt,
            vla,
            jnp.full((2,), 0.5),
            jnp.zeros((2, 1, 1)),
            jnp.asarray([[1.0, 0.5], [1.0, 0.5]]),
            source_indices=jnp.asarray([0, 0]),
            aux_decode_weight=0.0,
            low_gate_safety_weight=0.0,
            rank_weight=1.0,
            rank_margin=0.1,
            repair_weight=0.0,
        )
    np.testing.assert_allclose(components["rank"], jnp.full((2,), 6.1), atol=1e-6)
    decode.assert_called_once()


def test_bimanual_source_normalization_ignores_appended_inactive_source():
    def rank_mean(
        decoded_values: jax.Array, gates: jax.Array, sources: jax.Array
    ) -> jax.Array:
        gt = jnp.zeros((decoded_values.shape[0], 1, 20))
        vla = jnp.full_like(gt, 2.0)
        decoded = vla.at[..., :10].set(decoded_values[:, None, None])
        with mock.patch(
            "train_pi05_frs.utils.model.decode_bimanual_actions", return_value=decoded
        ):
            components = bimanual_loss_components_per_sample(
                _ConstantVelocity(jnp.zeros((20,))),
                jnp.zeros_like(gt),
                gt,
                vla,
                jnp.full((decoded_values.shape[0],), 0.5),
                jnp.zeros((decoded_values.shape[0], 1, 1)),
                gates,
                source_indices=sources,
                aux_decode_weight=0.0,
                low_gate_safety_weight=0.0,
                rank_weight=1.0,
                repair_weight=0.0,
            )
        return jnp.mean(components["rank"])

    active_values = jnp.asarray([1.5, 2.5, 2.0])
    active_gates = jnp.asarray([[1.0, 0.5], [1.0, 0.5], [1.0, 0.5]])
    active_sources = jnp.asarray([0, 0, 1])
    baseline = rank_mean(active_values, active_gates, active_sources)
    with_inactive_source = rank_mean(
        jnp.concatenate([active_values, jnp.asarray([2.0])]),
        jnp.concatenate([active_gates, jnp.asarray([[0.5, 0.5]])]),
        jnp.concatenate([active_sources, jnp.asarray([2])]),
    )

    np.testing.assert_allclose(baseline, 4.0, atol=1e-6)
    np.testing.assert_allclose(with_inactive_source, baseline, atol=1e-6)


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

    def test_direct_decoder_rejects_implicit_resume_symlink_to_cache(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            output = root / "output"
            cache_root = root / "cache"
            output.mkdir()
            cache_root.mkdir()
            (output / "last").symlink_to(cache_root, target_is_directory=True)

            with self.assertRaisesRegex(ValueError, "implicit resume.*same output"):
                _validate_training_path_boundaries(
                    cache_dir=cache_root,
                    cache_dirs=None,
                    tactile_encoder_dir=root / "encoder",
                    output_dir=output,
                    dataset_root=root / "dataset",
                    dataset_sources=None,
                    tactile_embedding_cache_root=None,
                    resume=True,
                    resume_from=None,
                )

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

    def test_public_bimanual_train_step_rejects_nonfinite_gate_before_jit(self):
        model = TactileConditionedFlowDecoder(
            DecoderConfig(
                action_dim=20,
                action_horizon=2,
                tactile_window=3,
                gru_hidden_dim=8,
                resnet_embedding_dim=4,
                model_dim=8,
                depth=1,
                num_heads=2,
            ),
            rngs=nnx.Rngs(220),
        )
        optimizer = make_optimizer(model, learning_rate=1e-3, weight_decay=0.0)
        zeros = jnp.zeros((1, 2, 20), dtype=jnp.float32)
        with mock.patch(
            "train_pi05_frs.utils.model._train_step_jit", create=True
        ) as compiled:
            with self.assertRaisesRegex(ValueError, "finite"):
                train_step(
                    model,
                    optimizer,
                    zeros,
                    zeros,
                    zeros,
                    jnp.zeros((1, 3, 4, 4), dtype=jnp.float32),
                    jnp.asarray([[jnp.nan, 0.5]], dtype=jnp.float32),
                    jax.random.key(221),
                    source_indices=jnp.asarray([0], dtype=jnp.int32),
                    loss_mode="bimanual_gated",
                    aux_decode_weight=0.0,
                )
        compiled.assert_not_called()

    def test_train_step_bimanual_gated_uses_composite_component(self):
        model = TactileConditionedFlowDecoder(
            DecoderConfig(
                action_dim=20,
                action_horizon=2,
                tactile_window=3,
                gru_hidden_dim=8,
                resnet_embedding_dim=4,
                model_dim=8,
                depth=1,
                num_heads=2,
            ),
            rngs=nnx.Rngs(90),
        )
        optimizer = make_optimizer(model, learning_rate=1e-3, weight_decay=0.0)
        x_base = jax.random.normal(jax.random.key(91), (2, 2, 20))
        gt = x_base + 1.0
        predicted = x_base - 1.0
        tactile = self._tactile_seq(jax.random.key(92), 2)

        loss, components = train_step(
            model,
            optimizer,
            x_base,
            gt,
            predicted,
            tactile,
            jnp.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=jnp.float32),
            jax.random.key(93),
            source_indices=jnp.asarray([0, 0], dtype=jnp.int32),
            loss_mode="bimanual_gated",
            aux_decode_weight=1.0,
            aux_decode_steps=1,
        )

        self.assertEqual(
            set(components),
            {
                "gt_fm",
                "vla_fm",
                "composite_fm",
                "low_safety",
                "decode",
                "rank",
                "repair",
            },
        )
        self.assertTrue(bool(jnp.isfinite(loss)))
        self.assertGreater(float(components["composite_fm"]), 0.0)
        self.assertEqual(float(components["gt_fm"]), 0.0)
        self.assertEqual(float(components["vla_fm"]), 0.0)

    def test_scalar_gated_train_step_keeps_existing_objective_value(self):
        model = self.make_model()
        optimizer = make_optimizer(model, learning_rate=1e-3, weight_decay=0.0)
        x_base = jax.random.normal(jax.random.key(50), (2, 6, 3))
        gt = x_base + 1.0
        predicted = x_base - 0.25
        tactile = self._tactile_seq(jax.random.key(51), 2)
        gate = jnp.full((2,), 0.5, dtype=jnp.float32)
        key = jax.random.key(52)
        time_key, _ = jax.random.split(key)
        t = jax.random.uniform(time_key, (2,), minval=0.0, maxval=1.0)
        expected = float(
            jnp.mean(
                gated_flow_matching_loss_per_sample(
                    model,
                    x_base,
                    gt,
                    predicted,
                    t,
                    tactile,
                    gate,
                    gate_lambda=2.0,
                    aux_decode_weight=0.0,
                )
            )
        )

        loss, components = train_step(
            model,
            optimizer,
            x_base,
            gt,
            predicted,
            tactile,
            gate,
            key,
            loss_mode="gated",
            gate_lambda=2.0,
            aux_decode_weight=0.0,
        )

        np.testing.assert_allclose(loss, expected, rtol=1e-3, atol=1e-5)
        self.assertEqual(
            set(components),
            {"gt_fm", "vla_fm", "low_safety", "decode", "rank", "repair"},
        )

    def test_scalar_composite_gated_uses_one_mixed_flow_endpoint(self):
        model = self.make_model()
        optimizer = make_optimizer(model, learning_rate=1e-3, weight_decay=0.0)
        x_base = jax.random.normal(jax.random.key(53), (3, 6, 3))
        gt = x_base + 1.0
        predicted = x_base - 0.25
        tactile = self._tactile_seq(jax.random.key(54), 3)
        gate = jnp.asarray([0.2, 0.5, 0.8], dtype=jnp.float32)
        key = jax.random.key(55)
        time_key, _ = jax.random.split(key)
        t = jax.random.uniform(time_key, (3,), minval=0.0, maxval=1.0)
        target, _ = composite_endpoint(gt, predicted, gate)
        expected = jnp.mean(
            flow_matching_loss_per_sample(model, x_base, target, t, tactile)
        )

        loss, components = train_step(
            model,
            optimizer,
            x_base,
            gt,
            predicted,
            tactile,
            gate,
            key,
            loss_mode="composite_gated",
            gate_lambda=0.0,
            aux_decode_weight=0.0,
        )

        np.testing.assert_allclose(loss, expected, rtol=1e-3, atol=1e-5)
        self.assertEqual(
            set(components),
            {
                "gt_fm",
                "vla_fm",
                "composite_fm",
                "low_safety",
                "decode",
                "rank",
                "repair",
            },
        )
        self.assertEqual(float(components["gt_fm"]), 0.0)
        self.assertEqual(float(components["vla_fm"]), 0.0)

    def test_decoder_accepts_two_tactile_tokens(self):
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
                num_tactile_tokens=2,
            ),
            rngs=nnx.Rngs(56),
        )
        tactile = jax.random.normal(jax.random.key(57), (2, 3, 2, 4))

        tokens = model.encode_tactile_tokens(tactile)

        self.assertEqual(tokens.shape, (2, 2, 16))

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
            set(components),
            {"gt_fm", "vla_fm", "low_safety", "decode", "rank", "repair"},
        )
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
        self.assertTrue(bool(jnp.isfinite(pred_step_loss)))
        self.assertEqual(
            set(pred_components),
            {"gt_fm", "vla_fm", "low_safety", "decode", "rank", "repair"},
        )

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
