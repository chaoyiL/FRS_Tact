from __future__ import annotations

import csv
import json

import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx

import train_smolvla_frs.evaluate as evaluate_module
import train_smolvla_frs.utils.metrics as metrics_module
from train_smolvla_frs.utils.metrics import bimanual_source_decode_metrics, evaluate_split
from train_smolvla_frs.utils.model import DecoderConfig, TactileConditionedFlowDecoder


def test_bimanual_evaluation_keeps_wrist_metrics_separate(monkeypatch) -> None:
    gt_action = np.zeros((2, 1, 20), dtype=np.float32)
    predicted_action = np.ones((2, 1, 20), dtype=np.float32)
    prediction = np.ones((2, 1, 20), dtype=np.float32)
    prediction[0, :, :10] = 0.0
    prediction[1, :, 10:] = 2.0

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

        def gate_current_tokens(self, indices, tactile_input):
            del indices
            return tactile_input[:, -1]

        def tactile_change_per_wrist_for_cache_indices(self, indices, current_tokens):
            del indices, current_tokens
            return np.asarray([[1.0, 0.0], [0.0, 0.8]], dtype=np.float32)

    monkeypatch.setattr(metrics_module, "encode_tactile_embeddings", lambda model, value: value)
    flow_targets: list[np.ndarray] = []

    def fake_flow(model, x_base, target, t, tactile_input, state):
        del model, x_base, t, tactile_input, state
        target_array = np.asarray(target)
        flow_targets.append(target_array)
        return np.mean(np.square(target_array), axis=(1, 2))

    monkeypatch.setattr(
        metrics_module,
        "flow_matching_loss_per_sample",
        fake_flow,
    )
    monkeypatch.setattr(
        metrics_module,
        "decode_actions",
        lambda model, x_base, tactile_input, num_steps, solver, state: prediction,
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

    np.testing.assert_allclose(result.sample_gate_w_left, [1.0, 0.0], atol=0.01)
    np.testing.assert_allclose(
        result.sample_gate_w_right,
        [1.0 / (1.0 + np.exp(5.0)), 1.0 / (1.0 + np.exp(-3.0))],
    )
    expected_composite = np.ones_like(gt_action)
    expected_composite[0, :, :10] = 0.0
    expected_composite[1, :, 10:] = 0.0
    np.testing.assert_allclose(flow_targets[2], expected_composite)
    np.testing.assert_allclose(result.sample_composite_fm, [0.5, 0.5])
    assert result.composite_fm == pytest.approx(0.5)
    np.testing.assert_allclose(result.sample_tactile_change_left, [1.0, 0.0])
    np.testing.assert_allclose(result.sample_tactile_change_right, [0.0, 0.8])
    assert result.gate_w_left != pytest.approx(result.gate_w_right)
    assert result.gate_w_p90_left != pytest.approx(result.gate_w_p90_right)
    assert result.tactile_change_p90_left != pytest.approx(
        result.tactile_change_p90_right
    )
    np.testing.assert_allclose(result.sample_mse_gt_left, [0.0, 1.0])
    np.testing.assert_allclose(result.sample_mse_gt_right, [1.0, 4.0])
    np.testing.assert_allclose(result.sample_mse_vla_left, [1.0, 0.0])
    np.testing.assert_allclose(result.sample_mse_vla_right, [0.0, 1.0])
    assert result.gt_gain_high_w_left == pytest.approx(1.0)
    assert result.gt_gain_high_w_right == pytest.approx(-3.0)
    assert result.rank_satisfied_high_frac_left == pytest.approx(1.0)
    assert result.rank_satisfied_high_frac_right == pytest.approx(0.0)
    assert result.low_unsafe_frac_left == pytest.approx(0.0)
    assert result.low_unsafe_frac_right == pytest.approx(0.0)
    assert result.bimanual_quadrants["high_low"]["n"] == 1
    assert result.bimanual_gate_region_counts.shape == (3, 3)

    result_with_actions = evaluate_split(
        object(),  # type: ignore[arg-type]
        FakeConditioner(),  # type: ignore[arg-type]
        split="val",
        batch_size=2,
        num_steps=1,
        keep_predictions=True,
        loss_mode="bimanual_gated",
        gate_tau=0.5,
        gate_temperature=0.1,
    )

    np.testing.assert_allclose(result_with_actions.gt_actions, gt_action)
    np.testing.assert_allclose(result_with_actions.vla_actions, predicted_action)
    np.testing.assert_allclose(result_with_actions.predictions, prediction)


def test_bimanual_source_metrics_keep_a_bad_dataset_from_being_pooled_away() -> None:
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
    assert rollups["worst_dataset_repair_penalty_high_w_left"] == pytest.approx(3.0)
    assert rollups["min_dataset_repair_satisfied_high_frac_left"] == pytest.approx(0.0)
    assert rollups["worst_dataset_repair_penalty_high_w_right"] == pytest.approx(0.0)
    assert rollups["min_dataset_repair_satisfied_high_frac_right"] == pytest.approx(1.0)

    _, missing_high_rollups = bimanual_source_decode_metrics(
        sample_mse_gt_left=np.asarray([0.0, 0.0, 4.0, 0.0]),
        sample_mse_gt_right=np.asarray([0.0, 0.0, 0.0, 0.0]),
        sample_mse_vla_left=np.asarray([1.0, 0.0, 1.0, 0.0]),
        sample_mse_vla_right=np.asarray([0.0, 1.0, 0.0, 1.0]),
        sample_mse_vla_gt_left=np.ones((4,)),
        sample_mse_vla_gt_right=np.ones((4,)),
        sample_gate_w_left=np.asarray([1.0, 0.0, 0.0, 0.0]),
        sample_gate_w_right=np.asarray([0.0, 1.0, 0.0, 1.0]),
        source_indices=np.asarray([0, 0, 1, 1]),
        num_sources=2,
        low_w_threshold=0.3,
        high_w_threshold=0.7,
        ranking_margin=0.0,
        repair_margin=0.0,
        low_safety_margin=0.0,
    )
    assert np.isnan(missing_high_rollups["worst_dataset_repair_penalty_high_w_left"])
    assert np.isnan(missing_high_rollups["min_dataset_repair_satisfied_high_frac_left"])
    assert missing_high_rollups["worst_dataset_repair_penalty_high_w_right"] == pytest.approx(0.0)


@pytest.mark.parametrize(
    ("loss_mode", "gate_metadata", "build_episode_baselines", "gate_kind"),
    [
        ("gated", {"gate_tau": 0.5, "gate_temperature": 0.1}, True, "scalar"),
        ("bimanual_gated", {"gate_tau": 0.5, "gate_temperature": 0.1}, True, "bimanual"),
        ("gt", {}, False, "none"),
    ],
)
def test_checkpoint_evaluation_tracks_gate_only_for_gated_loss_mode(
    tmp_path,
    monkeypatch,
    loss_mode,
    gate_metadata,
    build_episode_baselines,
    gate_kind,
):
    action_dim = 20 if gate_kind == "bimanual" else 1
    model = TactileConditionedFlowDecoder(
        DecoderConfig(
            action_dim=action_dim,
            action_horizon=2,
            tactile_window=2,
            gru_hidden_dim=4,
            resnet_embedding_dim=4,
            model_dim=4,
            depth=1,
            num_heads=1,
            num_tactile_tokens=1,
        ),
        rngs=nnx.Rngs(0),
    )

    class FakePairs:
        manifest = {
            "action_horizon": 2,
            "action_dim": action_dim,
            "state_dim": 0,
            "records_sha256": "test-digest",
        }
        arrays = {
            "dataset_index": np.asarray([0, 1], dtype=np.int64),
            "episode_index": np.asarray([0, 0], dtype=np.int64),
        }

        def __init__(self, cache_dir):
            del cache_dir

    class FakeConditioner:
        resnet_embedding_dim = 4

        def __init__(self, pairs, **kwargs):
            del pairs
            assert kwargs["build_episode_baselines"] is build_episode_baselines
            self.episode_baselines = (
                {0: np.zeros((1, 4), dtype=np.float32)} if build_episode_baselines else {}
            )

        def batches(self, split, *, batch_size, shuffle, seed):
            del batch_size, shuffle, seed
            assert split == "val"
            yield (
                np.asarray([0, 1], dtype=np.int64),
                np.zeros((2, 2, action_dim), dtype=np.float32),
                np.zeros((2, 2, action_dim), dtype=np.float32),
                np.ones((2, 2, action_dim), dtype=np.float32),
                np.zeros((2, 0), dtype=np.float32),
                jnp.ones((2, 2, 1, 4), dtype=jnp.float32),
            )

        def tactile_change_for_cache_indices(self, indices, current_tokens):
            del indices, current_tokens
            return np.asarray([0.1, 0.9], dtype=np.float32)

        def tactile_change_per_wrist_for_cache_indices(self, indices, current_tokens):
            del indices, current_tokens
            return np.asarray([[0.1, 0.9], [0.7, 0.2]], dtype=np.float32)

        def close(self):
            return None

    monkeypatch.setattr(evaluate_module, "CachedPairs", FakePairs)
    monkeypatch.setattr(evaluate_module, "TactileConditionedBatches", FakeConditioner)
    monkeypatch.setattr(
        evaluate_module,
        "load_checkpoint",
        lambda directory: (
            model,
            {
                "epoch": 1,
                "extra_metadata": {
                    "cache_records_sha256": "test-digest",
                    "loss_mode": loss_mode,
                    **gate_metadata,
                },
            },
        ),
    )
    decode_calls = 0
    real_decode_actions = metrics_module.decode_actions

    def counted_decode_actions(*args, **kwargs):
        nonlocal decode_calls
        decode_calls += 1
        return real_decode_actions(*args, **kwargs)

    monkeypatch.setattr(metrics_module, "decode_actions", counted_decode_actions)

    metrics = evaluate_module.evaluate_decoder(
        cache_dir=tmp_path / "cache",
        tactile_encoder_dir=tmp_path,
        checkpoint_dir=tmp_path / "checkpoint",
        output_dir=tmp_path / "output",
        dataset_repo_id="owner/data",
        dataset_root=None,
        tactile_window_divisor=None,
        history_stride=None,
        batch_size=2,
        num_steps=1,
        solver="euler",
        target=None,
        save_predictions=False,
        write_plots=gate_kind == "bimanual",
        num_trajectory_samples=0,
        num_episode_strips=0,
        num_workers=0,
        prefetch_batches=1,
        load_threads=1,
        pipeline_prefetch=1,
        image_cache_size=0,
    )

    written_metrics = json.loads((tmp_path / "output" / "metrics.json").read_text())
    assert decode_calls == 1
    if gate_kind == "scalar":
        assert metrics["n_high_w"] == 1
        assert metrics["n_low_w"] == 1
        assert "gate_w_mean" in metrics
        assert written_metrics["n_high_w"] == 1
        assert written_metrics["n_low_w"] == 1
    elif gate_kind == "bimanual":
        assert metrics["n_high_w_left"] == 1
        assert metrics["n_low_w_left"] == 1
        assert metrics["n_high_w_right"] == 1
        assert metrics["n_low_w_right"] == 1
        assert "n_high_w" not in metrics
        assert written_metrics["n_high_w_left"] == 1
        assert written_metrics["composite_fm"] >= 0.0
        assert written_metrics["gate_w_mean_left"] != pytest.approx(
            written_metrics["gate_w_mean_right"]
        )
        assert written_metrics["gate_w_p90_left"] != pytest.approx(
            written_metrics["gate_w_p90_right"]
        )
        assert "bimanual_quadrants" in written_metrics
        assert np.asarray(written_metrics["bimanual_gate_region_counts"]).shape == (3, 3)
        with (tmp_path / "output" / "per_sample.csv").open(newline="", encoding="utf-8") as file:
            rows = list(csv.DictReader(file))
        assert float(rows[0]["gate_w_left"]) == pytest.approx(1.0 / (1.0 + np.exp(4.0)))
        assert float(rows[0]["gate_w_right"]) == pytest.approx(1.0 / (1.0 + np.exp(-4.0)))
        assert float(rows[0]["tactile_change_left"]) == pytest.approx(0.1)
        assert float(rows[0]["tactile_change_right"]) == pytest.approx(0.9)
        assert float(rows[0]["composite_fm"]) >= 0.0
        assert float(rows[0]["mse_gt_left"]) >= 0.0
        assert float(rows[0]["mse_vla_right"]) >= 0.0
        assert rows[0]["mse_vla_gt_left"] != ""
        assert rows[0]["gate_region_left"] in {"low", "mid", "high"}
        assert rows[0]["gate_region_right"] in {"low", "mid", "high"}
        assert rows[0]["bimanual_quadrant"] in {"", "low_low", "high_low", "low_high", "high_high"}
        assert rows[0]["gate_region"] == ""
        assert rows[0]["gate_bin"] == ""
        for filename in ("gate_diagnostics.png", "bimanual_action_examples.png"):
            artifact = tmp_path / "output" / filename
            assert artifact.is_file() and artifact.stat().st_size > 0
        assert not (tmp_path / "output" / "predictions.npz").exists()
    else:
        assert "n_high_w" not in metrics
        assert "n_low_w" not in metrics
        assert "gate_w_mean" not in metrics
        assert "n_high_w" not in written_metrics
        assert "n_low_w" not in written_metrics
