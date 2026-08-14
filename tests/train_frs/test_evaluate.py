from __future__ import annotations

import json

import jax.numpy as jnp
import numpy as np
from flax import nnx

import train_frs.evaluate as evaluate_module
from train_frs.utils.model import DecoderConfig, TactileConditionedFlowDecoder


def test_gated_checkpoint_evaluation_reports_gate_labels_without_decoder_gate_input(tmp_path, monkeypatch):
    model = TactileConditionedFlowDecoder(
        DecoderConfig(
            action_dim=1,
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
            "action_dim": 1,
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
            assert kwargs["build_episode_baselines"] is True
            self.episode_baselines = {0: np.zeros((1, 4), dtype=np.float32)}

        def batches(self, split, *, batch_size, shuffle, seed):
            del batch_size, shuffle, seed
            assert split == "val"
            yield (
                np.asarray([0, 1], dtype=np.int64),
                np.zeros((2, 2, 1), dtype=np.float32),
                np.zeros((2, 2, 1), dtype=np.float32),
                np.ones((2, 2, 1), dtype=np.float32),
                np.zeros((2, 0), dtype=np.float32),
                jnp.ones((2, 2, 1, 4), dtype=jnp.float32),
            )

        def tactile_change_for_cache_indices(self, indices, current_tokens):
            del indices, current_tokens
            return np.asarray([0.1, 0.9], dtype=np.float32)

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
                    "loss_mode": "gated",
                    "gate_tau": 0.5,
                    "gate_temperature": 0.1,
                },
            },
        ),
    )

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
        write_plots=False,
        num_trajectory_samples=0,
        num_episode_strips=0,
        num_workers=0,
        prefetch_batches=1,
        load_threads=1,
        pipeline_prefetch=1,
        image_cache_size=0,
    )

    assert metrics["n_high_w"] == 1
    assert metrics["n_low_w"] == 1
    assert "gate_w_mean" in metrics
    written_metrics = json.loads((tmp_path / "output" / "metrics.json").read_text())
    assert written_metrics["n_high_w"] == 1
    assert written_metrics["n_low_w"] == 1
