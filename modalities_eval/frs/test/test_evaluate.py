import json
import sys
import types
from pathlib import Path

import numpy as np
import pytest

from modalities_eval.frs.evaluate import evaluate_batches
from modalities_eval.frs.interventions import DEFAULT_INTERVENTIONS


def test_evaluate_batches_keeps_x_base_fixed_for_every_condition():
    seen = []
    state = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)

    def decode(x_base, tactile, received_state):
        assert received_state.shape[0] == x_base.shape[0]
        np.testing.assert_array_equal(received_state, state)
        seen.append(np.array(x_base))
        return x_base

    indices = np.array([4, 7], dtype=np.int64)
    x_base = np.arange(4, dtype=np.float32).reshape(2, 1, 2)
    vla = np.ones_like(x_base)
    gt = np.zeros_like(x_base)
    tactile = np.array(
        [
            [
                [[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0]],
                [[0.0, 1.0], [1.0, 0.0], [0.0, 1.0], [1.0, 0.0]],
            ],
            [
                [[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0]],
                [[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0]],
            ],
        ],
        dtype=np.float32,
    )
    metadata = [
        {"episode_index": 10, "dataset_index": 100},
        {"episode_index": 20, "dataset_index": 200},
    ]
    baselines = np.array(
        [
            [[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0]],
            [[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0]],
        ],
        dtype=np.float32,
    )

    rows = evaluate_batches(
        batches=[(indices, x_base, vla, gt, state, tactile, metadata)],
        baseline_fn=lambda received_indices: baselines,
        decode_fn=decode,
        tau=0.4,
        temperature=0.1,
        interventions=DEFAULT_INTERVENTIONS,
    )

    assert len(seen) == 1 + len(DEFAULT_INTERVENTIONS)
    assert all(np.array_equal(value, seen[0]) for value in seen[1:])
    assert {row["condition"] for row in rows} >= {
        "full",
        "baseline_fixed",
        "baseline_recomputed",
    }


def test_cli_only_requires_training_config():
    from modalities_eval.frs.evaluate import build_parser

    args = build_parser().parse_args(["--config", "train.yaml"])

    assert args.config.name == "train.yaml"
    assert args.checkpoint_dir is None
    assert args.output_dir == Path("eval_outputs/frs_modalities")
    assert args.allow_unverified_provenance is False

    overridden = build_parser().parse_args(
        ["--config", "train.yaml", "--allow-unverified-provenance"]
    )
    assert overridden.allow_unverified_provenance is True


def test_evaluate_batches_rejects_decode_outputs_that_require_broadcasting():
    action = np.zeros((1, 1, 1), dtype=np.float32)
    tactile = np.ones((1, 2, 4, 1), dtype=np.float32)

    with pytest.raises(ValueError, match="decode output shape"):
        evaluate_batches(
            batches=[
                (
                    np.array([1], dtype=np.int64),
                    action,
                    action,
                    action,
                    np.ones((1, 2), dtype=np.float32),
                    tactile,
                    [{"episode_index": 1, "dataset_index": 1}],
                )
            ],
            baseline_fn=lambda indices: np.ones((len(indices), 4, 1), dtype=np.float32),
            decode_fn=lambda x_base, tactile, state: np.zeros((1, 1), dtype=np.float32),
            tau=0.4,
            temperature=0.1,
            interventions=("baseline_fixed",),
        )


def test_evaluate_from_config_uses_config_validation_steps_without_loading_reporting_early(monkeypatch, tmp_path):
    from modalities_eval.frs import evaluate

    decode_steps = []
    closed = []

    class FakeContext:
        gate_tau = 0.4
        gate_temperature = 0.1
        rank_low_gate_threshold = 0.2
        rank_high_gate_threshold = 0.8
        default_num_steps = 17
        provenance = {
            "status": "configuration_only",
            "strong_content_hashes_verified": False,
            "override_used": True,
            "warning": "test warning",
        }

        def batches(self, *, split, batch_size):
            assert split == "val"
            assert batch_size == 2
            yield (
                np.array([1], dtype=np.int64),
                np.zeros((1, 1, 1), dtype=np.float32),
                np.ones((1, 1, 1), dtype=np.float32),
                np.zeros((1, 1, 1), dtype=np.float32),
                np.ones((1, 2), dtype=np.float32),
                np.ones((1, 2, 4, 1), dtype=np.float32),
                [{"episode_index": 3, "dataset_index": 9, "source": "fake"}],
            )

        def baselines(self, indices):
            return np.ones((len(indices), 4, 1), dtype=np.float32)

        def decode(self, x_base, tactile, state, *, num_steps, solver):
            assert state.shape == (len(x_base), 2)
            decode_steps.append((num_steps, solver))
            return x_base

        def close(self):
            closed.append(True)

    reported = {}

    def write_report(
        rows,
        *,
        output_dir,
        bootstrap_samples,
        bootstrap_seed,
        rank_low_gate_threshold,
        rank_high_gate_threshold,
        provenance,
    ):
        reported.update(
            rows=rows,
            output_dir=output_dir,
            bootstrap_samples=bootstrap_samples,
            bootstrap_seed=bootstrap_seed,
            rank_low_gate_threshold=rank_low_gate_threshold,
            rank_high_gate_threshold=rank_high_gate_threshold,
            provenance=provenance,
        )
        return {"row_count": len(rows)}

    monkeypatch.setattr(evaluate, "load_evaluation_context", lambda **_: FakeContext())
    reporting = types.ModuleType("modalities_eval.frs.reporting")
    reporting.write_report = write_report
    monkeypatch.setitem(sys.modules, "modalities_eval.frs.reporting", reporting)

    result = evaluate.evaluate_from_config(
        config_path=tmp_path / "train.yaml",
        batch_size=2,
        interventions=("baseline_fixed",),
        bootstrap_samples=11,
        bootstrap_seed=5,
        allow_unverified_provenance=True,
    )

    assert result == {"row_count": 2}
    assert decode_steps == [(17, "euler"), (17, "euler")]
    assert closed == [True]
    assert reported["output_dir"] == Path("eval_outputs/frs_modalities")
    assert reported["rank_low_gate_threshold"] == 0.2
    assert reported["rank_high_gate_threshold"] == 0.8
    assert reported["provenance"] == FakeContext.provenance


def test_load_evaluation_context_uses_fakes_for_cache_checkpoint_and_source_metadata(monkeypatch, tmp_path):
    from modalities_eval.frs import evaluate

    calls = {}

    class FakePairs:
        manifest = {
            "records_sha256": "cache-digest",
            "action_horizon": 2,
            "action_dim": 1,
            "configuration": {"reverse_solver": "fireflow", "source_policy": "vla-a"},
        }
        source_names = ("source/a",)

        def __init__(self, cache_dirs, *, source_names):
            calls["cache_dirs"] = cache_dirs
            calls["source_names"] = source_names

        def source_and_local_indices(self, indices):
            return np.zeros(len(indices), dtype=np.int32), np.asarray(indices, dtype=np.int64) - 4

        def metadata_values(self, indices, key):
            values = {"dataset_index": 42, "episode_index": 3}
            return np.full(len(indices), values[key], dtype=np.int64)

    class FakeConditioner:
        resnet_embedding_dim = 3

        def __init__(self, pairs, **kwargs):
            calls["conditioner"] = (pairs, kwargs)
            self.episode_baselines = {(0, 3): np.ones((4, 3), dtype=np.float32)}

        def batches(self, split, *, batch_size, shuffle, seed):
            assert (split, batch_size, shuffle, seed) == ("val", 2, False, 0)
            yield (
                np.array([5], dtype=np.int64),
                np.zeros((1, 1, 1), dtype=np.float32),
                np.ones((1, 1, 1), dtype=np.float32),
                np.zeros((1, 1, 1), dtype=np.float32),
                np.ones((1, 2), dtype=np.float32),
                np.ones((1, 2, 4, 3), dtype=np.float32),
            )

        def close(self):
            calls["closed"] = True

    model = types.SimpleNamespace(
        config=types.SimpleNamespace(
            action_horizon=2,
            action_dim=1,
            gate_conditioning=False,
            num_tactile_tokens=4,
            tactile_window=2,
            resnet_embedding_dim=3,
        )
    )
    config = {
        "datasets": [{"repo_id": "source/a"}],
        "action_cache": {"root": str(tmp_path / "actions")},
        "tactile_embedding_cache": {"root": str(tmp_path / "tactile")},
        "model": {
            "tactile_encoder_path": str(tmp_path / "encoder"),
            "tactile_keys": ["a", "b", "c", "d"],
            "tactile_num_tokens": 4,
            "tactile_embedding_dim": 3,
            "tactile_image_size": 224,
        },
        "frs_training": {
            "output": str(tmp_path / "run"),
            "tactile_window_divisor": 1,
            "validation_steps": 17,
        },
    }
    train_config = types.ModuleType("train_frs.train_frs")
    train_config.load_config = lambda path: config
    train_config.source_cache_dir = lambda root, repo_id: tmp_path / "actions" / repo_id
    checkpoint_extra = {
        "cache_records_sha256": "cache-digest",
        "cache_configuration": {"reverse_solver": "fireflow", "source_policy": "vla-a"},
        "loss_mode": "gated",
        "decoder_input_version": 2,
        "gate_tau": 0.4,
        "gate_temperature": 0.1,
        "rank_low_gate_threshold": 0.2,
        "rank_high_gate_threshold": 0.8,
        "tactile_encoder_dir": str((tmp_path / "encoder").resolve()),
        "history_stride": 3,
        "tactile_window_divisor": 1,
    }
    checkpoint = types.ModuleType("train_frs.utils.checkpoint")
    checkpoint.load_checkpoint = lambda directory: (
        model,
        {"extra_metadata": checkpoint_extra},
    )
    data = types.ModuleType("train_frs.utils.data")
    data.CachedTactileEmbeddingBatches = FakeConditioner
    data.resolve_tactile_window = lambda **kwargs: kwargs["action_horizon"] // kwargs["window_divisor"]
    cache = types.ModuleType("utils.cache")
    cache.MultiCachedPairs = FakePairs
    monkeypatch.setitem(sys.modules, "train_frs.train_frs", train_config)
    monkeypatch.setitem(sys.modules, "train_frs.utils.checkpoint", checkpoint)
    monkeypatch.setitem(sys.modules, "train_frs.utils.data", data)
    monkeypatch.setitem(sys.modules, "utils.cache", cache)
    monkeypatch.setattr(
        evaluate,
        "_load_checkpoint_metadata_only",
        lambda directory: {"extra_metadata": checkpoint_extra},
    )

    with pytest.raises(ValueError, match="--allow-unverified-provenance"):
        evaluate.load_evaluation_context(config_path=tmp_path / "train.yaml")

    context = evaluate.load_evaluation_context(
        config_path=tmp_path / "train.yaml",
        allow_unverified_provenance=True,
    )
    batch = next(context.batches(split="val", batch_size=2))

    assert context.gate_tau == 0.4
    assert context.gate_temperature == 0.1
    assert context.rank_low_gate_threshold == 0.2
    assert context.rank_high_gate_threshold == 0.8
    assert context.provenance["status"] == "configuration_only"
    assert context.provenance["strong_content_hashes_verified"] is False
    assert context.provenance["override_used"] is True
    assert "array" in context.provenance["warning"]
    assert context.default_num_steps == 17
    assert calls["source_names"] == ["source/a"]
    assert calls["conditioner"][1]["history_stride"] == 3
    np.testing.assert_array_equal(batch[4], np.ones((1, 2), dtype=np.float32))
    assert batch[-1] == [
        {
            "cache_index": 5,
            "source": "source/a",
            "source_index": 0,
            "source_cache_index": 1,
            "dataset_index": 42,
            "episode_index": 3,
        }
    ]
    np.testing.assert_array_equal(context.baselines(np.array([5])), np.ones((1, 4, 3), dtype=np.float32))

    for metadata_key, mismatched_value in (
        ("cache_configuration", {"reverse_solver": "euler"}),
        ("tactile_encoder_dir", str((tmp_path / "other-encoder").resolve())),
        ("history_stride", 1),
        ("tactile_window_divisor", 2),
    ):
        checkpoint_extra[metadata_key] = mismatched_value
        with pytest.raises(ValueError, match="checkpoint.*(configuration|tactile_encoder_dir|history_stride|tactile_window_divisor)"):
            evaluate.load_evaluation_context(
                config_path=tmp_path / "train.yaml",
                allow_unverified_provenance=True,
            )
        checkpoint_extra[metadata_key] = {
            "cache_configuration": {"reverse_solver": "fireflow", "source_policy": "vla-a"},
            "tactile_encoder_dir": str((tmp_path / "encoder").resolve()),
            "history_stride": 3,
            "tactile_window_divisor": 1,
        }[metadata_key]

    checkpoint_extra["rank_low_gate_threshold"] = 0.8
    with pytest.raises(ValueError, match="0 <= low < high <= 1"):
        evaluate.load_evaluation_context(
            config_path=tmp_path / "train.yaml",
            allow_unverified_provenance=True,
        )
    checkpoint_extra["rank_low_gate_threshold"] = 0.2

    for invalid_version in (1, None, "2"):
        checkpoint_extra["decoder_input_version"] = invalid_version
        with pytest.raises(ValueError, match="decoder_input_version"):
            evaluate.load_evaluation_context(
                config_path=tmp_path / "train.yaml",
                allow_unverified_provenance=True,
            )
    checkpoint_extra["decoder_input_version"] = 2

    calls.pop("closed", None)
    config["frs_training"]["validation_steps"] = 0
    with pytest.raises(ValueError, match="validation_steps"):
        evaluate.load_evaluation_context(
            config_path=tmp_path / "train.yaml",
            allow_unverified_provenance=True,
        )
    assert calls["closed"] is True


def test_unverified_provenance_fails_before_action_cache_or_model_load(monkeypatch, tmp_path):
    from modalities_eval.frs import evaluate

    calls = {"pairs": 0, "checkpoint": 0}
    checkpoint_dir = tmp_path / "checkpoint"
    checkpoint_dir.mkdir()
    (checkpoint_dir / "checkpoint.json").write_text(
        json.dumps(
            {
                "version": 2,
                "extra_metadata": {
                    "cache_records_sha256": "records-only",
                    "cache_configuration": {"reverse_solver": "fireflow"},
                },
            }
        ),
        encoding="utf-8",
    )
    config = {
        "datasets": [{"repo_id": "source/a"}],
        "action_cache": {"root": str(tmp_path / "actions")},
        "tactile_embedding_cache": {"root": str(tmp_path / "tactile")},
        "model": {
            "tactile_encoder_path": str(tmp_path / "encoder"),
            "tactile_keys": ["a", "b", "c", "d"],
            "tactile_num_tokens": 4,
        },
        "frs_training": {"output": str(tmp_path / "unused")},
    }

    train_config = types.ModuleType("train_frs.train_frs")
    train_config.load_config = lambda path: config
    train_config.source_cache_dir = lambda root, repo_id: tmp_path / "actions" / repo_id

    checkpoint = types.ModuleType("train_frs.utils.checkpoint")

    def load_checkpoint(directory):
        calls["checkpoint"] += 1
        raise AssertionError("decoder checkpoint must not load before provenance opt-in")

    checkpoint.load_checkpoint = load_checkpoint
    data = types.ModuleType("train_frs.utils.data")
    data.CachedTactileEmbeddingBatches = object
    data.resolve_tactile_window = lambda **kwargs: 1
    cache = types.ModuleType("utils.cache")

    class FakePairs:
        def __init__(self, *args, **kwargs):
            calls["pairs"] += 1
            raise AssertionError("action caches must not load before provenance opt-in")

    cache.MultiCachedPairs = FakePairs
    monkeypatch.setitem(sys.modules, "train_frs.train_frs", train_config)
    monkeypatch.setitem(sys.modules, "train_frs.utils.checkpoint", checkpoint)
    monkeypatch.setitem(sys.modules, "train_frs.utils.data", data)
    monkeypatch.setitem(sys.modules, "utils.cache", cache)

    with pytest.raises(ValueError, match="--allow-unverified-provenance"):
        evaluate.load_evaluation_context(
            config_path=tmp_path / "train.yaml",
            checkpoint_dir=checkpoint_dir,
        )

    assert calls == {"pairs": 0, "checkpoint": 0}
