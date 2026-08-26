"""Synthetic CPU coverage for direct tactile decoder training and evaluation."""

from __future__ import annotations

import os
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from train_baseline_pi05.action_cache import ActionCache, ActionCacheWriter, SampleRecord
from train_baseline_pi05.config import TACTILE_KEYS, load_config
from train_baseline_pi05.tactile_cache import TactileEmbeddingCache, _identity


def _manifest() -> dict[str, object]:
    return {
        "dataset_identity": {"repo_id": "synthetic", "revision": "one"},
        "split": {"seed": 0}, "source_checkpoint": "frozen/pi05",
        "source_variant": {"name": "pi05"}, "norm_stats": {"asset_id": "synthetic"},
        "sample_steps": 10, "noise_seed": 0, "source_model_action_width": 20,
        "decoder_action_width": 20, "action_space": "normalized_pi05",
    }


def _caches(tmp_path: Path, *, indices: tuple[int, ...] = (1, 4, 7, 9), tactile_keys=TACTILE_KEYS):
    action_dir, tactile_dir, encoder = tmp_path / "action", tmp_path / "tactile", tmp_path / "encoder"
    encoder.mkdir(); (encoder / "weights").write_bytes(b"frozen")
    writer = ActionCacheWriter.create(action_dir, sample_count=len(indices), horizon=50, action_dim=20, manifest=_manifest())
    records = [SampleRecord(index, i // 2, index, 0 if i < 2 else 1) for i, index in enumerate(indices)]
    coarse = np.zeros((len(indices), 50, 20), dtype=np.float32)
    expert = np.ones((len(indices), 50, 20), dtype=np.float32)
    writer.write_batch(0, coarse=coarse, expert=expert, valid=np.ones((len(indices), 50), dtype=np.bool_), records=records)
    writer.finalize()
    tactile_dir.mkdir()
    embeddings = np.stack([np.full((4, 512), frame, dtype=np.float32) for frame in range(10)])
    np.save(tactile_dir / "embeddings.npy", embeddings)
    (tactile_dir / "manifest.json").write_text(json.dumps({
        "status": "complete", "total_frames": 10, "tactile_keys": list(tactile_keys),
        "embedding_dim": 512, "preprocess_version": "resize_with_pad_uint8_to_unit_v1",
        "encoder_identity": _identity(encoder), "dataset_identity": {"repo_id": "synthetic", "revision": "one"},
    }), encoding="utf-8")
    return ActionCache.open(action_dir), TactileEmbeddingCache.open(tactile_dir, encoder_path=encoder)


def test_dataset_aligns_absolute_action_dataset_indices_to_tactile_memmap_and_split(tmp_path: Path):
    from train_baseline_pi05.data import BaselineCacheDataset

    action, tactile = _caches(tmp_path)
    dataset = BaselineCacheDataset(action, tactile, "train")

    first, second = dataset[0], dataset[1]
    assert len(dataset) == 2
    assert first["dataset_index"].item() == 1
    assert second["dataset_index"].item() == 4
    assert first["tactile"].shape == (4, 512)
    assert torch.equal(first["tactile"], torch.full((4, 512), 1.0))
    assert first["coarse"].shape == first["target"].shape == (50, 20)
    assert first["valid"].shape == (50,)
    assert first["coarse"].device.type == first["tactile"].device.type == "cpu"


@pytest.mark.parametrize("case", ["out_of_range", "provenance", "key_order"])
def test_dataset_rejects_invalid_alignment_contract(tmp_path: Path, case: str):
    from train_baseline_pi05.data import BaselineCacheDataset

    if case == "out_of_range":
        action, tactile = _caches(tmp_path, indices=(1, 4, 7, 11))
    elif case == "key_order":
        action, tactile = _caches(tmp_path)
        tactile.metadata["tactile_keys"] = list(reversed(TACTILE_KEYS))
    else:
        action, tactile = _caches(tmp_path)
        tactile.metadata["dataset_identity"] = {"repo_id": "other", "revision": "one"}
    with pytest.raises((ValueError, IndexError), match="(tactile|dataset|key|range|align)"):
        BaselineCacheDataset(action, tactile, "train")


def test_seeded_train_loader_shuffles_reproducibly_while_evaluation_loaders_are_ordered(tmp_path: Path):
    from train_baseline_pi05.data import BaselineCacheDataset, make_loader

    action, tactile = _caches(tmp_path)
    train = BaselineCacheDataset(action, tactile, "train")
    validation = BaselineCacheDataset(action, tactile, "validation")
    first = [row.item() for batch in make_loader(train, batch_size=1, shuffle=True, seed=9) for row in batch["dataset_index"]]
    second = [row.item() for batch in make_loader(train, batch_size=1, shuffle=True, seed=9) for row in batch["dataset_index"]]
    ordered = [row.item() for batch in make_loader(validation, batch_size=1, shuffle=False, seed=9) for row in batch["dataset_index"]]
    assert first == second
    assert ordered == [7, 9]


def test_evaluation_masks_metrics_inverse_quantiles_and_deterministic_episode_shuffle(tmp_path: Path):
    from train_baseline_pi05.data import BaselineCacheDataset, make_loader
    from train_baseline_pi05.evaluate import evaluate_decoder

    action, tactile = _caches(tmp_path)
    dataset = BaselineCacheDataset(action, tactile, "validation")
    loader = make_loader(dataset, batch_size=2, shuffle=False, seed=0)
    class Zero(torch.nn.Module):
        def forward(self, coarse, tactile): return torch.zeros_like(coarse)
    metrics = evaluate_decoder(Zero(), loader, {"q01": np.zeros(20), "q99": np.full(20, 2.0)}, shuffle_tactile=True)
    assert metrics["decoder_smooth_l1"] == pytest.approx(0.5)
    assert metrics["coarse_smooth_l1"] == pytest.approx(0.5)
    assert metrics["decoder_mse"] == pytest.approx(1.0)
    assert metrics["coarse_mse"] == pytest.approx(1.0)
    assert metrics["relative_mse_reduction"] == pytest.approx(0.0)
    assert metrics["physical_mae"] == pytest.approx(1.0)
    assert metrics["normalized_gripper_mae_9"] == pytest.approx(1.0)
    assert metrics["normalized_gripper_mae_19"] == pytest.approx(1.0)
    assert metrics["shuffled_decoder_mse"] == pytest.approx(metrics["decoder_mse"])


def test_training_and_resume_only_update_decoder_and_preserve_source_inputs(tmp_path: Path, monkeypatch):
    from train_baseline_pi05.train import train_decoder

    action, tactile = _caches(tmp_path)
    source = tmp_path / "source.bin"; source.write_bytes(b"source")
    encoder = tmp_path / "encoder.bin"; encoder.write_bytes(b"encoder")
    config = SimpleNamespace(
        cache=SimpleNamespace(action_root=action.cache_dir, tactile_root=tactile.cache_dir),
        source=SimpleNamespace(checkpoint=source, norm_stats_dir=tmp_path, norm_stats_asset_id="synthetic", seed=0, sample_steps=10, paligemma_variant="p", action_expert_variant="a", use_quantile_norm=True),
        tactile=SimpleNamespace(encoder_checkpoint=encoder),
        decoder=SimpleNamespace(output=tmp_path / "output", batch_size=2, epochs=2, learning_rate=1e-3, weight_decay=0.0, seed=3, action_horizon=50, action_dim=20, tactile_dim=512, d_model=128, nhead=4, num_layers=2, dim_feedforward=256, dropout=0.1, tactile_keys=TACTILE_KEYS, workers=0, pin_memory=False, device="cpu", resume=False),
        dataset=SimpleNamespace(),
        action_cache=action,
        tactile_cache=tactile,
        norm_stats={"q01": np.zeros(20), "q99": np.ones(20)},
    )
    before = {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in (source, encoder, action.cache_dir / "coarse_actions.npy", tactile.cache_dir / "embeddings.npy")}
    import train_baseline_pi05.train as training
    last_epochs = []
    original_save_last = training.save_last_checkpoint
    monkeypatch.setattr(training, "save_last_checkpoint", lambda *args, **kwargs: last_epochs.append(kwargs["epoch"]) or original_save_last(*args, **kwargs))

    result = train_decoder(config, max_steps=1)
    assert result == config.decoder.output / "last.pt"
    assert (config.decoder.output / "best.pt").exists()
    assert torch.load(result, weights_only=True)["global_step"] == 1
    assert before == {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in before}
    config.decoder.resume = True
    resumed = train_decoder(config, max_steps=2)
    assert torch.load(resumed, weights_only=True)["global_step"] == 2
    resumed_payload = torch.load(resumed, weights_only=True)
    config.decoder.resume = False
    config.decoder.output = tmp_path / "uninterrupted"
    uninterrupted = torch.load(train_decoder(config, max_steps=2), weights_only=True)
    assert resumed_payload["decoder_state"].keys() == uninterrupted["decoder_state"].keys()

    assert all(torch.equal(resumed_payload["decoder_state"][key], uninterrupted["decoder_state"][key]) for key in resumed_payload["decoder_state"])
    assert 1 in last_epochs and 2 in last_epochs
    config.decoder.output = tmp_path / "output"
    config.decoder.resume = True
    os.utime(action.cache_dir / "coarse_actions.npy", None)
    with pytest.raises(ValueError, match="source contract"):
        train_decoder(config, max_steps=2)

def test_training_config_exposes_loader_resume_and_evaluation_fields():
    config = load_config(Path(__file__).resolve().parents[2] / "train_baseline_pi05/configs/train_baseline_pi05.yaml")

    assert config.decoder.workers == 0
    assert config.decoder.pin_memory is False
    assert config.decoder.device == "cpu"
    assert config.decoder.resume is False
    assert config.evaluation.split == "test"
    assert config.evaluation.batch_size == config.decoder.batch_size
    assert config.evaluation.shuffle_tactile is True


def test_episode_shuffle_uses_complete_cross_episode_permutation_when_available():
    from train_baseline_pi05.evaluate import _episode_shuffle

    episodes = torch.tensor([0, 0, 0, 1, 1, 1])
    tactile = torch.arange(6, dtype=torch.float32).reshape(6, 1, 1)
    shuffled = _episode_shuffle(tactile, episodes)

    assert torch.all(shuffled[:, 0, 0] != tactile[:, 0, 0])
    assert torch.all(episodes[shuffled[:, 0, 0].to(torch.int64)] != episodes)


def test_evaluate_cli_opens_configured_caches_and_computes_current_metrics(monkeypatch, tmp_path: Path):
    from train_baseline_pi05 import evaluate

    config = SimpleNamespace(
        cache=SimpleNamespace(action_root=tmp_path / "action", tactile_root=tmp_path / "tactile"),
        tactile=SimpleNamespace(encoder_checkpoint=tmp_path / "encoder"),
        decoder=SimpleNamespace(batch_size=3, device="cpu", seed=0, workers=0, pin_memory=False),
        source=SimpleNamespace(norm_stats_dir=tmp_path, norm_stats_asset_id="stats"),
        evaluation=SimpleNamespace(split="test", batch_size=2, shuffle_tactile=True, output=None),
    )
    observed = {}
    monkeypatch.setattr(evaluate, "load_config", lambda path: config, raising=False)
    monkeypatch.setattr(evaluate.ActionCache, "open", lambda path: "action")
    monkeypatch.setattr(evaluate.TactileEmbeddingCache, "open", lambda *args, **kwargs: "tactile", raising=False)
    monkeypatch.setattr(evaluate, "BaselineCacheDataset", lambda action, tactile, split: observed.setdefault("split", split) or "dataset", raising=False)
    monkeypatch.setattr(evaluate, "make_loader", lambda dataset, **kwargs: observed.setdefault("loader", kwargs) or "loader", raising=False)
    monkeypatch.setattr(evaluate, "load_decoder_checkpoint", lambda path, **kwargs: (torch.nn.Identity(), {}), raising=False)
    monkeypatch.setattr(evaluate, "_load_norm_stats", lambda cfg: {"q01": np.zeros(20), "q99": np.ones(20)}, raising=False)
    monkeypatch.setattr(evaluate, "evaluate_decoder", lambda *args, **kwargs: observed.setdefault("shuffle", kwargs["shuffle_tactile"]) or {"decoder_mse": 1.0})
    monkeypatch.setattr(evaluate, "write_metrics", lambda metrics, path: observed.setdefault("output", path))
    monkeypatch.setattr(sys, "argv", ["evaluate", "--config", "demo.yaml", "--checkpoint", "decoder.pt", "--split", "validation", "--output", "metrics.json"])
    evaluate.main()

    assert observed["split"] == "validation"
    assert observed["loader"]["batch_size"] == 2
    assert observed["shuffle"] is True
    assert observed["output"] == Path("metrics.json")


def test_training_modules_do_not_import_jax_flax_or_pi_runtime():
    completed = subprocess.run([sys.executable, "-c", "import train_baseline_pi05.data, train_baseline_pi05.train, train_baseline_pi05.evaluate; import sys; assert not any(name.startswith(('jax', 'flax', 'lerobot.policies.pi05')) for name in sys.modules)"], check=False)
    assert completed.returncode == 0
