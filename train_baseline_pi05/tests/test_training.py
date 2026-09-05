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


def _manifest(dataset_root: Path) -> dict[str, object]:
    return {
        "dataset_identity": {
            "repo_id": "synthetic",
            "root": str(dataset_root.resolve()),
            "revision": "one",
        },
        "split": {"seed": 0}, "source_checkpoint": "frozen/pi05",
        "source_variant": {"name": "pi05"}, "norm_stats": {"asset_id": "synthetic"},
        "sample_steps": 10, "noise_seed": 0, "source_model_action_width": 20,
        "decoder_action_width": 20, "action_space": "normalized_pi05",
    }


def _caches(
    tmp_path: Path,
    *,
    indices: tuple[int, ...] = (1, 4, 7, 9),
    episode_indices: tuple[int, ...] | None = None,
    tactile_keys=TACTILE_KEYS,
    action_dim: int = 20,
):
    action_dir, tactile_dir, encoder = tmp_path / "action", tmp_path / "tactile", tmp_path / "encoder"
    dataset_root = tmp_path / "dataset"
    encoder.mkdir(); (encoder / "weights").write_bytes(b"frozen")
    manifest = _manifest(dataset_root)
    manifest["source_model_action_width"] = action_dim
    manifest["decoder_action_width"] = action_dim
    writer = ActionCacheWriter.create(
        action_dir,
        sample_count=len(indices),
        horizon=50,
        action_dim=action_dim,
        manifest=manifest,
    )
    episodes = episode_indices or tuple(range(len(indices)))
    records = [SampleRecord(index, episodes[i], index, 0 if i < 2 else 1) for i, index in enumerate(indices)]
    coarse = np.zeros((len(indices), 50, action_dim), dtype=np.float32)
    expert = np.ones((len(indices), 50, action_dim), dtype=np.float32)
    writer.write_batch(0, coarse=coarse, expert=expert, valid=np.ones((len(indices), 50), dtype=np.bool_), records=records)
    writer.finalize()
    tactile_dir.mkdir()
    embeddings = np.stack([np.full((len(tactile_keys), 512), frame, dtype=np.float32) for frame in range(10)])
    np.save(tactile_dir / "embeddings.npy", embeddings)
    (tactile_dir / "manifest.json").write_text(json.dumps({
        "status": "complete", "total_frames": 10, "tactile_keys": list(tactile_keys),
        "embedding_dim": 512, "preprocess_version": "resize_with_pad_uint8_to_unit_v1",
        "encoder_identity": _identity(encoder),
        "dataset_identity": {
            "repo_id": "synthetic",
            "root": str(dataset_root.resolve()),
            "revision": "one",
        },
    }), encoding="utf-8")
    return ActionCache.open(action_dir), TactileEmbeddingCache.open(tactile_dir, tactile_keys=tactile_keys, encoder_path=encoder)


def test_training_opens_two_sensor_cache_from_config(tmp_path: Path) -> None:
    from train_baseline_pi05.train import _open_caches

    right_keys = (TACTILE_KEYS[2], TACTILE_KEYS[3])
    action, tactile = _caches(tmp_path, action_dim=10, tactile_keys=right_keys)
    config = SimpleNamespace(
        cache=SimpleNamespace(action_root=action.cache_dir, tactile_root=tactile.cache_dir),
        tactile=SimpleNamespace(encoder_checkpoint=tmp_path / "encoder"),
        decoder=SimpleNamespace(tactile_keys=right_keys),
    )

    _, opened_tactile = _open_caches(config)

    assert opened_tactile.get_many([1]).shape == (1, 2, 512)


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


def test_dataset_compares_canonical_dataset_roots(tmp_path: Path) -> None:
    from train_baseline_pi05.data import BaselineCacheDataset

    action, tactile = _caches(tmp_path)
    canonical = tmp_path / "dataset"
    action.manifest["dataset_identity"]["root"] = str(canonical / ".." / "dataset")
    BaselineCacheDataset(action, tactile, "train")

    tactile.metadata["dataset_identity"]["root"] = str(tmp_path / "different-dataset")
    with pytest.raises(ValueError, match="dataset provenance"):
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


def test_evaluation_uses_cross_episode_tactile_donors_across_batch_boundaries(tmp_path: Path) -> None:
    from train_baseline_pi05.data import BaselineCacheDataset, make_loader
    from train_baseline_pi05.evaluate import evaluate_decoder

    action, tactile = _caches(tmp_path)
    dataset = BaselineCacheDataset(action, tactile, "validation")
    loader = make_loader(dataset, batch_size=1, shuffle=False, seed=0)

    class Capture(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.tactile_frames: list[int] = []

        def forward(self, coarse, tactile):
            self.tactile_frames.extend(int(value) for value in tactile[:, 0, 0].tolist())
            return torch.zeros_like(coarse)

    model = Capture()
    evaluate_decoder(
        model,
        loader,
        {"q01": np.zeros(20), "q99": np.full(20, 2.0)},
        shuffle_tactile=True,
    )

    assert model.tactile_frames[0::2] == [7, 9]
    assert model.tactile_frames[1::2] == [9, 7]


@pytest.mark.parametrize("action_dim", (10, 20))
def test_evaluation_reports_only_present_grippers_and_masks_their_errors(action_dim: int) -> None:
    from train_baseline_pi05.evaluate import evaluate_decoder

    target = torch.ones(1, 2, action_dim)
    target[:, 1] = 1000.0
    target[:, 0, 9] = 2.0
    if action_dim == 20:
        target[:, 0, 19] = 3.0
    batch = {
        "coarse": torch.zeros_like(target),
        "target": target,
        "tactile": torch.zeros(1, 4, 512),
        "valid": torch.tensor([[True, False]]),
    }

    class Zero(torch.nn.Module):
        def forward(self, coarse, tactile):
            return torch.zeros_like(coarse)

    metrics = evaluate_decoder(
        Zero(), [batch], {"q01": np.zeros(action_dim), "q99": np.ones(action_dim)}
    )

    assert metrics["normalized_gripper_mae_9"] == pytest.approx(2.0)
    if action_dim == 20:
        assert metrics["normalized_gripper_mae_19"] == pytest.approx(3.0)
    else:
        assert "normalized_gripper_mae_19" not in metrics


def test_evaluation_rejects_single_episode_shuffled_tactile_split(tmp_path: Path) -> None:
    from train_baseline_pi05.data import BaselineCacheDataset, make_loader
    from train_baseline_pi05.evaluate import evaluate_decoder

    action, tactile = _caches(tmp_path, episode_indices=(0, 1, 2, 2))
    loader = make_loader(
        BaselineCacheDataset(action, tactile, "validation"),
        batch_size=1,
        shuffle=False,
        seed=0,
    )

    model = torch.nn.Identity()
    with pytest.raises(ValueError, match="cross-episode.*impossible"):
        evaluate_decoder(
            model,
            loader,
            {"q01": np.zeros(20), "q99": np.full(20, 2.0)},
            shuffle_tactile=True,
        )
    assert model.training is True


@pytest.mark.parametrize("action_dim", (10, 20))
@pytest.mark.parametrize("device", (
    "cpu",
    pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")),
))
def test_training_and_resume_only_update_decoder_and_preserve_source_inputs(tmp_path: Path, monkeypatch, action_dim: int, device: str):
    from train_baseline_pi05.train import train_decoder

    tactile_keys = (TACTILE_KEYS[2], TACTILE_KEYS[3]) if action_dim == 10 else TACTILE_KEYS
    action, tactile = _caches(tmp_path, action_dim=action_dim, tactile_keys=tactile_keys)
    source = tmp_path / "source.bin"; source.write_bytes(b"source")
    encoder = tmp_path / "encoder.bin"; encoder.write_bytes(b"encoder")
    config = SimpleNamespace(
        cache=SimpleNamespace(action_root=action.cache_dir, tactile_root=tactile.cache_dir),
        source=SimpleNamespace(checkpoint=source, norm_stats_dir=tmp_path, norm_stats_asset_id="synthetic", seed=0, sample_steps=10, paligemma_variant="p", action_expert_variant="a", use_quantile_norm=True),
        tactile=SimpleNamespace(encoder_checkpoint=encoder),
        decoder=SimpleNamespace(output=tmp_path / "output", batch_size=2, epochs=2, learning_rate=1e-3, weight_decay=0.0, seed=3, action_horizon=50, action_dim=action_dim, tactile_dim=512, d_model=128, nhead=4, num_layers=2, dim_feedforward=256, dropout=0.1, tactile_keys=tactile_keys, workers=0, pin_memory=False, device=device, resume=False),
        dataset=SimpleNamespace(),
        action_cache=action,
        tactile_cache=tactile,
        norm_stats={"q01": np.zeros(action_dim), "q99": np.ones(action_dim)},
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
    assert config.decoder.device == "cuda"
    assert config.decoder.resume is False
    assert config.evaluation.split == "test"
    assert config.evaluation.batch_size == config.decoder.batch_size
    assert config.evaluation.shuffle_tactile is True


@pytest.mark.parametrize("max_steps", (0, -1))
def test_train_decoder_rejects_nonpositive_max_steps_before_opening_caches(monkeypatch, max_steps: int):
    from train_baseline_pi05 import train

    monkeypatch.setattr(train, "_open_caches", lambda _config: pytest.fail("must not open caches"))
    with pytest.raises(ValueError, match="max_steps must be positive"):
        train.train_decoder(object(), max_steps=max_steps)


def test_train_cli_rejects_nonpositive_max_steps_before_loading_config(monkeypatch):
    from train_baseline_pi05 import train

    monkeypatch.setattr(sys, "argv", ["train", "--config", "missing.yaml", "--max-steps", "0"])
    with pytest.raises(ValueError, match="max_steps must be positive"):
        train.main()


def test_episode_shuffle_uses_complete_cross_episode_permutation_when_available():
    from train_baseline_pi05.evaluate import _episode_shuffle

    episodes = torch.tensor([0, 0, 0, 1, 1, 1])

    tactile = torch.arange(6, dtype=torch.float32).reshape(6, 1, 1)
    shuffled = _episode_shuffle(tactile, episodes)

    assert torch.all(shuffled[:, 0, 0] != tactile[:, 0, 0])
    assert torch.all(episodes[shuffled[:, 0, 0].to(torch.int64)] != episodes)
def test_episode_shuffle_deranges_non_cyclic_episode_order():
    from train_baseline_pi05.evaluate import _episode_shuffle

    episodes = torch.tensor([0, 0, 1, 2, 1])
    tactile = torch.arange(5, dtype=torch.float32).reshape(5, 1, 1)
    shuffled = _episode_shuffle(tactile, episodes)

    assert torch.all(episodes[shuffled[:, 0, 0].to(torch.int64)] != episodes)


@pytest.mark.parametrize("episodes", ([], [0], [0, 0], [0, 0, 0, 1]))
def test_episode_shuffle_rejects_impossible_cross_episode_permutation(episodes: list[int]) -> None:
    from train_baseline_pi05.evaluate import _episode_shuffle

    tactile = torch.arange(len(episodes), dtype=torch.float32).reshape(-1, 1, 1)
    with pytest.raises(ValueError, match="cross-episode.*impossible"):
        _episode_shuffle(tactile, torch.tensor(episodes))


def test_large_cache_fingerprint_uses_only_stat(monkeypatch, tmp_path: Path):
    from train_baseline_pi05.train import _small_file_fingerprint

    path = tmp_path / "large.npy"
    with path.open("wb") as handle:
        handle.truncate(17 * 1024 * 1024)
    monkeypatch.setattr(Path, "read_bytes", lambda self: (_ for _ in ()).throw(AssertionError("whole-file read")))

    fingerprint = _small_file_fingerprint(path)

    assert fingerprint["size"] == 17 * 1024 * 1024
    assert "sha256" not in fingerprint


def test_evaluate_cli_opens_configured_caches_and_computes_current_metrics(monkeypatch, tmp_path: Path):
    from train_baseline_pi05 import evaluate

    config = SimpleNamespace(
        cache=SimpleNamespace(action_root=tmp_path / "action", tactile_root=tmp_path / "tactile"),
        tactile=SimpleNamespace(encoder_checkpoint=tmp_path / "encoder"),
        decoder=SimpleNamespace(batch_size=3, device="cpu", seed=0, workers=0, pin_memory=False, tactile_keys=(TACTILE_KEYS[2], TACTILE_KEYS[3])),
        source=SimpleNamespace(norm_stats_dir=tmp_path, norm_stats_asset_id="stats"),
        evaluation=SimpleNamespace(split="test", batch_size=2, shuffle_tactile=True, output=None),
    )
    observed = {}
    monkeypatch.setattr(evaluate, "load_config", lambda path: config, raising=False)
    monkeypatch.setattr(evaluate.ActionCache, "open", lambda path: "action")
    monkeypatch.setattr(evaluate.TactileEmbeddingCache, "open", lambda *args, **kwargs: observed.setdefault("tactile_open", kwargs) and "tactile", raising=False)
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
    assert observed["tactile_open"]["tactile_keys"] == config.decoder.tactile_keys


def test_training_modules_do_not_import_jax_flax_or_pi_runtime():
    completed = subprocess.run([sys.executable, "-c", "import train_baseline_pi05.data, train_baseline_pi05.train, train_baseline_pi05.evaluate; import sys; assert not any(name.startswith(('jax', 'flax', 'lerobot.policies.pi05')) for name in sys.modules)"], check=False)
    assert completed.returncode == 0
