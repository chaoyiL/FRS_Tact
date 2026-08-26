"""Fast contract tests for the frozen direct-Pi0.5 cache producers."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
import sys
import types

import jax.numpy as jnp
import numpy as np


def _config(tmp_path: Path) -> SimpleNamespace:
    return SimpleNamespace(
        cache=SimpleNamespace(action_root=tmp_path / "actions", tactile_root=tmp_path / "tactile"),
        source=SimpleNamespace(
            checkpoint=tmp_path / "source", norm_stats_dir=tmp_path / "stats", norm_stats_asset_id="demo",
            seed=0, sample_steps=3, action_horizon=50, model_action_dim=20,
        ),
        decoder=SimpleNamespace(action_dim=20),
        tactile=SimpleNamespace(encoder_checkpoint=tmp_path / "encoder", embedding_dim=512),
        dataset=SimpleNamespace(repo_id="org/demo", revision="v1", root=tmp_path / "dataset", action_key="actions"),
    )


def test_fixed_noise_is_seed_zero_batch_one_sample_repeated_exactly() -> None:
    from train_baseline_pi05.source_model import fixed_noise

    noise = fixed_noise(3, seed=0, horizon=50, action_dim=32)
    reference = fixed_noise(1, seed=0, horizon=50, action_dim=32)

    assert noise.shape == (3, 50, 32)
    np.testing.assert_array_equal(np.asarray(noise), np.repeat(np.asarray(reference), 3, axis=0))


def test_forward_sampler_uses_only_native_sample_actions() -> None:
    from train_baseline_pi05.source_model import sample_coarse_actions

    class Model:
        def sample_actions(self, params, observation, *, noise, num_steps):
            self.call = (params, observation, noise, num_steps)
            return noise + 2

    model = Model()
    noise = jnp.zeros((2, 50, 23), dtype=jnp.float32)
    result = sample_coarse_actions(model, "params", "observation", noise, 3)

    assert model.call[3] == 3
    np.testing.assert_array_equal(result, np.full((2, 50, 23), 2, dtype=np.float32))


def test_default_dependencies_pass_source_variants_action_dim_and_camera_mapping(monkeypatch, tmp_path: Path) -> None:
    from train_baseline_pi05.prepare_action_cache import _default_dependencies

    config = _config(tmp_path)
    config.dataset.revision = None
    config.dataset.rename_map = {"observation.images.camera0": "observation.images.camera1"}
    config.dataset.camera_map = {"left_wrist_0_rgb": "observation.images.camera1"}
    config.source.model_action_dim = 32
    config.source.paligemma_variant = "gemma_2b_lora"
    config.source.action_expert_variant = "gemma_300m_lora"
    config.source.use_quantile_norm = True
    captured = {}

    class Metadata:
        pass

    class Dataset:
        pass

    class Processor:
        def __init__(self, **kwargs):
            captured.update(kwargs)
            self.config = object()

    class Model:
        action_dim = 32

    import lerobot.datasets
    monkeypatch.setattr(lerobot.datasets, "LeRobotDatasetMetadata", lambda *args, **kwargs: Metadata())
    monkeypatch.setattr(lerobot.datasets, "LeRobotDataset", lambda *args, **kwargs: Dataset())
    monkeypatch.setitem(sys.modules, "lerobot.policies.pi05_jax", types.SimpleNamespace(
        load_norm_stats=lambda *args: {"state": "state", "actions": "actions"}
    ))
    monkeypatch.setitem(sys.modules, "train_baseline_pi05.policy_inputs", types.SimpleNamespace(Pi05SampleProcessor=Processor))
    monkeypatch.setattr("train_baseline_pi05.prepare_action_cache.load_pi05_source_model", lambda *args, **kwargs: (Model(), 32))

    dependencies = _default_dependencies(config)

    assert captured["action_dim"] == 32
    assert captured["paligemma_variant"] == "gemma_2b_lora"
    assert captured["action_expert_variant"] == "gemma_300m_lora"
    assert captured["use_quantile_norm"] is True
    assert captured["rename_map"] == config.dataset.rename_map
    assert captured["camera_map"] == config.dataset.camera_map
    assert dependencies["params"] is not None


def test_default_tactile_encoder_uses_only_tactile_resnet_tree(monkeypatch, tmp_path: Path) -> None:
    from train_baseline_pi05.tactile_cache import _default_encoder

    captured = {}
    bundle = SimpleNamespace(params={"tactile_resnet": {"params": {}, "batch_stats": {}}, "future_projection": {"bad": 1}})
    monkeypatch.setattr("train_baseline_pi05.tactile_encoder.encoder_checkpoint.load_tactile_encoder", lambda path: bundle)

    def fake_encode(variables, images, *, train):
        captured["variables"] = variables
        captured["images"] = images
        return np.zeros((4, 512), dtype=np.float32), None

    monkeypatch.setattr("train_baseline_pi05.tactile_encoder.resnet.encode_resnet18", fake_encode)
    encoded = _default_encoder(tmp_path / "encoder")(np.zeros((4, 224, 224, 3), dtype=np.float32))

    assert captured["variables"] == bundle.params["tactile_resnet"]
    assert encoded.shape == (4, 512)


def test_producer_source_tree_has_no_reverse_or_frs_integration() -> None:
    root = Path("train_baseline_pi05")
    forbidden = (
        "sample_and_reverse",
        "reverse_integrate_actions",
        "pi05_jax.frs",
        "train_pi05_frs",
        "x_base",
        "integration",
    )
    paths = [root / "source_model.py", root / "prepare_action_cache.py"]
    paths.extend((root / "src" / "lerobot" / "policies" / "pi05_jax").rglob("*.py"))
    for path in paths:
        source = path.read_text(encoding="utf-8")
        assert not any(token in source for token in forbidden), path


def test_action_producer_writes_sliced_forward_actions_and_preserves_records(tmp_path: Path) -> None:
    from train_baseline_pi05.action_cache import ActionCache, build_records
    from train_baseline_pi05.prepare_action_cache import prepare_action_cache

    class Metadata:
        total_episodes = 1
        episodes = [{"dataset_from_index": 0, "dataset_to_index": 2}]

    class Dataset:
        def __getitem__(self, index: int) -> dict[str, object]:
            return {
                "observation.state": np.full(3, index), "task": "pick", "actions": np.full(20, index),
                "observation.images.tactile_left_0": np.zeros((3, 3, 3), dtype=np.uint8),
            }

    class Processor:
        def prepare_sample(self, sample):
            assert not any("tactile" in key for key in sample)
            return {"state": sample["observation.state"], "task": sample["task"]}, np.zeros((50, 20), np.float32), "pick"

    class Source:
        action_dim = 32

        def sample_actions(self, params, observation, *, noise, num_steps):
            assert observation["state"].shape == (2, 3)
            assert noise.shape[-1] == 32
            return np.broadcast_to(np.arange(32, dtype=np.float32), noise.shape)

    config = _config(tmp_path)
    config.source.model_action_dim = 32
    records = build_records(Metadata(), split_seed=7, frame_stride=1, fractions=(1, 0, 0))
    result = prepare_action_cache(
        config,
        dependencies={
            "metadata": Metadata(), "dataset": Dataset(), "records": records,
            "processor": Processor(), "model": Source(), "params": "weights", "batch_size": 2,
        },
    )

    cache = ActionCache.open(result)
    assert cache.coarse_actions.shape == (2, 50, 20)
    np.testing.assert_array_equal(cache.coarse_actions[0, 0], np.arange(20, dtype=np.float32))
    np.testing.assert_array_equal(cache.dataset_indices, [0, 1])
    np.testing.assert_array_equal(cache.valid_masks[0], np.r_[np.ones(2, dtype=bool), np.zeros(48, dtype=bool)])
    np.testing.assert_array_equal(cache.valid_masks[1], np.r_[True, np.zeros(49, dtype=bool)])
    manifest = json.loads((result / "manifest.json").read_text())
    assert manifest["source_model_action_width"] == 32
    assert manifest["decoder_action_width"] == 20


def test_action_producer_resumes_an_incomplete_cache(tmp_path: Path) -> None:
    from train_baseline_pi05.action_cache import ActionCache, ActionCacheWriter, SampleRecord
    from train_baseline_pi05.prepare_action_cache import prepare_action_cache

    class Metadata:
        total_episodes = 1
        episodes = [{"dataset_from_index": 0, "dataset_to_index": 2}]

    class Dataset:
        def __getitem__(self, index):
            return {"observation.state": np.full(3, index), "task": "pick", "actions": np.full(20, index)}

    class Processor:
        def prepare_sample(self, sample):
            return {"state": sample["observation.state"]}, np.zeros((50, 20), np.float32), "pick"

    class Source:
        action_dim = 32
        def sample_actions(self, rng, observation, *, noise, num_steps):
            return np.broadcast_to(np.arange(32, dtype=np.float32), noise.shape)

    config = _config(tmp_path)
    config.source.model_action_dim = 32
    records = (SampleRecord(0, 0, 0, 0), SampleRecord(1, 0, 1, 0))
    manifest = {
        "dataset_identity": {"repo_id": "org/demo", "root": str(config.dataset.root), "revision": "v1"},
        "split": {"seed": 0, "fractions": [0.8, 0.1, 0.1], "frame_stride": 1},
        "source_checkpoint": str(config.source.checkpoint),
        "source_variant": {"paligemma_variant": None, "action_expert_variant": None},
        "norm_stats": {"dir": str(config.source.norm_stats_dir), "asset_id": "demo", "use_quantile_norm": True},
        "sample_steps": 3, "noise_seed": 0, "source_model_action_width": 32,
        "decoder_action_width": 20, "action_space": "normalized_pi05",
    }
    with ActionCacheWriter.create(config.cache.action_root, sample_count=2, horizon=50, action_dim=20, manifest=manifest) as writer:
        writer.write_batch(0, coarse=np.zeros((1, 50, 20), np.float32), expert=np.zeros((1, 50, 20), np.float32), valid=np.concatenate([np.ones((1, 2), bool), np.zeros((1, 48), bool)], axis=1), records=records[:1])

    output = prepare_action_cache(config, dependencies={
        "metadata": Metadata(), "dataset": Dataset(), "records": records, "processor": Processor(),
        "model": Source(), "params": "rng", "batch_size": 1,
    })

    cache = ActionCache.open(output)
    assert cache.manifest["completed_samples"] == 2
    np.testing.assert_array_equal(cache.dataset_indices, [0, 1])


def test_tactile_producer_writes_current_four_sensor_rms_tokens_and_mmap_reader(tmp_path: Path) -> None:
    from train_baseline_pi05.tactile_cache import TactileEmbeddingCache, prepare_tactile_cache

    class Dataset:
        def __len__(self):
            return 2

        def __getitem__(self, index: int):
            return {
                "observation.images.tactile_left_0": np.full((6, 8, 3), 10 + index, np.uint8),
                "observation.images.tactile_right_0": np.full((6, 8, 3), 20 + index, np.uint8),
                "observation.images.tactile_left_1": np.full((6, 8, 3), 30 + index, np.uint8),
                "observation.images.tactile_right_1": np.full((6, 8, 3), 40 + index, np.uint8),
            }

    seen: list[tuple[int, ...]] = []
    def encoder(images: np.ndarray) -> np.ndarray:
        seen.append(images.shape)
        return np.tile(np.arange(1, 513, dtype=np.float32), (images.shape[0], 1))

    output = prepare_tactile_cache(_config(tmp_path), dependencies={"dataset": Dataset(), "encoder": encoder})
    reader = TactileEmbeddingCache.open(
        output,
        tactile_keys=(
            "observation.images.tactile_left_0", "observation.images.tactile_right_0",
            "observation.images.tactile_left_1", "observation.images.tactile_right_1",
        ),
        encoder_path=tmp_path / "encoder",
    )

    assert seen == [(4, 224, 224, 3), (4, 224, 224, 3)]
    assert isinstance(reader.embeddings, np.memmap)
    assert reader.embeddings.dtype == np.float32
    assert reader.embeddings.shape == (2, 4, 512)
    np.testing.assert_allclose(np.sqrt(np.mean(reader.embeddings[0, 0] ** 2)), 1.0, atol=1e-6)
