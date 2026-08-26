"""Fast contract tests for the frozen direct-Pi0.5 cache producers."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np


def _config(tmp_path: Path) -> SimpleNamespace:
    return SimpleNamespace(
        cache=SimpleNamespace(action_root=tmp_path / "actions", tactile_root=tmp_path / "tactile"),
        source=SimpleNamespace(checkpoint=tmp_path / "source", seed=0, sample_steps=3, action_horizon=50),
        decoder=SimpleNamespace(action_dim=20),
        tactile=SimpleNamespace(encoder_checkpoint=tmp_path / "encoder", embedding_dim=512),
        dataset=SimpleNamespace(repo_id="org/demo", revision="v1", root=tmp_path / "dataset"),
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
            return {"observation.state": np.full(3, index), "task": "pick", "actions": np.full(20, index)}

    class Processor:
        def prepare_sample(self, sample):
            assert "tactile" not in sample
            return {"state": sample["observation.state"], "task": sample["task"]}, np.zeros((50, 20), np.float32), "pick"

    class Source:
        action_dim = 32

        def sample_actions(self, params, observation, *, noise, num_steps):
            assert observation["state"].shape == (2, 3)
            return np.broadcast_to(np.arange(32, dtype=np.float32), noise.shape)

    config = _config(tmp_path)
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
