from __future__ import annotations

import dataclasses
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest
import torch
from safetensors.flax import load_file as load_safetensors_file

from lerobot.policies.smolvla_jax.configuration import JaxSmolVLAConfig
from lerobot.policies.smolvla_jax.data import (
    action_delta_timestamps,
    canonicalize_dataset_stats,
    ensure_stats_counts,
    parse_dataset_sources,
    prepare_lerobot_batch,
    rename_dataset_stats,
    resolve_action_key,
)
from lerobot.policies.smolvla_jax.preprocessing import JaxSmolVLAPreprocessor


def test_action_key_and_delta_timestamps() -> None:
    assert resolve_action_key({"actions": {}}) == "actions"
    assert resolve_action_key({"action": {}}) == "action"
    assert resolve_action_key({"custom": {}}, "custom") == "custom"
    with pytest.raises(ValueError):
        resolve_action_key({"action": {}, "actions": {}})
    assert action_delta_timestamps("actions", chunk_size=4, fps=20) == {"actions": [0.0, 0.05, 0.1, 0.15]}


def test_dataset_stats_use_canonical_action_name() -> None:
    stats = {
        "observation.state": {"mean": [1.0], "std": [2.0]},
        "actions": {"mean": [3.0], "std": [4.0]},
    }
    canonical = canonicalize_dataset_stats(stats, "actions")
    assert "actions" not in canonical
    assert canonical["action"] == stats["actions"]


class FakePreprocessor:
    def prepare(self, observation, tasks):
        assert tasks == ["task zero", "task one"]
        return {
            "images": jnp.asarray(observation["observation.images.main"])[:, None],
            "image_masks": jnp.ones((2, 1), dtype=jnp.bool_),
            "language_tokens": jnp.ones((2, 3), dtype=jnp.int32),
            "language_masks": jnp.ones((2, 3), dtype=jnp.bool_),
            "state": jnp.asarray(observation["observation.state"]),
        }

    def normalize_actions(self, actions):
        return actions * 2


def test_prepare_lerobot_batch_converts_torch_and_padding() -> None:
    config = dataclasses.replace(
        JaxSmolVLAConfig(),
        chunk_size=2,
        action_dim=3,
        max_action_dim=4,
    )
    raw = {
        "observation.state": torch.arange(8, dtype=torch.float32).reshape(2, 4),
        "observation.images.main": torch.zeros(2, 3, 4, 4),
        "actions": torch.ones(2, 2, 3),
        "actions_is_pad": torch.tensor([[False, False], [False, True]]),
        "task": ["task zero", "task one"],
    }
    batch = prepare_lerobot_batch(raw, FakePreprocessor(), config, "actions")
    assert batch["images"].shape == (2, 1, 3, 4, 4)
    np.testing.assert_array_equal(batch["actions"], np.full((2, 2, 3), 2.0))
    np.testing.assert_array_equal(
        batch["action_is_pad"],
        np.asarray([[False, False], [False, True]]),
    )


def test_training_stats_are_saved_for_future_inference(tmp_path: Path) -> None:
    processor = object.__new__(JaxSmolVLAPreprocessor)
    processor.stats = {
        "observation.state.mean": jnp.asarray([1.0, 2.0]),
        "action.std": jnp.asarray([3.0, 4.0]),
    }
    processor.post_stats = dict(processor.stats)
    processor.save_normalization_assets(tmp_path)

    pre = load_safetensors_file(tmp_path / "policy_preprocessor_step_5_normalizer_processor.safetensors")
    post = load_safetensors_file(tmp_path / "policy_postprocessor_step_0_unnormalizer_processor.safetensors")
    np.testing.assert_array_equal(pre["observation.state.mean"], [1.0, 2.0])
    np.testing.assert_array_equal(post["action.std"], [3.0, 4.0])


def test_parse_dataset_sources() -> None:
    sources = parse_dataset_sources(
        {
            "datasets": [
                {
                    "repo_id": "org/a",
                    "action_key": "actions",
                    "weight": 2.0,
                    "rename_map": {"observation.images.cam0": "observation.images.camera1"},
                },
                {
                    "repo_id": "org/b",
                    "action_key": "action",
                    "rename_map": {"observation.images.x": "observation.images.camera1"},
                },
            ],
        }
    )
    assert len(sources) == 2
    assert sources[0].repo_id == "org/a"
    assert sources[0].weight == 2.0
    assert sources[0].rename_map == {"observation.images.cam0": "observation.images.camera1"}
    assert sources[1].action_key == "action"
    with pytest.raises(ValueError):
        parse_dataset_sources({})
    with pytest.raises(ValueError):
        parse_dataset_sources({"datasets": []})


def test_rename_and_count_stats_for_aggregation() -> None:
    stats = canonicalize_dataset_stats(
        {
            "observation.state": {"mean": [0.0], "std": [1.0]},
            "actions": {"mean": [2.0], "std": [3.0]},
            "observation.images.cam0": {"mean": [0.5], "std": [0.1]},
        },
        "actions",
    )
    renamed = rename_dataset_stats(
        stats,
        {"observation.images.cam0": "observation.images.camera1"},
    )
    assert "action" in renamed
    assert "observation.images.camera1" in renamed
    assert "observation.images.cam0" not in renamed
    counted = ensure_stats_counts(renamed, frame_count=10)
    np.testing.assert_array_equal(counted["action"]["count"], [10])
