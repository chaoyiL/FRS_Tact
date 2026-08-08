from __future__ import annotations

import dataclasses
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest
import torch
from safetensors.flax import load_file as load_safetensors_file

from train_smolvla.configuration import JaxSmolVLAConfig
from train_smolvla.data import (
    DeterministicEpochBatchSampler,
    DatasetSource,
    _KeyMappedLeRobotDataset,
    action_delta_timestamps,
    canonicalize_dataset_stats,
    ensure_stats_counts,
    fixed_stratified_subset_indices,
    parse_dataset_sources,
    prepare_lerobot_batch,
    rename_dataset_stats,
    resolve_action_key,
    resolve_source_visual_keys,
    split_sources_train_val,
)
from train_smolvla.preprocessing import JaxSmolVLAPreprocessor


def test_epoch_batch_sampler_can_resume_without_replaying_prefix() -> None:
    sampler = DeterministicEpochBatchSampler(
        10,
        batch_size=2,
        drop_last=True,
        shuffle=True,
        seed=7,
    )
    full_epoch = list(sampler)
    sampler.set_position(epoch=0, start_batch=3)
    resumed = list(sampler)

    assert resumed == full_epoch[3:]
    sampler.set_position(epoch=1)
    assert list(sampler) != full_epoch


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


def test_key_mapped_dataset_augments_only_selected_visual_keys() -> None:
    class FakeDataset:
        def __len__(self):
            return 1

        def __getitem__(self, index):
            assert index == 0
            return {
                "actions": torch.zeros(2, 3),
                "observation.images.cam0": torch.zeros(3, 4, 4),
                "observation.images.auxiliary": torch.ones(3, 4, 4),
                "task": "pick cube",
            }

    dataset = _KeyMappedLeRobotDataset(
        FakeDataset(),
        action_key="actions",
        rename_map={"observation.images.cam0": "observation.images.camera1"},
        image_transforms=lambda image: image + 2,
        image_transform_keys=("observation.images.camera1",),
    )
    sample = dataset[0]
    np.testing.assert_array_equal(sample["observation.images.camera1"], np.full((3, 4, 4), 2.0))
    np.testing.assert_array_equal(sample["observation.images.auxiliary"], np.ones((3, 4, 4)))


@pytest.mark.parametrize(
    ("episode_index", "frame_index"),
    [
        (torch.tensor(3), torch.tensor(5)),
        (np.int64(3), np.int32(5)),
        (3, 5),
    ],
    ids=("torch", "numpy", "python"),
)
def test_key_mapped_dataset_loads_cached_embedding_by_absolute_frame(
    episode_index,
    frame_index,
) -> None:
    from train_vtsmolvla.data import _VTKeyMappedLeRobotDataset as VtKeyMappedDataset
    from train_vtsmolvla.tactile_cache import TACTILE_EMBEDDING_OBSERVATION_KEY

    class FakeCache:
        def __getitem__(self, index):
            assert index == 105
            return np.full((4, 8), index, dtype=np.float16)

    class FakeDataset:
        meta = type("Meta", (), {"episodes": {3: {"dataset_from_index": 100}}})()

        def __len__(self):
            return 1

        def __getitem__(self, index):
            assert index == 0
            return {
                "actions": torch.zeros(2, 3),
                "observation.images.cam0": torch.zeros(3, 4, 4),
                "episode_index": episode_index,
                "frame_index": frame_index,
                "task": "pick cube",
            }

    dataset = VtKeyMappedDataset(
        FakeDataset(),
        action_key="actions",
        rename_map={},
        tactile_embedding_cache=FakeCache(),
    )
    sample = dataset[0]
    np.testing.assert_array_equal(
        sample[TACTILE_EMBEDDING_OBSERVATION_KEY],
        np.full((4, 8), 105, dtype=np.float16),
    )


def test_training_stats_are_saved_for_future_inference(tmp_path: Path) -> None:
    processor = object.__new__(JaxSmolVLAPreprocessor)
    processor.checkpoint = tmp_path / "source"
    processor.config = JaxSmolVLAConfig()
    processor.rename_map = {}
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


def test_preprocessor_prepares_tactile_images_separately() -> None:
    from train_vtsmolvla.configuration import VTSmolVLAConfig as JaxVTSmolVLAConfig
    from train_vtsmolvla.preprocessing import (
        VTJaxSmolVLAPreprocessor as JaxVTSmolVLAPreprocessor,
    )

    processor = object.__new__(JaxVTSmolVLAPreprocessor)
    processor.config = dataclasses.replace(
        JaxVTSmolVLAConfig(),
        image_keys=("observation.images.camera1",),
        use_tactile_encoder=True,
        tactile_encoder_path="checkpoints/encoder/best",
        tactile_keys=("observation.images.tactile_left_0", "observation.images.tactile_right_0"),
        tactile_num_tokens=2,
        tactile_image_size=8,
        resize_height=8,
        resize_width=8,
    )
    processor.rename_map = {}
    processor.stats = {}
    processor.post_stats = {}
    processor.tokenize = lambda task: (jnp.ones((1, 3), dtype=jnp.int32), jnp.ones((1, 3), dtype=jnp.bool_))

    observation = {
        "observation.state": np.ones((4,), dtype=np.float32),
        "observation.images.camera1": np.zeros((8, 8, 3), dtype=np.uint8),
        "observation.images.tactile_left_0": np.full((8, 8, 3), 127, dtype=np.uint8),
        "observation.images.tactile_right_0": np.full((8, 8, 3), 255, dtype=np.uint8),
    }
    batch = processor.prepare(observation, "task")
    assert batch["images"].shape == (1, 1, 3, 8, 8)
    assert batch["image_masks"].shape == (1, 1)
    assert batch["tactile_images"].shape == (1, 2, 8, 8, 3)
    assert batch["tactile_masks"].shape == (1, 2)
    np.testing.assert_allclose(np.asarray(batch["tactile_images"][0, 1]), 1.0)


def test_preprocessor_accepts_cached_tactile_embeddings() -> None:
    from train_vtsmolvla.configuration import VTSmolVLAConfig as JaxVTSmolVLAConfig
    from train_vtsmolvla.preprocessing import (
        VTJaxSmolVLAPreprocessor as JaxVTSmolVLAPreprocessor,
    )
    from train_vtsmolvla.tactile_cache import TACTILE_EMBEDDING_OBSERVATION_KEY

    processor = object.__new__(JaxVTSmolVLAPreprocessor)
    processor.config = dataclasses.replace(
        JaxVTSmolVLAConfig(),
        image_keys=("observation.images.camera1",),
        use_tactile_encoder=True,
        tactile_encoder_path="checkpoints/encoder/best",
        tactile_keys=("observation.images.tactile_left_0", "observation.images.tactile_right_0"),
        tactile_num_tokens=2,
        tactile_embedding_dim=8,
        resize_height=8,
        resize_width=8,
    )
    processor.rename_map = {}
    processor.stats = {}
    processor.post_stats = {}
    processor.tokenize = lambda task: (
        jnp.ones((1, 3), dtype=jnp.int32),
        jnp.ones((1, 3), dtype=jnp.bool_),
    )
    cached = np.arange(16, dtype=np.float16).reshape(2, 8)
    observation = {
        "observation.state": np.ones((4,), dtype=np.float32),
        "observation.images.camera1": np.zeros((8, 8, 3), dtype=np.uint8),
        TACTILE_EMBEDDING_OBSERVATION_KEY: cached,
    }
    batch = processor.prepare(observation, "task")
    assert "tactile_images" not in batch
    assert batch["tactile_embeddings"].dtype == jnp.float16
    np.testing.assert_array_equal(batch["tactile_embeddings"][0], cached)
    np.testing.assert_array_equal(batch["tactile_masks"], [[True, True]])


def test_tactile_preprocessing_matches_encoder_for_non_square_bchw() -> None:
    from train_vtsmolvla.preprocessing import prepare_tactile_batch
    from tactile_encoder.utils.image_dataset import parse_image_to_unit

    image = np.arange(3 * 5 * 9, dtype=np.uint8).reshape(3, 5, 9)
    batch = np.stack((image, np.flip(image, axis=-1)), axis=0)

    actual = prepare_tactile_batch(batch, image_size=8)
    expected = np.stack(
        [parse_image_to_unit(frame, image_size=8) for frame in batch],
        axis=0,
    )

    assert actual.shape == (2, 8, 8, 3)
    np.testing.assert_array_equal(actual, expected)


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


def test_split_sources_train_val_uses_explicit_episodes(monkeypatch) -> None:
    class FakeMeta:
        def __init__(self, **kwargs):
            del kwargs
            self.total_episodes = 100

    monkeypatch.setattr(
        "train_smolvla.data.LeRobotDatasetMetadata",
        FakeMeta,
    )
    sources = [
        DatasetSource(
            repo_id="org/a",
            episodes=list(range(10)),
            action_key="actions",
            weight=1.0,
        )
    ]
    train, val = split_sources_train_val(sources, val_fraction=0.2, seed=0)
    assert len(train) == 1 and len(val) == 1
    assert set(train[0].episodes).isdisjoint(val[0].episodes)
    assert len(train[0].episodes) + len(val[0].episodes) == 10
    assert len(val[0].episodes) == 2


def test_fixed_stratified_subset_is_reproducible_and_covers_every_dataset() -> None:
    lengths = [1000, 100, 500, 200]
    first = fixed_stratified_subset_indices(lengths, sample_count=64, seed=23)
    second = fixed_stratified_subset_indices(lengths, sample_count=64, seed=23)
    different = fixed_stratified_subset_indices(lengths, sample_count=64, seed=24)

    assert first == second
    assert first != different
    assert len(first) == len(set(first)) == 64
    offsets = np.cumsum([0, *lengths])
    for start, stop in zip(offsets[:-1], offsets[1:], strict=True):
        assert any(start <= index < stop for index in first)


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


def test_resolve_source_visual_keys_with_rename_map() -> None:
    keys = resolve_source_visual_keys(
        ["observation.images.camera1", "observation.images.camera2"],
        {
            "observation.images.camera0": "observation.images.camera1",
            "observation.images.camera1": "observation.images.camera2",
        },
        [
            "observation.images.camera0",
            "observation.images.camera1",
            "observation.images.unused",
        ],
    )
    assert keys == ["observation.images.camera0", "observation.images.camera1"]

    with pytest.raises(KeyError, match="allow_missing=0"):
        resolve_source_visual_keys(
            ["observation.images.camera1", "observation.images.empty_camera_0"],
            {"observation.images.camera0": "observation.images.camera1"},
            ["observation.images.camera0"],
        )
    keys_with_placeholder = resolve_source_visual_keys(
        ["observation.images.camera1", "observation.images.empty_camera_0"],
        {"observation.images.camera0": "observation.images.camera1"},
        ["observation.images.camera0"],
        allow_missing=1,
    )
    assert keys_with_placeholder == ["observation.images.camera0"]

    # The checkpoint can contain more absent names than ``empty_cameras``.
    # SmolVLA uses the real cameras, fills only the configured number of empty
    # slots, and ignores the remaining missing names.
    keys_with_multiple_missing = resolve_source_visual_keys(
        [
            "observation.images.camera1",
            "observation.images.camera2",
            "observation.images.camera3",
            "observation.images.empty_camera_0",
        ],
        {
            "observation.images.camera0": "observation.images.camera1",
            "observation.images.camera1": "observation.images.camera2",
        },
        [
            "observation.images.camera0",
            "observation.images.camera1",
            "observation.images.unused",
        ],
        allow_missing=4,
    )
    assert keys_with_multiple_missing == [
        "observation.images.camera0",
        "observation.images.camera1",
    ]


def test_resolve_model_visual_keys_includes_tactile_when_enabled() -> None:
    from train_vtsmolvla.configuration import VTSmolVLAConfig as JaxVTSmolVLAConfig
    from train_vtsmolvla.data import resolve_model_visual_keys as resolve_vt_visual_keys

    config = dataclasses.replace(
        JaxVTSmolVLAConfig(),
        image_keys=("observation.images.camera1", "observation.images.camera2"),
        use_tactile_encoder=True,
        tactile_encoder_path="checkpoints/encoder/best",
        tactile_keys=(
            "observation.images.tactile_left_0",
            "observation.images.tactile_right_0",
        ),
        tactile_num_tokens=2,
    )
    assert resolve_vt_visual_keys(config) == (
        "observation.images.camera1",
        "observation.images.camera2",
        "observation.images.tactile_left_0",
        "observation.images.tactile_right_0",
    )
    assert resolve_vt_visual_keys(config, use_tactile_embedding_cache=True) == (
        "observation.images.camera1",
        "observation.images.camera2",
    )
