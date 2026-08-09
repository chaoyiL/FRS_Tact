from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch
from safetensors.flax import load_file as load_safetensors_file

from lerobot.policies.smolvla_jax.configuration import JaxSmolVLAConfig
from lerobot.policies.smolvla_jax.data import (
    DeterministicEpochBatchSampler,
    DatasetSource,
    LeRobotJaxDataLoader,
    _KeyMappedLeRobotDataset,
    action_delta_timestamps,
    canonicalize_dataset_stats,
    ensure_stats_counts,
    fixed_stratified_subset_indices,
    parse_dataset_sources,
    prepare_lerobot_batch,
    rename_dataset_stats,
    resolve_action_key,
    resolve_model_visual_keys,
    resolve_source_visual_keys,
    split_sources_train_val,
)
from lerobot.policies.smolvla_jax.preprocessing import JaxSmolVLAPreprocessor, prepare_tactile_batch
from lerobot.policies.smolvla_jax.tactile_cache import TACTILE_EMBEDDING_OBSERVATION_KEY
from tactile_encoder.utils.image_dataset import parse_image_to_unit


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


def test_key_mapped_dataset_augments_rgb_but_not_tactile() -> None:
    class FakeDataset:
        def __len__(self):
            return 1

        def __getitem__(self, index):
            assert index == 0
            return {
                "actions": torch.zeros(2, 3),
                "observation.images.cam0": torch.zeros(3, 4, 4),
                "observation.images.tactile": torch.ones(3, 4, 4),
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
    np.testing.assert_array_equal(sample["observation.images.tactile"], np.ones((3, 4, 4)))


def test_key_mapped_dataset_loads_cached_embedding_by_absolute_frame() -> None:
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
                "episode_index": torch.tensor(3),
                "frame_index": torch.tensor(5),
                "task": "pick cube",
            }

    dataset = _KeyMappedLeRobotDataset(
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
    processor = object.__new__(JaxSmolVLAPreprocessor)
    processor.config = dataclasses.replace(
        JaxSmolVLAConfig(),
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
    processor = object.__new__(JaxSmolVLAPreprocessor)
    processor.config = dataclasses.replace(
        JaxSmolVLAConfig(),
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
    monkeypatch.setattr(
        "lerobot.policies.smolvla_jax.data.LeRobotDatasetMetadata",
        lambda **kwargs: pytest.fail(
            f"explicit split must not construct full metadata: {kwargs}"
        ),
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


def test_split_sources_without_explicit_episodes_never_loads_full_stats(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "dataset"
    info_path = root / "meta" / "info.json"
    info_path.parent.mkdir(parents=True)
    info_path.write_text(
        json.dumps(
            {
                "codebase_version": "v3.0",
                "fps": 30,
                "total_episodes": 3,
                "features": {
                    "observation.state": {"dtype": "float32", "shape": [20]},
                    "actions": {"dtype": "float32", "shape": [20]},
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    episodes_path = root / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
    episodes_path.parent.mkdir(parents=True)
    pq.write_table(pa.Table.from_pylist([{"episode_index": index} for index in (2, 4, 7)]), episodes_path)
    monkeypatch.setattr(
        "lerobot.policies.smolvla_jax.data.LeRobotDatasetMetadata",
        lambda **kwargs: pytest.fail(f"full metadata/global stats were accessed: {kwargs}"),
    )

    train, val = split_sources_train_val(
        [DatasetSource(repo_id="org/a", root=root, action_key="actions")],
        val_fraction=1 / 3,
        seed=0,
    )

    selected = list(train[0].episodes or []) + list(val[0].episodes or [])
    assert sorted(selected) == [2, 4, 7]
    assert len(val[0].episodes or []) == 1


def test_remote_split_download_projects_only_info_and_episode_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    snapshot = tmp_path / "snapshot"
    info_path = snapshot / "meta" / "info.json"
    info_path.parent.mkdir(parents=True)
    info_path.write_text(
        json.dumps(
            {
                "codebase_version": "v3.0",
                "fps": 30,
                "total_episodes": 2,
                "features": {
                    "observation.state": {"dtype": "float32", "shape": [20]},
                    "actions": {"dtype": "float32", "shape": [20]},
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    episodes_path = snapshot / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
    episodes_path.parent.mkdir(parents=True)
    pq.write_table(pa.Table.from_pylist([{"episode_index": 0}, {"episode_index": 1}]), episodes_path)
    calls: list[dict[str, object]] = []

    def snapshot_download(repo_id: str, **kwargs) -> str:
        calls.append({"repo_id": repo_id, **kwargs})
        return str(snapshot)

    monkeypatch.setattr("lerobot.policies.smolvla_jax.data.HF_LEROBOT_HOME", tmp_path / "empty-cache")
    monkeypatch.setattr("lerobot.policies.smolvla_jax.data.snapshot_download", snapshot_download)
    monkeypatch.setattr(
        "lerobot.policies.smolvla_jax.data.LeRobotDatasetMetadata",
        lambda **kwargs: pytest.fail(f"full metadata/global stats were accessed: {kwargs}"),
    )

    train, val = split_sources_train_val(
        [DatasetSource(repo_id="org/remote", revision="commit-sha", action_key="actions")],
        val_fraction=0.5,
        seed=0,
    )

    assert sorted([*(train[0].episodes or []), *(val[0].episodes or [])]) == [0, 1]
    assert calls[0]["allow_patterns"] == [
        "meta/info.json",
        "meta/episodes/*/*.parquet",
    ]
    assert "meta/stats.json" not in calls[0]["allow_patterns"]


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


def test_explicit_preprocessor_does_not_read_full_dataset_stats(monkeypatch) -> None:
    class FakeMetadata:
        def __init__(self, **kwargs):
            del kwargs
            self.root = Path("/metadata-only")
            self.revision = "revision"
            self.features = {
                "observation.state": {"dtype": "float32", "shape": [20]},
                "actions": {"dtype": "float32", "shape": [20]},
                "observation.images.camera1": {"dtype": "video", "shape": [3, 8, 8]},
            }
            self.camera_keys = ["observation.images.camera1"]
            self.fps = 10

    class StatsMustNotBeRead:
        @property
        def stats(self):
            raise AssertionError("full-dataset stats must not be read")

    class FakeDataset:
        def __init__(self, **kwargs):
            del kwargs
            self.features = FakeMetadata().features
            self.meta = StatsMustNotBeRead()
            self.num_episodes = 1
            self.fps = 10

        def __len__(self):
            return 2

    monkeypatch.setattr(
        "lerobot.policies.smolvla_jax.data.LeRobotDatasetMetadata",
        FakeMetadata,
    )
    monkeypatch.setattr(
        "lerobot.policies.smolvla_jax.data.LeRobotDataset",
        FakeDataset,
    )
    config = dataclasses.replace(
        JaxSmolVLAConfig(),
        state_dim=20,
        action_dim=20,
        chunk_size=2,
        image_keys=("observation.images.camera1",),
    )
    preprocessor = object()

    loader = __import__(
        "lerobot.policies.smolvla_jax.data", fromlist=["LeRobotJaxDataLoader"]
    ).LeRobotJaxDataLoader(
        "checkpoint",
        config,
        sources=[DatasetSource(repo_id="org/dataset", episodes=[0], action_key="actions")],
        preprocessor=preprocessor,
        batch_size=1,
        num_workers=0,
        infinite=False,
        drop_last=False,
    )

    assert loader.preprocessor is preprocessor


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
            "observation.images.tactile_left_0",
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
            "observation.images.tactile_left_0",
        ],
        allow_missing=4,
    )
    assert keys_with_multiple_missing == [
        "observation.images.camera0",
        "observation.images.camera1",
    ]


def test_resolve_model_visual_keys_includes_tactile_when_enabled() -> None:
    config = dataclasses.replace(
        JaxSmolVLAConfig(),
        image_keys=("observation.images.camera1", "observation.images.camera2"),
        use_tactile_encoder=True,
        tactile_encoder_path="checkpoints/encoder/best",
        tactile_keys=(
            "observation.images.tactile_left_0",
            "observation.images.tactile_right_0",
        ),
        tactile_num_tokens=2,
    )
    assert resolve_model_visual_keys(config) == (
        "observation.images.camera1",
        "observation.images.camera2",
        "observation.images.tactile_left_0",
        "observation.images.tactile_right_0",
    )
    assert resolve_model_visual_keys(config, use_tactile_embedding_cache=True) == (
        "observation.images.camera1",
        "observation.images.camera2",
    )


def test_offline_loader_filters_episode_rows_and_resumes_without_rgb_or_tokenizer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    episodes = np.asarray([0, 0, 1, 1, 2, 2], dtype=np.int64)

    class FakeMetadata:
        def __init__(self, **kwargs):
            del kwargs
            self.root = tmp_path / "dataset"
            self.revision = "revision"
            self.total_frames = 6
            self.total_episodes = 3
            self.episodes = {
                index: {
                    "dataset_from_index": 2 * index,
                    "dataset_to_index": 2 * index + 2,
                }
                for index in range(3)
            }
            self.fps = 30
            self.camera_keys = ["cam0", "cam1", "touch0", "touch1"]
            self.features = {
                "observation.state": {"dtype": "float32", "shape": [3]},
                "actions": {"dtype": "float32", "shape": [2]},
                "cam0": {"dtype": "video", "shape": [3, 8, 8]},
                "cam1": {"dtype": "video", "shape": [3, 8, 8]},
                "touch0": {"dtype": "video", "shape": [3, 8, 8]},
                "touch1": {"dtype": "video", "shape": [3, 8, 8]},
            }

    class FakeOfflineCache:
        metadata = {"status": "complete"}
        spec = SimpleNamespace(camera_keys=("rgb0", "rgb1"))

        def __len__(self):
            return len(episodes)

        def __getitem__(self, index):
            index = int(index)
            return {
                "vision_tokens": np.full((2, 4, 960), index, dtype=jnp.bfloat16),
                "state": np.full((3,), index, dtype=np.float32),
                "actions": np.full((2, 2), index, dtype=np.float32),
                "action_is_pad": np.asarray([False, index % 2 == 1]),
                "language_tokens": np.asarray([index, index + 1, 0], dtype=np.int32),
                "language_masks": np.asarray([True, True, False]),
                "episode_index": episodes[index],
                "frame_index": np.asarray(index % 2, dtype=np.int64),
            }

    class FakeTactileCache:
        metadata = {"num_tactile_tokens": 2}

        def __getitem__(self, index):
            return np.full((2, 5), int(index), dtype=np.float16)

    class CacheOnlyPreprocessor:
        def prepare(self, *args, **kwargs):
            pytest.fail(f"cache mode decoded RGB: {args}, {kwargs}")

        def tokenize(self, *args, **kwargs):
            pytest.fail(f"cache mode invoked tokenizer: {args}, {kwargs}")

        def normalize_state(self, state):
            return state + 10

        def normalize_actions(self, actions):
            return actions * 2

    monkeypatch.setattr(
        "lerobot.policies.smolvla_jax.data.LeRobotDatasetMetadata", FakeMetadata
    )
    monkeypatch.setattr(
        "lerobot.policies.smolvla_jax.data.LeRobotDataset",
        lambda **kwargs: pytest.fail(f"cache mode constructed RGB dataset: {kwargs}"),
    )
    monkeypatch.setattr(
        "lerobot.policies.smolvla_jax.data.OfflineTrainingCache",
        lambda *args, **kwargs: FakeOfflineCache(),
    )
    monkeypatch.setattr(
        "lerobot.policies.smolvla_jax.data.TactileEmbeddingCache",
        lambda *args, **kwargs: FakeTactileCache(),
    )
    config = dataclasses.replace(
        JaxSmolVLAConfig(),
        state_dim=3,
        action_dim=2,
        max_state_dim=3,
        max_action_dim=2,
        chunk_size=2,
        resize_height=128,
        resize_width=128,
        image_keys=("rgb0", "rgb1"),
        use_tactile_encoder=True,
        tactile_encoder_path="unused",
        tactile_keys=("touch0", "touch1"),
        tactile_num_tokens=2,
        tactile_embedding_dim=5,
    )
    loader = LeRobotJaxDataLoader(
        "checkpoint",
        config,
        sources=[
            DatasetSource(
                repo_id="org/data",
                episodes=[0, 2],
                action_key="actions",
                rename_map={"cam0": "rgb0", "cam1": "rgb1"},
            )
        ],
        batch_size=2,
        num_workers=0,
        shuffle=False,
        infinite=False,
        drop_last=False,
        preprocessor=CacheOnlyPreprocessor(),
        tactile_embedding_cache_root=tmp_path / "tactile",
        offline_training_cache_root=tmp_path / "offline",
        host_prefetch_batches=2,
    )

    batch = next(loader.batches(start_batch=1))

    assert loader.full_dataset_size == 4
    assert "images" not in batch
    assert batch["vision_embeddings"].shape == (2, 2, 4, 960)
    assert batch["vision_embeddings"].dtype == jnp.bfloat16
    np.testing.assert_array_equal(
        batch["state"],
        np.asarray([[14, 14, 14], [15, 15, 15]], dtype=np.float32),
    )
    np.testing.assert_array_equal(batch["actions"][:, 0, 0], [8.0, 10.0])
    np.testing.assert_array_equal(batch["language_tokens"][:, 0], [4, 5])
    np.testing.assert_array_equal(batch["tactile_embeddings"][:, 0, 0], [4, 5])
    np.testing.assert_array_equal(batch["image_masks"], np.ones((2, 2), dtype=bool))
    np.testing.assert_array_equal(batch["tactile_masks"], np.ones((2, 2), dtype=bool))


def test_host_prefetch_surfaces_backing_iterator_failure() -> None:
    from lerobot.policies.smolvla_jax.data import _host_prefetch

    def backing():
        yield 1
        raise RuntimeError("prefetch exploded")

    batches = _host_prefetch(backing(), depth=1)
    assert next(batches) == 1
    with pytest.raises(RuntimeError, match="prefetch exploded"):
        next(batches)
