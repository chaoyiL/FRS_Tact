"""Both sensor layouts retain cache ordering and dataset alignment."""

import json
from types import SimpleNamespace

import numpy as np
import pytest

from train_baseline_pi05.config import TACTILE_KEYS
from train_baseline_pi05.data import BaselineCacheDataset
from train_baseline_pi05.tactile_cache import TactileEmbeddingCache, prepare_tactile_cache


RIGHT_KEYS = ("observation.images.tactile_left_1", "observation.images.tactile_right_1")
OLD_WRONG_RIGHT_KEYS = ("observation.images.tactile_right_0", "observation.images.tactile_right_1")


def test_right_arm_cache_selects_both_faces_of_camera1(tmp_path):
    config = SimpleNamespace(
        cache=SimpleNamespace(tactile_root=tmp_path / "tactile"),
        decoder=SimpleNamespace(tactile_keys=RIGHT_KEYS),
        tactile=SimpleNamespace(encoder_checkpoint=tmp_path / "encoder", embedding_dim=512),
        dataset=SimpleNamespace(repo_id="test", root=tmp_path / "dataset", revision=None),
    )
    sample = {key: np.full((4, 4, 3), value, np.uint8) for key, value in zip(TACTILE_KEYS, (10, 20, 30, 40), strict=True)}
    seen = []

    def encode(images):
        seen.append(images[:, 0, 0, 0].copy())
        return np.ones((len(images), 512), np.float32)

    output = prepare_tactile_cache(config, dependencies={"dataset": [sample], "encoder": encode})
    np.testing.assert_allclose(seen[0], np.array([30, 40]) / 255, atol=1e-6)
    assert TactileEmbeddingCache.open(output, encoder_path=config.tactile.encoder_checkpoint).metadata["tactile_keys"] == list(RIGHT_KEYS)


def test_cache_reader_rejects_old_cross_arm_pair_even_when_requested(tmp_path):
    (tmp_path / "manifest.json").write_text(json.dumps({"status": "complete", "tactile_keys": OLD_WRONG_RIGHT_KEYS}))
    with pytest.raises(ValueError, match="key order"):
        TactileEmbeddingCache.open(tmp_path, tactile_keys=OLD_WRONG_RIGHT_KEYS, encoder_path=tmp_path / "encoder")


@pytest.mark.parametrize("keys", [RIGHT_KEYS, TACTILE_KEYS])
def test_sensor_layout_round_trips_cache_and_aligned_dataset(tmp_path, keys):
    config = SimpleNamespace(
        cache=SimpleNamespace(tactile_root=tmp_path / "tactile"),
        decoder=SimpleNamespace(tactile_keys=keys),
        tactile=SimpleNamespace(encoder_checkpoint=tmp_path / "encoder", embedding_dim=512),
        dataset=SimpleNamespace(repo_id="test", root=tmp_path / "dataset", revision=None),
    )
    samples = [{key: np.full((4, 4, 3), 20 + sensor, np.uint8) for sensor, key in enumerate(keys)}]
    seen = []

    def encoder(images):
        seen.append(images[:, 0, 0, 0].copy())
        return np.stack([np.arange(1, 513, dtype=np.float32) + sensor for sensor in range(len(images))])

    output = prepare_tactile_cache(config, dependencies={"dataset": samples, "encoder": encoder})
    cache = TactileEmbeddingCache.open(output, encoder_path=config.tactile.encoder_checkpoint)
    assert cache.embeddings.shape == (1, len(keys), 512)
    assert cache.metadata["tactile_keys"] == list(keys)
    np.testing.assert_allclose(seen[0], np.arange(20, 20 + len(keys)) / 255, atol=1e-6)
    np.testing.assert_allclose(np.sqrt(np.mean(cache.embeddings ** 2, axis=-1)), 1, atol=1e-6)
    actions = SimpleNamespace(
        manifest={"dataset_identity": cache.metadata["dataset_identity"]},
        dataset_indices=np.array([0]), episode_indices=np.array([7]), indices=lambda split: np.array([0]),
        coarse_actions=np.ones((1, 50, 10), np.float32),
        expert_actions=np.ones((1, 50, 10), np.float32), valid_masks=np.ones((1, 50), bool),
    )
    sample = BaselineCacheDataset(actions, cache, "train")[0]
    assert sample["tactile"].shape == (len(keys), 512)
    np.testing.assert_array_equal(sample["tactile"].numpy(), cache.embeddings[0])

    with pytest.raises(ValueError, match="key order"):
        TactileEmbeddingCache.open(output, tactile_keys=keys[::-1], encoder_path=config.tactile.encoder_checkpoint)
    cache.metadata["tactile_keys"] = list(keys[::-1])
    with pytest.raises(ValueError, match="key order"):
        BaselineCacheDataset(actions, cache, "train")


def test_cache_reader_rejects_unsupported_manifest_sensor_layout(tmp_path):
    # Even a caller echoing a corrupt manifest must not authorize a new layout.
    keys = ["observation.images.tactile_left_0", "observation.images.tactile_left_1"]
    (tmp_path / "manifest.json").write_text(json.dumps({"status": "complete", "tactile_keys": keys}))
    with pytest.raises(ValueError, match="key order"):
        TactileEmbeddingCache.open(tmp_path, tactile_keys=keys, encoder_path=tmp_path / "encoder")


@pytest.mark.parametrize("keys", [RIGHT_KEYS, TACTILE_KEYS])
def test_batched_encoding_preserves_frame_sensor_order_and_partial_batch(tmp_path, keys):
    samples = [
        {key: np.full((4, 4, 3), 20 + frame * 4 + sensor, np.uint8) for sensor, key in enumerate(keys)}
        for frame in range(5)
    ]
    config = SimpleNamespace(
        cache=SimpleNamespace(tactile_root=tmp_path / "tactile", tactile_batch_size=2),
        decoder=SimpleNamespace(tactile_keys=keys),
        tactile=SimpleNamespace(encoder_checkpoint=tmp_path / "encoder", embedding_dim=512),
        dataset=SimpleNamespace(repo_id="test", root=tmp_path / "dataset", revision=None),
    )
    calls = []

    def encode(images):
        calls.append(images.shape[0])
        return np.arange(1, 513, dtype=np.float32)[None, :] + images[:, 0, 0, 0, None]

    output = prepare_tactile_cache(config, dependencies={"dataset": samples, "encoder": encode})
    assert calls == [2 * len(keys), 2 * len(keys), len(keys)]
    cached = TactileEmbeddingCache.open(output, encoder_path=config.tactile.encoder_checkpoint)
    for frame in range(5):
        for sensor in range(len(keys)):
            expected = np.arange(1, 513, dtype=np.float32) + (20 + frame * 4 + sensor) / 255
            expected /= np.sqrt(np.mean(expected ** 2))
            np.testing.assert_allclose(cached.embeddings[frame, sensor], expected, atol=1e-6)


def test_image_cache_skips_unselected_images_and_torch_transforms(tmp_path):
    import datasets
    from PIL import Image

    unused = "observation.images.camera0"
    columns = {key: [Image.fromarray(np.full((8, 8, 3), 40, np.uint8))] * 3 for key in RIGHT_KEYS}
    columns[unused] = [{"bytes": b"not an image", "path": None}] * 3
    raw = datasets.Dataset.from_dict(columns, features=datasets.Features({key: datasets.Image() for key in columns}))
    raw.set_transform(lambda batch: (_ for _ in ()).throw(AssertionError("torch transform must be bypassed")))

    class ImageDataset:
        meta = SimpleNamespace(image_keys=list(columns))
        hf_dataset = raw

        def __len__(self):
            return 3

        def __getitem__(self, index):
            raise AssertionError("image-only cache must use selected image columns")

    config = SimpleNamespace(
        cache=SimpleNamespace(tactile_root=tmp_path / "tactile", tactile_batch_size=2),
        decoder=SimpleNamespace(tactile_keys=RIGHT_KEYS),
        tactile=SimpleNamespace(encoder_checkpoint=tmp_path / "encoder", embedding_dim=512),
        dataset=SimpleNamespace(repo_id="test", root=tmp_path / "dataset", revision=None),
    )
    output = prepare_tactile_cache(config, dependencies={"dataset": ImageDataset(), "encoder": lambda images: np.ones((len(images), 512), np.float32)})
    assert np.load(output / "embeddings.npy").shape == (3, 2, 512)


@pytest.mark.parametrize("shape", [(224, 224), (320, 240), (120, 160)])
def test_raw_image_preprocessing_matches_legacy_float_resize(shape):
    from train_baseline_pi05.tactile_cache import _raw_image_to_unit
    from train_baseline_pi05.tactile_encoder.preprocess import parse_image_to_unit
    from PIL import Image

    pixels = np.random.default_rng(3).integers(0, 256, (*shape, 3), dtype=np.uint8)
    legacy = np.moveaxis(pixels.astype(np.float32) / np.float32(255), -1, 0)
    np.testing.assert_array_equal(
        _raw_image_to_unit(Image.fromarray(pixels)),
        parse_image_to_unit(legacy, image_size=224),
    )
