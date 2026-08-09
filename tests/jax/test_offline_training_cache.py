from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import jax.numpy as jnp
import ml_dtypes
import numpy as np
import pytest

from lerobot.policies.smolvla_jax.offline_training_cache import (
    ACTION_IS_PAD_NAME,
    ACTIONS_NAME,
    EPISODE_INDEX_NAME,
    FRAME_INDEX_NAME,
    LANGUAGE_MASKS_NAME,
    LANGUAGE_TOKENS_NAME,
    METADATA_NAME,
    OFFLINE_CACHE_SCHEMA_VERSION,
    STATE_NAME,
    VISION_TOKENS_NAME,
    OfflineCacheSpec,
    OfflineTrainingCache,
    bfloat16_to_uint16,
    offline_cache_dir,
    uint16_to_bfloat16,
)


def expected_spec() -> OfflineCacheSpec:
    return OfflineCacheSpec(
        repo_id="org/dataset",
        total_frames=3,
        camera_keys=("left", "right"),
        vision_tokens_per_camera=2,
        vision_hidden_size=4,
        state_dim=3,
        action_dim=2,
        chunk_size=4,
        tokenizer_max_length=5,
        checkpoint_source="hf://org/checkpoint@main",
        vision_mode="frozen",
        connector_mode="frozen",
    )


def make_complete_fixture(
    tmp_path: Path,
    *,
    status: str = "complete",
    camera_keys: tuple[str, ...] | None = None,
) -> Path:
    spec = expected_spec()
    keys = spec.camera_keys if camera_keys is None else camera_keys
    cache_dir = offline_cache_dir(tmp_path / "cache", spec.repo_id)
    cache_dir.mkdir(parents=True)
    metadata = {
        "version": OFFLINE_CACHE_SCHEMA_VERSION,
        "status": status,
        "repo_id": spec.repo_id,
        "total_frames": spec.total_frames,
        "camera_keys": list(keys),
        "vision_tokens_per_camera": spec.vision_tokens_per_camera,
        "vision_hidden_size": spec.vision_hidden_size,
        "state_dim": spec.state_dim,
        "action_dim": spec.action_dim,
        "chunk_size": spec.chunk_size,
        "tokenizer_max_length": spec.tokenizer_max_length,
        "checkpoint_source": spec.checkpoint_source,
        "vision_mode": spec.vision_mode,
        "connector_mode": spec.connector_mode,
    }
    (cache_dir / METADATA_NAME).write_text(json.dumps(metadata))
    vision = np.asarray(
        jnp.arange(
            spec.total_frames
            * len(keys)
            * spec.vision_tokens_per_camera
            * spec.vision_hidden_size,
            dtype=jnp.bfloat16,
        )
    ).reshape(
        spec.total_frames,
        len(keys),
        spec.vision_tokens_per_camera,
        spec.vision_hidden_size,
    )
    np.save(cache_dir / VISION_TOKENS_NAME, bfloat16_to_uint16(vision))
    np.save(cache_dir / STATE_NAME, np.zeros((spec.total_frames, spec.state_dim), dtype=np.float32))
    np.save(
        cache_dir / ACTIONS_NAME,
        np.zeros((spec.total_frames, spec.chunk_size, spec.action_dim), dtype=np.float32),
    )
    np.save(cache_dir / ACTION_IS_PAD_NAME, np.zeros((spec.total_frames, spec.chunk_size), dtype=bool))
    np.save(
        cache_dir / LANGUAGE_TOKENS_NAME,
        np.zeros((spec.total_frames, spec.tokenizer_max_length), dtype=np.int32),
    )
    np.save(
        cache_dir / LANGUAGE_MASKS_NAME,
        np.ones((spec.total_frames, spec.tokenizer_max_length), dtype=bool),
    )
    np.save(cache_dir / EPISODE_INDEX_NAME, np.arange(spec.total_frames, dtype=np.int64))
    np.save(cache_dir / FRAME_INDEX_NAME, np.arange(spec.total_frames, dtype=np.int64))
    return cache_dir


def test_offline_cache_dir_uses_repo_id_path_parts(tmp_path: Path) -> None:
    assert offline_cache_dir(tmp_path, "org/dataset") == tmp_path / "org" / "dataset"
    with pytest.raises(ValueError, match="repo id"):
        offline_cache_dir(tmp_path, "../dataset")


def test_bfloat16_uint16_roundtrip_is_bit_exact() -> None:
    source = np.asarray(jnp.array([0.0, -1.0, 1.2345, np.inf], dtype=jnp.bfloat16))
    stored = bfloat16_to_uint16(source)
    restored = uint16_to_bfloat16(stored)
    assert stored.dtype == np.uint16
    assert restored.dtype == ml_dtypes.bfloat16
    np.testing.assert_array_equal(source.view(np.uint16), restored.view(np.uint16))


def test_uint16_to_bfloat16_rejects_non_uint16_storage() -> None:
    with pytest.raises(TypeError, match="uint16"):
        uint16_to_bfloat16(np.zeros(1, dtype=np.float32))


def test_cache_exposes_logical_bfloat16_tokens_and_raw_small_arrays(tmp_path: Path) -> None:
    cache = OfflineTrainingCache(make_complete_fixture(tmp_path), expected_spec())

    first = cache[1]
    second = cache[1]

    assert cache.spec == expected_spec()
    assert len(cache) == expected_spec().total_frames
    assert first is not second
    assert first["vision_tokens"].dtype == ml_dtypes.bfloat16
    assert first["vision_tokens"].shape == (2, 2, 4)
    assert first["state"].dtype == np.float32
    assert first["actions"].dtype == np.float32
    assert first["action_is_pad"].dtype == np.bool_
    assert first["language_tokens"].dtype == np.int32
    assert first["language_masks"].dtype == np.bool_
    assert first["episode_index"].dtype == np.int64
    assert first["frame_index"].dtype == np.int64


def test_cache_rejects_incomplete_and_incompatible_metadata(tmp_path: Path) -> None:
    cache_dir = make_complete_fixture(tmp_path, status="incomplete")
    with pytest.raises(ValueError, match="incomplete"):
        OfflineTrainingCache(cache_dir, expected_spec())
    cache_dir = make_complete_fixture(tmp_path / "incompatible", camera_keys=("right", "left"))
    with pytest.raises(ValueError, match="camera_keys"):
        OfflineTrainingCache(cache_dir, expected_spec())


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("total_frames", 4),
        ("vision_tokens_per_camera", 3),
        ("vision_hidden_size", 5),
        ("chunk_size", 5),
        ("tokenizer_max_length", 6),
        ("checkpoint_source", "hf://org/other@main"),
        ("vision_mode", "trainable"),
        ("connector_mode", "trainable"),
    ],
)
def test_cache_rejects_each_incompatible_metadata_field(
    tmp_path: Path, field: str, replacement: object
) -> None:
    cache_dir = make_complete_fixture(tmp_path)
    with pytest.raises(ValueError, match=field):
        OfflineTrainingCache(cache_dir, replace(expected_spec(), **{field: replacement}))


def test_cache_rejects_missing_array_file(tmp_path: Path) -> None:
    cache_dir = make_complete_fixture(tmp_path)
    (cache_dir / ACTIONS_NAME).unlink()

    with pytest.raises(FileNotFoundError, match=ACTIONS_NAME):
        OfflineTrainingCache(cache_dir, expected_spec())


@pytest.mark.parametrize(
    ("filename", "replacement", "error"),
    [
        (VISION_TOKENS_NAME, np.zeros((3, 2, 2, 4), dtype=np.float32), "vision_tokens"),
        (STATE_NAME, np.zeros((3, 4), dtype=np.float32), "state"),
        (ACTIONS_NAME, np.zeros((3, 4, 2), dtype=np.float16), "actions"),
        (ACTION_IS_PAD_NAME, np.zeros((3, 5), dtype=bool), "action_is_pad"),
        (LANGUAGE_TOKENS_NAME, np.zeros((3, 5), dtype=np.int64), "language_tokens"),
        (LANGUAGE_MASKS_NAME, np.zeros((3, 6), dtype=bool), "language_masks"),
        (EPISODE_INDEX_NAME, np.zeros((3, 1), dtype=np.int64), "episode_index"),
        (FRAME_INDEX_NAME, np.zeros(3, dtype=np.int32), "frame_index"),
    ],
)
def test_cache_rejects_wrong_field_shape_or_dtype(
    tmp_path: Path, filename: str, replacement: np.ndarray, error: str
) -> None:
    cache_dir = make_complete_fixture(tmp_path)
    np.save(cache_dir / filename, replacement)

    with pytest.raises(ValueError, match=error):
        OfflineTrainingCache(cache_dir, expected_spec())
