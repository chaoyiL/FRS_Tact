from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from lerobot.datasets.tactile_cache import (
    TACTILE_EMBEDDINGS_NAME,
    TACTILE_METADATA_NAME,
    TactileEmbeddingCache,
    create_tactile_cache_metadata,
    tactile_cache_dir,
)


def _encoder(path: Path) -> Path:
    path.mkdir()
    (path / "checkpoint.json").write_text("{}\n")
    (path / "params.npz").write_bytes(b"frozen-resnet")
    return path


def test_tactile_embedding_cache_reads_frame_memmap(tmp_path: Path) -> None:
    encoder = _encoder(tmp_path / "encoder")
    cache_dir = tactile_cache_dir(tmp_path / "cache", "org/data")
    cache_dir.mkdir(parents=True)
    values = np.arange(5 * 2 * 3, dtype=np.float16).reshape(5, 2, 3)
    np.save(cache_dir / TACTILE_EMBEDDINGS_NAME, values)
    metadata = create_tactile_cache_metadata(
        repo_id="org/data",
        revision="v1",
        dataset_root=tmp_path / "dataset",
        total_frames=5,
        tactile_keys=("left", "right"),
        source_tactile_keys=("raw_left", "raw_right"),
        embedding_dim=3,
        image_size=8,
        dtype="float16",
        encoder_path=encoder,
        completed_frames=5,
        status="complete",
    )
    (cache_dir / TACTILE_METADATA_NAME).write_text(json.dumps(metadata))
    cache = TactileEmbeddingCache(
        cache_dir,
        repo_id="org/data",
        revision="v1",
        total_frames=5,
        tactile_keys=("left", "right"),
        source_tactile_keys=("raw_left", "raw_right"),
        embedding_dim=3,
        image_size=8,
        encoder_path=encoder,
    )
    np.testing.assert_array_equal(cache[3], values[3])
    assert cache[3].flags.writeable


def test_tactile_embedding_cache_rejects_incomplete_cache(tmp_path: Path) -> None:
    encoder = _encoder(tmp_path / "encoder")
    cache_dir = tactile_cache_dir(tmp_path / "cache", "org/data")
    cache_dir.mkdir(parents=True)
    np.save(cache_dir / TACTILE_EMBEDDINGS_NAME, np.zeros((2, 1, 2), dtype=np.float16))
    metadata = create_tactile_cache_metadata(
        repo_id="org/data",
        revision=None,
        dataset_root=tmp_path / "dataset",
        total_frames=2,
        tactile_keys=("touch",),
        source_tactile_keys=("touch",),
        embedding_dim=2,
        image_size=8,
        dtype="float16",
        encoder_path=encoder,
        completed_frames=1,
        status="incomplete",
    )
    (cache_dir / TACTILE_METADATA_NAME).write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="incomplete"):
        TactileEmbeddingCache(
            cache_dir,
            repo_id="org/data",
            revision=None,
            total_frames=2,
            tactile_keys=("touch",),
            source_tactile_keys=("touch",),
            embedding_dim=2,
            image_size=8,
            encoder_path=encoder,
        )
