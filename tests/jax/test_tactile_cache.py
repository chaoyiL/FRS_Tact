from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from lerobot.policies.smolvla_jax.tactile_cache import (
    TACTILE_ENCODER_PROVENANCE_FILENAME,
    TACTILE_EMBEDDINGS_NAME,
    TACTILE_METADATA_NAME,
    TactileEmbeddingCache,
    create_tactile_cache_metadata,
    tactile_cache_dir,
    write_tactile_encoder_provenance,
)


def _encoder(path: Path) -> Path:
    path.mkdir()
    (path / "checkpoint.json").write_text("{}\n")
    (path / "params.npz").write_bytes(b"frozen-resnet")
    write_tactile_encoder_provenance(
        path,
        repo_id="liuchaoyi/encoder_ckpt_05",
        requested_revision="main",
        resolved_revision="a" * 40,
    )
    return path


def test_tactile_embedding_cache_reads_frame_memmap(tmp_path: Path) -> None:
    encoder = _encoder(tmp_path / "encoder")
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    cache_dir = tactile_cache_dir(tmp_path / "cache", "org/data")
    cache_dir.mkdir(parents=True)
    values = np.arange(5 * 2 * 3, dtype=np.float16).reshape(5, 2, 3)
    np.save(cache_dir / TACTILE_EMBEDDINGS_NAME, values)
    metadata = create_tactile_cache_metadata(
        repo_id="org/data",
        revision="v1",
        dataset_root=dataset,
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
    assert metadata["encoder_repo_id"] == "liuchaoyi/encoder_ckpt_05"
    assert metadata["encoder_revision"] == "a" * 40
    assert metadata["encoder_provenance_sha256"]
    assert metadata["dataset_content_identity"]["sha256"]
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
        dataset_root=dataset,
    )
    np.testing.assert_array_equal(cache[3], values[3])
    assert cache[3].flags.writeable


def test_tactile_embedding_cache_rejects_incomplete_cache(tmp_path: Path) -> None:
    encoder = _encoder(tmp_path / "encoder")
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    cache_dir = tactile_cache_dir(tmp_path / "cache", "org/data")
    cache_dir.mkdir(parents=True)
    np.save(cache_dir / TACTILE_EMBEDDINGS_NAME, np.zeros((2, 1, 2), dtype=np.float16))
    metadata = create_tactile_cache_metadata(
        repo_id="org/data",
        revision=None,
        dataset_root=dataset,
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
            dataset_root=dataset,
        )


def test_tactile_embedding_cache_allows_identical_dataset_at_different_root(tmp_path: Path) -> None:
    encoder = _encoder(tmp_path / "encoder")
    original = tmp_path / "original"
    replacement = tmp_path / "replacement"
    original.mkdir()
    replacement.mkdir()
    (original / "data.parquet").write_bytes(b"same-content")
    (replacement / "data.parquet").write_bytes(b"same-content")
    cache_dir = tactile_cache_dir(tmp_path / "cache", "org/data")
    cache_dir.mkdir(parents=True)
    np.save(cache_dir / TACTILE_EMBEDDINGS_NAME, np.zeros((1, 1, 2), dtype=np.float16))
    metadata = create_tactile_cache_metadata(
        repo_id="org/data",
        revision=None,
        dataset_root=original,
        total_frames=1,
        tactile_keys=("touch",),
        source_tactile_keys=("touch",),
        embedding_dim=2,
        image_size=8,
        dtype="float16",
        encoder_path=encoder,
        completed_frames=1,
        status="complete",
    )
    (cache_dir / TACTILE_METADATA_NAME).write_text(json.dumps(metadata))

    cache = TactileEmbeddingCache(
        cache_dir,
        repo_id="org/data",
        revision=None,
        total_frames=1,
        tactile_keys=("touch",),
        source_tactile_keys=("touch",),
        embedding_dim=2,
        image_size=8,
        encoder_path=encoder,
        dataset_root=replacement,
    )
    np.testing.assert_array_equal(cache[0], np.zeros((1, 2), dtype=np.float16))


def test_tactile_embedding_cache_rejects_same_root_after_data_changes(tmp_path: Path) -> None:
    encoder = _encoder(tmp_path / "encoder")
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    source_file = dataset / "data.parquet"
    source_file.write_bytes(b"original")
    cache_dir = tactile_cache_dir(tmp_path / "cache", "org/data")
    cache_dir.mkdir(parents=True)
    np.save(cache_dir / TACTILE_EMBEDDINGS_NAME, np.zeros((1, 1, 2), dtype=np.float16))
    metadata = create_tactile_cache_metadata(
        repo_id="org/data",
        revision=None,
        dataset_root=dataset,
        total_frames=1,
        tactile_keys=("touch",),
        source_tactile_keys=("touch",),
        embedding_dim=2,
        image_size=8,
        dtype="float16",
        encoder_path=encoder,
        completed_frames=1,
        status="complete",
    )
    (cache_dir / TACTILE_METADATA_NAME).write_text(json.dumps(metadata))
    source_file.write_bytes(b"replacement-with-a-different-size")

    with pytest.raises(ValueError, match="dataset_fingerprint"):
        TactileEmbeddingCache(
            cache_dir,
            repo_id="org/data",
            revision=None,
            total_frames=1,
            tactile_keys=("touch",),
            source_tactile_keys=("touch",),
            embedding_dim=2,
            image_size=8,
            encoder_path=encoder,
            dataset_root=dataset,
        )


def test_cache_rejects_encoder_provenance_or_content_drift(tmp_path: Path) -> None:
    encoder = _encoder(tmp_path / "encoder")
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    (dataset / "data.parquet").write_bytes(b"data")
    metadata = create_tactile_cache_metadata(
        repo_id="org/data",
        revision="b" * 40,
        dataset_root=dataset,
        total_frames=1,
        tactile_keys=("touch",),
        source_tactile_keys=("touch",),
        embedding_dim=2,
        image_size=8,
        dtype="float16",
        encoder_path=encoder,
        completed_frames=1,
        status="complete",
    )
    cache_dir = tactile_cache_dir(tmp_path / "cache", "org/data")
    cache_dir.mkdir(parents=True)
    np.save(cache_dir / TACTILE_EMBEDDINGS_NAME, np.zeros((1, 1, 2), dtype=np.float16))
    (cache_dir / TACTILE_METADATA_NAME).write_text(json.dumps(metadata))

    provenance_path = encoder / TACTILE_ENCODER_PROVENANCE_FILENAME
    provenance = json.loads(provenance_path.read_text())
    provenance["repo_id"] = "attacker/replacement"
    provenance_path.write_text(json.dumps(provenance))
    with pytest.raises(ValueError, match="repo|provenance|encoder"):
        TactileEmbeddingCache(
            cache_dir,
            repo_id="org/data",
            revision="b" * 40,
            total_frames=1,
            tactile_keys=("touch",),
            source_tactile_keys=("touch",),
            embedding_dim=2,
            image_size=8,
            encoder_path=encoder,
            dataset_root=dataset,
        )
