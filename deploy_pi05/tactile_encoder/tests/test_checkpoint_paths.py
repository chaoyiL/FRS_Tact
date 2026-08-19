from pathlib import Path

import pytest

from tactile_encoder.utils.checkpoint import _resolve_checkpoint_file


def test_resolve_checkpoint_file_prefers_native_name(tmp_path: Path) -> None:
    native = tmp_path / "params.npz"
    native.touch()
    (tmp_path / "params-deadbeef.npz").touch()

    assert _resolve_checkpoint_file(tmp_path, "params.npz") == native


def test_resolve_checkpoint_file_accepts_hugging_face_hash_suffix(tmp_path: Path) -> None:
    uploaded = tmp_path / "params-235cb754d17b461b8be2d652c96fc169.npz"
    uploaded.touch()

    assert _resolve_checkpoint_file(tmp_path, "params.npz") == uploaded


def test_resolve_checkpoint_file_rejects_ambiguous_hashes(tmp_path: Path) -> None:
    (tmp_path / "params-first.npz").touch()
    (tmp_path / "params-second.npz").touch()

    with pytest.raises(FileNotFoundError, match="Multiple candidates"):
        _resolve_checkpoint_file(tmp_path, "params.npz")
