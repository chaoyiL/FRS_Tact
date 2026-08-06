from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from lerobot.policies.smolvla_jax import atomic_checkpoint
from lerobot.policies.smolvla_jax.atomic_checkpoint import assemble_checkpoint_atomically


def test_final_path_appears_only_after_validation(tmp_path: Path) -> None:
    final = tmp_path / "checkpoint-00000020"
    staging = final.with_name(final.name + ".incomplete")
    observed: list[tuple[str, bool, bool]] = []

    def writer(path: Path) -> None:
        observed.append(("writer", final.exists(), path == staging))
        (path / "marker").write_text("complete", encoding="utf-8")

    def validator(path: Path) -> None:
        observed.append(("validator", final.exists(), path == staging))
        assert (path / "marker").read_text(encoding="utf-8") == "complete"

    assemble_checkpoint_atomically(final, writer, validator)

    assert observed == [
        ("writer", False, True),
        ("validator", False, True),
    ]
    assert (final / "marker").read_text(encoding="utf-8") == "complete"
    assert not staging.exists()


def test_failed_validation_preserves_incomplete_directory(tmp_path: Path) -> None:
    final = tmp_path / "checkpoint-00000020"

    def fail_validation(path: Path) -> None:
        assert (path / "marker").is_file()
        raise ValueError("invalid checkpoint")

    with pytest.raises(ValueError, match="invalid checkpoint"):
        assemble_checkpoint_atomically(
            final,
            lambda path: (path / "marker").write_text("incomplete", encoding="utf-8"),
            fail_validation,
        )

    assert not final.exists()
    staging = final.with_name(final.name + ".incomplete")
    assert (staging / "marker").read_text(encoding="utf-8") == "incomplete"


def test_failed_writer_preserves_incomplete_directory(tmp_path: Path) -> None:
    final = tmp_path / "checkpoint-00000020"

    def fail_write(path: Path) -> None:
        (path / "partial").write_text("diagnostic", encoding="utf-8")
        raise RuntimeError("save failed")

    with pytest.raises(RuntimeError, match="save failed"):
        assemble_checkpoint_atomically(final, fail_write, lambda path: None)

    assert not final.exists()
    staging = final.with_name(final.name + ".incomplete")
    assert (staging / "partial").read_text(encoding="utf-8") == "diagnostic"


@pytest.mark.parametrize("existing_name", ("final", "staging"))
def test_existing_checkpoint_paths_are_rejected_without_modification(
    tmp_path: Path,
    existing_name: str,
) -> None:
    final = tmp_path / "checkpoint-00000020"
    staging = final.with_name(final.name + ".incomplete")
    existing = final if existing_name == "final" else staging
    existing.mkdir()
    (existing / "user-data").write_text("keep", encoding="utf-8")
    writer_called = False

    def writer(path: Path) -> None:
        nonlocal writer_called
        writer_called = True

    with pytest.raises(FileExistsError, match="already exists"):
        assemble_checkpoint_atomically(final, writer, lambda path: None)

    assert not writer_called
    assert (existing / "user-data").read_text(encoding="utf-8") == "keep"


def test_publish_does_not_replace_final_created_after_last_exists_check(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    final = tmp_path / "checkpoint-00000020"
    staging = final.with_name(final.name + ".incomplete")
    original_path_exists = atomic_checkpoint._path_exists
    final_checks = 0
    competitor_inode: int | None = None

    def path_exists_with_race(path: Path) -> bool:
        nonlocal competitor_inode, final_checks
        exists = original_path_exists(path)
        if path == final:
            final_checks += 1
            if final_checks == 2:
                assert not exists
                final.mkdir()
                competitor_inode = final.stat().st_ino
                # The competitor appears immediately after this check.
                return False
        return exists

    monkeypatch.setattr(atomic_checkpoint, "_path_exists", path_exists_with_race)

    with pytest.raises(FileExistsError, match="already exists"):
        assemble_checkpoint_atomically(
            final,
            lambda path: (path / "marker").write_text("checkpoint", encoding="utf-8"),
            lambda path: None,
        )

    assert competitor_inode is not None
    assert final.stat().st_ino == competitor_inode
    assert list(final.iterdir()) == []
    assert (staging / "marker").read_text(encoding="utf-8") == "checkpoint"


def test_unsupported_posix_refuses_unsafe_rename_fallback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    final = tmp_path / "checkpoint-00000020"
    staging = final.with_name(final.name + ".incomplete")
    monkeypatch.setattr(
        atomic_checkpoint,
        "sys",
        SimpleNamespace(platform="unsupported-posix"),
        raising=False,
    )

    with pytest.raises(RuntimeError, match="no-replace"):
        assemble_checkpoint_atomically(
            final,
            lambda path: (path / "marker").write_text("checkpoint", encoding="utf-8"),
            lambda path: None,
        )

    assert not final.exists()
    assert (staging / "marker").read_text(encoding="utf-8") == "checkpoint"
