"""Canonical output/cache path boundaries for the standalone training project."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path


TRAIN_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = TRAIN_ROOT.parent

_EXACT_PROTECTED = (
    Path("/"),
    REPO_ROOT / "pyproject.toml",
    REPO_ROOT / "uv.lock",
)
_ALLOWED_TRAIN_GENERATED = (TRAIN_ROOT / ".cache", TRAIN_ROOT / "outputs")
_PROTECTED_SUBTREES = tuple(
    REPO_ROOT / name
    for name in (
        ".git",
        ".superpowers",
        "deploy_pi05",
        "deploy_smolvla",
        "docs",
        "lerobot",
        "modalities_eval",
        "scripts",
        "tests",
        "tools",
        "train_encoder",
        "train_smolvla",
        "train_smolvla_frs",
        "train_vtsmolvla",
        "utils",
    )
) + tuple(
    TRAIN_ROOT / name
    for name in (
        "configs",
        "pi05_cache",
        "scripts",
        "src",
        "tests",
        "tools",
        "utils",
    )
)


def _is_same_or_descendant(path: Path, root: Path) -> bool:
    return path == root or root in path.parents


def _overlap(left: Path, right: Path) -> bool:
    return _is_same_or_descendant(left, right) or _is_same_or_descendant(right, left)


def validate_output_roots(
    roots: Mapping[str, str | Path],
    *,
    read_only_roots: Mapping[str, str | Path] | None = None,
) -> dict[str, Path]:
    """Resolve aliases and reject protected, writable, or read/write overlaps."""

    requested = {
        str(label): Path(value).expanduser().absolute()
        for label, value in roots.items()
    }
    resolved = {
        label: path.resolve(strict=False) for label, path in requested.items()
    }
    exact_protected = tuple(path.resolve(strict=False) for path in _EXACT_PROTECTED)
    protected_subtrees = tuple(path.resolve(strict=False) for path in _PROTECTED_SUBTREES)
    allowed_train_generated = tuple(
        (path.expanduser().absolute(), path.resolve(strict=False))
        for path in _ALLOWED_TRAIN_GENERATED
    )
    for label, path in resolved.items():
        requested_path = requested[label]
        inside_allowed_train_generated = any(
            lexical == canonical
            and _is_same_or_descendant(requested_path, lexical)
            and _is_same_or_descendant(path, canonical)
            for lexical, canonical in allowed_train_generated
        )
        inside_repository = _is_same_or_descendant(path, REPO_ROOT)
        claims_unsafe_generated_root = (
            any(
                _is_same_or_descendant(requested_path, lexical)
                for lexical, _ in allowed_train_generated
            )
            and not inside_allowed_train_generated
        )
        if (
            path in exact_protected
            or claims_unsafe_generated_root
            or (inside_repository and not inside_allowed_train_generated)
            or any(
                _is_same_or_descendant(path, protected)
                for protected in protected_subtrees
            )
        ):
            raise ValueError(f"{label} targets a protected repository/source path: {path}")

    items = list(resolved.items())
    for index, (left_label, left_path) in enumerate(items):
        for right_label, right_path in items[index + 1 :]:
            if _overlap(left_path, right_path):
                raise ValueError(
                    f"output/cache roots overlap: {left_label}={left_path} and "
                    f"{right_label}={right_path}"
                )
    resolved_read_only = {
        str(label): Path(value).expanduser().absolute().resolve(strict=False)
        for label, value in (read_only_roots or {}).items()
    }
    for writable_label, writable_path in items:
        for read_only_label, read_only_path in resolved_read_only.items():
            if _overlap(writable_path, read_only_path):
                raise ValueError(
                    "writable/read-only roots overlap: "
                    f"{writable_label}={writable_path} and "
                    f"{read_only_label}={read_only_path}"
                )
    return resolved


def validate_fresh_output_root(
    output_root: str | Path, *, owned_pipeline_log: str | Path | None = None
) -> None:
    """Reject stale fresh-run output while allowing this launcher's sole open log."""

    output = Path(output_root).expanduser().absolute().resolve(strict=False)
    if not output.exists():
        return
    if not output.is_dir():
        raise FileExistsError(f"fresh training output directory is not empty: {output}")
    entries = list(output.iterdir())
    if not entries:
        return
    if owned_pipeline_log is not None:
        log = Path(owned_pipeline_log).expanduser().absolute().resolve(strict=False)
        if (
            log.parent == output
            and log.is_file()
            and len(entries) == 1
            and entries[0].resolve(strict=False) == log
        ):
            return
    raise FileExistsError(f"fresh training output directory is not empty: {output}")


def validate_implicit_resume_root(output_root: str | Path) -> Path:
    """Allow only legacy ``last/`` or an in-output immutable generation target."""

    output = Path(output_root).expanduser().absolute().resolve(strict=False)
    last = output / "last"
    target = last.resolve(strict=False)
    if last.is_symlink():
        generations_path = output / ".checkpoint-generations"
        generations = generations_path.resolve(strict=False)
        valid = (
            generations == generations_path
            and generations.parent == output
            and target.parent == generations
            and target != generations
        )
    else:
        valid = target == last
    if not valid:
        raise ValueError(
            "implicit resume checkpoint must remain in the same output as legacy "
            f"last/ or .checkpoint-generations/<generation>: {last} -> {target}"
        )
    return target
