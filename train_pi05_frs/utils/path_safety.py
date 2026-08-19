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


def validate_output_roots(roots: Mapping[str, str | Path]) -> dict[str, Path]:
    """Resolve aliases and reject protected or mutually overlapping roots."""

    resolved = {
        str(label): Path(value).expanduser().resolve(strict=False)
        for label, value in roots.items()
    }
    exact_protected = tuple(path.resolve(strict=False) for path in _EXACT_PROTECTED)
    protected_subtrees = tuple(path.resolve(strict=False) for path in _PROTECTED_SUBTREES)
    allowed_train_generated = tuple(
        path.resolve(strict=False) for path in _ALLOWED_TRAIN_GENERATED
    )
    for label, path in resolved.items():
        inside_allowed_train_generated = any(
            _is_same_or_descendant(path, generated)
            for generated in allowed_train_generated
        )
        inside_repository = _is_same_or_descendant(path, REPO_ROOT)
        if (
            path in exact_protected
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
    return resolved
