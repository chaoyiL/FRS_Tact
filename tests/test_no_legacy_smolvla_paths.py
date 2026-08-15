"""Repository-level guards for the completed SmolVLA package migration."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
LEGACY_MODULE = "lerobot.policies.smolvla_jax"


def _python_sources() -> list[Path]:
    excluded_parts = {
        ".git",
        ".venv",
        ".worktrees",
        "build",
        "docs",
        "__pycache__",
        ".superpowers",
    }
    return [
        path
        for path in ROOT.rglob("*.py")
        if not excluded_parts.intersection(path.relative_to(ROOT).parts)
        and path != Path(__file__).resolve()
    ]


def _legacy_import_lines(source: str, *, filename: str) -> list[int]:
    lines: list[int] = []
    tree = ast.parse(source, filename=filename)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            if any(
                alias.name == LEGACY_MODULE or alias.name.startswith(f"{LEGACY_MODULE}.")
                for alias in node.names
            ):
                lines.append(node.lineno)
        elif isinstance(node, ast.ImportFrom):
            module = node.module
            absolute_legacy_import = module == LEGACY_MODULE or (
                module is not None and module.startswith(f"{LEGACY_MODULE}.")
            )
            legacy_parent_alias = module == "lerobot.policies" and any(
                alias.name == "smolvla_jax" for alias in node.names
            )
            relative_legacy_import = node.level > 0 and (
                (module is not None and (module == "smolvla_jax" or module.startswith("smolvla_jax.")))
                or (module is None and any(alias.name == "smolvla_jax" for alias in node.names))
            )
            if absolute_legacy_import or legacy_parent_alias or relative_legacy_import:
                lines.append(node.lineno)
    return lines


@pytest.mark.parametrize(
    ("source", "expected_lines"),
    (
        ("import lerobot.policies.smolvla_jax\n", [1]),
        ("from lerobot.policies.smolvla_jax import JaxSmolVLA\n", [1]),
        ("from lerobot.policies import smolvla_jax\n", [1]),
        ("from .smolvla_jax import JaxSmolVLA\n", [1]),
        ("from train_smolvla import JaxSmolVLA\n", []),
    ),
)
def test_legacy_import_line_detector(
    source: str, expected_lines: list[int]
) -> None:
    assert _legacy_import_lines(source, filename="legacy_import.py") == expected_lines


def test_repository_has_no_legacy_smolvla_imports() -> None:
    offenders: list[str] = []
    for path in _python_sources():
        for lineno in _legacy_import_lines(path.read_text(encoding="utf-8"), filename=str(path)):
            offenders.append(f"{path.relative_to(ROOT)}:{lineno}")

    assert offenders == []


def test_legacy_smolvla_source_and_removed_training_entrypoints_are_gone() -> None:
    legacy_paths = (
        ROOT / "lerobot/policies/smolvla_jax",
        ROOT / "tools/train_smolvla_jax.py",
        ROOT / "tools/train_vtsmolvla_jax.py",
        ROOT / "configs/train_smolvla_jax.yaml",
        ROOT / "configs/train_vtsmolvla_jax.yaml",
        ROOT / "scripts/start_vtsmolvla_train.sh",
    )
    assert [str(path.relative_to(ROOT)) for path in legacy_paths if path.exists()] == []
