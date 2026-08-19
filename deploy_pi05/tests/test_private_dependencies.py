"""Regression checks for Pi0.5's vendored runtime dependency boundary."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys


DEPLOY_ROOT = Path(__file__).resolve().parents[1]
PRIVATE_SOURCE = DEPLOY_ROOT / "src"
MANIFEST_PATH = DEPLOY_ROOT / "vendor_manifest.sha256"
TOOLS_MANIFEST_PATH = DEPLOY_ROOT / "pi05_tools_manifest.sha256"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_private_packages_resolve_from_deploy_pi05() -> None:
    """The launcher path ordering must select only the migration-local code."""
    probe = """
import importlib.util
import json
from pathlib import Path

packages = [
    \"lerobot\",
    \"lerobot.policies.pi05_jax\",
    \"utils\",
    \"utils.pi05_source_model\",
    \"train_pi05_frs\",
    \"tactile_encoder\",
]
print(json.dumps({name: importlib.util.find_spec(name).origin for name in packages}))
"""
    environment = os.environ | {
        "PYTHONPATH": os.pathsep.join((str(PRIVATE_SOURCE), str(DEPLOY_ROOT), str(DEPLOY_ROOT.parent))),
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    result = subprocess.run(
        [sys.executable, "-c", probe],
        check=True,
        capture_output=True,
        cwd=DEPLOY_ROOT,
        env=environment,
        text=True,
    )
    origins = {name: Path(origin).resolve() for name, origin in json.loads(result.stdout).items()}
    expected_roots = {
        "lerobot": PRIVATE_SOURCE / "lerobot",
        "utils": DEPLOY_ROOT / "utils",
        "train_pi05_frs": DEPLOY_ROOT / "train_pi05_frs",
        "tactile_encoder": DEPLOY_ROOT / "tactile_encoder",
    }

    for package, expected_root in expected_roots.items():
        assert origins[package].is_relative_to(expected_root.resolve())

    assert origins["lerobot.policies.pi05_jax"].is_relative_to(
        (PRIVATE_SOURCE / "lerobot" / "policies" / "pi05_jax").resolve()
    )
    assert origins["utils.pi05_source_model"].is_relative_to((DEPLOY_ROOT / "utils").resolve())


def _assert_manifest_matches_checked_in_bytes(manifest_path: Path) -> None:
    """The provenance manifest makes validation independent of the source tree."""
    entries = [
        line.split(maxsplit=1)
        for line in manifest_path.read_text(encoding="utf-8").splitlines()
        if line and not line.startswith("#")
    ]
    assert entries

    for expected_hash, relative_path in entries:
        path = DEPLOY_ROOT / relative_path
        assert path.is_file(), f"missing vendored file: {relative_path}"
        assert _sha256(path) == expected_hash, f"hash mismatch: {relative_path}"


def test_vendor_manifest_matches_checked_in_dependency_bytes() -> None:
    _assert_manifest_matches_checked_in_bytes(MANIFEST_PATH)


def test_pi05_tools_manifest_matches_checked_in_bytes() -> None:
    _assert_manifest_matches_checked_in_bytes(TOOLS_MANIFEST_PATH)
