"""Activate this standalone project's vendored LeRobot runtime on demand."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path


_VENDORED_ROOT = Path(__file__).resolve().parent / "src"


def _is_vendored(module: object) -> bool:
    filename = getattr(module, "__file__", None)
    if not filename:
        return False
    return _VENDORED_ROOT.resolve() in Path(filename).resolve().parents


def activate_vendored_lerobot() -> None:
    """Make the local Pi0.5 runtime win before any absolute ``lerobot`` import."""
    vendored = str(_VENDORED_ROOT.resolve())
    sys.path[:] = [entry for entry in sys.path if Path(entry or ".").resolve() != _VENDORED_ROOT.resolve()]
    sys.path.insert(0, vendored)
    loaded = sys.modules.get("lerobot")
    if loaded is not None and not _is_vendored(loaded):
        unrelated = [name for name in sys.modules if name.startswith("lerobot.") and not name.startswith("lerobot.policies")]
        if unrelated:
            raise RuntimeError("activate_vendored_lerobot must run before unrelated lerobot modules are imported")
        for name in [name for name in sys.modules if name == "lerobot" or name.startswith("lerobot.policies")]:
            sys.modules.pop(name, None)
    importlib.invalidate_caches()
    importlib.import_module("lerobot")
