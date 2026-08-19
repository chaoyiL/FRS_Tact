"""Standalone Pi0.5 action-cache producer and cache readers."""

from pathlib import Path
import sys


_PRIVATE_SRC = Path(__file__).resolve().parents[1] / "src"
_private_src_text = str(_PRIVATE_SRC)
_loaded_lerobot = sys.modules.get("lerobot")
if _loaded_lerobot is not None:
    _loaded_path = Path(_loaded_lerobot.__file__).resolve()
    if _PRIVATE_SRC not in _loaded_path.parents:
        raise RuntimeError(
            "train_pi05_frs requires its private lerobot before another lerobot package is imported; "
            f"already loaded {_loaded_path}"
        )

_previous_sys_path = list(sys.path)
sys.path[:] = [entry for entry in sys.path if entry != _private_src_text]
sys.path.insert(0, _private_src_text)

try:
    from .cache import CachedPairs, MultiCachedPairs
    from .prepare import prepare_cache
except Exception:
    sys.path[:] = _previous_sys_path
    raise

__all__ = ["CachedPairs", "MultiCachedPairs", "prepare_cache"]
