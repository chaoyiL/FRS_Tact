from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any


def _path_exists(path: Path) -> bool:
    """Treat broken symlinks as occupied paths too."""

    return path.exists() or path.is_symlink()


def assemble_checkpoint_atomically(
    final_path: str | Path,
    writer: Callable[[Path], Any],
    validator: Callable[[Path], Any],
) -> Path:
    """Build and validate a checkpoint before exposing its final directory name.

    Failed writes and validations intentionally leave the sibling ``.incomplete``
    directory untouched for diagnosis.
    """

    final = Path(final_path).expanduser()
    staging = final.with_name(final.name + ".incomplete")
    for path in (final, staging):
        if _path_exists(path):
            raise FileExistsError(f"checkpoint path already exists: {path}")

    staging.mkdir(parents=True)
    writer(staging)
    validator(staging)

    # Do not replace a checkpoint that appeared while the writer was running.
    if _path_exists(final):
        raise FileExistsError(f"checkpoint path already exists: {final}")
    staging.replace(final)
    return final
