from __future__ import annotations

import ctypes
import errno
import os
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

_AT_FDCWD = -100
_RENAME_NOREPLACE = 1


def _path_exists(path: Path) -> bool:
    """Treat broken symlinks as occupied paths too."""

    return path.exists() or path.is_symlink()


def _rename_noreplace(source: Path, destination: Path) -> None:
    """Atomically rename without ever replacing an existing destination."""

    if sys.platform.startswith("linux"):
        libc = ctypes.CDLL(None, use_errno=True)
        renameat2 = getattr(libc, "renameat2", None)
        if renameat2 is None:
            raise RuntimeError("atomic no-replace rename is unavailable on this Linux libc")
        renameat2.argtypes = (
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        )
        renameat2.restype = ctypes.c_int
        result = renameat2(
            _AT_FDCWD,
            os.fsencode(source),
            _AT_FDCWD,
            os.fsencode(destination),
            _RENAME_NOREPLACE,
        )
        if result == 0:
            return
        error = ctypes.get_errno()
        if error in (errno.EEXIST, errno.ENOTEMPTY):
            raise FileExistsError(
                error,
                f"checkpoint path already exists: {destination}",
                str(destination),
            )
        raise OSError(error, os.strerror(error), str(destination))

    if sys.platform.startswith("win"):
        # os.rename maps to a no-replace move on Windows.
        try:
            source.rename(destination)
        except FileExistsError as exc:
            raise FileExistsError(f"checkpoint path already exists: {destination}") from exc
        return

    # POSIX rename replaces an existing empty directory, so refusing to publish
    # is the only safe fallback when renameat2(RENAME_NOREPLACE) is unavailable.
    raise RuntimeError(f"atomic no-replace rename is unsupported on {sys.platform}")


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
    _rename_noreplace(staging, final)
    return final
