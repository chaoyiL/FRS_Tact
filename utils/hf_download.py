"""Retry Hub rate limits and repair individual downloads with size mismatches."""

from __future__ import annotations

import math
import re
import sys
import time
from email.utils import parsedate_to_datetime
from pathlib import Path, PurePosixPath


def consistency_error_file(error: OSError, expected_sizes: dict[str, int]) -> str | None:
    """Resolve a Hub size error only to one explicitly selected, pinned file."""
    match = re.match(
        r"Consistency check failed: file should be of size (\d+) but has size \d+ \(([^\n)]+)\)",
        str(error),
    )
    if match is None:
        return None
    expected, reported = int(match[1]), match[2]
    candidates = [
        name
        for name, size in expected_sizes.items()
        if size == expected and reported in (name, PurePosixPath(name).name)
    ]
    return candidates[0] if len(candidates) == 1 else None


def force_download_hf_file(
    repo_id: str, filename: str, *, repo_type: str, revision: str, local_dir: Path, expected_size: int
) -> None:
    """Discard only this file's broken partial download, with three bounded attempts."""
    from huggingface_hub import hf_hub_download

    for attempt in range(3):
        print(f"Redownloading {repo_id}/{filename} from byte 0 (attempt {attempt + 1}/3)", flush=True)
        try:
            hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                repo_type=repo_type,
                revision=revision,
                local_dir=local_dir,
                force_download=True,
            )
        except OSError as error:
            if attempt == 2 or consistency_error_file(error, {filename: expected_size}) is None:
                raise
            continue
        path = local_dir / filename
        if not path.is_file() or path.stat().st_size != expected_size:
            raise OSError(f"Forced download still has the wrong size: {path}")
        return


def retry_after_seconds(value: str | None) -> float:
    if value is None:
        return 0.0
    try:
        seconds = float(value)
    except ValueError:
        try:
            seconds = parsedate_to_datetime(value).timestamp() - time.time()
        except (ValueError, TypeError, OverflowError):
            return 0.0
    return max(0.0, seconds) if math.isfinite(seconds) else 0.0


def retry_hf_call(operation, /, *args, **kwargs):
    """Keep the original arguments/revision; retry 429 up to six total attempts."""
    # Callers configure HF_HOME before their first Hub call; keep imports lazy.
    from huggingface_hub.errors import HfHubHTTPError

    for attempt in range(6):
        try:
            return operation(*args, **kwargs)
        except HfHubHTTPError as error:
            if error.response is None or error.response.status_code != 429:
                raise
            if attempt == 5:
                print(
                    "Hugging Face 429 persists. Log in with the data-tools Python or set HF_TOKEN, "
                    "then retry after the server's rate-limit window. Keep existing downloads.",
                    file=sys.stderr,
                    flush=True,
                )
                raise
            delay = max(
                min(60 * 2**attempt, 600),
                retry_after_seconds(error.response.headers.get("Retry-After")),
            )
            print(
                f"Hugging Face 429: waiting {delay:.0f}s before attempt {attempt + 2}/6. "
                "Completed files will be reused; authenticate if downloading anonymously.",
                file=sys.stderr,
                flush=True,
            )
            time.sleep(delay)
