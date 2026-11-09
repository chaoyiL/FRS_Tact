"""Retry Hugging Face rate limits, including failures while acquiring Xet tokens."""

from __future__ import annotations

import math
import sys
import time
from email.utils import parsedate_to_datetime


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
