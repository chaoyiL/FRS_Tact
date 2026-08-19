"""Background iterator prefetch so GPU steps overlap with batch assembly."""

from __future__ import annotations

import queue
import threading
from collections.abc import Callable, Iterator
from typing import TypeVar

T = TypeVar("T")

_STOP = object()


def prefetch_iterator(
    iterator: Iterator[T],
    *,
    buffer_size: int = 4,
    on_item: Callable[[T], T] | None = None,
) -> Iterator[T]:
    """Pull from ``iterator`` on a daemon thread while the consumer runs."""

    if buffer_size <= 0:
        raise ValueError(f"buffer_size must be positive, got {buffer_size}.")

    ready: queue.Queue[object] = queue.Queue(maxsize=buffer_size)
    error: list[BaseException] = []

    def _producer() -> None:
        try:
            for item in iterator:
                if on_item is not None:
                    item = on_item(item)
                ready.put(item)
        except BaseException as exc:  # noqa: BLE001 — propagate to consumer
            error.append(exc)
        finally:
            ready.put(_STOP)

    thread = threading.Thread(target=_producer, name="tactile-clip-prefetch", daemon=True)
    thread.start()

    while True:
        item = ready.get()
        if item is _STOP:
            if error:
                raise error[0]
            return
        yield item  # type: ignore[misc]
