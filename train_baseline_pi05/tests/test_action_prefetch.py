"""Ordering, overlap, and cleanup contracts for action-cache CPU prefetch."""

from __future__ import annotations

import threading

import pytest

from train_baseline_pi05.prepare_action_cache import _prefetched_batches


@pytest.mark.parametrize("enabled", [False, True])
def test_batches_preserve_order_and_partial_tail(enabled: bool) -> None:
    records = tuple(range(11))
    seen = []

    def prepare(start):
        seen.append(start)
        return records[start : start + 4]

    output = list(_prefetched_batches(range(0, len(records), 4), prepare, enabled=enabled))

    assert seen == [0, 4, 8]
    assert [(start, value) for start, value, _ in output] == [
        (0, (0, 1, 2, 3)), (4, (4, 5, 6, 7)), (8, (8, 9, 10)),
    ]
    assert all(wait_seconds >= 0 for _, _, wait_seconds in output)


@pytest.mark.parametrize("enabled", [False, True])
def test_empty_batches_do_not_prepare_anything(enabled: bool) -> None:
    def prepare(_):
        pytest.fail("an empty sequence must not request a batch")

    assert list(_prefetched_batches(iter(()), prepare, enabled=enabled)) == []


def test_disabled_batches_prepare_synchronously_without_lookahead() -> None:
    main_thread = threading.current_thread()
    calls = []

    def prepare(start):
        calls.append((start, threading.current_thread()))
        return start * 2

    batches = _prefetched_batches([3, 7], prepare, enabled=False)
    try:
        assert calls == []
        assert next(batches)[:2] == (3, 6)
        assert calls == [(3, main_thread)]
        assert next(batches)[:2] == (7, 14)
        assert calls == [(3, main_thread), (7, main_thread)]
        with pytest.raises(StopIteration):
            next(batches)
    finally:
        batches.close()


def test_prefetch_overlaps_consumption_with_only_one_batch_ahead() -> None:
    second_started = threading.Event()
    release_second = threading.Event()
    second_finished = threading.Event()
    third_started = threading.Event()
    workers = set()

    def prepare(start):
        workers.add(threading.current_thread())
        if start == 1:
            second_started.set()
            assert release_second.wait(timeout=5), "test did not release the next batch"
            second_finished.set()
        elif start == 2:
            third_started.set()
        return start + 10

    batches = _prefetched_batches([0, 1, 2], prepare, enabled=True)
    try:
        assert next(batches)[:2] == (0, 10)
        # The consumer is paused after the yield, yet its next batch starts.
        assert second_started.wait(timeout=5)
        release_second.set()
        assert second_finished.wait(timeout=5)
        # A finished lookahead must not trigger an unbounded producer queue.
        assert not third_started.wait(timeout=0.1)
        assert next(batches)[:2] == (1, 11)
        assert third_started.wait(timeout=5)
        assert next(batches)[:2] == (2, 12)
        with pytest.raises(StopIteration):
            next(batches)
    finally:
        release_second.set()
        batches.close()

    assert len(workers) == 1
    assert threading.current_thread() not in workers
    assert all(not worker.is_alive() for worker in workers)


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("failure_start", [0, 1])
def test_prepare_errors_propagate_and_release_workers(enabled: bool, failure_start: int) -> None:
    workers = set()

    def prepare(start):
        workers.add(threading.current_thread())
        if start == failure_start:
            raise RuntimeError("cannot decode sample")
        return start

    batches = _prefetched_batches([0, 1, 2], prepare, enabled=enabled)
    try:
        if failure_start:
            assert next(batches)[:2] == (0, 0)
        with pytest.raises(RuntimeError, match="cannot decode sample"):
            next(batches)
    finally:
        batches.close()

    if enabled:
        assert all(not worker.is_alive() for worker in workers)


def test_closing_early_joins_the_prefetch_worker() -> None:
    second_started = threading.Event()
    release_second = threading.Event()
    workers = set()

    def prepare(start):
        workers.add(threading.current_thread())
        if start == 1:
            second_started.set()
            assert release_second.wait(timeout=5)
        return start

    batches = _prefetched_batches([0, 1, 2], prepare, enabled=True)
    try:
        assert next(batches)[:2] == (0, 0)
        assert second_started.wait(timeout=5)
    finally:
        release_second.set()
        batches.close()

    assert len(workers) == 1
    assert all(not worker.is_alive() for worker in workers)
