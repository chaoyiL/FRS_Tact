"""Spawn-based multiprocess batch loading for tactile CLIP pretraining.

Workers each open their own ImageOnlyLeRobotDataset (via repo_ids) so the
parent JAX/CUDA process never forks after GPU init and never pickles dataset handles.

``MpBatchLoader`` keeps workers alive across epochs so datasets are not reloaded
every epoch.
"""

from __future__ import annotations

import os
import traceback
from collections.abc import Iterator, Sequence
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from tactile_encoder.utils.data import DataKeys
    from tactile_encoder.utils.data import FutureRecord

# Sentinel: stop workers.
_STOP = None


def _worker_loop(task_q: Any, result_q: Any, init: dict[str, Any]) -> None:
    """Top-level worker entry (must be picklable for spawn)."""

    # Belt-and-suspenders: parent should already pass CPU-only env at spawn time.
    os.environ["JAX_PLATFORMS"] = "cpu"
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

    try:
        from tactile_encoder.utils.data import load_pair
        from tactile_encoder.utils.data import stack_pairs
        from tactile_encoder.utils.image_dataset import create_image_dataset

        repo_ids = init["repo_ids"]
        image_size = int(init["image_size"])
        cache_size = int(init["cache_size"])
        pair_threads = max(1, int(init.get("pair_threads", 8)))
        keys = init["keys"]
        records = tuple(init["records"])
        tactile_history = int(init.get("tactile_history", 0))
        history_stride = int(init.get("history_stride", 5))
        dataset = create_image_dataset(
            repo_ids,
            image_size=image_size,
            cache_size=cache_size,
        ).dataset
    except Exception as exc:  # noqa: BLE001 — surface init failure to parent
        result_q.put(("__init__", None, f"worker init failed: {exc}\n{traceback.format_exc()}"))
        return

    sides = keys.sides

    def _one(item: tuple[int, int]) -> dict[str, np.ndarray]:
        record_i, side_i = item
        return load_pair(
            dataset,
            records[int(record_i)],
            sides[int(side_i)],
            image_size=image_size,
            side_id=int(side_i),
            as_uint8=True,
            tactile_history=tactile_history,
            history_stride=history_stride,
        )

    with ThreadPoolExecutor(max_workers=pair_threads) as pair_pool:
        while True:
            item = task_q.get()
            if item is _STOP:
                break
            batch_id, selected = item
            try:
                if pair_threads <= 1 or len(selected) <= 1:
                    pairs = [_one(sample) for sample in selected]
                else:
                    pairs = list(pair_pool.map(_one, selected, chunksize=4))
                batch = stack_pairs(pairs, as_uint8=True)
                result_q.put((batch_id, batch, None))
            except Exception as exc:  # noqa: BLE001 — return error to parent
                result_q.put((batch_id, None, f"{exc}\n{traceback.format_exc()}"))


class MpBatchLoader:
    """Persistent spawn workers that reload only once for many epochs."""

    def __init__(
        self,
        *,
        repo_ids: str | Sequence[str],
        records: Sequence[FutureRecord],
        keys: DataKeys,
        image_size: int,
        image_cache_size: int,
        num_workers: int,
        prefetch_batches: int,
        pair_threads: int = 8,
        tactile_history: int = 0,
        history_stride: int = 5,
    ):
        import multiprocessing as mp

        if num_workers <= 1:
            raise ValueError("MpBatchLoader requires num_workers > 1.")
        if not repo_ids:
            raise ValueError("MpBatchLoader requires repo_ids.")

        self._num_workers = int(num_workers)
        self._prefetch = max(1, int(prefetch_batches))
        self._image_size = int(image_size)
        # Split total cache across workers; cap per worker to limit RAM blow-up.
        per_worker = max(4096, int(image_cache_size) // max(1, self._num_workers))
        self._worker_cache = min(per_worker, 16384)
        self._ctx = mp.get_context("spawn")
        self._task_q = self._ctx.Queue(maxsize=max(self._num_workers * 2, self._prefetch * 2))
        self._result_q = self._ctx.Queue(maxsize=max(self._num_workers * 2, self._prefetch * 2))
        init = {
            "repo_ids": repo_ids,
            "image_size": self._image_size,
            "cache_size": self._worker_cache,
            "pair_threads": int(pair_threads),
            "keys": keys,
            "records": tuple(records),
            "tactile_history": int(tactile_history),
            "history_stride": int(history_stride),
        }
        self._workers = [
            self._ctx.Process(
                target=_worker_loop,
                args=(self._task_q, self._result_q, init),
                name=f"tactile-clip-loader-{i}",
                daemon=True,
            )
            for i in range(self._num_workers)
        ]
        self._started = False

    def start(self) -> None:
        if self._started:
            return
        old_jax = os.environ.get("JAX_PLATFORMS")
        old_cuda = os.environ.get("CUDA_VISIBLE_DEVICES")
        os.environ["JAX_PLATFORMS"] = "cpu"
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        try:
            for proc in self._workers:
                proc.start()
        finally:
            if old_jax is None:
                os.environ.pop("JAX_PLATFORMS", None)
            else:
                os.environ["JAX_PLATFORMS"] = old_jax
            if old_cuda is None:
                os.environ.pop("CUDA_VISIBLE_DEVICES", None)
            else:
                os.environ["CUDA_VISIBLE_DEVICES"] = old_cuda
        self._started = True
        print(
            f"mp loader started: workers={self._num_workers} "
            f"per_worker_cache={self._worker_cache} prefetch={self._prefetch}",
            flush=True,
        )

    def close(self) -> None:
        if not self._started:
            return
        for _ in self._workers:
            try:
                self._task_q.put(_STOP, timeout=5)
            except Exception:  # noqa: BLE001
                pass
        for proc in self._workers:
            proc.join(timeout=30)
            if proc.is_alive():
                proc.terminate()
                proc.join(timeout=5)
        self._started = False

    def iter_batches(
        self,
        *,
        sample_index: Sequence[tuple[int, int]],
        order: np.ndarray,
        starts: Sequence[int],
        batch_size: int,
    ) -> Iterator[dict[str, np.ndarray]]:
        if not self._started:
            self.start()

        def _selected_for(start: int) -> list[tuple[int, int]]:
            return [sample_index[int(index)] for index in order[start : start + batch_size]]

        pending: set[int] = set()
        next_to_submit = 0
        next_to_yield = 0
        buffered: dict[int, dict[str, np.ndarray]] = {}

        while next_to_yield < len(starts):
            while next_to_submit < len(starts) and len(pending) < self._prefetch:
                start = starts[next_to_submit]
                self._task_q.put((next_to_submit, _selected_for(start)))
                pending.add(next_to_submit)
                next_to_submit += 1

            batch_id, batch, err = self._result_q.get()
            if batch_id == "__init__":
                raise RuntimeError(err or "worker init failed")
            if err is not None:
                raise RuntimeError(f"mp loader batch {batch_id} failed:\n{err}")
            if batch is None:
                raise RuntimeError(f"mp loader batch {batch_id} returned empty batch")
            batch_id_i = int(batch_id)
            buffered[batch_id_i] = batch
            pending.discard(batch_id_i)

            while next_to_yield in buffered:
                yield buffered.pop(next_to_yield)
                next_to_yield += 1


def iter_mp_batches(
    *,
    repo_ids: str | Sequence[str],
    records: Sequence[FutureRecord],
    keys: DataKeys,
    sample_index: Sequence[tuple[int, int]],
    order: np.ndarray,
    starts: Sequence[int],
    batch_size: int,
    image_size: int,
    image_cache_size: int,
    num_workers: int,
    prefetch_batches: int,
    pair_threads: int = 8,
    loader: MpBatchLoader | None = None,
    tactile_history: int = 0,
    history_stride: int = 5,
) -> Iterator[dict[str, np.ndarray]]:
    """Yield stacked batches using a spawn process pool.

    If ``loader`` is provided it must already be started (or will be on first use)
    and is not closed by this function. Otherwise a one-shot loader is created and
    shut down when iteration finishes.
    """

    owns_loader = loader is None
    if loader is None:
        loader = MpBatchLoader(
            repo_ids=repo_ids,
            records=records,
            keys=keys,
            image_size=image_size,
            image_cache_size=image_cache_size,
            num_workers=num_workers,
            prefetch_batches=prefetch_batches,
            pair_threads=pair_threads,
            tactile_history=tactile_history,
            history_stride=history_stride,
        )
        loader.start()
    try:
        yield from loader.iter_batches(
            sample_index=sample_index,
            order=order,
            starts=starts,
            batch_size=batch_size,
        )
    finally:
        if owns_loader:
            loader.close()
