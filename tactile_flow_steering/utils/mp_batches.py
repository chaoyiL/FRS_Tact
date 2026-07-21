"""Spawn-based multiprocess tactile-window loading for flow steering.

Workers each open their own ImageOnlyLeRobotDataset so the parent JAX/CUDA
process never forks after GPU init and never pickles dataset handles.

Workers only decode video frames to uint8 tensors; frozen ResNet encoding stays
in the parent process.
"""

from __future__ import annotations

import os
import traceback
from collections.abc import Iterator, Sequence
from typing import Any

import numpy as np

# Sentinel: stop workers.
_STOP = None


def _worker_loop(task_q: Any, result_q: Any, init: dict[str, Any]) -> None:
    """Top-level worker entry (must be picklable for spawn)."""

    os.environ["JAX_PLATFORMS"] = "cpu"
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["TACTILE_IO_LIGHT_IMPORT"] = "1"
    import importlib
    import sys

    try:
        from tactile_encoder.utils.image_dataset import create_image_dataset

        window_io = importlib.import_module("tactile_flow_steering.utils.window_io")
        TACTILE_KEYS = window_io.TACTILE_KEYS
        load_tactile_windows = window_io.load_tactile_windows

        repo_id = init["repo_id"]
        image_size = int(init["image_size"])
        cache_size = int(init["cache_size"])
        tactile_window = int(init["tactile_window"])
        history_stride = int(init["history_stride"])
        load_threads = max(1, int(init.get("load_threads", 8)))
        print(f"worker pid={os.getpid()} opening dataset {repo_id!r}...", flush=True)
        dataset = create_image_dataset(
            repo_id,
            image_size=image_size,
            cache_size=cache_size,
        ).dataset
        if "jax" in sys.modules:
            raise RuntimeError(
                "light import path unexpectedly pulled jax into an mp worker"
            )
        print(f"worker pid={os.getpid()} ready (jax_imported=False)", flush=True)
    except Exception as exc:  # noqa: BLE001 — surface init failure to parent
        result_q.put(("__init__", None, f"worker init failed: {exc}\n{traceback.format_exc()}"))
        return

    while True:
        item = task_q.get()
        if item is _STOP:
            break
        batch_id, samples = item
        try:
            images = load_tactile_windows(
                dataset,
                samples,
                tactile_window=tactile_window,
                history_stride=history_stride,
                tactile_keys=TACTILE_KEYS,
                load_threads=load_threads,
                as_float=False,
            )
            result_q.put((batch_id, images, None))
        except Exception as exc:  # noqa: BLE001 — return error to parent
            result_q.put((batch_id, None, f"{exc}\n{traceback.format_exc()}"))


class MpTactileWindowLoader:
    """Persistent spawn workers that decode tactile windows across epochs."""

    def __init__(
        self,
        *,
        repo_id: str,
        image_size: int,
        image_cache_size: int,
        tactile_window: int,
        history_stride: int,
        num_workers: int,
        prefetch_batches: int,
        load_threads: int = 8,
    ):
        import multiprocessing as mp

        if num_workers <= 1:
            raise ValueError("MpTactileWindowLoader requires num_workers > 1.")
        if not repo_id:
            raise ValueError("MpTactileWindowLoader requires repo_id.")

        self._num_workers = int(num_workers)
        self._prefetch = max(1, int(prefetch_batches))
        self._image_size = int(image_size)
        # Split LRU budget across workers; keep a small floor so tiny configs still cache.
        per_worker = max(256, int(image_cache_size) // max(1, self._num_workers))
        self._worker_cache = min(per_worker, 8192)
        self._ctx = mp.get_context("spawn")
        self._task_q = self._ctx.Queue(maxsize=max(self._num_workers * 2, self._prefetch * 2))
        self._result_q = self._ctx.Queue(maxsize=max(self._num_workers * 2, self._prefetch * 2))
        init = {
            "repo_id": repo_id,
            "image_size": self._image_size,
            "cache_size": self._worker_cache,
            "tactile_window": int(tactile_window),
            "history_stride": int(history_stride),
            "load_threads": int(load_threads),
        }
        self._workers = [
            self._ctx.Process(
                target=_worker_loop,
                args=(self._task_q, self._result_q, init),
                name=f"tactile-flow-loader-{i}",
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
            f"mp tactile loader started: workers={self._num_workers} "
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

    def iter_image_batches(
        self,
        batch_samples: Sequence[Sequence[tuple[int, int]]],
    ) -> Iterator[np.ndarray]:
        """Yield uint8 windows ``[B, T, 4, H, W, C]`` for each sample batch."""

        if not self._started:
            self.start()

        pending: set[int] = set()
        next_to_submit = 0
        next_to_yield = 0
        buffered: dict[int, np.ndarray] = {}
        total = len(batch_samples)

        while next_to_yield < total:
            while next_to_submit < total and len(pending) < self._prefetch:
                samples = list(batch_samples[next_to_submit])
                self._task_q.put((next_to_submit, samples))
                pending.add(next_to_submit)
                next_to_submit += 1

            batch_id, images, err = self._result_q.get()
            if batch_id == "__init__":
                raise RuntimeError(err or "worker init failed")
            if err is not None:
                raise RuntimeError(f"mp tactile loader batch {batch_id} failed:\n{err}")
            if images is None:
                raise RuntimeError(f"mp tactile loader batch {batch_id} returned empty batch")
            batch_id_i = int(batch_id)
            buffered[batch_id_i] = images
            pending.discard(batch_id_i)

            while next_to_yield in buffered:
                yield buffered.pop(next_to_yield)
                next_to_yield += 1
