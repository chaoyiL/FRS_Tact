"""JAX-free tactile window I/O for mp workers.

Kept separate from ``data.py`` so spawn workers never import JAX/CUDA plugins.
"""

from __future__ import annotations

from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor

import numpy as np

TACTILE_KEYS = (
    "tactile_left_0",
    "tactile_right_0",
    "tactile_left_1",
    "tactile_right_1",
)
NUM_TACTILE_STREAMS = len(TACTILE_KEYS)


def window_frame_indices(
    dataset,
    *,
    dataset_index: int,
    episode_index: int,
    window: int,
    history_stride: int = 1,
) -> tuple[int, ...]:
    """Oldest→newest frame indices of length ``window``, clamped to episode start."""

    if window <= 0:
        raise ValueError(f"window must be positive, got {window}.")
    if history_stride <= 0:
        raise ValueError(f"history_stride must be positive, got {history_stride}.")
    episode_frames = dataset.indices_for_episode(int(episode_index))
    if not episode_frames:
        raise ValueError(f"Episode {episode_index} has no frames.")
    episode_start = int(episode_frames[0])
    current = int(dataset_index)
    if window == 1:
        return (current,)
    past = tuple(
        max(episode_start, current - step * history_stride)
        for step in range(window - 1, 0, -1)
    )
    return past + (current,)


def _frame_streams_from_images(
    images: dict[str, np.ndarray],
    tactile_keys: Sequence[str],
) -> np.ndarray:
    return np.stack([np.asarray(images[key]) for key in tactile_keys], axis=0)


def load_tactile_windows(
    dataset,
    samples: Sequence[tuple[int, int]],
    *,
    tactile_window: int,
    history_stride: int,
    tactile_keys: Sequence[str] = TACTILE_KEYS,
    load_threads: int = 8,
    as_float: bool = False,
) -> np.ndarray:
    """Decode tactile windows for ``(dataset_index, episode_index)`` samples.

    Deduplicates frame indices within the batch and loads unique frames in a
    thread pool. Returns ``[B, T, 4, H, W, C]`` as float32 ``[0, 1]`` or uint8.
    """

    if len(samples) == 0:
        raise ValueError("samples must be non-empty.")
    if load_threads <= 0:
        raise ValueError(f"load_threads must be positive, got {load_threads}.")

    window_indices: list[tuple[int, ...]] = []
    unique_frames: list[int] = []
    seen: set[int] = set()
    for dataset_index, episode_index in samples:
        frames = window_frame_indices(
            dataset,
            dataset_index=int(dataset_index),
            episode_index=int(episode_index),
            window=tactile_window,
            history_stride=history_stride,
        )
        window_indices.append(frames)
        for frame_index in frames:
            if frame_index not in seen:
                seen.add(frame_index)
                unique_frames.append(int(frame_index))

    def _load_one(frame_index: int) -> tuple[int, np.ndarray]:
        images = dataset.get_images(int(frame_index), tactile_keys, as_float=as_float)
        stacked = _frame_streams_from_images(images, tactile_keys)
        if as_float:
            stacked = stacked.astype(np.float32, copy=False)
        else:
            stacked = np.asarray(stacked, dtype=np.uint8)
        return int(frame_index), stacked

    decoded: dict[int, np.ndarray] = {}
    if load_threads == 1 or len(unique_frames) <= 1:
        for frame_index in unique_frames:
            key, value = _load_one(frame_index)
            decoded[key] = value
    else:
        workers = min(load_threads, len(unique_frames))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for frame_index, stacked in pool.map(_load_one, unique_frames, chunksize=4):
                decoded[frame_index] = stacked

    windows = []
    for frames in window_indices:
        windows.append(np.stack([decoded[frame_index] for frame_index in frames], axis=0))
    return np.stack(windows, axis=0)
