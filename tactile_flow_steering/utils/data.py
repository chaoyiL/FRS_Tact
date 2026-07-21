"""Online tactile conditioning: frozen ResNet window features for a trainable GRU."""

from __future__ import annotations

import pathlib
from collections.abc import Iterator, Sequence
from concurrent.futures import ThreadPoolExecutor
from typing import Literal

import jax
import jax.numpy as jnp
import numpy as np

from tactile_encoder.utils.checkpoint import TactileEncoderBundle
from tactile_encoder.utils.checkpoint import load_tactile_encoder
from tactile_encoder.utils.image_dataset import create_image_dataset
from tactile_encoder.utils.model import encode_resnet18
from tactile_encoder.utils.model import tactile_clip_config_from_dict
from tactile_encoder.utils.prefetch import prefetch_iterator
from tactile_flow_steering.utils.mp_batches import MpTactileWindowLoader
from utils.cache import CachedPairs

Array = jax.Array
SplitName = Literal["train", "val"]
LossMode = Literal["gt", "gated"]

TACTILE_KEYS = (
    "tactile_left_0",
    "tactile_right_0",
    "tactile_left_1",
    "tactile_right_1",
)
NUM_TACTILE_STREAMS = len(TACTILE_KEYS)


def resolve_tactile_window(*, action_horizon: int, window_divisor: int) -> int:
    """Return window = action_horizon // window_divisor (must divide evenly)."""

    if action_horizon <= 0:
        raise ValueError(f"action_horizon must be positive, got {action_horizon}.")
    if window_divisor <= 0:
        raise ValueError(f"window_divisor must be positive, got {window_divisor}.")
    if action_horizon % window_divisor != 0:
        raise ValueError(
            f"action_horizon ({action_horizon}) must be divisible by "
            f"window_divisor ({window_divisor})."
        )
    window = action_horizon // window_divisor
    if window <= 0:
        raise ValueError(f"Resolved tactile window must be positive, got {window}.")
    return window


def resnet_embedding_dim_from_encoder(bundle: TactileEncoderBundle) -> int:
    config = tactile_clip_config_from_dict(bundle.metadata["tactile_clip_config"])
    return int(config.embedding_dim)


def resolve_dataset_repo_id(
    pairs: CachedPairs,
    *,
    dataset_repo_id: str | None = None,
) -> str:
    if dataset_repo_id is not None:
        return dataset_repo_id
    configuration = pairs.manifest.get("configuration") or {}
    repo_id = configuration.get("dataset_repo_id")
    if not repo_id:
        raise ValueError(
            "Cache manifest is missing configuration.dataset_repo_id; "
            "pass --dataset-repo-id explicitly."
        )
    if isinstance(repo_id, list):
        if len(repo_id) != 1:
            raise ValueError(
                f"Expected a single dataset_repo_id in cache manifest, got {repo_id!r}."
            )
        return str(repo_id[0])
    return str(repo_id)


def resolve_dataset_root(
    pairs: CachedPairs,
    *,
    dataset_root: pathlib.Path | None = None,
) -> pathlib.Path | None:
    if dataset_root is not None:
        return dataset_root
    configuration = pairs.manifest.get("configuration") or {}
    root = configuration.get("dataset_root")
    return pathlib.Path(root) if root else None


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
    thread pool (video decode releases the GIL). Returns
    ``[B, T, 4, H, W, C]`` as float32 ``[0, 1]`` or uint8.
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


def _l2_normalize(vectors: np.ndarray, *, eps: float = 1e-8) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=-1, keepdims=True)
    return vectors / np.maximum(norms, eps)


def tactile_change_from_tokens(
    current_tokens: np.ndarray,
    baseline_tokens: np.ndarray,
) -> np.ndarray:
    """Per-sample tactile change ``s = mean_i(1 - cos)`` for tokens ``[B, 4, D]``."""

    if current_tokens.ndim != 3 or baseline_tokens.ndim != 3:
        raise ValueError(
            f"Expected tokens [B, 4, D], got current={current_tokens.shape}, "
            f"baseline={baseline_tokens.shape}."
        )
    if current_tokens.shape != baseline_tokens.shape:
        raise ValueError(
            f"current/baseline shape mismatch: {current_tokens.shape} vs {baseline_tokens.shape}."
        )
    current_n = _l2_normalize(current_tokens.astype(np.float32))
    baseline_n = _l2_normalize(baseline_tokens.astype(np.float32))
    cosine = np.sum(current_n * baseline_n, axis=-1)  # [B, 4]
    return np.mean(1.0 - cosine, axis=-1).astype(np.float32)


def gate_weights_from_change(
    change: np.ndarray,
    *,
    tau: float,
    temperature: float,
) -> np.ndarray:
    """``w(s) = sigmoid((s - tau) / T)``."""

    if temperature <= 0:
        raise ValueError(f"temperature must be positive, got {temperature}.")
    logits = (np.asarray(change, dtype=np.float32) - float(tau)) / float(temperature)
    return (1.0 / (1.0 + np.exp(-logits))).astype(np.float32)


class TactileConditionedBatches:
    """Yields (indices, x_base, predicted, gt_action, tactile_seq) with frozen ResNet features.

    ``tactile_seq`` has shape ``[B, T, 4, D]`` (oldest→newest; 4 sensor streams).
    """

    def __init__(
        self,
        pairs: CachedPairs,
        *,
        tactile_encoder_dir: pathlib.Path,
        tactile_window: int,
        dataset_repo_id: str | None = None,
        dataset_root: pathlib.Path | None = None,
        image_cache_size: int = 4096,
        history_stride: int = 1,
        encode_batch_size: int = 64,
        build_episode_baselines: bool = False,
        num_workers: int = 4,
        prefetch_batches: int = 4,
        load_threads: int = 8,
        pipeline_prefetch: int = 2,
    ):
        if tactile_window <= 0:
            raise ValueError(f"tactile_window must be positive, got {tactile_window}.")
        if history_stride <= 0:
            raise ValueError(f"history_stride must be positive, got {history_stride}.")
        if encode_batch_size <= 0:
            raise ValueError(f"encode_batch_size must be positive, got {encode_batch_size}.")
        if num_workers < 0:
            raise ValueError(f"num_workers must be non-negative, got {num_workers}.")
        if prefetch_batches <= 0:
            raise ValueError(f"prefetch_batches must be positive, got {prefetch_batches}.")
        if load_threads <= 0:
            raise ValueError(f"load_threads must be positive, got {load_threads}.")
        if pipeline_prefetch <= 0:
            raise ValueError(f"pipeline_prefetch must be positive, got {pipeline_prefetch}.")

        self.pairs = pairs
        self.bundle = load_tactile_encoder(tactile_encoder_dir)
        self.tactile_window = int(tactile_window)
        self.history_stride = int(history_stride)
        self.encode_batch_size = int(encode_batch_size)
        self.num_workers = int(num_workers)
        self.prefetch_batches = int(prefetch_batches)
        self.load_threads = int(load_threads)
        self.pipeline_prefetch = int(pipeline_prefetch)
        self.image_cache_size = int(image_cache_size)
        config = tactile_clip_config_from_dict(self.bundle.metadata["tactile_clip_config"])
        self.resnet_embedding_dim = int(config.embedding_dim)
        self.image_size = int(config.tactile_image_size)
        if "tactile_resnet" not in self.bundle.params:
            raise KeyError("Tactile encoder checkpoint is missing tactile_resnet params.")

        self.repo_id = resolve_dataset_repo_id(pairs, dataset_repo_id=dataset_repo_id)
        del dataset_root
        self.dataset = create_image_dataset(
            self.repo_id,
            image_size=self.image_size,
            cache_size=image_cache_size,
        ).dataset
        self.tactile_keys = TACTILE_KEYS
        self.episode_baselines: dict[int, np.ndarray] = {}
        self._mp_loader: MpTactileWindowLoader | None = None
        if self.num_workers > 1:
            self._mp_loader = MpTactileWindowLoader(
                repo_id=self.repo_id,
                image_size=self.image_size,
                image_cache_size=self.image_cache_size,
                tactile_window=self.tactile_window,
                history_stride=self.history_stride,
                num_workers=self.num_workers,
                prefetch_batches=self.prefetch_batches,
                load_threads=self.load_threads,
            )
            self._mp_loader.start()
        if build_episode_baselines:
            self.build_episode_baseline_embeddings()

    def close(self) -> None:
        if self._mp_loader is not None:
            self._mp_loader.close()
            self._mp_loader = None

    def _encode_images_frozen(self, images: np.ndarray | Array) -> Array:
        """images: [M, H, W, C] → [M, D] with stop_gradient."""

        embeddings, _ = encode_resnet18(
            self.bundle.params["tactile_resnet"],
            jnp.asarray(images, dtype=jnp.float32),
            train=False,
            embedding_dim=self.resnet_embedding_dim,
        )
        return jax.lax.stop_gradient(embeddings)

    def _encode_window_images(self, batch_images: np.ndarray) -> Array:
        """``[B, T, 4, H, W, C]`` → ``[B, T, 4, D]`` via frozen ResNet microbatches."""

        if batch_images.dtype == np.uint8:
            batch_images = batch_images.astype(np.float32) / 255.0
        else:
            batch_images = np.asarray(batch_images, dtype=np.float32)
        batch_size, time_steps, num_streams = batch_images.shape[:3]
        flat = batch_images.reshape(
            (batch_size * time_steps * num_streams,) + batch_images.shape[3:]
        )
        encoded_parts: list[Array] = []
        for start in range(0, flat.shape[0], self.encode_batch_size):
            encoded_parts.append(
                self._encode_images_frozen(flat[start : start + self.encode_batch_size])
            )
        encoded = jnp.concatenate(encoded_parts, axis=0)
        return encoded.reshape(batch_size, time_steps, num_streams, self.resnet_embedding_dim)

    def _encode_frame_streams(self, frame_index: int) -> np.ndarray:
        """Encode one frame's four tactile images → ``[4, D]``."""

        images = self.dataset.get_images(int(frame_index), self.tactile_keys, as_float=True)
        stacked = _frame_streams_from_images(images, self.tactile_keys).astype(np.float32)
        encoded = self._encode_images_frozen(stacked)
        return np.asarray(encoded, dtype=np.float32)

    def build_episode_baseline_embeddings(self) -> dict[int, np.ndarray]:
        """Precompute episode-first-frame ResNet tokens for all cache episodes."""

        episode_indices = np.unique(np.asarray(self.pairs.arrays["episode_index"], dtype=np.int64))
        baselines: dict[int, np.ndarray] = {}
        for episode_index in episode_indices.tolist():
            episode_frames = self.dataset.indices_for_episode(int(episode_index))
            if not episode_frames:
                raise ValueError(f"Episode {episode_index} has no frames.")
            baselines[int(episode_index)] = self._encode_frame_streams(int(episode_frames[0]))
        self.episode_baselines = baselines
        print(f"episode_baselines={len(baselines)} (first-frame ResNet tokens)")
        return baselines

    def gate_weights_for_cache_indices(
        self,
        cache_indices: Sequence[int],
        current_tokens: np.ndarray | Array,
        *,
        tau: float,
        temperature: float,
    ) -> np.ndarray:
        """Compute ``w(s)`` using current-frame tokens vs episode baselines."""

        if not self.episode_baselines:
            raise RuntimeError(
                "Episode baselines are empty; call build_episode_baseline_embeddings() first."
            )
        current = np.asarray(current_tokens, dtype=np.float32)
        if current.ndim != 3 or current.shape[1] != NUM_TACTILE_STREAMS:
            raise ValueError(f"Expected current_tokens [B, 4, D], got {current.shape}.")
        baselines = []
        arrays = self.pairs.arrays
        for cache_index in cache_indices:
            episode_index = int(arrays["episode_index"][cache_index])
            if episode_index not in self.episode_baselines:
                raise KeyError(f"Missing episode baseline for episode_index={episode_index}.")
            baselines.append(self.episode_baselines[episode_index])
        baseline_batch = np.stack(baselines, axis=0)
        change = tactile_change_from_tokens(current, baseline_batch)
        return gate_weights_from_change(change, tau=tau, temperature=temperature)

    def _samples_for_cache_indices(
        self, cache_indices: Sequence[int]
    ) -> list[tuple[int, int]]:
        arrays = self.pairs.arrays
        return [
            (
                int(arrays["dataset_index"][cache_index]),
                int(arrays["episode_index"][cache_index]),
            )
            for cache_index in cache_indices
        ]

    def load_window_images(self, cache_indices: Sequence[int]) -> np.ndarray:
        """Load uint8 tactile windows ``[B, T, 4, H, W, C]`` in-process."""

        return load_tactile_windows(
            self.dataset,
            self._samples_for_cache_indices(cache_indices),
            tactile_window=self.tactile_window,
            history_stride=self.history_stride,
            tactile_keys=self.tactile_keys,
            load_threads=self.load_threads,
            as_float=False,
        )

    def encode_cache_indices(self, cache_indices: Sequence[int]) -> Array:
        """Load temporal windows for cache rows and return ``[B, T, 4, D]``."""

        if len(cache_indices) == 0:
            raise ValueError("cache_indices must be non-empty.")
        return self._encode_window_images(self.load_window_images(cache_indices))

    def encode_window_images(self, batch_images: np.ndarray) -> Array:
        """Encode a preloaded image batch ``[B, T, 4, H, W, C]`` → ``[B, T, 4, D]``."""

        return self._encode_window_images(batch_images)

    def _iter_pair_batches(
        self,
        split: SplitName,
        *,
        batch_size: int,
        shuffle: bool,
        seed: int,
    ) -> Iterator[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
        yield from self.pairs.batches(split, batch_size=batch_size, shuffle=shuffle, seed=seed)

    def batches(
        self,
        split: SplitName,
        *,
        batch_size: int,
        shuffle: bool,
        seed: int,
    ) -> Iterator[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Array]]:
        pair_batches = list(
            self._iter_pair_batches(split, batch_size=batch_size, shuffle=shuffle, seed=seed)
        )
        if not pair_batches:
            return

        if self._mp_loader is not None:
            sample_batches = [
                self._samples_for_cache_indices(indices) for indices, *_ in pair_batches
            ]

            def _produce() -> Iterator[
                tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
            ]:
                for pair_batch, images in zip(
                    pair_batches,
                    self._mp_loader.iter_image_batches(sample_batches),
                    strict=True,
                ):
                    indices, x_base, predicted, gt_action = pair_batch
                    yield indices, x_base, predicted, gt_action, images

            image_iter = prefetch_iterator(_produce(), buffer_size=self.pipeline_prefetch)
            for indices, x_base, predicted, gt_action, images in image_iter:
                tactile_seq = self._encode_window_images(images)
                yield indices, x_base, predicted, gt_action, tactile_seq
            return

        def _produce_local() -> Iterator[
            tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        ]:
            for indices, x_base, predicted, gt_action in pair_batches:
                images = self.load_window_images(indices)
                yield indices, x_base, predicted, gt_action, images

        for indices, x_base, predicted, gt_action, images in prefetch_iterator(
            _produce_local(),
            buffer_size=self.pipeline_prefetch,
        ):
            tactile_seq = self._encode_window_images(images)
            yield indices, x_base, predicted, gt_action, tactile_seq
