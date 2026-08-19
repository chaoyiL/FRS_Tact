"""Image-only LeRobot loading for tactile CLIP pretraining.

Loads only wrist RGB + tactile frames. Does not load state, actions, or prompts.
"""

from __future__ import annotations

import dataclasses
import importlib
import sys
import threading
import types
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import cv2
import numpy as np

# LRU stores uint8 HWC frames (~0.15MB/image; 6 cams ≈ 0.9MB/frame).
# Float32 would be ~3.6MB/frame; uint8 packs ~4x more frames in the same RAM.
# 4096 full frames ≈ 3.7GB; 16384 ≈ 14.7GB.
DEFAULT_IMAGE_CACHE_SIZE = 4096

# Raw LeRobot feature paths.
RAW_CAMERA0_KEY = "observation.images.camera0"
RAW_CAMERA1_KEY = "observation.images.camera1"
RAW_TACTILE_KEYS: tuple[str, ...] = (
    "observation.images.tactile_left_0",
    "observation.images.tactile_right_0",
    "observation.images.tactile_left_1",
    "observation.images.tactile_right_1",
)
RAW_IMAGE_KEYS: tuple[str, ...] = (RAW_CAMERA0_KEY, RAW_CAMERA1_KEY, *RAW_TACTILE_KEYS)

# Canonical names consumed by tactile CLIP wrist pairing.
CANONICAL_FROM_RAW: dict[str, str] = {
    RAW_CAMERA0_KEY: "left_image",
    RAW_CAMERA1_KEY: "right_image",
    "observation.images.tactile_left_0": "tactile_left_0",
    "observation.images.tactile_right_0": "tactile_right_0",
    "observation.images.tactile_left_1": "tactile_left_1",
    "observation.images.tactile_right_1": "tactile_right_1",
}
RAW_FROM_CANONICAL: dict[str, str] = {canonical: raw for raw, canonical in CANONICAL_FROM_RAW.items()}


def _import_lerobot_dataset_module() -> Any:
    """Import ``LeRobotDataset`` module.

    Mp data workers set ``TACTILE_IO_LIGHT_IMPORT=1`` so we skip
    ``lerobot.datasets.__init__`` (compute_stats → transformers → jax). Parent
    training keeps the normal import path.
    """

    import os

    module_name = "lerobot.datasets.lerobot_dataset"
    if module_name in sys.modules:
        return sys.modules[module_name]

    if os.environ.get("TACTILE_IO_LIGHT_IMPORT") != "1":
        return importlib.import_module(module_name)

    try:
        import lerobot
    except ModuleNotFoundError as exc:
        raise RuntimeError("lerobot is required for tactile CLIP image loading.") from exc

    datasets_pkg = "lerobot.datasets"
    datasets_path = Path(lerobot.__file__).resolve().parent / "datasets"
    if datasets_pkg not in sys.modules:
        pkg = types.ModuleType(datasets_pkg)
        pkg.__path__ = [str(datasets_path)]
        pkg.__file__ = str(datasets_path / "__init__.py")
        pkg.__package__ = datasets_pkg
        sys.modules[datasets_pkg] = pkg
    elif not hasattr(sys.modules[datasets_pkg], "__path__"):
        sys.modules[datasets_pkg].__path__ = [str(datasets_path)]

    return importlib.import_module(module_name)


def resize_with_pad(image: np.ndarray, height: int, width: int) -> np.ndarray:
    """Resize keeping aspect ratio and pad with black to ``(height, width)``.

    Accepts uint8 HWC ``[0, 255]`` or float32 HWC ``[0, 1]``.
    """

    if image.ndim != 3 or image.shape[-1] not in (1, 3):
        raise ValueError(f"Expected HWC image, got shape {image.shape}.")
    cur_h, cur_w = image.shape[:2]
    if cur_h == height and cur_w == width:
        return image
    ratio = max(cur_w / width, cur_h / height)
    resized_h = max(1, int(round(cur_h / ratio)))
    resized_w = max(1, int(round(cur_w / ratio)))
    interpolation = cv2.INTER_AREA if ratio > 1.0 else cv2.INTER_LINEAR
    resized = cv2.resize(image, (resized_w, resized_h), interpolation=interpolation)
    if image.dtype == np.uint8:
        resized = np.clip(np.rint(resized), 0, 255).astype(np.uint8)
        pad_value = 0
    else:
        resized = np.clip(resized, 0.0, 1.0).astype(np.float32)
        pad_value = 0.0
    pad_h0, rem_h = divmod(height - resized_h, 2)
    pad_h1 = pad_h0 + rem_h
    pad_w0, rem_w = divmod(width - resized_w, 2)
    pad_w1 = pad_w0 + rem_w
    return np.pad(
        resized,
        ((pad_h0, pad_h1), (pad_w0, pad_w1), (0, 0)),
        mode="constant",
        constant_values=pad_value,
    )


def _normalize_image_layout(image: Any) -> np.ndarray:
    """Normalize a raw LeRobot image to HWC with 3 channels (dtype unchanged)."""

    array = np.asarray(image)
    if hasattr(array, "detach"):
        array = array.detach().cpu().numpy()
    array = np.asarray(array)
    if array.ndim == 4 and array.shape[0] == 1:
        array = array[0]
    if array.ndim != 3:
        raise ValueError(f"Expected image with 3 dimensions, got shape {array.shape}.")
    if array.shape[0] in (1, 3) and array.shape[-1] not in (1, 3):
        array = np.moveaxis(array, 0, -1)
    if array.shape[-1] == 1:
        array = np.repeat(array, 3, axis=-1)
    if array.shape[-1] != 3:
        raise ValueError(f"Expected RGB image with 3 channels, got shape {array.shape}.")
    return array


def parse_image_to_uint8(image: Any, *, image_size: int) -> np.ndarray:
    """Convert a raw LeRobot image to uint8 HWC in ``[0, 255]`` at ``image_size``.

    Fast path: already-uint8 HWC at the target size skips resize and rescale.
    """

    array = _normalize_image_layout(image)
    if array.dtype == np.uint8 and array.shape[0] == image_size and array.shape[1] == image_size:
        return np.ascontiguousarray(array)

    if np.issubdtype(array.dtype, np.floating):
        array = array.astype(np.float32, copy=False)
        if array.size and float(np.nanmin(array)) < 0.0:
            array = (array + 1.0) * 0.5
        elif array.size and float(np.nanmax(array)) > 1.5:
            array = array / 255.0
        array = np.clip(array, 0.0, 1.0)
        resized = resize_with_pad(array, image_size, image_size)
        return np.clip(np.rint(resized * 255.0), 0, 255).astype(np.uint8)

    # Integer / other non-float: treat as 0..255 then resize as uint8.
    array = np.clip(array, 0, 255).astype(np.uint8, copy=False)
    return np.ascontiguousarray(resize_with_pad(array, image_size, image_size))


def parse_image_to_unit(image: Any, *, image_size: int) -> np.ndarray:
    """Convert a raw LeRobot image to float32 HWC in ``[0, 1]`` at ``image_size``."""

    return parse_image_to_uint8(image, image_size=image_size).astype(np.float32) * (1.0 / 255.0)


def _cached_image_to_unit(image: np.ndarray) -> np.ndarray:
    """Convert a cached uint8/float image to float32 ``[0, 1]``."""

    array = np.asarray(image)
    if array.dtype == np.uint8:
        return array.astype(np.float32) * (1.0 / 255.0)
    return np.asarray(array, dtype=np.float32)


def _unwrap_dataset(dataset: Any) -> Any:
    while hasattr(dataset, "_dataset"):
        dataset = dataset._dataset
    return dataset


def _lerobot_v21_episode_bounds(dataset: Any, episode_index: int) -> tuple[int, int] | None:
    episode_data_index = getattr(dataset, "episode_data_index", None)
    if episode_data_index is None:
        return None
    if "from" not in episode_data_index or "to" not in episode_data_index:
        return None
    starts = episode_data_index["from"]
    ends = episode_data_index["to"]
    if episode_index < 0 or episode_index >= len(starts):
        raise ValueError(
            f"Episode {episode_index} is out of range for this dataset. "
            f"Available episode indices are 0..{len(starts) - 1}."
        )
    return int(np.asarray(starts[episode_index])), int(np.asarray(ends[episode_index]))


def _lerobot_v30_episode_bounds(dataset: Any, episode_index: int) -> tuple[int, int] | None:
    meta = getattr(dataset, "meta", None)
    if meta is None:
        return None
    total_episodes = getattr(meta, "total_episodes", None)
    if total_episodes is None:
        return None
    if episode_index < 0 or episode_index >= total_episodes:
        raise ValueError(
            f"Episode {episode_index} is out of range for this dataset. "
            f"Available episode indices are 0..{total_episodes - 1}."
        )
    episodes = getattr(meta, "episodes", None)
    if episodes is None:
        return None
    try:
        start = int(np.asarray(episodes[episode_index]["dataset_from_index"]))
        end = int(np.asarray(episodes[episode_index]["dataset_to_index"]))
    except (KeyError, IndexError, TypeError):
        return None
    return start, end


def episode_count(dataset: Any) -> int:
    """Number of episodes in an image dataset or underlying LeRobot dataset."""

    if hasattr(dataset, "num_episodes"):
        return int(dataset.num_episodes)

    unwrapped = _unwrap_dataset(dataset)
    episode_data_index = getattr(unwrapped, "episode_data_index", None)
    if episode_data_index is not None and "from" in episode_data_index:
        return len(episode_data_index["from"])
    meta = getattr(unwrapped, "meta", None)
    if meta is not None:
        total = getattr(meta, "total_episodes", None)
        if total is not None:
            return int(total)
    raise ValueError(
        "Dataset does not expose LeRobot episode metadata. "
        "Expected episode_data_index (v2.1) or meta.total_episodes (v3.0)."
    )


def indices_for_episode(dataset: Any, episode_index: int | str) -> tuple[int, ...]:
    """Global dataset indices belonging to one episode."""

    try:
        episode_index = int(episode_index)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Episode {episode_index!r} not found in dataset metadata.") from exc

    if hasattr(dataset, "indices_for_episode"):
        return tuple(int(index) for index in dataset.indices_for_episode(episode_index))

    unwrapped = _unwrap_dataset(dataset)
    for bounds_fn in (_lerobot_v21_episode_bounds, _lerobot_v30_episode_bounds):
        bounds = bounds_fn(unwrapped, episode_index)
        if bounds is None:
            continue
        start, end = bounds
        if end <= start:
            break
        return tuple(range(start, end))
    raise ValueError(f"Episode {episode_index!r} not found in dataset metadata.")


def normalize_repo_ids(repo_ids: str | Sequence[str]) -> tuple[str, ...]:
    """Normalize one or more LeRobot dataset repo ids.

    Accepts a single id, a comma-separated string, or a sequence of ids.
    Multiple ids are concatenated by ``CombinedImageDataset`` for joint training.
    """

    if isinstance(repo_ids, str):
        parts = [part.strip() for part in repo_ids.split(",") if part.strip()]
    else:
        parts = [str(part).strip() for part in repo_ids if str(part).strip()]
    if not parts:
        raise ValueError("At least one dataset repo id is required.")
    return tuple(parts)


class ImageOnlyLeRobotDataset:
    """Random-access wrapper that decodes only requested vision/tactile streams."""

    def __init__(
        self,
        repo_id: str,
        *,
        image_size: int = 224,
        cache_size: int = DEFAULT_IMAGE_CACHE_SIZE,
    ):
        try:
            lerobot_dataset = _import_lerobot_dataset_module()
        except ModuleNotFoundError as exc:
            raise RuntimeError("lerobot is required for tactile CLIP image loading.") from exc

        self.repo_id = repo_id
        self.image_size = int(image_size)
        self._cache_size = max(0, int(cache_size))
        # Cache whole decoded frames by local index so left/right/future views
        # of the same parquet row only pay the decode cost once.
        self._cache: dict[int, dict[str, Any]] = {}
        self._cache_order: list[int] = []
        self._cache_lock = threading.RLock()
        # return_uint8 keeps decoded video frames in uint8 for our LRU cache.
        self._dataset = lerobot_dataset.LeRobotDataset(repo_id, return_uint8=True)
        self._video_keys = set(self._dataset.meta.video_keys)
        self._image_keys = set(self._dataset.meta.image_keys)
        missing = [key for key in RAW_IMAGE_KEYS if key not in self._video_keys and key not in self._image_keys]
        if missing:
            available = sorted(self._video_keys | self._image_keys)
            raise KeyError(
                f"Dataset {repo_id!r} is missing required image keys {missing}. "
                f"Available camera keys: {available}"
            )
        # Scalar columns only — never pull state/actions through select_columns.
        # Bypass LeRobot's hf_transform_to_torch (PIL → torch CHW float32): we only need
        # raw PIL/uint8 and convert via parse_image_to_uint8 ourselves.
        hf = self._dataset.hf_dataset
        hf_raw = hf.with_transform(None) if hasattr(hf, "with_transform") else hf
        if hasattr(hf_raw, "with_format"):
            hf_raw = hf_raw.with_format(None)
        scalar_cols = [col for col in ("episode_index", "timestamp", "index") if col in hf_raw.column_names]
        self._scalars = hf_raw.select_columns(scalar_cols)
        stored_image_cols = [key for key in RAW_IMAGE_KEYS if key in self._image_keys]
        self._stored_images = hf_raw.select_columns(stored_image_cols) if stored_image_cols else None
        self._canonical_image_keys = tuple(
            CANONICAL_FROM_RAW[raw] for raw in RAW_IMAGE_KEYS if raw in self._image_keys or raw in self._video_keys
        )
        self._frame_offset = 0
        self._episode_offset = 0
        self._preloaded = False

    @property
    def meta(self) -> Any:
        return self._dataset.meta

    @property
    def episode_data_index(self) -> Any:
        return getattr(self._dataset, "episode_data_index", None)

    @property
    def num_episodes(self) -> int:
        return episode_count(self._dataset)

    def __len__(self) -> int:
        return len(self._dataset)

    def indices_for_episode(self, episode_index: int) -> tuple[int, ...]:
        local_episode = int(episode_index) - self._episode_offset
        local_indices = indices_for_episode(self._dataset, local_episode)
        return tuple(self._frame_offset + index for index in local_indices)

    def _cache_get(self, local_index: int) -> dict[str, Any] | None:
        value = self._cache.get(local_index)
        if value is None:
            return None
        self._cache_order.remove(local_index)
        self._cache_order.append(local_index)
        return value

    def _cache_put(self, local_index: int, value: dict[str, Any]) -> None:
        if self._cache_size <= 0:
            return
        if local_index in self._cache:
            self._cache_order.remove(local_index)
        self._cache[local_index] = value
        self._cache_order.append(local_index)
        while len(self._cache_order) > self._cache_size:
            old = self._cache_order.pop(0)
            self._cache.pop(old, None)

    def _decode_frame(self, local_index: int, canonical_keys: Sequence[str]) -> dict[str, Any]:
        """Decode only the requested canonical image keys for one parquet/video row.

        Images are returned as uint8 HWC for LRU / preload storage.
        """

        keys = tuple(dict.fromkeys(canonical_keys))
        if not keys:
            raise ValueError("canonical_keys must be non-empty.")
        unknown = [key for key in keys if key not in RAW_FROM_CANONICAL]
        if unknown:
            raise KeyError(f"Unknown image keys {unknown}. Expected one of {sorted(RAW_FROM_CANONICAL)}.")

        row = self._scalars[local_index]
        episode_index = int(np.asarray(row["episode_index"]).item())
        timestamp = float(np.asarray(row["timestamp"]).item())

        raw_keys = tuple(RAW_FROM_CANONICAL[key] for key in keys)
        frames: dict[str, Any] = {}
        video_keys = [key for key in raw_keys if key in self._video_keys]
        if video_keys:
            query_timestamps = {key: [timestamp] for key in video_keys}
            reader = self._dataset._ensure_reader()
            frames.update(reader._query_videos(query_timestamps, episode_index))
        if self._stored_images is not None:
            stored = self._stored_images[local_index]
            for raw_key in raw_keys:
                if raw_key in self._image_keys:
                    frames[raw_key] = stored[raw_key]

        output: dict[str, Any] = {
            "episode_index": self._episode_offset + episode_index,
            "dataset_index": self._frame_offset + local_index,
        }
        missing_raw = [raw_key for raw_key in raw_keys if raw_key not in frames]
        if missing_raw:
            raise KeyError(
                f"Sample local_index={local_index} is missing image keys "
                f"{[CANONICAL_FROM_RAW[k] for k in missing_raw]}."
            )
        for raw_key in raw_keys:
            canonical = CANONICAL_FROM_RAW[raw_key]
            output[canonical] = parse_image_to_uint8(frames[raw_key], image_size=self.image_size)
        return output

    def get_images(
        self,
        index: int,
        keys: Sequence[str],
        *,
        as_float: bool = True,
    ) -> dict[str, Any]:
        """Decode only the requested canonical image keys for one frame.

        LRU entries store uint8 images and may be partial (only previously requested
        cameras). Missing keys are decoded on demand and merged into the cache.
        By default images are float32 in ``[0, 1]``; ``as_float=False`` keeps uint8.
        """

        canonical_keys = tuple(dict.fromkeys(keys))
        if not canonical_keys:
            raise ValueError("keys must be non-empty.")
        unknown = [key for key in canonical_keys if key not in RAW_FROM_CANONICAL]
        if unknown:
            raise KeyError(f"Unknown image keys {unknown}. Expected one of {sorted(RAW_FROM_CANONICAL)}.")

        local_index = int(index) - self._frame_offset
        if local_index < 0 or local_index >= len(self._dataset):
            raise IndexError(index)

        with self._cache_lock:
            cached = self._cache_get(local_index)
            cached_snapshot = None if cached is None else dict(cached)

        missing = (
            list(canonical_keys)
            if cached_snapshot is None
            else [key for key in canonical_keys if key not in cached_snapshot]
        )
        if missing:
            partial = self._decode_frame(local_index, missing)
            with self._cache_lock:
                existing = self._cache_get(local_index)
                if existing is None:
                    merged = partial
                else:
                    merged = dict(existing)
                    for key, value in partial.items():
                        if key in ("episode_index", "dataset_index") or key not in merged:
                            merged[key] = value
                self._cache_put(local_index, merged)
                cached_snapshot = merged

        assert cached_snapshot is not None
        still_missing = [key for key in canonical_keys if key not in cached_snapshot]
        if still_missing:
            raise KeyError(f"Sample {index} is missing image keys {still_missing}.")
        if as_float:
            images = {key: _cached_image_to_unit(cached_snapshot[key]) for key in canonical_keys}
        else:
            images = {
                key: np.ascontiguousarray(cached_snapshot[key], dtype=np.uint8)
                for key in canonical_keys
            }
        return {
            "episode_index": cached_snapshot["episode_index"],
            "dataset_index": cached_snapshot["dataset_index"],
            **images,
        }

    def preload(
        self,
        *,
        num_workers: int = 32,
        store_uint8: bool = True,
        indices: Sequence[int] | None = None,
    ) -> None:
        """Decode frames into RAM so training is no longer parquet-bound.

        ``indices`` are global dataset indices (same as ``get_images``). When
        provided, only those frames are decoded; otherwise every frame is loaded.
        Cache storage is uint8 by default (``store_uint8=True``).
        """

        if indices is None:
            local_indices = list(range(len(self._dataset)))
        else:
            local_indices = sorted(
                {
                    local
                    for index in indices
                    if 0 <= (local := int(index) - self._frame_offset) < len(self._dataset)
                }
            )
        if not local_indices:
            print(f"preloading {self.repo_id}: 0 frames requested, skip", flush=True)
            self._preloaded = True
            return
        if self._preloaded and all(local_index in self._cache for local_index in local_indices):
            return

        total = len(local_indices)
        self._cache_size = max(self._cache_size, total)
        workers = max(1, int(num_workers))
        all_keys = self._canonical_image_keys
        print(
            f"preloading {self.repo_id}: {total}/{len(self._dataset)} frames "
            f"(workers={workers}, uint8={store_uint8})...",
            flush=True,
        )

        def _one(local_index: int) -> tuple[int, dict[str, Any]]:
            frame = self._decode_frame(local_index, all_keys)
            if not store_uint8:
                for key in all_keys:
                    image = frame.get(key)
                    if image is None:
                        continue
                    frame[key] = _cached_image_to_unit(image)
            return local_index, frame

        from concurrent.futures import ThreadPoolExecutor

        done = 0
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for local_index, frame in pool.map(_one, local_indices, chunksize=16):
                with self._cache_lock:
                    self._cache_put(local_index, frame)
                done += 1
                if done % 5000 == 0 or done == total:
                    print(f"  preload {done}/{total}", flush=True)
        self._preloaded = True
        print(f"preloaded {self.repo_id}: {total} frames in RAM", flush=True)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.get_images(index, tuple(CANONICAL_FROM_RAW.values()))


class CombinedImageDataset:
    """Concatenate multiple image-only datasets with global episode/frame indices."""

    def __init__(self, datasets: Sequence[ImageOnlyLeRobotDataset]):
        if not datasets:
            raise ValueError("CombinedImageDataset requires at least one dataset.")
        self._datasets = list(datasets)
        frame_offset = 0
        episode_offset = 0
        self._episode_owners: list[tuple[int, int]] = []  # (dataset_i, local_episode)
        for dataset_i, dataset in enumerate(self._datasets):
            dataset._frame_offset = frame_offset
            dataset._episode_offset = episode_offset
            for local_episode in range(dataset.num_episodes):
                self._episode_owners.append((dataset_i, local_episode))
            frame_offset += len(dataset)
            episode_offset += dataset.num_episodes
        self._total_frames = frame_offset
        self._total_episodes = episode_offset

    @property
    def num_episodes(self) -> int:
        return self._total_episodes

    @property
    def repo_ids(self) -> tuple[str, ...]:
        return tuple(dataset.repo_id for dataset in self._datasets)

    def __len__(self) -> int:
        return self._total_frames

    def indices_for_episode(self, episode_index: int) -> tuple[int, ...]:
        if episode_index < 0 or episode_index >= self._total_episodes:
            raise ValueError(
                f"Episode {episode_index} is out of range. "
                f"Available episode indices are 0..{self._total_episodes - 1}."
            )
        dataset_i, local_episode = self._episode_owners[episode_index]
        dataset = self._datasets[dataset_i]
        return dataset.indices_for_episode(dataset._episode_offset + local_episode)

    def __getitem__(self, index: int) -> dict[str, Any]:
        index = int(index)
        if index < 0 or index >= self._total_frames:
            raise IndexError(index)
        for dataset in self._datasets:
            local = index - dataset._frame_offset
            if 0 <= local < len(dataset):
                return dataset[index]
        raise IndexError(index)

    def get_images(
        self,
        index: int,
        keys: Sequence[str],
        *,
        as_float: bool = True,
    ) -> dict[str, Any]:
        index = int(index)
        if index < 0 or index >= self._total_frames:
            raise IndexError(index)
        for dataset in self._datasets:
            local = index - dataset._frame_offset
            if 0 <= local < len(dataset):
                return dataset.get_images(index, keys, as_float=as_float)
        raise IndexError(index)

    def preload(
        self,
        *,
        num_workers: int = 32,
        store_uint8: bool = True,
        indices: Sequence[int] | None = None,
    ) -> None:
        for dataset in self._datasets:
            dataset.preload(num_workers=num_workers, store_uint8=store_uint8, indices=indices)


@dataclasses.dataclass(frozen=True)
class ImageDatasetInfo:
    config_name: str
    repo_id: str | tuple[str, ...]
    dataset: ImageOnlyLeRobotDataset | CombinedImageDataset


def create_image_dataset(
    repo_ids: str | Sequence[str],
    *,
    image_size: int = 224,
    cache_size: int = DEFAULT_IMAGE_CACHE_SIZE,
    config_name: str | None = None,
) -> ImageDatasetInfo:
    """Build an image-only dataset from one or more LeRobot repo ids."""

    resolved = normalize_repo_ids(repo_ids)
    datasets = [
        ImageOnlyLeRobotDataset(repo_id, image_size=image_size, cache_size=cache_size)
        for repo_id in resolved
    ]
    if len(datasets) == 1:
        dataset: ImageOnlyLeRobotDataset | CombinedImageDataset = datasets[0]
        repo_id: str | tuple[str, ...] = resolved[0]
    else:
        dataset = CombinedImageDataset(datasets)
        repo_id = resolved
    label = config_name or ("+".join(resolved) if len(resolved) > 1 else resolved[0])
    return ImageDatasetInfo(config_name=label, repo_id=repo_id, dataset=dataset)
