from __future__ import annotations

import dataclasses
import pathlib
from collections.abc import Iterator, Sequence
from typing import Any, Literal

import numpy as np

from utils.cache import SampleRecord
from utils.cache import split_episodes
from tactile_encoder.utils.image_dataset import ImageDatasetInfo
from tactile_encoder.utils.image_dataset import create_image_dataset
from tactile_encoder.utils.image_dataset import episode_count
from tactile_encoder.utils.image_dataset import indices_for_episode

# Per-wrist pairing: left wrist camera + tactile_*_0; right wrist camera + tactile_*_1.
DEFAULT_LEFT_TACTILE_IMAGE_KEYS = ("tactile_left_0", "tactile_right_0")
DEFAULT_RIGHT_TACTILE_IMAGE_KEYS = ("tactile_left_1", "tactile_right_1")


@dataclasses.dataclass(frozen=True)
class SideKeys:
    """One wrist side: current/future RGB plus the two tactile sensors on that side."""

    name: str
    current_rgb_key: str
    future_rgb_key: str
    tactile_keys: tuple[str, ...]
    masked_rgb_key: str | None = None


@dataclasses.dataclass(frozen=True)
class DataKeys:
    """Left/right wrist modality key groups used by tactile CLIP pretraining."""

    sides: tuple[SideKeys, ...]

    def __post_init__(self) -> None:
        if len(self.sides) != 2:
            raise ValueError(f"DataKeys.sides must contain exactly two wrist sides, got {len(self.sides)}.")
        counts = {len(side.tactile_keys) for side in self.sides}
        if len(counts) != 1:
            raise ValueError(
                f"All sides must use the same tactile image count, got {[len(s.tactile_keys) for s in self.sides]}."
            )

    @property
    def tactile_image_count(self) -> int:
        return len(self.sides[0].tactile_keys)

    @property
    def masked_rgb_key(self) -> str | None:
        masked = {side.masked_rgb_key for side in self.sides}
        if len(masked) == 1:
            return next(iter(masked))
        return None


def default_wrist_sides(*, masked_rgb_key: str | None = None) -> tuple[SideKeys, ...]:
    return (
        SideKeys(
            name="left",
            current_rgb_key="left_image",
            future_rgb_key="left_image",
            masked_rgb_key=masked_rgb_key,
            tactile_keys=DEFAULT_LEFT_TACTILE_IMAGE_KEYS,
        ),
        SideKeys(
            name="right",
            current_rgb_key="right_image",
            future_rgb_key="right_image",
            masked_rgb_key=masked_rgb_key,
            tactile_keys=DEFAULT_RIGHT_TACTILE_IMAGE_KEYS,
        ),
    )


@dataclasses.dataclass(frozen=True)
class FutureRecord:
    dataset_index: int
    future_dataset_index: int
    episode_index: int
    split: Literal["train", "val"]


@dataclasses.dataclass(frozen=True)
class FutureRecordSet:
    records: tuple[FutureRecord, ...]
    train_episodes: tuple[int, ...]
    val_episodes: tuple[int, ...]

    def split_records(self, split: Literal["train", "val"]) -> tuple[FutureRecord, ...]:
        return tuple(record for record in self.records if record.split == split)


def resolve_data_keys(
    data_config: Any = None,
    *,
    masked_rgb_key: str | None = None,
) -> DataKeys:
    """Return fixed left/right wrist pairings.

      left:  left_image  + tactile_{left,right}_0
      right: right_image + tactile_{left,right}_1
    """

    del data_config  # Kept for call-site compatibility; keys are fixed by wrist pairing.
    return DataKeys(sides=default_wrist_sides(masked_rgb_key=masked_rgb_key))


def history_dataset_indices(
    dataset: Any,
    record: FutureRecord,
    *,
    history: int,
    history_stride: int,
) -> tuple[int, ...]:
    """Return past dataset indices ``[t-s, t-2s, ...]`` clamped to the episode start."""

    if history < 0:
        raise ValueError(f"history must be non-negative, got {history}.")
    if history == 0:
        return ()
    if history_stride <= 0:
        raise ValueError(f"history_stride must be positive, got {history_stride}.")
    episode_indices = indices_for_episode(dataset, int(record.episode_index))
    if not episode_indices:
        raise ValueError(f"Episode {record.episode_index} has no frames.")
    episode_start = int(episode_indices[0])
    current = int(record.dataset_index)
    return tuple(
        max(episode_start, current - step * history_stride) for step in range(1, history + 1)
    )


def _load_frame_images(
    dataset: Any,
    dataset_index: int,
    keys: Sequence[str],
    *,
    as_uint8: bool,
) -> dict[str, Any]:
    if hasattr(dataset, "get_images"):
        return dataset.get_images(int(dataset_index), keys, as_float=not as_uint8)
    return dataset[int(dataset_index)]


def load_pair(
    dataset: Any,
    record: FutureRecord,
    side: SideKeys,
    *,
    image_size: int = 224,
    side_id: int | None = None,
    as_uint8: bool = False,
    tactile_history: int = 0,
    history_stride: int = 5,
) -> dict[str, np.ndarray]:
    """Load one contrastive sample for a single wrist side from an image-only dataset.

    When ``tactile_history == 0``, ``tactile`` has shape ``[num_sensors, H, W, C]``.
    Otherwise it has shape ``[T, num_sensors, H, W, C]`` with time ordered
    oldest → newest (``T = 1 + tactile_history``).
    """

    del image_size  # Images are already resized by ImageOnlyLeRobotDataset.
    if tactile_history < 0:
        raise ValueError(f"tactile_history must be non-negative, got {tactile_history}.")
    current_key = side.masked_rgb_key or side.current_rgb_key
    current_keys = (current_key, *side.tactile_keys)
    future_keys = (side.future_rgb_key,)
    current = _load_frame_images(
        dataset, int(record.dataset_index), current_keys, as_uint8=as_uint8
    )
    future = _load_frame_images(
        dataset, int(record.future_dataset_index), future_keys, as_uint8=as_uint8
    )
    if current_key not in current:
        raise KeyError(
            f"Sample {record.dataset_index} is missing RGB key {current_key!r}. "
            f"Available keys: {sorted(k for k in current if k.endswith('image') or k.startswith('tactile'))}"
        )
    missing_tactile = [key for key in side.tactile_keys if key not in current]
    if missing_tactile:
        raise KeyError(
            f"Sample {record.dataset_index} is missing tactile keys {missing_tactile}. "
            f"Available keys: {sorted(k for k in current if k.startswith('tactile'))}"
        )
    if side.future_rgb_key not in future:
        raise KeyError(
            f"Sample {record.future_dataset_index} is missing RGB key {side.future_rgb_key!r}."
        )
    if side_id is None:
        side_id = 0 if side.name == "left" else 1
    image_dtype = np.uint8 if as_uint8 else np.float32

    def _sensors_at(frame: dict[str, Any]) -> np.ndarray:
        missing = [key for key in side.tactile_keys if key not in frame]
        if missing:
            raise KeyError(f"Missing tactile keys {missing}.")
        return np.stack(
            [np.asarray(frame[key], dtype=image_dtype) for key in side.tactile_keys],
            axis=0,
        )

    current_tactile = _sensors_at(current)
    if tactile_history == 0:
        tactile = current_tactile
    else:
        # history_dataset_indices returns [t-s, t-2s, ...]; reverse to oldest→newest.
        past_indices = history_dataset_indices(
            dataset,
            record,
            history=tactile_history,
            history_stride=history_stride,
        )
        time_frames = []
        for hist_index in reversed(past_indices):
            past = _load_frame_images(dataset, hist_index, side.tactile_keys, as_uint8=as_uint8)
            time_frames.append(_sensors_at(past))
        time_frames.append(current_tactile)
        tactile = np.stack(time_frames, axis=0)

    return {
        "current_rgb": np.asarray(current[current_key], dtype=image_dtype),
        "future_rgb": np.asarray(future[side.future_rgb_key], dtype=image_dtype),
        "tactile": tactile,
        "dataset_index": np.asarray(record.dataset_index, dtype=np.int64),
        "future_dataset_index": np.asarray(record.future_dataset_index, dtype=np.int64),
        "episode_index": np.asarray(record.episode_index, dtype=np.int64),
        "side_id": np.asarray(side_id, dtype=np.int64),
    }


_UINT8_IMAGE_KEYS = frozenset({"current_rgb", "future_rgb", "tactile"})


def batch_uint8_to_float32(batch: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Convert uint8 image tensors from workers to float32 ``[0, 1]`` for JAX."""

    out: dict[str, np.ndarray] = {}
    inv = np.float32(1.0 / 255.0)
    for key, value in batch.items():
        if key in _UINT8_IMAGE_KEYS and value.dtype == np.uint8:
            out[key] = value.astype(np.float32) * inv
        else:
            out[key] = value
    return out


def load_record_pairs(
    dataset: Any,
    record: FutureRecord,
    keys: DataKeys,
    *,
    image_size: int = 224,
    tactile_history: int = 0,
    history_stride: int = 5,
) -> list[dict[str, np.ndarray]]:
    """Expand one temporal record into one sample per wrist side."""

    return [
        load_pair(
            dataset,
            record,
            side,
            image_size=image_size,
            side_id=side_i,
            tactile_history=tactile_history,
            history_stride=history_stride,
        )
        for side_i, side in enumerate(keys.sides)
    ]


def _build_base_records(
    dataset: Any,
    *,
    val_fraction: float,
    split_seed: int,
    frame_stride: int,
) -> tuple[list[SampleRecord], tuple[int, ...], tuple[int, ...]]:
    if frame_stride <= 0:
        raise ValueError(f"frame_stride must be positive, got {frame_stride}.")
    total_episodes = episode_count(dataset)
    episodes = list(range(total_episodes))
    train_episodes, val_episodes = split_episodes(episodes, val_fraction=val_fraction, seed=split_seed)
    val_set = set(val_episodes)

    records: list[SampleRecord] = []
    for episode_index in episodes:
        split = "val" if episode_index in val_set else "train"
        dataset_indices = indices_for_episode(dataset, episode_index)[::frame_stride]
        records.extend(SampleRecord(int(index), episode_index, split) for index in dataset_indices)
    if not records:
        raise ValueError("Dataset selection produced no samples.")
    return records, train_episodes, val_episodes


def build_future_records(
    dataset: Any,
    *,
    future_offset: int = 1,
    val_fraction: float = 0.1,
    split_seed: int = 0,
    frame_stride: int = 1,
) -> FutureRecordSet:
    if future_offset <= 0:
        raise ValueError(f"future_offset must be positive, got {future_offset}.")

    base_records, train_episodes, val_episodes = _build_base_records(
        dataset,
        val_fraction=val_fraction,
        split_seed=split_seed,
        frame_stride=frame_stride,
    )
    episode_indices = {
        episode: tuple(int(index) for index in indices_for_episode(dataset, episode))
        for episode in sorted({record.episode_index for record in base_records})
    }
    positions = {
        episode: {dataset_index: offset for offset, dataset_index in enumerate(indices)}
        for episode, indices in episode_indices.items()
    }

    future_records: list[FutureRecord] = []
    for record in base_records:
        indices = episode_indices[int(record.episode_index)]
        position = positions[int(record.episode_index)][int(record.dataset_index)]
        future_position = position + future_offset
        if future_position >= len(indices):
            continue
        future_index = int(indices[future_position])
        future_records.append(
            FutureRecord(
                dataset_index=int(record.dataset_index),
                future_dataset_index=future_index,
                episode_index=int(record.episode_index),
                split=record.split,
            )
        )
    if not any(record.split == "train" for record in future_records):
        raise ValueError("Dataset selection produced no training samples.")
    if not any(record.split == "val" for record in future_records):
        raise ValueError("Dataset selection produced no validation samples.")
    return FutureRecordSet(
        records=tuple(future_records),
        train_episodes=train_episodes,
        val_episodes=val_episodes,
    )


def stack_pairs(
    pairs: Sequence[dict[str, np.ndarray]],
    *,
    as_uint8: bool = False,
) -> dict[str, np.ndarray]:
    image_dtype = np.uint8 if as_uint8 else np.float32
    stacked = {
        "current_rgb": np.stack([pair["current_rgb"] for pair in pairs], axis=0).astype(
            image_dtype, copy=False
        ),
        "future_rgb": np.stack([pair["future_rgb"] for pair in pairs], axis=0).astype(
            image_dtype, copy=False
        ),
        "tactile": np.stack([pair["tactile"] for pair in pairs], axis=0).astype(image_dtype, copy=False),
        "dataset_index": np.asarray([pair["dataset_index"] for pair in pairs], dtype=np.int64),
        "future_dataset_index": np.asarray([pair["future_dataset_index"] for pair in pairs], dtype=np.int64),
        "episode_index": np.asarray([pair["episode_index"] for pair in pairs], dtype=np.int64),
    }
    if all("side_id" in pair for pair in pairs):
        stacked["side_id"] = np.asarray([pair["side_id"] for pair in pairs], dtype=np.int64)
    return stacked


def batches(
    dataset: Any,
    records: Sequence[FutureRecord],
    keys: DataKeys,
    *,
    batch_size: int,
    shuffle: bool,
    seed: int,
    image_size: int = 224,
    num_workers: int = 32,
    prefetch_batches: int = 4,
    dataset_repo_ids: str | Sequence[str] | None = None,
    loader: Literal["thread", "mp"] = "thread",
    image_cache_size: int = 4096,
    mp_loader: Any | None = None,
    pair_threads: int = 8,
    tactile_history: int = 0,
    history_stride: int = 5,
) -> Iterator[dict[str, np.ndarray]]:
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}.")
    if prefetch_batches <= 0:
        raise ValueError(f"prefetch_batches must be positive, got {prefetch_batches}.")
    if loader not in ("thread", "mp"):
        raise ValueError(f"loader must be 'thread' or 'mp', got {loader!r}.")
    if tactile_history < 0:
        raise ValueError(f"tactile_history must be non-negative, got {tactile_history}.")
    if history_stride <= 0:
        raise ValueError(f"history_stride must be positive, got {history_stride}.")
    # Expand each temporal record into one contrastive sample per wrist side.
    sample_index = [(record_i, side_i) for record_i in range(len(records)) for side_i in range(len(keys.sides))]
    order = np.arange(len(sample_index))
    if shuffle:
        order = np.random.default_rng(seed).permutation(order)

    starts = list(range(0, len(order), batch_size))
    if not starts:
        return

    worker_count = max(1, int(num_workers))
    prefetch_count = max(1, int(prefetch_batches))

    if loader == "mp" and (mp_loader is not None or worker_count > 1):
        if mp_loader is None and not dataset_repo_ids:
            raise ValueError("batches(loader='mp') requires dataset_repo_ids.")
        from tactile_encoder.utils.mp_batches import iter_mp_batches

        yield from iter_mp_batches(
            repo_ids=dataset_repo_ids or (),
            records=records,
            keys=keys,
            sample_index=sample_index,
            order=order,
            starts=starts,
            batch_size=batch_size,
            image_size=image_size,
            image_cache_size=image_cache_size,
            num_workers=worker_count,
            prefetch_batches=prefetch_count,
            pair_threads=pair_threads,
            loader=mp_loader,
            tactile_history=tactile_history,
            history_stride=history_stride,
        )
        return

    from concurrent.futures import ThreadPoolExecutor

    def _one(item: tuple[int, int]) -> dict[str, np.ndarray]:
        record_i, side_i = item
        return load_pair(
            dataset,
            records[record_i],
            keys.sides[side_i],
            image_size=image_size,
            side_id=side_i,
            tactile_history=tactile_history,
            history_stride=history_stride,
        )

    def _selected_for(start: int) -> list[tuple[int, int]]:
        return [sample_index[int(index)] for index in order[start : start + batch_size]]

    with ThreadPoolExecutor(max_workers=worker_count) as sample_pool, ThreadPoolExecutor(
        max_workers=prefetch_count
    ) as prefetch_pool:

        def _load_selected(selected: list[tuple[int, int]]) -> dict[str, np.ndarray]:
            if worker_count <= 1 or len(selected) <= 1:
                pairs = [_one(item) for item in selected]
            else:
                pairs = list(sample_pool.map(_one, selected))
            return stack_pairs(pairs)

        pending: dict[int, Any] = {}
        for start_i in range(min(prefetch_count, len(starts))):
            pending[start_i] = prefetch_pool.submit(_load_selected, _selected_for(starts[start_i]))

        for start_i, start in enumerate(starts):
            batch = pending.pop(start_i).result()
            next_i = start_i + prefetch_count
            if next_i < len(starts):
                pending[next_i] = prefetch_pool.submit(_load_selected, _selected_for(starts[next_i]))
            yield batch


# Backwards-compatible alias used by older call sites / docs.
def create_lerobot_datasets(
    repo_ids: str | Sequence[str],
    checkpoint_dir: pathlib.Path | None = None,
) -> ImageDatasetInfo:
    del checkpoint_dir
    return create_image_dataset(repo_ids)
