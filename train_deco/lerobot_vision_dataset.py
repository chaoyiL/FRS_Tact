"""Strict LeRobot v2.1 image adapter for the ``pick_tube_01`` layout.

The source state and action already carry the desired robot representation.
This module therefore performs no state/action derivation: it only validates,
normalizes, chunks and decodes two RGB camera streams plus optional tactile RGB streams.
"""

from __future__ import annotations

import io
import json
import os
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
from PIL import Image, UnidentifiedImageError
from torch.utils.data import Dataset

from .splits import split_episodes


DATASET_FORMAT = "lerobot-v2.1-parquet-vision-deco-v1"
CAMERA_NAMES = (
    "observation.images.camera0",
    "observation.images.camera1",
)
TACTILE_NAMES = (
    "observation.images.tactile_left_0",
    "observation.images.tactile_right_0",
    "observation.images.tactile_left_1",
    "observation.images.tactile_right_1",
)
STATE_KEY = "observation.state"
ACTION_KEY = "actions"
STATE_DIM = 20
ACTION_DIM = 20
STD_FLOOR = 1e-4

STATE_COLUMNS = (
    "robot0.relative_start.x",
    "robot0.relative_start.y",
    "robot0.relative_start.z",
    "robot0.relative_start.rx",
    "robot0.relative_start.ry",
    "robot0.relative_start.rz",
    "robot0.gripper_width",
    "robot1.relative_start.x",
    "robot1.relative_start.y",
    "robot1.relative_start.z",
    "robot1.relative_start.rx",
    "robot1.relative_start.ry",
    "robot1.relative_start.rz",
    "robot1.gripper_width",
    "left_relative_to_right.x",
    "left_relative_to_right.y",
    "left_relative_to_right.z",
    "left_relative_to_right.rx",
    "left_relative_to_right.ry",
    "left_relative_to_right.rz",
)

_ACTION_COMPONENTS = (
    "delta.x",
    "delta.y",
    "delta.z",
    "rotation_column_0.x",
    "rotation_column_0.y",
    "rotation_column_0.z",
    "rotation_column_1.x",
    "rotation_column_1.y",
    "rotation_column_1.z",
    "gripper_width",
)
ACTION_COLUMNS = tuple(
    f"robot{robot_index}.{component}"
    for robot_index in range(2)
    for component in _ACTION_COMPONENTS
)

_NUMERIC_COLUMNS = (
    STATE_KEY,
    ACTION_KEY,
    "frame_index",
    "episode_index",
    "task_index",
)


@dataclass(frozen=True)
class _EpisodeSpec:
    episode_id: int
    source_episode_id: int
    source_name: str
    length: int
    task_ids: tuple[int, ...]
    path: Path


def _read_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        raise ValueError(f"Required LeRobot metadata file is missing: {path}")
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSONL at {path}:{line_number}") from exc
    return rows


def _feature_shape(info: dict, key: str) -> tuple[int, ...]:
    feature = info.get("features", {}).get(key)
    if not isinstance(feature, dict):
        raise ValueError(f"LeRobot feature is missing: {key}")
    return tuple(int(value) for value in feature.get("shape", ()))


def _tactile_image_shape(info: dict) -> tuple[int, int]:
    tactile_shapes = []
    for name in TACTILE_NAMES:
        feature = info.get("features", {}).get(name)
        shape = _feature_shape(info, name)
        if feature.get("dtype") != "image":
            raise ValueError(f"{name} must be an image feature, got {feature.get('dtype')!r}")
        if len(shape) != 3 or shape[0] <= 0 or shape[1] <= 0 or shape[2] != 3:
            raise ValueError(
                f"{name} must be an RGB HWC image feature [H, W, 3], got {shape}"
            )
        tactile_shapes.append(shape)
    if len(set(tactile_shapes)) != 1:
        raise ValueError(f"The four tactile image shapes must match, got {tactile_shapes}")
    return tactile_shapes[0][0], tactile_shapes[0][1]


def _validate_info(
    root: Path, info: dict, *, include_tactile: bool = False
) -> tuple[int, int]:
    if info.get("codebase_version") != "v2.1":
        raise ValueError(
            f"Expected LeRobot codebase_version='v2.1', got {info.get('codebase_version')!r}"
        )
    if info.get("video_path") is not None or int(info.get("total_videos", -1)) != 0:
        raise ValueError("This adapter requires Parquet-embedded images, not video files")
    if _feature_shape(info, STATE_KEY) != (STATE_DIM,):
        raise ValueError(f"{STATE_KEY} must be exactly {STATE_DIM}D")
    if _feature_shape(info, ACTION_KEY) != (ACTION_DIM,):
        raise ValueError(f"{ACTION_KEY} must be exactly {ACTION_DIM}D")
    camera_shapes = [_feature_shape(info, name) for name in CAMERA_NAMES]
    if len(set(camera_shapes)) != 1 or len(camera_shapes[0]) != 3:
        raise ValueError(f"The two RGB camera shapes must match, got {camera_shapes}")
    height, width, channels = camera_shapes[0]
    if channels != 3 or height <= 0 or width <= 0:
        raise ValueError(f"Expected RGB HWC camera images, got {camera_shapes[0]}")
    if include_tactile:
        _tactile_image_shape(info)
    fps = float(info.get("fps", 0))
    if fps <= 0:
        raise ValueError(f"LeRobot fps must be positive, got {fps}")
    if not info.get("data_path"):
        raise ValueError(f"LeRobot data_path is missing: {root / 'meta/info.json'}")
    return height, width


def _episode_path(root: Path, info: dict, episode_id: int) -> Path:
    chunk_size = int(info.get("chunks_size", 1000))
    if chunk_size <= 0:
        raise ValueError(f"LeRobot chunks_size must be positive, got {chunk_size}")
    episode_chunk = episode_id // chunk_size
    relative = str(info["data_path"]).format(
        episode_chunk=episode_chunk,
        episode_index=episode_id,
        chunk_index=episode_chunk,
        file_index=episode_id,
    )
    path = root / relative
    if not path.is_file():
        raise ValueError(f"LeRobot episode Parquet is missing: {path}")
    return path


def _episode_specs(
    root: Path,
    info: dict,
    source_name: str,
    global_offset: int,
    preserve_source_ids: bool,
) -> list[_EpisodeSpec]:
    rows = _read_jsonl(root / "meta/episodes.jsonl")
    specs = []
    seen = set()
    for position, row in enumerate(sorted(rows, key=lambda item: int(item["episode_index"]))):
        source_episode_id = int(row["episode_index"])
        if source_episode_id in seen:
            raise ValueError(f"Duplicate episode_index in episodes.jsonl: {source_episode_id}")
        seen.add(source_episode_id)
        episode_id = source_episode_id if preserve_source_ids else global_offset + position
        length = int(row["length"])
        if length < 2:
            raise ValueError(
                f"Episode must contain a transition and terminal row: episode={episode_id}, length={length}"
            )
        specs.append(
            _EpisodeSpec(
                episode_id=episode_id,
                source_episode_id=source_episode_id,
                source_name=source_name,
                length=length,
                task_ids=tuple(int(value) for value in row.get("tasks", ())),
                path=_episode_path(root, info, source_episode_id),
            )
        )
    if len(specs) != int(info.get("total_episodes", -1)):
        raise ValueError(
            "LeRobot total_episodes disagrees with episodes.jsonl: "
            f"info={info.get('total_episodes')}, rows={len(specs)}"
        )
    if sum(spec.length for spec in specs) != int(info.get("total_frames", -1)):
        raise ValueError("LeRobot total_frames disagrees with episodes.jsonl")
    return sorted(specs, key=lambda spec: spec.episode_id)


def _fixed_vector_column(table, key: str, dimension: int) -> np.ndarray:
    values = np.asarray(table[key].to_pylist(), dtype=np.float32)
    if values.shape != (table.num_rows, dimension):
        raise ValueError(
            f"Parquet column {key!r} must have shape ({table.num_rows}, {dimension}), got {values.shape}"
        )
    return values


def _read_and_validate_numeric(spec: _EpisodeSpec) -> tuple[np.ndarray, np.ndarray]:
    try:
        table = pq.read_table(spec.path, columns=list(_NUMERIC_COLUMNS))
    except Exception as exc:
        raise ValueError(f"Cannot read LeRobot numeric columns: {spec.path}") from exc
    if table.num_rows != spec.length:
        raise ValueError(
            f"Episode row count mismatch: episode={spec.episode_id}, metadata={spec.length}, parquet={table.num_rows}"
        )
    states = _fixed_vector_column(table, STATE_KEY, STATE_DIM)
    actions = _fixed_vector_column(table, ACTION_KEY, ACTION_DIM)
    if not np.isfinite(states).all() or not np.isfinite(actions).all():
        raise ValueError(f"Non-finite state/action in episode {spec.episode_id}")
    frame_indices = np.asarray(table["frame_index"].to_numpy(), dtype=np.int64)
    if not np.array_equal(frame_indices, np.arange(spec.length, dtype=np.int64)):
        raise ValueError(
            f"frame_index must be contiguous within episode {spec.episode_id}"
        )
    episode_indices = np.asarray(table["episode_index"].to_numpy(), dtype=np.int64)
    if not np.all(episode_indices == spec.source_episode_id):
        raise ValueError(
            f"episode_index column mismatch in source={spec.source_name}, "
            f"episode={spec.source_episode_id}"
        )
    task_indices = np.asarray(table["task_index"].to_numpy(), dtype=np.int64)
    if spec.task_ids and not set(task_indices.tolist()).issubset(spec.task_ids):
        raise ValueError(f"task_index column mismatch in episode {spec.episode_id}")
    if not np.allclose(actions[-1], 0.0, rtol=0.0, atol=1e-7):
        raise ValueError(
            f"Expected an all-zero terminal action sentinel in episode {spec.episode_id}"
        )
    return states, actions


def _compute_train_stats(
    specs: list[_EpisodeSpec], train_episode_ids: set[int]
) -> dict[str, np.ndarray]:
    observation_sum = np.zeros(STATE_DIM, dtype=np.float64)
    observation_square_sum = np.zeros(STATE_DIM, dtype=np.float64)
    action_sum = np.zeros(ACTION_DIM, dtype=np.float64)
    action_square_sum = np.zeros(ACTION_DIM, dtype=np.float64)
    observation_count = 0
    action_count = 0
    for spec in specs:
        states, actions = _read_and_validate_numeric(spec)
        if spec.episode_id not in train_episode_ids:
            continue
        # The final state has no valid next action and is not an anchor. The
        # final all-zero action is an invalid Rotation-6D sentinel.
        valid_states = states[:-1].astype(np.float64)
        valid_actions = actions[:-1].astype(np.float64)
        observation_sum += valid_states.sum(axis=0)
        observation_square_sum += np.square(valid_states).sum(axis=0)
        action_sum += valid_actions.sum(axis=0)
        action_square_sum += np.square(valid_actions).sum(axis=0)
        observation_count += len(valid_states)
        action_count += len(valid_actions)
    if observation_count == 0 or action_count == 0:
        raise ValueError("Cannot compute statistics from an empty training split")
    observation_mean = observation_sum / observation_count
    action_mean = action_sum / action_count
    observation_variance = np.maximum(
        observation_square_sum / observation_count - np.square(observation_mean), 0.0
    )
    action_variance = np.maximum(
        action_square_sum / action_count - np.square(action_mean), 0.0
    )
    return {
        "observation_mean": observation_mean.astype(np.float32),
        "observation_std": np.clip(
            np.sqrt(observation_variance), STD_FLOOR, None
        ).astype(np.float32),
        "action_mean": action_mean.astype(np.float32),
        "action_std": np.clip(
            np.sqrt(action_variance), STD_FLOOR, None
        ).astype(np.float32),
    }


def _task_ids(root: Path, info: dict) -> tuple[list[str], list[dict]]:
    rows = _read_jsonl(root / "meta/tasks.jsonl")
    ids = sorted({int(row["task_index"]) for row in rows})
    if len(ids) != int(info.get("total_tasks", -1)) or not ids:
        raise ValueError("LeRobot total_tasks disagrees with tasks.jsonl")
    return [str(task_id) for task_id in ids], rows


def _load_sources(dataset_source: Path) -> tuple[list[dict], str]:
    if dataset_source.is_file():
        payload = json.loads(dataset_source.read_text(encoding="utf-8"))
        from .prepare_lerobot_multiroot import MANIFEST_FORMAT

        if payload.get("format") != MANIFEST_FORMAT:
            raise ValueError(
                f"Unsupported LeRobot multi-root manifest: {payload.get('format')!r}"
            )
        sources = [
            {"name": str(row["name"]), "root": Path(row["path"]).resolve()}
            for row in payload.get("sources", ())
        ]
        if not sources:
            raise ValueError("LeRobot multi-root manifest has no sources")
        return sorted(sources, key=lambda row: row["name"]), str(payload["dataset_id"])
    return [{"name": dataset_source.name, "root": dataset_source}], dataset_source.name


class LeRobotVisionDECODataset(Dataset):
    """Lazy visual view with optional four-stream tactile images."""

    def __init__(
        self,
        *,
        root: Path,
        split: str,
        specs: list[_EpisodeSpec],
        episode_ids: list[int],
        action_chunk_size: int,
        image_height: int,
        image_width: int,
        tactile_image_height: int | None,
        tactile_image_width: int | None,
        include_tactile: bool,
        metadata: dict,
        stats: dict[str, np.ndarray],
        task_ids: list[str],
        manifest: dict,
        limit: int | None,
    ):
        if split not in {"train", "val"}:
            raise ValueError(f"split must be train or val, got {split!r}")
        self.root = root
        self.split = split
        self.episode_ids = list(episode_ids)
        self.action_chunk_size = int(action_chunk_size)
        self.source_chunk_size = self.action_chunk_size
        self.image_height = int(image_height)
        self.image_width = int(image_width)
        self.include_tactile = bool(include_tactile)
        self.tactile_image_height = (
            int(tactile_image_height) if tactile_image_height is not None else None
        )
        self.tactile_image_width = (
            int(tactile_image_width) if tactile_image_width is not None else None
        )
        if self.include_tactile and (
            self.tactile_image_height is None or self.tactile_image_width is None
        ):
            raise ValueError("Tactile image dimensions are required when include_tactile=True")
        self.metadata = metadata
        self.stats = stats
        self.task_ids = task_ids
        self.manifest = manifest
        self.normalized = True
        self._task_to_index = {
            int(task_id): index for index, task_id in enumerate(self.task_ids)
        }
        self._spec_by_id = {spec.episode_id: spec for spec in specs}
        self.index = [
            (episode_id, row)
            for episode_id in self.episode_ids
            for row in range(self._spec_by_id[episode_id].length - 1)
        ]
        if limit is not None:
            if limit <= 0:
                raise ValueError(f"Dataset limit must be positive, got {limit}")
            self.index = self.index[:limit]
        if not self.index:
            raise ValueError(f"No nonterminal LeRobot samples in split={split}")
        self._episode_cache_size = max(
            1, int(os.environ.get("LEROBOT_EPISODE_CACHE_SIZE", "2"))
        )
        self._episode_lru: OrderedDict[int, dict] = OrderedDict()

    def __getstate__(self):
        state = dict(self.__dict__)
        state["_episode_lru"] = OrderedDict()
        return state

    def __len__(self) -> int:
        return len(self.index)

    def _load_episode(self, episode_id: int) -> dict:
        if episode_id in self._episode_lru:
            self._episode_lru.move_to_end(episode_id)
            return self._episode_lru[episode_id]
        spec = self._spec_by_id[episode_id]
        columns = [*CAMERA_NAMES, STATE_KEY, ACTION_KEY, "task_index"]
        if self.include_tactile:
            columns.extend(TACTILE_NAMES)
        try:
            table = pq.read_table(spec.path, columns=columns)
        except Exception as exc:
            raise ValueError(f"Cannot read LeRobot training sample data: {spec.path}") from exc
        if table.num_rows != spec.length:
            raise ValueError(f"Episode row count changed after validation: {spec.path}")
        episode = {
            "states": _fixed_vector_column(table, STATE_KEY, STATE_DIM),
            "actions": _fixed_vector_column(table, ACTION_KEY, ACTION_DIM),
            "task_indices": np.asarray(table["task_index"].to_numpy(), dtype=np.int64),
            "images": tuple(table[name] for name in CAMERA_NAMES),
        }
        if self.include_tactile:
            episode["tactile_images"] = tuple(table[name] for name in TACTILE_NAMES)
        self._episode_lru[episode_id] = episode
        while len(self._episode_lru) > self._episode_cache_size:
            self._episode_lru.popitem(last=False)
        return episode

    def _decode_image(
        self,
        image_column,
        row: int,
        camera_name: str,
        expected_height: int | None = None,
        expected_width: int | None = None,
    ) -> torch.Tensor:
        encoded = image_column[row].as_py()
        if not isinstance(encoded, dict) or not encoded.get("bytes"):
            raise ValueError(f"Missing embedded JPEG bytes: camera={camera_name}, row={row}")
        try:
            with Image.open(io.BytesIO(encoded["bytes"])) as image:
                array = np.array(image.convert("RGB"), dtype=np.uint8, copy=True)
        except (OSError, UnidentifiedImageError) as exc:
            raise ValueError(
                f"Cannot decode embedded JPEG: camera={camera_name}, row={row}"
            ) from exc
        expected = (
            self.image_height if expected_height is None else expected_height,
            self.image_width if expected_width is None else expected_width,
            3,
        )
        if array.shape != expected:
            raise ValueError(
                f"Decoded image shape mismatch: camera={camera_name}, got={array.shape}, expected={expected}"
            )
        return torch.from_numpy(array.transpose(2, 0, 1).copy()).float().div_(255.0)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        episode_id, row = self.index[index]
        episode = self._load_episode(episode_id)
        observation = episode["states"][row]
        valid_actions = episode["actions"][row:-1][: self.action_chunk_size]
        valid_action_count = len(valid_actions)
        if valid_action_count == 0:
            raise RuntimeError(f"Indexed a terminal action: episode={episode_id}, row={row}")
        if valid_action_count < self.action_chunk_size:
            actions = np.concatenate(
                (
                    valid_actions,
                    np.repeat(
                        valid_actions[-1:],
                        self.action_chunk_size - valid_action_count,
                        axis=0,
                    ),
                ),
                axis=0,
            )
        else:
            actions = valid_actions
        normalized_observation = (
            observation - self.stats["observation_mean"]
        ) / self.stats["observation_std"]
        normalized_actions = (
            actions - self.stats["action_mean"]
        ) / self.stats["action_std"]
        source_task_id = int(episode["task_indices"][row])
        if source_task_id not in self._task_to_index:
            raise ValueError(f"Unknown task_index {source_task_id} in episode {episode_id}")
        images = torch.stack(
            [
                self._decode_image(column, row, camera_name)
                for column, camera_name in zip(episode["images"], CAMERA_NAMES)
            ]
        )
        sample = {
            "observation": torch.from_numpy(
                normalized_observation.astype(np.float32)
            ),
            "action": torch.from_numpy(normalized_actions.astype(np.float32)),
            "images": images,
            "is_pad": torch.arange(self.action_chunk_size) >= valid_action_count,
            "task_index": torch.tensor(
                self._task_to_index[source_task_id], dtype=torch.long
            ),
        }
        if self.include_tactile:
            sample["tactile_images"] = torch.stack(
                [
                    self._decode_image(
                        column,
                        row,
                        tactile_name,
                        self.tactile_image_height,
                        self.tactile_image_width,
                    )
                    for column, tactile_name in zip(
                        episode["tactile_images"], TACTILE_NAMES
                    )
                ]
            )
        return sample


def build_lerobot_vision_datasets(
    dataset_dir: str | Path,
    *,
    action_chunk_size: int = 32,
    validation_ratio: float = 0.1,
    split_seed: int = 42,
    train_limit: int | None = None,
    val_limit: int | None = None,
    include_tactile: bool = False,
) -> tuple[LeRobotVisionDECODataset, LeRobotVisionDECODataset]:
    """Build train/validation views with one shared train-only statistics set."""

    source_path = Path(dataset_dir).expanduser().resolve()
    source_rows, manifest_dataset_id = _load_sources(source_path)
    if action_chunk_size <= 0:
        raise ValueError(f"action_chunk_size must be positive, got {action_chunk_size}")
    specs = []
    sources = []
    train_episode_ids = []
    val_episode_ids = []
    expected_image_shape = None
    expected_tactile_image_shape = None
    expected_fps = None
    expected_tasks = None
    tasks = None
    global_offset = 0
    preserve_source_ids = len(source_rows) == 1
    for source in source_rows:
        root = source["root"]
        info_path = root / "meta/info.json"
        if not info_path.is_file():
            raise ValueError(f"LeRobot info.json is missing: {info_path}")
        info = json.loads(info_path.read_text(encoding="utf-8"))
        image_shape = _validate_info(root, info, include_tactile=include_tactile)
        tactile_image_shape = (
            _tactile_image_shape(info) if include_tactile else None
        )
        fps = float(info["fps"])
        source_tasks, task_rows = _task_ids(root, info)
        if expected_image_shape is None:
            expected_image_shape = image_shape
            expected_tactile_image_shape = tactile_image_shape
            expected_fps = fps
            expected_tasks = task_rows
            tasks = source_tasks
        elif (image_shape, tactile_image_shape, fps, task_rows) != (
            expected_image_shape,
            expected_tactile_image_shape,
            expected_fps,
            expected_tasks,
        ):
            raise ValueError(f"LeRobot source contract differs: {root}")
        source_specs = _episode_specs(
            root,
            info,
            source["name"],
            global_offset,
            preserve_source_ids,
        )
        specs.extend(source_specs)
        source_episode_ids = [spec.episode_id for spec in source_specs]
        source_train, source_val = split_episodes(
            source_episode_ids, validation_ratio, split_seed
        )
        train_episode_ids.extend(source_train)
        val_episode_ids.extend(source_val)
        global_offset += len(source_specs)
        sources.append(
            {
                "name": source["name"],
                "path": str(root),
                "source_dataset_id": info.get("repo_id") or source["name"],
                "total_episodes": len(source_specs),
                "total_frames": int(info["total_frames"]),
            }
        )
    stats = _compute_train_stats(specs, set(train_episode_ids))
    dataset_id = (
        f"{manifest_dataset_id}@vision2-v1-val{validation_ratio:g}-seed{split_seed}"
    )
    metadata = {
        "source_format": DATASET_FORMAT,
        "obs_dim": STATE_DIM,
        "action_dim": ACTION_DIM,
        "chunk_size": int(action_chunk_size),
        "camera_names": list(CAMERA_NAMES),
        "state_columns": list(STATE_COLUMNS),
        "action_columns": list(ACTION_COLUMNS),
        "observation_indices": list(range(STATE_DIM)),
        "action_mode": "tcp_delta_absolute_gripper",
        "state_layout": "relative_start_pose6d_gripper_plus_left_relative_right",
        "rotation_representation": "rotation_6d_matrix_columns",
        "gripper_mode": "absolute",
        "terminal_action_policy": "excluded",
        "expected_sample_hz": expected_fps,
        "statistics_source": "train_episodes_nonterminal_rows_once",
    }
    if include_tactile:
        metadata["tactile_names"] = list(TACTILE_NAMES)
    manifest = {
        "format": DATASET_FORMAT,
        "dataset_id": dataset_id,
        "sources": sources,
        "split_seed": int(split_seed),
        "validation_ratio": float(validation_ratio),
        "splits": {
            "train": {
                "episode_ids": train_episode_ids,
                "episodes": [
                    {
                        "source": next(spec.source_name for spec in specs if spec.episode_id == episode_id),
                        "episode_index": next(spec.source_episode_id for spec in specs if spec.episode_id == episode_id),
                    }
                    for episode_id in train_episode_ids
                ],
            },
            "val": {
                "episode_ids": val_episode_ids,
                "episodes": [
                    {
                        "source": next(spec.source_name for spec in specs if spec.episode_id == episode_id),
                        "episode_index": next(spec.source_episode_id for spec in specs if spec.episode_id == episode_id),
                    }
                    for episode_id in val_episode_ids
                ],
            },
        },
    }
    shared = {
        "root": source_path,
        "specs": specs,
        "action_chunk_size": int(action_chunk_size),
        "image_height": expected_image_shape[0],
        "image_width": expected_image_shape[1],
        "tactile_image_height": (
            expected_tactile_image_shape[0] if include_tactile else None
        ),
        "tactile_image_width": (
            expected_tactile_image_shape[1] if include_tactile else None
        ),
        "include_tactile": include_tactile,
        "metadata": metadata,
        "stats": stats,
        "task_ids": tasks,
        "manifest": manifest,
    }
    return (
        LeRobotVisionDECODataset(
            split="train",
            episode_ids=train_episode_ids,
            limit=train_limit,
            **shared,
        ),
        LeRobotVisionDECODataset(
            split="val",
            episode_ids=val_episode_ids,
            limit=val_limit,
            **shared,
        ),
    )
