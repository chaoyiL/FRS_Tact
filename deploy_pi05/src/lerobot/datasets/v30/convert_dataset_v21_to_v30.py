#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Convert a LeRobot dataset from codebase version 2.1 to 3.0.

This module is based on Hugging Face LeRobot's official converter:
https://github.com/huggingface/lerobot/blob/bd74f673/src/lerobot/scripts/convert_dataset_v21_to_v30.py

It remains at the historical ``lerobot.datasets.v30`` import path so existing
commands keep working with this repository's trimmed LeRobot package.
"""

import argparse
import logging
import shutil
from pathlib import Path
from typing import Any

from lerobot.utils.import_utils import require_package

require_package("jsonlines", extra="dataset")

import jsonlines
import pandas as pd
import pyarrow as pa
import tqdm
from datasets import Dataset, Features, Image
from huggingface_hub import HfApi, snapshot_download
from requests import HTTPError

from lerobot.datasets import CODEBASE_VERSION, LeRobotDataset, aggregate_stats
from lerobot.datasets.io_utils import (
    cast_stats_to_numpy,
    get_file_size_in_mb,
    get_parquet_file_size_in_mb,
    get_parquet_num_frames,
    load_info,
    write_episodes,
    write_info,
    write_stats,
    write_tasks,
)
from lerobot.datasets.utils import (
    DEFAULT_CHUNK_SIZE,
    DEFAULT_DATA_FILE_SIZE_IN_MB,
    DEFAULT_DATA_PATH,
    DEFAULT_VIDEO_FILE_SIZE_IN_MB,
    DEFAULT_VIDEO_PATH,
    LEGACY_EPISODES_PATH,
    LEGACY_EPISODES_STATS_PATH,
    LEGACY_TASKS_PATH,
    update_chunk_file_indices,
)
from lerobot.datasets.video_utils import concatenate_video_files, get_video_duration_in_s
from lerobot.utils.constants import HF_LEROBOT_HOME
from lerobot.utils.utils import flatten_dict, init_logging

V21 = "v2.1"
V30 = "v3.0"


def load_jsonlines(fpath: Path) -> list[Any]:
    with jsonlines.open(fpath, "r") as reader:
        return list(reader)


def legacy_load_episodes(local_dir: Path) -> dict:
    episodes = load_jsonlines(local_dir / LEGACY_EPISODES_PATH)
    return {item["episode_index"]: item for item in sorted(episodes, key=lambda x: x["episode_index"])}


def legacy_load_episodes_stats(local_dir: Path) -> dict:
    episodes_stats = load_jsonlines(local_dir / LEGACY_EPISODES_STATS_PATH)
    return {
        item["episode_index"]: cast_stats_to_numpy(item["stats"])
        for item in sorted(episodes_stats, key=lambda x: x["episode_index"])
    }


def legacy_load_tasks(local_dir: Path) -> tuple[dict, dict]:
    tasks = load_jsonlines(local_dir / LEGACY_TASKS_PATH)
    tasks = {item["task_index"]: item["task"] for item in sorted(tasks, key=lambda x: x["task_index"])}
    task_to_task_index = {task: task_index for task_index, task in tasks.items()}
    return tasks, task_to_task_index


def validate_local_dataset_version(local_path: Path) -> None:
    info = load_info(local_path)
    dataset_version = info.codebase_version
    if dataset_version != V21:
        raise ValueError(
            f"Local dataset has codebase version '{dataset_version}', expected '{V21}'. "
            f"This script is specifically for converting v2.1 datasets to v3.0."
        )


def convert_tasks(root: Path, new_root: Path) -> None:
    logging.info("Converting tasks from %s to %s", root, new_root)
    tasks, _ = legacy_load_tasks(root)
    task_indices = tasks.keys()
    task_strings = tasks.values()
    df_tasks = pd.DataFrame(
        {"task_index": task_indices},
        index=pd.Index(task_strings, name="task"),
    )
    write_tasks(df_tasks, new_root)


def concat_data_files(
    paths_to_cat: list[Path],
    new_root: Path,
    chunk_idx: int,
    file_idx: int,
    image_keys: list[str],
) -> None:
    dataframes = [pd.read_parquet(file) for file in paths_to_cat]
    concatenated_df = pd.concat(dataframes, ignore_index=True)

    path = new_root / DEFAULT_DATA_PATH.format(chunk_index=chunk_idx, file_index=file_idx)
    path.parent.mkdir(parents=True, exist_ok=True)

    if image_keys:
        schema = pa.Schema.from_pandas(concatenated_df)
        features = Features.from_arrow_schema(schema)
        for key in image_keys:
            features[key] = Image()
        schema = features.arrow_schema
    else:
        schema = None

    concatenated_df.to_parquet(path, index=False, schema=schema)


def convert_data(root: Path, new_root: Path, data_file_size_in_mb: int) -> list[dict]:
    ep_paths = sorted((root / "data").glob("*/*.parquet"))
    image_keys = get_image_keys(root)

    chunk_idx = 0
    file_idx = 0
    size_in_mb = 0
    num_frames = 0
    paths_to_cat: list[Path] = []
    episodes_metadata = []

    logging.info("Converting data files from %d episodes", len(ep_paths))

    for ep_idx, ep_path in enumerate(tqdm.tqdm(ep_paths, desc="convert data files")):
        ep_size_in_mb = get_parquet_file_size_in_mb(ep_path)
        ep_num_frames = get_parquet_num_frames(ep_path)

        if size_in_mb + ep_size_in_mb >= data_file_size_in_mb and paths_to_cat:
            concat_data_files(paths_to_cat, new_root, chunk_idx, file_idx, image_keys)
            chunk_idx, file_idx = update_chunk_file_indices(chunk_idx, file_idx, DEFAULT_CHUNK_SIZE)
            size_in_mb = 0
            paths_to_cat = []

        episodes_metadata.append(
            {
                "episode_index": ep_idx,
                "data/chunk_index": chunk_idx,
                "data/file_index": file_idx,
                "dataset_from_index": num_frames,
                "dataset_to_index": num_frames + ep_num_frames,
            }
        )
        size_in_mb += ep_size_in_mb
        num_frames += ep_num_frames
        paths_to_cat.append(ep_path)

    if paths_to_cat:
        concat_data_files(paths_to_cat, new_root, chunk_idx, file_idx, image_keys)

    return episodes_metadata


def get_video_keys(root: Path) -> list[str]:
    info = load_info(root)
    return [key for key, feature in info.features.items() if feature["dtype"] == "video"]


def get_image_keys(root: Path) -> list[str]:
    info = load_info(root)
    return [key for key, feature in info.features.items() if feature["dtype"] == "image"]


def convert_videos(root: Path, new_root: Path, video_file_size_in_mb: int) -> list[dict] | None:
    logging.info("Converting videos from %s to %s", root, new_root)

    video_keys = sorted(get_video_keys(root))
    if not video_keys:
        return None

    eps_metadata_per_cam = [
        convert_videos_of_camera(root, new_root, camera, video_file_size_in_mb)
        for camera in video_keys
    ]

    num_eps_per_cam = [len(eps_cam_map) for eps_cam_map in eps_metadata_per_cam]
    if len(set(num_eps_per_cam)) != 1:
        raise ValueError(f"All cams dont have same number of episodes ({num_eps_per_cam}).")

    episodes_metadata = []
    num_episodes = num_eps_per_cam[0]
    for ep_idx in tqdm.tqdm(range(num_episodes), desc="convert videos"):
        ep_ids = [camera_metadata[ep_idx]["episode_index"] for camera_metadata in eps_metadata_per_cam]
        ep_ids.append(ep_idx)
        if len(set(ep_ids)) != 1:
            raise ValueError(f"All episode indices need to match ({ep_ids}).")

        ep_dict = {}
        for camera_metadata in eps_metadata_per_cam:
            ep_dict.update(camera_metadata[ep_idx])
        episodes_metadata.append(ep_dict)

    return episodes_metadata


def convert_videos_of_camera(
    root: Path,
    new_root: Path,
    video_key: str,
    video_file_size_in_mb: int,
) -> list[dict]:
    ep_paths = sorted((root / "videos").glob(f"*/{video_key}/*.mp4"))

    ep_idx = 0
    chunk_idx = 0
    file_idx = 0
    size_in_mb = 0
    duration_in_s = 0.0
    paths_to_cat: list[Path] = []
    episodes_metadata = []

    for ep_path in tqdm.tqdm(ep_paths, desc=f"convert videos of {video_key}"):
        ep_size_in_mb = get_file_size_in_mb(ep_path)
        ep_duration_in_s = get_video_duration_in_s(ep_path)

        if size_in_mb + ep_size_in_mb >= video_file_size_in_mb and paths_to_cat:
            concatenate_video_files(
                paths_to_cat,
                new_root
                / DEFAULT_VIDEO_PATH.format(
                    video_key=video_key,
                    chunk_index=chunk_idx,
                    file_index=file_idx,
                ),
            )

            for i, _ in enumerate(paths_to_cat):
                past_ep_idx = ep_idx - len(paths_to_cat) + i
                episodes_metadata[past_ep_idx][f"videos/{video_key}/chunk_index"] = chunk_idx
                episodes_metadata[past_ep_idx][f"videos/{video_key}/file_index"] = file_idx

            chunk_idx, file_idx = update_chunk_file_indices(chunk_idx, file_idx, DEFAULT_CHUNK_SIZE)
            size_in_mb = 0
            duration_in_s = 0.0
            paths_to_cat = []

        episodes_metadata.append(
            {
                "episode_index": ep_idx,
                f"videos/{video_key}/chunk_index": chunk_idx,
                f"videos/{video_key}/file_index": file_idx,
                f"videos/{video_key}/from_timestamp": duration_in_s,
                f"videos/{video_key}/to_timestamp": duration_in_s + ep_duration_in_s,
            }
        )

        paths_to_cat.append(ep_path)
        size_in_mb += ep_size_in_mb
        duration_in_s += ep_duration_in_s
        ep_idx += 1

    if paths_to_cat:
        concatenate_video_files(
            paths_to_cat,
            new_root
            / DEFAULT_VIDEO_PATH.format(
                video_key=video_key,
                chunk_index=chunk_idx,
                file_index=file_idx,
            ),
        )

        for i, _ in enumerate(paths_to_cat):
            past_ep_idx = ep_idx - len(paths_to_cat) + i
            episodes_metadata[past_ep_idx][f"videos/{video_key}/chunk_index"] = chunk_idx
            episodes_metadata[past_ep_idx][f"videos/{video_key}/file_index"] = file_idx

    return episodes_metadata


def generate_episode_metadata_dict(
    episodes_legacy_metadata: dict,
    episodes_metadata: list[dict],
    episodes_stats: dict,
    episodes_videos: list[dict] | None = None,
):
    num_episodes = len(episodes_metadata)
    episodes_legacy_metadata_vals = list(episodes_legacy_metadata.values())
    episodes_stats_vals = list(episodes_stats.values())
    episodes_stats_keys = list(episodes_stats.keys())

    for i in range(num_episodes):
        ep_legacy_metadata = episodes_legacy_metadata_vals[i]
        ep_metadata = episodes_metadata[i]
        ep_stats = episodes_stats_vals[i]

        ep_ids_set = {
            ep_legacy_metadata["episode_index"],
            ep_metadata["episode_index"],
            episodes_stats_keys[i],
        }

        if episodes_videos is None:
            ep_video = {}
        else:
            ep_video = episodes_videos[i]
            ep_ids_set.add(ep_video["episode_index"])

        if len(ep_ids_set) != 1:
            raise ValueError(f"Number of episodes is not the same ({ep_ids_set}).")

        ep_dict = {**ep_metadata, **ep_video, **ep_legacy_metadata, **flatten_dict({"stats": ep_stats})}
        ep_dict["meta/episodes/chunk_index"] = 0
        ep_dict["meta/episodes/file_index"] = 0
        yield ep_dict


def convert_episodes_metadata(
    root: Path,
    new_root: Path,
    episodes_metadata: list[dict],
    episodes_video_metadata: list[dict] | None = None,
) -> None:
    logging.info("Converting episodes metadata from %s to %s", root, new_root)

    episodes_legacy_metadata = legacy_load_episodes(root)
    episodes_stats = legacy_load_episodes_stats(root)

    num_eps_set = {len(episodes_legacy_metadata), len(episodes_metadata)}
    if episodes_video_metadata is not None:
        num_eps_set.add(len(episodes_video_metadata))

    if len(num_eps_set) != 1:
        raise ValueError(f"Number of episodes is not the same ({num_eps_set}).")

    ds_episodes = Dataset.from_generator(
        lambda: generate_episode_metadata_dict(
            episodes_legacy_metadata,
            episodes_metadata,
            episodes_stats,
            episodes_video_metadata,
        )
    )
    write_episodes(ds_episodes, new_root)
    write_stats(aggregate_stats(list(episodes_stats.values())), new_root)


def convert_info(
    root: Path,
    new_root: Path,
    data_file_size_in_mb: int,
    video_file_size_in_mb: int,
) -> None:
    info = load_info(root)
    # load_info() already ignores legacy-only fields such as total_chunks and
    # total_videos when constructing the typed DatasetInfo object.
    info.codebase_version = V30
    info.data_files_size_in_mb = data_file_size_in_mb
    info.video_files_size_in_mb = video_file_size_in_mb
    info.data_path = DEFAULT_DATA_PATH
    info.video_path = DEFAULT_VIDEO_PATH if info.video_path is not None else None
    info.fps = int(info.fps)

    logging.info("Converting info from %s to %s", root, new_root)
    for feature in info.features.values():
        if feature["dtype"] != "video":
            feature["fps"] = info.fps
    write_info(info, new_root)


def convert_dataset(
    repo_id: str,
    branch: str | None = None,
    data_file_size_in_mb: int | None = None,
    video_file_size_in_mb: int | None = None,
    root: str | Path | None = None,
    push_to_hub: bool = True,
    force_conversion: bool = False,
) -> None:
    data_file_size_in_mb = data_file_size_in_mb or DEFAULT_DATA_FILE_SIZE_IN_MB
    video_file_size_in_mb = video_file_size_in_mb or DEFAULT_VIDEO_FILE_SIZE_IN_MB

    if root is None and not force_conversion:
        try:
            print("Trying to download v3.0 version of the dataset from the hub...")
            snapshot_download(
                repo_id,
                repo_type="dataset",
                revision=V30,
                local_dir=HF_LEROBOT_HOME / repo_id,
            )
            return
        except Exception:
            print("Dataset does not have an uploaded v3.0 version. Continuing with conversion.")

    use_local_dataset = False
    root = HF_LEROBOT_HOME / repo_id if root is None else Path(root)
    if root.exists():
        validate_local_dataset_version(root)
        use_local_dataset = True
        print(f"Using local dataset at {root}")

    old_root = root.parent / f"{root.name}_old"
    new_root = root.parent / f"{root.name}_v30"

    if old_root.is_dir() and root.is_dir():
        shutil.rmtree(root)
        shutil.move(old_root, root)

    if new_root.is_dir():
        shutil.rmtree(new_root)

    if not use_local_dataset:
        snapshot_download(repo_id, repo_type="dataset", revision=V21, local_dir=root)

    convert_info(root, new_root, data_file_size_in_mb, video_file_size_in_mb)
    convert_tasks(root, new_root)
    episodes_metadata = convert_data(root, new_root, data_file_size_in_mb)
    episodes_videos_metadata = convert_videos(root, new_root, video_file_size_in_mb)
    convert_episodes_metadata(root, new_root, episodes_metadata, episodes_videos_metadata)

    shutil.move(root, old_root)
    shutil.move(new_root, root)

    if push_to_hub:
        hub_api = HfApi()
        try:
            hub_api.delete_tag(repo_id, tag=CODEBASE_VERSION, repo_type="dataset")
        except HTTPError as error:
            print(f"tag={CODEBASE_VERSION} probably doesn't exist. Skipping exception ({error})")
        hub_api.delete_files(
            delete_patterns=["data/chunk*/episode_*", "meta/*.jsonl", "videos/chunk*"],
            repo_id=repo_id,
            revision=branch,
            repo_type="dataset",
        )
        hub_api.create_tag(repo_id, tag=CODEBASE_VERSION, revision=branch, repo_type="dataset")
        LeRobotDataset(repo_id).push_to_hub()


def main() -> None:
    init_logging()
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repo-id",
        type=str,
        required=True,
        help="Repository identifier on Hugging Face, for example 'lerobot/pusht'.",
    )
    parser.add_argument(
        "--branch",
        type=str,
        default=None,
        help="Repository branch to push. Defaults to the main branch.",
    )
    parser.add_argument("--data-file-size-in-mb", type=int, default=None)
    parser.add_argument("--video-file-size-in-mb", type=int, default=None)
    parser.add_argument(
        "--root",
        type=str,
        default=None,
        help="Exact local dataset directory containing meta/, data/, and videos/.",
    )
    parser.add_argument(
        "--push-to-hub",
        type=lambda value: value.lower() == "true",
        default=True,
    )
    parser.add_argument("--force-conversion", action="store_true")
    args = parser.parse_args()
    convert_dataset(**vars(args))


if __name__ == "__main__":
    main()
