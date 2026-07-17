#!/usr/bin/env python
"""Run SmolVLA inference on a frame from a local LeRobot dataset.

Usage:
    uv sync
    uv run python tools/infer_smolvla.py \
        --dataset-path /path/to/dataset \
        --episode 0 \
        --frame 0

    uv run python tools/infer_smolvla.py \
        --dataset-path ~/.cache/huggingface/lerobot/chaoyi/black_smash_02 \
        --episode 0 --frame 0 \
        --model /path/to/checkpoint
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from lerobot.datasets import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.policies import make_pre_post_processors
from lerobot.policies.smolvla import SmolVLAPolicy
from lerobot.policies.utils import prepare_observation_for_inference
from lerobot.utils.constants import OBS_IMAGES, OBS_STATE


DEFAULT_MODEL = "lerobot/smolvla_base"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run SmolVLA inference on a LeRobot dataset frame.")
    parser.add_argument(
        "--dataset-path",
        required=True,
        type=Path,
        help="Local root directory of the LeRobot dataset.",
    )
    parser.add_argument(
        "--repo-id",
        default=None,
        help="Dataset repo id. Defaults to the dataset directory name.",
    )
    parser.add_argument("--episode", type=int, required=True, help="Episode index to read.")
    parser.add_argument("--frame", type=int, required=True, help="Frame index within the episode.")
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"Checkpoint path or Hugging Face model id (default: {DEFAULT_MODEL}).",
    )
    parser.add_argument("--device", default=None, help="cpu | cuda | mps (default: from model config)")
    parser.add_argument(
        "--task",
        default=None,
        help="Language instruction override. Defaults to the task stored in the dataset frame.",
    )
    parser.add_argument(
        "--rename-map",
        default=None,
        help='Optional JSON object for observation key renaming, e.g. \'{"observation.images.camera0": "observation.images.camera1"}\'.',
    )
    return parser.parse_args()


def _chw_to_hwc_uint8(image: np.ndarray | torch.Tensor) -> np.ndarray:
    if isinstance(image, torch.Tensor):
        image = image.detach().cpu().numpy()
    if image.ndim != 3:
        raise ValueError(f"Expected a 3D image tensor/array, got shape {image.shape}")
    if image.shape[0] in (1, 3):
        image = np.transpose(image, (1, 2, 0))
    if image.dtype == np.uint8:
        return image
    if np.issubdtype(image.dtype, np.floating) and image.max() <= 1.0:
        image = image * 255.0
    return image.astype(np.uint8)


def frame_to_raw_observation(sample: dict[str, Any], features: dict[str, dict]) -> dict[str, Any]:
    observation: dict[str, Any] = {}
    for key, feat in features.items():
        if key not in sample:
            continue
        value = sample[key]
        if feat.get("dtype") in ("image", "video"):
            observation[key] = _chw_to_hwc_uint8(value)
        elif key == OBS_STATE:
            state = value.detach().cpu().numpy() if isinstance(value, torch.Tensor) else np.asarray(value)
            observation[key] = state.astype(np.float32)
    return observation


def _to_numpy(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def load_dataset_frame(
    dataset_path: Path,
    repo_id: str,
    episode: int,
    frame: int,
) -> tuple[dict[str, Any], LeRobotDatasetMetadata]:
    meta = LeRobotDatasetMetadata(repo_id=repo_id, root=dataset_path)
    if episode < 0 or episode >= meta.info.total_episodes:
        raise IndexError(
            f"episode {episode} out of range [0, {meta.info.total_episodes - 1}] for dataset {dataset_path}"
        )

    dataset = LeRobotDataset(
        repo_id=repo_id,
        root=dataset_path,
        episodes=[episode],
        revision=meta.info.codebase_version,
    )
    if frame < 0 or frame >= len(dataset):
        raise IndexError(f"frame {frame} out of range [0, {len(dataset) - 1}] for episode {episode}")

    return dataset[frame], meta


def parse_rename_map(rename_map_arg: str | None) -> dict[str, str]:
    if not rename_map_arg:
        return {}
    rename_map = json.loads(rename_map_arg)
    if not isinstance(rename_map, dict):
        raise ValueError("--rename-map must be a JSON object")
    return {str(k): str(v) for k, v in rename_map.items()}


def main() -> None:
    args = parse_args()
    dataset_path = args.dataset_path.expanduser().resolve()
    repo_id = args.repo_id or dataset_path.name
    rename_map = parse_rename_map(args.rename_map)

    sample, meta = load_dataset_frame(dataset_path, repo_id, args.episode, args.frame)
    task = args.task if args.task is not None else sample.get("task", "")
    robot_type = meta.info.robot_type

    obs = frame_to_raw_observation(sample, meta.info.features)
    image_keys = [k for k in obs if k.startswith(f"{OBS_IMAGES}.")]
    state_shape = obs[OBS_STATE].shape if OBS_STATE in obs else None

    device = torch.device(args.device) if args.device else None
    policy = SmolVLAPolicy.from_pretrained(args.model)
    if device is not None:
        policy.config.device = str(device)
        policy.to(device)

    preprocessor_overrides: dict[str, Any] = {
        "device_processor": {"device": policy.config.device},
    }
    if rename_map:
        preprocessor_overrides["rename_observations_processor"] = {"rename_map": rename_map}

    preprocess, postprocess = make_pre_post_processors(
        policy.config,
        args.model,
        preprocessor_overrides=preprocessor_overrides,
    )

    observation = prepare_observation_for_inference(
        obs,
        policy.config.device,
        task=task,
        robot_type=robot_type,
    )
    batch = preprocess(observation)
    action = policy.select_action(batch)
    action = postprocess(action)

    print(f"dataset_path : {dataset_path}")
    print(f"repo_id      : {repo_id}")
    print(f"episode      : {args.episode}")
    print(f"frame        : {args.frame}")
    print(f"task         : {task!r}")
    print(f"robot_type   : {robot_type}")
    print(f"model        : {args.model}")
    print(f"device       : {policy.config.device}")
    print(f"image keys   : {image_keys}")
    print(f"state shape  : {state_shape}")
    print(f"action shape : {tuple(action.shape)}")
    print(f"action       : {action.cpu().numpy()}")

    if "actions" in sample:
        gt_action = _to_numpy(sample["actions"]).astype(np.float32)
        print(f"gt_action    : {gt_action}")
    elif "action" in sample:
        gt_action = _to_numpy(sample["action"]).astype(np.float32)
        print(f"gt_action    : {gt_action}")


if __name__ == "__main__":
    main()
