#!/usr/bin/env python
"""Precompute one dataset's complete frozen SmolVLA training cache."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import jax
import numpy as np

from lerobot.datasets import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.policies.smolvla_jax import JaxSmolVLA, JaxSmolVLAConfig
from lerobot.policies.smolvla_jax.checkpoint import load_params, resolve_checkpoint
from lerobot.policies.smolvla_jax.data import (
    DatasetSource,
    action_delta_timestamps,
    parse_dataset_sources,
    resolve_action_key,
    resolve_source_visual_keys,
)
from lerobot.policies.smolvla_jax.lora import resolve_module_modes
from lerobot.policies.smolvla_jax.offline_cache_precompute import OfflineCachePrecomputer
from lerobot.policies.smolvla_jax.offline_training_cache import OfflineCacheSpec, offline_cache_dir
from lerobot.policies.smolvla_jax.preprocessing import (
    JaxSmolVLAPreprocessor,
    _as_bchw,
    resize_with_pad,
)
from tools.train_smolvla_jax import apply_model_overrides, load_yaml_config, require


DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "train_vtsmolvla_jax_tactile16.yaml"


def nonnegative_dataset_index(value: str) -> int:
    index = int(value)
    if index < 0:
        raise argparse.ArgumentTypeError("dataset index must be non-negative")
    return index


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--dataset-index", type=nonnegative_dataset_index, required=True)
    return parser.parse_args(argv)


def select_dataset_source(config: Mapping[str, Any], dataset_index: int) -> DatasetSource:
    if dataset_index < 0:
        raise ValueError(f"dataset index must be non-negative, got {dataset_index}")
    sources = parse_dataset_sources(config)
    if dataset_index >= len(sources):
        raise ValueError(
            f"dataset index {dataset_index} is unavailable; config defines {len(sources)} sources"
        )
    return sources[dataset_index]


class _PrecomputeDataset:
    def __init__(
        self,
        dataset: LeRobotDataset,
        *,
        source: DatasetSource,
        action_key: str,
        image_keys: tuple[str, ...],
        empty_cameras: int,
    ) -> None:
        self.dataset = dataset
        self.source = source
        self.action_key = action_key
        self.image_keys = image_keys
        self.empty_cameras = int(empty_cameras)
        self.rename_map = dict(source.rename_map or {})

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> dict[str, Any]:
        raw = self.dataset[index]
        observation = {
            self.rename_map.get(key, key): value
            for key, value in raw.items()
            if key.startswith("observation.")
        }
        task = str(raw["task"])
        present_keys = [key for key in self.image_keys if key in observation]
        missing_keys = [key for key in self.image_keys if key not in observation]
        if not present_keys:
            raise ValueError(
                f"none of the expected image keys are present: "
                f"{self.image_keys}"
            )
        images = [_as_bchw(observation[key])[0] for key in present_keys]
        for _ in missing_keys[: self.empty_cameras]:
            images.append(np.zeros_like(images[-1]))
        actions = np.asarray(raw[self.action_key], dtype=np.float32)
        padding_key = f"{self.action_key}_is_pad"
        action_is_pad = np.asarray(
            raw.get(padding_key, np.zeros(actions.shape[0], dtype=np.bool_)),
            dtype=np.bool_,
        )
        return {
            "images": np.stack(images, axis=0),
            "state": np.asarray(observation["observation.state"], dtype=np.float32),
            "actions": actions,
            "action_is_pad": action_is_pad,
            "task": task,
            "episode_index": int(np.asarray(raw["episode_index"]).reshape(())),
            "frame_index": int(np.asarray(raw["frame_index"]).reshape(())),
        }


def _build_dataset(
    source: DatasetSource,
    *,
    config: JaxSmolVLAConfig,
    checkpoint: Path,
    local_files_only: bool,
    video_backend: str | None,
    return_uint8: bool,
) -> tuple[_PrecomputeDataset, JaxSmolVLAPreprocessor]:
    metadata = LeRobotDatasetMetadata(
        repo_id=source.repo_id,
        root=source.root,
        revision=source.revision,
    )
    action_key = resolve_action_key(metadata.features, source.action_key)
    visual_keys = resolve_source_visual_keys(
        config.image_keys,
        source.rename_map,
        metadata.camera_keys,
        allow_missing=config.empty_cameras,
    )
    dataset = LeRobotDataset(
        repo_id=source.repo_id,
        root=metadata.root,
        revision=metadata.revision,
        episodes=list(source.episodes) if source.episodes is not None else None,
        delta_timestamps=action_delta_timestamps(action_key, config.chunk_size, metadata.fps),
        image_transforms=None,
        video_backend=video_backend,
        return_uint8=return_uint8,
        download_videos=True,
        visual_keys=visual_keys,
    )
    vision_only_config = replace(
        config,
        use_tactile_encoder=False,
        tactile_keys=(),
    )
    preprocessor = JaxSmolVLAPreprocessor(
        checkpoint,
        vision_only_config,
        rename_map={},
        stats=None,
        local_files_only=local_files_only,
    )
    return (
        _PrecomputeDataset(
            dataset,
            source=source,
            action_key=action_key,
            image_keys=tuple(config.image_keys),
            empty_cameras=config.empty_cameras,
        ),
        preprocessor,
    )


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    cfg = load_yaml_config(args.config)
    source = select_dataset_source(cfg, args.dataset_index)
    cache_cfg = cfg.get("offline_training_cache") or {}
    if not isinstance(cache_cfg, Mapping):
        raise ValueError("offline_training_cache must be a mapping")
    if not bool(cache_cfg.get("enabled", False)):
        raise ValueError("offline_training_cache.enabled must be true")
    cache_root = cache_cfg.get("root")
    if not cache_root:
        raise ValueError("offline_training_cache.root is required")
    if str(cache_cfg.get("dtype", "bfloat16")) != "bfloat16":
        raise ValueError("offline_training_cache.dtype must be bfloat16")
    image_transforms = cfg.get("image_transforms") or {}
    if image_transforms and bool(image_transforms.get("enable", True)):
        raise ValueError("offline cache precompute requires disabled image augmentation")

    allow_download = bool(cfg.get("allow_download", False))
    checkpoint = resolve_checkpoint(
        require(cfg, "checkpoint"),
        revision=cfg.get("revision"),
        local_files_only=not allow_download,
    )
    config = apply_model_overrides(
        JaxSmolVLAConfig.from_pretrained(checkpoint),
        cfg.get("model"),
    )
    modes = resolve_module_modes(config)
    if modes["vision"] != "frozen" or modes["connector"] != "frozen":
        raise ValueError("offline cache precompute requires frozen vision and connector")

    params = load_params(checkpoint)
    model = JaxSmolVLA(config)

    @jax.jit
    def encode(images_bchw):
        images_bchw = resize_with_pad(
            images_bchw,
            config.resize_width,
            config.resize_height,
            pad_value=0.0,
        )
        return model.embed_image(params, images_bchw * 2.0 - 1.0)

    local_files_only = not (allow_download or bool(cfg.get("allow_tokenizer_download", False)))
    dataset, preprocessor = _build_dataset(
        source,
        config=config,
        checkpoint=checkpoint,
        local_files_only=local_files_only,
        video_backend=cfg.get("video_backend"),
        return_uint8=bool(cfg.get("return_uint8", True)),
    )
    tokens_height = config.resize_height // config.vision_patch_size // config.connector_scale_factor
    tokens_width = config.resize_width // config.vision_patch_size // config.connector_scale_factor
    spec = OfflineCacheSpec(
        repo_id=source.repo_id,
        total_frames=len(dataset),
        camera_keys=tuple(config.image_keys),
        vision_tokens_per_camera=tokens_height * tokens_width,
        vision_hidden_size=config.text_hidden_size,
        state_dim=config.state_dim,
        action_dim=config.action_dim,
        chunk_size=config.chunk_size,
        tokenizer_max_length=config.tokenizer_max_length,
        checkpoint_source=str(checkpoint),
        vision_mode=modes["vision"],
        connector_mode=modes["connector"],
    )

    def encode_cameras(images_bchw: np.ndarray) -> np.ndarray:
        batch, cameras = images_bchw.shape[:2]
        flat = images_bchw.reshape((batch * cameras,) + images_bchw.shape[2:])
        encoded = np.asarray(jax.device_get(encode(flat)))
        return encoded.reshape(
            batch,
            cameras,
            spec.vision_tokens_per_camera,
            spec.vision_hidden_size,
        )

    output_dir = offline_cache_dir(Path(cache_root), source.repo_id)
    result = OfflineCachePrecomputer(
        spec=spec,
        output_dir=output_dir,
        dataset=dataset,
        encode_vision=encode_cameras,
        tokenize=preprocessor.tokenize,
        batch_size=int(cache_cfg.get("precompute_batch_size", 64)),
        num_workers=int(cache_cfg.get("precompute_num_workers", 0)),
        prefetch_factor=int(cache_cfg.get("precompute_prefetch_factor", 2)),
    ).run()
    print(f"complete: {source.repo_id} -> {result}", flush=True)


if __name__ == "__main__":
    main()
