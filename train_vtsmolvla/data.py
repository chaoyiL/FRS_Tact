from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from train_smolvla.data import (
    DatasetSource,
    LeRobotDataset,
    LeRobotDatasetMetadata,
    LeRobotJaxDataLoader,
    _KeyMappedLeRobotDataset,
    _to_numpy,
    resolve_source_visual_keys,
)

from .configuration import VTSmolVLAConfig
from .preprocessing import VTJaxSmolVLAPreprocessor
from .tactile_cache import (
    TACTILE_EMBEDDING_OBSERVATION_KEY,
    TactileEmbeddingCache,
    tactile_cache_dir,
)


class _VTKeyMappedLeRobotDataset(_KeyMappedLeRobotDataset):
    """Visual key mapping extended with cached tactile embeddings."""

    def __init__(
        self,
        *args: Any,
        tactile_embedding_cache: TactileEmbeddingCache | None = None,
        **kwargs: Any,
    ):
        super().__init__(*args, **kwargs)
        self.tactile_embedding_cache = tactile_embedding_cache

    def _enrich_mapped_sample(
        self,
        source_sample: Mapping[str, Any],
        mapped: dict[str, Any],
    ) -> None:
        if self.tactile_embedding_cache is None:
            return
        episode_index = int(_to_numpy(source_sample["episode_index"]).reshape(()).item())
        frame_index = int(_to_numpy(source_sample["frame_index"]).reshape(()).item())
        episode = self.dataset.meta.episodes[episode_index]
        absolute_index = int(episode["dataset_from_index"]) + frame_index
        mapped[TACTILE_EMBEDDING_OBSERVATION_KEY] = self.tactile_embedding_cache[
            absolute_index
        ]


def resolve_model_visual_keys(
    config: VTSmolVLAConfig,
    *,
    use_tactile_embedding_cache: bool = False,
) -> tuple[str, ...]:
    keys = list(config.image_keys)
    if config.use_tactile_encoder and not use_tactile_embedding_cache:
        keys.extend(config.tactile_keys)
    return tuple(dict.fromkeys(keys))


class VTLeRobotJaxDataLoader(LeRobotJaxDataLoader):
    """LeRobot loader with optional frozen tactile-embedding cache support."""

    config: VTSmolVLAConfig

    def __init__(
        self,
        checkpoint: str | Path,
        config: VTSmolVLAConfig,
        *,
        tactile_embedding_cache_root: str | Path | None = None,
        **kwargs: Any,
    ):
        self.tactile_embedding_cache_root = (
            None
            if tactile_embedding_cache_root is None
            else Path(tactile_embedding_cache_root).expanduser()
        )
        if self.tactile_embedding_cache_root is not None and not config.use_tactile_encoder:
            raise ValueError("tactile embedding cache requires use_tactile_encoder=True")
        super().__init__(checkpoint, config, **kwargs)

    def _source_visual_keys_and_context(
        self,
        config: VTSmolVLAConfig,
        source: DatasetSource,
        metadata: LeRobotDatasetMetadata,
        source_rename: Mapping[str, str],
    ) -> tuple[list[str], dict[str, Any]]:
        visual_keys, context = super()._source_visual_keys_and_context(
            config,
            source,
            metadata,
            source_rename,
        )
        if not config.use_tactile_encoder:
            return visual_keys, context
        source_tactile_keys = tuple(
            resolve_source_visual_keys(
                config.tactile_keys,
                source_rename,
                metadata.camera_keys,
            )
        )
        tactile_embedding_cache = None
        if self.tactile_embedding_cache_root is not None:
            if not config.tactile_encoder_path:
                raise ValueError(
                    "tactile_encoder_path is required when using a tactile embedding cache"
                )
            cache_dir = tactile_cache_dir(
                self.tactile_embedding_cache_root,
                source.repo_id,
            )
            tactile_embedding_cache = TactileEmbeddingCache(
                cache_dir,
                repo_id=source.repo_id,
                revision=metadata.revision,
                total_frames=metadata.total_frames,
                tactile_keys=config.tactile_keys,
                source_tactile_keys=source_tactile_keys,
                embedding_dim=config.tactile_embedding_dim,
                image_size=config.tactile_image_size,
                encoder_path=config.tactile_encoder_path,
                dataset_root=metadata.root,
            )
        else:
            visual_keys = list(dict.fromkeys([*visual_keys, *source_tactile_keys]))
        return visual_keys, {
            "tactile_embedding_cache": tactile_embedding_cache,
        }

    def _make_mapped_dataset(
        self,
        dataset: LeRobotDataset,
        *,
        action_key: str,
        rename_map: Mapping[str, str],
        context: Mapping[str, Any],
    ) -> _VTKeyMappedLeRobotDataset:
        return _VTKeyMappedLeRobotDataset(
            dataset,
            action_key=action_key,
            rename_map=rename_map,
            image_transforms=self.image_transforms,
            image_transform_keys=self.config.image_keys,
            tactile_embedding_cache=context.get("tactile_embedding_cache"),
        )

    def _make_preprocessor(
        self,
        checkpoint: str | Path,
        config: VTSmolVLAConfig,
        *,
        stats: Mapping[str, Mapping[str, Any]],
        local_files_only: bool,
    ) -> VTJaxSmolVLAPreprocessor:
        return VTJaxSmolVLAPreprocessor(
            checkpoint,
            config,
            rename_map={},
            stats=stats,
            local_files_only=local_files_only,
        )

    def _dataset_summary_extension(self, context: Mapping[str, Any]) -> dict[str, Any]:
        cache = context.get("tactile_embedding_cache")
        return {
            "tactile_embedding_cache": None if cache is None else str(cache.cache_dir),
        }

    def _validate_features(
        self,
        config: VTSmolVLAConfig,
        features: Mapping[str, Any],
        action_key: str,
        rename_map: Mapping[str, str] | None,
        *,
        repo_id: str,
    ) -> None:
        super()._validate_features(
            config,
            features,
            action_key,
            rename_map,
            repo_id=repo_id,
        )
        if not config.use_tactile_encoder:
            return
        rename_map = dict(rename_map or {})
        dataset_cameras = {
            rename_map.get(key, key)
            for key, feature in features.items()
            if feature.get("dtype") in ("image", "video")
        }
        missing_tactile = sorted(set(config.tactile_keys) - dataset_cameras)
        if missing_tactile:
            raise ValueError(
                f"dataset {repo_id!r}: missing tactile keys after renaming: {missing_tactile}; "
                f"dataset={sorted(dataset_cameras)}"
            )


VTJaxSmolVLADataLoader = VTLeRobotJaxDataLoader

__all__ = [
    "DatasetSource",
    "VTJaxSmolVLADataLoader",
    "VTLeRobotJaxDataLoader",
    "resolve_model_visual_keys",
]
