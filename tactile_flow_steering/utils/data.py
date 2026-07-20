"""Online tactile conditioning from LeRobot frames + frozen tactile encoder."""

from __future__ import annotations

import pathlib
from collections.abc import Iterator, Sequence
from typing import Literal

import jax
import jax.numpy as jnp
import numpy as np

from tactile_encoder.utils.checkpoint import TactileEncoderBundle
from tactile_encoder.utils.checkpoint import load_tactile_encoder
from tactile_encoder.utils.data import DEFAULT_LEFT_TACTILE_IMAGE_KEYS
from tactile_encoder.utils.data import DEFAULT_RIGHT_TACTILE_IMAGE_KEYS
from tactile_encoder.utils.image_dataset import create_image_dataset
from tactile_encoder.utils.model import tactile_clip_config_from_dict
from utils.cache import CachedPairs

Array = jax.Array
SplitName = Literal["train", "val"]


def tactile_token_dim_from_encoder(bundle: TactileEncoderBundle) -> int:
    """Per-wrist encode dim used as each cross-attn token feature size."""

    config = tactile_clip_config_from_dict(bundle.metadata["tactile_clip_config"])
    if config.uses_gru:
        return int(config.tactile_image_count * config.gru_hidden_dim)
    return int(config.tactile_image_count * config.embedding_dim)


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


class TactileConditionedBatches:
    """Yields (indices, x_base, gt_action, tactile_tokens) with frozen encoder tokens."""

    def __init__(
        self,
        pairs: CachedPairs,
        *,
        tactile_encoder_dir: pathlib.Path,
        dataset_repo_id: str | None = None,
        dataset_root: pathlib.Path | None = None,
        image_cache_size: int = 4096,
    ):
        self.pairs = pairs
        self.bundle = load_tactile_encoder(tactile_encoder_dir)
        self.tactile_token_dim = tactile_token_dim_from_encoder(self.bundle)
        config = tactile_clip_config_from_dict(self.bundle.metadata["tactile_clip_config"])
        if config.tactile_history != 0:
            raise ValueError(
                "tactile_flow_steering currently requires tactile_history=0 encoder checkpoints; "
                f"got tactile_history={config.tactile_history}."
            )
        if config.tactile_image_count != 2:
            raise ValueError(
                f"Expected tactile_image_count=2, got {config.tactile_image_count}."
            )
        self.image_size = int(config.tactile_image_size)
        repo_id = resolve_dataset_repo_id(pairs, dataset_repo_id=dataset_repo_id)
        # create_image_dataset uses LeRobot hub root; optional local root is not wired there yet.
        del dataset_root  # reserved for future local-root override
        self.dataset = create_image_dataset(
            repo_id,
            image_size=self.image_size,
            cache_size=image_cache_size,
        ).dataset
        self.left_keys = DEFAULT_LEFT_TACTILE_IMAGE_KEYS
        self.right_keys = DEFAULT_RIGHT_TACTILE_IMAGE_KEYS

    def _encode_side(self, images: np.ndarray) -> Array:
        """images: [B, 2, H, W, C] float32 in [0, 1]."""

        tokens = self.bundle.encode(jnp.asarray(images, dtype=jnp.float32), train=False)
        return jax.lax.stop_gradient(tokens)

    def encode_indices(self, dataset_indices: Sequence[int]) -> Array:
        left_parts: list[np.ndarray] = []
        right_parts: list[np.ndarray] = []
        for dataset_index in dataset_indices:
            frame = self.dataset.get_images(
                int(dataset_index),
                [*self.left_keys, *self.right_keys],
                as_float=True,
            )
            left = np.stack([frame[key] for key in self.left_keys], axis=0)
            right = np.stack([frame[key] for key in self.right_keys], axis=0)
            left_parts.append(left)
            right_parts.append(right)
        left_batch = np.stack(left_parts, axis=0)
        right_batch = np.stack(right_parts, axis=0)
        left_tokens = self._encode_side(left_batch)
        right_tokens = self._encode_side(right_batch)
        return jnp.stack([left_tokens, right_tokens], axis=1)

    def batches(
        self,
        split: SplitName,
        *,
        batch_size: int,
        shuffle: bool,
        seed: int,
    ) -> Iterator[tuple[np.ndarray, np.ndarray, np.ndarray, Array]]:
        for indices, x_base, _predicted, gt_action in self.pairs.batches(
            split, batch_size=batch_size, shuffle=shuffle, seed=seed
        ):
            dataset_indices = [
                int(self.pairs.arrays["dataset_index"][cache_index]) for cache_index in indices
            ]
            tactile_tokens = self.encode_indices(dataset_indices)
            yield indices, x_base, gt_action, tactile_tokens
