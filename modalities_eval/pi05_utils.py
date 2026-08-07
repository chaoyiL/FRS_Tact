"""pi0.5 analogue of modalities_eval/utils.py's SmolVLAEvalModel: checkpoint + dataset-sample ->
Observation glue for FRS action_cache generation.

A separate class rather than a SmolVLAEvalModel subclass/branch: pi0.5's observation format
differs enough (fixed base_0_rgb/left_wrist_0_rgb/right_wrist_0_rgb image keys in HWC float32
[-1,1], and -- critically -- state baked into the tokenized prompt rather than passed as a
continuous input, see pi05_jax/tokenizer.py) that sharing SmolVLAEvalModel's code would mean more
branching than duplication.

UNTESTED, like the rest of the pi0.5 integration on this branch (see
src/lerobot/policies/pi05_jax/README.md and pi05_frs_plan.md at the repo root for the full status
and open questions -- especially where `state_stats`/`action_stats` should come from).
"""

from __future__ import annotations

import pathlib
from collections.abc import Mapping
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from lerobot.datasets import LeRobotDatasetMetadata
from lerobot.datasets.sample_utils import (
    action_delta_timestamps,
    lerobot_sample_to_observation,
    resolve_action_key,
)
from lerobot.policies.pi05_jax import Observation, PaligemmaTokenizer, Pi0, Pi0Config, load_pi0
from lerobot.policies.pi05_jax.model import IMAGE_KEYS, IMAGE_RESOLUTION
from lerobot.policies.pi05_jax.normalize import NormStats
from lerobot.policies.pi05_jax.normalize import apply as normalize_apply
from lerobot.policies.pi05_jax.normalize import unapply as normalize_unapply

Array = jax.Array


def _prepare_image(frame: Any) -> np.ndarray:
    """Dataset frame (HWC, uint8 0..255 or float 0..1) -> HWC float32 in [-1, 1].

    LeRobotDataset frames are HWC (see lerobot.policies.smolvla_jax.preprocessing._as_bchw's own
    comment/logic, which transposes *from* HWC) -- unlike SmolVLA's SigLIP (via transformers,
    channel-first), pi0.5's own siglip.py is flax.linen and expects channel-last, so no transpose
    here, only the value-range remap (matches SmolVLA's `image * 2.0 - 1.0` after a 0..1 rescale).
    """
    image = np.asarray(frame, dtype=np.float32)
    if image.max(initial=0.0) > 1.0:
        image = image / 255.0
    return image * 2.0 - 1.0


def _resize(image: np.ndarray) -> np.ndarray:
    from lerobot.policies.pi05_jax import image_tools

    resized = image_tools.resize_with_pad(jnp.asarray(image)[None, ...], *IMAGE_RESOLUTION)
    return np.asarray(resized[0])


class Pi05EvalModel:
    """Checkpoint, normalization and observation-building bundled for FRS action-cache scripts."""

    def __init__(
        self,
        checkpoint: str | pathlib.Path,
        *,
        dataset_repo_id: str,
        dataset_root: str | pathlib.Path | None = None,
        dataset_revision: str | None = None,
        action_key: str | None = None,
        rename_map: Mapping[str, str] | None = None,
        camera_map: Mapping[str, str],
        state_stats: NormStats,
        action_stats: NormStats,
        use_quantile_norm: bool = True,
        action_dim: int = 32,
        action_horizon: int = 50,
        max_token_len: int | None = None,
    ):
        """
        Args:
            camera_map: pi0.5 image key (subset of `base_0_rgb`/`left_wrist_0_rgb`/
                `right_wrist_0_rgb`) -> dataset observation key, *after* `rename_map` is applied
                (e.g. `{"base_0_rgb": "observation.images.camera1"}`). Keys from `IMAGE_KEYS` not
                present in `camera_map` are filled with a masked (all -1, mask=False) empty
                camera, mirroring SmolVLA's `empty_cameras` handling
                (`smolvla_jax/preprocessing.py`).
            state_stats / action_stats: required, not loaded automatically -- see this module's
                docstring and pi05_frs_plan.md for why (no norm stats exist for a brand-new
                dataset in the pretrained pi05_base checkpoint's assets).
            use_quantile_norm: pi0.5 (pi05=True) bakes the discretized *state* into the tokenized
                prompt assuming it is already in [-1, 1] (see pi05_jax/tokenizer.py) -- quantile
                normalization guarantees that range; z-score normalization does not. Leave True
                unless you have a specific reason and have re-checked that assumption.
        """
        missing_cameras = set(camera_map) - set(IMAGE_KEYS)
        if missing_cameras:
            raise ValueError(f"camera_map keys must be a subset of {IMAGE_KEYS}, got extra {missing_cameras}")

        self.config = Pi0Config(pi05=True, action_dim=action_dim, action_horizon=action_horizon, max_token_len=max_token_len)
        self.model: Pi0 = load_pi0(checkpoint, config=self.config)
        self.tokenizer = PaligemmaTokenizer(max_len=self.config.max_token_len)

        self.dataset_repo_id = dataset_repo_id
        self.dataset_root = pathlib.Path(dataset_root).expanduser() if dataset_root is not None else None
        self.dataset_revision = dataset_revision
        metadata = LeRobotDatasetMetadata(dataset_repo_id, root=self.dataset_root, revision=dataset_revision)
        self.dataset_root = metadata.root
        self.dataset_revision = metadata.revision
        self.action_key = resolve_action_key(metadata.features, action_key)
        self.fps = metadata.fps

        self.rename_map = dict(rename_map or {})
        self.camera_map = dict(camera_map)
        self.state_stats = state_stats
        self.action_stats = action_stats
        self.use_quantile_norm = use_quantile_norm

    @property
    def action_horizon(self) -> int:
        return self.config.action_horizon

    @property
    def action_dim(self) -> int:
        return self.config.action_dim

    def _renamed_observation(self, sample: Mapping[str, Any]) -> dict[str, Any]:
        observation = lerobot_sample_to_observation(sample)
        return {self.rename_map.get(key, key): value for key, value in observation.items()}

    def prepare_sample(self, sample: Mapping[str, Any]) -> tuple[Observation, Array, str]:
        renamed = self._renamed_observation(sample)
        if "observation.state" not in renamed:
            raise KeyError("observation.state is required")

        raw_state = np.asarray(renamed["observation.state"], dtype=np.float32)
        norm_state = normalize_apply(raw_state, self.state_stats, use_quantiles=self.use_quantile_norm)
        padded_state = np.zeros((self.action_dim,), dtype=np.float32)
        padded_state[: norm_state.shape[-1]] = norm_state[: self.action_dim]

        images: dict[str, np.ndarray] = {}
        image_masks: dict[str, np.ndarray] = {}
        empty_image = None
        for key in IMAGE_KEYS:
            source_key = self.camera_map.get(key)
            if source_key is not None and source_key in renamed:
                image = _resize(_prepare_image(renamed[source_key]))
                images[key] = image
                image_masks[key] = np.asarray(True)
                empty_image = image
            else:
                images[key] = None  # filled below once we know the resolved image shape
                image_masks[key] = np.asarray(False)
        if empty_image is None:
            raise ValueError(f"camera_map matched none of {IMAGE_KEYS} for sample keys {list(renamed)}")
        for key in IMAGE_KEYS:
            if images[key] is None:
                images[key] = -np.ones_like(empty_image)

        prompt = str(sample.get("task", ""))
        # pi0.5: state is discretized into the tokenized prompt, not a continuous model input
        # (see pi05_jax/tokenizer.py's module docstring / pi0.py:embed_suffix).
        tokens, token_mask = self.tokenizer.tokenize(prompt, state=norm_state)

        observation = Observation(
            images={key: jnp.asarray(value) for key, value in images.items()},
            image_masks={key: jnp.asarray(value) for key, value in image_masks.items()},
            state=jnp.asarray(padded_state),
            tokenized_prompt=jnp.asarray(tokens, dtype=jnp.int32),
            tokenized_prompt_mask=jnp.asarray(token_mask, dtype=jnp.bool_),
        )

        raw_actions = np.asarray(sample[self.action_key], dtype=np.float32)
        norm_actions = normalize_apply(raw_actions, self.action_stats, use_quantiles=self.use_quantile_norm)
        padded_actions = np.zeros((norm_actions.shape[0], self.action_dim), dtype=np.float32)
        padded_actions[:, : norm_actions.shape[-1]] = norm_actions[:, : self.action_dim]

        return observation, jnp.asarray(padded_actions), prompt

    def unnormalize_actions(self, actions: Array) -> Array:
        actions_np = np.asarray(actions)
        real_dim = self.action_stats.mean.shape[-1]
        return jnp.asarray(normalize_unapply(actions_np[..., :real_dim], self.action_stats, use_quantiles=self.use_quantile_norm))


def stack_observations(observations: list[Observation]) -> Observation:
    if not observations:
        raise ValueError("at least one observation is required")
    return jax.tree.map(lambda *values: jnp.stack(values, axis=0), *observations)


def action_delta_timestamps_for(model: Pi05EvalModel) -> dict[str, list[float]]:
    return action_delta_timestamps(model.action_key, model.action_horizon, model.fps)
