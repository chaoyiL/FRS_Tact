from __future__ import annotations

import argparse
import json
import pathlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from flax import struct

from lerobot.datasets import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.policies.smolvla_jax import JaxSmolVLA, JaxSmolVLAConfig
from lerobot.policies.smolvla_jax.checkpoint import load_params, resolve_checkpoint
from lerobot.policies.smolvla_jax.data import (
    action_delta_timestamps,
    canonicalize_dataset_stats,
    lerobot_sample_to_observation,
    resolve_action_key,
)
from lerobot.policies.smolvla_jax.modeling import PrefixContext
from lerobot.policies.smolvla_jax.preprocessing import JaxSmolVLAPreprocessor

Array = jax.Array


@struct.dataclass
class EvalObservation:
    images: Array
    image_masks: Array
    language_tokens: Array
    language_masks: Array
    state: Array
    image_keys: tuple[str, ...] = struct.field(pytree_node=False)


@struct.dataclass
class VelocityContext:
    pad_mask: Array
    cache: tuple[tuple[Array, Array], ...]


@dataclass(frozen=True)
class EpisodeData:
    """One normalized LeRobot episode ready for SmolVLA JAX evaluation."""

    indices: tuple[int, ...]
    frames: tuple[int, ...]
    raw_samples: tuple[dict[str, Any], ...]
    observations: tuple[EvalObservation, ...]
    actions: tuple[Array, ...]
    prompts: tuple[str, ...]


class SmolVLAEvalModel:
    """Checkpoint, preprocessing and functional model bundled for eval scripts."""

    def __init__(
        self,
        checkpoint: str | pathlib.Path,
        *,
        dataset_repo_id: str,
        dataset_root: str | pathlib.Path | None = None,
        dataset_revision: str | None = None,
        action_key: str | None = None,
        rename_map: Mapping[str, str] | None = None,
        local_files_only: bool = True,
    ):
        self.checkpoint = resolve_checkpoint(checkpoint, local_files_only=local_files_only)
        self.config = JaxSmolVLAConfig.from_pretrained(self.checkpoint)
        self.params = load_params(self.checkpoint)
        self.model = JaxSmolVLA(self.config)
        self.dataset_repo_id = dataset_repo_id
        self.dataset_root = pathlib.Path(dataset_root).expanduser() if dataset_root is not None else None
        self.dataset_revision = dataset_revision

        metadata = LeRobotDatasetMetadata(
            dataset_repo_id,
            root=self.dataset_root,
            revision=dataset_revision,
        )
        self.dataset_root = metadata.root
        self.dataset_revision = metadata.revision
        self.action_key = resolve_action_key(metadata.features, action_key)
        stats = canonicalize_dataset_stats(metadata.stats, self.action_key)
        self.preprocessor = JaxSmolVLAPreprocessor(
            self.checkpoint,
            self.config,
            rename_map=rename_map,
            stats=stats,
            local_files_only=local_files_only,
        )
        self._sample_cache: dict[int, Any] = {}

    @property
    def action_horizon(self) -> int:
        return self.config.chunk_size

    @property
    def action_dim(self) -> int:
        return self.config.action_dim

    def image_keys_for_sample(self, sample: Mapping[str, Any]) -> tuple[str, ...]:
        source_by_target = {
            self.preprocessor.rename_map.get(key, key): key
            for key in sample
            if key.startswith("observation.images.")
        }
        renamed_keys = set(source_by_target)
        present = [key for key in self.config.image_keys if key in renamed_keys]
        missing = [key for key in self.config.image_keys if key not in renamed_keys]
        selected = present + missing[: self.config.empty_cameras]
        return tuple(source_by_target.get(key, key) for key in selected)

    def prepare_sample(self, sample: Mapping[str, Any]) -> tuple[EvalObservation, Array, str]:
        prompt = str(sample.get("task", ""))
        prepared = self.preprocessor.prepare(lerobot_sample_to_observation(sample), prompt)
        observation = EvalObservation(
            images=prepared["images"][0],
            image_masks=prepared["image_masks"][0],
            language_tokens=prepared["language_tokens"][0],
            language_masks=prepared["language_masks"][0],
            state=prepared["state"][0],
            image_keys=self.image_keys_for_sample(sample),
        )
        actions = self.preprocessor.normalize_actions(
            jnp.asarray(np.asarray(sample[self.action_key]), dtype=jnp.float32)
        )
        return observation, actions, prompt

    def sample_actions(
        self,
        rng: Array,
        observation: EvalObservation,
        *,
        num_steps: int,
        noise: Array | None = None,
    ) -> Array:
        if num_steps not in self._sample_cache:
            functional_model = self.model

            def sample(params, key, obs, initial_noise):
                return functional_model.sample_actions(
                    params,
                    obs.images,
                    obs.image_masks,
                    obs.language_tokens,
                    obs.language_masks,
                    obs.state,
                    key,
                    noise=initial_noise,
                    num_steps=num_steps,
                )

            self._sample_cache[num_steps] = jax.jit(sample)
        if noise is None:
            noise = jax.random.normal(
                rng,
                (
                    observation.state.shape[0],
                    self.config.chunk_size,
                    self.config.max_action_dim,
                ),
                dtype=jnp.float32,
            )
        elif noise.shape[-1] == self.config.action_dim:
            noise = jnp.pad(noise, ((0, 0), (0, 0), (0, self.config.max_action_dim - noise.shape[-1])))
        return self._sample_cache[num_steps](self.params, rng, observation, noise)


def _as_scalar(value: Any) -> Any:
    value = np.asarray(value)
    if value.shape == ():
        return value.item()
    if value.size == 1:
        return value.reshape(()).item()
    return value


def _scalar(value: Any) -> float:
    return float(np.asarray(jax.device_get(value)).reshape(-1)[0])


def _add_batch_dim(data: Any) -> Any:
    return jax.tree.map(lambda value: jnp.asarray(value)[None, ...], data)


def _batch_observation(observation: EvalObservation) -> EvalObservation:
    return _add_batch_dim(observation)


def _batch_actions(actions: Array) -> Array:
    return jnp.asarray(actions)[None, ...]


def _stack_observations(*observations: EvalObservation) -> EvalObservation:
    if not observations:
        raise ValueError("at least one observation is required")
    return jax.tree.map(
        lambda *values: jnp.stack([jnp.asarray(value) for value in values], axis=0),
        *observations,
    )


def load_model(
    checkpoint_dir: str | pathlib.Path,
    *,
    dataset_repo_id: str,
    dataset_root: str | pathlib.Path | None = None,
    dataset_revision: str | None = None,
    action_key: str | None = None,
    rename_map: Mapping[str, str] | None = None,
    local_files_only: bool = True,
) -> SmolVLAEvalModel:
    return SmolVLAEvalModel(
        checkpoint_dir,
        dataset_repo_id=dataset_repo_id,
        dataset_root=dataset_root,
        dataset_revision=dataset_revision,
        action_key=action_key,
        rename_map=rename_map,
        local_files_only=local_files_only,
    )


def load_episode(
    model: SmolVLAEvalModel,
    episode_index: int | str,
    *,
    start_frame: int = 0,
    sample_interval: int | None = None,
    max_frames: int | None = None,
    frame_indices: Sequence[int] | None = None,
) -> EpisodeData:
    episode_index = int(episode_index)
    dataset = LeRobotDataset(
        model.dataset_repo_id,
        root=model.dataset_root,
        revision=model.dataset_revision,
        episodes=[episode_index],
        delta_timestamps=action_delta_timestamps(
            model.action_key,
            model.config.chunk_size,
            LeRobotDatasetMetadata(
                model.dataset_repo_id,
                root=model.dataset_root,
                revision=model.dataset_revision,
            ).fps,
        ),
    )
    if frame_indices is not None and sample_interval is not None:
        raise ValueError("frame_indices and sample_interval cannot both be set")
    limit = len(dataset) if max_frames is None else min(len(dataset), max_frames)
    if frame_indices is None:
        interval = 1 if sample_interval is None else sample_interval
        if interval <= 0:
            raise ValueError(f"sample_interval must be positive, got {interval}")
        frame_indices = tuple(range(start_frame, limit, interval))

    raw_samples = []
    observations = []
    actions = []
    prompts = []
    indices = []
    frames = []
    for frame in frame_indices:
        if frame < 0 or frame >= len(dataset):
            raise ValueError(
                f"frame {frame} is out of range for episode {episode_index}; "
                f"available frames are 0..{len(dataset) - 1}"
            )
        sample = dataset[frame]
        observation, action, prompt = model.prepare_sample(sample)
        raw_samples.append(sample)
        observations.append(observation)
        actions.append(action)
        prompts.append(prompt)
        indices.append(int(_as_scalar(sample["index"])))
        frames.append(int(frame))
    if not frames:
        raise ValueError(f"no frames selected for episode {episode_index}")
    return EpisodeData(
        indices=tuple(indices),
        frames=tuple(frames),
        raw_samples=tuple(raw_samples),
        observations=tuple(observations),
        actions=tuple(actions),
        prompts=tuple(prompts),
    )


def ablate_modality_observation(
    observation: EvalObservation,
    *,
    modality: str,
    **_: Any,
) -> EvalObservation:
    if modality in ("vision", "tactile"):
        tactile = np.asarray(["tactile" in key.lower() for key in observation.image_keys])
        selected = tactile if modality == "tactile" else ~tactile
        if not np.any(selected):
            raise ValueError(
                f"checkpoint observation has no {modality} image slots: {observation.image_keys}"
            )
        masks = jnp.where(jnp.asarray(selected), False, observation.image_masks)
        return observation.replace(image_masks=masks)
    if modality == "state":
        return observation.replace(state=jnp.zeros_like(observation.state))
    if modality in ("language", "language_prompt"):
        return observation.replace(language_masks=jnp.zeros_like(observation.language_masks))
    raise ValueError(
        f"unsupported modality {modality!r}; expected vision, tactile, state, or language_prompt"
    )


def create_velocity_context(
    model: SmolVLAEvalModel,
    observation: EvalObservation,
) -> VelocityContext:
    prefix = model.model.build_prefix_context(
        model.params,
        observation.images,
        observation.image_masks,
        observation.language_tokens,
        observation.language_masks,
        observation.state,
    )
    return VelocityContext(pad_mask=prefix.pad_mask, cache=prefix.cache)


def predict_velocity_with_context(
    model: SmolVLAEvalModel,
    context: VelocityContext,
    x: Array,
    t: Array,
) -> Array:
    x = jnp.asarray(x, dtype=jnp.float32)
    padded_x = x
    if x.shape[-1] < model.config.max_action_dim:
        padded_x = jnp.pad(x, ((0, 0), (0, 0), (0, model.config.max_action_dim - x.shape[-1])))
    t = jnp.asarray(t, dtype=jnp.float32)
    if t.ndim == 0:
        t = jnp.full((x.shape[0],), t)
    velocity = model.model.denoise_step(
        model.params,
        PrefixContext(pad_mask=context.pad_mask, cache=context.cache),
        padded_x,
        t,
    )
    return velocity[..., : x.shape[-1]].astype(jnp.float32)


def parse_rename_map(value: str | None) -> dict[str, str] | None:
    if value is None:
        return None
    parsed = json.loads(value)
    if not isinstance(parsed, dict):
        raise ValueError("--rename-map must be a JSON object")
    return {str(key): str(target) for key, target in parsed.items()}


def add_eval_data_arguments(parser: argparse.ArgumentParser, *, required: bool = True) -> None:
    parser.add_argument("--checkpoint-dir", required=required, type=pathlib.Path)
    parser.add_argument("--dataset-repo-id", required=required)
    parser.add_argument("--dataset-root", type=pathlib.Path)
    parser.add_argument("--dataset-revision")
    parser.add_argument("--action-key")
    parser.add_argument("--rename-map", help="JSON object overriding checkpoint observation renames")
    parser.add_argument("--allow-download", action="store_true")


def load_model_from_args(args: argparse.Namespace) -> SmolVLAEvalModel:
    return load_model(
        args.checkpoint_dir,
        dataset_repo_id=args.dataset_repo_id,
        dataset_root=args.dataset_root,
        dataset_revision=args.dataset_revision,
        action_key=args.action_key,
        rename_map=parse_rename_map(args.rename_map),
        local_files_only=not args.allow_download,
    )
