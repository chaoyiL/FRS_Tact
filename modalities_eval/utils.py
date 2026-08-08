from __future__ import annotations

import argparse
import json
import pathlib
import warnings
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
from lerobot.policies.smolvla_jax.normalization_protocol import (
    NORMALIZATION_MANIFEST_FILENAME,
    validate_normalization_protocol_integrity,
)
from lerobot.policies.smolvla_jax.preprocessing import JaxSmolVLAPreprocessor
from lerobot.policies.smolvla_jax.training import prepare_params_for_compute
from lerobot.policies.smolvla_jax.validation import contract_from_config, validate_checkpoint

Array = jax.Array


@struct.dataclass
class EvalObservation:
    images: Array
    image_masks: Array
    language_tokens: Array
    language_masks: Array
    state: Array
    image_keys: tuple[str, ...] = struct.field(pytree_node=False)
    state_mask: Array | None = None
    tactile_images: Array | None = None
    tactile_embeddings: Array | None = None
    tactile_masks: Array | None = None
    tactile_keys: tuple[str, ...] = struct.field(pytree_node=False, default=())


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
    action_is_pad: tuple[Array, ...]
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
        normalization_source: str = "checkpoint",
        unsafe_legacy_dataset_normalization: bool = False,
        local_files_only: bool = True,
    ):
        self.checkpoint = resolve_checkpoint(checkpoint, local_files_only=local_files_only)
        self.config = JaxSmolVLAConfig.from_pretrained(self.checkpoint)
        if normalization_source not in ("checkpoint", "dataset"):
            raise ValueError(
                "normalization_source must be 'checkpoint' or 'dataset', "
                f"got {normalization_source!r}"
            )
        protocol_checkpoint = (self.checkpoint / NORMALIZATION_MANIFEST_FILENAME).is_file()
        if normalization_source == "dataset":
            if not unsafe_legacy_dataset_normalization:
                detail = (
                    "protocol checkpoint train-only normalization"
                    if protocol_checkpoint
                    else "legacy dataset normalization"
                )
                raise ValueError(
                    f"{detail} may only be overridden with "
                    "unsafe_legacy_dataset_normalization=True"
                )
            warnings.warn(
                "Unsafe legacy dataset normalization ignores checkpoint train-only stats and "
                "can change evaluation results.",
                RuntimeWarning,
                stacklevel=2,
            )
            if protocol_checkpoint:
                validate_normalization_protocol_integrity(
                    self.checkpoint,
                    required=True,
                )
        else:
            validate_normalization_protocol_integrity(
                self.checkpoint,
                required=protocol_checkpoint,
            )
        if protocol_checkpoint:
            validate_checkpoint(
                self.checkpoint,
                expected=contract_from_config(self.config),
                require_weight=True,
            ).require_valid()
        self.params = prepare_params_for_compute(load_params(self.checkpoint), self.config)
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
        stats = (
            canonicalize_dataset_stats(metadata.stats, self.action_key)
            if normalization_source == "dataset"
            else None
        )
        self.normalization_source = normalization_source
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

    def prepare_sample(self, sample: Mapping[str, Any]) -> tuple[EvalObservation, Array, Array, str]:
        prompt = str(sample.get("task", ""))
        prepared = self.preprocessor.prepare(lerobot_sample_to_observation(sample), prompt)
        observation = EvalObservation(
            images=prepared["images"][0],
            image_masks=prepared["image_masks"][0],
            language_tokens=prepared["language_tokens"][0],
            language_masks=prepared["language_masks"][0],
            state=prepared["state"][0],
            image_keys=self.image_keys_for_sample(sample),
            state_mask=jnp.asarray(True, dtype=jnp.bool_),
            tactile_images=(
                None if prepared.get("tactile_images") is None else prepared["tactile_images"][0]
            ),
            tactile_embeddings=(
                None
                if prepared.get("tactile_embeddings") is None
                else prepared["tactile_embeddings"][0]
            ),
            tactile_masks=(
                None if prepared.get("tactile_masks") is None else prepared["tactile_masks"][0]
            ),
            tactile_keys=tuple(self.config.tactile_keys or ()),
        )
        actions = self.preprocessor.normalize_actions(
            jnp.asarray(np.asarray(sample[self.action_key]), dtype=jnp.float32)
        )
        padding_key = "action_is_pad" if self.action_key == "action" else f"{self.action_key}_is_pad"
        action_is_pad = sample.get(padding_key)
        if action_is_pad is None:
            action_is_pad = jnp.zeros(actions.shape[0], dtype=jnp.bool_)
        else:
            action_is_pad = jnp.asarray(np.asarray(action_is_pad), dtype=jnp.bool_)
        return observation, actions, action_is_pad, prompt

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
                    state_mask=obs.state_mask,
                    tactile_images=obs.tactile_images,
                    tactile_embeddings=obs.tactile_embeddings,
                    tactile_masks=obs.tactile_masks,
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


def require_unpadded_action_chunks(action_is_pad: Sequence[Array], *, operation: str) -> None:
    """Reject selected frames whose fixed-size action chunks contain dataset padding."""

    padded_offsets = tuple(
        offset
        for offset, padding in enumerate(action_is_pad)
        if bool(np.any(np.asarray(jax.device_get(padding), dtype=np.bool_)))
    )
    if padded_offsets:
        raise ValueError(
            f"{operation} requires complete action chunks without padding; selected frame offsets "
            f"{padded_offsets} contain action_is_pad=True. Choose H_safe complete frames."
        )


def load_model(
    checkpoint_dir: str | pathlib.Path,
    *,
    dataset_repo_id: str,
    dataset_root: str | pathlib.Path | None = None,
    dataset_revision: str | None = None,
    action_key: str | None = None,
    rename_map: Mapping[str, str] | None = None,
    normalization_source: str = "checkpoint",
    unsafe_legacy_dataset_normalization: bool = False,
    local_files_only: bool = True,
) -> SmolVLAEvalModel:
    return SmolVLAEvalModel(
        checkpoint_dir,
        dataset_repo_id=dataset_repo_id,
        dataset_root=dataset_root,
        dataset_revision=dataset_revision,
        action_key=action_key,
        rename_map=rename_map,
        normalization_source=normalization_source,
        unsafe_legacy_dataset_normalization=unsafe_legacy_dataset_normalization,
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
    action_is_pad = []
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
        observation, action, padding, prompt = model.prepare_sample(sample)
        raw_samples.append(sample)
        observations.append(observation)
        actions.append(action)
        action_is_pad.append(padding)
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
        action_is_pad=tuple(action_is_pad),
        prompts=tuple(prompts),
    )


def ablate_modality_observation(
    observation: EvalObservation,
    *,
    modality: str,
    **_: Any,
) -> EvalObservation:
    if modality == "vision":
        if not observation.image_keys:
            raise ValueError("checkpoint observation has no vision image slots")
        return observation.replace(image_masks=jnp.zeros_like(observation.image_masks))
    if modality == "tactile":
        if observation.tactile_masks is None or not observation.tactile_keys:
            raise ValueError("checkpoint does not use tactile inputs; tactile ablation is not applicable")
        return observation.replace(tactile_masks=jnp.zeros_like(observation.tactile_masks))
    if modality == "state":
        return observation.replace(
            state_mask=jnp.zeros(observation.state.shape[:-1], dtype=jnp.bool_)
        )
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
        state_mask=observation.state_mask,
        tactile_images=observation.tactile_images,
        tactile_embeddings=observation.tactile_embeddings,
        tactile_masks=observation.tactile_masks,
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
    parser.add_argument(
        "--normalization-source",
        choices=("checkpoint", "dataset"),
        default="checkpoint",
        help=(
            "Use train-only normalization assets saved with the checkpoint (default), or the "
            "selected dataset's global stats with the explicit unsafe override."
        ),
    )
    parser.add_argument(
        "--unsafe-legacy-dataset-normalization",
        action="store_true",
        help=(
            "Allow --normalization-source dataset even though it ignores checkpoint train-only "
            "stats and can invalidate comparisons."
        ),
    )
    parser.add_argument("--allow-download", action="store_true")


def load_model_from_args(args: argparse.Namespace) -> SmolVLAEvalModel:
    return load_model(
        args.checkpoint_dir,
        dataset_repo_id=args.dataset_repo_id,
        dataset_root=args.dataset_root,
        dataset_revision=args.dataset_revision,
        action_key=args.action_key,
        rename_map=parse_rename_map(args.rename_map),
        normalization_source=args.normalization_source,
        unsafe_legacy_dataset_normalization=args.unsafe_legacy_dataset_normalization,
        local_files_only=not args.allow_download,
    )
