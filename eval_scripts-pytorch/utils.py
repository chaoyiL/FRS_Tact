from __future__ import annotations

import dataclasses
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch

from lerobot.datasets import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.policies import make_pre_post_processors
from lerobot.policies.smolvla import SmolVLAPolicy
from lerobot.policies.utils import prepare_observation_for_inference
from lerobot.utils.constants import ACTION, OBS_IMAGES, OBS_STATE


DEFAULT_MODEL = "lerobot/smolvla_base"


@dataclasses.dataclass(frozen=True)
class EpisodeData:
    """Selected, preprocessed frames from one LeRobot episode."""

    frames: tuple[int, ...]
    dataset_indices: tuple[int, ...]
    raw_samples: tuple[dict[str, Any], ...]
    batches: tuple[dict[str, Any], ...]
    actions: tuple[torch.Tensor, ...]
    prompts: tuple[str, ...]
    image_keys: tuple[str, ...]


@dataclasses.dataclass(frozen=True)
class LoadedPolicy:
    policy: SmolVLAPolicy
    preprocess: Any


def parse_json_map(value: str | None, *, argument_name: str = "--rename-map") -> dict[str, str]:
    if value is None:
        return {}
    parsed = json.loads(value)
    if not isinstance(parsed, dict):
        raise ValueError(f"{argument_name} must be a JSON object")
    return {str(key): str(mapped) for key, mapped in parsed.items()}


def parse_key_list(values: Sequence[str] | None) -> tuple[str, ...] | None:
    if values is None:
        return None
    keys: list[str] = []
    for value in values:
        keys.extend(part.strip() for part in value.split(",") if part.strip())
    return tuple(dict.fromkeys(keys))


def load_policy(
    model: str | Path = DEFAULT_MODEL,
    *,
    device: str | None = None,
    rename_map: dict[str, str] | None = None,
) -> LoadedPolicy:
    """Load SmolVLA and its checkpoint-owned preprocessing pipeline."""

    model = str(model)
    policy = SmolVLAPolicy.from_pretrained(model)
    if device is not None:
        policy.config.device = str(torch.device(device))
        policy.to(policy.config.device)
    policy.eval()
    policy.requires_grad_(False)
    policy.config.use_cache = True

    overrides: dict[str, Any] = {
        "device_processor": {"device": policy.config.device},
    }
    if rename_map:
        overrides["rename_observations_processor"] = {"rename_map": rename_map}

    preprocess, _ = make_pre_post_processors(
        policy.config,
        model,
        preprocessor_overrides=overrides,
    )
    return LoadedPolicy(policy=policy, preprocess=preprocess)


def _chw_to_hwc_uint8(image: np.ndarray | torch.Tensor) -> np.ndarray:
    if isinstance(image, torch.Tensor):
        image = image.detach().cpu().numpy()
    if image.ndim != 3:
        raise ValueError(f"Expected a 3D image, got shape {image.shape}")
    if image.shape[0] in (1, 3):
        image = np.transpose(image, (1, 2, 0))
    if image.dtype == np.uint8:
        return np.ascontiguousarray(image)
    if np.issubdtype(image.dtype, np.floating) and image.size and image.max() <= 1.0:
        image = image * 255.0
    return np.ascontiguousarray(np.clip(image, 0, 255).astype(np.uint8))


def _as_numpy_float32(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=np.float32)


def _raw_observation(sample: dict[str, Any], features: dict[str, dict]) -> dict[str, np.ndarray]:
    observation: dict[str, np.ndarray] = {}
    for key, feature in features.items():
        if key not in sample:
            continue
        if feature.get("dtype") in ("image", "video"):
            observation[key] = _chw_to_hwc_uint8(sample[key])
        elif key == OBS_STATE:
            observation[key] = _as_numpy_float32(sample[key])
    if OBS_STATE not in observation:
        raise KeyError(f"Dataset frame is missing required feature {OBS_STATE!r}")
    return observation


def _scalar_int(value: Any) -> int:
    if isinstance(value, torch.Tensor):
        return int(value.detach().cpu().item())
    return int(np.asarray(value).item())


def _select_frames(
    episode_length: int,
    *,
    start_frame: int,
    sample_interval: int | None,
    max_frames: int | None,
    frame_indices: Sequence[int] | None,
) -> tuple[int, ...]:
    if start_frame < 0:
        raise ValueError(f"start_frame must be non-negative, got {start_frame}")
    if frame_indices is not None and sample_interval is not None:
        raise ValueError("frame_indices and sample_interval cannot both be set")

    effective_length = episode_length if max_frames is None else min(episode_length, max_frames)
    if frame_indices is not None:
        selected = tuple(int(frame) for frame in frame_indices)
    elif sample_interval is None:
        selected = (start_frame,)
    else:
        if sample_interval <= 0:
            raise ValueError(f"sample_interval must be positive, got {sample_interval}")
        selected = tuple(range(start_frame, effective_length, sample_interval))

    for frame in selected:
        if frame < 0 or frame >= effective_length:
            raise IndexError(
                f"Frame {frame} is out of range [0, {effective_length - 1}] "
                f"for the selected episode window"
            )
    if not selected:
        raise ValueError("No episode frames were selected")
    return selected


def _validate_action_shape(
    action_chunk: np.ndarray,
    *,
    chunk_size: int,
    action_dim: int,
    action_key: str,
) -> None:
    expected = (chunk_size, action_dim)
    if action_chunk.shape != expected:
        raise ValueError(
            f"Dataset feature {action_key!r} produced action chunk shape {action_chunk.shape}, "
            f"but checkpoint expects {expected}. Check --action-key and use a checkpoint trained "
            "for this dataset/action space."
        )


def load_episode(
    loaded_policy: LoadedPolicy,
    dataset_path: str | Path,
    episode_index: int,
    *,
    action_key: str = "actions",
    start_frame: int = 0,
    sample_interval: int | None = None,
    max_frames: int | None = None,
    frame_indices: Sequence[int] | None = None,
    task_override: str | None = None,
    robot_type: str | None = None,
) -> tuple[EpisodeData, LeRobotDatasetMetadata]:
    """Load one episode with future action chunks and checkpoint preprocessing."""

    policy = loaded_policy.policy
    action_feature = policy.config.action_feature
    if action_feature is None:
        raise ValueError("Checkpoint config does not define an action feature")

    dataset_path = Path(dataset_path).expanduser().resolve()
    if not (dataset_path / "meta" / "info.json").is_file():
        raise FileNotFoundError(f"Missing LeRobot metadata: {dataset_path / 'meta/info.json'}")
    repo_id = dataset_path.name

    meta = LeRobotDatasetMetadata(repo_id=repo_id, root=dataset_path)
    if episode_index < 0 or episode_index >= meta.info.total_episodes:
        raise IndexError(
            f"Episode {episode_index} is out of range [0, {meta.info.total_episodes - 1}]"
        )
    if action_key not in meta.info.features:
        raise KeyError(
            f"Action feature {action_key!r} is not in the dataset. "
            f"Available features: {sorted(meta.info.features)}"
        )

    dataset_action_shape = tuple(meta.info.features[action_key].get("shape", ()))
    expected_action_dim = int(action_feature.shape[0])
    if dataset_action_shape != (expected_action_dim,):
        raise ValueError(
            f"Dataset {action_key!r} has shape {dataset_action_shape}, while checkpoint expects "
            f"action dimension {expected_action_dim}. Use a matching checkpoint or --action-key."
        )

    chunk_size = int(policy.config.chunk_size)
    delta_timestamps = {
        action_key: [step / meta.info.fps for step in range(chunk_size)],
    }
    dataset = LeRobotDataset(
        repo_id=repo_id,
        root=dataset_path,
        episodes=[episode_index],
        revision=meta.info.codebase_version,
        delta_timestamps=delta_timestamps,
    )
    selected_frames = _select_frames(
        len(dataset),
        start_frame=start_frame,
        sample_interval=sample_interval,
        max_frames=max_frames,
        frame_indices=frame_indices,
    )

    batches: list[dict[str, Any]] = []
    actions: list[torch.Tensor] = []
    samples: list[dict[str, Any]] = []
    prompts: list[str] = []
    dataset_indices: list[int] = []

    for frame in selected_frames:
        sample = dataset[frame]
        action_chunk = _as_numpy_float32(sample[action_key])
        _validate_action_shape(
            action_chunk,
            chunk_size=chunk_size,
            action_dim=expected_action_dim,
            action_key=action_key,
        )

        task = str(task_override if task_override is not None else sample.get("task", ""))
        observation = _raw_observation(sample, meta.info.features)
        observation[ACTION] = action_chunk
        inference_frame = prepare_observation_for_inference(
            observation,
            policy.config.device,
            task=task,
            robot_type=robot_type if robot_type is not None else meta.info.robot_type,
        )
        batch = loaded_policy.preprocess(inference_frame)
        if ACTION not in batch:
            raise KeyError("Checkpoint preprocessor did not return normalized 'action'")

        if policy.config.adapt_to_pi_aloha:
            batch[OBS_STATE] = policy._pi_aloha_decode_state(batch[OBS_STATE])
            batch[ACTION] = policy._pi_aloha_encode_actions_inv(batch[ACTION])

        prepared_actions = policy.prepare_action(batch).squeeze(0).to(dtype=torch.float32)
        batches.append(batch)
        actions.append(prepared_actions)
        samples.append(sample)
        prompts.append(task)
        dataset_indices.append(_scalar_int(sample["index"]))

    first_batch = batches[0]
    image_keys = tuple(
        key for key in policy.config.image_features if key in first_batch
    )
    if not image_keys:
        raise ValueError(
            "No checkpoint image features are present after preprocessing. "
            "Use --rename-map to align dataset and checkpoint camera keys."
        )

    return (
        EpisodeData(
            frames=selected_frames,
            dataset_indices=tuple(dataset_indices),
            raw_samples=tuple(samples),
            batches=tuple(batches),
            actions=tuple(actions),
            prompts=tuple(prompts),
            image_keys=image_keys,
        ),
        meta,
    )


def stack_frame_batches(batches: Sequence[dict[str, Any]], keys: Sequence[str]) -> dict[str, torch.Tensor]:
    """Concatenate model-relevant tensors from independently preprocessed frames."""

    if not batches:
        raise ValueError("Cannot stack an empty batch sequence")
    result: dict[str, torch.Tensor] = {}
    for key in keys:
        values = [batch[key] for batch in batches if key in batch]
        if len(values) != len(batches):
            raise KeyError(f"Feature {key!r} is missing from one or more frames")
        if not all(isinstance(value, torch.Tensor) for value in values):
            raise TypeError(f"Feature {key!r} is not tensor-valued after preprocessing")
        result[key] = torch.cat(values, dim=0)
    return result


def infer_modality_keys(
    image_keys: Sequence[str],
    *,
    vision_keys: Sequence[str] | None = None,
    tactile_keys: Sequence[str] | None = None,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Classify image features; explicit CLI lists override name-based defaults."""

    all_keys = tuple(image_keys)
    all_set = set(all_keys)
    explicit_vision = parse_key_list(vision_keys)
    explicit_tactile = parse_key_list(tactile_keys)

    tactile = (
        tuple(explicit_tactile)
        if explicit_tactile is not None
        else tuple(key for key in all_keys if "tactile" in key.lower())
    )
    vision = (
        tuple(explicit_vision)
        if explicit_vision is not None
        else tuple(key for key in all_keys if key not in set(tactile))
    )

    unknown = (set(vision) | set(tactile)) - all_set
    if unknown:
        raise ValueError(
            f"Modality image keys are not present in the checkpoint input: {sorted(unknown)}; "
            f"available keys: {sorted(all_set)}"
        )
    overlap = set(vision) & set(tactile)
    if overlap:
        raise ValueError(f"Vision and tactile key sets overlap: {sorted(overlap)}")
    return vision, tactile


def observation_tensor_keys(episode: EpisodeData) -> tuple[str, ...]:
    return (*episode.image_keys, OBS_STATE, "observation.language.tokens", "observation.language.attention_mask")

