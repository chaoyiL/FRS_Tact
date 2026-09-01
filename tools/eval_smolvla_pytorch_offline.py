"""SmolVLA offline prediction alignment, raw dataset inference, metrics, and reports."""

import argparse
import contextlib
import csv
import hashlib
import importlib.metadata
import io
import json
import math
import os
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np

ACTION_DIM = 20
LEFT_TRANSLATION = slice(0, 3)
LEFT_ROTATION = slice(3, 9)
LEFT_GRIPPER = 9
RIGHT_TRANSLATION = slice(10, 13)
RIGHT_ROTATION = slice(13, 19)
RIGHT_GRIPPER = 19
REQUIRED_CONTRACT = (20, 20, 20)
DEFAULT_EPISODE_SELECTION = "202-211"
PER_HORIZON_COLUMNS = (
    "lead_step",
    "valid_steps",
    "mae",
    "rmse",
    "left_translation_mae",
    "right_translation_mae",
    "left_rotation_geodesic_deg",
    "right_rotation_geodesic_deg",
    "left_gripper_mae",
    "right_gripper_mae",
)
PER_EPISODE_COLUMNS = (
    "episode_index",
    "frame_count",
    "valid_steps",
    "mae",
    "rmse",
    "first_10_mae",
    "first_1_mae",
    "left_translation_mae",
    "right_translation_mae",
    "left_rotation_geodesic_deg",
    "right_rotation_geodesic_deg",
    "left_gripper_mae",
    "right_gripper_mae",
)
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_SHARD_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class RawEpisode:
    episode_index: int
    frame_indices: np.ndarray
    states: np.ndarray
    actions: np.ndarray
    observations: list[dict[str, Any]]


@dataclass(frozen=True)
class EvalRuntime:
    config_path: Path
    config: Mapping[str, Any]
    checkpoint: str
    device: Any
    policy: Any
    preprocess: Any
    postprocess: Any
    prepare_frame: Callable[[Mapping[str, Any]], dict[str, Any]]
    torch: Any
    horizon: int
    state_dim: int
    action_dim: int
    model_image_keys: tuple[str, ...]
    dataset_image_keys: tuple[str, ...]


@dataclass(frozen=True)
class EpisodeInferenceResult:
    episode_index: int
    frame_indices: np.ndarray
    pred: np.ndarray
    gt: np.ndarray
    valid: np.ndarray


def _smolvla_runtime():
    """Import the PyTorch deployment runtime that shields official LeRobot from local shadowing."""
    project_root = str(_PROJECT_ROOT)
    inserted = False
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
        inserted = True
    try:
        from deploy_smolvla import pytorch_remote_client
    finally:
        if inserted:
            try:
                sys.path.remove(project_root)
            except ValueError:
                pass
    return pytorch_remote_client


def parse_episode_selection(value: str | None) -> tuple[int, ...] | None:
    if value is None or not str(value).strip():
        return None
    selected: set[int] = set()
    for raw_token in str(value).split(","):
        token = raw_token.strip()
        if not token:
            continue
        if "-" in token:
            parts = token.split("-", maxsplit=1)
            try:
                start, stop = (int(part.strip()) for part in parts)
            except ValueError as exc:
                raise ValueError(f"invalid episode range: {token}") from exc
            if start < 0 or stop < 0 or start > stop:
                raise ValueError(f"invalid episode range: {token}")
            selected.update(range(start, stop + 1))
            continue
        try:
            episode = int(token)
        except ValueError as exc:
            raise ValueError(f"invalid episode: {token}") from exc
        if episode < 0:
            raise ValueError(f"invalid episode: {token}")
        selected.add(episode)
    return tuple(sorted(selected)) if selected else None


def _require_mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return value


def _runtime_torch(client: Any):
    torch_module = getattr(client, "torch", None)
    if torch_module is not None:
        return torch_module
    import torch

    return torch


def _config_section(config: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    return _require_mapping(config.get(name), name)


def _runtime_action_dim(runtime: EvalRuntime | Any) -> int:
    if hasattr(runtime, "action_dim"):
        return int(runtime.action_dim)
    policy = runtime.policy
    config = policy.config
    action_feature = getattr(config, "action_feature", None)
    shape = getattr(action_feature, "shape", None)
    if shape is None and isinstance(action_feature, Mapping):
        shape = action_feature.get("shape")
    if shape:
        return int(tuple(shape)[0])
    return ACTION_DIM


def _runtime_horizon(runtime: EvalRuntime | Any) -> int:
    if hasattr(runtime, "horizon"):
        return int(runtime.horizon)
    return int(runtime.policy.config.chunk_size)


def file_sha256(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_identity(path: Path, *, root: Path | None = None) -> dict[str, Any]:
    stat = path.stat()
    label = str(path.relative_to(root)) if root is not None else str(path)
    return {
        "path": label,
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
        "sha256": file_sha256(path),
    }


def _stable_fingerprint(parts: Any) -> str:
    encoded = json.dumps(_json_safe(parts), allow_nan=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _checkpoint_file_paths(checkpoint: Path) -> list[Path]:
    if checkpoint.is_file():
        return [checkpoint]
    if not checkpoint.is_dir():
        return []
    preferred_names = {
        "adapter_config.json",
        "adapter_model.safetensors",
        "config.json",
        "policy_preprocessor.json",
        "policy_postprocessor.json",
        "preprocessor_config.json",
        "postprocessor_config.json",
        "train_config.json",
        "training_args.bin",
    }
    files = [
        path
        for path in checkpoint.rglob("*")
        if path.is_file()
        and (
            path.name in preferred_names
            or (
                path.name.startswith("policy_preprocessor_step_")
                and path.name.endswith("_processor.safetensors")
            )
            or (
                path.name.startswith("policy_postprocessor_step_")
                and path.name.endswith("_processor.safetensors")
            )
        )
    ]
    return sorted(files or [path for path in checkpoint.rglob("*") if path.is_file()])


def _checkpoint_identity(checkpoint: Path | str) -> dict[str, Any]:
    checkpoint_path = Path(checkpoint)
    files = {
        str(path.relative_to(checkpoint_path)) if checkpoint_path.is_dir() else path.name: _file_identity(
            path, root=checkpoint_path if checkpoint_path.is_dir() else None
        )
        for path in _checkpoint_file_paths(checkpoint_path)
    }
    identity = {"path": str(checkpoint_path), "files": files}
    identity["fingerprint"] = _stable_fingerprint(files)
    return identity


def _dataset_revision(dataset_root: Path | str) -> str | None:
    tree_root = Path(dataset_root) / ".cache" / "huggingface" / "trees"
    if not tree_root.is_dir():
        return None
    for path in sorted(tree_root.glob("*.json")):
        try:
            document = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(document, Mapping):
            for key in ("commit_hash", "revision", "sha", "commit"):
                value = document.get(key)
                if isinstance(value, str) and value:
                    return value
        stem = path.stem
        if len(stem) == 40 and all(character in "0123456789abcdefABCDEF" for character in stem):
            return stem
    return None


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def runtime_versions(runtime: EvalRuntime | Any | None = None) -> dict[str, str | None]:
    torch_module = getattr(runtime, "torch", None) if runtime is not None else None
    torch_version = getattr(torch_module, "__version__", None)
    return {
        "torch": str(torch_version) if torch_version is not None else _package_version("torch"),
        "lerobot": _package_version("lerobot"),
        "peft": _package_version("peft"),
    }


def _deployment_metadata(config: Mapping[str, Any]) -> dict[str, Any]:
    observation = config.get("observation") if isinstance(config.get("observation"), Mapping) else {}
    return {
        "prompt": str(observation.get("language_prompt", "")),
        "rename_map": {str(key): str(value) for key, value in (config.get("rename_map") or {}).items()},
        "gripper_thresholds": _close_thresholds_from_config(config),
    }


def build_reproducibility_metadata(
    *,
    config_path: Path | str,
    dataset_root: Path | str,
    parquet_paths: tuple[Path, ...] | list[Path],
    checkpoint: Path | str,
    config: Mapping[str, Any],
    selected_episodes: tuple[int, ...] | None,
    device: str,
    seed: int,
    versions: Mapping[str, Any] | None = None,
    fps: float | None = None,
) -> dict[str, Any]:
    root = Path(dataset_root)
    parquet_files = [_file_identity(Path(path), root=root) for path in sorted(Path(path) for path in parquet_paths)]
    dataset = {
        "root": str(root),
        "revision": _dataset_revision(root),
        "parquet_files": parquet_files,
    }
    dataset["fingerprint"] = _stable_fingerprint(dataset)
    metadata = {
        "dataset": dataset,
        "checkpoint": _checkpoint_identity(checkpoint),
        "config": _file_identity(Path(config_path)),
        "deployment": _deployment_metadata(config),
        "selected_episodes": None if selected_episodes is None else [int(value) for value in selected_episodes],
        "device": str(device),
        "seed": int(seed),
        "versions": dict(versions or runtime_versions()),
        "fps": 30.0 if fps is None else float(fps),
    }
    metadata["run_fingerprint"] = _stable_fingerprint(metadata)
    return metadata


def _validate_required_contract(runtime: EvalRuntime | Any) -> None:
    actual = (int(runtime.state_dim), int(runtime.action_dim), int(runtime.horizon))
    if actual != REQUIRED_CONTRACT:
        raise ValueError(
            f"offline PyTorch SmolVLA eval requires state/action/horizon contract 20/20/20, got "
            f"{actual[0]}/{actual[1]}/{actual[2]}"
        )


def load_eval_runtime(config_path: Path | str, *, device: str | None = None) -> EvalRuntime:
    client = _smolvla_runtime()
    torch_module = _runtime_torch(client)
    path = Path(config_path)
    config = client._load_config(path)
    control = _config_section(config, "control")
    observation = _config_section(config, "observation")
    checkpoint = client._resolve_checkpoint(str(config["checkpoint"]), path)
    device_value = str(device if device is not None else config.get("device", "cuda"))
    torch_device = torch_module.device(device_value)
    if torch_device.type == "cuda" and not torch_module.cuda.is_available():
        raise RuntimeError("config requests CUDA but torch.cuda.is_available() is false")

    policy = client._load_policy(
        checkpoint,
        revision=None if config.get("revision") is None else str(config["revision"]),
        allow_download=bool(config.get("allow_download", False)),
    )
    policy.config.device = str(torch_device)
    policy.to(torch_device).eval()
    if hasattr(policy, "reset"):
        policy.reset()
    preprocess, postprocess = client.make_pre_post_processors(
        policy.config,
        checkpoint,
        preprocessor_overrides={"device_processor": {"device": str(torch_device)}},
    )
    state_dim, action_dim, model_image_keys = client._policy_contract(policy)
    horizon = int(control["action_horizon"])
    if int(policy.config.chunk_size) != horizon:
        raise ValueError(
            f"checkpoint chunk_size={policy.config.chunk_size} does not match action_horizon={horizon}"
        )
    rename_map = {str(key): str(value) for key, value in (config.get("rename_map") or {}).items()}
    reverse_rename = {model: robot for robot, model in rename_map.items()}
    dataset_image_keys = tuple(reverse_rename.get(key, key) for key in model_image_keys)
    task = str(observation.get("language_prompt", ""))

    def prepare_frame(observation_frame: Mapping[str, Any]) -> dict[str, Any]:
        return client._prepare_frame(
            observation_frame,
            task=task,
            device=torch_device,
            state_dim=state_dim,
            model_image_keys=model_image_keys,
            rename_map=rename_map,
        )

    return EvalRuntime(
        config_path=path,
        config=config,
        checkpoint=checkpoint,
        device=torch_device,
        policy=policy,
        preprocess=preprocess,
        postprocess=postprocess,
        prepare_frame=prepare_frame,
        torch=torch_module,
        horizon=horizon,
        state_dim=int(state_dim),
        action_dim=int(action_dim),
        model_image_keys=tuple(model_image_keys),
        dataset_image_keys=dataset_image_keys,
    )


def decode_image_cell(cell: Any, *, dataset_root: Path | str) -> np.ndarray:
    from PIL import Image

    root = Path(dataset_root)
    source: io.BytesIO | Path
    if isinstance(cell, Mapping):
        payload = cell.get("bytes")
        if payload is not None and len(payload) > 0:
            source = io.BytesIO(bytes(payload))
        else:
            raw_path = cell.get("path")
            if raw_path is None or not str(raw_path):
                raise ValueError("image cell must contain bytes or path")
            path = Path(str(raw_path)).expanduser()
            source = path if path.is_absolute() else root / path
    else:
        array = np.asarray(cell)
        if array.ndim == 3 and array.shape[-1] == 3:
            return array.astype(np.uint8, copy=False)
        raise ValueError(f"unsupported image cell type: {type(cell).__name__}")

    with Image.open(source) as image:
        decoded = np.asarray(image.convert("RGB"), dtype=np.uint8)
    if decoded.ndim != 3 or decoded.shape[-1] != 3:
        raise ValueError(f"decoded image must be HWC RGB, got {decoded.shape}")
    return decoded


def _parquet_paths(dataset_root: Path) -> list[Path]:
    data_root = dataset_root / "data"
    paths = sorted(data_root.glob("*/*.parquet"))
    if not paths:
        paths = sorted(data_root.glob("*.parquet"))
    if not paths:
        paths = sorted(dataset_root.glob("*.parquet"))
    if not paths:
        raise FileNotFoundError(f"no parquet files found under {dataset_root}")
    return paths


def _read_parquet_rows(
    dataset_root: Path, *, episodes: tuple[int, ...] | None, columns: list[str]
) -> list[dict[str, Any]]:
    import pyarrow.dataset as pa_ds

    dataset = pa_ds.dataset([str(path) for path in _parquet_paths(dataset_root)], format="parquet")
    available = set(dataset.schema.names)
    missing = [column for column in columns if column not in available]
    if missing:
        raise KeyError(f"raw LeRobot parquet is missing required columns: {missing}")
    filter_expr = pa_ds.field("episode_index").isin(list(episodes)) if episodes is not None else None
    table = dataset.to_table(columns=columns, filter=filter_expr)
    return table.to_pylist()


def _episode_length(row: Mapping[str, Any]) -> int | None:
    for key in ("episode_length", "length", "num_frames", "frame_count"):
        value = row.get(key)
        if value is not None:
            return int(value)
    if row.get("dataset_from_index") is not None and row.get("dataset_to_index") is not None:
        return int(row["dataset_to_index"]) - int(row["dataset_from_index"])
    return None


def load_episode_lengths(dataset_root: Path | str) -> dict[int, int]:
    path = Path(dataset_root) / "meta" / "episodes.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"complete evaluation requires metadata file: {path}")
    lengths: dict[int, int] = {}
    with path.open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            document = json.loads(line)
            if not isinstance(document, Mapping) or "episode_index" not in document:
                raise ValueError(f"invalid episode metadata line {line_number}: missing episode_index")
            length = _episode_length(document)
            if length is None:
                continue
            if length < 0:
                raise ValueError(f"invalid episode length for episode {document['episode_index']}: {length}")
            lengths[int(document["episode_index"])] = int(length)
    return lengths


def load_dataset_fps(dataset_root: Path | str) -> float:
    info_path = Path(dataset_root) / "meta" / "info.json"
    if not info_path.exists():
        return 30.0
    try:
        info = json.loads(info_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid dataset info JSON: {info_path}") from exc
    if not isinstance(info, Mapping) or info.get("fps") is None:
        return 30.0
    fps = float(info["fps"])
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError(f"dataset FPS must be positive, got {fps}")
    return fps


def _validate_episode_rows(
    *,
    requested: tuple[int, ...] | None,
    grouped: Mapping[int, list[dict[str, Any]]],
    expected_lengths: Mapping[int, int],
) -> None:
    if requested is not None:
        missing = sorted(set(int(value) for value in requested) - set(grouped))
        if missing:
            raise ValueError(f"missing requested episodes in parquet data: {missing}")
    for episode_index, episode_rows in grouped.items():
        frame_indices = [int(row["frame_index"]) for row in episode_rows]
        duplicates = sorted({frame for frame in frame_indices if frame_indices.count(frame) > 1})
        if duplicates:
            raise ValueError(f"duplicate frame_index values for episode {episode_index}: {duplicates}")
        expected_length = expected_lengths.get(int(episode_index), len(frame_indices))
        expected_frames = list(range(expected_length))
        actual_frames = sorted(frame_indices)
        if actual_frames != expected_frames:
            raise ValueError(
                f"episode {episode_index} frame_index values must be contiguous 0..{expected_length - 1}, "
                f"got {actual_frames}"
            )


def _as_vector(value: Any, *, expected_dim: int, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.shape != (expected_dim,) or not np.isfinite(array).all():
        raise ValueError(f"{name} must be finite shape {(expected_dim,)}, got {array.shape}")
    return array


def load_raw_lerobot_episodes(
    *,
    dataset_root: Path | str,
    episodes: tuple[int, ...] | None,
    image_keys: tuple[str, ...],
    state_dim: int,
    action_dim: int,
) -> list[RawEpisode]:
    root = Path(dataset_root)
    columns = ["episode_index", "frame_index", "observation.state", "actions", *image_keys]
    rows = _read_parquet_rows(root, episodes=episodes, columns=list(dict.fromkeys(columns)))
    grouped: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(int(row["episode_index"]), []).append(row)
    _validate_episode_rows(
        requested=episodes,
        grouped=grouped,
        expected_lengths=load_episode_lengths(root),
    )

    raw_episodes: list[RawEpisode] = []
    for episode_index in sorted(grouped):
        episode_rows = sorted(grouped[episode_index], key=lambda row: int(row["frame_index"]))
        frame_indices: list[int] = []
        states: list[np.ndarray] = []
        actions: list[np.ndarray] = []
        observations: list[dict[str, Any]] = []
        for row in episode_rows:
            state = _as_vector(row["observation.state"], expected_dim=state_dim, name="observation.state")
            action = _as_vector(row["actions"], expected_dim=action_dim, name="actions")
            observation: dict[str, Any] = {"observation.state": state.copy()}
            for image_key in image_keys:
                observation[image_key] = decode_image_cell(row[image_key], dataset_root=root)
            frame_indices.append(int(row["frame_index"]))
            states.append(state)
            actions.append(action)
            observations.append(observation)
        raw_episodes.append(
            RawEpisode(
                episode_index=episode_index,
                frame_indices=np.asarray(frame_indices, dtype=np.int64),
                states=np.stack(states).astype(np.float32, copy=False),
                actions=np.stack(actions).astype(np.float32, copy=False),
                observations=observations,
            )
        )
    return raw_episodes


def frame_seed(seed: int, *, episode_index: int, frame_index: int) -> int:
    return (int(seed) + int(episode_index) * 1_000_003 + int(frame_index)) % (2**63 - 1)


def _predict_chunk(runtime: EvalRuntime | Any, observation: Mapping[str, Any]) -> np.ndarray:
    frame = runtime.prepare_frame(observation)
    inference_mode = getattr(runtime.torch, "inference_mode", None)
    context = inference_mode() if callable(inference_mode) else contextlib.nullcontext()
    with context:
        action = runtime.postprocess(runtime.policy.predict_action_chunk(runtime.preprocess(frame)))
    if hasattr(action, "detach"):
        action = action.detach().cpu().numpy()
    action_array = np.asarray(action, dtype=np.float32)
    expected = (1, _runtime_horizon(runtime), _runtime_action_dim(runtime))
    if action_array.shape != expected:
        raise ValueError(f"Expected PyTorch SmolVLA action shaped {expected}, got {action_array.shape}")
    if not np.isfinite(action_array).all():
        raise ValueError("SmolVLA action contains NaN or Inf")
    return action_array[0].astype(np.float32, copy=False)


def run_episode_inference(episode: RawEpisode, *, runtime: EvalRuntime | Any, seed: int) -> EpisodeInferenceResult:
    if hasattr(runtime.policy, "reset"):
        runtime.policy.reset()
    horizon = _runtime_horizon(runtime)
    predictions: list[np.ndarray] = []
    for frame_index, observation in zip(episode.frame_indices, episode.observations, strict=True):
        runtime.torch.manual_seed(frame_seed(seed, episode_index=episode.episode_index, frame_index=int(frame_index)))
        predictions.append(_predict_chunk(runtime, observation))
    pred = np.stack(predictions).astype(np.float32, copy=False)
    gt, valid = build_gt_chunks(episode.actions, horizon)
    return EpisodeInferenceResult(
        episode_index=episode.episode_index,
        frame_indices=episode.frame_indices.astype(np.int64, copy=False),
        pred=pred,
        gt=gt,
        valid=valid,
    )


def _shard_path(output_dir: Path | str, episode_index: int) -> Path:
    return Path(output_dir) / "shards" / f"episode_{int(episode_index):06d}.npz"


def episode_shard_metadata(
    *,
    config_path: Path | str,
    checkpoint: str,
    dataset_root: Path | str,
    seed: int,
    device: str,
    episode_index: int,
    frame_count: int,
    horizon: int,
    state_dim: int,
    action_dim: int,
    image_keys: tuple[str, ...],
    reproducibility: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    metadata = {
        "schema_version": _SHARD_SCHEMA_VERSION,
        "complete": True,
        "config_path": str(Path(config_path)),
        "checkpoint": str(checkpoint),
        "dataset_root": str(Path(dataset_root)),
        "seed": int(seed),
        "device": str(device),
        "episode_index": int(episode_index),
        "frame_count": int(frame_count),
        "horizon": int(horizon),
        "state_dim": int(state_dim),
        "action_dim": int(action_dim),
        "image_keys": list(image_keys),
    }
    if reproducibility is not None:
        metadata["reproducibility"] = _json_safe(dict(reproducibility))
    return metadata


def save_episode_shard(
    *,
    output_dir: Path | str,
    episode_index: int,
    pred: Any,
    gt: Any,
    valid: Any,
    frame_indices: Any,
    metadata: Mapping[str, Any],
) -> Path:
    pred_array = _as_actions(pred, expected_ndim=3, name="pred").astype(np.float32)
    gt_array = _as_actions(gt, expected_ndim=3, name="gt").astype(np.float32)
    if pred_array.shape != gt_array.shape:
        raise ValueError(f"pred and gt must share a shape, got {pred_array.shape} vs {gt_array.shape}")
    valid_mask = _as_valid_mask(valid, shape=pred_array.shape[:2])
    frames = np.asarray(frame_indices, dtype=np.int64)
    if frames.shape != (pred_array.shape[0],):
        raise ValueError(f"frame_indices must have shape {(pred_array.shape[0],)}, got {frames.shape}")
    path = _shard_path(output_dir, episode_index)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(".tmp.npz")
    np.savez_compressed(
        tmp_path,
        pred=pred_array,
        gt=gt_array,
        valid=valid_mask,
        frame_indices=frames,
        metadata_json=json.dumps(_json_safe(dict(metadata)), allow_nan=False, sort_keys=True),
    )
    os.replace(tmp_path, path)
    return path


def load_matching_episode_shard(
    output_dir: Path | str, *, episode_index: int, expected_metadata: Mapping[str, Any]
) -> dict[str, Any] | None:
    path = _shard_path(output_dir, episode_index)
    if not path.exists():
        return None
    try:
        with np.load(path, allow_pickle=False) as archive:
            metadata = json.loads(str(archive["metadata_json"].item()))
            if metadata != _json_safe(dict(expected_metadata)) or metadata.get("complete") is not True:
                return None
            return {
                "pred": archive["pred"].copy(),
                "gt": archive["gt"].copy(),
                "valid": archive["valid"].copy(),
                "frame_indices": archive["frame_indices"].copy(),
                "metadata": metadata,
            }
    except (OSError, KeyError, ValueError, json.JSONDecodeError):
        return None


def _close_thresholds_from_config(config: Mapping[str, Any]) -> dict[str, float]:
    gripper = config.get("gripper")
    if isinstance(gripper, Mapping):
        return {
            "left": float(gripper.get("left_close_threshold", 0.5)),
            "right": float(gripper.get("right_close_threshold", 0.5)),
        }
    return {"left": 0.5, "right": 0.5}


def run_offline_eval(
    *,
    config_path: Path | str,
    dataset_root: Path | str,
    episodes: str | tuple[int, ...] | None,
    output_dir: Path | str,
    device: str | None = None,
    seed: int = 0,
) -> dict[str, Any]:
    selected_episodes = parse_episode_selection(episodes) if isinstance(episodes, str) else episodes
    if selected_episodes is None:
        selected_episodes = parse_episode_selection(DEFAULT_EPISODE_SELECTION)
    runtime = load_eval_runtime(config_path, device=device)
    _validate_required_contract(runtime)
    dataset_path = Path(dataset_root)
    parquet_paths = tuple(_parquet_paths(dataset_path))
    fps = load_dataset_fps(dataset_path)
    run_metadata = build_reproducibility_metadata(
        config_path=runtime.config_path,
        dataset_root=dataset_path,
        parquet_paths=parquet_paths,
        checkpoint=runtime.checkpoint,
        config=runtime.config,
        selected_episodes=selected_episodes,
        device=str(runtime.device),
        seed=seed,
        versions=runtime_versions(runtime),
        fps=fps,
    )
    raw_episodes = load_raw_lerobot_episodes(
        dataset_root=dataset_path,
        episodes=selected_episodes,
        image_keys=runtime.dataset_image_keys,
        state_dim=runtime.state_dim,
        action_dim=runtime.action_dim,
    )
    if not raw_episodes:
        raise ValueError("no frames matched the requested episodes")

    episode_payloads: list[dict[str, Any]] = []
    for episode in raw_episodes:
        metadata = episode_shard_metadata(
            config_path=runtime.config_path,
            checkpoint=runtime.checkpoint,
            dataset_root=dataset_root,
            seed=seed,
            device=str(runtime.device),
            episode_index=episode.episode_index,
            frame_count=len(episode.frame_indices),
            horizon=runtime.horizon,
            state_dim=runtime.state_dim,
            action_dim=runtime.action_dim,
            image_keys=runtime.dataset_image_keys,
            reproducibility=run_metadata,
        )
        loaded = load_matching_episode_shard(output_dir, episode_index=episode.episode_index, expected_metadata=metadata)
        if loaded is None:
            result = run_episode_inference(episode, runtime=runtime, seed=seed)
            save_episode_shard(
                output_dir=output_dir,
                episode_index=episode.episode_index,
                pred=result.pred,
                gt=result.gt,
                valid=result.valid,
                frame_indices=result.frame_indices,
                metadata=metadata,
            )
            loaded = load_matching_episode_shard(
                output_dir, episode_index=episode.episode_index, expected_metadata=metadata
            )
            if loaded is None:
                raise RuntimeError(f"failed to save complete shard for episode {episode.episode_index}")
        loaded["episode_index"] = episode.episode_index
        episode_payloads.append(loaded)

    pred = np.concatenate([payload["pred"] for payload in episode_payloads], axis=0)
    gt = np.concatenate([payload["gt"] for payload in episode_payloads], axis=0)
    valid = np.concatenate([payload["valid"] for payload in episode_payloads], axis=0)
    frame_indices = np.concatenate([payload["frame_indices"] for payload in episode_payloads], axis=0)
    episode_indices = np.concatenate(
        [
            np.full(payload["frame_indices"].shape, int(payload["episode_index"]), dtype=np.int64)
            for payload in episode_payloads
        ],
        axis=0,
    )
    return write_reports(
        output_dir=output_dir,
        pred=pred,
        gt=gt,
        valid=valid,
        episode_indices=episode_indices,
        frame_indices=frame_indices,
        close_thresholds=_close_thresholds_from_config(runtime.config),
        metadata=run_metadata,
    )


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path, help="PyTorch SmolVLA deploy YAML.")
    parser.add_argument("--dataset-root", required=True, type=Path, help="Raw LeRobot v2.1 dataset root.")
    parser.add_argument(
        "--episodes",
        default=DEFAULT_EPISODE_SELECTION,
        help=f"Episode selection as CSV/ranges, e.g. '0,2-4'. Defaults to {DEFAULT_EPISODE_SELECTION}.",
    )
    parser.add_argument("--output-dir", required=True, type=Path, help="Directory for shards and reports.")
    parser.add_argument("--device", default=None, help="Torch device override, e.g. cpu or cuda:0.")
    parser.add_argument("--seed", default=0, type=int, help="Base deterministic inference seed.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    paths = run_offline_eval(
        config_path=args.config,
        dataset_root=args.dataset_root,
        episodes=args.episodes,
        output_dir=args.output_dir,
        device=args.device,
        seed=args.seed,
    )
    print(f"[eval] Wrote predictions: {paths['predictions']}")
    print(f"[eval] Wrote metrics: {paths['metrics']}")
    return 0


def _as_actions(value: Any, *, expected_ndim: int, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != expected_ndim:
        raise ValueError(f"{name} must have rank {expected_ndim}, got shape {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


def _as_valid_mask(valid: Any, *, shape: tuple[int, int]) -> np.ndarray:
    mask = np.asarray(valid, dtype=bool)
    if mask.shape != shape:
        raise ValueError(f"valid must have shape {shape}, got {mask.shape}")
    return mask


def _close_thresholds_dict(close_thresholds: Mapping[str, Any]) -> dict[str, float]:
    try:
        left = float(close_thresholds["left"])
        right = float(close_thresholds["right"])
    except KeyError as exc:
        raise ValueError("close_thresholds must define left and right") from exc
    thresholds = {"left": left, "right": right}
    for name, value in thresholds.items():
        if not math.isfinite(value):
            raise ValueError(f"{name} close threshold must be finite")
    return thresholds


def build_gt_chunks(actions: Any, horizon: int) -> tuple[np.ndarray, np.ndarray]:
    action_array = _as_actions(actions, expected_ndim=2, name="actions")
    if horizon <= 0:
        raise ValueError("horizon must be positive")

    steps, action_dim = action_array.shape
    gt = np.zeros((steps, horizon, action_dim), dtype=action_array.dtype)
    valid = np.zeros((steps, horizon), dtype=bool)
    terminal = np.all(action_array == 0.0, axis=1)

    for start in range(steps):
        seen_terminal = False
        for lead in range(horizon):
            index = start + lead
            if index >= steps:
                continue
            gt[start, lead] = action_array[index]
            if seen_terminal or terminal[index]:
                seen_terminal = True
                continue
            valid[start, lead] = True
    return gt.astype(np.float32), valid


def _masked_element_metrics(
    pred: np.ndarray, gt: np.ndarray, valid: np.ndarray
) -> tuple[int, int, float | None, float | None]:
    valid_steps = int(np.count_nonzero(valid))
    if valid_steps == 0:
        return 0, 0, None, None

    diff = pred - gt
    valid_elements = int(valid_steps * pred.shape[-1])
    weights = valid[..., None].astype(np.float64)
    mae = float((np.abs(diff) * weights).sum() / valid_elements)
    rmse = float(np.sqrt((np.square(diff) * weights).sum() / valid_elements))
    return valid_steps, valid_elements, mae, rmse


def _group_mae(pred: np.ndarray, gt: np.ndarray, valid: np.ndarray, dim_slice: slice | int) -> float | None:
    pred_group = pred[..., dim_slice]
    gt_group = gt[..., dim_slice]
    if pred_group.ndim == 2:
        pred_group = pred_group[..., None]
        gt_group = gt_group[..., None]
    valid_steps = int(np.count_nonzero(valid))
    if valid_steps == 0:
        return None
    valid_elements = valid_steps * pred_group.shape[-1]
    weights = valid[..., None].astype(np.float64)
    return float((np.abs(pred_group - gt_group) * weights).sum() / valid_elements)


def _rotation6d_to_matrix(rotation6d: np.ndarray) -> np.ndarray:
    first = rotation6d[..., 0:3]
    second = rotation6d[..., 3:6]

    first_norm = np.linalg.norm(first, axis=-1, keepdims=True)
    second_residual = second - np.sum(first * second, axis=-1, keepdims=True) * first / np.clip(
        first_norm**2, 1e-12, None
    )
    second_norm = np.linalg.norm(second_residual, axis=-1, keepdims=True)

    if np.any(first_norm < 1e-12) or np.any(second_norm < 1e-12):
        raise ValueError("rotation 6D vectors must contain two non-collinear finite axes")

    basis_x = first / first_norm
    basis_y = second_residual / second_norm
    basis_z = np.cross(basis_x, basis_y, axis=-1)
    return np.stack((basis_x, basis_y, basis_z), axis=-1)


def _rotation_geodesic_deg(
    pred: np.ndarray, gt: np.ndarray, valid: np.ndarray, rotation_slice: slice
) -> float | None:
    valid_steps = int(np.count_nonzero(valid))
    if valid_steps == 0:
        return None
    pred_rot = pred[..., rotation_slice][valid]
    gt_rot = gt[..., rotation_slice][valid]
    pred_matrix = _rotation6d_to_matrix(pred_rot)
    gt_matrix = _rotation6d_to_matrix(gt_rot)
    relative = np.matmul(pred_matrix, np.swapaxes(gt_matrix, -1, -2))
    trace = np.trace(relative, axis1=-2, axis2=-1)
    cosine = np.clip((trace - 1.0) / 2.0, -1.0, 1.0)
    return float(np.rad2deg(np.arccos(cosine)).mean())


def _safe_ratio(numerator: float, denominator: float) -> float | None:
    if denominator == 0.0:
        return None
    return numerator / denominator


def _f1(precision: float | None, recall: float | None) -> float | None:
    if precision is None or recall is None or precision + recall == 0.0:
        return None
    return 2.0 * precision * recall / (precision + recall)


def _first_close_onset(values: np.ndarray, valid: np.ndarray, threshold: float) -> int | None:
    valid_indices = np.flatnonzero(valid)
    if valid_indices.size < 2:
        return None
    previous_closed = bool(values[valid_indices[0]] <= threshold)
    for index in valid_indices[1:]:
        current_closed = bool(values[index] <= threshold)
        if not previous_closed and current_closed:
            return int(index)
        previous_closed = current_closed
    return None


def _gripper_binary_metrics(
    pred: np.ndarray, gt: np.ndarray, valid: np.ndarray, threshold: float
) -> dict[str, Any]:
    pred_closed = pred[valid] <= threshold
    gt_closed = gt[valid] <= threshold
    true_positive = float(np.count_nonzero(pred_closed & gt_closed))
    false_positive = float(np.count_nonzero(pred_closed & ~gt_closed))
    false_negative = float(np.count_nonzero(~pred_closed & gt_closed))
    precision = _safe_ratio(true_positive, true_positive + false_positive)
    recall = _safe_ratio(true_positive, true_positive + false_negative)
    return {
        "precision": precision,
        "recall": recall,
        "f1": _f1(precision, recall),
        "support": int(np.count_nonzero(gt_closed)),
        "predicted_positive_count": int(np.count_nonzero(pred_closed)),
    }


def _gripper_event_metrics(
    pred: np.ndarray, gt: np.ndarray, valid: np.ndarray, threshold: float
) -> dict[str, Any]:
    true_positive = 0.0
    false_positive = 0.0
    false_negative = 0.0
    timing_errors: list[float] = []

    for row_index in range(pred.shape[0]):
        gt_onset = _first_close_onset(gt[row_index], valid[row_index], threshold)
        pred_onset = _first_close_onset(pred[row_index], valid[row_index], threshold)
        if gt_onset is None and pred_onset is None:
            continue
        if gt_onset is not None and pred_onset is not None:
            true_positive += 1.0
            timing_errors.append(abs(pred_onset - gt_onset))
            continue
        if gt_onset is None:
            false_positive += 1.0
        else:
            false_negative += 1.0

    precision = _safe_ratio(true_positive, true_positive + false_positive)
    recall = _safe_ratio(true_positive, true_positive + false_negative)
    return {
        "precision": precision,
        "recall": recall,
        "f1": _f1(precision, recall),
        "support": int(true_positive + false_negative),
        "predicted_positive_count": int(true_positive + false_positive),
        "onset_timing_mae": float(np.mean(timing_errors)) if timing_errors else None,
    }


def compute_timeline_gripper_metrics(
    *,
    pred: Any,
    gt: Any,
    valid: Any,
    episode_indices: Any,
    frame_indices: Any,
    close_thresholds: Mapping[str, Any],
    fps: float,
) -> dict[str, Any]:
    pred_array = _as_actions(pred, expected_ndim=3, name="pred")
    gt_array = _as_actions(gt, expected_ndim=3, name="gt")
    if pred_array.shape != gt_array.shape:
        raise ValueError(f"pred and gt must share a shape, got {pred_array.shape} vs {gt_array.shape}")
    valid_mask = _as_valid_mask(valid, shape=pred_array.shape[:2])
    episodes = np.asarray(episode_indices, dtype=np.int64)
    frames = np.asarray(frame_indices, dtype=np.int64)
    if episodes.shape != (pred_array.shape[0],):
        raise ValueError(f"episode_indices must have shape {(pred_array.shape[0],)}, got {episodes.shape}")
    if frames.shape != (pred_array.shape[0],):
        raise ValueError(f"frame_indices must have shape {(pred_array.shape[0],)}, got {frames.shape}")
    parsed_fps = float(fps)
    if not math.isfinite(parsed_fps) or parsed_fps <= 0:
        raise ValueError(f"fps must be positive, got {fps}")

    thresholds = _close_thresholds_dict(close_thresholds)
    timeline: dict[str, Any] = {}
    for name, index in (("left", LEFT_GRIPPER), ("right", RIGHT_GRIPPER)):
        lead_valid = valid_mask[:, 0]
        state = _gripper_binary_metrics(pred_array[:, 0, index], gt_array[:, 0, index], lead_valid, thresholds[name])
        true_positive = 0.0
        false_positive = 0.0
        false_negative = 0.0
        timing_errors: list[float] = []
        for episode_index in np.unique(episodes):
            mask = episodes == episode_index
            ordering = np.argsort(frames[mask], kind="stable")
            episode_frames = frames[mask][ordering]
            episode_valid = valid_mask[mask, 0][ordering]
            pred_values = pred_array[mask, 0, index][ordering]
            gt_values = gt_array[mask, 0, index][ordering]
            gt_onset = _first_close_onset(gt_values, episode_valid, thresholds[name])
            pred_onset = _first_close_onset(pred_values, episode_valid, thresholds[name])
            if gt_onset is None and pred_onset is None:
                continue
            if gt_onset is not None and pred_onset is not None:
                true_positive += 1.0
                timing_errors.append(abs(float(episode_frames[pred_onset] - episode_frames[gt_onset])))
                continue
            if gt_onset is None:
                false_positive += 1.0
            else:
                false_negative += 1.0
        precision = _safe_ratio(true_positive, true_positive + false_positive)
        recall = _safe_ratio(true_positive, true_positive + false_negative)
        timing_mae_frames = float(np.mean(timing_errors)) if timing_errors else None
        timeline[name] = {
            "threshold": thresholds[name],
            "state": state,
            "close_event": {
                "precision": precision,
                "recall": recall,
                "f1": _f1(precision, recall),
                "support": int(true_positive + false_negative),
                "predicted_positive_count": int(true_positive + false_positive),
                "onset_timing_mae_frames": timing_mae_frames,
                "onset_timing_mae_seconds": (
                    None if timing_mae_frames is None else timing_mae_frames / parsed_fps
                ),
            },
        }
    return timeline


def _window_metrics(pred: np.ndarray, gt: np.ndarray, valid: np.ndarray) -> dict[str, Any]:
    valid_steps, valid_elements, mae, rmse = _masked_element_metrics(pred, gt, valid)
    return {
        "valid_steps": valid_steps,
        "valid_action_elements": valid_elements,
        "mae": mae,
        "rmse": rmse,
        "left_translation_mae": _group_mae(pred, gt, valid, LEFT_TRANSLATION),
        "right_translation_mae": _group_mae(pred, gt, valid, RIGHT_TRANSLATION),
        "left_rotation_geodesic_deg": _rotation_geodesic_deg(pred, gt, valid, LEFT_ROTATION),
        "right_rotation_geodesic_deg": _rotation_geodesic_deg(pred, gt, valid, RIGHT_ROTATION),
        "left_gripper_mae": _group_mae(pred, gt, valid, LEFT_GRIPPER),
        "right_gripper_mae": _group_mae(pred, gt, valid, RIGHT_GRIPPER),
    }


def compute_metrics(
    pred: Any, gt: Any, valid: Any, close_thresholds: Mapping[str, Any]
) -> dict[str, Any]:
    pred_array = _as_actions(pred, expected_ndim=3, name="pred")
    gt_array = _as_actions(gt, expected_ndim=3, name="gt")
    if pred_array.shape != gt_array.shape:
        raise ValueError(f"pred and gt must share a shape, got {pred_array.shape} vs {gt_array.shape}")
    if pred_array.shape[-1] != ACTION_DIM:
        raise ValueError(f"expected action dimension {ACTION_DIM}, got {pred_array.shape[-1]}")
    valid_mask = _as_valid_mask(valid, shape=pred_array.shape[:2])
    thresholds = _close_thresholds_dict(close_thresholds)

    counts = {
        "samples": int(pred_array.shape[0]),
        "horizon": int(pred_array.shape[1]),
        "action_dim": int(pred_array.shape[2]),
        "valid_steps": int(np.count_nonzero(valid_mask)),
        "valid_action_elements": int(np.count_nonzero(valid_mask) * pred_array.shape[2]),
    }
    windows = {
        "full": _window_metrics(pred_array, gt_array, valid_mask),
        "first_10": _window_metrics(
            pred_array[:, : min(10, pred_array.shape[1])],
            gt_array[:, : min(10, gt_array.shape[1])],
            valid_mask[:, : min(10, valid_mask.shape[1])],
        ),
        "first_1": _window_metrics(pred_array[:, :1], gt_array[:, :1], valid_mask[:, :1]),
    }

    per_horizon = []
    for lead_step in range(pred_array.shape[1]):
        horizon_valid = valid_mask[:, lead_step : lead_step + 1]
        horizon_pred = pred_array[:, lead_step : lead_step + 1]
        horizon_gt = gt_array[:, lead_step : lead_step + 1]
        row = {"lead_step": lead_step}
        row.update(_window_metrics(horizon_pred, horizon_gt, horizon_valid))
        per_horizon.append(row)

    gripper = {}
    for name, index in (("left", LEFT_GRIPPER), ("right", RIGHT_GRIPPER)):
        state = _gripper_binary_metrics(
            pred_array[..., index], gt_array[..., index], valid_mask, thresholds[name]
        )
        close_event = _gripper_event_metrics(
            pred_array[..., index], gt_array[..., index], valid_mask, thresholds[name]
        )
        first_10_steps = min(10, pred_array.shape[1])
        first_10_pred = pred_array[:, :first_10_steps, index]
        first_10_gt = gt_array[:, :first_10_steps, index]
        first_10_valid = valid_mask[:, :first_10_steps]
        gripper[name] = {
            "threshold": thresholds[name],
            "chunk_forecast": {"state": state, "close_event": close_event},
            "first_10_forecast": {
                "state": _gripper_binary_metrics(
                    first_10_pred, first_10_gt, first_10_valid, thresholds[name]
                ),
                "close_event": _gripper_event_metrics(
                    first_10_pred, first_10_gt, first_10_valid, thresholds[name]
                ),
            },
            "state": state,
        }

    return {
        "counts": counts,
        "windows": windows,
        "per_horizon": per_horizon,
        "gripper": gripper,
        "close_thresholds": thresholds,
    }


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def _write_csv(path: Path, fieldnames: tuple[str, ...], rows: list[dict[str, Any]]) -> Path:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name) for name in fieldnames})
    return path


def _matplotlib_pyplot():
    cache_dir = Path("/tmp/smolvla_eval_mpl")
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ["MPLBACKEND"] = "Agg"
    os.environ["MPLCONFIGDIR"] = str(cache_dir)
    import matplotlib

    matplotlib.use("Agg", force=True)
    from matplotlib import pyplot as plt

    return plt


def _per_dim_mae(pred: np.ndarray, gt: np.ndarray, valid: np.ndarray) -> np.ndarray:
    diff = np.abs(pred - gt)
    numerator = (diff * valid[..., None]).sum(axis=0)
    denominator = valid.sum(axis=0, keepdims=True).T
    result = np.full((pred.shape[1], pred.shape[2]), np.nan, dtype=np.float64)
    np.divide(numerator, denominator, out=result, where=denominator > 0)
    return result


def _nan_or_none(values: list[float | None]) -> np.ndarray:
    return np.asarray([np.nan if value is None else float(value) for value in values], dtype=np.float64)


def _write_heatmap(path: Path, pred: np.ndarray, gt: np.ndarray, valid: np.ndarray) -> Path:
    plt = _matplotlib_pyplot()
    heatmap = _per_dim_mae(pred, gt, valid)
    figure, axis = plt.subplots(figsize=(12.0, 5.0))
    image = axis.imshow(heatmap.T, aspect="auto", origin="lower", cmap="magma")
    axis.set_xlabel("Lead step")
    axis.set_ylabel("Action dimension")
    axis.set_title("Action MAE by lead step and action dimension")
    figure.colorbar(image, ax=axis, fraction=0.035, pad=0.02, label="MAE")
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return path


def _write_group_error_plot(path: Path, per_horizon: list[dict[str, Any]]) -> Path:
    plt = _matplotlib_pyplot()
    lead_steps = np.asarray([row["lead_step"] for row in per_horizon], dtype=np.int64)
    series = {
        "left_translation_mae": _nan_or_none([row["left_translation_mae"] for row in per_horizon]),
        "right_translation_mae": _nan_or_none([row["right_translation_mae"] for row in per_horizon]),
        "left_rotation_geodesic_deg": _nan_or_none(
            [row["left_rotation_geodesic_deg"] for row in per_horizon]
        ),
        "right_rotation_geodesic_deg": _nan_or_none(
            [row["right_rotation_geodesic_deg"] for row in per_horizon]
        ),
        "left_gripper_mae": _nan_or_none([row["left_gripper_mae"] for row in per_horizon]),
        "right_gripper_mae": _nan_or_none([row["right_gripper_mae"] for row in per_horizon]),
    }

    figure, axes = plt.subplots(3, 1, figsize=(10.0, 9.0), sharex=True)
    groups = (
        ("Translation MAE", ("left_translation_mae", "right_translation_mae")),
        ("Rotation geodesic error (deg)", ("left_rotation_geodesic_deg", "right_rotation_geodesic_deg")),
        ("Gripper MAE", ("left_gripper_mae", "right_gripper_mae")),
    )
    for axis, (title, keys) in zip(axes, groups, strict=True):
        for key in keys:
            axis.plot(lead_steps, series[key], marker="o", linewidth=1.4, label=key)
        axis.set_ylabel(title)
        axis.grid(True, alpha=0.3)
        axis.legend(loc="best", fontsize=8)
    axes[-1].set_xlabel("Lead step")
    figure.suptitle("Grouped error by prediction horizon", y=0.995)
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return path


def _write_gripper_timeline(
    path: Path,
    pred: np.ndarray,
    gt: np.ndarray,
    frame_indices: np.ndarray,
    close_thresholds: Mapping[str, float],
) -> Path:
    plt = _matplotlib_pyplot()
    figure, axes = plt.subplots(2, 1, figsize=(10.0, 6.0), sharex=True)
    for axis, title, index, threshold in (
        (axes[0], "Left gripper", LEFT_GRIPPER, close_thresholds["left"]),
        (axes[1], "Right gripper", RIGHT_GRIPPER, close_thresholds["right"]),
    ):
        axis.plot(frame_indices, gt[:, 0, index], linewidth=1.5, label="gt")
        axis.plot(frame_indices, pred[:, 0, index], linewidth=1.5, label="pred")
        axis.axhline(threshold, linestyle="--", linewidth=1.0, color="black", label="close threshold")
        axis.set_ylabel("Width")
        axis.set_title(title)
        axis.grid(True, alpha=0.3)
        axis.legend(loc="best", fontsize=8)
    axes[-1].set_xlabel("Frame index")
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return path


def _timeline_lead0_slice(
    pred: Any, gt: Any, valid: Any, frame_indices: Any
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pred_array = _as_actions(pred, expected_ndim=3, name="pred")
    gt_array = _as_actions(gt, expected_ndim=3, name="gt")
    if pred_array.shape != gt_array.shape:
        raise ValueError(f"pred and gt must share a shape, got {pred_array.shape} vs {gt_array.shape}")
    valid_mask = _as_valid_mask(valid, shape=pred_array.shape[:2])
    frames = np.asarray(frame_indices, dtype=np.int64)
    if frames.shape != (pred_array.shape[0],):
        raise ValueError(f"frame_indices must have shape {(pred_array.shape[0],)}, got {frames.shape}")
    lead_valid = valid_mask[:, 0]
    return pred_array[lead_valid], gt_array[lead_valid], frames[lead_valid]


def _write_action_timeline(path: Path, pred: np.ndarray, gt: np.ndarray, frame_indices: np.ndarray) -> Path:
    plt = _matplotlib_pyplot()
    figure, axes = plt.subplots(5, 4, figsize=(14.0, 10.0), sharex=True)
    for dim, axis in enumerate(axes.flat):
        axis.plot(frame_indices, gt[:, 0, dim], linewidth=1.1, label="gt")
        axis.plot(frame_indices, pred[:, 0, dim], linewidth=1.1, label="pred")
        axis.set_title(f"dim {dim}", fontsize=9)
        axis.grid(True, alpha=0.3)
    axes[0, 0].legend(loc="best", fontsize=8)
    for axis in axes[-1]:
        axis.set_xlabel("Frame index")
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return path


def _episode_metrics_rows(
    pred: np.ndarray,
    gt: np.ndarray,
    valid: np.ndarray,
    episode_indices: np.ndarray,
    close_thresholds: Mapping[str, float],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows = []
    json_rows = []
    for episode_index in np.unique(episode_indices):
        mask = episode_indices == episode_index
        metrics = compute_metrics(pred[mask], gt[mask], valid[mask], close_thresholds)
        row = {
            "episode_index": int(episode_index),
            "frame_count": int(np.count_nonzero(mask)),
            "valid_steps": metrics["counts"]["valid_steps"],
            "mae": metrics["windows"]["full"]["mae"],
            "rmse": metrics["windows"]["full"]["rmse"],
            "first_10_mae": metrics["windows"]["first_10"]["mae"],
            "first_1_mae": metrics["windows"]["first_1"]["mae"],
            "left_translation_mae": metrics["windows"]["full"]["left_translation_mae"],
            "right_translation_mae": metrics["windows"]["full"]["right_translation_mae"],
            "left_rotation_geodesic_deg": metrics["windows"]["full"]["left_rotation_geodesic_deg"],
            "right_rotation_geodesic_deg": metrics["windows"]["full"]["right_rotation_geodesic_deg"],
            "left_gripper_mae": metrics["windows"]["full"]["left_gripper_mae"],
            "right_gripper_mae": metrics["windows"]["full"]["right_gripper_mae"],
        }
        rows.append(row)
        json_rows.append({"episode_index": int(episode_index), **metrics})
    return rows, json_rows


def write_reports(
    *,
    output_dir: Path | str,
    pred: Any,
    gt: Any,
    valid: Any,
    episode_indices: Any,
    frame_indices: Any,
    close_thresholds: Mapping[str, Any],
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    pred_array = _as_actions(pred, expected_ndim=3, name="pred").astype(np.float32)
    gt_array = _as_actions(gt, expected_ndim=3, name="gt").astype(np.float32)
    if pred_array.shape != gt_array.shape:
        raise ValueError(f"pred and gt must share a shape, got {pred_array.shape} vs {gt_array.shape}")
    valid_mask = _as_valid_mask(valid, shape=pred_array.shape[:2])
    thresholds = _close_thresholds_dict(close_thresholds)
    episodes = np.asarray(episode_indices, dtype=np.int64)
    frames = np.asarray(frame_indices, dtype=np.int64)
    if episodes.shape != (pred_array.shape[0],):
        raise ValueError(f"episode_indices must have shape {(pred_array.shape[0],)}, got {episodes.shape}")
    if frames.shape != (pred_array.shape[0],):
        raise ValueError(f"frame_indices must have shape {(pred_array.shape[0],)}, got {frames.shape}")

    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    run_metadata = _json_safe(dict(metadata or {}))
    fps = float(run_metadata.get("fps", 30.0)) if isinstance(run_metadata, Mapping) else 30.0
    metrics = compute_metrics(pred_array, gt_array, valid_mask, thresholds)
    episode_rows, per_episode_metrics = _episode_metrics_rows(
        pred_array, gt_array, valid_mask, episodes, thresholds
    )
    metrics_document = dict(metrics)
    metrics_document["per_episode"] = per_episode_metrics
    metrics_document["gripper_timeline"] = compute_timeline_gripper_metrics(
        pred=pred_array,
        gt=gt_array,
        valid=valid_mask,
        episode_indices=episodes,
        frame_indices=frames,
        close_thresholds=thresholds,
        fps=fps,
    )
    metrics_document["run_metadata"] = run_metadata

    predictions_path = output_root / "predictions.npz"
    np.savez_compressed(
        predictions_path,
        pred=pred_array,
        gt=gt_array,
        valid=valid_mask,
        episode_indices=episodes,
        frame_indices=frames,
        metadata_json=json.dumps(run_metadata, allow_nan=False, sort_keys=True),
    )

    metrics_path = output_root / "metrics.json"
    with metrics_path.open("w", encoding="utf-8") as file:
        json.dump(_json_safe(metrics_document), file, allow_nan=False, indent=2, sort_keys=True)
        file.write("\n")

    per_horizon_path = _write_csv(
        output_root / "per_horizon.csv",
        PER_HORIZON_COLUMNS,
        [{column: row.get(column) for column in PER_HORIZON_COLUMNS} for row in metrics["per_horizon"]],
    )
    episode_metrics_path = _write_csv(output_root / "episode_metrics.csv", PER_EPISODE_COLUMNS, episode_rows)
    heatmap_path = _write_heatmap(output_root / "action_error_heatmap.png", pred_array, gt_array, valid_mask)
    group_plot_path = _write_group_error_plot(
        output_root / "group_error_by_horizon.png", metrics["per_horizon"]
    )

    gripper_timeline_paths = []
    action_timeline_paths = []
    for episode_index in np.unique(episodes):
        episode_mask = episodes == episode_index
        ordering = np.argsort(frames[episode_mask], kind="stable")
        episode_frames = frames[episode_mask][ordering]
        episode_pred = pred_array[episode_mask][ordering]
        episode_gt = gt_array[episode_mask][ordering]
        episode_valid = valid_mask[episode_mask][ordering]
        episode_pred, episode_gt, episode_frames = _timeline_lead0_slice(
            episode_pred, episode_gt, episode_valid, episode_frames
        )
        gripper_timeline_paths.append(
            _write_gripper_timeline(
                output_root / f"gripper_timeline_episode_{int(episode_index)}.png",
                episode_pred,
                episode_gt,
                episode_frames,
                thresholds,
            )
        )
        action_timeline_paths.append(
            _write_action_timeline(
                output_root / f"action_timeline_episode_{int(episode_index)}.png",
                episode_pred,
                episode_gt,
                episode_frames,
            )
        )

    return {
        "predictions": predictions_path,
        "metrics": metrics_path,
        "per_horizon": per_horizon_path,
        "episode_metrics": episode_metrics_path,
        "action_error_heatmap": heatmap_path,
        "group_error_by_horizon": group_plot_path,
        "gripper_timelines": gripper_timeline_paths,
        "action_timelines": action_timeline_paths,
    }


if __name__ == "__main__":
    raise SystemExit(main())
