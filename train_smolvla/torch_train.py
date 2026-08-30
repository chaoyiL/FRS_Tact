"""Launch official PyTorch LeRobot SmolVLA training from the project YAML."""

from __future__ import annotations

import argparse
from bisect import bisect_right
import copy
from datetime import timedelta
import inspect
import json
import netrc as netrc_module
import os
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

import yaml

if TYPE_CHECKING:
    from train_smolvla.image_augmentation import ImageAugmentationConfig

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = Path(__file__).resolve().parent / "configs" / "train_smolvla.yaml"
OUTPUT_DIR_OVERRIDE_ENV = "FRS_SMOLVLA_OUTPUT_DIR"


def _load(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as file:
        config = yaml.safe_load(file) or {}
    if not isinstance(config, dict):
        raise ValueError(f"config root must be a mapping: {path}")
    for section in ("dataset", "policy", "training", "distributed", "wandb"):
        if not isinstance(config.get(section), dict):
            raise ValueError(f"missing YAML section: {section}")
    if not isinstance(config.get("peft", {}), dict):
        raise ValueError("YAML section peft must be a mapping")
    return config


def _bool(value: Any) -> str:
    return str(bool(value)).lower()


def _json(value: Any) -> str:
    return json.dumps(value, separators=(",", ":"))


def _wandb_has_credentials() -> bool:
    if os.environ.get("WANDB_API_KEY"):
        return True
    try:
        credentials = netrc_module.netrc().authenticators("api.wandb.ai")
    except (FileNotFoundError, netrc_module.NetrcParseError, OSError, UnicodeError):
        return False
    return bool(credentials and credentials[2])


def resolve_wandb_mode(config: dict[str, Any]) -> str:
    """Resolve a non-interactive W&B mode before distributed workers launch."""
    wandb = config["wandb"]
    if not bool(wandb.get("enable", True)):
        return "disabled"
    requested = str(wandb.get("mode", "auto")).lower()
    allowed = {"auto", "online", "offline", "disabled"}
    if requested not in allowed:
        raise ValueError(f"wandb.mode must be one of {sorted(allowed)}, got {requested!r}")
    environment_mode = os.environ.get("WANDB_MODE", "").lower()
    if requested == "auto" and environment_mode in allowed - {"auto"}:
        if environment_mode == "online" and not _wandb_has_credentials():
            return "offline"
        return environment_mode
    if requested == "auto":
        return "online" if _wandb_has_credentials() else "offline"
    return requested


def validate_constructed_policy(
    policy: Any,
    config: dict[str, Any],
    training_image_keys: tuple[str, ...],
) -> None:
    """Fail before the first batch if pretrained defaults leaked into the policy."""
    policy_config = policy.config
    expected_inputs = {
        "observation.state",
        *training_image_keys,
    }
    actual_inputs = set(policy_config.input_features)
    if actual_inputs != expected_inputs:
        raise ValueError(
            "constructed SmolVLA inputs do not match the FRS_Tact contract: "
            f"expected {sorted(expected_inputs)}, got {sorted(actual_inputs)}"
        )
    shape_contract = {
        "observation.state": int(config["dataset"]["state_dim"]),
        "action": int(config["dataset"]["action_dim"]),
    }
    all_features = {
        **policy_config.input_features,
        **policy_config.output_features,
    }
    for key, expected_dim in shape_contract.items():
        feature = all_features.get(key)
        actual_shape = None if feature is None else list(feature.shape)
        if actual_shape != [expected_dim]:
            raise ValueError(
                f"constructed SmolVLA {key} must have shape [{expected_dim}], "
                f"got {actual_shape}"
            )
    for name in ("chunk_size", "n_action_steps", "num_vlm_layers", "num_expert_layers"):
        expected = int(config["policy"][name])
        actual = int(getattr(policy_config, name))
        if actual != expected:
            raise ValueError(f"constructed SmolVLA {name} must be {expected}, got {actual}")
    if int(os.environ.get("LOCAL_RANK", "0")) == 0:
        print(
            "[smolvla] policy contract ready: "
            f"state={shape_contract['observation.state']}D "
            f"action={shape_contract['action']}D "
            f"cameras={list(training_image_keys)}"
        )


def validate_cuda_runtime(config: dict[str, Any]) -> None:
    """Validate the requested GPU count before starting distributed workers."""
    if not str(config["policy"].get("device", "cuda")).startswith("cuda"):
        return
    import torch

    requested = int(config["distributed"].get("num_gpus", 1))
    available = torch.cuda.device_count()
    if not torch.cuda.is_available() or available < requested:
        raise RuntimeError(
            f"SmolVLA requested {requested} CUDA GPU(s), but PyTorch detects {available}"
        )
    batch_per_gpu = int(config["training"]["batch_size"])
    print(
        f"[smolvla] CUDA ready: {available} GPU(s); using {requested}, "
        f"batch_per_gpu={batch_per_gpu}, global_batch={batch_per_gpu * requested}"
    )


def _configure_single_gpu_precision(config: dict[str, Any]) -> None:
    """Make a direct single-GPU launch honor distributed.mixed_precision."""
    precision = str(config["distributed"].get("mixed_precision", "no")).lower()
    if precision not in {"no", "fp16", "bf16"}:
        raise ValueError(
            "distributed.mixed_precision must be one of ['bf16', 'fp16', 'no'], "
            f"got {precision!r}"
        )
    os.environ["ACCELERATE_MIXED_PRECISION"] = precision
    print(f"[smolvla] single-GPU mixed precision={precision}")


def _install_accelerate_timeout(config: dict[str, Any]) -> Callable[[], None]:
    """Inject a longer process-group timeout before LeRobot creates Accelerator."""
    import accelerate
    from accelerate.utils import InitProcessGroupKwargs

    timeout_seconds = int(config["distributed"].get("timeout_seconds", 7200))
    if timeout_seconds <= 0:
        raise ValueError("distributed.timeout_seconds must be greater than zero")
    upstream_accelerator = accelerate.Accelerator

    def accelerator_with_timeout(*args, **kwargs):
        handlers = list(kwargs.pop("kwargs_handlers", None) or [])
        if any(isinstance(handler, InitProcessGroupKwargs) for handler in handlers):
            raise ValueError("LeRobot already supplied an InitProcessGroupKwargs handler")
        handlers.append(InitProcessGroupKwargs(timeout=timedelta(seconds=timeout_seconds)))
        return upstream_accelerator(*args, kwargs_handlers=handlers, **kwargs)

    accelerate.Accelerator = accelerator_with_timeout
    if int(os.environ.get("LOCAL_RANK", "0")) == 0:
        print(f"[smolvla] distributed process-group timeout={timeout_seconds}s")

    def restore() -> None:
        accelerate.Accelerator = upstream_accelerator

    return restore


def _effective_output_dir(config: dict[str, Any]) -> Path:
    configured = Path(str(config["training"]["output_dir"])).expanduser()
    override = os.environ.get(OUTPUT_DIR_OVERRIDE_ENV)
    return Path(override).expanduser() if override else configured


def _prepare_output_dir(config: dict[str, Any]) -> Path:
    """Choose a fresh output directory without deleting a previous run."""
    output_dir = Path(str(config["training"]["output_dir"])).expanduser()
    if config["training"].get("resume_from") or not output_dir.exists():
        os.environ[OUTPUT_DIR_OVERRIDE_ENV] = str(output_dir)
        return output_dir
    policy = str(config["training"].get("existing_output", "error")).lower()
    if policy != "increment":
        raise FileExistsError(
            f"training output directory already exists: {output_dir}; "
            "set training.existing_output=increment or configure training.resume_from"
        )
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    candidate = output_dir.with_name(f"{output_dir.name}-{timestamp}")
    suffix = 1
    while candidate.exists():
        candidate = output_dir.with_name(f"{output_dir.name}-{timestamp}-{suffix}")
        suffix += 1
    os.environ[OUTPUT_DIR_OVERRIDE_ENV] = str(candidate)
    print(
        f"[smolvla] output directory exists; preserving it and using {candidate}",
        flush=True,
    )
    return candidate


def dataset_sources(config: dict[str, Any]) -> list[dict[str, Any]]:
    """Return the canonical FRS-style list of SmolVLA dataset sources."""
    raw = config.get("datasets")
    if raw is None:
        dataset = config["dataset"]
        raw = [{"repo_id": dataset.get("repo_id"), "root": dataset.get("root")}]
    if not isinstance(raw, list) or not raw:
        raise ValueError("datasets must be a non-empty list of dataset mappings")
    sources: list[dict[str, Any]] = []
    seen_repo_ids: set[str] = set()
    for index, item in enumerate(raw):
        if not isinstance(item, dict):
            raise ValueError(f"datasets[{index}] must be a mapping")
        unknown = set(item) - {"repo_id", "root", "revision"}
        if unknown:
            raise ValueError(f"unknown datasets[{index}] fields: {sorted(unknown)}")
        if not item.get("repo_id"):
            raise ValueError(f"datasets[{index}].repo_id is required")
        if not item.get("root"):
            raise ValueError(f"datasets[{index}].root is required")
        repo_id = str(item["repo_id"])
        if repo_id in seen_repo_ids:
            raise ValueError(f"duplicate dataset repo_id: {repo_id}")
        seen_repo_ids.add(repo_id)
        sources.append(
            {
                "repo_id": repo_id,
                "root": str(item["root"]),
                "revision": None if item.get("revision") is None else str(item["revision"]),
            }
        )
    return sources


def validate_dataset_contract(config: dict[str, Any]) -> None:
    """Validate every v3.0 source against one shared state/action/camera contract."""
    dataset = config["dataset"]
    reference_features: dict[str, Any] | None = None
    reference_repo_id: str | None = None
    for source in dataset_sources(config):
        root = Path(source["root"]).expanduser()
        info_path = root / "meta" / "info.json"
        if not info_path.is_file():
            raise FileNotFoundError(
                f"LeRobot v3 metadata not found for {source['repo_id']}: {info_path}; "
                "run scripts/download_data.sh first"
            )
        with info_path.open(encoding="utf-8") as file:
            info = json.load(file)
        if info.get("fps") != int(dataset["expected_fps"]):
            raise ValueError(
                f"dataset {source['repo_id']} FPS must be {dataset['expected_fps']}, "
                f"got {info.get('fps')}"
            )

        features = info.get("features", {})
        if "actions" in features:
            raise ValueError(
                f"dataset {source['repo_id']} contains legacy 'actions'; expected singular 'action'"
            )
        for key, dimension in (
            ("observation.state", int(dataset["state_dim"])),
            ("action", int(dataset["action_dim"])),
        ):
            actual = features.get(key, {}).get("shape")
            if actual != [dimension]:
                raise ValueError(
                    f"dataset {source['repo_id']} {key} must have shape [{dimension}], got {actual}"
                )
        expected_images = set(dataset["image_keys"])
        actual_images = {key for key in features if key.startswith("observation.images.")}
        missing_images = expected_images - actual_images
        if missing_images:
            raise ValueError(
                f"dataset {source['repo_id']} is missing required cameras "
                f"{sorted(missing_images)}; available cameras are {sorted(actual_images)}"
            )
        selected_features = {
            key: {
                field: features[key].get(field)
                for field in ("dtype", "shape", "names")
                if features[key].get(field) is not None
            }
            for key in ("observation.state", "action", *sorted(expected_images))
        }
        if reference_features is None:
            reference_features = selected_features
            reference_repo_id = source["repo_id"]
        elif selected_features != reference_features:
            raise ValueError(
                f"selected feature schemas differ between {reference_repo_id} and "
                f"{source['repo_id']}; multi-dataset SmolVLA requires identical state, "
                "action, and selected camera schemas"
            )


def _stats_tensors_to_numpy(value: Any) -> Any:
    """Recursively convert Torch statistic leaves for LeRobot aggregation."""
    if isinstance(value, dict):
        return {key: _stats_tensors_to_numpy(item) for key, item in value.items()}
    detach = getattr(value, "detach", None)
    if callable(detach):
        detached = detach()
        cpu = getattr(detached, "cpu", None)
        numpy = getattr(cpu() if callable(cpu) else detached, "numpy", None)
        if callable(numpy):
            return numpy()
    return value


class CombinedLeRobotDataset:
    """Map-style concatenation with aggregate stats and global episode boundaries."""

    def __init__(self, datasets: list[Any], aggregate_stats: Any):
        if not datasets:
            raise ValueError("at least one LeRobot dataset is required")
        self._datasets = datasets
        self._ends: list[int] = []
        frame_total = 0
        for child in datasets:
            frame_total += len(child)
            self._ends.append(frame_total)

        self.meta = copy.copy(datasets[0].meta)
        child_stats = [child.meta.stats for child in datasets]
        if len(child_stats) == 1:
            # Preserve official single-dataset behavior. In particular,
            # use_imagenet_stats replaces visual mean/std with Torch tensors,
            # while LeRobot's multi-dataset aggregate_stats only accepts NumPy.
            self.meta.stats = copy.deepcopy(child_stats[0])
        else:
            self.meta.stats = aggregate_stats(
                [_stats_tensors_to_numpy(stats) for stats in child_stats]
            )
        starts: list[int] = []
        stops: list[int] = []
        tasks: list[Any] = []
        cursor = 0
        for child in datasets:
            child_episodes = (
                list(child.episodes)
                if child.episodes is not None
                else list(range(child.num_episodes))
            )
            for episode_index in child_episodes:
                source_start = int(child.meta.episodes["dataset_from_index"][episode_index])
                source_stop = int(child.meta.episodes["dataset_to_index"][episode_index])
                cursor_next = cursor + source_stop - source_start
                starts.append(cursor)
                stops.append(cursor_next)
                cursor = cursor_next
                if "tasks" in child.meta.episodes:
                    tasks.append(child.meta.episodes["tasks"][episode_index])
                else:
                    tasks.append([])
        if cursor != frame_total:
            raise ValueError(
                f"combined episode lengths ({cursor}) do not match dataset frames ({frame_total})"
            )
        self.meta.episodes = {
            "dataset_from_index": starts,
            "dataset_to_index": stops,
            "tasks": tasks,
        }
        self.episodes = list(range(len(starts)))
        self.absolute_to_relative_idx = None

    @property
    def num_frames(self) -> int:
        return len(self)

    @property
    def num_episodes(self) -> int:
        return len(self.episodes)

    def __len__(self) -> int:
        return self._ends[-1]

    def __getitem__(self, index: int) -> Any:
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        dataset_index = bisect_right(self._ends, index)
        start = 0 if dataset_index == 0 else self._ends[dataset_index - 1]
        return self._datasets[dataset_index][index - start]


def _select_dataset_cameras(dataset: Any, selected_images: set[str]) -> Any:
    """Hide unselected camera features before LeRobot returns training samples."""
    meta = dataset.meta
    features = dict(meta.features)
    missing = selected_images - set(features)
    if missing:
        raise KeyError(f"dataset is missing selected cameras: {sorted(missing)}")
    selected_features = {
        key: value
        for key, value in features.items()
        if not key.startswith("observation.images.") or key in selected_images
    }
    # LeRobotDatasetMetadata properties read from ``info`` dynamically.  The
    # DatasetReader holds this same metadata object, so pruning it here prevents
    # tactile/unused videos from being decoded by DataLoader workers.
    if isinstance(meta.info, dict):
        meta.info["features"] = selected_features
    else:
        meta.info.features = selected_features
    meta.stats = {
        key: value for key, value in meta.stats.items() if key in selected_features
    }
    if getattr(dataset, "delta_timestamps", None) is not None:
        dataset.delta_timestamps = {
            key: value
            for key, value in dataset.delta_timestamps.items()
            if key in selected_features
        }
    reader = getattr(dataset, "reader", None)
    if reader is not None and reader.delta_indices is not None:
        reader.delta_indices = {
            key: value
            for key, value in reader.delta_indices.items()
            if key in selected_features
        }
    # Official LeRobot 0.6.1 loads ``dtype: image`` cameras as columns in the
    # Hugging Face Dataset before this selector runs. Pruning metadata is enough
    # for MP4-backed video keys, but embedded Parquet image columns would still
    # be returned by ``dataset[index]``. Remove those unused columns from the
    # already-loaded reader as well, so tactile images never reach DataLoader.
    hf_dataset = None if reader is None else getattr(reader, "hf_dataset", None)
    if hf_dataset is not None:
        unused_image_columns = [
            key
            for key in getattr(hf_dataset, "column_names", ())
            if key.startswith("observation.images.") and key not in selected_images
        ]
        if unused_image_columns:
            reader.hf_dataset = hf_dataset.remove_columns(unused_image_columns)
    return dataset


def make_multi_dataset_factory(config: dict[str, Any], upstream_factory: Any) -> Any:
    """Create a LeRobot-compatible factory that concatenates all configured sources."""
    sources = dataset_sources(config)
    selected_images = set(config["dataset"]["image_keys"])

    def make_train_eval_datasets(cfg):
        from lerobot.datasets.compute_stats import aggregate_stats

        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        saved = {
            "repo_id": cfg.dataset.repo_id,
            "root": cfg.dataset.root,
            "revision": cfg.dataset.revision,
        }
        train_datasets: list[Any] = []
        eval_datasets: list[Any] = []
        try:
            for source_index, source in enumerate(sources, start=1):
                started_at = time.monotonic()
                print(
                    f"[smolvla][rank {local_rank}] loading dataset "
                    f"{source_index}/{len(sources)}: {source['repo_id']}",
                    flush=True,
                )
                cfg.dataset.repo_id = source["repo_id"]
                cfg.dataset.root = source["root"]
                cfg.dataset.revision = source["revision"]
                train_dataset, eval_dataset = upstream_factory(cfg)
                train_datasets.append(_select_dataset_cameras(train_dataset, selected_images))
                if eval_dataset is not None:
                    eval_datasets.append(_select_dataset_cameras(eval_dataset, selected_images))
                print(
                    f"[smolvla][rank {local_rank}] dataset ready: {source['repo_id']} "
                    f"train_frames={len(train_dataset)} "
                    f"eval_frames={0 if eval_dataset is None else len(eval_dataset)} "
                    f"elapsed={time.monotonic() - started_at:.1f}s",
                    flush=True,
                )
        finally:
            cfg.dataset.repo_id = saved["repo_id"]
            cfg.dataset.root = saved["root"]
            cfg.dataset.revision = saved["revision"]
        combined_train = CombinedLeRobotDataset(train_datasets, aggregate_stats)
        combined_eval = (
            CombinedLeRobotDataset(eval_datasets, aggregate_stats) if eval_datasets else None
        )
        return combined_train, combined_eval

    return make_train_eval_datasets


def resolve_smolvla_augmentation(config: dict[str, Any]) -> ImageAugmentationConfig | None:
    """Resolve the SmolVLA image augmentation preset selected by the YAML."""
    from train_smolvla.image_augmentation import (
        AUGMENTATION_PRESET_NAMES,
        augmentation_preset,
        validate_augmentation_config,
    )

    raw = config.get("augmentation")
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise ValueError("YAML section augmentation must be a mapping")
    unknown = set(raw) - {"preset", "enabled"}
    if unknown:
        raise ValueError(f"unknown augmentation fields: {sorted(unknown)}")
    preset = str(raw.get("preset", "balanced-light-v2"))
    if preset not in AUGMENTATION_PRESET_NAMES:
        raise ValueError(
            f"unknown SmolVLA image augmentation preset {preset!r}; "
            f"expected one of {AUGMENTATION_PRESET_NAMES}"
        )
    resolved = augmentation_preset(preset, enabled=bool(raw.get("enabled", True)))
    validate_augmentation_config(resolved)
    if resolved.enabled and bool(config["dataset"].get("image_transforms", {}).get("enable", False)):
        raise ValueError(
            "dataset.image_transforms must be disabled when SmolVLA image augmentation is enabled"
        )
    return resolved


def augment_smolvla_training_batch(
    batch: Any,
    image_keys: tuple[str, ...],
    augmentation: ImageAugmentationConfig | None,
) -> Any:
    """Apply SmolVLA augmentation jointly to all views in each training sample."""
    if augmentation is None or not augmentation.enabled:
        return batch
    import torch

    from train_smolvla.image_augmentation import augment_training_images

    if not isinstance(batch, dict):
        raise TypeError(f"SmolVLA training batch must be a dict, got {type(batch).__name__}")
    missing = [key for key in image_keys if key not in batch]
    if missing:
        raise KeyError(f"SmolVLA training batch is missing image keys: {missing}")
    images = [batch[key] for key in image_keys]
    if not all(isinstance(image, torch.Tensor) for image in images):
        raise TypeError("SmolVLA camera batch values must be torch tensors")
    shapes = {tuple(image.shape) for image in images}
    if len(shapes) != 1 or images[0].ndim not in (4, 5) or images[0].shape[-3] != 3:
        raise ValueError(
            "SmolVLA image augmentation requires equally shaped RGB camera batches "
            "[B,C,H,W] or [B,T,C,H,W], "
            f"got {[tuple(image.shape) for image in images]}"
        )
    if not all(image.is_floating_point() for image in images):
        raise TypeError("SmolVLA image augmentation requires floating-point camera tensors")

    # LeRobot keeps an observation-time axis even when n_obs_steps=1. Flatten
    # camera and time into the augmentation view axis [B,N,C,H,W], so it
    # implementation samples exactly one transform per batch sample and shares
    # it across every camera and observation time.
    stacked = torch.stack(images, dim=1)
    if images[0].ndim == 5:
        batch_size, cameras, times, channels, height, width = stacked.shape
        deco_images = stacked.reshape(
            batch_size, cameras * times, channels, height, width
        )
        augmented = augment_training_images(deco_images, augmentation).reshape_as(stacked)
    else:
        augmented = augment_training_images(stacked, augmentation)
    for camera_index, key in enumerate(image_keys):
        batch[key] = augmented[:, camera_index]
    return batch


def build_command(config: dict[str, Any]) -> list[str]:
    """Build the official LeRobot arguments for FRS_Tact training."""
    dataset = config["dataset"]
    policy = config["policy"]
    training = config["training"]
    wandb = config["wandb"]
    peft = config.get("peft", {})
    transforms = dataset.get("image_transforms", {})
    primary_source = dataset_sources(config)[0]
    wandb_mode = resolve_wandb_mode(config)

    steps = int(training["steps"])
    resume_from = training.get("resume_from")
    command: list[str] = []
    if resume_from:
        command.extend(
            [
                f"--config_path={Path(str(resume_from)).expanduser()}",
                "--resume=true",
            ]
        )
    else:
        command.extend(
            [
                f"--policy.path={policy.get('path', 'lerobot/smolvla_base')}",
                f"--rename_map={_json(config.get('rename_map') or {})}",
            ]
        )

    output_dir = _effective_output_dir(config)
    command.extend(
        [
            f"--policy.chunk_size={int(policy['chunk_size'])}",
            f"--policy.n_action_steps={int(policy['n_action_steps'])}",
            f"--policy.empty_cameras={int(policy.get('empty_cameras', 0))}",
            f"--policy.num_vlm_layers={int(policy['num_vlm_layers'])}",
            f"--policy.freeze_vision_encoder={_bool(policy['freeze_vision_encoder'])}",
            f"--policy.train_expert_only={_bool(policy['train_expert_only'])}",
            f"--policy.train_state_proj={_bool(policy['train_state_proj'])}",
            f"--policy.optimizer_lr={float(policy['optimizer_lr'])}",
            f"--policy.scheduler_warmup_steps={int(policy['scheduler_warmup_steps'])}",
            f"--policy.scheduler_decay_steps={int(policy.get('scheduler_decay_steps', steps))}",
            f"--policy.push_to_hub={_bool(policy.get('push_to_hub', False))}",
            f"--policy.device={policy.get('device', 'cuda')}",
            f"--dataset.repo_id={primary_source['repo_id']}",
            f"--dataset.root={Path(primary_source['root']).expanduser()}",
            f"--dataset.eval_split={float(dataset.get('eval_split', 0.1))}",
            f"--dataset.image_transforms.enable={_bool(transforms.get('enable', False))}",
            (
                "--dataset.image_transforms.max_num_transforms="
                f"{int(transforms.get('max_num_transforms', 0))}"
            ),
            f"--dataset.image_transforms.random_order={_bool(transforms.get('random_order', False))}",
            f"--dataset.image_transforms.tfs={_json(transforms.get('tfs') or {})}",
            f"--batch_size={int(training['batch_size'])}",
            f"--num_workers={int(training['num_workers'])}",
            f"--steps={steps}",
            f"--eval_steps={int(training['eval_steps'])}",
            f"--max_eval_samples={int(training.get('max_eval_samples', 0))}",
            f"--save_freq={int(training['save_freq'])}",
            f"--log_freq={int(training['log_freq'])}",
            f"--seed={int(training['seed'])}",
            f"--output_dir={output_dir}",
            f"--job_name={training.get('job_name') or output_dir.name}",
            f"--wandb.enable={_bool(wandb.get('enable', True))}",
            f"--wandb.project={wandb['project']}",
            f"--wandb.mode={wandb_mode}",
            f"--wandb.disable_artifact={_bool(wandb.get('disable_artifact', True))}",
            f"--wandb.add_tags={_bool(wandb.get('add_tags', True))}",
        ]
    )
    if policy.get("num_expert_layers") is not None:
        command.append(f"--policy.num_expert_layers={int(policy['num_expert_layers'])}")

    if peft.get("enable", False):
        target_modules = peft.get("target_modules")
        if not target_modules:
            raise ValueError("peft.target_modules is required when peft.enable=true")
        if policy.get("train_expert_only", False) and "vlm" in str(target_modules):
            raise ValueError("VLM LoRA requires policy.train_expert_only=false")
        if not isinstance(target_modules, str):
            target_modules = _json(list(target_modules))
        command.extend(
            [
                f"--peft.method_type={peft.get('method_type', 'LORA')}",
                f"--peft.target_modules={target_modules}",
                f"--peft.full_training_modules={_json(peft.get('full_training_modules') or [])}",
                f"--peft.r={int(peft.get('rank', 16))}",
            ]
        )
        if peft.get("lora_alpha") is not None:
            command.append(f"--peft.lora_alpha={int(peft['lora_alpha'])}")

    for name in ("entity", "notes", "run_id"):
        if wandb.get(name):
            command.append(f"--wandb.{name}={wandb[name]}")
    return command


def _import_official_lerobot():
    removed: list[tuple[int, str]] = []
    for index in range(len(sys.path) - 1, -1, -1):
        entry = sys.path[index]
        try:
            resolved = Path(entry or os.getcwd()).resolve()
        except OSError:
            continue
        if resolved == ROOT:
            removed.append((index, sys.path.pop(index)))
    try:
        from lerobot.scripts import lerobot_train
        from lerobot.utils.feature_utils import dataset_to_policy_features
    except ImportError as error:
        raise ImportError(
            "PyTorch training requires FRS_Tact's isolated official LeRobot environment; "
            "run `bash scripts/setup_env.sh --smolvla`"
        ) from error
    finally:
        for index, entry in sorted(removed):
            sys.path.insert(index, entry)
    return lerobot_train, dataset_to_policy_features


def _accelerate_command(config_path: Path, config: dict[str, Any], *, dry_run: bool) -> list[str]:
    distributed = config["distributed"]
    accelerate = shutil.which("accelerate")
    if accelerate is None:
        if not dry_run:
            raise FileNotFoundError("distributed.num_gpus > 1 but accelerate is not installed")
        accelerate = "accelerate"
    return [
        accelerate,
        "launch",
        "--multi_gpu",
        f"--num_processes={int(distributed['num_gpus'])}",
        f"--num_machines={int(distributed.get('num_machines', 1))}",
        f"--mixed_precision={distributed.get('mixed_precision', 'bf16')}",
        "--dynamo_backend=no",
        "--module",
        "train_smolvla.torch_train",
        "--config",
        str(config_path),
        "--worker",
    ]


def _run_worker(config: dict[str, Any], command: list[str]) -> None:
    lerobot_train, dataset_to_policy_features = _import_official_lerobot()
    upstream_update = lerobot_train.update_policy
    upstream_make = lerobot_train.make_policy
    upstream_make_datasets = lerobot_train.make_train_eval_datasets
    update_signature = inspect.signature(upstream_update)
    augmentation = resolve_smolvla_augmentation(config)
    rename_map = config.get("rename_map") or {}
    training_image_keys = tuple(
        rename_map.get(key, key) for key in config["dataset"]["image_keys"]
    )

    def update_policy(*args, **kwargs):
        bound = update_signature.bind_partial(*args, **kwargs)
        if "batch" not in bound.arguments:
            raise TypeError("official LeRobot update_policy call does not expose a batch argument")
        augment_smolvla_training_batch(
            bound.arguments["batch"], training_image_keys, augmentation
        )
        metrics, output = upstream_update(*args, **kwargs)
        if output and "loss" in output:
            output = {"loss_step" if key == "loss" else key: value for key, value in output.items()}
        return metrics, output

    def make_policy(*, cfg, ds_meta=None, env_cfg=None, rename_map=None):
        if ds_meta is not None:
            features = dataset_to_policy_features(ds_meta.features)
            rename_map = rename_map or {}
            selected_inputs = {
                "observation.state",
                *config["dataset"]["image_keys"],
            }
            cfg.input_features = {
                rename_map.get(name, name): feature
                for name, feature in features.items()
                if name in selected_inputs
            }
        policy = upstream_make(cfg=cfg, ds_meta=ds_meta, env_cfg=env_cfg, rename_map=rename_map)
        validate_constructed_policy(policy, config, training_image_keys)
        return policy

    lerobot_train.update_policy = update_policy
    lerobot_train.make_policy = make_policy
    lerobot_train.make_train_eval_datasets = make_multi_dataset_factory(
        config, upstream_make_datasets
    )
    sys.argv = ["lerobot-train", *command]
    restore_accelerator = _install_accelerate_timeout(config)
    try:
        lerobot_train.main()
    finally:
        restore_accelerator()
        import torch.distributed as distributed

        if distributed.is_available() and distributed.is_initialized():
            distributed.destroy_process_group()


def run(config_path: Path, *, dry_run: bool = False, worker: bool = False) -> None:
    config_path = config_path.expanduser().resolve()
    config = _load(config_path)
    wandb_mode = resolve_wandb_mode(config)
    if not worker and config["wandb"].get("enable", True):
        requested_mode = str(config["wandb"].get("mode", "auto")).lower()
        if requested_mode == "auto":
            print(f"[smolvla] wandb.mode auto -> {wandb_mode}")
    num_gpus = int(config["distributed"].get("num_gpus", 1))
    if num_gpus < 1:
        raise ValueError("distributed.num_gpus must be at least 1")
    if not dry_run and not worker:
        _prepare_output_dir(config)
        validate_dataset_contract(config)
        validate_cuda_runtime(config)
        if num_gpus == 1:
            _configure_single_gpu_precision(config)
    if num_gpus > 1 and not worker:
        launcher = _accelerate_command(config_path, config, dry_run=dry_run)
        if dry_run:
            print(shlex.join(launcher))
            return
        subprocess.run(launcher, check=True)
        return

    command = build_command(config)
    if dry_run:
        print("python -m lerobot.scripts.lerobot_train " + shlex.join(command))
        return
    _run_worker(config, command)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    run(args.config, dry_run=args.dry_run, worker=args.worker)


if __name__ == "__main__":
    main()
