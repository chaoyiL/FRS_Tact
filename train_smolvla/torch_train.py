"""Launch official PyTorch LeRobot SmolVLA training from the project YAML."""

from __future__ import annotations

import argparse
from bisect import bisect_right
import copy
import inspect
import json
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml

if TYPE_CHECKING:
    from train_deco.input_adapter import LowLightAugmentationConfig

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = Path(__file__).resolve().parent / "configs" / "train_pytorch.yaml"


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
        if actual_images != expected_images:
            raise ValueError(
                f"dataset {source['repo_id']} cameras must be {sorted(expected_images)}, "
                f"got {sorted(actual_images)}"
            )
        if reference_features is None:
            reference_features = features
            reference_repo_id = source["repo_id"]
        elif features != reference_features:
            raise ValueError(
                f"dataset feature schemas differ between {reference_repo_id} and "
                f"{source['repo_id']}; multi-dataset SmolVLA requires identical feature schemas"
            )


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
        self.meta.stats = aggregate_stats([child.meta.stats for child in datasets])
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


def make_multi_dataset_factory(config: dict[str, Any], upstream_factory: Any) -> Any:
    """Create a LeRobot-compatible factory that concatenates all configured sources."""
    sources = dataset_sources(config)
    if len(sources) == 1:
        return upstream_factory

    def make_train_eval_datasets(cfg):
        from lerobot.datasets.compute_stats import aggregate_stats

        saved = {
            "repo_id": cfg.dataset.repo_id,
            "root": cfg.dataset.root,
            "revision": cfg.dataset.revision,
        }
        train_datasets: list[Any] = []
        eval_datasets: list[Any] = []
        try:
            for source in sources:
                cfg.dataset.repo_id = source["repo_id"]
                cfg.dataset.root = source["root"]
                cfg.dataset.revision = source["revision"]
                train_dataset, eval_dataset = upstream_factory(cfg)
                train_datasets.append(train_dataset)
                if eval_dataset is not None:
                    eval_datasets.append(eval_dataset)
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


def resolve_deco_augmentation(config: dict[str, Any]) -> LowLightAugmentationConfig | None:
    """Resolve the exact DECO batch-level augmentation selected by the YAML."""
    from train_deco.input_adapter import (
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
            f"unknown DECO augmentation preset {preset!r}; "
            f"expected one of {AUGMENTATION_PRESET_NAMES}"
        )
    resolved = augmentation_preset(preset, enabled=bool(raw.get("enabled", True)))
    validate_augmentation_config(resolved)
    if resolved.enabled and bool(config["dataset"].get("image_transforms", {}).get("enable", False)):
        raise ValueError(
            "dataset.image_transforms must be disabled when DECO augmentation is enabled"
        )
    return resolved


def augment_smolvla_training_batch(
    batch: Any,
    image_keys: tuple[str, ...],
    augmentation: LowLightAugmentationConfig | None,
) -> Any:
    """Apply the selected DECO preset jointly to all camera views in each training sample."""
    if augmentation is None or not augmentation.enabled:
        return batch
    import torch

    from train_deco.input_adapter import augment_training_images

    if not isinstance(batch, dict):
        raise TypeError(f"SmolVLA training batch must be a dict, got {type(batch).__name__}")
    missing = [key for key in image_keys if key not in batch]
    if missing:
        raise KeyError(f"SmolVLA training batch is missing image keys: {missing}")
    images = [batch[key] for key in image_keys]
    if not all(isinstance(image, torch.Tensor) for image in images):
        raise TypeError("SmolVLA camera batch values must be torch tensors")
    shapes = {tuple(image.shape) for image in images}
    if len(shapes) != 1 or images[0].ndim != 4:
        raise ValueError(
            "DECO image augmentation requires equally shaped camera batches [B,C,H,W], "
            f"got {[tuple(image.shape) for image in images]}"
        )
    if not all(image.is_floating_point() for image in images):
        raise TypeError("DECO image augmentation requires floating-point camera tensors")

    augmented = augment_training_images(torch.stack(images, dim=1), augmentation)
    for camera_index, key in enumerate(image_keys):
        batch[key] = augmented[:, camera_index]
    return batch


def build_command(config: dict[str, Any]) -> list[str]:
    """Build the official LeRobot arguments, following VB3's training contract."""
    dataset = config["dataset"]
    policy = config["policy"]
    training = config["training"]
    wandb = config["wandb"]
    peft = config.get("peft", {})
    transforms = dataset.get("image_transforms", {})
    primary_source = dataset_sources(config)[0]

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

    output_dir = Path(str(training["output_dir"])).expanduser()
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
            f"--wandb.mode={wandb.get('mode', 'online')}",
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
            "PyTorch training requires the official LeRobot environment used by VB3; "
            "set SMOLVLA_TORCH_PYTHON to that environment's Python"
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
    augmentation = resolve_deco_augmentation(config)
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
            cfg.input_features = {
                rename_map.get(name, name): feature
                for name, feature in features.items()
                if not name.startswith("action")
            }
        return upstream_make(cfg=cfg, ds_meta=ds_meta, env_cfg=env_cfg, rename_map=rename_map)

    lerobot_train.update_policy = update_policy
    lerobot_train.make_policy = make_policy
    lerobot_train.make_train_eval_datasets = make_multi_dataset_factory(
        config, upstream_make_datasets
    )
    sys.argv = ["lerobot-train", *command]
    lerobot_train.main()


def run(config_path: Path, *, dry_run: bool = False, worker: bool = False) -> None:
    config_path = config_path.expanduser().resolve()
    config = _load(config_path)
    num_gpus = int(config["distributed"].get("num_gpus", 1))
    if num_gpus < 1:
        raise ValueError("distributed.num_gpus must be at least 1")
    if not dry_run and not worker:
        validate_dataset_contract(config)
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
