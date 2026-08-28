"""Launch official PyTorch LeRobot SmolVLA training from the project YAML."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

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


def validate_dataset_contract(config: dict[str, Any]) -> None:
    """Validate the same v3.0 state/action/camera contract checked by VB3."""
    dataset = config["dataset"]
    root = Path(str(dataset["root"])).expanduser()
    info_path = root / "meta" / "info.json"
    if not info_path.is_file():
        raise FileNotFoundError(
            f"LeRobot v3 metadata not found: {info_path}; run scripts/download_data.sh first"
        )
    with info_path.open(encoding="utf-8") as file:
        info = json.load(file)
    if info.get("fps") != int(dataset["expected_fps"]):
        raise ValueError(f"dataset FPS must be {dataset['expected_fps']}, got {info.get('fps')}")

    features = info.get("features", {})
    if "actions" in features:
        raise ValueError("dataset contains legacy 'actions'; expected singular 'action'")
    for key, dimension in (
        ("observation.state", int(dataset["state_dim"])),
        ("action", int(dataset["action_dim"])),
    ):
        actual = features.get(key, {}).get("shape")
        if actual != [dimension]:
            raise ValueError(f"dataset {key} must have shape [{dimension}], got {actual}")
    expected_images = set(dataset["image_keys"])
    actual_images = {key for key in features if key.startswith("observation.images.")}
    if actual_images != expected_images:
        raise ValueError(
            f"dataset cameras must be {sorted(expected_images)}, got {sorted(actual_images)}"
        )


def build_command(config: dict[str, Any]) -> list[str]:
    """Build the official LeRobot arguments, following VB3's training contract."""
    dataset = config["dataset"]
    policy = config["policy"]
    training = config["training"]
    wandb = config["wandb"]
    peft = config.get("peft", {})
    transforms = dataset.get("image_transforms", {})

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
            f"--dataset.repo_id={dataset['repo_id']}",
            f"--dataset.root={Path(str(dataset['root'])).expanduser()}",
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

    def update_policy(*args, **kwargs):
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
