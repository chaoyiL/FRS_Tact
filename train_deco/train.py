import argparse
import json
import math
import os
import random
import time
from contextlib import nullcontext
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler, Subset

from .checkpoint import (
    atomic_torch_save,
    capture_rng_state,
    load_checkpoint,
    prune_old_checkpoints,
    restore_rng_state,
)
from .export_torchscript import copy_torchscript_artifact, export_policy
from .input_adapter import (
    LowLightAugmentationConfig,
    augment_training_images,
    letterbox_and_normalize,
    select_deco_observation,
    validate_augmentation_config,
)
from .metrics import MetricsLogger
from .model_factory import MODEL_TYPE, build_model, observation_indices
from .preprocessed_dataset import PreprocessedDECODataset, verify_preprocessed_dataset
from .resume import validate_resume_config
from .training_utils import (
    DistributedEvalSampler,
    backbone_cosine_multiplier,
    constant_lr_scheduler,
    deterministic_subset_indices,
    masked_error_sums,
    metric_totals,
    optimizer_parameter_groups,
    optimizer_partition_lr,
    override_optimizer_partition_lrs,
    seed_training_rng,
    set_backbone_batch_norm_eval,
    warmup_cosine_multiplier,
)


def is_dist() -> bool:
    return int(os.environ.get("WORLD_SIZE", "1")) > 1


def setup_dist():
    if not is_dist():
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu"), 0, 1
    dist.init_process_group(backend="nccl" if torch.cuda.is_available() else "gloo")
    rank = dist.get_rank()
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cpu")
    return device, rank, dist.get_world_size()


def cleanup_dist():
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def env_bool(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    return default if value is None else value.strip().lower() in {"1", "true", "yes", "on"}


def env_tuple(name: str, default: tuple, value_type=float) -> tuple:
    value = os.environ.get(name)
    if value is None:
        return default
    return tuple(value_type(item.strip()) for item in value.split(","))


def augmentation_config_from_args(args) -> LowLightAugmentationConfig:
    config = LowLightAugmentationConfig(
        enabled=args.augmentation_enabled,
        identity_probability=args.augmentation_identity_probability,
        low_light_probability=args.augmentation_low_light_probability,
        mild_probability=args.augmentation_mild_probability,
        exposure_probability=args.augmentation_exposure_probability,
        exposure_range=tuple(args.augmentation_exposure_range),
        gamma_range=tuple(args.augmentation_gamma_range),
        mild_brightness_range=tuple(args.augmentation_mild_brightness_range),
        contrast_range=tuple(args.augmentation_contrast_range),
        saturation_range=tuple(args.augmentation_saturation_range),
        blur_probability=args.augmentation_blur_probability,
        blur_kernel_sizes=tuple(args.augmentation_blur_kernel_sizes),
        blur_sigma_range=tuple(args.augmentation_blur_sigma_range),
    )
    validate_augmentation_config(config)
    return config


def jsonable_stats(stats: dict) -> dict:
    return {key: value.tolist() for key, value in stats.items()}


def resolve_dataset_action_mode(contract: dict) -> str:
    """Resolve old homogeneous actions and the pick-tube mixed action contract."""
    action_mode = contract.get("action_mode")
    legacy_delta_flag = contract.get("use_delta_action")
    if action_mode is None:
        return "delta" if bool(legacy_delta_flag) else "absolute"
    if action_mode not in {
        "absolute",
        "delta",
        "tcp_delta_absolute_gripper",
    }:
        raise ValueError(f"Unsupported dataset action_mode: {action_mode!r}")
    if action_mode == "tcp_delta_absolute_gripper":
        if legacy_delta_flag is not None:
            raise ValueError(
                "use_delta_action cannot describe the mixed "
                "tcp_delta_absolute_gripper action contract"
            )
        return action_mode
    if (
        legacy_delta_flag is not None
        and bool(legacy_delta_flag) != (action_mode == "delta")
    ):
        raise ValueError("Dataset action_mode and use_delta_action disagree")
    return action_mode


def action_mode_config_fields(action_mode: str) -> dict:
    """Emit the legacy flag only when it can represent the whole action."""
    if action_mode == "delta":
        return {"use_delta_action": True}
    if action_mode == "absolute":
        return {"use_delta_action": False}
    if action_mode == "tcp_delta_absolute_gripper":
        return {}
    raise ValueError(f"Unsupported dataset action_mode: {action_mode!r}")


def training_dataset_source(args) -> str:
    if args.dataset_manifest:
        if args.dataset_format != "lerobot-v21":
            raise ValueError("--dataset-manifest is only valid for lerobot-v21")
        return str(args.dataset_manifest)
    if args.dataset_dir:
        return str(args.dataset_dir)
    raise ValueError(
        "--dataset-dir/--dataset-manifest (or the corresponding environment variable) is required"
    )


def create_training_datasets(args):
    """Create the selected dataset backend with a common training interface."""
    val_limit = max(1, args.limit_samples // 5) if args.limit_samples else None
    if args.dataset_format == "preprocessed":
        return (
            PreprocessedDECODataset(
                args.dataset_dir,
                "train",
                args.limit_samples,
                action_chunk_size=args.action_chunk_size,
            ),
            PreprocessedDECODataset(
                args.dataset_dir,
                "val",
                val_limit,
                action_chunk_size=args.action_chunk_size,
            ),
        )
    if args.dataset_format == "lerobot-v21":
        from .lerobot_vision_dataset import build_lerobot_vision_datasets

        return build_lerobot_vision_datasets(
            training_dataset_source(args),
            action_chunk_size=args.action_chunk_size or 32,
            validation_ratio=args.validation_ratio,
            split_seed=args.episode_split_seed,
            train_limit=args.limit_samples,
            val_limit=val_limit,
        )
    raise ValueError(f"Unsupported dataset format: {args.dataset_format!r}")


def make_loader(dataset, batch_size, workers, rank, world_size, shuffle, seed):
    sampler = None
    if world_size > 1:
        if shuffle:
            sampler = DistributedSampler(
                dataset,
                num_replicas=world_size,
                rank=rank,
                shuffle=True,
                seed=seed,
                drop_last=False,
            )
        else:
            sampler = DistributedEvalSampler(dataset, world_size, rank)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=shuffle and sampler is None,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        # persistent_workers disabled: each worker holds ~120k open file
        # descriptors (146k mmap'd npy shards) and /dev/shm tensor objects;
        # keeping workers alive across epochs accumulates fd/shm until the node
        # hits fs.file-max (Errno 23 "Too many open files in system") and crashes
        # DDP mid-epoch. Recreating workers each epoch releases them.
        persistent_workers=False,
        drop_last=False,
    ), sampler


def validate_nonempty_training_loader(dataset, loader, batch_size, world_size, rank, device):
    batches = torch.tensor(len(loader), dtype=torch.int64, device=device)
    if world_size > 1:
        dist.all_reduce(batches, op=dist.ReduceOp.MIN)
    if batches.item() > 0:
        return
    cleanup_dist()
    raise ValueError(
        "DECO-C02: training DataLoader produced zero batches. "
        f"samples={len(dataset)}, batch_size={batch_size}, world_size={world_size}, rank={rank}"
    )


def reduce_totals(totals: torch.Tensor, world_size: int) -> dict:
    if world_size > 1:
        dist.all_reduce(totals, op=dist.ReduceOp.SUM)
    count = totals[2].item()
    if count <= 0:
        raise ValueError("Metric aggregation received zero valid action elements")
    return {
        "loss": totals[0].item() / count,
        "velocity_mae": totals[1].item() / count,
        "element_count": count,
    }


def masked_action_metrics(prediction, target, is_pad):
    squared_sum, absolute_sum, element_count = masked_error_sums(
        prediction, target, is_pad
    )
    if element_count.item() <= 0:
        raise ValueError("Metric batch contains zero valid action elements")
    return squared_sum / element_count, absolute_sum / element_count


def sync_model_buffers(model, world_size: int) -> None:
    """Synchronize DDP buffers once before uneven no-padding validation."""

    if world_size <= 1:
        return
    for buffer in model.buffers():
        dist.broadcast(buffer, src=0)


def average_validation_metrics(records: list[dict]) -> dict:
    if not records:
        raise ValueError("At least one validation pass is required")
    counts = {record["element_count"] for record in records}
    if len(counts) != 1:
        raise ValueError(f"Validation element counts changed between seeds: {counts}")
    losses = np.asarray([record["loss"] for record in records], dtype=np.float64)
    maes = np.asarray([record["velocity_mae"] for record in records], dtype=np.float64)
    return {
        "loss": float(losses.mean()),
        "loss_std": float(losses.std()),
        "velocity_mae": float(maes.mean()),
        "velocity_mae_std": float(maes.std()),
        "element_count": counts.pop(),
        "noise_seeds": len(records),
    }


def run_epoch(
    model,
    loader,
    device,
    optimizer,
    scheduler,
    scaler,
    observation_index,
    image_size,
    use_task_condition,
    train,
    world_size,
    backbone_parameters=(),
    backbone_freeze_steps=0,
    backbone_bn_eval=False,
    mask_training_padding=True,
    epoch=0,
    global_step=0,
    log_every_steps=0,
    metric_callback=None,
    validation_seed=12345,
    rank=0,
    augmentation_config=None,
):
    model.train(train)
    raw_model = model.module if isinstance(model, DDP) else model
    if train and backbone_bn_eval:
        set_backbone_batch_norm_eval(raw_model)
    totals = torch.zeros(3, dtype=torch.float64, device=device)
    interval = torch.zeros(3, dtype=torch.float64, device=device)
    total_batches = len(loader)
    devices = [device.index] if device.type == "cuda" else []
    rng_context = nullcontext() if train else torch.random.fork_rng(devices=devices)
    grad_context = torch.enable_grad() if train else torch.inference_mode()
    with rng_context:
        if not train:
            torch.manual_seed(validation_seed + rank)
            if device.type == "cuda":
                torch.cuda.manual_seed(validation_seed + rank)
        with grad_context:
            for batch_index, batch in enumerate(loader, start=1):
                source_observation = batch["observation"].to(device, non_blocking=True)
                observation = select_deco_observation(source_observation, observation_index)
                images = batch["images"].to(device, non_blocking=True)
                if train:
                    images = augment_training_images(images, augmentation_config)
                images = letterbox_and_normalize(images, image_size)
                actions = batch["action"].to(device, non_blocking=True)
                is_pad = batch["is_pad"].to(device, non_blocking=True)
                task_index = (
                    batch["task_index"].to(device, non_blocking=True)
                    if use_task_condition else None
                )
                if train:
                    optimizer.zero_grad(set_to_none=True)
                    step_policy_lr = optimizer_partition_lr(optimizer, "policy")
                    step_backbone_lr = optimizer_partition_lr(
                        optimizer, "backbone"
                    )
                    backbone_frozen = global_step < backbone_freeze_steps
                with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                    camera_images = (
                        (images[:, 0], images[:, 1])
                        if images.shape[1] == 2
                        else (images[:, 0], images[:, 1], images[:, 2])
                    )
                    if train:
                        predicted, noise = model(
                            *camera_images, obs=observation,
                            act=actions, task_idx=task_index, training=True,
                        )
                        target = noise - actions
                        squared_sum, absolute_sum, element_count = masked_error_sums(
                            predicted,
                            target,
                            is_pad if mask_training_padding else None,
                        )
                        global_element_count = element_count.detach().to(
                            dtype=squared_sum.dtype
                        )
                        if world_size > 1:
                            dist.all_reduce(
                                global_element_count, op=dist.ReduceOp.SUM
                            )
                        if global_element_count.item() <= 0:
                            raise ValueError(
                                "Training batch contains zero valid action elements"
                            )
                        # DDP averages rank gradients. Scaling by world_size makes
                        # this exactly the global valid-element MSE gradient.
                        loss = (
                            squared_sum * world_size / global_element_count
                        )
                    else:
                        prediction = model(
                            *camera_images, obs=observation,
                            task_idx=task_index, training=False,
                        )
                        squared_sum, absolute_sum, element_count = masked_error_sums(
                            prediction, actions, is_pad
                        )
                if train:
                    scaler.scale(loss).backward()
                    if backbone_frozen:
                        for parameter in backbone_parameters:
                            parameter.grad = None
                    scale_before = scaler.get_scale()
                    scaler.step(optimizer)
                    scaler.update()
                    step_succeeded = (
                        not scaler.is_enabled() or scaler.get_scale() >= scale_before
                    )
                    if step_succeeded:
                        scheduler.step()
                        global_step += 1
                batch_values = metric_totals(
                    squared_sum, absolute_sum, element_count
                )
                totals += batch_values
                if train:
                    interval += batch_values
                    should_log = log_every_steps > 0 and (
                        batch_index % log_every_steps == 0 or batch_index == total_batches
                    )
                    if should_log:
                        record = reduce_totals(interval.clone(), world_size)
                        if metric_callback is not None:
                            metric_callback({
                                "event": "train_step", "epoch": epoch,
                                "batch": batch_index, "batches_in_epoch": total_batches,
                                "global_step": global_step,
                                "lr": step_policy_lr,
                                "backbone_lr": step_backbone_lr,
                                "backbone_frozen": backbone_frozen,
                                **record,
                            })
                        interval.zero_()
    metrics = reduce_totals(totals, world_size)
    if train:
        metrics.update(
            {
                "last_lr": step_policy_lr,
                "last_backbone_lr": step_backbone_lr,
                "last_backbone_frozen": backbone_frozen,
            }
        )
    return metrics, global_step


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", default=os.environ.get("PREPROCESSED_DATASET_DIR"))
    parser.add_argument(
        "--dataset-manifest",
        default=os.environ.get("LEROBOT_DATASET_MANIFEST"),
        help="Multi-root manifest generated by scripts/pick_tube_vision/02_prepare_data.sh",
    )
    parser.add_argument(
        "--dataset-format",
        choices=("preprocessed", "lerobot-v21"),
        default=os.environ.get("DATASET_FORMAT", "preprocessed"),
        help="Read the existing mmap snapshot or LeRobot v2.1 episode Parquet directly",
    )
    action_chunk_size_env = os.environ.get("ACTION_CHUNK_SIZE")
    parser.add_argument(
        "--action-chunk-size",
        type=int,
        default=(
            int(action_chunk_size_env)
            if action_chunk_size_env is not None
            else None
        ),
        help=(
            "Training action horizon. Preprocessed data defaults to its source "
            "chunk size; LeRobot v2.1 defaults to 32"
        ),
    )
    parser.add_argument("--output-dir", default=os.environ.get("OUTPUT_DIR", "/workspace/output"))
    parser.add_argument("--run-id", default=os.environ.get("RUN_ID"))
    parser.add_argument("--resume-from", default=os.environ.get("RESUME_FROM"))
    parser.add_argument(
        "--resume-mode",
        choices=("exact", "finetune"),
        default=os.environ.get("RESUME_MODE", "exact"),
    )
    parser.add_argument(
        "--lr-scheduler",
        choices=("warmup_cosine", "constant"),
        default=os.environ.get("LR_SCHEDULER", "warmup_cosine"),
    )
    parser.add_argument("--seed", type=int, default=int(os.environ.get("SEED", "42")))
    parser.add_argument("--epochs", type=int, default=int(os.environ.get("EPOCHS", "100")))
    parser.add_argument("--batch-size", type=int, default=int(os.environ.get("BATCH_SIZE", "8")))
    parser.add_argument("--lr", type=float, default=float(os.environ.get("LR", "1e-4")))
    parser.add_argument("--lr-final", type=float, default=float(os.environ.get("LR_FINAL", "5e-6")))
    parser.add_argument(
        "--backbone-lr",
        type=float,
        default=float(os.environ.get("BACKBONE_LR", "1e-5")),
    )
    parser.add_argument(
        "--backbone-lr-final",
        type=float,
        default=float(os.environ.get("BACKBONE_LR_FINAL", "5e-7")),
    )
    parser.add_argument("--weight-decay", type=float, default=float(os.environ.get("WEIGHT_DECAY", "1e-6")))
    parser.add_argument(
        "--warmup-epochs",
        type=int,
        default=int(os.environ.get("WARMUP_EPOCHS", "0")),
    )
    parser.add_argument(
        "--cosine-t-max-epochs",
        type=int,
        default=int(os.environ.get("COSINE_T_MAX_EPOCHS", os.environ.get("EPOCHS", "100"))),
    )
    parser.add_argument(
        "--backbone-freeze-epochs",
        type=int,
        default=int(os.environ.get("BACKBONE_FREEZE_EPOCHS", "0")),
    )
    parser.add_argument(
        "--backbone-bn-eval",
        action=argparse.BooleanOptionalAction,
        default=env_bool("BACKBONE_BN_EVAL", False),
    )
    parser.add_argument(
        "--mask-training-padding",
        action=argparse.BooleanOptionalAction,
        default=env_bool("MASK_TRAINING_PADDING", True),
    )
    parser.add_argument("--hidden-dim", type=int, default=int(os.environ.get("HIDDEN_DIM", "512")))
    parser.add_argument("--layers", type=int, default=int(os.environ.get("LAYERS", "6")))
    parser.add_argument("--heads", type=int, default=int(os.environ.get("HEADS", "8")))
    parser.add_argument("--image-size", type=int, default=int(os.environ.get("DECO_IMAGE_SIZE", "256")))
    parser.add_argument("--inference-steps", type=int, default=int(os.environ.get("DECO_INFERENCE_STEPS", "5")))
    parser.add_argument(
        "--backbone-weights", default=os.environ.get("DECO_BACKBONE_WEIGHTS")
    )
    parser.add_argument("--rope-height", type=int, default=int(os.environ.get("DECO_ROPE_HEIGHT", "256")))
    parser.add_argument("--rope-width", type=int, default=int(os.environ.get("DECO_ROPE_WIDTH", "256")))
    parser.add_argument("--use-task-condition", action=argparse.BooleanOptionalAction,
                        default=env_bool("DECO_USE_TASK_CONDITION", False))
    parser.add_argument(
        "--augmentation-enabled",
        action=argparse.BooleanOptionalAction,
        default=env_bool("DECO_AUGMENTATION_ENABLED", True),
    )
    parser.add_argument(
        "--augmentation-identity-probability", type=float,
        default=float(os.environ.get("DECO_AUGMENTATION_IDENTITY_PROBABILITY", "0.25")),
    )
    parser.add_argument(
        "--augmentation-low-light-probability", type=float,
        default=float(os.environ.get("DECO_AUGMENTATION_LOW_LIGHT_PROBABILITY", "0.55")),
    )
    parser.add_argument(
        "--augmentation-mild-probability", type=float,
        default=float(os.environ.get("DECO_AUGMENTATION_MILD_PROBABILITY", "0.20")),
    )
    parser.add_argument(
        "--augmentation-exposure-probability", type=float,
        default=float(os.environ.get("DECO_AUGMENTATION_EXPOSURE_PROBABILITY", "0.5")),
    )
    parser.add_argument(
        "--augmentation-exposure-range", type=float, nargs=2,
        default=env_tuple("DECO_AUGMENTATION_EXPOSURE_RANGE", (0.58, 0.90)),
    )
    parser.add_argument(
        "--augmentation-gamma-range", type=float, nargs=2,
        default=env_tuple("DECO_AUGMENTATION_GAMMA_RANGE", (1.10, 1.50)),
    )
    parser.add_argument(
        "--augmentation-mild-brightness-range", type=float, nargs=2,
        default=env_tuple("DECO_AUGMENTATION_MILD_BRIGHTNESS_RANGE", (0.90, 1.10)),
    )
    parser.add_argument(
        "--augmentation-contrast-range", type=float, nargs=2,
        default=env_tuple("DECO_AUGMENTATION_CONTRAST_RANGE", (0.85, 1.10)),
    )
    parser.add_argument(
        "--augmentation-saturation-range", type=float, nargs=2,
        default=env_tuple("DECO_AUGMENTATION_SATURATION_RANGE", (0.90, 1.10)),
    )
    parser.add_argument(
        "--augmentation-blur-probability", type=float,
        default=float(os.environ.get("DECO_AUGMENTATION_BLUR_PROBABILITY", "0.20")),
    )
    parser.add_argument(
        "--augmentation-blur-kernel-sizes", type=int, nargs="+",
        default=env_tuple("DECO_AUGMENTATION_BLUR_KERNEL_SIZES", (3, 5), int),
    )
    parser.add_argument(
        "--augmentation-blur-sigma-range", type=float, nargs=2,
        default=env_tuple("DECO_AUGMENTATION_BLUR_SIGMA_RANGE", (0.1, 1.0)),
    )
    parser.add_argument("--workers", type=int, default=int(os.environ.get("DATALOADER_WORKERS", "2")))
    parser.add_argument("--limit-samples", type=int, default=None)
    parser.add_argument(
        "--validation-ratio",
        type=float,
        default=float(os.environ.get("VALIDATION_RATIO", "0.1")),
        help="Episode-level validation fraction for --dataset-format lerobot-v21",
    )
    parser.add_argument(
        "--episode-split-seed",
        type=int,
        default=int(os.environ.get("EPISODE_SPLIT_SEED", "42")),
        help="Deterministic episode split seed for --dataset-format lerobot-v21",
    )
    parser.add_argument("--save-every", type=int, default=1)
    parser.add_argument("--keep-last-checkpoints", type=int,
                        default=int(os.environ.get("KEEP_LAST_CHECKPOINTS", "5")))
    parser.add_argument("--log-every-steps", type=int, default=int(os.environ.get("LOG_EVERY_STEPS", "100")))
    parser.add_argument("--validation-seed", type=int, default=int(os.environ.get("VALIDATION_SEED", "12345")))
    parser.add_argument(
        "--validation-noise-seeds",
        type=int,
        default=int(os.environ.get("VALIDATION_NOISE_SEEDS", "1")),
    )
    parser.add_argument(
        "--train-subset-validation-samples",
        type=int,
        default=int(os.environ.get("TRAIN_SUBSET_VALIDATION_SAMPLES", "0")),
        help=(
            "Number of fixed training samples used for in-distribution validation; "
            "zero uses the unseen validation sample count"
        ),
    )
    parser.add_argument(
        "--train-subset-validation-seed",
        type=int,
        default=int(os.environ.get("TRAIN_SUBSET_VALIDATION_SEED", "12345")),
    )
    parser.add_argument("--early-stopping-patience", type=int, default=int(os.environ.get("EARLY_STOPPING_PATIENCE", "5")))
    parser.add_argument(
        "--early-stopping-min-delta",
        type=float,
        default=float(os.environ.get("EARLY_STOPPING_MIN_DELTA", "0")),
    )
    parser.add_argument("--torchscript-image-height", type=int, default=int(os.environ.get("TORCHSCRIPT_IMAGE_HEIGHT", "208")))
    parser.add_argument("--torchscript-image-width", type=int, default=int(os.environ.get("TORCHSCRIPT_IMAGE_WIDTH", "320")))
    args = parser.parse_args()
    augmentation_config = augmentation_config_from_args(args)
    dataset_source = training_dataset_source(args)
    if args.action_chunk_size is not None and args.action_chunk_size <= 0:
        raise ValueError("ACTION_CHUNK_SIZE must be positive")
    if args.dataset_format == "lerobot-v21" and not 0.0 < args.validation_ratio < 1.0:
        raise ValueError("VALIDATION_RATIO must be between 0 and 1")
    if args.epochs <= 0 or args.batch_size <= 0:
        raise ValueError("epochs and batch_size must be positive")
    if args.lr <= 0 or not 0 <= args.lr_final <= args.lr:
        raise ValueError("LR must be positive and LR_FINAL must be in [0, LR]")
    if args.backbone_lr <= 0 or not 0 <= args.backbone_lr_final <= args.backbone_lr:
        raise ValueError(
            "BACKBONE_LR must be positive and BACKBONE_LR_FINAL must be in "
            "[0, BACKBONE_LR]"
        )
    if args.warmup_epochs < 0 or args.cosine_t_max_epochs <= 0:
        raise ValueError("WARMUP_EPOCHS must be non-negative and COSINE_T_MAX_EPOCHS positive")
    if args.backbone_freeze_epochs < 0:
        raise ValueError("BACKBONE_FREEZE_EPOCHS must be non-negative")
    if args.validation_noise_seeds <= 0:
        raise ValueError("VALIDATION_NOISE_SEEDS must be positive")
    if args.train_subset_validation_samples < 0:
        raise ValueError("TRAIN_SUBSET_VALIDATION_SAMPLES must be non-negative")
    if args.early_stopping_min_delta < 0:
        raise ValueError("EARLY_STOPPING_MIN_DELTA must be non-negative")
    if args.resume_mode == "finetune" and not args.resume_from:
        raise ValueError("RESUME_MODE=finetune requires RESUME_FROM")
    if args.resume_mode == "finetune" and args.lr_scheduler != "constant":
        raise ValueError(
            "RESUME_MODE=finetune requires LR_SCHEDULER=constant"
        )
    if args.lr_scheduler == "constant":
        if args.warmup_epochs != 0:
            raise ValueError("Constant LR scheduling requires WARMUP_EPOCHS=0")
        if args.lr_final != args.lr:
            raise ValueError("Constant LR scheduling requires LR_FINAL=LR")
        if args.backbone_lr_final != args.backbone_lr:
            raise ValueError(
                "Constant LR scheduling requires BACKBONE_LR_FINAL=BACKBONE_LR"
            )

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device, rank, world_size = setup_dist()
    main_rank = rank == 0
    args.run_id = args.run_id or os.environ.get("HOSTNAME")
    if not args.run_id:
        raise ValueError("K8S-C01: RUN_ID is required to isolate training outputs")
    output_dir = Path(args.output_dir) / args.run_id
    if main_rank:
        output_dir.mkdir(parents=True, exist_ok=True)

    # Verify the preprocessed dataset. By default run on ALL ranks in parallel
    # (not just rank 0): the dataset is read-only/shared so every rank reaches the
    # same verdict, and parallelizing avoids a long stall on rank 0 that makes the
    # other ranks' NCCL communicator setup time out (store->get wait timeout after
    # 600000ms) when the dataset lives on a slow shared filesystem (CPFS/NFS).
    #
    # SKIP_DATASET_VERIFY=1 skips it entirely: the dataset is immutable, so once
    # it has been verified (READY file present) re-stat'ing ~33k shards x N files
    # on NFS every launch is wasteful (adds 10-30 min startup on CPFS). The
    # PreprocessedDECODataset constructors below still sanity-check READY/manifest.
    skip_verify = env_bool("SKIP_DATASET_VERIFY")
    verification = [None]
    if args.dataset_format == "preprocessed" and not skip_verify:
        try:
            verify_preprocessed_dataset(dataset_source)
        except Exception as exc:
            verification[0] = f"{type(exc).__name__}: {exc}"
    if world_size > 1:
        dist.broadcast_object_list(verification, src=0, device=device)
    if verification[0] is not None:
        cleanup_dist()
        raise ValueError(f"Preprocessed dataset integrity verification failed: {verification[0]}")

    train_dataset, val_dataset = create_training_datasets(args)
    contract = train_dataset.metadata
    if val_dataset.metadata != contract:
        raise ValueError("Train/validation dataset contracts differ")
    action_mode = resolve_dataset_action_mode(contract)
    if len(contract["camera_names"]) not in (2, 3):
        raise ValueError("DECO Stage 1 requires two or three camera streams")
    obs_index_list = observation_indices(contract)
    observation_index = torch.tensor(obs_index_list, dtype=torch.long, device=device)
    action_dim = int(contract["action_dim"])
    chunk_size = int(contract["chunk_size"])
    if train_dataset.task_ids != val_dataset.task_ids:
        raise ValueError("Preprocessed train/val task mappings differ")
    num_tasks = len(train_dataset.task_ids)

    train_subset_val_samples = args.train_subset_validation_samples or len(val_dataset)
    train_subset_val_indices = deterministic_subset_indices(
        len(train_dataset),
        min(train_subset_val_samples, len(train_dataset)),
        args.train_subset_validation_seed,
    )
    train_subset_val_dataset = Subset(train_dataset, train_subset_val_indices)

    train_loader, train_sampler = make_loader(
        train_dataset, args.batch_size, args.workers, rank, world_size, True, args.seed
    )
    val_loader, val_sampler = make_loader(
        val_dataset, args.batch_size, args.workers, rank, world_size, False, args.seed
    )
    train_subset_val_loader, _ = make_loader(
        train_subset_val_dataset,
        args.batch_size,
        args.workers,
        rank,
        world_size,
        False,
        args.seed,
    )
    validate_nonempty_training_loader(
        train_dataset, train_loader, args.batch_size, world_size, rank, device
    )
    steps_per_epoch = len(train_loader)
    warmup_steps = args.warmup_epochs * steps_per_epoch
    cosine_t_max_steps = args.cosine_t_max_epochs * steps_per_epoch
    backbone_freeze_steps = args.backbone_freeze_epochs * steps_per_epoch

    config = vars(args) | {
        "model_type": MODEL_TYPE,
        "source_obs_dim": int(contract["obs_dim"]),
        "obs_dim": action_dim,
        "action_dim": action_dim,
        "chunk_size": chunk_size,
        "source_chunk_size": train_dataset.source_chunk_size,
        "observation_indices": obs_index_list,
        "state_columns": contract["state_columns"],
        "action_columns": contract["action_columns"],
        "action_mode": action_mode,
        **action_mode_config_fields(action_mode),
        "expected_sample_hz": contract.get("expected_sample_hz"),
        "source_format": contract.get("source_format"),
        "state_layout": contract.get("state_layout"),
        "rotation_representation": contract.get("rotation_representation"),
        "gripper_mode": contract.get("gripper_mode"),
        "terminal_action_policy": contract.get("terminal_action_policy"),
        "statistics_source": contract.get("statistics_source"),
        "world_size": world_size,
        "camera_names": contract["camera_names"],
        "task_ids": train_dataset.task_ids,
        "num_tasks": num_tasks,
        "dataset_id": train_dataset.manifest["dataset_id"],
        "train_samples": len(train_dataset),
        "val_samples": len(val_dataset),
        "val_unseen_samples": len(val_dataset),
        "val_train_subset_samples": len(train_subset_val_dataset),
        "val_train_subset_seed": args.train_subset_validation_seed,
        "steps_per_epoch": steps_per_epoch,
        "warmup_steps": warmup_steps,
        "cosine_t_max_steps": cosine_t_max_steps,
        "backbone_freeze_steps": backbone_freeze_steps,
        "training_state_version": 2,
        "objective_version": (
            "masked-flow-mse-v1"
            if args.mask_training_padding else "unmasked-flow-mse-v1"
        ),
        "validation_metric_version": "no-repeat-masked-element-mean-v1",
        "dual_validation_version": "train-subset-and-unseen-v1",
        "validation_sampler": "contiguous-no-padding-v1",
        "scheduler_type": (
            "per-step-constant-v1"
            if args.lr_scheduler == "constant"
            else "per-step-warmup-cosine-v1"
        ),
        "rank_seed_scheme": "base-plus-rank-v1",
        "augmentation": asdict(augmentation_config),
    }
    model = build_model(config).to(device)
    if world_size > 1:
        model = DDP(
            model,
            device_ids=[device.index] if device.type == "cuda" else None,
            find_unused_parameters=True,
        )
    raw_model = model.module if isinstance(model, DDP) else model
    parameter_groups, backbone_parameters = optimizer_parameter_groups(
        raw_model,
        policy_lr=args.lr,
        backbone_lr=args.backbone_lr,
        weight_decay=args.weight_decay,
    )
    optimizer = torch.optim.AdamW(
        parameter_groups,
        betas=(0.95, 0.999),
    )
    if args.lr_scheduler == "constant":
        scheduler = constant_lr_scheduler(optimizer)
    else:
        policy_final_ratio = args.lr_final / args.lr
        backbone_final_ratio = args.backbone_lr_final / args.backbone_lr
        lr_lambdas = []
        for group in optimizer.param_groups:
            group_name = str(group["group_name"])
            if group_name.startswith("policy"):
                lr_lambdas.append(
                    lambda step, warmup_steps=warmup_steps,
                    cosine_t_max_steps=cosine_t_max_steps,
                    final_ratio=policy_final_ratio: warmup_cosine_multiplier(
                        step, warmup_steps, cosine_t_max_steps, final_ratio
                    )
                )
            elif group_name.startswith("backbone"):
                lr_lambdas.append(
                    lambda step, freeze_steps=backbone_freeze_steps,
                    cosine_t_max_steps=cosine_t_max_steps,
                    final_ratio=backbone_final_ratio: backbone_cosine_multiplier(
                        step, freeze_steps, cosine_t_max_steps, final_ratio
                    )
                )
            else:
                raise ValueError(f"Unknown optimizer group: {group_name}")
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda=lr_lambdas
        )
    config["optimizer_group_names"] = [
        str(group["group_name"]) for group in optimizer.param_groups
    ]
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    start_epoch = 0
    global_step = 0
    best_val = float("inf")
    patience_best_val = float("inf")
    stale_epochs = 0
    resume_record = None
    if args.resume_from:
        resumed = load_checkpoint(args.resume_from, device)
        validate_resume_config(
            resumed.get("config", {}),
            config,
            resume_mode=args.resume_mode,
        )
        if len(resumed.get("rng_states", [])) != world_size:
            raise ValueError("Cannot restore per-rank RNG with a different world size")
        raw_model.load_state_dict(resumed["model"], strict=True)
        optimizer.load_state_dict(resumed["optimizer"])
        scaler.load_state_dict(resumed["scaler"])
        start_epoch = int(resumed["epoch"])
        global_step = int(resumed.get("global_step", 0))
        source_scheduler_step = int(
            resumed.get("scheduler", {}).get("last_epoch", -1)
        )
        if source_scheduler_step != global_step:
            raise ValueError(
                "Source checkpoint scheduler/global_step mismatch: "
                f"{source_scheduler_step} != {global_step}"
            )
        if args.resume_mode == "finetune":
            source_val = float(resumed["val_loss"])
            source_best_val = float(resumed["best_val"])
            if not math.isclose(
                source_val,
                source_best_val,
                rel_tol=0.0,
                abs_tol=1e-12,
            ):
                raise ValueError(
                    "Fine-tune resume requires a best checkpoint: "
                    f"val_loss={source_val}, best_val={source_best_val}"
                )
            override_optimizer_partition_lrs(
                optimizer,
                policy_lr=args.lr,
                backbone_lr=args.backbone_lr,
            )
            scheduler = constant_lr_scheduler(
                optimizer,
                global_step=global_step,
            )
            best_val = source_val
            patience_best_val = source_val
            stale_epochs = 0
        else:
            scheduler.load_state_dict(resumed["scheduler"])
            best_val = float(resumed["best_val"])
            patience_best_val = float(
                resumed.get("patience_best_val", best_val)
            )
            stale_epochs = int(resumed.get("stale_epochs", 0))
        if scheduler.last_epoch != global_step:
            raise ValueError(
                "Checkpoint scheduler/global_step mismatch: "
                f"{scheduler.last_epoch} != {global_step}"
            )
        restore_rng_state(resumed["rng_states"][rank])
        resume_record = {
            "event": "resume",
            "mode": args.resume_mode,
            "source": args.resume_from,
            "source_run_id": resumed.get("run_id"),
            "source_epoch": start_epoch,
            "global_step": global_step,
            "best_val": best_val,
            "stale_epochs": stale_epochs,
            "lr": optimizer_partition_lr(optimizer, "policy"),
            "backbone_lr": optimizer_partition_lr(optimizer, "backbone"),
            "scheduler_type": config["scheduler_type"],
        }
    else:
        seed_training_rng(args.seed, rank)
    if start_epoch >= args.epochs:
        raise ValueError(
            f"Checkpoint epoch {start_epoch} must be less than EPOCHS "
            f"{args.epochs}"
        )

    stats = train_dataset.stats
    if main_rank:
        (output_dir / "config.json").write_text(json.dumps(config, indent=2))
        (output_dir / "dataset_stats.json").write_text(
            json.dumps(jsonable_stats(stats), indent=2)
        )
        print(json.dumps({"event": "start", **config}), flush=True)
        metrics_logger = MetricsLogger(output_dir)
        if resume_record is not None:
            metrics_logger.log(resume_record)
            print(json.dumps(resume_record), flush=True)
    else:
        metrics_logger = None
    started = time.monotonic()

    def log_step(record):
        if main_rank:
            record.setdefault(
                "lr", optimizer_partition_lr(optimizer, "policy")
            )
            record.setdefault(
                "backbone_lr",
                optimizer_partition_lr(optimizer, "backbone"),
            )
            record.setdefault(
                "backbone_frozen",
                record["global_step"] < backbone_freeze_steps,
            )
            record["elapsed_seconds"] = time.monotonic() - started
            metrics_logger.log(record)
            print(json.dumps(record), flush=True)

    for epoch in range(start_epoch, args.epochs):
        if train_sampler:
            train_sampler.set_epoch(epoch)
        train_metrics, global_step = run_epoch(
            model=model,
            loader=train_loader,
            device=device,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            observation_index=observation_index,
            image_size=args.image_size,
            use_task_condition=args.use_task_condition,
            train=True,
            world_size=world_size,
            backbone_parameters=backbone_parameters,
            backbone_freeze_steps=backbone_freeze_steps,
            backbone_bn_eval=args.backbone_bn_eval,
            mask_training_padding=args.mask_training_padding,
            epoch=epoch + 1,
            global_step=global_step,
            log_every_steps=args.log_every_steps,
            metric_callback=log_step,
            rank=rank,
            augmentation_config=augmentation_config,
        )
        sync_model_buffers(raw_model, world_size)
        validation_metrics = {}
        for validation_name, validation_loader in (
            ("train", train_subset_val_loader),
            ("unseen", val_loader),
        ):
            validation_records = []
            for seed_index in range(args.validation_noise_seeds):
                seed_metrics, _ = run_epoch(
                    model=raw_model,
                    loader=validation_loader,
                    device=device,
                    optimizer=None,
                    scheduler=None,
                    scaler=scaler,
                    observation_index=observation_index,
                    image_size=args.image_size,
                    use_task_condition=args.use_task_condition,
                    train=False,
                    world_size=world_size,
                    validation_seed=args.validation_seed + seed_index * 100_003,
                    rank=rank,
                )
                validation_records.append(seed_metrics)
            validation_metrics[validation_name] = average_validation_metrics(
                validation_records
            )
        val_train_metrics = validation_metrics["train"]
        val_unseen_metrics = validation_metrics["unseen"]
        if not all(
            math.isfinite(metrics["loss"])
            for metrics in (val_train_metrics, val_unseen_metrics)
        ):
            cleanup_dist()
            raise FloatingPointError(
                "Validation loss is not finite: "
                f"train={val_train_metrics['loss']}, "
                f"unseen={val_unseen_metrics['loss']}"
            )
        epoch_lr = train_metrics["last_lr"]
        epoch_backbone_lr = train_metrics["last_backbone_lr"]
        backbone_frozen = train_metrics["last_backbone_frozen"]
        local_rng = capture_rng_state()
        if world_size > 1:
            rng_states = [None] * world_size if main_rank else None
            dist.gather_object(local_rng, rng_states, dst=0)
        else:
            rng_states = [local_rng]

        stop = False
        if main_rank:
            absolute_improved = val_unseen_metrics["loss"] < best_val
            significant_improved = (
                val_unseen_metrics["loss"]
                < patience_best_val - args.early_stopping_min_delta
            )
            if absolute_improved:
                best_val = val_unseen_metrics["loss"]
            if significant_improved:
                patience_best_val = val_unseen_metrics["loss"]
                stale_epochs = 0
            else:
                stale_epochs += 1
            record = {
                "event": "epoch", "epoch": epoch + 1, "global_step": global_step,
                "train": train_metrics,
                # Keep val as an alias for unseen so existing plots/checkpoints
                # and exact-resume semantics remain backward compatible.
                "val": val_unseen_metrics,
                "val_train": val_train_metrics,
                "val_unseen": val_unseen_metrics,
                "lr": epoch_lr,
                "backbone_lr": epoch_backbone_lr,
                "backbone_frozen": backbone_frozen,
                "elapsed_seconds": time.monotonic() - started,
            }
            metrics_logger.log_epoch(record)
            print(json.dumps(record), flush=True)
            checkpoint = {
                "model": raw_model.state_dict(), "config": config,
                "stats": jsonable_stats(stats), "epoch": epoch + 1,
                "val_loss": val_unseen_metrics["loss"],
                "val_train_loss": val_train_metrics["loss"],
                "val_unseen_loss": val_unseen_metrics["loss"],
                "best_val": best_val,
                "patience_best_val": patience_best_val,
                "stale_epochs": stale_epochs, "optimizer": optimizer.state_dict(),
                "scaler": scaler.state_dict(), "scheduler": scheduler.state_dict(),
                "rng_states": rng_states, "run_id": args.run_id,
                "global_step": global_step,
            }
            periodic = (epoch + 1) % args.save_every == 0
            if periodic:
                atomic_torch_save(checkpoint, output_dir / f"deco_stage1_epoch_{epoch + 1}.pt")
                atomic_torch_save(checkpoint, output_dir / "deco_stage1_latest.pt")
            if absolute_improved:
                atomic_torch_save(checkpoint, output_dir / "deco_stage1_best.pt")
            if periodic or absolute_improved:
                epoch_ts = output_dir / f"deco_stage1_epoch_{epoch + 1}.ts"
                artifact_rng = capture_rng_state()
                try:
                    metadata = export_policy(
                        raw_model, jsonable_stats(stats), config, epoch_ts,
                        args.torchscript_image_height, args.torchscript_image_width,
                        epoch + 1, val_unseen_metrics["loss"],
                    )
                    if periodic:
                        copy_torchscript_artifact(
                            epoch_ts, output_dir / "deco_stage1_latest.ts"
                        )
                    if absolute_improved:
                        copy_torchscript_artifact(
                            epoch_ts, output_dir / "deco_stage1_best.ts"
                        )
                finally:
                    restore_rng_state(artifact_rng)
                print(json.dumps({"event": "torchscript_saved", **metadata}), flush=True)
            if periodic:
                removed = prune_old_checkpoints(output_dir, args.keep_last_checkpoints)
                if removed:
                    print(json.dumps({
                        "event": "checkpoints_pruned", "epoch": epoch + 1,
                        "keep_last": args.keep_last_checkpoints, "removed": removed,
                    }), flush=True)
            stop = args.early_stopping_patience > 0 and stale_epochs >= args.early_stopping_patience
            if stop:
                print(json.dumps({
                    "event": "early_stopping", "epoch": epoch + 1,
                    "patience": args.early_stopping_patience,
                    "min_delta": args.early_stopping_min_delta,
                    "best_val": best_val,
                    "patience_best_val": patience_best_val,
                }), flush=True)
        if world_size > 1:
            stop_tensor = torch.tensor(int(stop), device=device)
            dist.broadcast(stop_tensor, src=0)
            stop = bool(stop_tensor.item())
            dist.barrier()
        if stop:
            break
    if device.type == "cuda" and rank == 0:
        peak_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)
        peak_reserved_gb = torch.cuda.max_memory_reserved() / (1024 ** 3)
        print(json.dumps({
            "event": "training_peak_memory",
            "peak_alloc_gb": round(peak_gb, 3),
            "peak_reserved_gb": round(peak_reserved_gb, 3),
        }), flush=True)
    cleanup_dist()


if __name__ == "__main__":
    main()
