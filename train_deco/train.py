import argparse
import hashlib
import json
import math
import os
import random
import time
from contextlib import contextmanager, nullcontext
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
    AUGMENTATION_PRESET_NAMES,
    LowLightAugmentationConfig,
    augmentation_preset,
    augment_training_images,
    letterbox_and_normalize,
    select_deco_observation,
    letterbox_tactile_images,
    validate_augmentation_config,
)
from .lerobot_vision_dataset import TACTILE_NAMES
from .metrics import MetricsLogger, WandbMetricsLogger
from .model_factory import (
    MODEL_TYPE,
    STAGE2_MODEL_TYPE,
    build_model,
    build_stage2_model,
    observation_indices,
)
from .preprocessed_dataset import PreprocessedDECODataset, verify_preprocessed_dataset
from .resume import validate_resume_config
from .stage2_initialization import (
    configure_stage2_trainability,
    initialize_stage2_from_stage1,
    load_stage1_reference,
    validate_stage1_checkpoint_contract,
    verify_stage2_stage1_parity,
)
from .tactile_encoder_conversion import ResolvedTactileEncoder, load_tactile_encoder_weights
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
    stage2_gradient_diagnostics,
    stage2_optimizer_parameter_groups,
)



STAGE2_CHECKPOINT_SCHEMA_VERSION = 1

_STAGE2_CHECKPOINT_DRIVEN_CONFIG_KEYS = (
    "hidden_dim",
    "layers",
    "heads",
    "image_size",
    "inference_steps",
    "rope_height",
    "rope_width",
    "use_task_condition",
    "tactile_adapter_rank",
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
    legacy_config = LowLightAugmentationConfig(
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
    name = getattr(args, "augmentation_preset", None)
    if name is None:
        validate_augmentation_config(legacy_config)
        return legacy_config
    expected_legacy = LowLightAugmentationConfig(enabled=args.augmentation_enabled)
    if legacy_config != expected_legacy:
        raise ValueError(
            "--augmentation-preset cannot be combined with fine-grained "
            "augmentation overrides"
        )
    return augmentation_preset(name, enabled=args.augmentation_enabled)


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

        datasets = build_lerobot_vision_datasets(
            training_dataset_source(args),
            action_chunk_size=args.action_chunk_size or 32,
            validation_ratio=args.validation_ratio,
            split_seed=args.episode_split_seed,
            train_limit=args.limit_samples,
            val_limit=val_limit,
            include_tactile=getattr(args, "stage", 1) == 2,
        )
        if getattr(args, "bread_phase", False):
            from .bread_phase.dataset import build_bread_phase_datasets

            return build_bread_phase_datasets(*datasets)
        return datasets
    raise ValueError(f"Unsupported dataset format: {args.dataset_format!r}")



def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_tactile_encoder_distributed(
    source: str | Path,
    cache_root: str | Path,
    *,
    rank: int,
    world_size: int,
    device: torch.device,
    resolver=None,
) -> ResolvedTactileEncoder:
    """Resolve once on rank zero, then broadcast one immutable artifact contract."""

    payload = [None]
    if rank == 0:
        try:
            if resolver is None:
                from .tactile_encoder_conversion import resolve_tactile_encoder

                resolver = resolve_tactile_encoder
            artifact = resolver(source, cache_root)
            payload[0] = {
                "ok": True,
                "weights_path": str(artifact.weights_path),
                "metadata_path": str(artifact.metadata_path),
                "source_sha256": artifact.source_sha256,
                "architecture": artifact.architecture,
                "embedding_dim": artifact.embedding_dim,
            }
        except Exception as error:
            payload[0] = {"ok": False, "error": f"{type(error).__name__}: {error}"}
    if world_size > 1:
        dist.broadcast_object_list(payload, src=0, device=device)
        dist.barrier()
    result = payload[0]
    if not isinstance(result, dict) or not result.get("ok"):
        error = result.get("error", "unknown conversion error") if isinstance(result, dict) else "missing conversion result"
        raise RuntimeError(f"Distributed tactile encoder resolution failed: {error}")
    return ResolvedTactileEncoder(
        weights_path=Path(result["weights_path"]),
        metadata_path=Path(result["metadata_path"]),
        source_sha256=str(result["source_sha256"]),
        architecture=str(result["architecture"]),
        embedding_dim=int(result["embedding_dim"]),
    )


def _parameter_categories(parameter_report) -> dict[str, dict[str, list[str]]]:
    return {
        "trainable": {
            category: list(names)
            for category, names in parameter_report.trainable_by_category.items()
        },
        "frozen": {
            category: list(names)
            for category, names in parameter_report.frozen_by_category.items()
        },
    }


def build_stage2_checkpoint_metadata(
    model,
    parameter_report,
    *,
    stage1_checkpoint: str | Path,
    tactile_artifact: ResolvedTactileEncoder,
    tactile_adapter_rank: int,
) -> dict:
    artifact_metadata = json.loads(
        tactile_artifact.metadata_path.read_text(encoding="utf-8")
    )
    return {
        "model_type": STAGE2_MODEL_TYPE,
        "tactile_field_order": list(TACTILE_NAMES),
        "tactile_encoder": {
            "source_sha256": tactile_artifact.source_sha256,
            "artifact_sha256": artifact_metadata.get("weights_sha256")
            or _sha256_file(tactile_artifact.weights_path),
            "artifact_path": str(tactile_artifact.weights_path.resolve()),
            "metadata_path": str(tactile_artifact.metadata_path.resolve()),
            "architecture": tactile_artifact.architecture,
            "embedding_dim": tactile_artifact.embedding_dim,
        },
        "tactile_adapter_rank": int(tactile_adapter_rank),
        "gate_values": stage2_gradient_diagnostics(model, parameter_report)["gate_values"],
        "parameter_categories": _parameter_categories(parameter_report),
        "parameter_counts": {
            "total": parameter_report.total_parameters,
            "trainable": parameter_report.trainable_parameters,
        },
        "stage1_checkpoint": {
            "path": str(Path(stage1_checkpoint).expanduser().resolve()),
            "sha256": _sha256_file(stage1_checkpoint),
        },
    }

def build_stage2_parity_inputs(
    config: dict, device: torch.device, *, seed: int = 20260827
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    """Create fixed production-shaped inputs without changing training RNG."""
    generator = torch.Generator(device="cpu").manual_seed(seed)
    camera_count = len(config["camera_names"])
    image_size = int(config["image_size"])
    images = [
        torch.randn(1, 3, image_size, image_size, generator=generator).to(device)
        for _ in range(camera_count)
    ]
    inputs = {
        "img1": images[0],
        "img2": images[1],
        "obs": torch.randn(
            1, int(config["action_dim"]), generator=generator
        ).to(device),
        "act": torch.randn(
            1, int(config["chunk_size"]), int(config["action_dim"]),
            generator=generator,
        ).to(device),
    }
    if camera_count == 3:
        inputs["img3"] = images[2]
    if config.get("use_task_condition", False):
        inputs["task_idx"] = torch.zeros(1, dtype=torch.long, device=device)
    tactile = torch.rand(1, 4, 3, 224, 224, generator=generator).to(device)
    return inputs, tactile



def export_stage2_torchscript_artifacts(
    *,
    policy,
    stats: dict,
    config: dict,
    stage2_metadata: dict,
    output_dir: str | Path,
    epoch: int,
    val_loss: float,
    image_height: int,
    image_width: int,
    periodic: bool,
    improved: bool,
    exporter=export_policy,
    copier=copy_torchscript_artifact,
) -> list[dict]:
    """Export Stage2 aliases after PT durability; report failures without raising."""
    if not (periodic or improved):
        return []
    output_dir = Path(output_dir)
    epoch_ts = output_dir / f"deco_stage2_epoch_{epoch}.ts"
    artifact_rng = capture_rng_state()
    try:
        metadata = exporter(
            policy,
            stats,
            config,
            epoch_ts,
            image_height,
            image_width,
            epoch,
            val_loss,
            checkpoint_schema_version=STAGE2_CHECKPOINT_SCHEMA_VERSION,
            stage2_metadata=stage2_metadata,
        )
        if periodic:
            copier(epoch_ts, output_dir / "deco_stage2_latest.ts")
        if improved:
            copier(epoch_ts, output_dir / "deco_stage2_best.ts")
        return [{"event": "torchscript_saved", "stage": 2, **metadata}]
    except Exception as error:
        return [{
            "event": "torchscript_export_failed",
            "stage": 2,
            "epoch": int(epoch),
            "error": f"{type(error).__name__}: {error}",
        }]
    finally:
        restore_rng_state(artifact_rng)


def _require_metadata_mapping(value, label: str) -> dict:
    if not isinstance(value, dict):
        raise ValueError(f"Stage2 resume {label} must be a mapping")
    return value


def _require_sha256(value, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"Stage2 resume {label} must be a lowercase SHA256 digest")
    return value


def _require_nonempty_path(value, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Stage2 resume {label} must be a non-empty path")
    return value


def validate_stage2_resume_checkpoint(checkpoint: dict) -> dict:
    config = _require_metadata_mapping(checkpoint.get("config"), "config")
    if (
        checkpoint.get("stage") != 2
        or checkpoint.get("model_type") != STAGE2_MODEL_TYPE
        or config.get("model_type") != STAGE2_MODEL_TYPE
    ):
        raise ValueError("Stage2 exact resume rejected a Stage1 or non-Stage2 checkpoint")
    if checkpoint.get("checkpoint_schema_version") != STAGE2_CHECKPOINT_SCHEMA_VERSION:
        raise ValueError("Stage2 resume checkpoint schema/version is incompatible")
    metadata = _require_metadata_mapping(
        checkpoint.get("stage2_metadata"), "stage2_metadata"
    )
    required = {
        "model_type", "tactile_field_order", "tactile_encoder",
        "tactile_adapter_rank", "gate_values", "parameter_categories",
        "parameter_counts", "stage1_checkpoint",
    }
    missing = sorted(required - set(metadata))
    if missing:
        raise ValueError(f"Stage2 resume checkpoint metadata is missing: {missing}")
    if metadata["model_type"] != STAGE2_MODEL_TYPE:
        raise ValueError("Stage2 resume metadata model_type is incompatible")
    if metadata["tactile_field_order"] != list(TACTILE_NAMES):
        raise ValueError("Stage2 resume tactile field order is incompatible")

    adapter_rank = metadata["tactile_adapter_rank"]
    if isinstance(adapter_rank, bool) or not isinstance(adapter_rank, int) or adapter_rank <= 0:
        raise ValueError("Stage2 resume tactile_adapter_rank must be a positive integer")
    if config.get("tactile_adapter_rank") != adapter_rank:
        raise ValueError(
            "Stage2 resume adapter rank disagrees between config and metadata"
        )

    encoder = _require_metadata_mapping(
        metadata["tactile_encoder"], "tactile_encoder"
    )
    for key in ("source_sha256", "artifact_sha256"):
        _require_sha256(encoder.get(key), f"tactile_encoder.{key}")
    for key in ("artifact_path", "metadata_path"):
        _require_nonempty_path(encoder.get(key), f"tactile_encoder.{key}")
    if encoder.get("architecture") != "resnet18" or encoder.get("embedding_dim") != 512:
        raise ValueError("Stage2 resume tactile encoder architecture is incompatible")

    stage1 = _require_metadata_mapping(
        metadata["stage1_checkpoint"], "stage1_checkpoint"
    )
    _require_nonempty_path(stage1.get("path"), "stage1_checkpoint.path")
    _require_sha256(stage1.get("sha256"), "stage1_checkpoint.sha256")

    gates = _require_metadata_mapping(metadata["gate_values"], "gate_values")
    for name, value in gates.items():
        if (
            not isinstance(name, str)
            or not name.endswith(".tactile_gate")
            or isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
        ):
            raise ValueError("Stage2 resume gate_values schema is invalid")

    categories = _require_metadata_mapping(
        metadata["parameter_categories"], "parameter_categories"
    )
    for boundary in ("trainable", "frozen"):
        grouped = _require_metadata_mapping(
            categories.get(boundary), f"parameter_categories.{boundary}"
        )
        for names in grouped.values():
            if (
                not isinstance(names, list)
                or not all(isinstance(name, str) and name for name in names)
                or len(names) != len(set(names))
            ):
                raise ValueError(
                    f"Stage2 resume parameter_categories.{boundary} is invalid"
                )

    counts = _require_metadata_mapping(metadata["parameter_counts"], "parameter_counts")
    total = counts.get("total")
    trainable = counts.get("trainable")
    if (
        isinstance(total, bool)
        or isinstance(trainable, bool)
        or not isinstance(total, int)
        or not isinstance(trainable, int)
        or total <= 0
        or trainable <= 0
        or trainable >= total
    ):
        raise ValueError("Stage2 resume parameter_counts is invalid")
    return metadata


def stage2_config_from_resume_checkpoint(
    checkpoint: dict, current_config: dict
) -> dict:
    """Use checkpoint-owned architecture fields before constructing Stage2."""
    validate_stage2_resume_checkpoint(checkpoint)
    saved_config = checkpoint["config"]
    resolved = dict(current_config)
    missing = [
        key for key in _STAGE2_CHECKPOINT_DRIVEN_CONFIG_KEYS
        if key not in saved_config
    ]
    if missing:
        raise ValueError(
            f"Stage2 resume config is missing model fields: {missing}"
        )
    for key in _STAGE2_CHECKPOINT_DRIVEN_CONFIG_KEYS:
        resolved[key] = saved_config[key]
    return resolved

_STAGE2_RESUME_RUNTIME_ARGUMENT_KEYS = frozenset(
    {
        "resume_from",
        "resume_mode",
        "output_dir",
        "run_id",
        "epochs",
        "workers",
        "log_every_steps",
        "save_every",
        "keep_last_checkpoints",
        "torchscript_image_height",
        "torchscript_image_width",
        "validation_seed",
        "validation_noise_seeds",
        "train_subset_validation_samples",
        "train_subset_validation_seed",
        "stage1_checkpoint",
        "tactile_encoder_checkpoint",
        "tactile_encoder_cache",
        "wandb_enabled",
        "wandb_project",
        "wandb_entity",
        "wandb_group",
        "wandb_tags",
        "wandb_mode",
        "wandb_run_id",
    }
)


def restore_stage2_resume_arguments(args, *, checkpoint_loader=load_checkpoint) -> dict | None:
    """Load Stage2 exact-resume config before constructing stateful objects."""
    if args.stage != 2 or not args.resume_from:
        return None
    checkpoint = checkpoint_loader(args.resume_from, "cpu")
    validate_stage2_resume_checkpoint(checkpoint)
    saved_config = checkpoint["config"]
    for name in vars(args):
        if name in _STAGE2_RESUME_RUNTIME_ARGUMENT_KEYS:
            continue
        if name in saved_config:
            setattr(args, name, saved_config[name])
    args.augmentation_preset = saved_config.get("augmentation_preset")
    # Fresh-initialization sources are provenance only during exact resume.
    args.stage1_checkpoint = None
    args.tactile_encoder_checkpoint = None
    return checkpoint



def apply_restored_dataset_stats(stats: dict, *datasets) -> dict[str, np.ndarray]:
    """Apply checkpoint normalization statistics to every resume dataset."""
    required = {
        "observation_mean", "observation_std", "action_mean", "action_std"
    }
    if not isinstance(stats, dict) or not required.issubset(stats):
        raise ValueError("Stage2 resume checkpoint stats are incomplete")
    restored = {
        key: np.asarray(value, dtype=np.float32) for key, value in stats.items()
    }
    for key, value in restored.items():
        if value.size == 0 or not np.isfinite(value).all():
            raise ValueError(f"Stage2 resume checkpoint stat {key!r} is invalid")
    for key in ("observation_std", "action_std"):
        if np.any(restored[key] <= 0):
            raise ValueError(f"Stage2 resume checkpoint stat {key!r} must be positive")
    for dataset in datasets:
        dataset.stats = {key: value.copy() for key, value in restored.items()}
    return restored


def _validate_stage2_model_metadata(model, metadata: dict) -> None:
    named_parameters = dict(model.named_parameters())
    saved_gates = metadata["gate_values"]
    actual_gates = {
        name: float(parameter.detach().float().item())
        for name, parameter in named_parameters.items()
        if name.endswith(".tactile_gate")
    }
    if set(saved_gates) != set(actual_gates):
        raise ValueError("Stage2 resume gate names disagree with the model")
    for name, value in saved_gates.items():
        if not math.isclose(float(value), actual_gates[name], rel_tol=1e-6, abs_tol=1e-7):
            raise ValueError(f"Stage2 resume gate value disagrees for {name!r}")

    categories = metadata["parameter_categories"]
    saved_trainable = {
        name
        for names in categories["trainable"].values()
        for name in names
    }
    saved_frozen = {
        name
        for names in categories["frozen"].values()
        for name in names
    }
    actual_trainable = {
        name for name, parameter in named_parameters.items()
        if parameter.requires_grad
    }
    actual_frozen = set(named_parameters) - actual_trainable
    if saved_trainable != actual_trainable or saved_frozen != actual_frozen:
        raise ValueError(
            "Stage2 resume parameter categories disagree with the model"
        )


def restore_stage2_training_state(
    checkpoint: dict,
    *,
    model,
    optimizer,
    scheduler,
    scaler,
    current_config: dict,
    world_size: int,
    rank: int,
) -> dict:
    """Strictly restore every stateful component of an exact Stage2 resume."""

    metadata = validate_stage2_resume_checkpoint(checkpoint)
    validate_resume_config(
        checkpoint.get("config", {}),
        current_config,
        resume_mode="exact",
        expected_training_state_version=3,
        allowed_overrides={"validation_seed"},
    )
    if len(checkpoint.get("rng_states", [])) != world_size:
        raise ValueError("Cannot restore per-rank RNG with a different world size")
    model.load_state_dict(checkpoint["model"], strict=True)
    _validate_stage2_model_metadata(model, metadata)
    optimizer.load_state_dict(checkpoint["optimizer"])
    scheduler.load_state_dict(checkpoint["scheduler"])
    scaler.load_state_dict(checkpoint["scaler"])
    global_step = int(checkpoint.get("global_step", 0))
    if scheduler.last_epoch != global_step:
        raise ValueError(
            "Checkpoint scheduler/global_step mismatch: "
            f"{scheduler.last_epoch} != {global_step}"
        )
    restore_rng_state(checkpoint["rng_states"][rank])
    return {
        "epoch": int(checkpoint["epoch"]),
        "global_step": global_step,
        "best_val": float(checkpoint["best_val"]),
        "patience_best_val": float(checkpoint.get("patience_best_val", checkpoint["best_val"])),
        "stale_epochs": int(checkpoint.get("stale_epochs", 0)),
        "stats": checkpoint["stats"],
        "config": checkpoint["config"],
        "stage2_metadata": metadata,
    }

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


@contextmanager
def temporarily_zero_tactile_gates(model, enabled: bool):
    if not enabled:
        yield
        return
    raw_model = model.module if isinstance(model, DDP) else model
    gates = [
        parameter for name, parameter in raw_model.named_parameters()
        if name.endswith(".tactile_gate")
    ]
    if not gates:
        raise ValueError("tactile-disabled validation found no tactile gates")
    saved = [gate.detach().clone() for gate in gates]
    try:
        with torch.no_grad():
            for gate in gates:
                gate.zero_()
        yield
    finally:
        with torch.no_grad():
            for gate, value in zip(gates, saved):
                gate.copy_(value)


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
    stage=1,
    stage2_parameter_report=None,
    tactile_ablation=None,
):
    if tactile_ablation not in (None, "disabled", "shuffled"):
        raise ValueError(f"Unknown tactile ablation: {tactile_ablation!r}")
    if tactile_ablation is not None and (train or stage != 2):
        raise ValueError(
            "Tactile ablations are only valid for Stage2 validation"
        )
    model.train(train)
    raw_model = model.module if isinstance(model, DDP) else model
    if train and (backbone_bn_eval or stage == 2):
        set_backbone_batch_norm_eval(raw_model)
    totals = torch.zeros(3, dtype=torch.float64, device=device)
    interval = torch.zeros(3, dtype=torch.float64, device=device)
    stage2_diagnostics = {}
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
                tactile_images = None
                if stage == 2:
                    tactile_images = letterbox_tactile_images(
                        batch["tactile_images"].to(device, non_blocking=True),
                        (224, 224),
                    )
                actions = batch["action"].to(device, non_blocking=True)
                if tactile_ablation == "shuffled" and tactile_images.shape[0] > 1:
                    shift = 1 + (
                        (validation_seed + batch_index - 1)
                        % (tactile_images.shape[0] - 1)
                    )
                    tactile_images = torch.roll(tactile_images, shift, dims=0)
                is_pad = batch["is_pad"].to(device, non_blocking=True)
                task_index = (
                    batch["task_index"].to(device, non_blocking=True)
                    if use_task_condition else None
                )
                if train:
                    optimizer.zero_grad(set_to_none=True)
                    step_policy_lr = optimizer_partition_lr(optimizer, "policy")
                    step_backbone_lr = (
                        0.0 if stage == 2
                        else optimizer_partition_lr(optimizer, "backbone")
                    )
                    backbone_frozen = stage == 2 or global_step < backbone_freeze_steps
                with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                    camera_images = (
                        (images[:, 0], images[:, 1])
                        if images.shape[1] == 2
                        else (images[:, 0], images[:, 1], images[:, 2])
                    )
                    if train:
                        if stage == 2:
                            predicted, noise = model(
                                *camera_images, obs=observation, act=actions,
                                task_idx=task_index, training=True,
                                tactile_images=tactile_images,
                            )
                        else:
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
                        if stage == 2:
                            with temporarily_zero_tactile_gates(
                                model, tactile_ablation == "disabled"
                            ):
                                prediction = model(
                                    *camera_images, obs=observation,
                                    task_idx=task_index, training=False,
                                    tactile_images=tactile_images,
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
                    if stage == 2 and stage2_parameter_report is not None:
                        scaler.unscale_(optimizer)
                        collect_diagnostics = (
                            batch_index == total_batches
                            or (
                                log_every_steps > 0
                                and batch_index % log_every_steps == 0
                            )
                        )
                        if collect_diagnostics:
                            stage2_diagnostics = stage2_gradient_diagnostics(
                                raw_model, stage2_parameter_report
                            )
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
                                **stage2_diagnostics,
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
                **stage2_diagnostics,
            }
        )
    return metrics, global_step


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage", type=int, choices=(1, 2),
        default=int(os.environ.get("DECO_STAGE", "1")),
    )
    parser.add_argument(
        "--bread-phase",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--stage1-checkpoint", default=os.environ.get("STAGE1_CHECKPOINT")
    )
    parser.add_argument(
        "--tactile-encoder-checkpoint",
        default=os.environ.get("TACTILE_ENCODER_CHECKPOINT"),
    )
    parser.add_argument(
        "--tactile-encoder-cache",
        default=os.environ.get(
            "TACTILE_ENCODER_CACHE", "checkpoints/deco/tactile_encoder_cache"
        ),
    )
    parser.add_argument(
        "--tactile-adapter-rank", type=int,
        default=int(os.environ.get("TACTILE_ADAPTER_RANK", "32")),
    )
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
    parser.add_argument(
        "--resume", "--resume-from", dest="resume_from",
        default=os.environ.get("RESUME_FROM"),
    )
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
        "--augmentation-preset",
        choices=AUGMENTATION_PRESET_NAMES,
        default=None,
    )
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
    parser.add_argument(
        "--wandb-enabled",
        action=argparse.BooleanOptionalAction,
        default=env_bool("WANDB_ENABLED", False),
    )
    parser.add_argument(
        "--wandb-project", default=os.environ.get("WANDB_PROJECT", "deco-stage2")
    )
    parser.add_argument("--wandb-entity", default=os.environ.get("WANDB_ENTITY"))
    parser.add_argument("--wandb-group", default=os.environ.get("WANDB_GROUP"))
    parser.add_argument("--wandb-tags", default=os.environ.get("WANDB_TAGS", ""))
    parser.add_argument(
        "--wandb-mode",
        choices=("online", "offline", "disabled"),
        default=os.environ.get("WANDB_MODE", "online"),
    )
    parser.add_argument("--wandb-run-id", default=os.environ.get("WANDB_RUN_ID"))
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
    return parser


def validate_stage_arguments(args) -> None:
    if args.tactile_adapter_rank <= 0:
        raise ValueError("--tactile-adapter-rank must be positive")
    if args.stage1_checkpoint and args.resume_from:
        raise ValueError("--stage1-checkpoint and --resume are mutually exclusive")
    if args.stage == 1:
        return
    if args.resume_from:
        if args.resume_mode != "exact":
            raise ValueError("Stage2 supports exact --resume only")
        if args.dataset_format != "lerobot-v21":
            raise ValueError("Stage2 currently requires --dataset-format lerobot-v21")
        return
    if not args.stage1_checkpoint:
        raise ValueError("Fresh Stage2 training requires --stage1-checkpoint")
    if not args.tactile_encoder_checkpoint:
        raise ValueError(
            "Fresh Stage2 training requires --tactile-encoder-checkpoint"
        )
    if args.dataset_format != "lerobot-v21":
        raise ValueError("Stage2 currently requires --dataset-format lerobot-v21")


def main(argv=None):
    args = build_argument_parser().parse_args(argv)
    if args.bread_phase:
        if args.stage != 1 or args.dataset_format != "lerobot-v21":
            raise ValueError("Bread phase training requires Stage 1 LeRobot v2.1 data")
        args.use_task_condition = True
    resumed_stage2 = restore_stage2_resume_arguments(args)
    validate_stage_arguments(args)
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
    deco_obs_dim = len(obs_index_list)
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
        "model_type": STAGE2_MODEL_TYPE if args.stage == 2 else MODEL_TYPE,
        "source_obs_dim": int(contract["obs_dim"]),
        "obs_dim": deco_obs_dim,
        "action_dim": action_dim,
        "chunk_size": chunk_size,
        "source_chunk_size": train_dataset.source_chunk_size,
        "observation_indices": obs_index_list,
        "state_columns": contract["state_columns"],
        "action_columns": contract["action_columns"],
        "state_action_profile": contract.get("state_action_profile"),
        "controlled_arms": contract.get("controlled_arms"),
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
        "tactile_field_order": list(TACTILE_NAMES) if args.stage == 2 else None,
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
        "training_state_version": 3 if args.stage == 2 else 2,
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
        "augmentation_preset": augmentation_config.version,
        "augmentation": asdict(augmentation_config),
        "bread_phase_version": "bread-phase-v1" if args.bread_phase else None,
    }
    tactile_artifact = None
    stage1_payload = None
    stage2_parameter_report = None
    stage2_metadata = None
    if args.stage == 2:
        if args.resume_from:
            stage2_metadata = validate_stage2_resume_checkpoint(resumed_stage2)
            config = stage2_config_from_resume_checkpoint(resumed_stage2, config)
            for key in _STAGE2_CHECKPOINT_DRIVEN_CONFIG_KEYS:
                setattr(args, key, config[key])
        else:
            stage1_payload = validate_stage1_checkpoint_contract(
                args.stage1_checkpoint,
                current_config=config,
                current_stats=train_dataset.stats,
                map_location="cpu",
            )
            tactile_artifact = resolve_tactile_encoder_distributed(
                args.tactile_encoder_checkpoint,
                args.tactile_encoder_cache,
                rank=rank,
                world_size=world_size,
                device=device,
            )
        model = build_stage2_model(config).to(device)
        if tactile_artifact is not None:
            load_tactile_encoder_weights(model.tactile_encoder, tactile_artifact)
            initialization = initialize_stage2_from_stage1(
                model, stage1_payload, map_location="cpu"
            )
            stage2_parameter_report = initialization.parameters
            stage1_reference = build_model(config, load_backbone=False).to(device)
            load_stage1_reference(stage1_reference, stage1_payload)
            parity_inputs, parity_tactile = build_stage2_parity_inputs(config, device)
            parity_report = verify_stage2_stage1_parity(
                stage1_reference,
                model,
                inputs=parity_inputs,
                tactile_images=parity_tactile,
                seed=20260827,
            )
            del stage1_reference, parity_inputs, parity_tactile
            if device.type == "cuda":
                torch.cuda.empty_cache()
            stage2_metadata = build_stage2_checkpoint_metadata(
                model,
                stage2_parameter_report,
                stage1_checkpoint=args.stage1_checkpoint,
                tactile_artifact=tactile_artifact,
                tactile_adapter_rank=args.tactile_adapter_rank,
            )
            stage2_metadata["stage1_parity"] = parity_report
        else:
            stage2_parameter_report = configure_stage2_trainability(model)
        config["parameter_counts"] = {
            "total": stage2_parameter_report.total_parameters,
            "trainable": stage2_parameter_report.trainable_parameters,
        }
        config["parameter_categories"] = _parameter_categories(
            stage2_parameter_report
        )
    else:
        model = build_model(config).to(device)
    if world_size > 1:
        model = DDP(
            model,
            device_ids=[device.index] if device.type == "cuda" else None,
            find_unused_parameters=True,
        )
    raw_model = model.module if isinstance(model, DDP) else model
    if args.stage == 2:
        parameter_groups = stage2_optimizer_parameter_groups(
            raw_model,
            learning_rate=args.lr,
            weight_decay=args.weight_decay,
        )
        backbone_parameters = ()
    else:
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
    restored_stats = None
    if args.resume_from and args.stage == 2:
        restored = restore_stage2_training_state(
            resumed_stage2,
            model=raw_model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            current_config=config,
            world_size=world_size,
            rank=rank,
        )
        start_epoch = restored["epoch"]
        global_step = restored["global_step"]
        best_val = restored["best_val"]
        patience_best_val = restored["patience_best_val"]
        stale_epochs = restored["stale_epochs"]
        restored_stats = restored["stats"]
        stage2_metadata = restored["stage2_metadata"]
        resume_record = {
            "event": "resume", "mode": "exact", "source": args.resume_from,
            "source_run_id": resumed_stage2.get("run_id"),
            "source_epoch": start_epoch, "global_step": global_step,
            "best_val": best_val, "stale_epochs": stale_epochs,
            "lr": optimizer_partition_lr(optimizer, "policy"),
            "backbone_lr": 0.0,
            "scheduler_type": config["scheduler_type"],
        }
    elif args.resume_from:
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
    if restored_stats is not None:
        stats = apply_restored_dataset_stats(restored_stats, train_dataset, val_dataset)
    if main_rank:
        (output_dir / "config.json").write_text(json.dumps(config, indent=2))
        (output_dir / "dataset_stats.json").write_text(
            json.dumps(jsonable_stats(stats), indent=2)
        )
        print(json.dumps({"event": "start", **config}), flush=True)
        metrics_logger = MetricsLogger(output_dir)
        wandb_logger = None
        if args.wandb_enabled:
            wandb_logger = WandbMetricsLogger(
                project=args.wandb_project,
                entity=args.wandb_entity,
                name=args.run_id,
                run_id=args.wandb_run_id or args.run_id,
                group=args.wandb_group,
                tags=[
                    tag.strip() for tag in args.wandb_tags.split(",")
                    if tag.strip()
                ],
                mode=args.wandb_mode,
                output_dir=output_dir,
                config=config,
            )
            print(json.dumps({
                "event": "wandb_initialized", "url": wandb_logger.url,
            }), flush=True)
        if resume_record is not None:
            metrics_logger.log(resume_record)
            if wandb_logger is not None:
                wandb_logger.log(resume_record)
            print(json.dumps(resume_record), flush=True)
    else:
        metrics_logger = None
        wandb_logger = None
    started = time.monotonic()

    def log_step(record):
        if main_rank:
            record.setdefault(
                "lr", optimizer_partition_lr(optimizer, "policy")
            )
            record.setdefault(
                "backbone_lr",
                0.0 if args.stage == 2
                else optimizer_partition_lr(optimizer, "backbone"),
            )
            record.setdefault(
                "backbone_frozen",
                args.stage == 2 or record["global_step"] < backbone_freeze_steps,
            )
            record["elapsed_seconds"] = time.monotonic() - started
            metrics_logger.log(record)
            if wandb_logger is not None:
                wandb_logger.log(record)
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
            stage=args.stage,
            stage2_parameter_report=stage2_parameter_report,
        )
        sync_model_buffers(raw_model, world_size)
        validation_metrics = {}
        validation_ablation_metrics = {}
        for validation_name, validation_loader in (
            ("train", train_subset_val_loader),
            ("unseen", val_loader),
        ):
            ablation_modes = (("normal", None),)
            if args.stage == 2:
                ablation_modes += (
                    ("tactile_disabled", "disabled"),
                    ("shuffled_tactile", "shuffled"),
                )
            validation_records = {name: [] for name, _ in ablation_modes}
            for seed_index in range(args.validation_noise_seeds):
                validation_noise_seed = (
                    args.validation_seed + seed_index * 100_003
                )
                for ablation_name, tactile_ablation in ablation_modes:
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
                        validation_seed=validation_noise_seed,
                        rank=rank,
                        stage=args.stage,
                        tactile_ablation=tactile_ablation,
                    )
                    validation_records[ablation_name].append(seed_metrics)
            validation_metrics[validation_name] = average_validation_metrics(
                validation_records["normal"]
            )
            if args.stage == 2:
                validation_ablation_metrics[validation_name] = {
                    name: average_validation_metrics(records)
                    for name, records in validation_records.items()
                    if name != "normal"
                }
        val_train_metrics = validation_metrics["train"]
        val_unseen_metrics = validation_metrics["unseen"]
        ablation_record = {}
        all_validation_metrics = [val_train_metrics, val_unseen_metrics]
        if args.stage == 2:
            ablation_record = {
                "val_train_tactile_disabled": validation_ablation_metrics["train"]["tactile_disabled"],
                "val_train_shuffled_tactile": validation_ablation_metrics["train"]["shuffled_tactile"],
                "val_unseen_tactile_disabled": validation_ablation_metrics["unseen"]["tactile_disabled"],
                "val_unseen_shuffled_tactile": validation_ablation_metrics["unseen"]["shuffled_tactile"],
            }
            all_validation_metrics.extend(ablation_record.values())
        if not all(
            math.isfinite(metrics["loss"])
            for metrics in all_validation_metrics
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
            record.update(ablation_record)
            metrics_logger.log_epoch(record)
            if wandb_logger is not None:
                wandb_logger.log(record)
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
            checkpoint_stem = f"deco_stage{args.stage}"
            if args.stage == 2:
                checkpoint.update({
                    f"{name}_loss": metrics["loss"]
                    for name, metrics in ablation_record.items()
                })
            if args.stage == 2:
                stage2_metadata = {
                    **stage2_metadata,
                    "gate_values": stage2_gradient_diagnostics(
                        raw_model, stage2_parameter_report
                    )["gate_values"],
                }
                checkpoint.update({
                    "checkpoint_schema_version": STAGE2_CHECKPOINT_SCHEMA_VERSION,
                    "stage": 2,
                    "model_type": STAGE2_MODEL_TYPE,
                    "stage2_metadata": stage2_metadata,
                })
            periodic = (epoch + 1) % args.save_every == 0
            if periodic:
                atomic_torch_save(checkpoint, output_dir / f"{checkpoint_stem}_epoch_{epoch + 1}.pt")
                atomic_torch_save(checkpoint, output_dir / f"{checkpoint_stem}_latest.pt")
            if absolute_improved:
                atomic_torch_save(checkpoint, output_dir / f"{checkpoint_stem}_best.pt")
            if args.stage == 2 and (periodic or absolute_improved):
                export_events = export_stage2_torchscript_artifacts(
                    policy=raw_model,
                    stats=jsonable_stats(stats),
                    config=config,
                    stage2_metadata=stage2_metadata,
                    output_dir=output_dir,
                    epoch=epoch + 1,
                    val_loss=val_unseen_metrics["loss"],
                    image_height=args.torchscript_image_height,
                    image_width=args.torchscript_image_width,
                    periodic=periodic,
                    improved=absolute_improved,
                )
                for export_event in export_events:
                    metrics_logger.log(export_event)
                    print(json.dumps(export_event), flush=True)
            if args.stage == 1 and (periodic or absolute_improved):
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
            if args.stage == 1 and periodic:
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
        memory_record = {
            "event": "training_peak_memory",
            "peak_alloc_gb": round(peak_gb, 3),
            "peak_reserved_gb": round(peak_reserved_gb, 3),
            "global_step": global_step,
        }
        if wandb_logger is not None:
            wandb_logger.log(memory_record)
        print(json.dumps(memory_record), flush=True)
    if wandb_logger is not None:
        wandb_logger.finish()
    cleanup_dist()


if __name__ == "__main__":
    main()
