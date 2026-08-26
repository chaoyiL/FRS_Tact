"""Small, testable utilities shared by the distributed training loop."""

from __future__ import annotations

import math
import random
from collections.abc import Iterator, Sized
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import Sampler


class DistributedEvalSampler(Sampler[int]):
    """Partition evaluation data across ranks without padding or repetition."""

    def __init__(self, dataset: Sized, num_replicas: int, rank: int):
        if num_replicas <= 0:
            raise ValueError("num_replicas must be positive")
        if rank < 0 or rank >= num_replicas:
            raise ValueError(f"rank must be in [0, {num_replicas}), got {rank}")
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank

        quotient, remainder = divmod(len(dataset), num_replicas)
        self._size = quotient + int(rank < remainder)
        self._start = rank * quotient + min(rank, remainder)

    def __iter__(self) -> Iterator[int]:
        return iter(range(self._start, self._start + self._size))

    def __len__(self) -> int:
        return self._size


def deterministic_subset_indices(
    dataset_size: int, subset_size: int, seed: int
) -> list[int]:
    """Choose a fixed, sorted subset without changing global RNG state."""

    if dataset_size <= 0:
        raise ValueError("dataset_size must be positive")
    if subset_size <= 0:
        raise ValueError("subset_size must be positive")
    if subset_size >= dataset_size:
        return list(range(dataset_size))
    generator = np.random.default_rng(seed)
    return np.sort(
        generator.choice(dataset_size, size=subset_size, replace=False)
    ).tolist()


def masked_error_sums(
    prediction: torch.Tensor,
    target: torch.Tensor,
    is_pad: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return SSE, SAE, and scalar element count before any batch averaging."""

    if prediction.shape != target.shape:
        raise ValueError(
            f"prediction and target shapes differ: {prediction.shape} != {target.shape}"
        )
    if prediction.ndim < 2:
        raise ValueError("prediction and target must include an action dimension")

    error = prediction.float() - target.float()
    if is_pad is None:
        squared_sum = error.square().sum()
        absolute_sum = error.abs().sum()
        element_count = torch.tensor(error.numel(), device=error.device)
        return squared_sum, absolute_sum, element_count

    expected_pad_shape = prediction.shape[:-1]
    if tuple(is_pad.shape) != tuple(expected_pad_shape):
        raise ValueError(
            f"is_pad shape must be {expected_pad_shape}, got {tuple(is_pad.shape)}"
        )
    valid = (~is_pad.bool()).unsqueeze(-1)
    squared_sum = (error.square() * valid).sum()
    absolute_sum = (error.abs() * valid).sum()
    element_count = valid.sum() * prediction.shape[-1]
    return squared_sum, absolute_sum, element_count


def metric_totals(
    squared_sum: torch.Tensor,
    absolute_sum: torch.Tensor,
    element_count: torch.Tensor,
) -> torch.Tensor:
    """Detach per-batch sums into an FP64 accumulator payload."""

    return torch.stack(
        (
            squared_sum.detach().double(),
            absolute_sum.detach().double(),
            element_count.detach().double(),
        )
    )


def warmup_cosine_multiplier(
    step: int,
    warmup_steps: int,
    cosine_steps: int,
    final_ratio: float,
) -> float:
    """Closed-form warmup followed by cosine decay, clamped at ``final_ratio``."""

    if step < 0:
        raise ValueError("step must be non-negative")
    if warmup_steps < 0:
        raise ValueError("warmup_steps must be non-negative")
    if cosine_steps <= 0:
        raise ValueError("cosine_steps must be positive")
    if not 0.0 <= final_ratio <= 1.0:
        raise ValueError("final_ratio must be between zero and one")
    if warmup_steps and step < warmup_steps:
        return (step + 1) / warmup_steps

    progress = min(max((step - warmup_steps) / cosine_steps, 0.0), 1.0)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return final_ratio + (1.0 - final_ratio) * cosine


def backbone_cosine_multiplier(
    step: int,
    freeze_steps: int,
    cosine_steps: int,
    final_ratio: float,
) -> float:
    """Keep a backbone frozen, then cosine-decay it from its base LR."""

    if freeze_steps < 0:
        raise ValueError("freeze_steps must be non-negative")
    if step < freeze_steps:
        return 0.0
    return warmup_cosine_multiplier(
        step - freeze_steps,
        warmup_steps=0,
        cosine_steps=cosine_steps,
        final_ratio=final_ratio,
    )


def optimizer_parameter_groups(
    model: nn.Module,
    policy_lr: float,
    backbone_lr: float,
    weight_decay: float,
) -> tuple[list[dict], list[nn.Parameter]]:
    """Create stable policy/backbone and decay/no-decay AdamW groups."""

    if not hasattr(model, "img_encoder"):
        raise ValueError("DECO model is missing img_encoder")
    backbone_parameters = list(model.img_encoder.parameters())
    backbone_ids = {id(parameter) for parameter in backbone_parameters}
    buckets: dict[str, list[nn.Parameter]] = {
        "policy_decay": [],
        "policy_no_decay": [],
        "backbone_decay": [],
        "backbone_no_decay": [],
    }

    seen: set[int] = set()
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        parameter_id = id(parameter)
        if parameter_id in seen:
            raise ValueError(f"parameter appears more than once: {name}")
        seen.add(parameter_id)
        partition = "backbone" if parameter_id in backbone_ids else "policy"
        decay = "no_decay" if parameter.ndim <= 1 or name.endswith(".bias") else "decay"
        buckets[f"{partition}_{decay}"].append(parameter)

    expected = {id(parameter) for parameter in model.parameters() if parameter.requires_grad}
    if seen != expected:
        raise ValueError("optimizer groups do not cover every trainable parameter exactly once")
    if not backbone_parameters or not any(id(parameter) in seen for parameter in backbone_parameters):
        raise ValueError("backbone optimizer partition is empty")
    if not buckets["policy_decay"] and not buckets["policy_no_decay"]:
        raise ValueError("policy optimizer partition is empty")

    groups = []
    for name in (
        "policy_decay",
        "policy_no_decay",
        "backbone_decay",
        "backbone_no_decay",
    ):
        partition_lr = backbone_lr if name.startswith("backbone") else policy_lr
        group_weight_decay = 0.0 if name.endswith("no_decay") else weight_decay
        groups.append(
            {
                "group_name": name,
                "params": buckets[name],
                "lr": partition_lr,
                "weight_decay": group_weight_decay,
            }
        )
    return groups, backbone_parameters


def _is_stage2_trainable_name(name: str) -> bool:
    return (
        name == "sensor_embeddings.weight"
        or (
            name.startswith("mmattn.")
            and (
                ".tactile_key." in name
                or ".tactile_value." in name
                or name.endswith(".tactile_gate")
                or "_pi." in name
            )
        )
    )


def stage2_optimizer_parameter_groups(
    model: nn.Module,
    learning_rate: float,
    weight_decay: float,
) -> list[dict[str, Any]]:
    """Build AdamW groups from the exact Stage2 trainable allowlist only."""

    if learning_rate <= 0:
        raise ValueError("Stage2 learning rate must be positive")
    if weight_decay < 0:
        raise ValueError("Stage2 weight decay must be non-negative")
    trainable = [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    ]
    if not trainable:
        raise ValueError("Stage2 trainable parameter set is empty")
    forbidden = [
        name for name, _ in trainable if not _is_stage2_trainable_name(name)
    ]
    if forbidden:
        raise ValueError(
            "Stage2 optimizer received Stage1/tactile-encoder parameters: "
            f"{forbidden}"
        )

    buckets: dict[str, list[nn.Parameter]] = {
        "policy_decay": [],
        "policy_no_decay": [],
    }
    for name, parameter in trainable:
        bucket = (
            "policy_no_decay"
            if parameter.ndim <= 1 or name.endswith(".bias")
            else "policy_decay"
        )
        buckets[bucket].append(parameter)
    groups = []
    for name, parameters in buckets.items():
        if not parameters:
            continue
        groups.append(
            {
                "group_name": name,
                "params": parameters,
                "lr": learning_rate,
                "weight_decay": 0.0 if name.endswith("no_decay") else weight_decay,
            }
        )
    optimized = {id(parameter) for group in groups for parameter in group["params"]}
    expected = {id(parameter) for _, parameter in trainable}
    if optimized != expected:
        raise ValueError(
            "Stage2 optimizer groups do not cover every trainable parameter exactly once"
        )
    return groups


def stage2_gradient_diagnostics(model: nn.Module, parameter_report) -> dict[str, Any]:
    """Return cheap gate and gradient-norm diagnostics for the Stage2 boundary."""

    named_parameters = dict(model.named_parameters())

    def category_norm(categories: dict[str, tuple[str, ...]]) -> float:
        squared = 0.0
        for names in categories.values():
            for name in names:
                parameter = named_parameters.get(name)
                if parameter is not None and parameter.grad is not None:
                    squared += float(parameter.grad.detach().float().square().sum())
        return math.sqrt(squared)

    gates = {
        name: float(parameter.detach().float().item())
        for name, parameter in named_parameters.items()
        if name.endswith(".tactile_gate")
    }
    return {
        "gate_values": gates,
        "gradient_norms": {
            "trainable": category_norm(parameter_report.trainable_by_category),
            "frozen": category_norm(parameter_report.frozen_by_category),
        },
    }


def optimizer_partition_lr(optimizer: torch.optim.Optimizer, partition: str) -> float:
    """Return the common LR for one logical optimizer partition."""

    values = {
        float(group["lr"])
        for group in optimizer.param_groups
        if str(group.get("group_name", "")).startswith(partition)
    }
    if len(values) != 1:
        raise ValueError(f"optimizer partition {partition!r} has inconsistent LRs: {values}")
    return values.pop()


def override_optimizer_partition_lrs(
    optimizer: torch.optim.Optimizer,
    policy_lr: float,
    backbone_lr: float,
) -> None:
    """Override loaded optimizer LRs without discarding moment estimates."""

    if policy_lr <= 0 or backbone_lr <= 0:
        raise ValueError("fine-tune learning rates must be positive")
    counts = {"policy": 0, "backbone": 0}
    for group in optimizer.param_groups:
        group_name = str(group.get("group_name", ""))
        if group_name.startswith("policy"):
            partition = "policy"
            learning_rate = policy_lr
        elif group_name.startswith("backbone"):
            partition = "backbone"
            learning_rate = backbone_lr
        else:
            raise ValueError(
                f"Cannot override LR for unknown optimizer group: {group_name!r}"
            )
        group["lr"] = learning_rate
        # LambdaLR requires ``initial_lr`` when it starts at a nonzero step.
        # Optimizer state loading restores the old scheduler's base LR here,
        # so it must be replaced along with the live LR.
        group["initial_lr"] = learning_rate
        counts[partition] += 1
    missing = [partition for partition, count in counts.items() if count == 0]
    if missing:
        raise ValueError(f"Optimizer is missing partitions: {missing}")


def constant_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    global_step: int = 0,
) -> torch.optim.lr_scheduler.LambdaLR:
    """Build a constant per-step scheduler aligned with a global step."""

    if global_step < 0:
        raise ValueError("global_step must be non-negative")
    if global_step > 0:
        missing = [
            str(group.get("group_name", ""))
            for group in optimizer.param_groups
            if "initial_lr" not in group
        ]
        if missing:
            raise ValueError(
                "Nonzero scheduler start requires initial_lr for groups: "
                f"{missing}"
            )
    return torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=[lambda _step: 1.0 for _ in optimizer.param_groups],
        last_epoch=global_step - 1,
    )


def seed_training_rng(seed: int, rank: int) -> None:
    """Seed stochastic training operations independently on every DDP rank."""

    rank_seed = seed + rank
    random.seed(rank_seed)
    np.random.seed(rank_seed % (2**32))
    torch.manual_seed(rank_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(rank_seed)


def set_backbone_batch_norm_eval(model: nn.Module) -> None:
    """Keep pretrained backbone BatchNorm statistics fixed for small local batches."""

    backbone_names = ["img_encoder"]
    if getattr(model, "tactile_image_mode", False) or hasattr(model, "tactile_encoder"):
        backbone_names.append("tactile_encoder")
    for backbone_name in backbone_names:
        backbone = getattr(model, backbone_name, None)
        if backbone is None:
            continue
        for module in backbone.modules():
            if isinstance(module, nn.modules.batchnorm._BatchNorm):
                module.eval()
