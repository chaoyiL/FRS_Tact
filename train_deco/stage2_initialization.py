"""Strict Stage1-to-Stage2 loading and parameter-boundary audits."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn


_KNOWN_WRAPPER_PREFIXES = ("module.", "_orig_mod.")
_TRAINABLE_CATEGORIES = (
    "sensor_embeddings",
    "tactile_attention",
    "tactile_gates",
    "pi_adapters",
)


@dataclass(frozen=True)
class Stage2ParameterReport:
    trainable_by_category: dict[str, tuple[str, ...]]
    frozen_by_category: dict[str, tuple[str, ...]]
    total_parameters: int
    trainable_parameters: int


@dataclass(frozen=True)
class Stage2InitializationReport:
    loaded_stage1_keys: tuple[str, ...]
    stage2_only_keys: tuple[str, ...]
    parameters: Stage2ParameterReport


def _stage2_category(name: str) -> str | None:
    if name.startswith("tactile_encoder."):
        return "tactile_encoder"
    if name == "sensor_embeddings.weight":
        return "sensor_embeddings"
    if not name.startswith("mmattn."):
        return None
    if ".tactile_key." in name or ".tactile_value." in name:
        return "tactile_attention"
    if name.endswith(".tactile_gate"):
        return "tactile_gates"
    if "_pi." in name:
        return "pi_adapters"
    return None


def _strip_known_wrapper_prefixes(
    state: Mapping[str, Any],
) -> dict[str, Any]:
    normalized = dict(state)
    changed = True
    while normalized and changed:
        changed = False
        for prefix in _KNOWN_WRAPPER_PREFIXES:
            if all(name.startswith(prefix) for name in normalized):
                stripped = {name[len(prefix):]: value for name, value in normalized.items()}
                if len(stripped) != len(normalized):
                    raise ValueError(
                        f"Stage1 checkpoint keys collide after stripping known prefix {prefix!r}"
                    )
                normalized = stripped
                changed = True
                break
    return normalized


def _checkpoint_payload(
    checkpoint: str | Path | Mapping[str, Any],
    map_location: str | torch.device,
) -> Mapping[str, Any]:
    if isinstance(checkpoint, (str, Path)):
        payload = torch.load(
            Path(checkpoint),
            map_location=map_location,
            weights_only=True,
        )
    else:
        payload = checkpoint
    if not isinstance(payload, Mapping) or "model" not in payload:
        raise ValueError(
            "Stage1 initialization requires a training checkpoint containing a 'model' state dict"
        )
    model_state = payload["model"]
    if not isinstance(model_state, Mapping):
        raise ValueError("Stage1 training checkpoint 'model' must be a state dict mapping")
    if not all(isinstance(name, str) for name in model_state):
        raise ValueError("Stage1 training checkpoint model keys must be strings")
    return model_state


def _stage2_state_partition(
    model: nn.Module,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    if not getattr(model, "tactile_image_mode", False):
        raise ValueError("Stage1 initialization target must be a tactile-image Stage2 model")
    stage1_keys: list[str] = []
    stage2_keys: list[str] = []
    for name in model.state_dict():
        if _stage2_category(name) is None:
            stage1_keys.append(name)
        else:
            stage2_keys.append(name)
    if not stage2_keys:
        raise ValueError("Stage2 model exposes no tactile or PI-adapter state")
    return tuple(stage1_keys), tuple(stage2_keys)


def configure_stage2_trainability(model: nn.Module) -> Stage2ParameterReport:
    """Apply and report the exact Stage2 trainable-parameter allowlist."""
    if not getattr(model, "tactile_image_mode", False):
        raise ValueError("trainability configuration requires a tactile-image Stage2 model")
    trainable: dict[str, list[str]] = {
        category: [] for category in _TRAINABLE_CATEGORIES
    }
    frozen: dict[str, list[str]] = {
        "stage1": [],
        "tactile_encoder": [],
    }
    total_parameters = 0
    trainable_parameters = 0
    for name, parameter in model.named_parameters():
        category = _stage2_category(name)
        should_train = category in _TRAINABLE_CATEGORIES
        parameter.requires_grad_(should_train)
        total_parameters += parameter.numel()
        if should_train:
            trainable[category].append(name)
            trainable_parameters += parameter.numel()
        elif category == "tactile_encoder":
            frozen["tactile_encoder"].append(name)
        else:
            frozen["stage1"].append(name)
    empty = [category for category, names in trainable.items() if not names]
    if empty:
        raise ValueError(
            "Stage2 model is missing required trainable parameter categories: "
            + ", ".join(empty)
        )
    tactile_encoder = getattr(model, "tactile_encoder", None)
    if tactile_encoder is None:
        raise ValueError("Stage2 model is missing its shared tactile encoder")
    tactile_encoder.eval()
    return Stage2ParameterReport(
        trainable_by_category={
            category: tuple(names) for category, names in trainable.items()
        },
        frozen_by_category={
            category: tuple(names) for category, names in frozen.items()
        },
        total_parameters=total_parameters,
        trainable_parameters=trainable_parameters,
    )


def initialize_stage2_from_stage1(
    model: nn.Module,
    checkpoint: str | Path | Mapping[str, Any],
    *,
    map_location: str | torch.device = "cpu",
) -> Stage2InitializationReport:
    """Strictly load a wrapped Stage1 state dict into the Stage2 superset."""
    expected_stage1_keys, stage2_only_keys = _stage2_state_partition(model)
    source = _strip_known_wrapper_prefixes(
        _checkpoint_payload(checkpoint, map_location)
    )
    expected = set(expected_stage1_keys)
    received = set(source)
    missing = sorted(expected - received)
    unexpected = sorted(received - expected)
    if missing or unexpected:
        details: list[str] = []
        if missing:
            details.append(f"missing Stage1 keys: {missing}")
        if unexpected:
            details.append(f"unexpected Stage1 keys: {unexpected}")
        raise ValueError("; ".join(details))

    target_state = model.state_dict()
    for name in expected_stage1_keys:
        value = source[name]
        if not isinstance(value, torch.Tensor):
            raise ValueError(f"Stage1 checkpoint tensor {name!r} is not a torch.Tensor")
        expected_shape = tuple(target_state[name].shape)
        if tuple(value.shape) != expected_shape:
            raise ValueError(
                f"Stage1 shape mismatch for {name!r}: expected "
                f"{expected_shape}, got {tuple(value.shape)}"
            )

    complete_state = dict(target_state)
    complete_state.update(source)
    model.load_state_dict(complete_state, strict=True)
    parameters = configure_stage2_trainability(model)
    return Stage2InitializationReport(
        loaded_stage1_keys=expected_stage1_keys,
        stage2_only_keys=stage2_only_keys,
        parameters=parameters,
    )
