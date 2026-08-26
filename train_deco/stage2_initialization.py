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
_STAGE1_MODEL_TYPE = "upstream-deco-stage1"
_STAGE1_CONTRACT_KEYS = (
    "action_dim", "obs_dim", "source_obs_dim", "chunk_size", "camera_names",
    "hidden_dim", "layers", "heads", "image_size", "inference_steps",
    "rope_height", "rope_width", "use_task_condition", "num_tasks",
    "task_ids", "action_mode", "objective_version", "dataset_id", "observation_indices",
    "state_columns", "action_columns",
)
_NORMALIZATION_STAT_KEYS = (
    "observation_mean", "observation_std", "action_mean", "action_std",
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

def _training_checkpoint_payload(
    checkpoint: str | Path | Mapping[str, Any],
    map_location: str | torch.device,
) -> Mapping[str, Any]:
    if isinstance(checkpoint, (str, Path)):
        payload = torch.load(
            Path(checkpoint), map_location=map_location, weights_only=True
        )
    else:
        payload = checkpoint
    if not isinstance(payload, Mapping) or "model" not in payload:
        raise ValueError(
            "Stage1 initialization requires a training checkpoint containing a 'model' state dict"
        )
    return payload


def validate_stage1_checkpoint_contract(
    checkpoint: str | Path | Mapping[str, Any],
    *,
    current_config: Mapping[str, Any],
    current_stats: Mapping[str, Any],
    map_location: str | torch.device = "cpu",
) -> Mapping[str, Any]:
    """Reject a fresh Stage1 wrapper that differs from the Stage2 contract."""
    payload = _training_checkpoint_payload(checkpoint, map_location)
    saved_config = payload.get("config")
    if not isinstance(saved_config, Mapping):
        raise ValueError("Stage1 checkpoint config must be a mapping")
    if saved_config.get("model_type") != _STAGE1_MODEL_TYPE:
        raise ValueError("Stage1 checkpoint model_type is not upstream-deco-stage1")
    mismatches = [
        key for key in _STAGE1_CONTRACT_KEYS
        if key not in saved_config or saved_config.get(key) != current_config.get(key)
    ]
    if mismatches:
        details = {
            key: {"stage1": saved_config.get(key), "stage2": current_config.get(key)}
            for key in mismatches
        }
        raise ValueError(f"Stage1 checkpoint contract mismatch: {details}")
    saved_stats = payload.get("stats")
    if not isinstance(saved_stats, Mapping):
        raise ValueError("Stage1 checkpoint normalization stats must be a mapping")
    for key in _NORMALIZATION_STAT_KEYS:
        if key not in saved_stats or key not in current_stats:
            raise ValueError(f"Stage1 checkpoint normalization stat {key} is missing")
        saved = torch.as_tensor(saved_stats[key], dtype=torch.float64)
        current = torch.as_tensor(current_stats[key], dtype=torch.float64)
        if saved.shape != current.shape or not torch.equal(saved, current):
            raise ValueError(f"Stage1 checkpoint normalization stat {key} mismatch")
    return payload


def load_stage1_reference(
    model: nn.Module,
    checkpoint: str | Path | Mapping[str, Any],
    *,
    map_location: str | torch.device = "cpu",
) -> None:
    """Strictly load an independent Stage1 reference from its wrapper."""
    source = _strip_known_wrapper_prefixes(_checkpoint_payload(checkpoint, map_location))
    model.load_state_dict(source, strict=True)


def verify_stage2_stage1_parity(
    stage1: nn.Module,
    stage2: nn.Module,
    *,
    inputs: Mapping[str, torch.Tensor],
    tactile_images: torch.Tensor,
    seed: int,
    rtol: float = 1e-6,
    atol: float = 1e-7,
) -> dict[str, float | int]:
    """Abort unless fresh zero-gate/zero-adapter Stage2 matches Stage1."""
    named = dict(stage2.named_parameters())
    nonzero_gates = [
        name for name, value in named.items()
        if name.endswith(".tactile_gate")
        and int(torch.count_nonzero(value.detach()).item()) != 0
    ]
    if nonzero_gates:
        raise ValueError(f"Fresh Stage2 requires zero tactile gates: {nonzero_gates}")
    nonzero_adapters = [
        name for name, value in named.items()
        if "_pi.up." in name
        and int(torch.count_nonzero(value.detach()).item()) != 0
    ]
    if nonzero_adapters:
        raise ValueError(
            "Fresh Stage2 requires every zero-initialized PI adapter output to be zero: "
            f"{nonzero_adapters}"
        )

    stage1_was_training = stage1.training
    stage2_was_training = stage2.training
    stage1.eval()
    stage2.eval()
    input_device = next(iter(inputs.values())).device
    devices = [input_device.index or 0] if input_device.type == "cuda" else []
    try:
        with torch.no_grad(), torch.random.fork_rng(devices=devices):
            torch.manual_seed(seed)
            if input_device.type == "cuda":
                torch.cuda.manual_seed(seed)
            stage1_prediction, stage1_noise = stage1(**inputs, training=True)
            torch.manual_seed(seed)
            if input_device.type == "cuda":
                torch.cuda.manual_seed(seed)
            stage2_prediction, stage2_noise = stage2(
                **inputs, tactile_images=tactile_images, training=True
            )
    finally:
        stage1.train(stage1_was_training)
        stage2.train(stage2_was_training)
    if not torch.equal(stage2_noise, stage1_noise):
        raise ValueError("Fresh Stage2 parity failed: fixed RNG produced different noise")
    max_abs = float(
        (stage2_prediction - stage1_prediction).detach().abs().max().item()
    )
    if not torch.allclose(
        stage2_prediction, stage1_prediction, rtol=rtol, atol=atol
    ):
        raise ValueError(
            f"Fresh Stage2 deterministic parity failed: max_abs={max_abs}"
        )
    return {"seed": int(seed), "max_abs_prediction": max_abs}



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
