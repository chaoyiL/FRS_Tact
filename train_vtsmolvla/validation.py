from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from train_smolvla.validation import (
    CheckpointContract as VisualCheckpointContract,
    CheckpointValidationExtension,
    CheckpointValidationReport,
    validate_checkpoint as validate_visual_checkpoint,
)

_TACTILE_ENCODER_PARAMS_PREFIX = "model.tactile_encoder.params/"


@dataclass(frozen=True)
class CheckpointContract(VisualCheckpointContract):
    tactile_keys: tuple[str, ...] = ()
    tactile_embedding_dim: int = 512
    tactile_num_tokens: int = 0
    tactile_proj_mode: str | None = None

    def __post_init__(self) -> None:
        mode = self.tactile_proj_mode
        if mode is None or str(mode).lower() == "auto":
            mode = "full" if self.tactile_keys or self.tactile_num_tokens else "frozen"
        else:
            mode = str(mode).lower()
        object.__setattr__(self, "tactile_proj_mode", mode)


def contract_from_config(config: Any) -> CheckpointContract:
    use_tactile_encoder = bool(config.use_tactile_encoder)
    raw_module_modes = getattr(config, "module_modes", None)
    module_modes = raw_module_modes if isinstance(raw_module_modes, Mapping) else {}
    return CheckpointContract(
        state_dim=int(config.state_dim),
        action_dim=int(config.action_dim),
        chunk_size=int(config.chunk_size),
        image_keys=tuple(config.image_keys),
        lora_rank=int(config.lora_rank),
        vlm_lora_target_modules=tuple(config.vlm_lora_target_modules),
        tactile_keys=tuple(config.tactile_keys) if use_tactile_encoder else (),
        tactile_embedding_dim=int(config.tactile_embedding_dim),
        tactile_num_tokens=int(config.tactile_num_tokens) if use_tactile_encoder else 0,
        tactile_proj_mode=str(
            module_modes.get("tactile_proj", "full" if use_tactile_encoder else "frozen")
        ).lower(),
    )


@dataclass(frozen=True)
class _VTConfigView:
    use_tactile_encoder: bool
    tactile_keys: tuple[str, ...]
    tactile_embedding_dim: int | None
    tactile_num_tokens: int | None
    tactile_proj_mode: str


def _integer(
    value: Any,
    field: str,
    issues: list[str],
    *,
    default: int | None = None,
) -> int | None:
    if value is None:
        return default
    if isinstance(value, bool):
        issues.append(f"config {field} must be an integer, got {value!r}")
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        issues.append(f"config {field} must be an integer, got {value!r}")
        return None


def _string_tuple(value: Any, field: str, issues: list[str]) -> tuple[str, ...]:
    if value is None:
        return ()
    if not isinstance(value, list | tuple) or any(not isinstance(item, str) for item in value):
        issues.append(f"config {field} must be a list of strings, got {value!r}")
        return ()
    return tuple(value)


class _VTValidationExtension(CheckpointValidationExtension):
    module_names = frozenset({"tactile_proj"})

    def parse_config(self, raw: Mapping[str, Any], issues: list[str]) -> _VTConfigView:
        use_tactile = bool(raw.get("use_tactile_encoder", False))
        tactile_num_tokens = _integer(
            raw.get("tactile_num_tokens"),
            "tactile_num_tokens",
            issues,
            default=0,
        )
        if not use_tactile:
            tactile_num_tokens = 0
        module_modes = raw.get("module_modes")
        tactile_proj_mode = (
            str(module_modes.get("tactile_proj", "full" if use_tactile else "frozen")).lower()
            if isinstance(module_modes, Mapping)
            else ("full" if use_tactile else "frozen")
        )
        return _VTConfigView(
            use_tactile_encoder=use_tactile,
            tactile_keys=_string_tuple(raw.get("tactile_keys"), "tactile_keys", issues),
            tactile_embedding_dim=_integer(
                raw.get("tactile_embedding_dim"),
                "tactile_embedding_dim",
                issues,
                default=512,
            ),
            tactile_num_tokens=tactile_num_tokens,
            tactile_proj_mode=tactile_proj_mode,
        )

    def build_contract(
        self,
        base: VisualCheckpointContract,
        config: _VTConfigView,
    ) -> CheckpointContract | None:
        if config.tactile_embedding_dim is None or config.tactile_num_tokens is None:
            return None
        return CheckpointContract(
            state_dim=base.state_dim,
            action_dim=base.action_dim,
            chunk_size=base.chunk_size,
            image_keys=base.image_keys,
            lora_rank=base.lora_rank,
            vlm_lora_target_modules=base.vlm_lora_target_modules,
            tactile_keys=config.tactile_keys,
            tactile_embedding_dim=int(config.tactile_embedding_dim),
            tactile_num_tokens=int(config.tactile_num_tokens),
            tactile_proj_mode=config.tactile_proj_mode,
        )

    def compare_contract(
        self,
        config: _VTConfigView,
        expected: VisualCheckpointContract,
        issues: list[str],
    ) -> None:
        actual_fields = {
            "tactile_keys": config.tactile_keys,
            "tactile_embedding_dim": config.tactile_embedding_dim,
            "tactile_num_tokens": config.tactile_num_tokens,
            "tactile_proj_mode": config.tactile_proj_mode,
        }
        for field, actual in actual_fields.items():
            wanted = getattr(expected, field, () if field == "tactile_keys" else 0)
            if actual is not None and actual != wanted:
                issues.append(f"config {field} expected {wanted!r}, got {actual!r}")
        expected_tactile = bool(
            getattr(expected, "tactile_keys", ())
            or getattr(expected, "tactile_num_tokens", 0)
        )
        if config.use_tactile_encoder != expected_tactile:
            issues.append(
                "config use_tactile_encoder expected "
                f"{expected_tactile!r}, got {config.use_tactile_encoder!r}"
            )

    def validate_config(
        self,
        raw: Mapping[str, Any],
        config: _VTConfigView,
        image_keys: tuple[str, ...],
        issues: list[str],
    ) -> None:
        if len(config.tactile_keys) != len(set(config.tactile_keys)):
            issues.append(f"config tactile_keys contain duplicates: {config.tactile_keys!r}")
        overlap = tuple(key for key in image_keys if key in set(config.tactile_keys))
        if overlap:
            issues.append(f"config RGB and tactile keys must be disjoint, found {overlap!r}")
        if config.use_tactile_encoder:
            if not config.tactile_keys:
                issues.append("config enables the tactile encoder but tactile_keys is empty")
            if config.tactile_num_tokens != len(config.tactile_keys):
                issues.append(
                    "config tactile_num_tokens must equal the number of tactile_keys, "
                    f"got {config.tactile_num_tokens!r} and {len(config.tactile_keys)}"
                )
            if config.tactile_embedding_dim is not None and config.tactile_embedding_dim <= 0:
                issues.append(
                    "config tactile_embedding_dim must be positive, got "
                    f"{config.tactile_embedding_dim}"
                )
        elif config.tactile_keys:
            issues.append("config has tactile_keys but use_tactile_encoder is false")
        if config.tactile_proj_mode == "lora":
            rank = raw.get("lora_rank")
            if isinstance(rank, bool) or not isinstance(rank, int) or rank <= 0:
                issues.append("config lora_rank must be positive when tactile_proj uses LoRA")

    def needs_model_structure(self, contract: VisualCheckpointContract) -> bool:
        return bool(
            getattr(contract, "tactile_keys", ())
            or getattr(contract, "tactile_num_tokens", 0)
        )

    def validate_model(
        self,
        tensors: Any,
        keys: set[str],
        *,
        path: Path,
        contract: VisualCheckpointContract,
        hidden_size: int,
        issues: list[str],
    ) -> None:
        if not self.needs_model_structure(contract):
            return
        if not any(key.startswith(_TACTILE_ENCODER_PARAMS_PREFIX) for key in keys):
            issues.append(
                f"{path.name}: missing tactile encoder tensors under "
                f"{_TACTILE_ENCODER_PARAMS_PREFIX!r}"
            )
        projection_weight = "model.tactile_proj.weight"
        projection_bias = "model.tactile_proj.bias"
        for key in (projection_weight, projection_bias):
            if key not in keys:
                issues.append(f"{path.name}: missing tensor {key!r}")
        embedding_dim = int(getattr(contract, "tactile_embedding_dim", 0))
        if projection_weight in keys:
            weight_shape = list(tensors.get_slice(projection_weight).get_shape())
            expected_shape = [hidden_size, embedding_dim]
            if weight_shape != expected_shape:
                issues.append(
                    f"{path.name}: tensor {projection_weight!r} expected shape "
                    f"{expected_shape}, got {weight_shape}"
                )
        if projection_bias in keys:
            bias_shape = list(tensors.get_slice(projection_bias).get_shape())
            if bias_shape != [hidden_size]:
                issues.append(
                    f"{path.name}: tensor {projection_bias!r} expected shape "
                    f"[{hidden_size}], got {bias_shape}"
                )
        if getattr(contract, "tactile_proj_mode", "full") == "lora":
            rank = int(contract.lora_rank)
            adapter_keys = {
                "lora_a": "model.tactile_proj.lora_a",
                "lora_b": "model.tactile_proj.lora_b",
                "lora_scale": "model.tactile_proj.lora_scale",
            }
            missing = [name for name, key in adapter_keys.items() if key not in keys]
            if missing:
                issues.append(f"{path.name}: missing tactile_proj LoRA tensors: {missing}")
            expected_shapes = {
                "lora_a": [rank, embedding_dim],
                "lora_b": [hidden_size, rank],
                "lora_scale": [],
            }
            for name, expected_shape in expected_shapes.items():
                key = adapter_keys[name]
                if key not in keys:
                    continue
                actual_shape = list(tensors.get_slice(key).get_shape())
                if actual_shape != expected_shape:
                    issues.append(
                        f"{path.name}: tactile_proj {name} expected shape "
                        f"{expected_shape}, got {actual_shape}"
                    )


_VT_EXTENSION = _VTValidationExtension()


def validate_checkpoint(
    path: str | Path,
    *,
    expected: CheckpointContract | None = None,
    base_sidecars: str | Path | None = None,
    require_weight: bool = True,
) -> CheckpointValidationReport:
    return validate_visual_checkpoint(
        path,
        expected=expected,
        base_sidecars=base_sidecars,
        require_weight=require_weight,
        extension=_VT_EXTENSION,
    )


__all__ = [
    "CheckpointContract",
    "CheckpointValidationReport",
    "contract_from_config",
    "validate_checkpoint",
]
