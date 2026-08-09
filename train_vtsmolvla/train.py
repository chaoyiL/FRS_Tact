#!/usr/bin/env python
"""Fine-tune the vision-tactile JAX SmolVLA from a YAML configuration."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import yaml

from train_smolvla.train import (
    ALLOWED_TOP_LEVEL_KEYS,
    TrainingComponents,
    parse_args,
    run_training,
)
from train_vtsmolvla.checkpoint import (
    count_expert_layers,
    count_vlm_layers,
    extend_vlm_layers,
    initialize_tactile_fusion_params,
    load_params,
    resolve_checkpoint,
)
from train_vtsmolvla.configuration import VTSmolVLAConfig
from train_vtsmolvla.data import VTLeRobotJaxDataLoader
from train_vtsmolvla.lora import resolve_module_modes
from train_vtsmolvla.modeling import VTJaxSmolVLA
from train_vtsmolvla.training import VTJaxSmolVLATrainer
from train_vtsmolvla.validation import contract_from_config, validate_checkpoint

DEFAULT_CONFIG = Path(__file__).resolve().parent / "configs" / "train.yaml"
VT_ALLOWED_TOP_LEVEL_KEYS = ALLOWED_TOP_LEVEL_KEYS | {"tactile_embedding_cache"}


def _validate_vt_config(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"配置文件不存在：{path}")
    with path.open(encoding="utf-8") as file:
        cfg = yaml.safe_load(file) or {}
    if not isinstance(cfg, dict):
        raise ValueError(f"配置根节点必须是 mapping：{path}")
    model = cfg.get("model")
    if not isinstance(model, dict):
        raise ValueError(f"{path} 缺少 model 配置块")
    if not bool(model.get("use_tactile_encoder", False)):
        raise ValueError(
            f"{path} 不是触觉融合配置：model.use_tactile_encoder 必须为 true"
        )
    if model.get("freeze_tactile_encoder") is not True:
        raise NotImplementedError("第一版 VT-SmolVLA 只支持 freeze_tactile_encoder=True")

    required = (
        "tactile_encoder_path",
        "tactile_keys",
        "tactile_embedding_dim",
        "tactile_num_tokens",
    )
    missing = [name for name in required if model.get(name) in (None, "", [])]
    if missing:
        raise ValueError(f"{path} 缺少触觉配置字段：{missing}")

    tactile_keys = model["tactile_keys"]
    if not isinstance(tactile_keys, list | tuple) or len(tactile_keys) != int(
        model["tactile_num_tokens"]
    ):
        raise ValueError(
            "model.tactile_keys 数量必须等于 model.tactile_num_tokens "
            f"({len(tactile_keys) if isinstance(tactile_keys, list | tuple) else 'invalid'} "
            f"!= {model['tactile_num_tokens']})"
        )

    image_keys = model.get("image_keys") or []
    overlap = sorted(set(image_keys) & set(tactile_keys))
    if overlap:
        raise ValueError(
            "触觉 keys 不应该放进 model.image_keys；它们会走 tactile encoder。"
            f" 重复字段：{overlap}"
        )

    cache = cfg.get("tactile_embedding_cache") or {}
    if not isinstance(cache, dict):
        raise ValueError("tactile_embedding_cache 必须是 mapping")
    if bool(cache.get("enabled", False)) and not cache.get("root"):
        raise ValueError(
            "tactile_embedding_cache.enabled=true 时必须配置 root"
        )


def _extra_loader_kwargs(cfg: Mapping[str, Any]) -> dict[str, Any]:
    cache = cfg.get("tactile_embedding_cache") or {}
    if not isinstance(cache, dict):
        raise ValueError("tactile_embedding_cache 必须是 mapping")
    enabled = bool(cache.get("enabled", False))
    root = cache.get("root") if enabled else None
    if enabled and not root:
        raise ValueError("tactile_embedding_cache.enabled=true 时必须配置 root")
    return {"tactile_embedding_cache_root": root}


VT_COMPONENTS = TrainingComponents(
    config_type=VTSmolVLAConfig,
    model_type=VTJaxSmolVLA,
    loader_type=VTLeRobotJaxDataLoader,
    trainer_type=VTJaxSmolVLATrainer,
    resolve_checkpoint=resolve_checkpoint,
    load_params=load_params,
    count_vlm_layers=count_vlm_layers,
    count_expert_layers=count_expert_layers,
    extend_vlm_layers=extend_vlm_layers,
    resolve_module_modes=resolve_module_modes,
    contract_from_config=contract_from_config,
    validate_checkpoint=validate_checkpoint,
    prepare_params=initialize_tactile_fusion_params,
    extra_loader_kwargs=_extra_loader_kwargs,
    allowed_top_level_keys=VT_ALLOWED_TOP_LEVEL_KEYS,
)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(
        argv,
        default_config=DEFAULT_CONFIG,
        description=__doc__ or "VT-SmolVLA training",
    )
    _validate_vt_config(args.config)
    run_training(args.config, components=VT_COMPONENTS)


if __name__ == "__main__":
    main()
