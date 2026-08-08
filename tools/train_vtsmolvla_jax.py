#!/usr/bin/env python
"""训练视觉 + 触觉 encoder 版 JAX SmolVLA。

这个入口默认使用 ``configs/train_vtsmolvla_jax.yaml``，并在进入通用
SmolVLA JAX 训练流程前检查触觉融合配置。训练循环复用
``tools/train_smolvla_jax.py``，触觉数据与模型分支由配置开启。
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Sequence

import yaml

from lerobot.policies.smolvla_jax.atomic_checkpoint import paths_overlap


DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "train_vtsmolvla_jax.yaml"


def _config_arg(argv: Sequence[str]) -> Path | None:
    values: list[str] = []
    for index, arg in enumerate(argv):
        if arg == "--config":
            if index + 1 >= len(argv):
                raise ValueError("--config 后面需要跟一个 YAML 路径")
            value = argv[index + 1]
            if not value or value.startswith("--"):
                raise ValueError("--config 后面需要跟一个 YAML 路径")
            values.append(value)
        if arg.startswith("--config="):
            value = arg.split("=", 1)[1]
            if not value:
                raise ValueError("--config 后面需要跟一个 YAML 路径")
            values.append(value)
    if len(values) > 1:
        raise ValueError("--config 只能指定一次")
    return Path(values[0]) if values else None


def _argv_with_default_config(argv: Sequence[str]) -> list[str]:
    if _config_arg(argv) is not None:
        return list(argv)
    return ["--config", str(DEFAULT_CONFIG), *argv]


def _validate_vt_config(path: Path) -> dict[str, Any]:
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

    required = (
        "tactile_encoder_path",
        "tactile_encoder_repo_id",
        "tactile_keys",
        "tactile_embedding_dim",
        "tactile_num_tokens",
    )
    missing = [name for name in required if model.get(name) in (None, "", [])]
    if missing:
        raise ValueError(f"{path} 缺少触觉配置字段：{missing}")
    if model["tactile_encoder_repo_id"] != "liuchaoyi/encoder_ckpt_05":
        raise ValueError(
            "model.tactile_encoder_repo_id 必须是审批的 "
            "liuchaoyi/encoder_ckpt_05"
        )

    tactile_keys = model["tactile_keys"]
    if not isinstance(tactile_keys, list | tuple) or len(tactile_keys) != int(
        model["tactile_num_tokens"]
    ):
        raise ValueError(
            "model.tactile_keys 数量必须等于 model.tactile_num_tokens "
            f"({len(tactile_keys) if isinstance(tactile_keys, list | tuple) else 'invalid'} "
            f"!= {model['tactile_num_tokens']})"
        )

    repeat_factor = model.get("tactile_token_repeat_factor", 1)
    if (
        isinstance(repeat_factor, bool)
        or not isinstance(repeat_factor, int)
        or repeat_factor < 1
    ):
        raise ValueError(
            "model.tactile_token_repeat_factor 必须是正整数，"
            f"当前值：{repeat_factor!r}"
        )
    normalization = cfg.get("normalization")
    if repeat_factor > 1:
        if not isinstance(normalization, dict) or not normalization.get("protocol_dir"):
            raise ValueError(
                "K8/K21 配置必须显式设置 normalization.protocol_dir"
            )
        output = cfg.get("output")
        if output and paths_overlap(normalization["protocol_dir"], output):
            raise ValueError("normalization.protocol_dir 必须独立于单个 K 的 output")

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
    return cfg


def _validate_runtime_devices(config: dict[str, Any], devices: Sequence[Any]) -> None:
    """Enforce the paper K8/K21 hardware contract before any cache work starts."""

    repeat_factor = int((config.get("model") or {}).get("tactile_token_repeat_factor", 1))
    visible = list(devices)
    if repeat_factor > 1:
        exact_h100_pair = len(visible) == 2 and all(
            getattr(device, "platform", None) == "gpu"
            and "H100" in str(getattr(device, "device_kind", "")).upper()
            for device in visible
        )
        if not exact_h100_pair:
            raise RuntimeError(
                "K8/K21 paper runs require exactly two visible NVIDIA H100 GPUs; "
                f"got {visible!r}"
            )
        return
    if not any(getattr(device, "platform", None) == "gpu" for device in visible):
        raise RuntimeError("K1 VT training requires at least one visible GPU")


def main() -> None:
    argv = _argv_with_default_config(sys.argv[1:])
    config_path = _config_arg(argv)
    assert config_path is not None
    config = _validate_vt_config(config_path)

    import jax

    _validate_runtime_devices(config, jax.devices())
    sys.argv = [sys.argv[0], *argv]

    import train_smolvla_jax as base

    base.main()


if __name__ == "__main__":
    main()
