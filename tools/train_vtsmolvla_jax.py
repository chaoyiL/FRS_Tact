#!/usr/bin/env python
"""训练视觉 + 触觉 encoder 版 JAX SmolVLA。

这个入口默认使用 ``configs/train_vtsmolvla_jax.yaml``，并在进入通用
SmolVLA JAX 训练流程前检查触觉融合配置。训练循环复用
``tools/train_smolvla_jax.py``，触觉数据与模型分支由配置开启。
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Sequence

import yaml


DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "train_vtsmolvla_jax.yaml"


def _config_arg(argv: Sequence[str]) -> Path | None:
    for index, arg in enumerate(argv):
        if arg == "--config":
            if index + 1 >= len(argv):
                raise ValueError("--config 后面需要跟一个 YAML 路径")
            return Path(argv[index + 1])
        if arg.startswith("--config="):
            return Path(arg.split("=", 1)[1])
    return None


def _argv_with_default_config(argv: Sequence[str]) -> list[str]:
    if _config_arg(argv) is not None:
        return list(argv)
    return ["--config", str(DEFAULT_CONFIG), *argv]


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


def main() -> None:
    argv = _argv_with_default_config(sys.argv[1:])
    config_path = _config_arg(argv)
    assert config_path is not None
    _validate_vt_config(config_path)
    sys.argv = [sys.argv[0], *argv]

    import train_smolvla_jax as base

    base.main()


if __name__ == "__main__":
    main()
