#!/usr/bin/env python3
"""从 Hugging Face 下载并校验 tactile encoder checkpoint。

uv run --frozen python download_ckpt.py
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
import json
import os
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
from huggingface_hub import HfApi, snapshot_download
from huggingface_hub.errors import HfHubHTTPError


DEFAULT_REPO_ID = "liuchaoyi/encoder_ckpt_05"
DEFAULT_OUTPUT_DIR = Path("/workspace/checkpoints/encoder_ckpt_05")
MINIMAL_CHECKPOINT_PATTERNS = ("checkpoint.json", "params.npz", "params-*.npz")
FULL_CHECKPOINT_PATTERNS = (
    "checkpoint.json",
    "params.npz",
    "params-*.npz",
    "opt_state.npz",
    "opt_state-*.npz",
    "opt_state.treedef.pkl",
    "opt_state-*.treedef.pkl",
    "memory_bank.npz",
    "memory_bank-*.npz",
)


def normalize_repo_id(value: str) -> str:
    """同时接受 ``namespace/repo`` 和完整 Hugging Face URL。"""

    value = value.strip().rstrip("/")
    if value.startswith(("http://", "https://")):
        parsed = urlparse(value)
        if parsed.netloc not in {"huggingface.co", "www.huggingface.co"}:
            raise ValueError(f"不是 Hugging Face 仓库 URL：{value}")
        parts = [part for part in parsed.path.split("/") if part]
        if len(parts) < 2:
            raise ValueError(f"无法从 URL 解析 repo id：{value}")
        value = "/".join(parts[:2])
    if value.count("/") != 1:
        raise ValueError(f"repo id 应为 namespace/name，实际为：{value!r}")
    return value


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-id",
        default=DEFAULT_REPO_ID,
        help=f"模型仓库 ID 或 URL（默认：{DEFAULT_REPO_ID}）",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"下载目录（默认：{DEFAULT_OUTPUT_DIR}）",
    )
    parser.add_argument("--revision", default="main", help="分支、tag 或 commit（默认：main）")
    parser.add_argument("--cache-dir", type=Path, default=None, help="可选 Hugging Face 缓存目录")
    parser.add_argument(
        "--token",
        default=None,
        help="可选 HF token；默认读取 HF_TOKEN 或本机 hf auth login 凭据",
    )
    parser.add_argument(
        "--minimal",
        action="store_true",
        help="只下载推理/VT-SmolVLA 所需的 checkpoint.json 和参数归档",
    )
    parser.add_argument("--force-download", action="store_true", help="强制重新下载文件")
    return parser.parse_args(argv)


def verify_checkpoint(directory: Path) -> dict:
    """检查 checkpoint 元数据和参数归档是否能够被 tactile loader 读取。"""

    checkpoint_path = directory / "checkpoint.json"
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"checkpoint 缺少必要文件：{checkpoint_path}")
    with checkpoint_path.open(encoding="utf-8") as file:
        metadata = json.load(file)
    params_name = str(metadata.get("params_file", "params.npz"))
    params_path = directory / params_name
    if not params_path.is_file():
        raise FileNotFoundError(f"checkpoint 缺少参数归档：{params_path}")
    tactile_config = metadata.get("tactile_clip_config")
    if not isinstance(tactile_config, dict):
        raise ValueError("checkpoint.json 缺少 tactile_clip_config")
    parameter_paths = metadata.get("parameter_paths")
    if not isinstance(parameter_paths, list) or not parameter_paths:
        raise ValueError("checkpoint.json 缺少 parameter_paths")
    if not any(str(path).startswith("tactile_resnet/") for path in parameter_paths):
        raise ValueError("checkpoint 中没有 tactile_resnet 参数")

    with np.load(params_path) as archive:
        if len(archive.files) != len(parameter_paths):
            raise ValueError(
                f"{params_name} 参数数量与 checkpoint.json 不一致："
                f"{len(archive.files)} != {len(parameter_paths)}"
            )
    return metadata


def main() -> None:
    args = parse_args()
    repo_id = normalize_repo_id(args.repo_id)
    output_dir = args.output_dir.expanduser().resolve()
    cache_dir = None if args.cache_dir is None else args.cache_dir.expanduser().resolve()
    token = args.token or os.environ.get("HF_TOKEN") or None

    print(f"仓库：{repo_id}")
    print(f"目标：{output_dir}")
    print(f"模式：{'最小训练依赖' if args.minimal else '完整 checkpoint'}")
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        info = HfApi(token=token).model_info(repo_id, revision=args.revision)
        resolved_revision = info.sha
        print(f"版本：{args.revision} -> {resolved_revision}")
        snapshot_download(
            repo_id=repo_id,
            repo_type="model",
            revision=resolved_revision,
            local_dir=output_dir,
            cache_dir=cache_dir,
            token=token,
            allow_patterns=list(
                MINIMAL_CHECKPOINT_PATTERNS if args.minimal else FULL_CHECKPOINT_PATTERNS
            ),
            force_download=args.force_download,
        )
    except HfHubHTTPError as error:
        if error.response is not None and error.response.status_code in {401, 403}:
            raise SystemExit(
                "没有权限访问仓库。请先执行 `uv run hf auth login`，"
                "或设置环境变量 HF_TOKEN。"
            ) from error
        raise

    metadata = verify_checkpoint(output_dir)
    tactile_config = metadata["tactile_clip_config"]
    print("校验通过：")
    print(f"  epoch={metadata.get('epoch')}")
    print(f"  backbone={metadata.get('tactile_backbone')}")
    print(f"  embedding_dim={tactile_config.get('embedding_dim')}")
    print(f"  tactile_image_size={tactile_config.get('tactile_image_size')}")
    print(f"  tactile_history={tactile_config.get('tactile_history')}")
    print(f"  本地路径={output_dir}")


if __name__ == "__main__":
    main()
