#!/usr/bin/env bash
# 示例：
# bash scripts/upload_frs_ckpt.sh \
#   KaiyueChen/smolvla_frs_pick_tube_05_bimanual_best \
#   /workspace/frs_pick_tube_05/frs_bimanual_gated_01/best

set -Eeuo pipefail

log() { echo "[upload-frs] $*"; }
fail() { echo "[upload-frs] 错误：$*" >&2; exit 1; }

usage() {
    printf '%s\n' \
        "用法：" \
        "  bash scripts/upload_frs_ckpt.sh OWNER/REPO /path/to/checkpoint [--figures-dir PATH]" \
        "" \
        "将指定 FRS checkpoint 上传到 public Hugging Face model 仓库根目录，" \
        "并将训练图像上传到仓库的 figures/ 目录。"
}

if (($# == 1)) && [[ "$1" == "-h" || "$1" == "--help" ]]; then
    usage
    exit 0
fi
if (($# != 2 && $# != 4)); then
    usage >&2
    fail "必须提供仓库 ID、checkpoint 路径，以及可选的 --figures-dir PATH"
fi

repo_id="$1"
checkpoint_input="$2"
if [[ ! "${repo_id}" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*$ ]]; then
    fail "无效的 Hugging Face 仓库 ID：${repo_id}；应为 OWNER/REPO"
fi

if (($# == 4)); then
    [[ "$3" == "--figures-dir" ]] || fail "未知选项：$3"
    figures_input="$4"
else
    figures_input="$(dirname -- "${checkpoint_input}")"
fi

[[ ! -L "${checkpoint_input}" ]] || fail "checkpoint 目录不能是符号链接：${checkpoint_input}"
[[ ! -L "${figures_input}" ]] || fail "图像目录不能是符号链接：${figures_input}"
[[ -d "${checkpoint_input}" ]] || fail "checkpoint 目录不存在：${checkpoint_input}"
[[ -d "${figures_input}" ]] || fail "图像目录不存在：${figures_input}"
checkpoint_dir="$(cd -- "${checkpoint_input}" && pwd -P)"
figures_dir="$(cd -- "${figures_input}" && pwd -P)"
[[ -f "${checkpoint_dir}/checkpoint.json" ]] || fail "缺少 checkpoint.json：${checkpoint_dir}"
[[ ! -L "${checkpoint_dir}/checkpoint.json" ]] || fail "checkpoint.json 不能是符号链接"
if [[ -z "$(find "${figures_dir}" -maxdepth 1 -type f -name '*.png' -print -quit)" ]]; then
    fail "图像目录中没有顶层 PNG：${figures_dir}"
fi

if command -v uv >/dev/null 2>&1; then
    UV_BIN="$(command -v uv)"
elif [[ -n "${HOME:-}" && -x "${HOME}/.local/bin/uv" ]]; then
    UV_BIN="${HOME}/.local/bin/uv"
else
    fail "找不到 uv；请先运行 scripts/setup_env.sh"
fi

if ! params_path="$(
    "${UV_BIN}" run --no-sync python - \
        "${checkpoint_dir}/checkpoint.json" "${checkpoint_dir}" <<'PY'
import hashlib
import json
from pathlib import Path
import sys

metadata_path = Path(sys.argv[1])
checkpoint_dir = Path(sys.argv[2]).resolve(strict=True)
try:
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
except (OSError, UnicodeError, json.JSONDecodeError) as exc:
    raise SystemExit(f"无法读取 checkpoint.json：{exc}") from exc


def sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def canonical_json(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True) + "\n").encode("utf-8")


try:
    version = int(metadata.get("version", 1))
except (TypeError, ValueError) as exc:
    raise SystemExit("checkpoint.json 的 version 无效") from exc

if version >= 3:
    generation = metadata.get("generation")
    if not isinstance(generation, str) or not generation:
        raise SystemExit("checkpoint v3 缺少有效的 generation")
    if checkpoint_dir.parent.name == ".checkpoint-generations" and checkpoint_dir.name != generation:
        raise SystemExit(
            f"checkpoint generation 不匹配：metadata={generation!r}, directory={checkpoint_dir.name!r}"
        )

    files = metadata.get("files")
    if not isinstance(files, dict):
        raise SystemExit("checkpoint v3 缺少有效的 files 清单")
    expected_files = {"params.npz", "checkpoint.json"}
    if metadata.get("has_opt_state"):
        expected_files.update(("opt_state.npz", "opt_state.treedef.pkl"))
    if set(files) != expected_files:
        raise SystemExit(
            f"checkpoint 文件清单不匹配：{sorted(files)} != {sorted(expected_files)}"
        )

    for name in sorted(expected_files):
        record = files.get(name)
        if not isinstance(record, dict):
            raise SystemExit(f"checkpoint 文件记录无效：{name}")
        if record.get("generation") != generation:
            raise SystemExit(f"checkpoint 文件 generation 不匹配：{name}")
        path = checkpoint_dir / name
        if path.is_symlink() or not path.is_file():
            raise SystemExit(f"checkpoint 文件不存在或不是普通文件：{name}")
        if path.resolve(strict=True).parent != checkpoint_dir:
            raise SystemExit(f"checkpoint 文件越出 checkpoint 目录：{name}")
        if name == "checkpoint.json":
            canonical = json.loads(json.dumps(metadata))
            canonical["files"]["checkpoint.json"]["sha256"] = ""
            payload = canonical_json(canonical)
        else:
            payload = path.read_bytes()
        if record.get("size") != len(payload):
            raise SystemExit(f"checkpoint 文件大小不匹配：{name}")
        if record.get("sha256") != sha256(payload):
            raise SystemExit(f"checkpoint 文件校验和不匹配：{name}")
    params_path = checkpoint_dir / "params.npz"
else:
    params_file = metadata.get("params_file")
    if not isinstance(params_file, str) or not params_file:
        raise SystemExit("旧版 checkpoint.json 缺少有效的 params_file")
    relative_path = Path(params_file)
    if relative_path.is_absolute() or relative_path.parent != Path("."):
        raise SystemExit(f"params_file 必须是 checkpoint 目录中的文件名：{params_file}")
    params_path = checkpoint_dir / relative_path
    if params_path.is_symlink() or not params_path.is_file():
        raise SystemExit(f"params_file 不存在或不是普通文件：{params_file}")
    if params_path.resolve(strict=True).parent != checkpoint_dir:
        raise SystemExit(f"params_file 越出 checkpoint 目录：{params_file}")

print(params_path.resolve(strict=True))
PY
)"; then
    fail "checkpoint generation 校验失败"
fi

log "仓库：${repo_id}（public）"
log "checkpoint：${checkpoint_dir}"
log "参数文件：${params_path}"
log "图像目录：${figures_dir}"
"${UV_BIN}" run --no-sync hf auth whoami
"${UV_BIN}" run --no-sync hf repo create "${repo_id}" \
    --repo-type model \
    --exist-ok
"${UV_BIN}" run --no-sync hf upload "${repo_id}" \
    "${checkpoint_dir}" . \
    --repo-type model \
    --commit-message "Upload FRS checkpoint"
"${UV_BIN}" run --no-sync hf upload "${repo_id}" \
    "${figures_dir}" figures \
    --repo-type model \
    --include "*.png" \
    --commit-message "Upload FRS training figures"

log "上传完成：https://huggingface.co/${repo_id}"
