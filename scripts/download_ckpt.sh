#!/usr/bin/env bash

set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
ENV_FILE="${PROJECT_ROOT}/.env.frs"

log() { echo "[download-ckpt] $*"; }
fail() { echo "[download-ckpt] 错误：$*" >&2; exit 1; }
trap 'status=$?; echo "[download-ckpt] 下载失败，退出码 ${status}" >&2; exit "${status}"' ERR

usage() {
    printf '%s\n' \
        "用法：" \
        "  bash scripts/download_ckpt.sh OWNER/REPO" \
        "  bash scripts/download_ckpt.sh --OWNER/REPO" \
        "" \
        "完整下载 Hugging Face 模型到：" \
        "  <项目根目录>/checkpoints/model/<仓库名>"
}

if (($# != 1)); then
    usage >&2
    fail "必须提供一个 Hugging Face 模型仓库 ID"
fi

case "$1" in
    -h|--help)
        usage
        exit 0
        ;;
    --*)
        repo_id="${1#--}"
        ;;
    *)
        repo_id="$1"
        ;;
esac

if [[ ! "${repo_id}" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*$ ]]; then
    fail "无效的 Hugging Face 仓库 ID：${repo_id}；应为 OWNER/REPO"
fi

repo_name="${repo_id##*/}"
checkpoint_root="${PROJECT_ROOT}/checkpoints"
model_root="${checkpoint_root}/model"
metadata_root="${model_root}/.frs_hf_repos"
output_dir="${model_root}/${repo_name}"
repo_marker="${metadata_root}/${repo_name}.repo-id"

if [[ -f "${ENV_FILE}" ]]; then
    # shellcheck disable=SC1090
    source "${ENV_FILE}"
fi

if command -v uv >/dev/null 2>&1; then
    UV_BIN="$(command -v uv)"
elif [[ -n "${HOME:-}" && -x "${HOME}/.local/bin/uv" ]]; then
    UV_BIN="${HOME}/.local/bin/uv"
else
    fail "找不到 uv；请先运行 scripts/setup_env.sh"
fi

if [[ -L "${checkpoint_root}" || -L "${model_root}" || -L "${metadata_root}" ]]; then
    fail "checkpoint 下载路径不能是符号链接"
fi
mkdir -p "${metadata_root}"
if [[ -L "${output_dir}" ]]; then
    fail "模型下载目录不能是符号链接：${output_dir}"
fi
if [[ -e "${output_dir}" && ! -d "${output_dir}" ]]; then
    fail "下载目标不是目录：${output_dir}"
fi
if [[ -L "${repo_marker}" ]]; then
    fail "仓库归属标记不能是符号链接：${repo_marker}"
fi
if [[ -f "${repo_marker}" ]]; then
    owned_repo_id="$(<"${repo_marker}")"
    if [[ "${owned_repo_id}" != "${repo_id}" ]]; then
        fail "目录 ${output_dir} 已经属于 ${owned_repo_id}，拒绝混入 ${repo_id}"
    fi
elif [[ -d "${output_dir}" && -n "$(find "${output_dir}" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
    fail "目录 ${output_dir} 已存在但缺少仓库归属标记，请先确认或移走该目录"
else
    mkdir -p "${output_dir}"
    printf '%s\n' "${repo_id}" >"${repo_marker}"
fi

log "模型仓库：${repo_id}"
log "下载目录：${output_dir}"
"${UV_BIN}" run --no-sync hf download "${repo_id}" \
    --repo-type model \
    --local-dir "${output_dir}"
