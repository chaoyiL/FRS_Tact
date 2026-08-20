#!/usr/bin/env bash
# 示例：
# bash scripts/download_encoder.sh \
#   --repo-id KaiyueChen/encoder_ckpt_0809 \
#   --output-dir /workspace/FRS_Tact/checkpoints/encoder_ckpt_0809/best

set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
ENV_FILE="${PROJECT_ROOT}/.env.frs"

log() { echo "[download-encoder] $*"; }
fail() { echo "[download-encoder] 错误：$*" >&2; exit 1; }
trap 'status=$?; echo "[download-encoder] 下载失败，退出码 ${status}" >&2; exit "${status}"' ERR

usage() {
    printf '%s\n' \
        "用法：scripts/download_encoder.sh [--full] [download_ckpt.py 参数]" \
        "" \
        "默认下载 FRS 所需的最小 tactile encoder checkpoint。" \
        "  --full                 下载 optimizer 和 memory bank 等完整文件" \
        "  --repo-id ID           覆盖 Hugging Face 仓库" \
        "  --output-dir PATH      覆盖下载目录" \
        "  --revision REVISION    指定分支、tag 或 commit" \
        "  --cache-dir PATH       指定 Hugging Face 缓存目录" \
        "  --force-download       强制重新下载"
}

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

# FRS 只需要 encoder 参数，默认不下载 optimizer/memory bank。
# 传入 --full 可下载完整 checkpoint；其余参数原样传给 download_ckpt.py。
minimal=1
forwarded=()
for argument in "$@"; do
    case "${argument}" in
        -h|--help)
            usage
            exit 0
            ;;
        --full)
            minimal=0
            ;;
        *)
            forwarded+=("${argument}")
            ;;
    esac
done

command=(
    "${UV_BIN}" run --no-sync python "${PROJECT_ROOT}/deploy_smolvla/src/download_ckpt.py"
)
if ((minimal)); then
    command+=(--minimal)
fi
command+=("${forwarded[@]}")

log "下载 tactile encoder checkpoint"
log "默认仓库：liuchaoyi/encoder_ckpt_06"
log "默认目录：${PROJECT_ROOT}/checkpoints/encoder/encoder_ckpt_06"
"${command[@]}"
