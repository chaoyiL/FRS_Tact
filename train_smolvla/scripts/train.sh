#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
CONFIG_PATH="${1:-${PROJECT_ROOT}/train_smolvla/configs/train.yaml}"

if [[ -f "${PROJECT_ROOT}/.env.frs" ]]; then
    # shellcheck disable=SC1091
    source "${PROJECT_ROOT}/.env.frs"
fi

UV_BIN="$(command -v uv)" || {
    echo "[smolvla] 错误：找不到 uv，请先运行 scripts/setup_env.sh" >&2
    exit 1
}

cd "${PROJECT_ROOT}"
exec "${UV_BIN}" run --no-sync python -m train_smolvla.launcher --config "${CONFIG_PATH}"
