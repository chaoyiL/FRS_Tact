#!/usr/bin/env bash
set -Eeuo pipefail

# 右手单臂纯视觉 SmolVLA（PyTorch）训练，要求数据为 7D state / 10D action：
#   bash train_smolvla/scripts/start_smolvla_right_train.sh
# 只生成并显示 LeRobot 命令，不开始训练：
#   bash train_smolvla/scripts/start_smolvla_right_train.sh --dry-run

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
CONFIG_PATH="${PROJECT_ROOT}/train_smolvla/configs/train_pytorch_right.yaml"

if [[ -f "${PROJECT_ROOT}/.env.frs" ]]; then
    # shellcheck disable=SC1091
    source "${PROJECT_ROOT}/.env.frs"
fi

cd "${PROJECT_ROOT}"
if [[ -n "${SMOLVLA_TORCH_PYTHON:-}" ]]; then
    PYTHON_BIN="${SMOLVLA_TORCH_PYTHON}"
elif [[ -x "/home/typhon/vb3/.venv/bin/python" ]]; then
    PYTHON_BIN="/home/typhon/vb3/.venv/bin/python"
else
    echo "[smolvla-right] 请设置 SMOLVLA_TORCH_PYTHON，指向 VB3/官方 LeRobot 环境的 Python" >&2
    exit 1
fi

exec "${PYTHON_BIN}" -m train_smolvla.torch_train --config "${CONFIG_PATH}" "$@"
