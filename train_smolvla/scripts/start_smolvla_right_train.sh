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
elif [[ -x "${WORKSPACE_ROOT:-/workspace}/venvs/smolvla_torch/bin/python" ]]; then
    PYTHON_BIN="${WORKSPACE_ROOT:-/workspace}/venvs/smolvla_torch/bin/python"
elif [[ -x "${PROJECT_ROOT}/.venv-smolvla-torch/bin/python" ]]; then
    PYTHON_BIN="${PROJECT_ROOT}/.venv-smolvla-torch/bin/python"
else
    echo "[smolvla-right] 未找到 FRS_Tact 的官方 LeRobot 训练环境。" >&2
    echo "[smolvla-right] 请先运行：bash scripts/setup_env.sh --smolvla" >&2
    exit 1
fi

if [[ ! -x "${PYTHON_BIN}" ]]; then
    echo "[smolvla-right] Python 不存在或不可执行：${PYTHON_BIN}" >&2
    echo "[smolvla-right] 请重新运行：bash scripts/setup_env.sh --smolvla" >&2
    exit 1
fi

# 让多卡训练时由包装器找到同一环境中的 accelerate。
export PATH="$(dirname -- "${PYTHON_BIN}"):${PATH}"
exec "${PYTHON_BIN}" -m train_smolvla.torch_train --config "${CONFIG_PATH}" "$@"
