#!/usr/bin/env bash
set -Eeuo pipefail

# 双臂纯视觉 SmolVLA（PyTorch）训练：
#   bash train_smolvla/scripts/start_smolvla_train.sh
# 只生成并显示 LeRobot 命令，不开始训练：
#   bash train_smolvla/scripts/start_smolvla_train.sh --dry-run

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
CONFIG_PATH="${SMOLVLA_TRAIN_CONFIG:-${PROJECT_ROOT}/train_smolvla/configs/train_smolvla.yaml}"

ENV_FILE="${PROJECT_ROOT}/env_path"
PREVIOUS_ENV_FILE="${PROJECT_ROOT}/environment_paths.sh"
LEGACY_ENV_FILE="${PROJECT_ROOT}/.env.frs"
if [[ ! -f "${ENV_FILE}" ]]; then
    if [[ -f "${PREVIOUS_ENV_FILE}" ]]; then
        ENV_FILE="${PREVIOUS_ENV_FILE}"
    elif [[ -f "${LEGACY_ENV_FILE}" ]]; then
        ENV_FILE="${LEGACY_ENV_FILE}"
    fi
fi
if [[ -f "${ENV_FILE}" ]]; then
    # shellcheck disable=SC1091
    source "${ENV_FILE}"
fi

# Hugging Face Datasets materializes image-bearing Parquet files as Arrow caches.
# A five-dataset run can exceed RunPod's small local overlay, so formal training
# keeps these files under the persistent workspace by default. Local /tmp is an
# explicit opt-in for pods whose overlay is known to be large enough.
if [[ "${SMOLVLA_USE_LOCAL_ARROW_CACHE:-0}" == "1" ]]; then
    SMOLVLA_LOCAL_CACHE_ROOT="${SMOLVLA_LOCAL_CACHE_ROOT:-/tmp/frs_tact_smolvla}"
    export HF_DATASETS_CACHE="${SMOLVLA_LOCAL_CACHE_ROOT}/datasets_arrow"
    export TMPDIR="${SMOLVLA_LOCAL_CACHE_ROOT}/tmp"
    mkdir -p "${HF_DATASETS_CACHE}" "${TMPDIR}"
    [[ -w "${HF_DATASETS_CACHE}" && -w "${TMPDIR}" ]] || {
        echo "[smolvla] 本地 Arrow/临时缓存目录不可写：${SMOLVLA_LOCAL_CACHE_ROOT}" >&2
        exit 1
    }
    echo "[smolvla] local Arrow cache: ${HF_DATASETS_CACHE}"
    echo "[smolvla] local temp dir: ${TMPDIR}"
else
    SMOLVLA_PERSISTENT_CACHE_ROOT="${SMOLVLA_PERSISTENT_CACHE_ROOT:-${WORKSPACE_ROOT:-/workspace}}"
    export HF_DATASETS_CACHE="${SMOLVLA_PERSISTENT_CACHE_ROOT}/huggingface/datasets_arrow"
    export TMPDIR="${SMOLVLA_PERSISTENT_CACHE_ROOT}/tmp"
    mkdir -p "${HF_DATASETS_CACHE}" "${TMPDIR}"
    [[ -w "${HF_DATASETS_CACHE}" && -w "${TMPDIR}" ]] || {
        echo "[smolvla] Workspace Arrow/temp cache is not writable: ${SMOLVLA_PERSISTENT_CACHE_ROOT}" >&2
        exit 1
    }
    echo "[smolvla] persistent Arrow cache: ${HF_DATASETS_CACHE}"
    echo "[smolvla] persistent temp dir: ${TMPDIR}"
fi

cd "${PROJECT_ROOT}"
if [[ -n "${SMOLVLA_TORCH_PYTHON:-}" ]]; then
    PYTHON_BIN="${SMOLVLA_TORCH_PYTHON}"
elif [[ -x "${WORKSPACE_ROOT:-/workspace}/venvs/smolvla_torch/bin/python" ]]; then
    PYTHON_BIN="${WORKSPACE_ROOT:-/workspace}/venvs/smolvla_torch/bin/python"
elif [[ -x "${PROJECT_ROOT}/.venv-smolvla-torch/bin/python" ]]; then
    PYTHON_BIN="${PROJECT_ROOT}/.venv-smolvla-torch/bin/python"
else
    echo "[smolvla] 未找到 FRS_Tact 的官方 LeRobot 训练环境。" >&2
    echo "[smolvla] 请先运行：bash scripts/setup_env.sh --smolvla" >&2
    exit 1
fi

if [[ ! -x "${PYTHON_BIN}" ]]; then
    echo "[smolvla] Python 不存在或不可执行：${PYTHON_BIN}" >&2
    echo "[smolvla] 请重新运行：bash scripts/setup_env.sh --smolvla" >&2
    exit 1
fi

# 让多卡训练时由包装器找到同一环境中的 accelerate。
export PATH="$(dirname -- "${PYTHON_BIN}"):${PATH}"
exec "${PYTHON_BIN}" -m train_smolvla.torch_train --config "${CONFIG_PATH}" "$@"
