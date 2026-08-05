#!/usr/bin/env bash

set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
PYTHON_VERSION="3.12"
ENV_FILE="${PROJECT_ROOT}/.env.frs"

if [[ "${PROJECT_ROOT}" == /workspace/* ]]; then
    DEFAULT_VENV_DIR="/opt/venvs/frs_tact"
    DEFAULT_STORAGE_ROOT="/workspace"
else
    DEFAULT_VENV_DIR="${PROJECT_ROOT}/.venv"
    DEFAULT_STORAGE_ROOT="${PROJECT_ROOT}/.cache"
fi
VENV_DIR="${FRS_VENV_DIR:-${DEFAULT_VENV_DIR}}"
UV_CACHE_DIR_VALUE="${UV_CACHE_DIR:-${HOME}/.cache/uv}"
STORAGE_ROOT="${FRS_STORAGE_ROOT:-${DEFAULT_STORAGE_ROOT}}"
HF_HOME_VALUE="${STORAGE_ROOT}/huggingface"
HF_HUB_CACHE_VALUE="${HF_HOME_VALUE}/hub"
HF_DATASETS_CACHE_VALUE="${HF_HOME_VALUE}/datasets_arrow"
HF_LEROBOT_HOME_VALUE="${HF_HOME_VALUE}/lerobot"
TMPDIR_VALUE="${STORAGE_ROOT}/tmp"
UV_BIN=""

log() {
    echo "[setup] $*"
}

warn() {
    echo "[setup] 警告：$*" >&2
}

fail() {
    echo "[setup] 错误：$*" >&2
    exit 1
}

install_system_dependencies() {
    local -a packages=()
    command -v curl >/dev/null 2>&1 || packages+=(curl)
    command -v git >/dev/null 2>&1 || packages+=(git)
    command -v tmux >/dev/null 2>&1 || packages+=(tmux)
    command -v rsync >/dev/null 2>&1 || packages+=(rsync)
    command -v ffmpeg >/dev/null 2>&1 || packages+=(ffmpeg)
    [[ -f /etc/ssl/certs/ca-certificates.crt ]] || packages+=(ca-certificates)
    if ((${#packages[@]} == 0)); then
        log "系统依赖已安装：tmux=$(tmux -V 2>/dev/null || true)"
        return
    fi
    command -v apt-get >/dev/null 2>&1 || fail "当前系统没有 apt-get，请手动安装：${packages[*]}"
    local -a apt_command
    if [[ "${EUID}" -eq 0 ]]; then
        apt_command=(apt-get)
    elif command -v sudo >/dev/null 2>&1; then
        apt_command=(sudo apt-get)
    else
        fail "安装系统依赖需要 root 或 sudo 权限：${packages[*]}"
    fi
    log "安装系统依赖：${packages[*]}"
    "${apt_command[@]}" update
    DEBIAN_FRONTEND=noninteractive "${apt_command[@]}" install -y "${packages[@]}"
}

find_uv() {
    if command -v uv >/dev/null 2>&1; then
        UV_BIN="$(command -v uv)"
    elif [[ -x "${HOME}/.local/bin/uv" ]]; then
        UV_BIN="${HOME}/.local/bin/uv"
    elif [[ -x "${HOME}/.cargo/bin/uv" ]]; then
        UV_BIN="${HOME}/.cargo/bin/uv"
    fi
}

install_uv() {
    find_uv
    if [[ -n "${UV_BIN}" ]]; then
        log "uv 已安装：$(${UV_BIN} --version) (${UV_BIN})"
        return
    fi
    log "正在安装 uv"
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="${HOME}/.local/bin:${HOME}/.cargo/bin:${PATH}"
    hash -r
    find_uv
    [[ -n "${UV_BIN}" ]] || fail "uv 安装完成但命令仍不可用"
    log "uv 已安装：$(${UV_BIN} --version) (${UV_BIN})"
}

persist_uv_path() {
    local marker="# FRS_Tact uv PATH"
    local line="export PATH=\"\${HOME}/.local/bin:\${HOME}/.cargo/bin:\${PATH}\""
    if [[ -w "${HOME}" ]] && ! grep -Fq "${marker}" "${HOME}/.bashrc" 2>/dev/null; then
        printf '\n%s\n%s\n' "${marker}" "${line}" >>"${HOME}/.bashrc"
        log "已把 uv PATH 写入 ${HOME}/.bashrc"
    fi
}

check_existing_uv_processes() {
    local running
    running="$(ps -eo pid=,etime=,cmd= | awk '/[u]v sync|[u]v run/ {print}')"
    if [[ -n "${running}" && "${FRS_IGNORE_UV_PROCESSES:-0}" != "1" ]]; then
        echo "${running}" >&2
        fail "检测到其他 uv sync/run 进程。请等待其结束，或确认后设置 FRS_IGNORE_UV_PROCESSES=1。"
    fi
}

configure_uv_storage() {
    mkdir -p "$(dirname -- "${VENV_DIR}")" "${UV_CACHE_DIR_VALUE}"
    export UV_PROJECT_ENVIRONMENT="${VENV_DIR}"
    export UV_CACHE_DIR="${UV_CACHE_DIR_VALUE}"

    local env_device cache_device
    env_device="$(stat -c '%d' "$(dirname -- "${VENV_DIR}")")"
    cache_device="$(stat -c '%d' "${UV_CACHE_DIR_VALUE}")"
    if [[ "${env_device}" != "${cache_device}" ]]; then
        export UV_LINK_MODE="copy"
        warn "uv cache 和虚拟环境不在同一文件系统，使用 UV_LINK_MODE=copy"
    fi
}

configure_runtime_storage() {
    mkdir -p \
        "${HF_HUB_CACHE_VALUE}" \
        "${HF_DATASETS_CACHE_VALUE}" \
        "${HF_LEROBOT_HOME_VALUE}" \
        "${TMPDIR_VALUE}"
    export HF_HOME="${HF_HOME_VALUE}"
    export HF_HUB_CACHE="${HF_HUB_CACHE_VALUE}"
    export HF_DATASETS_CACHE="${HF_DATASETS_CACHE_VALUE}"
    export HF_LEROBOT_HOME="${HF_LEROBOT_HOME_VALUE}"
    export TMPDIR="${TMPDIR_VALUE}"
}

write_environment_file() {
    {
        echo "# 由 setup_env.sh 生成；供训练脚本复用。"
        printf 'export PATH=%q\n' "${HOME}/.local/bin:${HOME}/.cargo/bin:${PATH}"
        printf 'export UV_PROJECT_ENVIRONMENT=%q\n' "${UV_PROJECT_ENVIRONMENT}"
        printf 'export UV_CACHE_DIR=%q\n' "${UV_CACHE_DIR}"
        printf 'export HF_HOME=%q\n' "${HF_HOME}"
        printf 'export HF_HUB_CACHE=%q\n' "${HF_HUB_CACHE}"
        printf 'export HF_DATASETS_CACHE=%q\n' "${HF_DATASETS_CACHE}"
        printf 'export HF_LEROBOT_HOME=%q\n' "${HF_LEROBOT_HOME}"
        printf 'export TMPDIR=%q\n' "${TMPDIR}"
        if [[ -n "${UV_LINK_MODE:-}" ]]; then
            printf 'export UV_LINK_MODE=%q\n' "${UV_LINK_MODE}"
        fi
    } >"${ENV_FILE}"
    chmod 600 "${ENV_FILE}"
}

sync_environment() {
    cd "${PROJECT_ROOT}"
    check_existing_uv_processes
    log "安装 Python ${PYTHON_VERSION}"
    "${UV_BIN}" python install "${PYTHON_VERSION}"
    log "环境目录：${UV_PROJECT_ENVIRONMENT}"
    log "uv cache：${UV_CACHE_DIR}"
    log "按照 uv.lock 同步依赖；大型 PyTorch/CUDA wheel 第一次需要数分钟"
    "${UV_BIN}" sync --frozen --python "${PYTHON_VERSION}"
    write_environment_file
}

verify_python_environment() {
    cd "${PROJECT_ROOT}"
    log "验证 Python、CLI 和训练依赖（不再执行 sync）"
    "${UV_BIN}" run --no-sync python - <<'PY'
import sys

import flax
import huggingface_hub
import jax
import optax
import torch
import torchcodec
import wandb

if sys.version_info[:2] != (3, 12):
    raise RuntimeError(f"需要 Python 3.12，当前为 {sys.version}")
print(f"python={sys.version.split()[0]}")
print(f"huggingface_hub={huggingface_hub.__version__}")
print(f"wandb={wandb.__version__}")
print(f"jax={jax.__version__}")
print(f"torch={torch.__version__} cuda={torch.version.cuda}")
print(f"torchcodec={getattr(torchcodec, '__version__', 'installed')}")
print(f"flax={flax.__version__}")
print(f"optax={optax.__version__}")
PY
    "${UV_BIN}" run --no-sync hf version
    "${UV_BIN}" run --no-sync wandb --version
}

check_gpu() {
    cd "${PROJECT_ROOT}"
    log "检查 NVIDIA、PyTorch 和 JAX 设备"
    local expect_gpu=0
    if command -v nvidia-smi >/dev/null 2>&1; then
        expect_gpu=1
        nvidia-smi --query-gpu=index,name,driver_version,memory.total --format=csv,noheader
    else
        warn "没有找到 nvidia-smi；本机只能做 CPU 开发，不能用于正式训练"
    fi
    FRS_EXPECT_GPU="${expect_gpu}" "${UV_BIN}" run --no-sync python - <<'PY'
import os

import jax
import torch

devices = jax.devices()
print(f"JAX devices: {devices}")
print(f"PyTorch CUDA available: {torch.cuda.is_available()}")
if os.environ.get("FRS_EXPECT_GPU") == "1":
    if not torch.cuda.is_available():
        raise RuntimeError("nvidia-smi 可用，但 PyTorch 没有识别到 CUDA")
    if not any(device.platform == "gpu" for device in devices):
        raise RuntimeError("nvidia-smi 可用，但 JAX 没有识别到 GPU")
PY
}

main() {
    install_system_dependencies
    install_uv
    persist_uv_path
    configure_uv_storage
    configure_runtime_storage
    sync_environment
    verify_python_environment
    check_gpu

    log "环境安装完成"
    echo
    echo "环境目录：${VENV_DIR}"
    echo "环境变量：${ENV_FILE}"
    echo "Hugging Face 缓存：${HF_HOME}"
    echo "Arrow 数据缓存：${HF_DATASETS_CACHE}"
    echo
    echo "首次使用时登录："
    echo "  cd ${PROJECT_ROOT}"
    echo "  source ${ENV_FILE}"
    echo "  ${UV_BIN} run --no-sync hf auth login"
    echo "  ${UV_BIN} run --no-sync wandb login"
    echo
    echo "一键启动 VT-SmolVLA："
    echo "  bash ${PROJECT_ROOT}/scripts/start_vtsmolvla_train.sh"
}

main "$@"
