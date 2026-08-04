#!/usr/bin/env bash

set -Eeuo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

log() {
    echo "[setup] $*"
}

fail() {
    echo "[setup] 错误：$*" >&2
    exit 1
}

apt_install() {
    if ! command -v apt-get >/dev/null 2>&1; then
        fail "当前系统没有 apt-get，请手动安装：$*"
    fi

    local -a apt_command
    if [[ "${EUID}" -eq 0 ]]; then
        apt_command=(apt-get)
    elif command -v sudo >/dev/null 2>&1; then
        apt_command=(sudo apt-get)
    else
        fail "安装系统依赖需要 root 或 sudo 权限：$*"
    fi

    "${apt_command[@]}" update
    "${apt_command[@]}" install -y "$@"
}

install_tmux() {
    if command -v tmux >/dev/null 2>&1; then
        log "tmux 已安装：$(tmux -V)"
        return
    fi
    log "正在安装 tmux"
    apt_install tmux
}

install_uv() {
    if command -v uv >/dev/null 2>&1; then
        log "uv 已安装：$(uv --version)"
        return
    fi

    if ! command -v curl >/dev/null 2>&1; then
        log "正在安装 curl"
        apt_install curl ca-certificates
    fi

    log "正在安装 uv"
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="${HOME}/.local/bin:${HOME}/.cargo/bin:${PATH}"
    hash -r
    command -v uv >/dev/null 2>&1 || fail "uv 安装完成但命令仍不可用，请重新打开终端后再运行"
    log "uv 已安装：$(uv --version)"
}

verify_python_environment() {
    log "验证 Python 依赖"
    uv run --frozen python - <<'PY'
import flax
import huggingface_hub
import jax
import lerobot
import optax
import torch
import wandb

print(f"huggingface_hub={huggingface_hub.__version__}")
print(f"wandb={wandb.__version__}")
print(f"jax={jax.__version__}")
print(f"torch={torch.__version__}")
print(f"flax={flax.__version__}")
print(f"optax={optax.__version__}")
PY

    uv run --frozen hf version
    uv run --frozen wandb --version
}

check_gpu() {
    log "检查 NVIDIA/JAX 设备"
    if command -v nvidia-smi >/dev/null 2>&1; then
        nvidia-smi --query-gpu=index,name,driver_version,memory.total --format=csv,noheader || true
    else
        echo "[setup] 警告：没有找到 nvidia-smi。" >&2
    fi

    uv run --frozen python - <<'PY'
import jax

devices = jax.devices()
print(f"JAX devices: {devices}")
if not any(device.platform == "gpu" for device in devices):
    print("警告：JAX 当前没有识别到 GPU，请检查 NVIDIA 驱动和 CUDA。")
PY
}

main() {
    install_tmux
    install_uv

    cd "${PROJECT_ROOT}"
    log "项目目录：${PROJECT_ROOT}"
    log "按照 uv.lock 创建/同步 .venv"
    uv sync --frozen

    verify_python_environment
    check_gpu

    log "环境安装完成"
    echo
    echo "首次使用时分别登录："
    echo "  uv run hf auth login"
    echo "  uv run wandb login"
    echo
    echo "训练命令："
    echo "  uv run python tools/train_vtsmolvla_jax.py --config configs/train_vtsmolvla_jax.yaml"
}

main "$@"
