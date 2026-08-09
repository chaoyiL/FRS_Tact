#!/usr/bin/env bash

set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
PYTHON_VERSION="3.12"
ENV_FILE="${PROJECT_ROOT}/.env.frs"

STORAGE_ROOT="${FRS_STORAGE_ROOT:-/DATA/ljl/substage}"
VENV_DIR="${FRS_VENV_DIR:-/home/ljl/.venvs/frs_tact}"
UV_PROJECT_ENVIRONMENT="${VENV_DIR}"
UV_CACHE_DIR_VALUE="${FRS_UV_CACHE_DIR:-${STORAGE_ROOT}/.cache/uv}"
HF_HOME_VALUE="${STORAGE_ROOT}/huggingface"
HF_HUB_CACHE_VALUE="${HF_HOME_VALUE}/hub"
HF_DATASETS_CACHE_VALUE="${HF_HOME_VALUE}/datasets_arrow"
HF_LEROBOT_HOME_VALUE="${HF_HOME_VALUE}/lerobot"
TMPDIR_VALUE="${STORAGE_ROOT}/tmp"
FRS_LOG_DIR_VALUE="${STORAGE_ROOT}/logs"
WANDB_DIR_VALUE="${FRS_LOG_DIR_VALUE}/wandb"
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

acquire_project_lock() {
    command -v flock >/dev/null 2>&1 || fail "找不到 flock（util-linux）"
    mkdir -p "${STORAGE_ROOT}/.locks"
    exec 9>"${STORAGE_ROOT}/.locks/frs-setup.lock"
    flock -n 9 || fail "另一个 FRS 环境安装正在运行"
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
        "${TMPDIR_VALUE}" \
        "${WANDB_DIR_VALUE}"
    export HF_HOME="${HF_HOME_VALUE}"
    export HF_HUB_CACHE="${HF_HUB_CACHE_VALUE}"
    export HF_DATASETS_CACHE="${HF_DATASETS_CACHE_VALUE}"
    export HF_LEROBOT_HOME="${HF_LEROBOT_HOME_VALUE}"
    export TMPDIR="${TMPDIR_VALUE}"
    export FRS_LOG_DIR="${FRS_LOG_DIR_VALUE}"
    export WANDB_DIR="${WANDB_DIR_VALUE}"
}

write_environment_file() {
    local env_tmp
    env_tmp="$(mktemp --tmpdir="${PROJECT_ROOT}" .env.frs.XXXXXX)"
    {
        echo "# 由 setup_env.sh 生成；供训练脚本复用。"
        printf 'export PATH=%q\n' "${HOME}/.local/bin:${HOME}/.cargo/bin:${PATH}"
        printf 'export FRS_STORAGE_ROOT=%q\n' "${STORAGE_ROOT}"
        printf 'export FRS_VENV_DIR=%q\n' "${VENV_DIR}"
        printf 'export UV_PROJECT_ENVIRONMENT=%q\n' "${UV_PROJECT_ENVIRONMENT}"
        printf 'export UV_CACHE_DIR=%q\n' "${UV_CACHE_DIR}"
        printf 'export HF_HOME=%q\n' "${HF_HOME}"
        printf 'export HF_HUB_CACHE=%q\n' "${HF_HUB_CACHE}"
        printf 'export HF_DATASETS_CACHE=%q\n' "${HF_DATASETS_CACHE}"
        printf 'export HF_LEROBOT_HOME=%q\n' "${HF_LEROBOT_HOME}"
        printf 'export TMPDIR=%q\n' "${TMPDIR}"
        printf 'export FRS_LOG_DIR=%q\n' "${FRS_LOG_DIR_VALUE}"
        printf 'export WANDB_DIR=%q\n' "${WANDB_DIR_VALUE}"
        if [[ -n "${UV_LINK_MODE:-}" ]]; then
            printf 'export UV_LINK_MODE=%q\n' "${UV_LINK_MODE}"
        fi
    } >"${env_tmp}"
    chmod 600 "${env_tmp}"
    mv "${env_tmp}" "${ENV_FILE}"
}

sync_environment() {
    cd "${PROJECT_ROOT}"
    log "安装 Python ${PYTHON_VERSION}"
    "${UV_BIN}" python install "${PYTHON_VERSION}"
    log "环境目录：${UV_PROJECT_ENVIRONMENT}"
    log "uv cache：${UV_CACHE_DIR}"
    log "按照 uv.lock 同步依赖；大型 PyTorch/CUDA wheel 第一次需要数分钟"
    "${UV_BIN}" sync --frozen --python "${PYTHON_VERSION}"
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
    log "检查四张 RTX PRO 6000 Blackwell、CUDA 运行库、PyTorch 和 JAX 设备"
    command -v nvidia-smi >/dev/null 2>&1 || fail "找不到 nvidia-smi，无法验证四卡环境"
    local -a gpu_rows
    mapfile -t gpu_rows < <(
        nvidia-smi --query-gpu=name,driver_version --format=csv,noheader,nounits
    )
    ((${#gpu_rows[@]} == 4)) || fail "需要恰好四张 RTX PRO 6000 Blackwell，nvidia-smi 报告 ${#gpu_rows[@]} 张 GPU"
    local row name driver oldest
    for row in "${gpu_rows[@]}"; do
        name="${row%,*}"
        driver="${row##*,}"
        name="${name#"${name%%[![:space:]]*}"}"
        name="${name%"${name##*[![:space:]]}"}"
        driver="${driver#"${driver%%[![:space:]]*}"}"
        driver="${driver%"${driver##*[![:space:]]}"}"
        [[ "${name}" == *"RTX PRO 6000 Blackwell Server Edition"* ]] || \
            fail "GPU 不是 RTX PRO 6000 Blackwell Server Edition：${name}"
        [[ "${driver}" =~ ^[0-9]+([.][0-9]+)*$ ]] || fail "无法解析 NVIDIA driver 版本：${driver}"
        oldest="$(printf '%s\n' "570.86" "${driver}" | sort -V | head -n 1)"
        [[ "${oldest}" == "570.86" ]] || fail "NVIDIA driver ${driver} 低于最低要求 570.86"
        log "GPU=${name} driver=${driver}"
    done

    "${UV_BIN}" run --no-sync python - <<'PY'
import ctypes
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import torch
from jax._src.lib import cuda_versions
from jax.sharding import Mesh, NamedSharding, PartitionSpec


def find_nvidia_library(package_name: str, library_name: str) -> Path:
    module = __import__(package_name, fromlist=["__path__"])
    for package_path in module.__path__:
        candidate = Path(package_path) / "lib" / library_name
        if candidate.is_file():
            return candidate
    raise RuntimeError(f"找不到 {library_name}（包 {package_name}）")


cuda_version = cuda_versions.cuda_runtime_get_version()
cudnn_version = cuda_versions.cudnn_get_version()
if cuda_version < 12010:
    raise RuntimeError(f"CUDA runtime 需要 >=12.1，当前编码版本为 {cuda_version}")
if cudnn_version < 90800:
    raise RuntimeError(f"cuDNN 需要 >=9.8，当前编码版本为 {cudnn_version}")

nccl_path = find_nvidia_library("nvidia.nccl", "libnccl.so.2")
nccl = ctypes.CDLL(str(nccl_path))
nccl_version_value = ctypes.c_int()
if nccl.ncclGetVersion(ctypes.byref(nccl_version_value)) != 0:
    raise RuntimeError("ncclGetVersion 调用失败")
if nccl_version_value.value < 21800:
    raise RuntimeError(f"NCCL 需要 >=2.18，当前编码版本为 {nccl_version_value.value}")

cuda_nvcc = __import__("nvidia.cuda_nvcc", fromlist=["__path__"])
libdevice_paths = [
    Path(package_path) / "nvvm" / "libdevice" / "libdevice.10.bc"
    for package_path in cuda_nvcc.__path__
]
if not any(path.is_file() for path in libdevice_paths):
    raise RuntimeError("JAX local-CUDA 环境找不到 libdevice.10.bc")

if not torch.cuda.is_available():
    raise RuntimeError("PyTorch 没有识别到 CUDA")
if torch.cuda.device_count() != 4:
    raise RuntimeError(f"PyTorch 必须识别四张 GPU，当前为 {torch.cuda.device_count()}")

gpu_devices = [device for device in jax.devices() if device.platform == "gpu"]
print(f"JAX devices: {gpu_devices}")
if len(gpu_devices) != 4:
    raise RuntimeError(f"JAX 必须识别四张 GPU，当前为 {len(gpu_devices)}")

host = np.arange(16, dtype=np.float32).reshape(4, 4)
mesh = Mesh(np.asarray(gpu_devices), ("data",))
sharding = NamedSharding(mesh, PartitionSpec("data", None))
sharded = jax.device_put(host, sharding)
actual = np.asarray(jax.device_get(jnp.sum(sharded, axis=0)))
np.testing.assert_allclose(actual, host.sum(axis=0))
print(
    f"CUDA={cuda_version} cuDNN={cudnn_version} NCCL={nccl_version_value.value} "
    f"libdevice={next(path for path in libdevice_paths if path.is_file())}"
)
print(f"four-device sharded sum={actual.tolist()}")
PY
}

main() {
    acquire_project_lock
    install_system_dependencies
    install_uv
    configure_uv_storage
    configure_runtime_storage
    sync_environment
    verify_python_environment
    check_gpu
    write_environment_file

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
    echo "先生成共享离线缓存："
    echo "  bash ${PROJECT_ROOT}/scripts/precompute_smolvla_training_cache.sh"
    echo "并行启动 K8(0,1) 与 K21(2,3)："
    echo "  bash ${PROJECT_ROOT}/scripts/start_vtsmolvla_dual_train.sh"
}

main "$@"
