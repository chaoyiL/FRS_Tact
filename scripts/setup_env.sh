#!/usr/bin/env bash
# FRS_Tact 统一环境安装脚本。请在仓库根目录执行。
#
# 可用方式：
#
#   1. 不传参数：保持原有默认行为，同时安装以下两套环境：
#      - SmolVLA/FRS 环境（仓库根目录的 pyproject.toml 和 uv.lock）；
#      - 官方 LeRobot SmolVLA PyTorch 训练环境；
#      - Pi0.5 部署环境（deploy_pi05/pyproject.toml 和 uv.lock）。
#      此方式不会安装纯视觉 Pi0.5 训练环境。
#
#      bash scripts/setup_env.sh
#
#   2. --smolvla：安装两套互相隔离的 SmolVLA 环境，均使用 Python 3.12。
#      FRS_Tact 根环境供以下内容共用：
#      - train_smolvla_frs：SmolVLA-FRS 训练及缓存生成；
#      - deploy_smolvla：SmolVLA 和 SmolVLA-FRS 部署；
#      - 仓库根目录下依赖 SmolVLA/FRS 的下载、转换和检查工具。
#      官方 LeRobot 环境只供 train_smolvla 的 PyTorch 纯视觉训练使用。
#
#      bash scripts/setup_env.sh --smolvla
#
#   3. --pi05_deploy：只安装 Pi0.5 部署环境，使用 Python 3.12。
#      该环境对应 deploy_pi05，供纯视觉 Pi0.5 和 Pi0.5-FRS 部署使用。
#
#      bash scripts/setup_env.sh --pi05_deploy
#
#   4. --pi05_train：安装两套职责分离的环境：
#      - 纯视觉 Pi0.5 JAX 训练环境，使用 Python 3.11，对应 train_pi05；
#      - LeRobot 数据下载/转换环境，使用 Python 3.12，对应 data_tools。
#      Pi0.5 训练项目不包含 AnyTouch 或触觉训练分支。download_data.sh 会自动
#      使用 Python 3.12 数据环境，不会错误地调用 Python 3.11 训练环境。
#
#      bash scripts/setup_env.sh --pi05_train
#
#   5. --pi05_frs_train：只安装 Pi0.5 FRS 训练环境，使用 Python 3.12，
#      对应 train_pi05_frs。该环境独立于纯视觉 Pi0.5 训练和 Pi0.5 部署。
#
#      bash scripts/setup_env.sh --pi05_frs_train
#
#   6. 查看帮助，不执行安装：
#      bash scripts/setup_env.sh --help
#
# 每次安装会完成：
#   - 检查并安装缺少的系统工具、uv 和所需 Python 版本；
#   - 用所选项目的依赖配置创建或同步独立虚拟环境；
#   - 导入关键 Python 包，并检查 NVIDIA GPU、PyTorch/JAX 设备；
#   - 生成仓库根目录的 env_path，记录每套环境唯一的 Python 路径；启动和下载脚本会自动读取。
#
# 默认虚拟环境目录：
#   普通路径：
#     SmolVLA/FRS   <仓库根目录>/.venv
#     SmolVLA Torch <仓库根目录>/.venv-smolvla-torch
#     Pi0.5 部署    <仓库根目录>/deploy_pi05/.venv
#     Pi0.5 训练    <仓库根目录>/train_pi05/.venv
#     Pi0.5 FRS训练 <仓库根目录>/train_pi05_frs/.venv
#     数据下载/转换 <仓库根目录>/data_tools/.venv
#   当仓库位于 /workspace 下时：
#     SmolVLA/FRS   /workspace/venvs/frs_tact
#     SmolVLA Torch /workspace/venvs/smolvla_torch
#     Pi0.5 部署    /workspace/venvs/pi05_deploy
#     Pi0.5 训练    /workspace/venvs/pi05_train
#     Pi0.5 FRS训练 /workspace/venvs/pi05_frs_train
#     数据下载/转换 /workspace/venvs/lerobot_data_tools

set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
PYTHON_VERSION="3.12"
SMOLVLA_TORCH_VERSION="${SMOLVLA_TORCH_VERSION:-0.6.1}"
SMOLVLA_TORCH_VERSION_PYTORCH="${SMOLVLA_TORCH_VERSION_PYTORCH:-2.7.1}"
SMOLVLA_TORCH_VERSION_TORCHVISION="${SMOLVLA_TORCH_VERSION_TORCHVISION:-0.22.1}"
SMOLVLA_TORCHCODEC_VERSION="${SMOLVLA_TORCHCODEC_VERSION:-0.5.0}"
# train_pi05 的依赖约束与另外两套项目不同，因此固定使用独立的 Python 3.11 环境。
PI05_TRAIN_PYTHON_VERSION="3.11"
ENV_FILE="${PROJECT_ROOT}/env_path"

if [[ "${PROJECT_ROOT}" == /workspace/* ]]; then
    DEFAULT_WORKSPACE_ROOT="/workspace"
else
    DEFAULT_WORKSPACE_ROOT="${PROJECT_ROOT}/.cache"
fi
WORKSPACE_ROOT_VALUE="${FRS_WORKSPACE_ROOT:-${DEFAULT_WORKSPACE_ROOT}}"
if [[ "${PROJECT_ROOT}" == /workspace/* ]]; then
    DEFAULT_VENV_DIR="${WORKSPACE_ROOT_VALUE}/venvs/frs_tact"
    DEFAULT_SMOLVLA_TORCH_VENV_DIR="${WORKSPACE_ROOT_VALUE}/venvs/smolvla_torch"
    DEFAULT_PI05_VENV_DIR="${WORKSPACE_ROOT_VALUE}/venvs/pi05_deploy"
    DEFAULT_PI05_TRAIN_VENV_DIR="${WORKSPACE_ROOT_VALUE}/venvs/pi05_train"
    DEFAULT_PI05_FRS_TRAIN_VENV_DIR="${WORKSPACE_ROOT_VALUE}/venvs/pi05_frs_train"
    DEFAULT_DATA_TOOLS_VENV_DIR="${WORKSPACE_ROOT_VALUE}/venvs/lerobot_data_tools"
    DEFAULT_UV_CACHE_DIR="${WORKSPACE_ROOT_VALUE}/uv-cache"
    DEFAULT_STORAGE_ROOT="${WORKSPACE_ROOT_VALUE}"
else
    DEFAULT_VENV_DIR="${PROJECT_ROOT}/.venv"
    DEFAULT_SMOLVLA_TORCH_VENV_DIR="${PROJECT_ROOT}/.venv-smolvla-torch"
    DEFAULT_PI05_VENV_DIR="${PROJECT_ROOT}/deploy_pi05/.venv"
    DEFAULT_PI05_TRAIN_VENV_DIR="${PROJECT_ROOT}/train_pi05/.venv"
    DEFAULT_PI05_FRS_TRAIN_VENV_DIR="${PROJECT_ROOT}/train_pi05_frs/.venv"
    DEFAULT_DATA_TOOLS_VENV_DIR="${PROJECT_ROOT}/data_tools/.venv"
    DEFAULT_UV_CACHE_DIR="${HOME}/.cache/uv"
    DEFAULT_STORAGE_ROOT="${PROJECT_ROOT}/.cache"
fi
VENV_DIR="${FRS_VENV_DIR:-${DEFAULT_VENV_DIR}}"
SMOLVLA_TORCH_VENV_DIR="${SMOLVLA_TORCH_VENV_DIR:-${DEFAULT_SMOLVLA_TORCH_VENV_DIR}}"
PI05_PROJECT_ROOT="${PROJECT_ROOT}/deploy_pi05"
PI05_VENV_DIR="${PI05_VENV_DIR:-${DEFAULT_PI05_VENV_DIR}}"
PI05_TRAIN_PROJECT_ROOT="${PROJECT_ROOT}/train_pi05"
PI05_TRAIN_VENV_DIR="${PI05_TRAIN_VENV_DIR:-${DEFAULT_PI05_TRAIN_VENV_DIR}}"
PI05_FRS_TRAIN_PROJECT_ROOT="${PROJECT_ROOT}/train_pi05_frs"
PI05_FRS_TRAIN_VENV_DIR="${PI05_FRS_TRAIN_VENV_DIR:-${DEFAULT_PI05_FRS_TRAIN_VENV_DIR}}"
DATA_TOOLS_PROJECT_ROOT="${PROJECT_ROOT}/data_tools"
DATA_TOOLS_VENV_DIR="${DATA_TOOLS_VENV_DIR:-${DEFAULT_DATA_TOOLS_VENV_DIR}}"
UV_CACHE_DIR_VALUE="${UV_CACHE_DIR:-${DEFAULT_UV_CACHE_DIR}}"
STORAGE_ROOT="${FRS_STORAGE_ROOT:-${DEFAULT_STORAGE_ROOT}}"
HF_HOME_VALUE="${STORAGE_ROOT}/huggingface"
HF_HUB_CACHE_VALUE="${HF_HOME_VALUE}/hub"
HF_DATASETS_CACHE_VALUE="${HF_HOME_VALUE}/datasets_arrow"
HF_LEROBOT_HOME_VALUE="${HF_HOME_VALUE}/lerobot"
OPENPI_DATA_HOME_VALUE="${STORAGE_ROOT}/openpi-cache"
TMPDIR_VALUE="${STORAGE_ROOT}/tmp"
UV_BIN=""
# all 是无参数时的兼容模式：只包含 SmolVLA 和 Pi0.5 部署。
# 两套 Pi0.5 训练环境必须显式选择；默认安装不包含纯视觉或 FRS 训练环境。
SETUP_MODE="all"
SHOW_HELP=0

usage() {
    cat <<'EOF'
用法：bash scripts/setup_env.sh [--smolvla | --pi05_deploy | --pi05_train | --pi05_frs_train]

不传参数       同时安装 SmolVLA 环境和 Pi0.5 部署环境；不安装 Pi0.5 训练环境
--smolvla      安装 FRS_Tact 环境及独立的官方 LeRobot SmolVLA 训练环境（Python 3.12）
--pi05_deploy  只安装 Pi0.5 与 Pi0.5-FRS 部署环境（Python 3.12）
--pi05_train   安装纯视觉 Pi0.5 JAX 训练环境（Python 3.11）和数据转换环境（Python 3.12）
--pi05_frs_train 只安装 Pi0.5 FRS 训练环境（Python 3.12）
-h, --help     只显示本帮助，不执行安装

四个显式参数不能组合使用。
EOF
}

usage_error() {
    echo "[setup] 错误：$*" >&2
    usage >&2
    return 2
}

parse_args() {
    local selected=""
    SHOW_HELP=0
    while (($#)); do
        case "$1" in
            -h|--help)
                if [[ -n "${selected}" ]] || (($# != 1)); then
                    usage_error "--help 不能与其他参数一起使用"
                    return $?
                fi
                SHOW_HELP=1
                return 0
                ;;
            --smolvla)
                if [[ -n "${selected}" ]]; then
                    usage_error "一次只能选择一个安装环境"
                    return $?
                fi
                selected="smolvla"
                ;;
            --pi05_deploy)
                if [[ -n "${selected}" ]]; then
                    usage_error "一次只能选择一个安装环境"
                    return $?
                fi
                selected="pi05_deploy"
                ;;
            --pi05_train)
                if [[ -n "${selected}" ]]; then
                    usage_error "一次只能选择一个安装环境"
                    return $?
                fi
                selected="pi05_train"
                ;;
            --pi05_frs_train)
                if [[ -n "${selected}" ]]; then
                    usage_error "一次只能选择一个安装环境"
                    return $?
                fi
                selected="pi05_frs_train"
                ;;
            *)
                usage_error "未知参数：$1"
                return $?
                ;;
        esac
        shift
    done
    SETUP_MODE="${selected:-all}"
}

should_setup_smolvla() {
    [[ "${SETUP_MODE}" == "all" || "${SETUP_MODE}" == "smolvla" ]]
}

should_setup_pi05() {
    [[ "${SETUP_MODE}" == "all" || "${SETUP_MODE}" == "pi05_deploy" ]]
}

should_setup_pi05_train() {
    [[ "${SETUP_MODE}" == "pi05_train" ]]
}

should_setup_pi05_frs_train() {
    [[ "${SETUP_MODE}" == "pi05_frs_train" ]]
}

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

validate_environment_targets() {
    # 相对路径统一按仓库根目录解析，避免从不同工作目录执行时生成不同环境。
    if [[ "${VENV_DIR}" != /* ]]; then
        VENV_DIR="${PROJECT_ROOT}/${VENV_DIR}"
    fi
    if [[ "${SMOLVLA_TORCH_VENV_DIR}" != /* ]]; then
        SMOLVLA_TORCH_VENV_DIR="${PROJECT_ROOT}/${SMOLVLA_TORCH_VENV_DIR}"
    fi
    if [[ "${PI05_VENV_DIR}" != /* ]]; then
        PI05_VENV_DIR="${PROJECT_ROOT}/${PI05_VENV_DIR}"
    fi
    if [[ "${PI05_TRAIN_VENV_DIR}" != /* ]]; then
        PI05_TRAIN_VENV_DIR="${PROJECT_ROOT}/${PI05_TRAIN_VENV_DIR}"
    fi
    if [[ "${PI05_FRS_TRAIN_VENV_DIR}" != /* ]]; then
        PI05_FRS_TRAIN_VENV_DIR="${PROJECT_ROOT}/${PI05_FRS_TRAIN_VENV_DIR}"
    fi
    if [[ "${DATA_TOOLS_VENV_DIR}" != /* ]]; then
        DATA_TOOLS_VENV_DIR="${PROJECT_ROOT}/${DATA_TOOLS_VENV_DIR}"
    fi
    VENV_DIR="$(realpath -m -- "${VENV_DIR}")"
    SMOLVLA_TORCH_VENV_DIR="$(realpath -m -- "${SMOLVLA_TORCH_VENV_DIR}")"
    PI05_VENV_DIR="$(realpath -m -- "${PI05_VENV_DIR}")"
    PI05_TRAIN_VENV_DIR="$(realpath -m -- "${PI05_TRAIN_VENV_DIR}")"
    PI05_FRS_TRAIN_VENV_DIR="$(realpath -m -- "${PI05_FRS_TRAIN_VENV_DIR}")"
    DATA_TOOLS_VENV_DIR="$(realpath -m -- "${DATA_TOOLS_VENV_DIR}")"
    # FRS_Tact 包含同名的精简 lerobot 包，不能与官方 LeRobot 安装在同一环境。
    if [[ "${VENV_DIR}" == "${SMOLVLA_TORCH_VENV_DIR}" ]]; then
        fail "FRS_Tact 与官方 LeRobot SmolVLA 训练必须使用不同环境：${VENV_DIR}"
    fi
    # 各项目的 Python 版本和依赖约束并不完全兼容，禁止复用同一个虚拟环境。
    if [[ "${VENV_DIR}" == "${PI05_VENV_DIR}" ]]; then
        fail "SmolVLA 与 Pi0.5 部署必须使用不同的虚拟环境目录：${VENV_DIR}"
    fi
    if [[ "${SMOLVLA_TORCH_VENV_DIR}" == "${PI05_VENV_DIR}" || \
          "${SMOLVLA_TORCH_VENV_DIR}" == "${PI05_TRAIN_VENV_DIR}" ]]; then
        fail "官方 LeRobot SmolVLA 训练环境不能与 Pi0.5 共用：${SMOLVLA_TORCH_VENV_DIR}"
    fi
    if [[ "${PI05_TRAIN_VENV_DIR}" == "${VENV_DIR}" || \
          "${PI05_TRAIN_VENV_DIR}" == "${PI05_VENV_DIR}" ]]; then
        fail "Pi0.5 训练必须使用独立的虚拟环境目录：${PI05_TRAIN_VENV_DIR}"
    fi
    if [[ "${PI05_FRS_TRAIN_VENV_DIR}" == "${VENV_DIR}" || \
          "${PI05_FRS_TRAIN_VENV_DIR}" == "${SMOLVLA_TORCH_VENV_DIR}" || \
          "${PI05_FRS_TRAIN_VENV_DIR}" == "${PI05_VENV_DIR}" || \
          "${PI05_FRS_TRAIN_VENV_DIR}" == "${PI05_TRAIN_VENV_DIR}" || \
          "${PI05_FRS_TRAIN_VENV_DIR}" == "${DATA_TOOLS_VENV_DIR}" ]]; then
        fail "Pi0.5 FRS 训练必须使用独立的虚拟环境目录：${PI05_FRS_TRAIN_VENV_DIR}"
    fi
    if [[ "${DATA_TOOLS_VENV_DIR}" == "${PI05_TRAIN_VENV_DIR}" || \
          "${DATA_TOOLS_VENV_DIR}" == "${VENV_DIR}" || \
          "${DATA_TOOLS_VENV_DIR}" == "${PI05_VENV_DIR}" || \
          "${DATA_TOOLS_VENV_DIR}" == "${SMOLVLA_TORCH_VENV_DIR}" ]]; then
        fail "LeRobot 数据工具必须使用独立的 Python 3.12 环境：${DATA_TOOLS_VENV_DIR}"
    fi
}

configure_uv_storage() {
    validate_environment_targets
    mkdir -p "${UV_CACHE_DIR_VALUE}"
    if should_setup_smolvla; then
        mkdir -p "$(dirname -- "${VENV_DIR}")"
        mkdir -p "$(dirname -- "${SMOLVLA_TORCH_VENV_DIR}")"
    fi
    if should_setup_pi05; then
        mkdir -p "$(dirname -- "${PI05_VENV_DIR}")"
    fi
    if should_setup_pi05_train; then
        mkdir -p "$(dirname -- "${PI05_TRAIN_VENV_DIR}")"
        mkdir -p "$(dirname -- "${DATA_TOOLS_VENV_DIR}")"
        export UV_PROJECT_ENVIRONMENT="${PI05_TRAIN_VENV_DIR}"
    elif should_setup_pi05_frs_train; then
        mkdir -p "$(dirname -- "${PI05_FRS_TRAIN_VENV_DIR}")"
        export UV_PROJECT_ENVIRONMENT="${PI05_FRS_TRAIN_VENV_DIR}"
    else
        export UV_PROJECT_ENVIRONMENT="${VENV_DIR}"
    fi
    export UV_CACHE_DIR="${UV_CACHE_DIR_VALUE}"

    local cache_device env_device smolvla_torch_env_device pi05_env_device pi05_train_env_device pi05_frs_train_env_device data_tools_env_device needs_copy=0
    cache_device="$(stat -c '%d' "${UV_CACHE_DIR_VALUE}")"
    if should_setup_smolvla; then
        env_device="$(stat -c '%d' "$(dirname -- "${VENV_DIR}")")"
        [[ "${env_device}" == "${cache_device}" ]] || needs_copy=1
        smolvla_torch_env_device="$(stat -c '%d' "$(dirname -- "${SMOLVLA_TORCH_VENV_DIR}")")"
        [[ "${smolvla_torch_env_device}" == "${cache_device}" ]] || needs_copy=1
    fi
    if should_setup_pi05; then
        pi05_env_device="$(stat -c '%d' "$(dirname -- "${PI05_VENV_DIR}")")"
        [[ "${pi05_env_device}" == "${cache_device}" ]] || needs_copy=1
    fi
    if should_setup_pi05_train; then
        pi05_train_env_device="$(stat -c '%d' "$(dirname -- "${PI05_TRAIN_VENV_DIR}")")"
        [[ "${pi05_train_env_device}" == "${cache_device}" ]] || needs_copy=1
        data_tools_env_device="$(stat -c '%d' "$(dirname -- "${DATA_TOOLS_VENV_DIR}")")"
        [[ "${data_tools_env_device}" == "${cache_device}" ]] || needs_copy=1
    fi
    if should_setup_pi05_frs_train; then
        pi05_frs_train_env_device="$(stat -c '%d' "$(dirname -- "${PI05_FRS_TRAIN_VENV_DIR}")")"
        [[ "${pi05_frs_train_env_device}" == "${cache_device}" ]] || needs_copy=1
    fi
    if ((needs_copy)); then
        export UV_LINK_MODE="copy"
        warn "uv cache 和虚拟环境不在同一文件系统，使用 UV_LINK_MODE=copy"
    fi
}

configure_runtime_storage() {
    mkdir -p \
        "${HF_HUB_CACHE_VALUE}" \
        "${HF_DATASETS_CACHE_VALUE}" \
        "${HF_LEROBOT_HOME_VALUE}" \
        "${OPENPI_DATA_HOME_VALUE}" \
        "${TMPDIR_VALUE}"
    export WORKSPACE_ROOT="${WORKSPACE_ROOT_VALUE}"
    export HF_HOME="${HF_HOME_VALUE}"
    export HF_HUB_CACHE="${HF_HUB_CACHE_VALUE}"
    export HF_DATASETS_CACHE="${HF_DATASETS_CACHE_VALUE}"
    export HF_LEROBOT_HOME="${HF_LEROBOT_HOME_VALUE}"
    export OPENPI_DATA_HOME="${OPENPI_DATA_HOME_VALUE}"
    export TMPDIR="${TMPDIR_VALUE}"
}

write_environment_file() {
    # env_path 是所有项目解释器和缓存路径的统一索引，不等同于激活某一个 venv。
    # 启动/下载脚本会自动读取对应的 *_PYTHON；手动执行工具时可 source 一次后
    # 显式调用所需解释器，因此 /workspace 和普通目录使用完全相同的命令结构。
    {
        echo "# 由 setup_env.sh 生成；供训练脚本复用。"
        printf 'export PATH=%q\n' "${HOME}/.local/bin:${HOME}/.cargo/bin:${PATH}"
        printf 'export UV_PROJECT_ENVIRONMENT=%q\n' "${UV_PROJECT_ENVIRONMENT}"
        printf 'export FRS_PYTHON=%q\n' "${VENV_DIR}/bin/python"
        printf 'export SMOLVLA_TORCH_PYTHON=%q\n' "${SMOLVLA_TORCH_VENV_DIR}/bin/python"
        printf 'export PI05_PYTHON=%q\n' "${PI05_VENV_DIR}/bin/python"
        printf 'export PI05_FRS_PYTHON=%q\n' "${PI05_VENV_DIR}/bin/python"
        printf 'export TRAIN_PI05_PYTHON=%q\n' "${PI05_TRAIN_VENV_DIR}/bin/python"
        printf 'export TRAIN_PI05_FRS_PYTHON=%q\n' "${PI05_FRS_TRAIN_VENV_DIR}/bin/python"
        printf 'export DATA_TOOL_PYTHON=%q\n' "${DATA_TOOLS_VENV_DIR}/bin/python"
        printf 'export UV_CACHE_DIR=%q\n' "${UV_CACHE_DIR}"
        printf 'export WORKSPACE_ROOT=%q\n' "${WORKSPACE_ROOT}"
        printf 'export HF_HOME=%q\n' "${HF_HOME}"
        printf 'export HF_HUB_CACHE=%q\n' "${HF_HUB_CACHE}"
        printf 'export HF_DATASETS_CACHE=%q\n' "${HF_DATASETS_CACHE}"
        printf 'export HF_LEROBOT_HOME=%q\n' "${HF_LEROBOT_HOME}"
        printf 'export OPENPI_DATA_HOME=%q\n' "${OPENPI_DATA_HOME}"
        printf 'export TMPDIR=%q\n' "${TMPDIR}"
        printf 'export FRS_PROJECT_ROOT=%q\n' "${PROJECT_ROOT}"
        printf 'source %q\n' "${PROJECT_ROOT}/scripts/env_shortcuts.sh"
        if [[ -n "${UV_LINK_MODE:-}" ]]; then
            printf 'export UV_LINK_MODE=%q\n' "${UV_LINK_MODE}"
        fi
    } >"${ENV_FILE}"
    chmod 600 "${ENV_FILE}"
}

validate_selected_projects() {
    if should_setup_smolvla; then
        [[ -f "${PROJECT_ROOT}/pyproject.toml" ]] || fail "缺少 SmolVLA 项目配置：${PROJECT_ROOT}/pyproject.toml"
        [[ -f "${PROJECT_ROOT}/uv.lock" ]] || fail "缺少 SmolVLA 依赖锁文件：${PROJECT_ROOT}/uv.lock"
    fi
    if should_setup_pi05; then
        [[ -f "${PI05_PROJECT_ROOT}/pyproject.toml" ]] || \
            fail "缺少 Pi0.5 部署项目：${PI05_PROJECT_ROOT}/pyproject.toml"
        [[ -f "${PI05_PROJECT_ROOT}/uv.lock" ]] || \
            fail "缺少 Pi0.5 部署锁文件：${PI05_PROJECT_ROOT}/uv.lock"
    fi
    if should_setup_pi05_train; then
        [[ -f "${PI05_TRAIN_PROJECT_ROOT}/pyproject.toml" ]] || \
            fail "缺少 Pi0.5 训练项目：${PI05_TRAIN_PROJECT_ROOT}/pyproject.toml"
        [[ -f "${DATA_TOOLS_PROJECT_ROOT}/pyproject.toml" ]] || \
            fail "缺少 LeRobot 数据工具项目：${DATA_TOOLS_PROJECT_ROOT}/pyproject.toml"
    fi
    if should_setup_pi05_frs_train; then
        [[ -f "${PI05_FRS_TRAIN_PROJECT_ROOT}/pyproject.toml" ]] || \
            fail "缺少 Pi0.5 FRS 训练项目：${PI05_FRS_TRAIN_PROJECT_ROOT}/pyproject.toml"
        [[ -f "${PI05_FRS_TRAIN_PROJECT_ROOT}/uv.lock" ]] || \
            fail "缺少 Pi0.5 FRS 训练锁文件：${PI05_FRS_TRAIN_PROJECT_ROOT}/uv.lock"
    fi
}

install_python() {
    cd "${PROJECT_ROOT}"
    check_existing_uv_processes
    if should_setup_smolvla || should_setup_pi05 || should_setup_pi05_frs_train; then
        log "安装 Python ${PYTHON_VERSION}"
        "${UV_BIN}" python install "${PYTHON_VERSION}"
    fi
    if should_setup_pi05_train; then
        log "为数据下载/转换安装 Python ${PYTHON_VERSION}"
        "${UV_BIN}" python install "${PYTHON_VERSION}"
        log "为纯视觉 Pi0.5 训练安装 Python ${PI05_TRAIN_PYTHON_VERSION}"
        "${UV_BIN}" python install "${PI05_TRAIN_PYTHON_VERSION}"
    fi
}

sync_root_environment() {
    cd "${PROJECT_ROOT}"
    log "SmolVLA 环境目录：${VENV_DIR}"
    log "uv cache：${UV_CACHE_DIR}"
    log "按照仓库根目录 uv.lock 同步 SmolVLA/SmolVLA-FRS 环境"
    UV_PROJECT_ENVIRONMENT="${VENV_DIR}" \
        "${UV_BIN}" sync --frozen --python "${PYTHON_VERSION}"
}

sync_smolvla_torch_environment() {
    local smolvla_python="${SMOLVLA_TORCH_VENV_DIR}/bin/python"
    log "官方 LeRobot SmolVLA 训练环境目录：${SMOLVLA_TORCH_VENV_DIR}"
    if [[ ! -x "${smolvla_python}" ]]; then
        "${UV_BIN}" venv --python "${PYTHON_VERSION}" "${SMOLVLA_TORCH_VENV_DIR}"
    fi
    # 先固定安装 CUDA 12.8 PyTorch，避免 PyPI 解析到 CPU wheel。
    "${UV_BIN}" pip install --python "${smolvla_python}" \
        "torch==${SMOLVLA_TORCH_VERSION_PYTORCH}" \
        "torchvision==${SMOLVLA_TORCH_VERSION_TORCHVISION}" \
        --index https://download.pytorch.org/whl/cu128
    "${UV_BIN}" pip install --python "${smolvla_python}" \
        "lerobot[training,smolvla,peft]==${SMOLVLA_TORCH_VERSION}" \
        "websockets>=13.0,<16.0" \
        "msgpack>=1.0.0,<2.0.0"
    # torchcodec 0.5 is the newest release compatible with torch 2.7.
    "${UV_BIN}" pip install --python "${smolvla_python}" \
        "torchcodec==${SMOLVLA_TORCHCODEC_VERSION}"
}

verify_smolvla_torch_environment() {
    local smolvla_python="${SMOLVLA_TORCH_VENV_DIR}/bin/python"
    [[ -x "${smolvla_python}" ]] || fail "SmolVLA PyTorch Python 不可执行：${smolvla_python}"
    log "验证官方 LeRobot ${SMOLVLA_TORCH_VERSION}、SmolVLA、PEFT 与 CUDA"
    (
        cd /
        FRS_EXPECT_GPU="$([[ -x "$(command -v nvidia-smi 2>/dev/null || true)" ]] && echo 1 || echo 0)" \
            FRS_SMOLVLA_TORCH_VERSION="${SMOLVLA_TORCH_VERSION_PYTORCH}" \
            FRS_SMOLVLA_TORCHCODEC_VERSION="${SMOLVLA_TORCHCODEC_VERSION}" \
            "${smolvla_python}" - <<'PY'
import os
from importlib.metadata import version
from packaging.version import Version

import lerobot
import torch
import torchcodec
from lerobot.scripts import lerobot_train
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig

del lerobot_train, SmolVLAConfig
print(f"official lerobot={lerobot.__version__}")
print(f"torchcodec={getattr(torchcodec, '__version__', 'installed')}")
print(f"SmolVLA PyTorch CUDA available: {torch.cuda.is_available()}")
expected_torch = os.environ["FRS_SMOLVLA_TORCH_VERSION"]
expected_torchcodec = os.environ["FRS_SMOLVLA_TORCHCODEC_VERSION"]
if torch.__version__.split("+", 1)[0] != expected_torch:
    raise RuntimeError(f"需要 torch {expected_torch}，当前为 {torch.__version__}")
if Version(version("torchcodec")) != Version(expected_torchcodec):
    raise RuntimeError(
        f"需要 torchcodec {expected_torchcodec}，当前为 {version('torchcodec')}"
    )
if os.environ.get("FRS_EXPECT_GPU") == "1" and not torch.cuda.is_available():
    raise RuntimeError("nvidia-smi 可用，但 SmolVLA PyTorch 环境没有识别到 CUDA")
PY
    )
}

sync_pi05_environment() {
    log "Pi0.5 部署环境目录：${PI05_VENV_DIR}"
    log "按照 deploy_pi05/uv.lock 同步独立 Pi0.5 环境"
    UV_PROJECT_ENVIRONMENT="${PI05_VENV_DIR}" \
        "${UV_BIN}" sync --frozen --python "${PYTHON_VERSION}" \
        --project "${PI05_PROJECT_ROOT}"
}

sync_pi05_train_environment() {
    log "Pi0.5 训练环境目录：${PI05_TRAIN_VENV_DIR}"
    log "同步纯视觉 Pi0.5 JAX 训练项目（显示详细下载进度）"
    # train_pi05 当前没有要求预先存在 uv.lock；首次同步时允许 uv 解析并生成锁文件。
    # 网络较慢时增加读取超时和重试次数；FRS_PYPI_MIRROR 可临时指定 PyPI 镜像，
    # 例如 https://pypi.tuna.tsinghua.edu.cn/simple。
    (
        local -a pi05_sync_indexes=()
        export UV_HTTP_CONNECT_TIMEOUT="${UV_HTTP_CONNECT_TIMEOUT:-15}"
        export UV_HTTP_TIMEOUT="${UV_HTTP_TIMEOUT:-120}"
        export UV_HTTP_RETRIES="${UV_HTTP_RETRIES:-8}"
        if [[ -n "${FRS_PYPI_MIRROR:-}" ]]; then
            export UV_DEFAULT_INDEX="${FRS_PYPI_MIRROR}"
            log "Pi0.5 训练使用 PyPI 镜像：${UV_DEFAULT_INDEX}"
        fi
        if [[ -n "${FRS_PYTORCH_INDEX:-}" ]]; then
            pi05_sync_indexes=(--index "pytorch-cpu=${FRS_PYTORCH_INDEX}")
            log "Pi0.5 训练使用 PyTorch CPU 镜像：${FRS_PYTORCH_INDEX}"
        fi
        UV_PROJECT_ENVIRONMENT="${PI05_TRAIN_VENV_DIR}" \
            "${UV_BIN}" sync --verbose --no-dev --python "${PI05_TRAIN_PYTHON_VERSION}" \
            --project "${PI05_TRAIN_PROJECT_ROOT}" "${pi05_sync_indexes[@]}"
    )
}

sync_pi05_frs_train_environment() {
    log "Pi0.5 FRS 训练环境目录：${PI05_FRS_TRAIN_VENV_DIR}"
    log "按照 train_pi05_frs/uv.lock 同步独立 Python 3.12 训练环境"
    (
        export UV_HTTP_CONNECT_TIMEOUT="${UV_HTTP_CONNECT_TIMEOUT:-15}"
        export UV_HTTP_TIMEOUT="${UV_HTTP_TIMEOUT:-120}"
        export UV_HTTP_RETRIES="${UV_HTTP_RETRIES:-8}"
        if [[ -n "${FRS_PYPI_MIRROR:-}" ]]; then
            export UV_DEFAULT_INDEX="${FRS_PYPI_MIRROR}"
            log "Pi0.5 FRS 训练使用 PyPI 镜像：${UV_DEFAULT_INDEX}"
        fi
        UV_PROJECT_ENVIRONMENT="${PI05_FRS_TRAIN_VENV_DIR}" \
            "${UV_BIN}" sync --frozen --python "${PYTHON_VERSION}" \
            --project "${PI05_FRS_TRAIN_PROJECT_ROOT}"
    )
}

sync_data_tools_environment() {
    log "LeRobot 数据工具环境目录：${DATA_TOOLS_VENV_DIR}"
    log "同步 Python 3.12 下载与 v2.1 -> v3.0 转换依赖"
    (
        export UV_HTTP_CONNECT_TIMEOUT="${UV_HTTP_CONNECT_TIMEOUT:-15}"
        export UV_HTTP_TIMEOUT="${UV_HTTP_TIMEOUT:-120}"
        export UV_HTTP_RETRIES="${UV_HTTP_RETRIES:-8}"
        if [[ -n "${FRS_PYPI_MIRROR:-}" ]]; then
            export UV_DEFAULT_INDEX="${FRS_PYPI_MIRROR}"
        fi
        UV_PROJECT_ENVIRONMENT="${DATA_TOOLS_VENV_DIR}" \
            "${UV_BIN}" sync --verbose --python "${PYTHON_VERSION}" \
            --project "${DATA_TOOLS_PROJECT_ROOT}"
    )
}

verify_data_tools_environment() {
    local data_python="${DATA_TOOLS_VENV_DIR}/bin/python"
    local data_hf="${DATA_TOOLS_VENV_DIR}/bin/hf"
    [[ -x "${data_python}" ]] || fail "LeRobot 数据工具 Python 不可执行：${data_python}"
    [[ -x "${data_hf}" ]] || fail "LeRobot 数据工具缺少 hf 命令：${data_hf}"
    log "验证 Python 3.12 LeRobot 数据下载与 v2.1 -> v3.0 转换环境"
    (
        cd "${PROJECT_ROOT}"
        PYTHONDONTWRITEBYTECODE=1 \
        PYTHONPATH="${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" \
            "${data_python}" - <<'PY'
import sys

import jsonlines
import torch
from lerobot.datasets.v30 import convert_dataset_v21_to_v30

del jsonlines, convert_dataset_v21_to_v30
if sys.version_info[:2] != (3, 12):
    raise RuntimeError(f"LeRobot data conversion requires Python 3.12, got {sys.version}")
print(f"LeRobot data tools python={sys.version.split()[0]}")
print(f"LeRobot data tools torch={torch.__version__}")
PY
        "${data_hf}" version
    )
}

# Preserve the source-level helper used by existing tests and shell consumers.
sync_environments() {
    validate_environment_targets
    validate_selected_projects
    install_python
    sync_root_environment
    sync_pi05_environment
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

verify_pi05_environment() {
    local pi05_python="${PI05_VENV_DIR}/bin/python"
    [[ -x "${pi05_python}" ]] || fail "Pi0.5 Python 不可执行：${pi05_python}"
    log "验证独立 Pi0.5 纯视觉与 FRS 部署依赖（不再执行 sync）"
    (
        cd "${PI05_PROJECT_ROOT}"
        PYTHONDONTWRITEBYTECODE=1 \
        PYTHONPATH="${PI05_PROJECT_ROOT}/src:${PI05_PROJECT_ROOT}:${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" \
            "${pi05_python}" - <<'PY'
import sys

import flax
import jax
import orbax.checkpoint
import transformers

import deploy_pi05.pi05_client
import deploy_pi05.remote_client

if sys.version_info[:2] != (3, 12):
    raise RuntimeError(f"Pi0.5 需要 Python 3.12，当前为 {sys.version}")
print(f"pi05 python={sys.version.split()[0]}")
print(f"pi05 jax={jax.__version__}")
print(f"pi05 flax={flax.__version__}")
print(f"pi05 orbax={orbax.checkpoint.__version__}")
print(f"pi05 transformers={transformers.__version__}")
PY
    )
}

verify_pi05_train_environment() {
    local train_python="${PI05_TRAIN_VENV_DIR}/bin/python"
    [[ -x "${train_python}" ]] || fail "Pi0.5 训练 Python 不可执行：${train_python}"
    log "验证纯视觉 Pi0.5 JAX 训练环境"
    (
        cd "${PI05_TRAIN_PROJECT_ROOT}"
        PYTHONDONTWRITEBYTECODE=1 \
        PYTHONPATH="${PI05_TRAIN_PROJECT_ROOT}/src:${PI05_TRAIN_PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" \
            "${train_python}" - <<'PY'
import sys

import flax
import jax
import orbax.checkpoint
import torch
import yaml

import openpi.training.config
import openpi.training.data_loader

if sys.version_info[:2] != (3, 11):
    raise RuntimeError(f"Pi0.5 training requires Python 3.11, got {sys.version}")
print(f"pi05 train python={sys.version.split()[0]}")
print(f"pi05 train jax={jax.__version__}")
print(f"pi05 train flax={flax.__version__}")
print(f"pi05 train torch={torch.__version__}")
PY
    )
}

verify_pi05_frs_train_environment() {
    local train_python="${PI05_FRS_TRAIN_VENV_DIR}/bin/python"
    [[ -x "${train_python}" ]] || \
        fail "Pi0.5 FRS 训练 Python 不可执行：${train_python}"
    log "验证独立 Pi0.5 FRS 训练依赖"
    (
        cd "${PI05_FRS_TRAIN_PROJECT_ROOT}"
        PYTHONDONTWRITEBYTECODE=1 \
        PYTHONPATH="${PI05_FRS_TRAIN_PROJECT_ROOT}/src:${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" \
            "${train_python}" - <<'PY'
import sys

import flax
import jax
import matplotlib
import orbax.checkpoint
import torch
import torchcodec
import transformers

from train_pi05_frs.utils.checkpoint import load_checkpoint

del load_checkpoint
if sys.version_info[:2] != (3, 12):
    raise RuntimeError(f"Pi0.5 FRS training requires Python 3.12, got {sys.version}")
print(f"pi05 frs train python={sys.version.split()[0]}")
print(f"pi05 frs train jax={jax.__version__}")
print(f"pi05 frs train flax={flax.__version__}")
print(f"pi05 frs train torch={torch.__version__} cuda={torch.version.cuda}")
print(f"pi05 frs train torchcodec={getattr(torchcodec, '__version__', 'installed')}")
print(f"pi05 frs train matplotlib={matplotlib.__version__}")
print(f"pi05 frs train transformers={transformers.__version__}")
PY
    )
}

check_root_gpu() {
    cd "${PROJECT_ROOT}"
    log "检查 SmolVLA 环境的 NVIDIA、PyTorch 和 JAX 设备"
    local expect_gpu=0
    if command -v nvidia-smi >/dev/null 2>&1; then
        expect_gpu=1
        nvidia-smi --query-gpu=index,name,driver_version,memory.total --format=csv,noheader
    else
        warn "没有找到 nvidia-smi；本机只能做 CPU 开发，不能用于正式训练"
    fi
    FRS_EXPECT_GPU="${expect_gpu}" UV_PROJECT_ENVIRONMENT="${VENV_DIR}" \
        "${UV_BIN}" run --no-sync python - <<'PY'
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

check_pi05_gpu() {
    log "检查 Pi0.5 环境 NVIDIA 和 JAX 设备"
    local expect_gpu=0
    if command -v nvidia-smi >/dev/null 2>&1; then
        expect_gpu=1
        nvidia-smi --query-gpu=index,name,driver_version,memory.total --format=csv,noheader
    else
        warn "没有找到 nvidia-smi；本机只能做 CPU 开发，不能用于正式训练"
    fi
    (
        cd "${PI05_PROJECT_ROOT}"
        FRS_EXPECT_GPU="${expect_gpu}" \
        PYTHONDONTWRITEBYTECODE=1 \
        PYTHONPATH="${PI05_PROJECT_ROOT}/src:${PI05_PROJECT_ROOT}:${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" \
            "${PI05_VENV_DIR}/bin/python" - <<'PY'
import os

import jax

devices = jax.devices()
print(f"Pi0.5 JAX devices: {devices}")
if os.environ.get("FRS_EXPECT_GPU") == "1" and not any(
    device.platform == "gpu" for device in devices
):
    raise RuntimeError("nvidia-smi 可用，但 Pi0.5 JAX 没有识别到 GPU")
PY
    )
}

check_pi05_train_gpu() {
    log "检查 Pi0.5 训练环境的 NVIDIA 和 JAX 设备"
    local expect_gpu=0
    if command -v nvidia-smi >/dev/null 2>&1; then
        expect_gpu=1
        nvidia-smi --query-gpu=index,name,driver_version,memory.total --format=csv,noheader
    else
        warn "没有找到 nvidia-smi；正式 Pi0.5 训练需要 NVIDIA GPU"
    fi
    (
        cd "${PI05_TRAIN_PROJECT_ROOT}"
        FRS_EXPECT_GPU="${expect_gpu}" \
        PYTHONDONTWRITEBYTECODE=1 \
        PYTHONPATH="${PI05_TRAIN_PROJECT_ROOT}/src:${PI05_TRAIN_PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" \
            "${PI05_TRAIN_VENV_DIR}/bin/python" - <<'PY'
import os
import jax

devices = jax.devices()
print(f"Pi0.5 training JAX devices: {devices}")
if os.environ.get("FRS_EXPECT_GPU") == "1" and not any(
    device.platform == "gpu" for device in devices
):
    raise RuntimeError("nvidia-smi is available, but Pi0.5 training JAX did not detect a GPU")
PY
    )
}

check_pi05_frs_train_gpu() {
    log "检查 Pi0.5 FRS 训练环境的 NVIDIA、JAX 和 PyTorch 设备"
    local expect_gpu=0
    if command -v nvidia-smi >/dev/null 2>&1; then
        expect_gpu=1
        nvidia-smi --query-gpu=index,name,driver_version,memory.total --format=csv,noheader
    else
        warn "没有找到 nvidia-smi；正式 Pi0.5 FRS 训练需要 NVIDIA GPU"
    fi
    (
        cd "${PI05_FRS_TRAIN_PROJECT_ROOT}"
        FRS_EXPECT_GPU="${expect_gpu}" \
        PYTHONDONTWRITEBYTECODE=1 \
        PYTHONPATH="${PI05_FRS_TRAIN_PROJECT_ROOT}/src:${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" \
            "${PI05_FRS_TRAIN_VENV_DIR}/bin/python" - <<'PY'
import os

import jax
import torch

devices = jax.devices()
print(f"Pi0.5 FRS training JAX devices: {devices}")
print(f"Pi0.5 FRS training PyTorch CUDA available: {torch.cuda.is_available()}")
if os.environ.get("FRS_EXPECT_GPU") == "1":
    if not any(device.platform == "gpu" for device in devices):
        raise RuntimeError("nvidia-smi is available, but Pi0.5 FRS JAX did not detect a GPU")
    if not torch.cuda.is_available():
        raise RuntimeError("nvidia-smi is available, but Pi0.5 FRS PyTorch did not detect CUDA")
PY
    )
}

print_summary() {
    if should_setup_pi05_train; then
        echo "Pi0.5 训练环境（已安装）：${PI05_TRAIN_VENV_DIR}"
        echo "LeRobot 数据工具环境（已安装）：${DATA_TOOLS_VENV_DIR}"
    else
        echo "Pi0.5 训练环境（本次未安装）：${PI05_TRAIN_VENV_DIR}"
        echo "LeRobot 数据工具环境（本次未安装）：${DATA_TOOLS_VENV_DIR}"
    fi
    if should_setup_pi05_frs_train; then
        echo "Pi0.5 FRS 训练环境（已安装）：${PI05_FRS_TRAIN_VENV_DIR}"
    else
        echo "Pi0.5 FRS 训练环境（本次未安装）：${PI05_FRS_TRAIN_VENV_DIR}"
    fi
    log "环境安装完成：${SETUP_MODE}"
    echo
    if should_setup_smolvla; then
        echo "SmolVLA/FRS 环境（已安装）：${VENV_DIR}"
        echo "SmolVLA PyTorch 环境（已安装）：${SMOLVLA_TORCH_VENV_DIR}"
    else
        echo "SmolVLA/FRS 环境（本次未安装）：${VENV_DIR}"
        echo "SmolVLA PyTorch 环境（本次未安装）：${SMOLVLA_TORCH_VENV_DIR}"
    fi
    if should_setup_pi05; then
        echo "Pi0.5 部署环境（已安装）：${PI05_VENV_DIR}"
    else
        echo "Pi0.5 部署环境（本次未安装）：${PI05_VENV_DIR}"
    fi
    echo "环境变量：${ENV_FILE}"
    echo "Workspace 根目录：${WORKSPACE_ROOT}"
    echo "Hugging Face 缓存：${HF_HOME}"
    echo "OpenPI 缓存：${OPENPI_DATA_HOME}"
    echo "Arrow 数据缓存：${HF_DATASETS_CACHE}"
    echo
    echo "首次使用时登录："
    echo "  cd ${PROJECT_ROOT}"
    echo "  source ${ENV_FILE}"
    if should_setup_pi05_train; then
        echo "  ${DATA_TOOLS_VENV_DIR}/bin/hf auth login"
        echo "  ${PI05_TRAIN_VENV_DIR}/bin/wandb login"
    elif should_setup_pi05_frs_train; then
        echo "  ${PI05_FRS_TRAIN_VENV_DIR}/bin/hf auth login"
    elif should_setup_smolvla; then
        echo "  ${VENV_DIR}/bin/hf auth login"
        echo "  ${VENV_DIR}/bin/wandb login"
    elif should_setup_pi05; then
        echo "  ${PI05_VENV_DIR}/bin/hf auth login"
    fi
    echo
    echo "一键启动视觉 SmolVLA："
    echo "  bash ${PROJECT_ROOT}/train_smolvla/scripts/start_smolvla_train.sh"
    echo "一键启动右手单臂视觉 SmolVLA："
    echo "  bash ${PROJECT_ROOT}/train_smolvla/scripts/start_smolvla_right_train.sh"
    echo "一键启动纯视觉 Pi0.5："
    echo "  bash ${PROJECT_ROOT}/deploy_pi05/scripts/start_pi05.sh"
    echo "一键启动 Pi0.5 + FRS："
    echo "  bash ${PROJECT_ROOT}/deploy_pi05/scripts/start_pi05_frs.sh"
    echo "一键启动纯视觉 Pi0.5 训练："
    echo "  bash ${PROJECT_ROOT}/train_pi05/scripts/start_pi05_train.sh"
    echo "一键启动右手单臂 Pi0.5 FRS 训练："
    echo "  bash ${PROJECT_ROOT}/train_pi05_frs/scripts/start_frs_pi05_right_train.sh"
}

main() {
    parse_args "$@" || return $?
    if ((SHOW_HELP)); then
        usage
        return 0
    fi

    validate_environment_targets
    validate_selected_projects
    install_system_dependencies
    install_uv
    persist_uv_path
    configure_uv_storage
    configure_runtime_storage
    install_python
    should_setup_smolvla && sync_root_environment
    should_setup_smolvla && sync_smolvla_torch_environment
    should_setup_pi05 && sync_pi05_environment
    should_setup_pi05_train && sync_data_tools_environment
    should_setup_pi05_train && sync_pi05_train_environment
    should_setup_pi05_frs_train && sync_pi05_frs_train_environment
    write_environment_file
    should_setup_smolvla && verify_python_environment
    should_setup_smolvla && verify_smolvla_torch_environment
    should_setup_pi05 && verify_pi05_environment
    should_setup_pi05_train && verify_data_tools_environment
    should_setup_pi05_train && verify_pi05_train_environment
    should_setup_pi05_frs_train && verify_pi05_frs_train_environment
    should_setup_smolvla && check_root_gpu
    should_setup_pi05 && check_pi05_gpu
    should_setup_pi05_train && check_pi05_train_gpu
    should_setup_pi05_frs_train && check_pi05_frs_train_gpu
    print_summary
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    main "$@"
fi
