#!/usr/bin/env bash

set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
K8_CONFIG="${PROJECT_ROOT}/configs/train_vtsmolvla_jax_tactile16.yaml"
K21_CONFIG="${PROJECT_ROOT}/configs/train_vtsmolvla_jax_tactile32.yaml"
ENV_FILE="${PROJECT_ROOT}/.env.frs"
SCRIPT_PATH="${SCRIPT_DIR}/start_vtsmolvla_train.sh"
TMUX_SESSION="${FRS_TMUX_SESSION:-vtsmolvla_train}"
GPUS="0,1"
EXPERIMENT="both"
CONFIG_PATH=""
ORIGINAL_ARGS=("$@")

if [[ "${PROJECT_ROOT}" == /workspace/* ]]; then
    STORAGE_ROOT="${FRS_STORAGE_ROOT:-/workspace}"
else
    STORAGE_ROOT="${FRS_STORAGE_ROOT:-${PROJECT_ROOT}/.cache}"
fi

log() {
    echo "[vtsmolvla] $*"
}

fail() {
    echo "[vtsmolvla] 错误：$*" >&2
    exit 1
}

usage() {
    printf '%s\n' \
        "用法：scripts/start_vtsmolvla_train.sh [--experiment both|k8|k21] [--gpus 0,1] [--foreground] [--session NAME]" \
        "       scripts/start_vtsmolvla_train.sh --config PATH [--gpus 0,1] [--foreground] [--session NAME]" \
        "" \
        "默认在一条链路中预计算一次 cache，随后依次训练 K8 与 K21。" \
        "--config 保持单配置兼容模式；相对路径以项目根目录为基准。"
}

parse_arguments() {
    local config_set=0 experiment_set=0
    while (($#)); do
        case "$1" in
            -h|--help)
                usage
                exit 0
                ;;
            --config)
                (($# >= 2)) || fail "--config 需要一个路径"
                ((config_set == 0)) || fail "--config 只能指定一次"
                [[ -n "$2" ]] || fail "--config 需要一个路径"
                CONFIG_PATH="$2"
                config_set=1
                shift 2
                ;;
            --config=*)
                ((config_set == 0)) || fail "--config 只能指定一次"
                CONFIG_PATH="${1#--config=}"
                [[ -n "${CONFIG_PATH}" ]] || fail "--config 需要一个路径"
                config_set=1
                shift
                ;;
            --experiment)
                (($# >= 2)) || fail "--experiment 需要 both、k8 或 k21"
                ((experiment_set == 0)) || fail "--experiment 只能指定一次"
                EXPERIMENT="$2"
                experiment_set=1
                shift 2
                ;;
            --experiment=*)
                ((experiment_set == 0)) || fail "--experiment 只能指定一次"
                EXPERIMENT="${1#--experiment=}"
                experiment_set=1
                shift
                ;;
            --gpus)
                (($# >= 2)) || fail "--gpus 需要两个以逗号分隔的 GPU ID"
                GPUS="$2"
                shift 2
                ;;
            --gpus=*)
                GPUS="${1#--gpus=}"
                shift
                ;;
            --foreground)
                export FRS_FOREGROUND=1
                shift
                ;;
            --session)
                (($# >= 2)) || fail "--session 需要名称"
                [[ -n "$2" ]] || fail "--session 需要名称"
                TMUX_SESSION="$2"
                shift 2
                ;;
            --session=*)
                TMUX_SESSION="${1#--session=}"
                [[ -n "${TMUX_SESSION}" ]] || fail "--session 需要名称"
                shift
                ;;
            *)
                fail "未知参数：$1"
                ;;
        esac
    done

    case "${EXPERIMENT}" in
        both|k8|k21) ;;
        *) fail "--experiment 只能是 both、k8 或 k21" ;;
    esac
    ((config_set && experiment_set)) && fail "--config 与 --experiment 不能同时使用"
    if ((config_set)) && [[ "${CONFIG_PATH}" != /* ]]; then
        CONFIG_PATH="${PROJECT_ROOT}/${CONFIG_PATH}"
    fi
    [[ "${GPUS}" =~ ^[0-9]+,[0-9]+$ ]] || fail "--gpus 必须是两个以逗号分隔的 GPU ID，例如 0,1"
    local first_gpu second_gpu
    IFS=, read -r first_gpu second_gpu <<< "${GPUS}"
    [[ "${first_gpu}" != "${second_gpu}" ]] || fail "--gpus 必须指定两张不同的 GPU"
    export CUDA_VISIBLE_DEVICES="${GPUS}"
}

on_error() {
    local status=$?
    echo "[vtsmolvla] 训练链路在第 ${BASH_LINENO[0]} 行失败，退出码 ${status}" >&2
    exit "${status}"
}
trap on_error ERR

load_environment() {
    if [[ -f "${ENV_FILE}" ]]; then
        # shellcheck disable=SC1090
        source "${ENV_FILE}"
    elif [[ -d "/opt/venvs/frs_tact" ]]; then
        export UV_PROJECT_ENVIRONMENT="/opt/venvs/frs_tact"
        export UV_CACHE_DIR="${UV_CACHE_DIR:-${HOME}/.cache/uv}"
    elif [[ ! -d "${PROJECT_ROOT}/.venv" ]]; then
        fail "没有找到 Python 环境。请先运行：bash ${PROJECT_ROOT}/scripts/setup_env.sh"
    fi

    if command -v uv >/dev/null 2>&1; then
        UV_BIN="$(command -v uv)"
    elif [[ -x "${HOME}/.local/bin/uv" ]]; then
        UV_BIN="${HOME}/.local/bin/uv"
    elif [[ -x "${HOME}/.cargo/bin/uv" ]]; then
        UV_BIN="${HOME}/.cargo/bin/uv"
    else
        fail "找不到 uv。请先运行：bash ${PROJECT_ROOT}/scripts/setup_env.sh"
    fi
    export PATH="${HOME}/.local/bin:${HOME}/.cargo/bin:${PATH}"
    export HF_HOME="${STORAGE_ROOT}/huggingface"
    export HF_HUB_CACHE="${HF_HOME}/hub"
    export HF_DATASETS_CACHE="${HF_HOME}/datasets_arrow"
    export HF_LEROBOT_HOME="${HF_HOME}/lerobot"
    export TMPDIR="${STORAGE_ROOT}/tmp"
    mkdir -p "${HF_HUB_CACHE}" "${HF_DATASETS_CACHE}" "${HF_LEROBOT_HOME}" "${TMPDIR}"
}

start_tmux_if_needed() {
    if [[ "${FRS_FOREGROUND:-0}" == "1" || -n "${TMUX:-}" ]]; then
        return
    fi
    if ! command -v tmux >/dev/null 2>&1; then
        return
    fi
    if tmux has-session -t "${TMUX_SESSION}" 2>/dev/null; then
        fail "tmux session ${TMUX_SESSION} 已存在。查看：tmux attach -t ${TMUX_SESSION}"
    fi
    local inner_command quoted_argument
    printf -v inner_command 'FRS_FOREGROUND=1 bash %q' "${SCRIPT_PATH}"
    for quoted_argument in "${ORIGINAL_ARGS[@]}"; do
        printf -v inner_command '%s %q' "${inner_command}" "${quoted_argument}"
    done
    tmux new-session -d -s "${TMUX_SESSION}" -c "${PROJECT_ROOT}" "${inner_command}"
    log "训练已在 tmux 后台启动：${TMUX_SESSION}"
    log "查看实时输出：tmux attach -t ${TMUX_SESSION}"
    exit 0
}

read_yaml_settings() {
    [[ -f "${CONFIG_PATH}" ]] || fail "配置不存在：${CONFIG_PATH}"
    mapfile -t YAML_SETTINGS < <(
        "${UV_BIN}" run --no-sync python - "${CONFIG_PATH}" <<'PY'
import pathlib
import sys

import yaml

path = pathlib.Path(sys.argv[1])
with path.open(encoding="utf-8") as file:
    config = yaml.safe_load(file) or {}
output = config.get("output")
if not output:
    raise ValueError("YAML 缺少 output")
cache = config.get("tactile_embedding_cache") or {}
print(pathlib.Path(output).expanduser())
print("1" if cache.get("enabled", False) else "0")
print("" if config.get("resume") in (None, "") else pathlib.Path(config["resume"]).expanduser())
PY
    )
    ((${#YAML_SETTINGS[@]} == 3)) || fail "无法从 YAML 读取 output/cache/resume"
    OUTPUT_DIR="${YAML_SETTINGS[0]}"
    CACHE_ENABLED="${YAML_SETTINGS[1]}"
    RESUME_PATH="${YAML_SETTINGS[2]}"
}

preflight() {
    cd "${PROJECT_ROOT}"
    "${UV_BIN}" run --no-sync python - "${CONFIG_PATH}" <<'PY'
import pathlib
import sys

import jax
import yaml

with pathlib.Path(sys.argv[1]).open(encoding="utf-8") as file:
    config = yaml.safe_load(file) or {}
missing = []
for dataset in config.get("datasets") or []:
    root = dataset.get("root")
    if root and not pathlib.Path(root).is_dir():
        missing.append(f"dataset root: {root}")
encoder = (config.get("model") or {}).get("tactile_encoder_path")
if encoder and not pathlib.Path(encoder).is_dir():
    missing.append(f"tactile encoder: {encoder}")
if missing:
    raise FileNotFoundError("缺少训练输入：\n  " + "\n  ".join(missing))
devices = jax.devices()
if len(devices) != 2:
    raise RuntimeError(f"需要恰好两张 H100，JAX 看到 {len(devices)} 张设备：{devices}")
if any(device.platform != "gpu" or "H100" not in device.device_kind.upper() for device in devices):
    raise RuntimeError(f"需要两张 H100，JAX devices={devices}")
print(f"JAX devices={devices}")
PY
    if compgen -G "${OUTPUT_DIR}/checkpoint-*" >/dev/null && [[ -z "${RESUME_PATH}" ]]; then
        fail "${OUTPUT_DIR} 已有 checkpoint，但 YAML 的 resume 为空。请设置新 output 或在 YAML 配置 resume。"
    fi
}

run_config() {
    local config_path="$1" precompute="$2" timestamp precompute_log train_log
    CONFIG_PATH="${config_path}"
    read_yaml_settings
    preflight
    mkdir -p "${OUTPUT_DIR}"
    timestamp="$(date +%Y%m%d_%H%M%S)"
    precompute_log="${OUTPUT_DIR}/precompute_${timestamp}.log"
    train_log="${OUTPUT_DIR}/train_${timestamp}.log"
    export PYTHONUNBUFFERED=1
    export TOKENIZERS_PARALLELISM=false

    log "config=${CONFIG_PATH}"
    log "output=${OUTPUT_DIR}"
    if [[ "${precompute}" == "1" && "${CACHE_ENABLED}" == "1" ]]; then
        log "检查并补齐 tactile embedding cache"
        "${UV_BIN}" run --no-sync python tools/precompute_tactile_embeddings.py --config "${CONFIG_PATH}" \
            2>&1 | tee -a "${precompute_log}"
    fi
    log "开始 VT-SmolVLA 训练，日志=${train_log}"
    "${UV_BIN}" run --no-sync python tools/train_vtsmolvla_jax.py --config "${CONFIG_PATH}" \
        2>&1 | tee -a "${train_log}"
}

run_pipeline() {
    if [[ -n "${CONFIG_PATH}" ]]; then
        run_config "${CONFIG_PATH}" 1
        return
    fi
    case "${EXPERIMENT}" in
        both)
            run_config "${K8_CONFIG}" 1
            run_config "${K21_CONFIG}" 0
            ;;
        k8) run_config "${K8_CONFIG}" 1 ;;
        k21) run_config "${K21_CONFIG}" 1 ;;
    esac
}

parse_arguments "$@"
start_tmux_if_needed
load_environment
run_pipeline
