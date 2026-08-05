#!/usr/bin/env bash

set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
CONFIG_PATH="${PROJECT_ROOT}/configs/train_vtsmolvla_jax.yaml"
ENV_FILE="${PROJECT_ROOT}/.env.frs"
SCRIPT_PATH="${SCRIPT_DIR}/start_vtsmolvla_train.sh"
TMUX_SESSION="${FRS_TMUX_SESSION:-vtsmolvla_train}"

log() {
    echo "[vtsmolvla] $*"
}

fail() {
    echo "[vtsmolvla] 错误：$*" >&2
    exit 1
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
        fail "没有找到 Python 环境。请先运行：bash ${PROJECT_ROOT}/srcipts/setup_env.sh"
    fi

    if command -v uv >/dev/null 2>&1; then
        UV_BIN="$(command -v uv)"
    elif [[ -x "${HOME}/.local/bin/uv" ]]; then
        UV_BIN="${HOME}/.local/bin/uv"
    elif [[ -x "${HOME}/.cargo/bin/uv" ]]; then
        UV_BIN="${HOME}/.cargo/bin/uv"
    else
        fail "找不到 uv。请先运行：bash ${PROJECT_ROOT}/srcipts/setup_env.sh"
    fi
    export PATH="${HOME}/.local/bin:${HOME}/.cargo/bin:${PATH}"
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
    local inner_command
    printf -v inner_command 'FRS_FOREGROUND=1 bash %q' "${SCRIPT_PATH}"
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
resume = config.get("resume")
print(pathlib.Path(output).expanduser())
print("1" if cache.get("enabled", False) else "0")
print("" if resume in (None, "") else pathlib.Path(resume).expanduser())
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
print(f"JAX devices={devices}")
if not any(device.platform == "gpu" for device in devices):
    raise RuntimeError("JAX 没有识别到 GPU，拒绝启动正式训练")
PY
    if compgen -G "${OUTPUT_DIR}/checkpoint-*" >/dev/null && [[ -z "${RESUME_PATH}" ]]; then
        fail "${OUTPUT_DIR} 已有 checkpoint，但 YAML 的 resume 为空。请设置新 output 或在 YAML 配置 resume。"
    fi
}

run_pipeline() {
    mkdir -p "${OUTPUT_DIR}"
    local timestamp precompute_log train_log
    timestamp="$(date +%Y%m%d_%H%M%S)"
    precompute_log="${OUTPUT_DIR}/precompute_${timestamp}.log"
    train_log="${OUTPUT_DIR}/train_${timestamp}.log"
    export PYTHONUNBUFFERED=1
    export TOKENIZERS_PARALLELISM=false

    log "config=${CONFIG_PATH}"
    log "output=${OUTPUT_DIR}"
    if [[ "${CACHE_ENABLED}" == "1" ]]; then
        log "检查并补齐 tactile embedding cache"
        "${UV_BIN}" run --no-sync python tools/precompute_tactile_embeddings.py \
            --config "${CONFIG_PATH}" \
            2>&1 | tee -a "${precompute_log}"
    fi

    log "开始 VT-SmolVLA 训练，日志=${train_log}"
    "${UV_BIN}" run --no-sync python tools/train_vtsmolvla_jax.py \
        --config "${CONFIG_PATH}" \
        2>&1 | tee -a "${train_log}"
}

load_environment
read_yaml_settings
preflight
start_tmux_if_needed
run_pipeline
