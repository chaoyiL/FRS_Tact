#!/usr/bin/env bash

set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
BASE_LAUNCHER="${SCRIPT_DIR}/start_vtsmolvla_train.sh"
K1_CONFIG="${PROJECT_ROOT}/configs/train_vtsmolvla_jax_tactile_k1.yaml"
K4_CONFIG="${PROJECT_ROOT}/configs/train_vtsmolvla_jax_tactile_8pct.yaml"
SCRIPT_PATH="${SCRIPT_DIR}/start_vtsmolvla_low_token_train.sh"
GPUS="0,1"
TMUX_SESSION="${FRS_TMUX_SESSION:-vtsmolvla_low_token}"
ORIGINAL_ARGS=("$@")

log() {
    echo "[vtsmolvla-low-token] $*"
}

fail() {
    echo "[vtsmolvla-low-token] 错误：$*" >&2
    exit 1
}

usage() {
    printf '%s\n' \
        "用法：scripts/start_vtsmolvla_low_token_train.sh [--gpus 0,1] [--foreground] [--session NAME]" \
        "" \
        "在同一对 GPU 上先训练 K=1（4 tactile tokens，2.21%），完成后训练 K=4（16 tactile tokens，8.29%）。" \
        "两组均训练 80000 steps，每 10000 steps 保存一次。"
}

parse_arguments() {
    while (($#)); do
        case "$1" in
            -h|--help)
                usage
                exit 0
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

    [[ "${GPUS}" =~ ^[0-9]+,[0-9]+$ ]] || fail "--gpus 必须是两个以逗号分隔的 GPU ID，例如 0,1"
    local first_gpu second_gpu
    IFS=, read -r first_gpu second_gpu <<< "${GPUS}"
    [[ "${first_gpu}" != "${second_gpu}" ]] || fail "--gpus 必须指定两张不同的 GPU"
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

run_experiment() {
    local label="$1" config="$2"
    log "开始 ${label}：${config}，GPU=${GPUS}"
    FRS_FOREGROUND=1 bash "${BASE_LAUNCHER}" \
        --config "${config}" \
        --gpus "${GPUS}" \
        --foreground
}

parse_arguments "$@"
start_tmux_if_needed
cd "${PROJECT_ROOT}"
run_experiment "K=1 / tactile 2.21%" "${K1_CONFIG}"
run_experiment "K=4 / tactile 8.29%" "${K4_CONFIG}"
log "K=1 与 K=4 训练均已完成"
