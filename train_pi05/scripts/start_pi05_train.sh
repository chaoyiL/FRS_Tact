#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
TRAIN_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
PROJECT_ROOT="$(cd -- "${TRAIN_ROOT}/.." && pwd)"
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

# setup_env.sh --pi05_train 会把唯一的训练解释器路径写入 env_path。
# 调用者显式传入的 TRAIN_PI05_PYTHON 优先级最高，便于临时覆盖。
TRAIN_PYTHON_OVERRIDE="${TRAIN_PI05_PYTHON:-}"
if [[ -f "${ENV_FILE}" ]]; then
    # shellcheck disable=SC1090
    source "${ENV_FILE}"
fi
TRAIN_PYTHON="${TRAIN_PYTHON_OVERRIDE:-${TRAIN_PI05_PYTHON:-${TRAIN_ROOT}/.venv/bin/python}}"
export PYTHONPATH="${TRAIN_ROOT}/src:${TRAIN_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONSAFEPATH=1
TMUX_SESSION="${PI05_TMUX_SESSION:-pi05_vision_train}"

fail() { printf '[pi05] error: %s\n' "$*" >&2; return 1; }
log() { printf '[pi05] %s\n' "$*"; }

check_only=0
attach_tmux=0
config_path=""
while (($#)); do
    case "$1" in
        --check) check_only=1 ;;
        --attach) attach_tmux=1 ;;
        --detach) attach_tmux=0 ;; # 兼容旧命令；现在默认就是后台启动。
        -*) fail "usage: $0 [--check] [--attach] [CONFIG]"; exit 2 ;;
        *) [[ -z "${config_path}" ]] || { fail "only one CONFIG is allowed"; exit 2; }; config_path="$1" ;;
    esac
    shift
done
config_path="${config_path:-${TRAIN_ROOT}/configs/train_pi05.yaml}"

[[ -x "${TRAIN_PYTHON}" ]] || fail \
    "training Python is not executable: ${TRAIN_PYTHON}; run: bash ${PROJECT_ROOT}/scripts/setup_env.sh --pi05_train"
[[ -f "${config_path}" ]] || fail "configuration does not exist: ${config_path}"
config_path="$(realpath -- "${config_path}")"
cd "${TRAIN_ROOT}"

output_dir="$(${TRAIN_PYTHON} train.py --config "${config_path}" --check --print-output)"
if [[ "${check_only}" == 1 ]]; then
    log "configuration and dataset paths are valid: ${config_path}"
    exit 0
fi

norm_stats_file="$(${TRAIN_PYTHON} train.py --config "${config_path}" --check --print-norm-stats)"
norm_batch_size="${PI05_NORM_BATCH_SIZE:-1024}"
norm_num_workers="${PI05_NORM_NUM_WORKERS:-8}"
log_dir="${output_dir}/logs"
log_file="${log_dir}/train_$(date '+%Y%m%d_%H%M%S').log"
inner=""
prepare_log_command() {
    mkdir -p -- "${log_dir}"
    ln -sfn -- "$(basename -- "${log_file}")" "${log_dir}/latest.log"
    printf -v inner 'set -Eeuo pipefail; { if [[ ! -s %q ]]; then printf "[pi05] norm stats missing; computing: %%s\n" %q; env PYTHONPATH=%q PYTHONSAFEPATH=1 %q tools/compute_norm_stats.py --config-name %q --batch-size %q --num-workers %q; else printf "[pi05] using norm stats: %%s\n" %q; fi; env PYTHONPATH=%q PYTHONSAFEPATH=1 %q train.py --config %q; } 2>&1 | tee -a %q' \
        "${norm_stats_file}" "${norm_stats_file}" \
        "${PYTHONPATH}" "${TRAIN_PYTHON}" "${config_path}" "${norm_batch_size}" "${norm_num_workers}" \
        "${norm_stats_file}" "${PYTHONPATH}" "${TRAIN_PYTHON}" "${config_path}" "${log_file}"
}

if [[ "${PI05_FOREGROUND:-0}" != 1 && -z "${TMUX:-}" ]] && command -v tmux >/dev/null 2>&1; then
    if tmux has-session -t "${TMUX_SESSION}" 2>/dev/null; then
        # 上一次训练已经结束时，自动清理只剩 dead pane 的会话；正在训练的会话绝不覆盖。
        if [[ "$(tmux display-message -p -t "${TMUX_SESSION}:0.0" '#{pane_dead}')" == 1 ]]; then
            tmux kill-session -t "${TMUX_SESSION}"
        else
            fail "tmux session already exists: ${TMUX_SESSION}; attach with: tmux attach -t ${TMUX_SESSION}"
            exit 1
        fi
    fi
    prepare_log_command
    tmux_command=""
    printf -v tmux_command 'bash -lc %q' "${inner}"
    tmux new-session -d -s "${TMUX_SESSION}" -c "${TRAIN_ROOT}" \
        "${tmux_command}"

    log "started detached tmux session: ${TMUX_SESSION}"
    log "log file: ${log_file}"
    log "follow log: tail -F ${log_dir}/latest.log"

    if [[ "${attach_tmux}" == 1 && "${PI05_TMUX_DETACHED:-0}" != 1 && -t 0 && -t 1 ]]; then
        exec tmux attach-session -t "${TMUX_SESSION}"
    fi
    exit 0
fi

prepare_log_command
log "training output: ${output_dir}"
log "log file: ${log_file}"
exec bash -lc "${inner}"
