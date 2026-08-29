#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
TRAIN_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
TRAIN_PYTHON="${TRAIN_PI05_PYTHON:-${TRAIN_ROOT}/.venv/bin/python}"
export PYTHONPATH="${TRAIN_ROOT}/src:${TRAIN_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONSAFEPATH=1
TMUX_SESSION="${PI05_TMUX_SESSION:-pi05_vision_train}"

fail() { printf '[pi05] error: %s\n' "$*" >&2; return 1; }
log() { printf '[pi05] %s\n' "$*"; }

check_only=0
config_path=""
while (($#)); do
    case "$1" in
        --check) check_only=1 ;;
        -*) fail "usage: $0 [--check] [CONFIG]"; exit 2 ;;
        *) [[ -z "${config_path}" ]] || { fail "only one CONFIG is allowed"; exit 2; }; config_path="$1" ;;
    esac
    shift
done
config_path="${config_path:-${TRAIN_ROOT}/configs/train_pi05.yaml}"

[[ -x "${TRAIN_PYTHON}" ]] || fail \
    "training Python is not executable: ${TRAIN_PYTHON}; run: bash ${TRAIN_ROOT}/../scripts/setup_env.sh --pi05_train"
[[ -f "${config_path}" ]] || fail "configuration does not exist: ${config_path}"
config_path="$(realpath -- "${config_path}")"
cd "${TRAIN_ROOT}"

output_dir="$(${TRAIN_PYTHON} train.py --config "${config_path}" --check --print-output)"
if [[ "${check_only}" == 1 ]]; then
    log "configuration and dataset paths are valid: ${config_path}"
    exit 0
fi

if [[ "${PI05_FOREGROUND:-0}" != 1 && -z "${TMUX:-}" ]] && command -v tmux >/dev/null 2>&1; then
    inner=""
    tmux has-session -t "${TMUX_SESSION}" 2>/dev/null \
        && fail "tmux session already exists: ${TMUX_SESSION}"
    printf -v inner 'env PYTHONPATH=%q PYTHONSAFEPATH=1 %q train.py --config %q' \
        "${PYTHONPATH}" "${TRAIN_PYTHON}" "${config_path}"
    tmux new-session -d -s "${TMUX_SESSION}" -c "${TRAIN_ROOT}" \
        "${inner}"
    log "started tmux session ${TMUX_SESSION}; attach with: tmux attach -t ${TMUX_SESSION}"
    exit 0
fi

mkdir -p -- "${output_dir}"
log "training output: ${output_dir}"
exec "${TRAIN_PYTHON}" train.py --config "${config_path}"
