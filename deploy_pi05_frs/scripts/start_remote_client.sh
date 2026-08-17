#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CONFIG=""
TOKEN_FILE="${VB3_TOKEN_FILE:-/home/typhon/vb3_robot_server/token_list.txt}"
CHECK_ONLY=false

if [[ -n "${PI05_FRS_PYTHON:-}" ]]; then
    PYTHON_BIN="${PI05_FRS_PYTHON}"
elif [[ -n "${VB3_PYTHON:-}" ]]; then
    PYTHON_BIN="${VB3_PYTHON}"
elif [[ -x "${ROOT}/.venv/bin/python" ]]; then
    PYTHON_BIN="${ROOT}/.venv/bin/python"
else
    PYTHON_BIN="python3"
fi

usage() {
    cat <<'EOF'
Usage: bash deploy_pi05_frs/scripts/start_remote_client.sh --config PATH [--check]

Environment overrides:
  VB_ROBOT_TOKEN    Robot authentication token
  VB3_TOKEN_FILE    Token file used when VB_ROBOT_TOKEN is unset
  PI05_FRS_PYTHON   Python executable (highest priority)
  VB3_PYTHON        Python executable fallback
  OPENPI_DATA_HOME  openpi checkpoint cache
EOF
}

while (( $# > 0 )); do
    case "$1" in
        --check) CHECK_ONLY=true; shift ;;
        --config)
            (( $# >= 2 )) || { echo "--config requires a path" >&2; exit 2; }
            CONFIG="$2"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
    esac
done

[[ -n "${CONFIG}" ]] || { echo "--config is required" >&2; exit 2; }
[[ -r "${CONFIG}" ]] || { echo "Deployment config is not readable: ${CONFIG}" >&2; exit 2; }

token_source="environment:VB_ROBOT_TOKEN"
if [[ -z "${VB_ROBOT_TOKEN:-}" ]]; then
    [[ -r "${TOKEN_FILE}" ]] || {
        echo "VB_ROBOT_TOKEN is unset and token file is not readable: ${TOKEN_FILE}" >&2
        exit 2
    }
    while IFS= read -r candidate || [[ -n "${candidate}" ]]; do
        if [[ -n "${candidate}" ]]; then
            export VB_ROBOT_TOKEN="${candidate}"
            break
        fi
    done <"${TOKEN_FILE}"
    [[ -n "${VB_ROBOT_TOKEN:-}" ]] || { echo "Token file is empty: ${TOKEN_FILE}" >&2; exit 2; }
    token_source="file:${TOKEN_FILE}"
fi

if [[ "${CHECK_ONLY}" == true ]]; then
    echo "config=${CONFIG}"
    echo "token_source=${token_source}"
    echo "python=${PYTHON_BIN}"
    echo "entrypoint=deploy_pi05_frs.remote_client"
    exit 0
fi

export PYTHONUNBUFFERED=1
export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
exec "${PYTHON_BIN}" -m deploy_pi05_frs.remote_client --config "${CONFIG}"
