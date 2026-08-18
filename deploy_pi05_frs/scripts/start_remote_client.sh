#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
MODE="frs"
CONFIG=""
TOKEN_FILE="${VB3_TOKEN_FILE:-/home/typhon/vb3_robot_server/token_list.txt}"
CHECK_ONLY=false
MAX_ITERATIONS=""

usage() {
    cat <<'EOF'
Usage: bash deploy_pi05_frs/scripts/start_remote_client.sh --config PATH [--mode pi05|frs] [--check] [--max-iterations N]

Environment overrides:
  VB_ROBOT_TOKEN    Robot authentication token
  VB3_TOKEN_FILE    Token file used when VB_ROBOT_TOKEN is unset
  PI05_PYTHON       Python executable for pi05 mode (highest priority)
  PI05_FRS_PYTHON   Python executable for frs mode (highest priority)
  VB3_PYTHON        Python executable fallback
  OPENPI_DATA_HOME  openpi checkpoint cache
EOF
}

while (( $# > 0 )); do
    case "$1" in
        --check) CHECK_ONLY=true; shift ;;
        --mode)
            (( $# >= 2 )) || { echo "--mode requires a value" >&2; exit 2; }
            MODE="$2"; shift 2 ;;
        --config)
            (( $# >= 2 )) || { echo "--config requires a path" >&2; exit 2; }
            CONFIG="$2"; shift 2 ;;
        --max-iterations)
            (( $# >= 2 )) || { echo "--max-iterations requires a value" >&2; exit 2; }
            MAX_ITERATIONS="$2"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
    esac
done

if [[ -n "${MAX_ITERATIONS}" && ! "${MAX_ITERATIONS}" =~ ^[0-9]+$ ]]; then
    echo "--max-iterations must be a non-negative integer" >&2
    exit 2
fi

case "${MODE}" in
    pi05)
        ENTRYPOINT="deploy_pi05_frs.pi05_client"
        MODE_PYTHON="${PI05_PYTHON:-}"
        ;;
    frs)
        ENTRYPOINT="deploy_pi05_frs.remote_client"
        MODE_PYTHON="${PI05_FRS_PYTHON:-}"
        ;;
    *) echo "Unsupported mode: ${MODE}" >&2; exit 2 ;;
esac

if [[ -n "${MODE_PYTHON}" ]]; then
    PYTHON_BIN="${MODE_PYTHON}"
elif [[ -n "${VB3_PYTHON:-}" ]]; then
    PYTHON_BIN="${VB3_PYTHON}"
elif [[ -x "${ROOT}/.venv/bin/python" ]]; then
    PYTHON_BIN="${ROOT}/.venv/bin/python"
else
    PYTHON_BIN="python3"
fi

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
    echo "mode=${MODE}"
    echo "config=${CONFIG}"
    echo "token_source=${token_source}"
    echo "python=${PYTHON_BIN}"
    echo "entrypoint=${ENTRYPOINT}"
    exit 0
fi

export PYTHONUNBUFFERED=1
export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
PYTHON_ARGS=(-m "${ENTRYPOINT}" --config "${CONFIG}")
if [[ -n "${MAX_ITERATIONS}" ]]; then
    PYTHON_ARGS+=(--max-iterations "${MAX_ITERATIONS}")
fi
exec "${PYTHON_BIN}" "${PYTHON_ARGS[@]}"
