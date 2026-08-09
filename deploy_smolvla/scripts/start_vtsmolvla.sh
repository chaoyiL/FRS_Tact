#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CHECKPOINTS_DIR="${ROOT}/checkpoints"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${CHECKPOINTS_DIR}/model}"
mkdir -p "${HF_HUB_CACHE}" "${CHECKPOINTS_DIR}/encoder"
CONFIG="${FRS_DEPLOY_CONFIG:-${ROOT}/deploy_smolvla/configs/deploy_smolvla_jax.yaml}"
TOKEN_FILE="${VB3_TOKEN_FILE:-/home/typhon/vb3_robot_server/token_list.txt}"
CHECK_ONLY=false
if [[ -n "${FRS_PYTHON:-}" ]]; then
    PYTHON_BIN="${FRS_PYTHON}"
elif [[ -n "${VB3_PYTHON:-}" ]]; then
    PYTHON_BIN="${VB3_PYTHON}"
elif [[ -x "${ROOT}/.venv/bin/python" ]]; then
    PYTHON_BIN="${ROOT}/.venv/bin/python"
else
    PYTHON_BIN="python3"
fi

usage() {
    cat <<'EOF'
Usage: bash ./deploy_smolvla/scripts/start_vtsmolvla.sh [--check] [--config PATH]

Environment overrides:
  VB_ROBOT_TOKEN    Robot authentication token (preferred when already set)
  VB3_TOKEN_FILE    Token file used when VB_ROBOT_TOKEN is unset
  FRS_DEPLOY_CONFIG Deployment YAML path
  FRS_PYTHON        Python executable (highest priority)
  VB3_PYTHON        Python executable fallback
  HF_HUB_CACHE      Hugging Face Hub cache (default: <project>/checkpoints/model)
EOF
}

while (( $# > 0 )); do
    case "$1" in
        --check)
            CHECK_ONLY=true
            shift
            ;;
        --config)
            if (( $# < 2 )); then
                echo "--config requires a path" >&2
                exit 2
            fi
            CONFIG="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown argument: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

if [[ ! -r "${CONFIG}" ]]; then
    echo "Deployment config is not readable: ${CONFIG}" >&2
    exit 2
fi

token_source="environment:VB_ROBOT_TOKEN"
if [[ -z "${VB_ROBOT_TOKEN:-}" ]]; then
    if [[ ! -r "${TOKEN_FILE}" ]]; then
        echo "VB_ROBOT_TOKEN is unset and token file is not readable: ${TOKEN_FILE}" >&2
        exit 2
    fi
    VB_ROBOT_TOKEN=""
    while IFS= read -r candidate || [[ -n "${candidate}" ]]; do
        if [[ -n "${candidate}" ]]; then
            VB_ROBOT_TOKEN="${candidate}"
            break
        fi
    done <"${TOKEN_FILE}"
    if [[ -z "${VB_ROBOT_TOKEN}" ]]; then
        echo "VB_ROBOT_TOKEN is unset and token file has no non-empty token: ${TOKEN_FILE}" >&2
        exit 2
    fi
    export VB_ROBOT_TOKEN
    token_source="file:${TOKEN_FILE}"
fi

if [[ "${CHECK_ONLY}" == true ]]; then
    echo "config=${CONFIG}"
    echo "token_source=${token_source}"
    echo "model_cache=${HF_HUB_CACHE}"
    echo "python=${PYTHON_BIN}"
    echo "entrypoint=deploy_smolvla.remote_client"
    exit 0
fi

export PYTHONUNBUFFERED=1
export PYTHONPATH="${ROOT}/src:${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
exec "${PYTHON_BIN}" -m deploy_smolvla.remote_client --config "${CONFIG}"
