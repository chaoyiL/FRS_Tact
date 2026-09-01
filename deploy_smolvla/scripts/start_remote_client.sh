#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CHECKPOINTS_DIR="${ROOT}/checkpoints"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${CHECKPOINTS_DIR}/model}"
HF_HUB_CACHE_SELECTED="${HF_HUB_CACHE}"
mkdir -p "${HF_HUB_CACHE}" "${CHECKPOINTS_DIR}/encoder"
CONFIG=""
TOKEN_FILE="${VB3_TOKEN_FILE:-/home/typhon/vb3_robot_server/token_list.txt}"
CHECK_ONLY=false
MAX_ITERATIONS=""

usage() {
    cat <<'EOF'
Usage: bash ./deploy_smolvla/scripts/start_remote_client.sh --config PATH [--max-iterations N] [--check]

Environment overrides:
  VB_ROBOT_TOKEN    Robot authentication token (preferred when already set)
  VB3_TOKEN_FILE    Token file used when VB_ROBOT_TOKEN is unset
  SMOLVLA_TORCH_PYTHON  Official LeRobot+PEFT Python for pytorch_smolvla
  FRS_PYTHON        JAX/FRS Python executable
  VB3_PYTHON        JAX/FRS Python executable fallback
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
        --max-iterations)
            if (( $# < 2 )) || [[ ! "$2" =~ ^[1-9][0-9]*$ ]]; then
                echo "--max-iterations must be a positive integer" >&2
                exit 2
            fi
            MAX_ITERATIONS="$2"
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

if [[ -z "${CONFIG}" ]]; then
    echo "--config is required" >&2
    exit 2
fi

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

SMOLVLA_TORCH_PYTHON_OVERRIDE="${SMOLVLA_TORCH_PYTHON:-}"
FRS_PYTHON_OVERRIDE="${FRS_PYTHON:-}"
VB3_PYTHON_OVERRIDE="${VB3_PYTHON:-}"
WORKSPACE_ROOT_OVERRIDE="${WORKSPACE_ROOT:-}"
ENV_FILE="${ROOT}/env_path"
if [[ ! -f "${ENV_FILE}" ]]; then
    if [[ -f "${ROOT}/environment_paths.sh" ]]; then
        ENV_FILE="${ROOT}/environment_paths.sh"
    elif [[ -f "${ROOT}/.env.frs" ]]; then
        ENV_FILE="${ROOT}/.env.frs"
    fi
fi
if [[ -f "${ENV_FILE}" ]]; then
    # shellcheck disable=SC1090
    source "${ENV_FILE}"
fi
export HF_HUB_CACHE="${HF_HUB_CACHE_SELECTED}"
[[ -z "${SMOLVLA_TORCH_PYTHON_OVERRIDE}" ]] || \
    SMOLVLA_TORCH_PYTHON="${SMOLVLA_TORCH_PYTHON_OVERRIDE}"
[[ -z "${FRS_PYTHON_OVERRIDE}" ]] || FRS_PYTHON="${FRS_PYTHON_OVERRIDE}"
[[ -z "${VB3_PYTHON_OVERRIDE}" ]] || VB3_PYTHON="${VB3_PYTHON_OVERRIDE}"
[[ -z "${WORKSPACE_ROOT_OVERRIDE}" ]] || WORKSPACE_ROOT="${WORKSPACE_ROOT_OVERRIDE}"

BACKEND="$(
    awk '
        /^backend[[:space:]]*:/ {
            sub(/^backend[[:space:]]*:[[:space:]]*/, "")
            sub(/[[:space:]]+#.*$/, "")
            sub(/^[[:space:]]+/, "")
            sub(/[[:space:]]+$/, "")
            print
            exit
        }
    ' "${CONFIG}"
)"
case "${BACKEND}" in
    \"*\") BACKEND="${BACKEND#\"}"; BACKEND="${BACKEND%\"}" ;;
    \'*\') BACKEND="${BACKEND#\'}"; BACKEND="${BACKEND%\'}" ;;
esac
BACKEND="${BACKEND:-pytorch_smolvla}"
if [[ "${BACKEND}" == "pytorch_smolvla" ]]; then
    if [[ -n "${SMOLVLA_TORCH_PYTHON:-}" ]]; then
        PYTHON_BIN="${SMOLVLA_TORCH_PYTHON}"
    elif [[ -x "${WORKSPACE_ROOT:-/workspace}/venvs/smolvla_torch/bin/python" ]]; then
        PYTHON_BIN="${WORKSPACE_ROOT:-/workspace}/venvs/smolvla_torch/bin/python"
    elif [[ -x "${ROOT}/.venv-smolvla-torch/bin/python" ]]; then
        PYTHON_BIN="${ROOT}/.venv-smolvla-torch/bin/python"
    else
        echo "Official LeRobot SmolVLA Python is unavailable." >&2
        echo "Run: bash ${ROOT}/scripts/setup_env.sh --smolvla" >&2
        exit 2
    fi
    if [[ ! -x "${PYTHON_BIN}" ]]; then
        echo "SMOLVLA_TORCH_PYTHON is not executable: ${PYTHON_BIN}" >&2
        echo "Run: bash ${ROOT}/scripts/setup_env.sh --smolvla" >&2
        exit 2
    fi
elif [[ -n "${FRS_PYTHON:-}" ]]; then
    PYTHON_BIN="${FRS_PYTHON}"
elif [[ -n "${VB3_PYTHON:-}" ]]; then
    PYTHON_BIN="${VB3_PYTHON}"
elif [[ -x "${ROOT}/.venv/bin/python" ]]; then
    PYTHON_BIN="${ROOT}/.venv/bin/python"
else
    PYTHON_BIN="python3"
fi

if [[ "${CHECK_ONLY}" == true ]]; then
    echo "config=${CONFIG}"
    echo "token_source=${token_source}"
    echo "model_cache=${HF_HUB_CACHE}"
    echo "python=${PYTHON_BIN}"
    echo "entrypoint=deploy_smolvla.remote_client"
    if [[ -n "${MAX_ITERATIONS}" ]]; then
        echo "max_iterations=${MAX_ITERATIONS}"
    fi
    exit 0
fi

unset TRANSFORMERS_CACHE PYTORCH_TRANSFORMERS_CACHE PYTORCH_PRETRAINED_BERT_CACHE
export PYTHONUNBUFFERED=1
export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
CLIENT_ARGS=()
if [[ -n "${MAX_ITERATIONS}" ]]; then
    CLIENT_ARGS+=(--max-iterations "${MAX_ITERATIONS}")
fi
exec "${PYTHON_BIN}" -m deploy_smolvla.remote_client \
    --config "${CONFIG}" "${CLIENT_ARGS[@]}"
