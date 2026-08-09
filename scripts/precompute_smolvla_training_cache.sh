#!/usr/bin/env bash

set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"
K8_CONFIG="${PROJECT_ROOT}/configs/train_vtsmolvla_jax_tactile16.yaml"
ENV_FILE="${PROJECT_ROOT}/.env.frs"
LOG_ROOT=""

fail() {
    echo "[vtsmolvla-precompute] error: $*" >&2
    exit 1
}

while (($#)); do
    case "$1" in
        --config)
            (($# >= 2)) || fail "--config requires a path"
            K8_CONFIG="$2"
            shift 2
            ;;
        --log-root)
            (($# >= 2)) || fail "--log-root requires a path"
            LOG_ROOT="$2"
            shift 2
            ;;
        *) fail "unknown argument: $1" ;;
    esac
done

if [[ "${K8_CONFIG}" != /* ]]; then
    K8_CONFIG="${PROJECT_ROOT}/${K8_CONFIG}"
fi
[[ -f "${K8_CONFIG}" ]] || fail "config does not exist: ${K8_CONFIG}"

if [[ -f "${ENV_FILE}" ]]; then
    # shellcheck disable=SC1090
    source "${ENV_FILE}"
fi

if command -v uv >/dev/null 2>&1; then
    UV_BIN="$(command -v uv)"
elif [[ -x "${HOME}/.local/bin/uv" ]]; then
    UV_BIN="${HOME}/.local/bin/uv"
else
    fail "uv is required; run scripts/setup_env.sh first"
fi

STORAGE_ROOT="${FRS_STORAGE_ROOT:-${PROJECT_ROOT}/.cache}"
LOG_ROOT="${LOG_ROOT:-${STORAGE_ROOT}/logs/vtsmolvla-precompute}"
mkdir -p "${LOG_ROOT}"

echo "[vtsmolvla-precompute] tactile cache on GPU 0"
CUDA_VISIBLE_DEVICES=0 "${UV_BIN}" run --no-sync python tools/precompute_tactile_embeddings.py \
    --config "${K8_CONFIG}" 2>&1 | tee -a "${LOG_ROOT}/tactile.log"

declare -a cache_pids=()
for dataset_index in 0 1 2 3; do
    echo "[vtsmolvla-precompute] offline dataset ${dataset_index} on GPU ${dataset_index}"
    CUDA_VISIBLE_DEVICES="${dataset_index}" \
        "${UV_BIN}" run --no-sync python tools/precompute_smolvla_training_cache.py \
        --config "${K8_CONFIG}" --dataset-index "${dataset_index}" \
        > >(tee -a "${LOG_ROOT}/offline_dataset_${dataset_index}.log") 2>&1 &
    cache_pids+=("$!")
done

cache_failed=0
for dataset_index in 0 1 2 3; do
    if wait "${cache_pids[dataset_index]}"; then
        status=0
    else
        status=$?
    fi
    if ((status != 0)); then
        echo "[vtsmolvla-precompute] offline dataset ${dataset_index} failed with status ${status}" >&2
        cache_failed=1
    fi
done

((cache_failed == 0)) || exit 1
echo "[vtsmolvla-precompute] all four offline datasets are complete"
