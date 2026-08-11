#!/usr/bin/env bash

set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"
K8_CONFIG="${PROJECT_ROOT}/configs/train_vtsmolvla_jax_tactile16.yaml"
ENV_FILE="${PROJECT_ROOT}/.env.frs"
LOG_ROOT=""
GPUS="0,1,2,3"

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
        --gpus)
            (($# >= 2)) || fail "--gpus requires comma-separated GPU IDs"
            GPUS="$2"
            shift 2
            ;;
        *) fail "unknown argument: $1" ;;
    esac
done

if [[ "${K8_CONFIG}" != /* ]]; then
    K8_CONFIG="${PROJECT_ROOT}/${K8_CONFIG}"
fi
[[ -f "${K8_CONFIG}" ]] || fail "config does not exist: ${K8_CONFIG}"

[[ "${GPUS}" =~ ^[0-9]+(,[0-9]+){0,3}$ ]] || \
    fail "--gpus must contain one to four comma-separated GPU IDs"
IFS=, read -r -a GPU_IDS <<< "${GPUS}"
declare -A SEEN_GPUS=()
for gpu_index in "${GPU_IDS[@]}"; do
    [[ -z "${SEEN_GPUS[${gpu_index}]:-}" ]] || fail "duplicate GPU ID: ${gpu_index}"
    SEEN_GPUS["${gpu_index}"]=1
done

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

USE_TACTILE="$("${UV_BIN}" run --no-sync python - "${K8_CONFIG}" <<'PY'
import pathlib
import sys

import yaml

with pathlib.Path(sys.argv[1]).open(encoding="utf-8") as file:
    config = yaml.safe_load(file) or {}
print("1" if bool((config.get("model") or {}).get("use_tactile_encoder", False)) else "0")
PY
)"

if [[ "${USE_TACTILE}" == "1" ]]; then
    echo "[vtsmolvla-precompute] tactile cache on GPU ${GPU_IDS[0]}"
    CUDA_VISIBLE_DEVICES="${GPU_IDS[0]}" "${UV_BIN}" run --no-sync python tools/precompute_tactile_embeddings.py \
        --config "${K8_CONFIG}" 2>&1 | tee -a "${LOG_ROOT}/tactile.log"
else
    echo "[vtsmolvla-precompute] visual-only config: skip tactile cache"
fi

OFFLINE_ENABLED="$("${UV_BIN}" run --no-sync python - "${K8_CONFIG}" <<'PY'
import pathlib
import sys

import yaml

with pathlib.Path(sys.argv[1]).open(encoding="utf-8") as file:
    config = yaml.safe_load(file) or {}
print("1" if bool((config.get("offline_training_cache") or {}).get("enabled", False)) else "0")
PY
)"
if [[ "${OFFLINE_ENABLED}" != "1" ]]; then
    echo "[vtsmolvla-precompute] offline training cache disabled: skip vision cache"
    exit 0
fi

dataset_count=6
gpu_count="${#GPU_IDS[@]}"

run_gpu_queue() {
    local gpu_slot="$1"
    local gpu_index="${GPU_IDS[gpu_slot]}"
    local dataset_index status
    for ((dataset_index = gpu_slot; dataset_index < dataset_count; dataset_index += gpu_count)); do
        echo "[vtsmolvla-precompute] offline dataset ${dataset_index} on GPU ${gpu_index}"
        if CUDA_VISIBLE_DEVICES="${gpu_index}" \
            "${UV_BIN}" run --no-sync python tools/precompute_smolvla_training_cache.py \
            --config "${K8_CONFIG}" --dataset-index "${dataset_index}" \
            > >(tee -a "${LOG_ROOT}/offline_dataset_${dataset_index}.log") 2>&1
        then
            continue
        else
            status=$?
            echo "[vtsmolvla-precompute] offline dataset ${dataset_index} failed with status ${status}" >&2
            return "${status}"
        fi
    done
}

queue_pids=()
for ((gpu_slot = 0; gpu_slot < gpu_count; gpu_slot++)); do
    run_gpu_queue "${gpu_slot}" &
    queue_pids+=("$!")
done

queue_failed=0
for queue_pid in "${queue_pids[@]}"; do
    if ! wait "${queue_pid}"; then
        queue_failed=1
    fi
done
((queue_failed == 0)) || exit 1
echo "[vtsmolvla-precompute] all six offline datasets are complete"
