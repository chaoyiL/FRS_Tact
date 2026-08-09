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

echo "[vtsmolvla-precompute] tactile cache on GPU ${GPU_IDS[0]}"
CUDA_VISIBLE_DEVICES="${GPU_IDS[0]}" "${UV_BIN}" run --no-sync python tools/precompute_tactile_embeddings.py \
    --config "${K8_CONFIG}" 2>&1 | tee -a "${LOG_ROOT}/tactile.log"

run_cache_wave() {
    local -a dataset_indices=("$@")
    local -a cache_pids=()
    local dataset_index gpu_index pid_index status wave_failed=0
    for dataset_index in "${dataset_indices[@]}"; do
        gpu_index="${GPU_IDS[${#cache_pids[@]}]}"
        echo "[vtsmolvla-precompute] offline dataset ${dataset_index} on GPU ${gpu_index}"
        CUDA_VISIBLE_DEVICES="${gpu_index}" \
            "${UV_BIN}" run --no-sync python tools/precompute_smolvla_training_cache.py \
            --config "${K8_CONFIG}" --dataset-index "${dataset_index}" \
            > >(tee -a "${LOG_ROOT}/offline_dataset_${dataset_index}.log") 2>&1 &
        cache_pids+=("$!")
    done

    for pid_index in "${!cache_pids[@]}"; do
        dataset_index="${dataset_indices[pid_index]}"
        if wait "${cache_pids[pid_index]}"; then
            status=0
        else
            status=$?
        fi
        if ((status != 0)); then
            echo "[vtsmolvla-precompute] offline dataset ${dataset_index} failed with status ${status}" >&2
            wave_failed=1
        fi
    done
    ((wave_failed == 0))
}

dataset_count=6
gpu_count="${#GPU_IDS[@]}"
wave_start=0
while ((wave_start < dataset_count)); do
    wave_indices=()
    for ((gpu_slot = 0; gpu_slot < gpu_count; gpu_slot++)); do
        dataset_index=$((wave_start + gpu_slot))
        ((dataset_index < dataset_count)) || break
        wave_indices+=("${dataset_index}")
    done
    run_cache_wave "${wave_indices[@]}" || exit 1
    wave_start=$((wave_start + gpu_count))
done
echo "[vtsmolvla-precompute] all six offline datasets are complete"
