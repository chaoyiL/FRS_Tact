#!/usr/bin/env bash

set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"
K8_CONFIG="${PROJECT_ROOT}/configs/train_vtsmolvla_jax_tactile16.yaml"
K21_CONFIG="${PROJECT_ROOT}/configs/train_vtsmolvla_jax_tactile32.yaml"
PRECOMPUTE_SCRIPT="${SCRIPT_DIR}/precompute_smolvla_training_cache.sh"
ENV_FILE="${PROJECT_ROOT}/.env.frs"
APPROVED_GPU="NVIDIA RTX PRO 6000 Blackwell Server Edition"
LOG_ROOT=""
FOREGROUND=0
COORDINATOR=0
CHILD=""
ORIGINAL_ARGS=("$@")

fail() {
    echo "[vtsmolvla-dual] error: $*" >&2
    exit 1
}

while (($#)); do
    case "$1" in
        --foreground) FOREGROUND=1; shift ;;
        --coordinator) COORDINATOR=1; shift ;;
        --child)
            (($# >= 2)) || fail "--child requires k8 or k21"
            CHILD="$2"
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
LOG_ROOT="${LOG_ROOT:-${STORAGE_ROOT}/logs/vtsmolvla-train}"
mkdir -p "${LOG_ROOT}"

require_four_approved_gpus() {
    command -v nvidia-smi >/dev/null 2>&1 || fail "nvidia-smi is required"
    mapfile -t gpu_names < <(nvidia-smi --query-gpu=name --format=csv,noheader)
    ((${#gpu_names[@]} == 4)) || fail "expected exactly four GPUs, found ${#gpu_names[@]}"
    local gpu_name
    for gpu_name in "${gpu_names[@]}"; do
        [[ "${gpu_name}" == *"${APPROVED_GPU}"* ]] || fail "unsupported GPU: ${gpu_name}"
    done
}

ensure_training_sessions_absent() {
    local session
    for session in vtsmolvla_k8 vtsmolvla_k21; do
        if tmux has-session -t "${session}" 2>/dev/null; then
            fail "tmux session ${session} already exists; attach or remove it explicitly"
        fi
    done
}

preflight_child() {
    "${UV_BIN}" run --no-sync python - <<'PY'
import jax

approved = "NVIDIA RTX PRO 6000 BLACKWELL SERVER EDITION"
devices = jax.devices()
if len(devices) != 2:
    raise RuntimeError(f"expected exactly two JAX devices, got {len(devices)}: {devices}")
if any(device.platform != "gpu" or approved not in device.device_kind.upper() for device in devices):
    raise RuntimeError(f"expected two approved RTX PRO 6000 Blackwell devices: {devices}")
print(f"JAX devices={devices}")
PY
}

run_child() {
    local name="$1" gpus="$2" config="$3" log_file="$4"
    export CUDA_VISIBLE_DEVICES="${gpus}"
    preflight_child
    echo "[vtsmolvla-dual] ${name} on GPUs ${gpus}"
    "${UV_BIN}" run --no-sync python tools/train_vtsmolvla_jax.py --config "${config}" \
        2>&1 | tee -a "${log_file}"
}

run_foreground_training() {
    run_child K8 0,1 "${K8_CONFIG}" "${LOG_ROOT}/k8.log" &
    k8_pid=$!
    run_child K21 2,3 "${K21_CONFIG}" "${LOG_ROOT}/k21.log" &
    k21_pid=$!

    if wait "${k8_pid}"; then k8_status=0; else k8_status=$?; fi
    if wait "${k21_pid}"; then k21_status=0; else k21_status=$?; fi
    echo "[vtsmolvla-dual] K8 status=${k8_status}" >&2
    echo "[vtsmolvla-dual] K21 status=${k21_status}" >&2
    ((k8_status == 0 && k21_status == 0))
}

child_main() {
    case "${CHILD}" in
        k8) run_child K8 0,1 "${K8_CONFIG}" "${LOG_ROOT}/k8.log" ;;
        k21) run_child K21 2,3 "${K21_CONFIG}" "${LOG_ROOT}/k21.log" ;;
        *) fail "--child must be k8 or k21" ;;
    esac
}

launch_tmux_children() {
    ensure_training_sessions_absent
    local name inner_command
    for name in k8 k21; do
        printf -v inner_command 'bash %q --child %q --log-root %q' "${BASH_SOURCE[0]}" "${name}" "${LOG_ROOT}"
        tmux new-session -d -s "vtsmolvla_${name}" -c "${PROJECT_ROOT}" "${inner_command}"
    done
    echo "[vtsmolvla-dual] training sessions started: vtsmolvla_k8, vtsmolvla_k21"
}

start_coordinator_tmux() {
    command -v tmux >/dev/null 2>&1 || fail "tmux is required unless --foreground is used"
    ensure_training_sessions_absent
    if tmux has-session -t vtsmolvla_prepare 2>/dev/null; then
        fail "tmux session vtsmolvla_prepare already exists; attach or remove it explicitly"
    fi
    local inner_command argument
    printf -v inner_command 'bash %q --coordinator' "${BASH_SOURCE[0]}"
    for argument in "${ORIGINAL_ARGS[@]}"; do
        [[ "${argument}" == "--foreground" ]] || printf -v inner_command '%s %q' "${inner_command}" "${argument}"
    done
    tmux new-session -d -s vtsmolvla_prepare -c "${PROJECT_ROOT}" "${inner_command}"
    echo "[vtsmolvla-dual] preparation coordinator started: vtsmolvla_prepare"
}

if [[ -n "${CHILD}" ]]; then
    child_main
    exit 0
fi

if ((FOREGROUND == 0 && COORDINATOR == 0)); then
    start_coordinator_tmux
    exit 0
fi

require_four_approved_gpus
if ! bash "${PRECOMPUTE_SCRIPT}" --config "${K8_CONFIG}" --gpus 0,1,2,3 --log-root "${LOG_ROOT}"; then
    fail "preparation failed; K8 and K21 were not started"
fi

if ((COORDINATOR)); then
    launch_tmux_children
else
    run_foreground_training
fi
