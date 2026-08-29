#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
RDP_DIR=$(cd -- "${SCRIPT_DIR}/.." && pwd)
cd "${RDP_DIR}"

STAGE=${1:-help}
PROFILE=${2:-4090}
PYTHON_BIN=${PYTHON_BIN:-${RDP_DIR}/.venv/bin/python}
ACCELERATE_BIN=${ACCELERATE_BIN:-${RDP_DIR}/.venv/bin/accelerate}
DATASET_PATH=${DATASET_PATH:-${RDP_DIR}/data/pick_tube_01_06_pca30_rdp_zarr}
OUTPUT_ROOT=${OUTPUT_ROOT:-${RDP_DIR}/data/outputs/pick_tube_01_06}
RUN_ID=${RUN_ID:-$(date +%Y%m%d_%H%M%S)}
LOGGING_MODE=${LOGGING_MODE:-offline}
DRY_RUN=${DRY_RUN:-0}
MASTER_PORT=${MASTER_PORT:-29500}
BASELINE_JSON=${BASELINE_JSON:-}

usage() {
  cat <<USAGE
Usage: $0 <at|ldp|all|smoke> <4090|rtxpro6000x2>

Examples:
  $0 all 4090
  $0 at rtxpro6000x2
  AT_CKPT=/path/to/latest.ckpt $0 ldp rtxpro6000x2

AT is single-GPU. The rtxpro6000x2 profile uses both GPUs only for LDP.
Common overrides: GPU_IDS, DATASET_PATH, OUTPUT_ROOT, AT_CKPT,
AT_BATCH, LDP_BATCH, AT_EPOCHS, LDP_EPOCHS, NUM_WORKERS, LOGGING_MODE,
BASELINE_JSON=/absolute/path/to/frozen_v1_validation_metrics.json.
Use DRY_RUN=1 to print commands without training.
USAGE
}

case "${PROFILE}" in
  4090)
    GPU_IDS=${GPU_IDS:-0}
    LDP_PROCESSES=1
    AT_BATCH=${AT_BATCH:-64}
    LDP_BATCH=${LDP_BATCH:-64}
    NUM_WORKERS=${NUM_WORKERS:-8}
    ;;
  rtxpro6000x2)
    GPU_IDS=${GPU_IDS:-0,1}
    LDP_PROCESSES=2
    AT_BATCH=${AT_BATCH:-64}
    LDP_BATCH=${LDP_BATCH:-64}
    NUM_WORKERS=${NUM_WORKERS:-8}
    ;;
  *)
    usage >&2
    exit 2
    ;;
esac

AT_EPOCHS=${AT_EPOCHS:-20}
LDP_EPOCHS=${LDP_EPOCHS:-10}
AT_PERIODIC_KEEP=${AT_PERIODIC_KEEP:-10}
LDP_PERIODIC_KEEP=${LDP_PERIODIC_KEEP:-10}
AT_DIR=${AT_DIR:-${OUTPUT_ROOT}/at_${RUN_ID}}
LDP_DIR=${LDP_DIR:-${OUTPUT_ROOT}/ldp_${RUN_ID}}

if (( ${#RUN_ID} > 64 )); then
  echo "RUN_ID must be at most 64 characters for W&B tags (${#RUN_ID}): ${RUN_ID}" >&2
  exit 2
fi

run() {
  printf '+ '
  printf '%q ' "$@"
  printf '\n'
  if [[ "${DRY_RUN}" != "1" ]]; then
    "$@"
  fi
}

require_data() {
  if [[ ! -d "${DATASET_PATH}/replay_buffer.zarr" ]]; then
    echo "Dataset not found: ${DATASET_PATH}/replay_buffer.zarr" >&2
    echo "Run scripts/setup_pick_tube_data.sh validate first." >&2
    exit 1
  fi
}

require_baseline() {
  if [[ -z "${BASELINE_JSON}" ]]; then
    echo "warning: BASELINE_JSON is unset; training will run, but checkpoints will remain non-deployable and will not enter top-k." >&2
    return
  fi
  if [[ ! -f "${BASELINE_JSON}" ]]; then
    echo "Validation baseline JSON not found: ${BASELINE_JSON}" >&2
    exit 1
  fi
}

train_at() {
  local debug=$1
  local at_gpu=${GPU_IDS%%,*}
  local args=(
    "${PYTHON_BIN}" train.py
    --config-name=train_pick_tube_at_workspace
    "task.dataset_path=${DATASET_PATH}"
    "hydra.run.dir=${AT_DIR}"
    "logging.mode=${LOGGING_MODE}"
    "training.num_epochs=${AT_EPOCHS}"
    "dataloader.batch_size=${AT_BATCH}"
    "val_dataloader.batch_size=${AT_BATCH}"
    "dataloader.num_workers=${NUM_WORKERS}"
    "val_dataloader.num_workers=${NUM_WORKERS}"
    "validation.baseline_json=${BASELINE_JSON}"
    "checkpoint.periodic.keep=${AT_PERIODIC_KEEP}"
  )
  if [[ "${debug}" == "1" ]]; then
    args+=(training.debug=true task.dataset.max_train_episodes=1)
  fi
  run env "CUDA_VISIBLE_DEVICES=${at_gpu}" "${args[@]}"

  AT_CKPT=${AT_DIR}/checkpoints/latest.ckpt
  if [[ "${DRY_RUN}" != "1" ]]; then
    if [[ ! -f "${AT_CKPT}" ]]; then
      echo "AT checkpoint was not produced: ${AT_CKPT}" >&2
      exit 1
    fi
  fi
  echo "AT checkpoint: ${AT_CKPT}"
}

resolve_at_checkpoint() {
  AT_CKPT=${AT_CKPT:-${AT_DIR}/checkpoints/latest.ckpt}
}

train_ldp() {
  local debug=$1
  resolve_at_checkpoint
  if [[ "${DRY_RUN}" != "1" && ! -f "${AT_CKPT}" ]]; then
    echo "AT checkpoint not found: ${AT_CKPT}" >&2
    exit 1
  fi

  local launch=(
    env "CUDA_VISIBLE_DEVICES=${GPU_IDS}"
    "${ACCELERATE_BIN}" launch
    --num_processes "${LDP_PROCESSES}"
    --num_machines 1
    --dynamo_backend no
  )
  if [[ "${LDP_PROCESSES}" == "2" ]]; then
    launch+=(--multi_gpu --main_process_port "${MASTER_PORT}")
  fi
  local args=(
    train.py
    --config-name=train_pick_tube_ldp_workspace
    "task.dataset_path=${DATASET_PATH}"
    "at_load_dir=${AT_CKPT}"
    "hydra.run.dir=${LDP_DIR}"
    "logging.mode=${LOGGING_MODE}"
    "training.num_epochs=${LDP_EPOCHS}"
    "dataloader.batch_size=${LDP_BATCH}"
    "val_dataloader.batch_size=${LDP_BATCH}"
    "dataloader.num_workers=${NUM_WORKERS}"
    "val_dataloader.num_workers=${NUM_WORKERS}"
    "validation.baseline_json=${BASELINE_JSON}"
    "checkpoint.periodic.keep=${LDP_PERIODIC_KEEP}"
  )
  if [[ "${debug}" == "1" ]]; then
    args+=(training.debug=true task.dataset.max_train_episodes=1)
  fi
  run "${launch[@]}" "${args[@]}"
  echo "LDP output: ${LDP_DIR}"
}

case "${STAGE}" in
  at)
    require_baseline
    require_data
    train_at 0
    ;;
  ldp)
    require_baseline
    require_data
    train_ldp 0
    ;;
  all)
    require_baseline
    require_data
    train_at 0
    train_ldp 0
    ;;
  smoke)
    require_baseline
    require_data
    AT_DIR=${AT_DIR}_smoke
    LDP_DIR=${LDP_DIR}_smoke
    AT_BATCH=${AT_SMOKE_BATCH:-4}
    LDP_BATCH=${LDP_SMOKE_BATCH:-2}
    train_at 1
    train_ldp 1
    ;;
  -h|--help|help)
    usage
    ;;
  *)
    usage >&2
    exit 2
    ;;
esac
