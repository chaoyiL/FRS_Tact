#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN=${PYTHON_BIN:-python3}
GPU_ID=${GPU_ID:-0}
DATASET_PATH=${DATASET_PATH:-data/pick_tube_01_06_pca30_rdp_zarr}
OUTPUT_ROOT=${OUTPUT_ROOT:-data/outputs/pick_tube}
LOGGING_MODE=${LOGGING_MODE:-online}
STAGE=${1:-all}
TIMESTAMP=${TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}
AT_OUTPUT_DIR=${AT_OUTPUT_DIR:-${OUTPUT_ROOT}/at_${TIMESTAMP}}
LDP_OUTPUT_DIR=${LDP_OUTPUT_DIR:-${OUTPUT_ROOT}/ldp_${TIMESTAMP}}
BASELINE_JSON=${BASELINE_JSON:-}

if [[ -z "${BASELINE_JSON}" ]]; then
  echo "warning: BASELINE_JSON is unset; training will run, but checkpoints will remain non-deployable and will not enter top-k." >&2
elif [[ ! -f "${BASELINE_JSON}" ]]; then
  echo "Validation baseline JSON not found: ${BASELINE_JSON}" >&2
  exit 1
fi

if [[ "${STAGE}" == "at" || "${STAGE}" == "all" ]]; then
  CUDA_VISIBLE_DEVICES=${GPU_ID} "${PYTHON_BIN}" train.py \
    --config-name=train_pick_tube_at_workspace \
    task.dataset_path="${DATASET_PATH}" \
    hydra.run.dir="${AT_OUTPUT_DIR}" \
    validation.baseline_json="${BASELINE_JSON}" \
    logging.mode="${LOGGING_MODE}"
fi

AT_LOAD_DIR=${AT_LOAD_DIR:-${AT_OUTPUT_DIR}/checkpoints/latest.ckpt}
if [[ "${STAGE}" == "ldp" || "${STAGE}" == "all" ]]; then
  if [[ ! -f "${AT_LOAD_DIR}" ]]; then
    echo "AT checkpoint not found: ${AT_LOAD_DIR}" >&2
    exit 1
  fi
  CUDA_VISIBLE_DEVICES=${GPU_ID} accelerate launch train.py \
    --config-name=train_pick_tube_ldp_workspace \
    task.dataset_path="${DATASET_PATH}" \
    at_load_dir="${AT_LOAD_DIR}" \
    hydra.run.dir="${LDP_OUTPUT_DIR}" \
    validation.baseline_json="${BASELINE_JSON}" \
    logging.mode="${LOGGING_MODE}"
fi
