#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
RDP_DIR=$(cd -- "${SCRIPT_DIR}/.." && pwd)
cd "${RDP_DIR}"

STAGE=${1:-help}
TASK_SELECTION=${2:-both}

LEROBOT_ROOT=${LEROBOT_ROOT:-/DATA/ljl/substages/lerobot_v21/KaiyueChen}
WORK_ROOT=${WORK_ROOT:-/DATA/ljl/substages/rdp_two_tasks}
TACTILE_CACHE_ROOT=${TACTILE_CACHE_ROOT:-${WORK_ROOT}/tactile_embeddings_encoder0809}
PCA_ROOT=${PCA_ROOT:-${WORK_ROOT}/pca}
DATA_ROOT=${DATA_ROOT:-${WORK_ROOT}/datasets}
OUTPUT_ROOT=${OUTPUT_ROOT:-${WORK_ROOT}/outputs}
ENCODER_DIR=${ENCODER_DIR:-${RDP_DIR}/data/encoder_ckpt_0809}

PYTHON_BIN=${PYTHON_BIN:-${RDP_DIR}/.venv-cu128/bin/python}
ACCELERATE_BIN=${ACCELERATE_BIN:-${RDP_DIR}/.venv-cu128/bin/accelerate}
JAX_PYTHON=${JAX_PYTHON:-${RDP_DIR}/.venv-jax/bin/python}
GPU_ID=${GPU_ID:-0}
MIXED_PRECISION=${MIXED_PRECISION:-bf16}
LOGGING_MODE=${LOGGING_MODE:-online}
AT_BATCH=${AT_BATCH:-64}
LDP_BATCH=${LDP_BATCH:-64}
NUM_WORKERS=${NUM_WORKERS:-8}
AT_EPOCHS=${AT_EPOCHS:-20}
LDP_EPOCHS=${LDP_EPOCHS:-20}
AT_CHECKPOINT_EVERY=${AT_CHECKPOINT_EVERY:-2}
LDP_CHECKPOINT_EVERY=${LDP_CHECKPOINT_EVERY:-2}
AT_CHECKPOINT_KEEP=${AT_CHECKPOINT_KEEP:-20}
LDP_CHECKPOINT_KEEP=${LDP_CHECKPOINT_KEEP:-20}
PRECOMPUTE_BATCH=${PRECOMPUTE_BATCH:-64}
PRECOMPUTE_WORKERS=${PRECOMPUTE_WORKERS:-4}
EXPERIMENT_ID=${EXPERIMENT_ID:-$(date +%Y%m%d_%H%M%S)}
RESUME=${RESUME:-false}
FORCE_PREPARE=${FORCE_PREPARE:-0}
OVERWRITE_TACTILE=${OVERWRITE_TACTILE:-0}
DRY_RUN=${DRY_RUN:-0}

usage() {
  cat <<'USAGE'
Usage: bash scripts/run_two_task_rdp_experiments.sh <stage> <task>

Stages:
  precompute  Encode the four tactile streams into four 512D embeddings.
  prepare     Fit an independent arm-wise PCA30 and convert to RDP Zarr.
  train       Train an independent AT20 followed by LDP20.
  all         Run precompute, prepare, and train sequentially.

Tasks:
  two_tubes   Merge two_tubes_01 and two_tubes_02.
  task2       Merge task2_01 and task2_02.
  both        Run two_tubes first, then task2.

Important overrides:
  LEROBOT_ROOT=/DATA/ljl/substages/lerobot_v21/KaiyueChen
  WORK_ROOT=/DATA/ljl/substages/rdp_two_tasks
  PYTHON_BIN=/home/ljl/FRS_Tact/train_RDP/.venv-cu128/bin/python
  ACCELERATE_BIN=/home/ljl/FRS_Tact/train_RDP/.venv-cu128/bin/accelerate
  JAX_PYTHON=/home/ljl/FRS_Tact/train_RDP/.venv-jax/bin/python
  ENCODER_DIR=/home/ljl/FRS_Tact/train_RDP/data/encoder_ckpt_0809
  GPU_ID=0 NUM_WORKERS=8 LOGGING_MODE=online MIXED_PRECISION=bf16
  EXPERIMENT_ID=baseline_v1 RESUME=false DRY_RUN=1

Each task gets its own tactile cache inputs, PCA artifact, RDP Zarr, AT output,
and LDP output. Checkpoints are saved every two epochs and up to 20 named
checkpoints are retained in addition to latest.ckpt.
USAGE
}

run() {
  printf '+ '
  printf '%q ' "$@"
  printf '\n'
  if [[ "${DRY_RUN}" != "1" ]]; then
    "$@"
  fi
}

select_task() {
  local task=$1
  case "${task}" in
    two_tubes)
      TASK_TAG=two_tubes
      DATASETS=(two_tubes_01 two_tubes_02)
      ;;
    task2)
      TASK_TAG=task2
      DATASETS=(task2_01 task2_02)
      ;;
    *)
      echo "Unknown task: ${task}" >&2
      exit 2
      ;;
  esac
  PCA_PATH=${PCA_ROOT}/tactile_pca_${TASK_TAG}_2x15.npz
  DATASET_PATH=${DATA_ROOT}/${TASK_TAG}_pca30_rdp_zarr
  TASK_OUTPUT_ROOT=${OUTPUT_ROOT}/${TASK_TAG}
}

check_executable() {
  local path=$1
  local label=$2
  if [[ ! -x "${path}" ]]; then
    echo "${label} is not executable: ${path}" >&2
    exit 1
  fi
}

check_dataset_contract() {
  local dataset
  for dataset in "${DATASETS[@]}"; do
    if [[ ! -f "${LEROBOT_ROOT}/${dataset}/meta/info.json" ]]; then
      echo "Dataset metadata not found: ${LEROBOT_ROOT}/${dataset}/meta/info.json" >&2
      exit 1
    fi
    if [[ ! -f "${LEROBOT_ROOT}/${dataset}/meta/episodes.jsonl" ]]; then
      echo "LeRobot v2.1 episodes metadata not found: ${LEROBOT_ROOT}/${dataset}/meta/episodes.jsonl" >&2
      exit 1
    fi
  done

  "${PYTHON_BIN}" - "${LEROBOT_ROOT}" "${DATASETS[@]}" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
required = {
    "observation.images.camera0": [224, 224, 3],
    "observation.images.camera1": [224, 224, 3],
    "observation.images.tactile_left_0": [224, 224, 3],
    "observation.images.tactile_right_0": [224, 224, 3],
    "observation.images.tactile_left_1": [224, 224, 3],
    "observation.images.tactile_right_1": [224, 224, 3],
    "observation.state": [20],
    "actions": [20],
}
for dataset in sys.argv[2:]:
    info = json.loads((root / dataset / "meta" / "info.json").read_text())
    if info.get("codebase_version") != "v2.1":
        raise SystemExit(f"{dataset}: expected LeRobot v2.1, got {info.get('codebase_version')}")
    if int(info.get("fps", -1)) != 30:
        raise SystemExit(f"{dataset}: expected 30 fps, got {info.get('fps')}")
    features = info.get("features", {})
    for key, shape in required.items():
        actual = features.get(key, {}).get("shape")
        if actual != shape:
            raise SystemExit(f"{dataset}: {key} expected {shape}, got {actual}")
    print(
        f"contract OK: {dataset}, episodes={info['total_episodes']}, "
        f"frames={info['total_frames']}"
    )
PY
}

check_encoder() {
  local params
  if [[ ! -f "${ENCODER_DIR}/checkpoint.json" ]]; then
    echo "Tactile encoder metadata not found: ${ENCODER_DIR}/checkpoint.json" >&2
    exit 1
  fi
  params=$("${PYTHON_BIN}" - "${ENCODER_DIR}" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
metadata = json.loads((root / "checkpoint.json").read_text())
print(root / metadata["params_file"])
PY
)
  if [[ ! -f "${params}" ]]; then
    echo "Tactile encoder parameters not found: ${params}" >&2
    exit 1
  fi
}

precompute_task() {
  check_executable "${JAX_PYTHON}" "JAX Python"
  check_encoder
  local args=(
    "${JAX_PYTHON}" precompute_pick_tube_v21_tactile_embeddings.py
    --dataset-root "${LEROBOT_ROOT}"
    --cache-root "${TACTILE_CACHE_ROOT}"
    --encoder-path "${ENCODER_DIR}"
    --batch-size "${PRECOMPUTE_BATCH}"
    --num-workers "${PRECOMPUTE_WORKERS}"
    --datasets "${DATASETS[@]}"
  )
  if [[ "${OVERWRITE_TACTILE}" == "1" ]]; then
    args+=(--overwrite)
  fi
  run "${args[@]}"
}

check_tactile_cache() {
  local dataset
  for dataset in "${DATASETS[@]}"; do
    if [[ ! -f "${TACTILE_CACHE_ROOT}/KaiyueChen/${dataset}/embeddings.npy" ]]; then
      if [[ "${DRY_RUN}" == "1" ]]; then
        echo "Dry run: tactile embeddings will be required for ${dataset}."
        continue
      fi
      echo "Tactile embeddings not found: ${TACTILE_CACHE_ROOT}/KaiyueChen/${dataset}/embeddings.npy" >&2
      echo "Run the precompute stage first." >&2
      exit 1
    fi
    if [[ ! -f "${TACTILE_CACHE_ROOT}/KaiyueChen/${dataset}/metadata.json" ]]; then
      if [[ "${DRY_RUN}" == "1" ]]; then
        echo "Dry run: completed tactile metadata will be required for ${dataset}."
        continue
      fi
      echo "Completed tactile metadata not found for ${dataset}; precompute may be incomplete." >&2
      exit 1
    fi
  done
}

prepare_task() {
  check_tactile_cache
  if [[ "${FORCE_PREPARE}" == "1" || ! -f "${PCA_PATH}" ]]; then
    run "${PYTHON_BIN}" fit_pick_tube_tactile_pca.py \
      --tactile-cache-root "${TACTILE_CACHE_ROOT}" \
      --output "${PCA_PATH}" \
      --components-per-arm 15 \
      --datasets "${DATASETS[@]}"
  else
    echo "Using existing PCA: ${PCA_PATH}"
  fi

  if [[ "${FORCE_PREPARE}" == "1" || ! -d "${DATASET_PATH}/replay_buffer.zarr" ]]; then
    local args=(
      "${PYTHON_BIN}" convert_pick_tube_lerobot_to_rdp_zarr.py
      --dataset-root "${LEROBOT_ROOT}"
      --tactile-cache-root "${TACTILE_CACHE_ROOT}"
      --output-dir "${DATASET_PATH}"
      --tactile-pca-path "${PCA_PATH}"
      --datasets "${DATASETS[@]}"
      --dataset-repeats
    )
    if [[ "${FORCE_PREPARE}" == "1" ]]; then
      args+=(--overwrite)
    fi
    run "${args[@]}"
  else
    echo "Using existing RDP dataset: ${DATASET_PATH}"
  fi

  if [[ "${DRY_RUN}" != "1" ]]; then
    run env \
      "PYTHON_BIN=${PYTHON_BIN}" \
      "DATASET_PATH=${DATASET_PATH}" \
      bash scripts/setup_pick_tube_data.sh validate
  fi
}

train_task() {
  if [[ ! -d "${DATASET_PATH}/replay_buffer.zarr" ]]; then
    if [[ "${DRY_RUN}" == "1" ]]; then
      echo "Dry run: RDP dataset will be required at ${DATASET_PATH}/replay_buffer.zarr."
    else
      echo "RDP dataset not found: ${DATASET_PATH}/replay_buffer.zarr" >&2
      echo "Run the prepare stage first." >&2
      exit 1
    fi
  fi
  run env \
    "PYTHON_BIN=${PYTHON_BIN}" \
    "ACCELERATE_BIN=${ACCELERATE_BIN}" \
    "DATASET_PATH=${DATASET_PATH}" \
    "OUTPUT_ROOT=${TASK_OUTPUT_ROOT}" \
    "RUN_ID=${TASK_TAG}_pca30_at${AT_EPOCHS}_ldp${LDP_EPOCHS}_${EXPERIMENT_ID}" \
    "GPU_ID=${GPU_ID}" \
    "LOGGING_MODE=${LOGGING_MODE}" \
    "MIXED_PRECISION=${MIXED_PRECISION}" \
    "TACTILE_DIM=30" \
    "AT_EPOCHS=${AT_EPOCHS}" \
    "LDP_EPOCHS=${LDP_EPOCHS}" \
    "AT_BATCH=${AT_BATCH}" \
    "LDP_BATCH=${LDP_BATCH}" \
    "NUM_WORKERS=${NUM_WORKERS}" \
    "AT_CHECKPOINT_EVERY=${AT_CHECKPOINT_EVERY}" \
    "LDP_CHECKPOINT_EVERY=${LDP_CHECKPOINT_EVERY}" \
    "AT_CHECKPOINT_KEEP=${AT_CHECKPOINT_KEEP}" \
    "LDP_CHECKPOINT_KEEP=${LDP_CHECKPOINT_KEEP}" \
    "RESUME=${RESUME}" \
    "VALIDATE_DATASET=0" \
    "DRY_RUN=${DRY_RUN}" \
    bash scripts/train_pick_tube_single_gpu.sh all
}

run_task() {
  local task=$1
  select_task "${task}"
  check_dataset_contract
  case "${STAGE}" in
    precompute)
      precompute_task
      ;;
    prepare)
      prepare_task
      ;;
    train)
      train_task
      ;;
    all)
      precompute_task
      prepare_task
      train_task
      ;;
  esac
  printf '\nCompleted stage %s for task %s\n' "${STAGE}" "${TASK_TAG}"
}

case "${STAGE}" in
  precompute|prepare|train|all)
    ;;
  help|-h|--help)
    usage
    exit 0
    ;;
  *)
    usage >&2
    exit 2
    ;;
esac

case "${TASK_SELECTION}" in
  two_tubes|task2)
    check_executable "${PYTHON_BIN}" "Training Python"
    run_task "${TASK_SELECTION}"
    ;;
  both)
    check_executable "${PYTHON_BIN}" "Training Python"
    run_task two_tubes
    run_task task2
    ;;
  *)
    usage >&2
    exit 2
    ;;
esac
