#!/usr/bin/env bash
set -euo pipefail

# Single-GPU pick-tube PCA30 training entry point for an RTX PRO 6000.
# AT is trained first; LDP then loads the AT latest checkpoint.

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
RDP_DIR=$(cd -- "${SCRIPT_DIR}/.." && pwd)
cd "${RDP_DIR}"

STAGE=${1:-help}
GPU_ID=${GPU_ID:-0}
PYTHON_BIN=${PYTHON_BIN:-${RDP_DIR}/.venv/bin/python}
ACCELERATE_BIN=${ACCELERATE_BIN:-${RDP_DIR}/.venv/bin/accelerate}
DATASET_PATH=${DATASET_PATH:-${RDP_DIR}/data/pick_tube_01_06_pca30_rdp_zarr}
OUTPUT_ROOT=${OUTPUT_ROOT:-${RDP_DIR}/data/outputs/pick_tube_01_06}
RUN_ID=${RUN_ID:-pca30_latent32_full6_$(date +%Y%m%d_%H%M%S)}
LOGGING_MODE=${LOGGING_MODE:-offline}
MIXED_PRECISION=${MIXED_PRECISION:-bf16}
AT_EPOCHS=${AT_EPOCHS:-20}
LDP_EPOCHS=${LDP_EPOCHS:-10}
AT_BATCH=${AT_BATCH:-64}
LDP_BATCH=${LDP_BATCH:-64}
NUM_WORKERS=${NUM_WORKERS:-8}
TACTILE_DIM=${TACTILE_DIM:-30}
AT_CHECKPOINT_EVERY=${AT_CHECKPOINT_EVERY:-1}
LDP_CHECKPOINT_EVERY=${LDP_CHECKPOINT_EVERY:-1}
AT_CHECKPOINT_KEEP=${AT_CHECKPOINT_KEEP:-1}
LDP_CHECKPOINT_KEEP=${LDP_CHECKPOINT_KEEP:-1}
AT_PERIODIC_KEEP=${AT_PERIODIC_KEEP:-10}
LDP_PERIODIC_KEEP=${LDP_PERIODIC_KEEP:-10}
RESUME=${RESUME:-true}
VALIDATE_DATASET=${VALIDATE_DATASET:-1}
DRY_RUN=${DRY_RUN:-0}
BASELINE_JSON=${BASELINE_JSON:-}

AT_DIR=${AT_DIR:-${OUTPUT_ROOT}/at_${RUN_ID}}
LDP_DIR=${LDP_DIR:-${OUTPUT_ROOT}/ldp_${RUN_ID}}

usage() {
  cat <<'USAGE'
Usage: bash scripts/train_pick_tube_single_gpu.sh <at|ldp|all>

Examples:
  # Full six-dataset training on physical GPU 0
  RUN_ID=pca30_latent32_full6_v1 \
    bash scripts/train_pick_tube_single_gpu.sh all

  # Resume the same run; num_epochs is the target total, not extra epochs
  RUN_ID=pca30_latent32_full6_v1 \
    bash scripts/train_pick_tube_single_gpu.sh all

  # Train only LDP from an existing AT checkpoint
  AT_CKPT=/absolute/path/to/at/checkpoints/latest.ckpt \
    RUN_ID=pca30_latent32_full6_v2 \
    bash scripts/train_pick_tube_single_gpu.sh ldp

Common environment overrides:
  GPU_ID=0
  DATASET_PATH=/absolute/path/to/pick_tube_01_06_pca30_rdp_zarr
  OUTPUT_ROOT=/absolute/path/to/outputs
  RUN_ID=pca30_latent32_full6_v1
  AT_EPOCHS=20 LDP_EPOCHS=10
  AT_BATCH=64 LDP_BATCH=64 NUM_WORKERS=8
  TACTILE_DIM=30              # total PCA output dimension across both arms
  MIXED_PRECISION=bf16        # use "no" to disable Accelerate mixed precision
  AT_CHECKPOINT_EVERY=1 LDP_CHECKPOINT_EVERY=1
  AT_CHECKPOINT_KEEP=1 LDP_CHECKPOINT_KEEP=1
  AT_PERIODIC_KEEP=10 LDP_PERIODIC_KEEP=10
  RESUME=true VALIDATE_DATASET=1 LOGGING_MODE=offline
  BASELINE_JSON=/absolute/path/to/frozen_v1_validation_metrics.json
  DRY_RUN=1                  # print commands without starting training
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

case "${STAGE}" in
  at|ldp|all)
    ;;
  -h|--help|help)
    usage
    exit 0
    ;;
  *)
    usage >&2
    exit 2
    ;;
esac

if [[ ! "${GPU_ID}" =~ ^[0-9]+$ ]]; then
  echo "GPU_ID must be a non-negative integer, got: ${GPU_ID}" >&2
  exit 2
fi
if [[ ! "${TACTILE_DIM}" =~ ^[1-9][0-9]*$ ]]; then
  echo "TACTILE_DIM must be a positive integer, got: ${TACTILE_DIM}" >&2
  exit 2
fi
if [[ "${RESUME}" != "true" && "${RESUME}" != "false" ]]; then
  echo "RESUME must be true or false, got: ${RESUME}" >&2
  exit 2
fi
if [[ "${MIXED_PRECISION}" != "no" && "${MIXED_PRECISION}" != "fp16" && "${MIXED_PRECISION}" != "bf16" ]]; then
  echo "MIXED_PRECISION must be no, fp16, or bf16, got: ${MIXED_PRECISION}" >&2
  exit 2
fi
for value_name in AT_CHECKPOINT_KEEP LDP_CHECKPOINT_KEEP AT_PERIODIC_KEEP LDP_PERIODIC_KEEP; do
  value=${!value_name}
  if [[ ! "${value}" =~ ^[0-9]+$ ]]; then
    echo "${value_name} must be a non-negative integer, got: ${value}" >&2
    exit 2
  fi
done
if (( ${#RUN_ID} > 64 )); then
  echo "RUN_ID must be at most 64 characters for W&B tags (${#RUN_ID}): ${RUN_ID}" >&2
  exit 2
fi
if [[ -z "${BASELINE_JSON}" ]]; then
  echo "warning: BASELINE_JSON is unset; training will run, but checkpoints will remain non-deployable and will not enter top-k." >&2
elif [[ ! -f "${BASELINE_JSON}" ]]; then
  echo "Validation baseline JSON not found: ${BASELINE_JSON}" >&2
  exit 1
fi

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Training Python not found: ${PYTHON_BIN}" >&2
  echo "Run: bash scripts/install_pick_tube_training_env.sh" >&2
  exit 1
fi
if [[ ! -x "${ACCELERATE_BIN}" ]]; then
  echo "Accelerate CLI not found: ${ACCELERATE_BIN}" >&2
  echo "Run: bash scripts/install_pick_tube_training_env.sh" >&2
  exit 1
fi
if [[ ! -d "${DATASET_PATH}/replay_buffer.zarr" ]]; then
  echo "Dataset not found: ${DATASET_PATH}/replay_buffer.zarr" >&2
  echo "Generate it with scripts/setup_pick_tube_data.sh convert." >&2
  exit 1
fi

if [[ "${DRY_RUN}" != "1" ]]; then
  CUDA_VISIBLE_DEVICES=${GPU_ID} "${PYTHON_BIN}" - <<'PY'
import torch

if not torch.cuda.is_available():
    raise SystemExit("CUDA is unavailable in the training environment")
if torch.cuda.device_count() != 1:
    raise SystemExit(
        f"single-GPU launcher expected one visible CUDA device, got {torch.cuda.device_count()}"
    )
name = torch.cuda.get_device_name(0)
capability = torch.cuda.get_device_capability(0)
print(f"Using CUDA device 0: {name} (compute capability {capability[0]}.{capability[1]})")
if "RTX PRO 6000" not in name.upper():
    print("warning: the selected GPU is not identified as an RTX PRO 6000")
PY
fi

if [[ "${VALIDATE_DATASET}" == "1" ]]; then
  run env \
    "PYTHON_BIN=${PYTHON_BIN}" \
    "DATASET_PATH=${DATASET_PATH}" \
    bash scripts/setup_pick_tube_data.sh validate
fi

mkdir -p "${OUTPUT_ROOT}"

train_at() {
  local args=(
    "${PYTHON_BIN}" train.py
    --config-name=train_pick_tube_at_workspace
    "task.dataset_path=${DATASET_PATH}"
    "task.tactile_embedding_dim=${TACTILE_DIM}"
    "exp_name=${RUN_ID}"
    "hydra.run.dir=${AT_DIR}"
    "logging.mode=${LOGGING_MODE}"
    "training.device=cuda:0"
    "training.resume=${RESUME}"
    "training.num_epochs=${AT_EPOCHS}"
    "training.checkpoint_every=${AT_CHECKPOINT_EVERY}"
    "checkpoint.topk.k=${AT_CHECKPOINT_KEEP}"
    "checkpoint.periodic.keep=${AT_PERIODIC_KEEP}"
    "+training.mixed_precision=${MIXED_PRECISION}"
    "dataloader.batch_size=${AT_BATCH}"
    "val_dataloader.batch_size=${AT_BATCH}"
    "dataloader.num_workers=${NUM_WORKERS}"
    "val_dataloader.num_workers=${NUM_WORKERS}"
    "validation.baseline_json=${BASELINE_JSON}"
  )
  run env "CUDA_VISIBLE_DEVICES=${GPU_ID}" "${args[@]}"

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
  resolve_at_checkpoint
  if [[ "${DRY_RUN}" != "1" && ! -f "${AT_CKPT}" ]]; then
    echo "AT checkpoint not found: ${AT_CKPT}" >&2
    exit 1
  fi

  local args=(
    "${ACCELERATE_BIN}" launch
    --num_processes 1
    --num_machines 1
    --dynamo_backend no
    --mixed_precision "${MIXED_PRECISION}"
    train.py
    --config-name=train_pick_tube_ldp_workspace
    "task.dataset_path=${DATASET_PATH}"
    "task.tactile_embedding_dim=${TACTILE_DIM}"
    "exp_name=${RUN_ID}"
    # Keep the checkpoint path quoted for Hydra's override parser. Checkpoint
    # filenames commonly contain '=' (for example epoch=0019-train_loss=...).
    "at_load_dir='${AT_CKPT}'"
    "hydra.run.dir=${LDP_DIR}"
    "logging.mode=${LOGGING_MODE}"
    "training.resume=${RESUME}"
    "training.num_epochs=${LDP_EPOCHS}"
    "training.checkpoint_every=${LDP_CHECKPOINT_EVERY}"
    "checkpoint.topk.k=${LDP_CHECKPOINT_KEEP}"
    "checkpoint.periodic.keep=${LDP_PERIODIC_KEEP}"
    "dataloader.batch_size=${LDP_BATCH}"
    "val_dataloader.batch_size=${LDP_BATCH}"
    "dataloader.num_workers=${NUM_WORKERS}"
    "val_dataloader.num_workers=${NUM_WORKERS}"
    "validation.baseline_json=${BASELINE_JSON}"
  )
  run env "CUDA_VISIBLE_DEVICES=${GPU_ID}" "${args[@]}"
  echo "LDP output: ${LDP_DIR}"
}

case "${STAGE}" in
  at)
    train_at
    ;;
  ldp)
    train_ldp
    ;;
  all)
    train_at
    train_ldp
    ;;
esac

printf '\nTraining stage complete.\n'
printf 'AT directory:  %s\n' "${AT_DIR}"
printf 'LDP directory: %s\n' "${LDP_DIR}"
