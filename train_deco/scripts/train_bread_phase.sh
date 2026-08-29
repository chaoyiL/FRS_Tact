#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-${PROJECT_ROOT}/train_deco/.venv/bin/python}"

MANIFEST="${BREAD_MANIFEST:-${PROJECT_ROOT}/train_deco/data_manifests/bread_01_03.json}"
DRY_RUN=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --manifest) MANIFEST="$2"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help)
      echo "用法: bash train_deco/scripts/train_bread_phase.sh [--manifest PATH] [--dry-run]"
      exit 0
      ;;
    *) echo "未知参数: $1" >&2; exit 2 ;;
  esac
done

RUN_ID="${RUN_ID:-bread-deco-phase-v3}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/train_deco/outputs}"
BATCH_SIZE="${BATCH_SIZE:-8}"
WORKERS="${WORKERS:-4}"
EPOCHS="${EPOCHS:-100}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
BACKBONE_WEIGHTS="${BACKBONE_WEIGHTS:-${PROJECT_ROOT}/train_deco/pretrained/resnet34-b627a593}"

COMMAND=(
  "${PYTHON_BIN}" -m train_deco.bread_phase.train
  --dataset-manifest "${MANIFEST}"
  --output-dir "${OUTPUT_DIR}"
  --run-id "${RUN_ID}"
  --epochs "${EPOCHS}"
  --batch-size "${BATCH_SIZE}"
  --workers "${WORKERS}"
  --lr 1e-4 --lr-final 5e-6
  --backbone-lr 1e-5 --backbone-lr-final 5e-7
  --hidden-dim 512 --layers 6 --heads 8
  --image-size 256 --rope-height 256 --rope-width 256
  --inference-steps 5
  --backbone-weights "${BACKBONE_WEIGHTS}"
  --save-every 10 --keep-last-checkpoints 5
  --stage 1 --dataset-format lerobot-v21 --action-chunk-size 32
  --bread-phase --use-task-condition
  --augmentation-enabled
  --augmentation-identity-probability 0.25
  --augmentation-low-light-probability 0.0
  --augmentation-mild-probability 0.75
  --augmentation-exposure-probability 0.5
  --augmentation-exposure-range 0.8 1.2
  --augmentation-gamma-range 0.9 1.1
  --augmentation-mild-brightness-range 0.8 1.2
  --augmentation-contrast-range 0.85 1.30
  --augmentation-saturation-range 0.80 1.15
  --augmentation-blur-probability 0.20
  --augmentation-blur-kernel-sizes 3 5
  --augmentation-blur-sigma-range 0.1 1.0
)

printf 'CUDA_VISIBLE_DEVICES=%q ' "${CUDA_VISIBLE_DEVICES}"
printf '%q ' "${COMMAND[@]}"
printf '\n'
if [[ ${DRY_RUN} -eq 1 ]]; then
  exit 0
fi

export CUDA_VISIBLE_DEVICES
cd "${PROJECT_ROOT}"
exec "${COMMAND[@]}"
