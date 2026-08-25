#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
PACKAGE_ROOT="${PACKAGE_ROOT:-${PROJECT_ROOT}/train_deco}"
VENV_PATH="${VENV_PATH:-${PACKAGE_ROOT}/.venv}"
PYTHON_BIN="${PYTHON_BIN:-${VENV_PATH}/bin/python}"
TORCHRUN_BIN="${TORCHRUN_BIN:-${VENV_PATH}/bin/torchrun}"

MODE="local-smoke"
MANIFEST=""
RUN_ID=""
RESUME_FROM="${RESUME_FROM:-}"
DRY_RUN=0

usage() {
  cat <<'EOF'
用法：
  bash scripts/pick_tube_vision/03_train.sh --mode local-smoke [选项]
  bash scripts/pick_tube_vision/03_train.sh --mode local-train [选项]
  bash scripts/pick_tube_vision/03_train.sh --mode server-train [选项]

模式：
  local-smoke   本机 RTX 4090 单卡最小链路：16 samples、1 epoch、小模型
  local-train   本机 RTX 4090 单卡正式参数
  server-train  服务器第 3、4 张卡（CUDA 索引 2,3），torchrun/DDP 两进程

选项：
  --manifest PATH    02_prepare_data.sh 生成的 manifest
  --run-id ID        输出运行名；默认自动加入时间戳
  --resume-from PATH 从 checkpoint 恢复
  --dry-run          只打印命令，不启动训练
  -h, --help         显示帮助

常用环境变量覆盖：
  OUTPUT_DIR BATCH_SIZE WORKERS EPOCHS SAVE_EVERY ACTION_CHUNK_SIZE LR BACKBONE_LR
  HIDDEN_DIM LAYERS HEADS IMAGE_SIZE INFERENCE_STEPS CUDA_VISIBLE_DEVICES

--batch-size 是每个 GPU 的 batch size；server-train 默认使用 CUDA 索引 2,3，16/GPU，总 batch 32。
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --mode)
      MODE="$2"
      shift 2
      ;;
    --manifest)
      MANIFEST="$2"
      shift 2
      ;;
    --run-id)
      RUN_ID="$2"
      shift 2
      ;;
    --resume-from)
      RESUME_FROM="$2"
      shift 2
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "未知参数：$1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
BACKBONE_WEIGHTS="${BACKBONE_WEIGHTS:-${PACKAGE_ROOT}/pretrained/resnet34-b627a593}"
VALIDATION_RATIO="${VALIDATION_RATIO:-0.1}"
EPISODE_SPLIT_SEED="${EPISODE_SPLIT_SEED:-42}"

case "${MODE}" in
  local-smoke)
    MANIFEST="${MANIFEST:-${PACKAGE_ROOT}/data_manifests/pick_tube_local.json}"
    RUN_ID="${RUN_ID:-pick_tube_vision_local_smoke_${TIMESTAMP}}"
    OUTPUT_DIR="${OUTPUT_DIR:-${PACKAGE_ROOT}/outputs}"
    BATCH_SIZE="${BATCH_SIZE:-2}"
    WORKERS="${WORKERS:-0}"
    EPOCHS="${EPOCHS:-1}"
    ACTION_CHUNK_SIZE="${ACTION_CHUNK_SIZE:-4}"
    HIDDEN_DIM="${HIDDEN_DIM:-64}"
    LAYERS="${LAYERS:-1}"
    HEADS="${HEADS:-4}"
    IMAGE_SIZE="${IMAGE_SIZE:-64}"
    ROPE_HEIGHT="${ROPE_HEIGHT:-64}"
    ROPE_WIDTH="${ROPE_WIDTH:-64}"
    INFERENCE_STEPS="${INFERENCE_STEPS:-2}"
    SAVE_EVERY="${SAVE_EVERY:-1}"
    CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
    LAUNCH=("${PYTHON_BIN}" -m train_deco.train)
    EXTRA_ARGS=(--limit-samples 16 --log-every-steps 1)
    ;;
  local-train)
    MANIFEST="${MANIFEST:-${PACKAGE_ROOT}/data_manifests/pick_tube_local.json}"
    RUN_ID="${RUN_ID:-pick_tube_vision_local_${TIMESTAMP}}"
    OUTPUT_DIR="${OUTPUT_DIR:-${PACKAGE_ROOT}/outputs}"
    BATCH_SIZE="${BATCH_SIZE:-8}"
    WORKERS="${WORKERS:-4}"
    EPOCHS="${EPOCHS:-100}"
    ACTION_CHUNK_SIZE="${ACTION_CHUNK_SIZE:-32}"
    HIDDEN_DIM="${HIDDEN_DIM:-512}"
    LAYERS="${LAYERS:-6}"
    HEADS="${HEADS:-8}"
    IMAGE_SIZE="${IMAGE_SIZE:-256}"
    ROPE_HEIGHT="${ROPE_HEIGHT:-256}"
    ROPE_WIDTH="${ROPE_WIDTH:-256}"
    INFERENCE_STEPS="${INFERENCE_STEPS:-5}"
    SAVE_EVERY="${SAVE_EVERY:-10}"
    CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
    LAUNCH=("${PYTHON_BIN}" -m train_deco.train)
    EXTRA_ARGS=()
    ;;
  server-train)
    MANIFEST="${MANIFEST:-${PACKAGE_ROOT}/data_manifests/pick_tube_01_06.json}"
    RUN_ID="${RUN_ID:-pick_tube_vision_01_06_ddp2_${TIMESTAMP}}"
    OUTPUT_DIR="${OUTPUT_DIR:-/DATA/ljl/substage/deco_runs}"
    BATCH_SIZE="${BATCH_SIZE:-16}"
    WORKERS="${WORKERS:-4}"
    EPOCHS="${EPOCHS:-100}"
    ACTION_CHUNK_SIZE="${ACTION_CHUNK_SIZE:-32}"
    HIDDEN_DIM="${HIDDEN_DIM:-512}"
    LAYERS="${LAYERS:-6}"
    HEADS="${HEADS:-8}"
    IMAGE_SIZE="${IMAGE_SIZE:-256}"
    ROPE_HEIGHT="${ROPE_HEIGHT:-256}"
    ROPE_WIDTH="${ROPE_WIDTH:-256}"
    INFERENCE_STEPS="${INFERENCE_STEPS:-5}"
    SAVE_EVERY="${SAVE_EVERY:-10}"
    CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2,3}"
    LAUNCH=("${TORCHRUN_BIN}" --standalone --nnodes=1 --nproc_per_node=2 -m train_deco.train)
    EXTRA_ARGS=()
    ;;
  *)
    echo "--mode 只能是 local-smoke、local-train 或 server-train，当前为：${MODE}" >&2
    exit 2
    ;;
esac

LR="${LR:-1e-4}"
LR_FINAL="${LR_FINAL:-5e-6}"
BACKBONE_LR="${BACKBONE_LR:-1e-5}"
BACKBONE_LR_FINAL="${BACKBONE_LR_FINAL:-5e-7}"

COMMAND=(
  "${LAUNCH[@]}"
  --dataset-format lerobot-v21
  --dataset-manifest "${MANIFEST}"
  --output-dir "${OUTPUT_DIR}"
  --run-id "${RUN_ID}"
  --epochs "${EPOCHS}"
  --batch-size "${BATCH_SIZE}"
  --workers "${WORKERS}"
  --action-chunk-size "${ACTION_CHUNK_SIZE}"
  --validation-ratio "${VALIDATION_RATIO}"
  --episode-split-seed "${EPISODE_SPLIT_SEED}"
  --lr "${LR}"
  --lr-final "${LR_FINAL}"
  --backbone-lr "${BACKBONE_LR}"
  --backbone-lr-final "${BACKBONE_LR_FINAL}"
  --hidden-dim "${HIDDEN_DIM}"
  --layers "${LAYERS}"
  --heads "${HEADS}"
  --image-size "${IMAGE_SIZE}"
  --rope-height "${ROPE_HEIGHT}"
  --rope-width "${ROPE_WIDTH}"
  --inference-steps "${INFERENCE_STEPS}"
  --backbone-weights "${BACKBONE_WEIGHTS}"
  --torchscript-image-height 224
  --torchscript-image-width 224
  --save-every "${SAVE_EVERY}"
  --keep-last-checkpoints 5
  "${EXTRA_ARGS[@]}"
)

if [[ -n "${RESUME_FROM}" ]]; then
  COMMAND+=(--resume-from "${RESUME_FROM}" --resume-mode exact)
fi

printf 'CUDA_VISIBLE_DEVICES=%q ' "${CUDA_VISIBLE_DEVICES}"
printf '%q ' "${COMMAND[@]}"
printf '\n'

if [[ ${DRY_RUN} -eq 1 ]]; then
  exit 0
fi

if [[ ! -f "${MANIFEST}" ]]; then
  echo "找不到数据 manifest：${MANIFEST}" >&2
  echo "请先运行 02_prepare_data.sh。" >&2
  exit 1
fi
if [[ "${MODE}" == "server-train" ]]; then
  if [[ ! -x "${TORCHRUN_BIN}" ]]; then
    echo "找不到 torchrun：${TORCHRUN_BIN}" >&2
    exit 1
  fi
elif [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "找不到 Python：${PYTHON_BIN}" >&2
  exit 1
fi

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"

cd "${PROJECT_ROOT}"
exec "${COMMAND[@]}"
