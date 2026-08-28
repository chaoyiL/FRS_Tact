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
RUN_ID="${RUN_ID:-}"
RESUME_FROM="${RESUME_FROM:-}"
STAGE1_CHECKPOINT="${STAGE1_CHECKPOINT:-}"
TACTILE_ENCODER_CHECKPOINT="${TACTILE_ENCODER_CHECKPOINT:-}"
TACTILE_ENCODER_CACHE="${TACTILE_ENCODER_CACHE:-${PROJECT_ROOT}/checkpoints/deco/tactile_encoder_cache}"
STAGE=1
DRY_RUN=0

usage() {
  cat <<'EOF'
用法：
  bash scripts/pick_tube_vision/03_train.sh --mode local-smoke [选项]
  bash scripts/pick_tube_vision/03_train.sh --mode local-train [选项]
  bash scripts/pick_tube_vision/03_train.sh --mode server-train [选项]
  bash scripts/pick_tube_vision/03_train.sh --mode local-stage2 [选项]
  bash scripts/pick_tube_vision/03_train.sh --mode server-stage2 [选项]

模式：
  local-smoke   本机 RTX 4090 单卡最小链路：16 samples、1 epoch、小模型
  local-train   本机 RTX 4090 单卡正式参数
  server-train  服务器第 3、4 张卡（CUDA 索引 2,3），torchrun/DDP 两进程
  local-stage2  本机单卡 Stage2，从 Stage1 和触觉编码器初始化
  server-stage2 服务器双卡 Stage2，rank 0 自动转换触觉编码器

选项：
  --manifest PATH    02_prepare_data.sh 生成的 manifest
  --run-id ID        输出运行名；默认自动加入时间戳
  --resume-from PATH 从 checkpoint 恢复
  --stage1-checkpoint PATH          Stage1 训练 checkpoint
  --tactile-encoder-checkpoint PATH JAX目录或已转换 safetensors
  --dry-run          只打印命令，不启动训练
  -h, --help         显示帮助

常用环境变量覆盖：
  OUTPUT_DIR BATCH_SIZE WORKERS EPOCHS SAVE_EVERY ACTION_CHUNK_SIZE LR BACKBONE_LR
  HIDDEN_DIM LAYERS HEADS IMAGE_SIZE INFERENCE_STEPS CUDA_VISIBLE_DEVICES
  AUGMENTATION_PRESET AUGMENTATION_ENABLED AUGMENTATION_IDENTITY_PROBABILITY
  AUGMENTATION_LOW_LIGHT_PROBABILITY AUGMENTATION_MILD_PROBABILITY

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
    --resume-from|--resume)
      RESUME_FROM="$2"
      shift 2
      ;;
    --stage1-checkpoint)
      STAGE1_CHECKPOINT="$2"
      shift 2
      ;;
    --tactile-encoder-checkpoint)
      TACTILE_ENCODER_CHECKPOINT="$2"
      shift 2
      ;;
    --tactile-encoder-cache)
      TACTILE_ENCODER_CACHE="$2"
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
  local-stage2)
    STAGE=2
    MANIFEST="${MANIFEST:-${PACKAGE_ROOT}/data_manifests/pick_tube_local.json}"
    RUN_ID="${RUN_ID:-pick_tube_stage2_local_${TIMESTAMP}}"
    OUTPUT_DIR="${OUTPUT_DIR:-${PACKAGE_ROOT}/outputs}"
    BATCH_SIZE="${BATCH_SIZE:-8}"
    WORKERS="${WORKERS:-4}"
    EPOCHS="${EPOCHS:-50}"
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
  server-train|server-stage2)
    if [[ "${MODE}" == "server-stage2" ]]; then
      STAGE=2
      RUN_ID="${RUN_ID:-pick_tube_stage2_01_06_ddp2_${TIMESTAMP}}"
      EPOCHS="${EPOCHS:-50}"
    else
      RUN_ID="${RUN_ID:-pick_tube_vision_01_06_ddp2_${TIMESTAMP}}"
      EPOCHS="${EPOCHS:-100}"
    fi
    MANIFEST="${MANIFEST:-${PACKAGE_ROOT}/data_manifests/pick_tube_01_06.json}"
    OUTPUT_DIR="${OUTPUT_DIR:-/DATA/ljl/substage/deco_runs}"
    BATCH_SIZE="${BATCH_SIZE:-16}"
    WORKERS="${WORKERS:-4}"
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
    echo "--mode 只能是 local-smoke、local-train、local-stage2、server-train 或 server-stage2，当前为：${MODE}" >&2
    exit 2
    ;;
esac

LR="${LR:-1e-4}"
LR_FINAL="${LR_FINAL:-5e-6}"
BACKBONE_LR="${BACKBONE_LR:-1e-5}"
BACKBONE_LR_FINAL="${BACKBONE_LR_FINAL:-5e-7}"
AUGMENTATION_PRESET="${AUGMENTATION_PRESET:-balanced-light-v2}"
AUGMENTATION_ENABLED="${AUGMENTATION_ENABLED:-1}"
AUGMENTATION_IDENTITY_PROBABILITY="${AUGMENTATION_IDENTITY_PROBABILITY:-0.25}"
AUGMENTATION_LOW_LIGHT_PROBABILITY="${AUGMENTATION_LOW_LIGHT_PROBABILITY:-0.55}"
AUGMENTATION_MILD_PROBABILITY="${AUGMENTATION_MILD_PROBABILITY:-0.20}"
AUGMENTATION_EXPOSURE_PROBABILITY="${AUGMENTATION_EXPOSURE_PROBABILITY:-0.5}"
AUGMENTATION_EXPOSURE_MIN="${AUGMENTATION_EXPOSURE_MIN:-0.58}"
AUGMENTATION_EXPOSURE_MAX="${AUGMENTATION_EXPOSURE_MAX:-0.90}"
AUGMENTATION_GAMMA_MIN="${AUGMENTATION_GAMMA_MIN:-1.10}"
AUGMENTATION_GAMMA_MAX="${AUGMENTATION_GAMMA_MAX:-1.50}"
AUGMENTATION_MILD_BRIGHTNESS_MIN="${AUGMENTATION_MILD_BRIGHTNESS_MIN:-0.90}"
AUGMENTATION_MILD_BRIGHTNESS_MAX="${AUGMENTATION_MILD_BRIGHTNESS_MAX:-1.10}"
AUGMENTATION_CONTRAST_MIN="${AUGMENTATION_CONTRAST_MIN:-0.85}"
AUGMENTATION_CONTRAST_MAX="${AUGMENTATION_CONTRAST_MAX:-1.10}"
AUGMENTATION_SATURATION_MIN="${AUGMENTATION_SATURATION_MIN:-0.90}"
AUGMENTATION_SATURATION_MAX="${AUGMENTATION_SATURATION_MAX:-1.10}"
AUGMENTATION_BLUR_PROBABILITY="${AUGMENTATION_BLUR_PROBABILITY:-0.20}"
AUGMENTATION_BLUR_SIGMA_MIN="${AUGMENTATION_BLUR_SIGMA_MIN:-0.1}"
AUGMENTATION_BLUR_SIGMA_MAX="${AUGMENTATION_BLUR_SIGMA_MAX:-1.0}"

case "${AUGMENTATION_ENABLED}" in
  1|[Tt][Rr][Uu][Ee]|[Yy][Ee][Ss]|[Oo][Nn])
    AUGMENTATION_FLAG=(--augmentation-enabled)
    ;;
  *) AUGMENTATION_FLAG=(--no-augmentation-enabled) ;;
esac

COMMAND=(
  "${LAUNCH[@]}"
  --dataset-format lerobot-v21
  --dataset-manifest "${MANIFEST}"
  --output-dir "${OUTPUT_DIR}"
  --run-id "${RUN_ID}"
  --stage "${STAGE}"
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
  "${AUGMENTATION_FLAG[@]}"
  --augmentation-preset "${AUGMENTATION_PRESET}"
  --augmentation-identity-probability "${AUGMENTATION_IDENTITY_PROBABILITY}"
  --augmentation-low-light-probability "${AUGMENTATION_LOW_LIGHT_PROBABILITY}"
  --augmentation-mild-probability "${AUGMENTATION_MILD_PROBABILITY}"
  --augmentation-exposure-probability "${AUGMENTATION_EXPOSURE_PROBABILITY}"
  --augmentation-exposure-range "${AUGMENTATION_EXPOSURE_MIN}" "${AUGMENTATION_EXPOSURE_MAX}"
  --augmentation-gamma-range "${AUGMENTATION_GAMMA_MIN}" "${AUGMENTATION_GAMMA_MAX}"
  --augmentation-mild-brightness-range "${AUGMENTATION_MILD_BRIGHTNESS_MIN}" "${AUGMENTATION_MILD_BRIGHTNESS_MAX}"
  --augmentation-contrast-range "${AUGMENTATION_CONTRAST_MIN}" "${AUGMENTATION_CONTRAST_MAX}"
  --augmentation-saturation-range "${AUGMENTATION_SATURATION_MIN}" "${AUGMENTATION_SATURATION_MAX}"
  --augmentation-blur-probability "${AUGMENTATION_BLUR_PROBABILITY}"
  --augmentation-blur-kernel-sizes 3 5
  --augmentation-blur-sigma-range "${AUGMENTATION_BLUR_SIGMA_MIN}" "${AUGMENTATION_BLUR_SIGMA_MAX}"
  --torchscript-image-height 224
  --torchscript-image-width 224
  --save-every "${SAVE_EVERY}"
  --keep-last-checkpoints 5
)

if [[ "${MODE}" == "local-smoke" ]]; then
  COMMAND+=(--limit-samples 16 --log-every-steps 1)
fi

if [[ -n "${RESUME_FROM}" ]]; then
  COMMAND+=(--resume "${RESUME_FROM}" --resume-mode exact)
fi


if [[ ${STAGE} -eq 2 && -z "${RESUME_FROM}" ]]; then
  if [[ -z "${STAGE1_CHECKPOINT}" || -z "${TACTILE_ENCODER_CHECKPOINT}" ]]; then
    echo "Stage2 需要 --stage1-checkpoint 和 --tactile-encoder-checkpoint" >&2
    exit 2
  fi
  COMMAND+=(
    --stage1-checkpoint "${STAGE1_CHECKPOINT}"
    --tactile-encoder-checkpoint "${TACTILE_ENCODER_CHECKPOINT}"
    --tactile-encoder-cache "${TACTILE_ENCODER_CACHE}"
  )
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
if [[ "${MODE}" == "server-train" || "${MODE}" == "server-stage2" ]]; then
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
