#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
PACKAGE_ROOT="${PACKAGE_ROOT:-${PROJECT_ROOT}/train_deco}"
VENV_PATH="${VENV_PATH:-${PACKAGE_ROOT}/.venv}"
PYTHON_BIN="${PYTHON_BIN:-${VENV_PATH}/bin/python}"
SERVER_DATA_PARENT="${SERVER_DATA_PARENT:-/DATA/ljl/substage/lerobot_v30_work/KaiyueChen}"

MODE="local"
OUTPUT=""
DATASET_ID=""
STATE_ACTION_PROFILE=""
ROOTS=()

usage() {
  cat <<'EOF'
用法：
  bash scripts/pick_tube_vision/02_prepare_data.sh --mode local [选项]
  bash scripts/pick_tube_vision/02_prepare_data.sh --mode server [选项]

模式：
  local   使用本机 pick_tube_01，供 RTX 4090 冒烟测试
  server  合并服务器的 pick_tube_01、pick_tube_02、pick_tube_03、
          pick_tube_04、pick_tube_05、pick_tube_06，六个数据集都会进入训练/验证划分

选项：
  --root PATH        自定义数据根目录，可重复；一旦指定就替换模式默认值
  --output PATH      输出多数据集 manifest JSON
  --dataset-id ID    manifest 中的数据集名称
  --state-action-profile PROFILE
                     状态/动作合同：dual-arm-20x20 或 single-right-arm-7x10
  -h, --help         显示帮助

默认路径：
  local:
    /home/yunjing/.cache/huggingface/lerobot/KaiyueChen/pick_tube_01
  server:
    /DATA/ljl/substage/lerobot_v30_work/KaiyueChen/pick_tube_01 ... pick_tube_06

这一步不复制 JPEG/Parquet；它会验证六个 LeRobot v2.1 数据集的结构与字段合同，
并生成训练可直接读取的轻量 manifest。训练启动时再统一计算仅基于训练 episode 的统计量。
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --mode)
      MODE="$2"
      shift 2
      ;;
    --root)
      ROOTS+=("$2")
      shift 2
      ;;
    --output)
      OUTPUT="$2"
      shift 2
      ;;
    --dataset-id)
      DATASET_ID="$2"
      shift 2
      ;;
    --state-action-profile)
      STATE_ACTION_PROFILE="$2"
      shift 2
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

case "${MODE}" in
  local)
    if [[ ${#ROOTS[@]} -eq 0 ]]; then
      ROOTS=("/home/yunjing/.cache/huggingface/lerobot/KaiyueChen/pick_tube_01")
    fi
    OUTPUT="${OUTPUT:-${PACKAGE_ROOT}/data_manifests/pick_tube_local.json}"
    DATASET_ID="${DATASET_ID:-pick_tube_01_local}"
    NEXT_MODE="local-smoke"
    ;;
  server)
    if [[ ${#ROOTS[@]} -eq 0 ]]; then
      for suffix in 01 02 03 04 05 06; do
        ROOTS+=("${SERVER_DATA_PARENT}/pick_tube_${suffix}")
      done
    fi
    OUTPUT="${OUTPUT:-${PACKAGE_ROOT}/data_manifests/pick_tube_01_06.json}"
    DATASET_ID="${DATASET_ID:-pick_tube_01_02_03_04_05_06}"
    NEXT_MODE="server-train"
    ;;
  *)
    echo "--mode 只能是 local 或 server，当前为：${MODE}" >&2
    exit 2
    ;;
esac

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "找不到 Python：${PYTHON_BIN}" >&2
  echo "请先运行 01_setup_env.sh，或设置 PYTHON_BIN。" >&2
  exit 1
fi

for root in "${ROOTS[@]}"; do
  if [[ ! -f "${root}/meta/info.json" ]]; then
    echo "无效数据集，缺少 ${root}/meta/info.json" >&2
    exit 1
  fi
done

cd "${PROJECT_ROOT}"
PREPARE_COMMAND=(
  "${PYTHON_BIN}" -m train_deco.prepare_lerobot_multiroot
  --output "${OUTPUT}"
  --dataset-id "${DATASET_ID}"
)
if [[ -n "${STATE_ACTION_PROFILE}" ]]; then
  PREPARE_COMMAND+=(--state-action-profile "${STATE_ACTION_PROFILE}")
fi
PREPARE_COMMAND+=("${ROOTS[@]}")
"${PREPARE_COMMAND[@]}"

echo "数据 manifest 已生成：${OUTPUT}"
echo "数据集数量：${#ROOTS[@]}"
echo "下一步：bash ${PACKAGE_ROOT}/scripts/train.sh --mode ${NEXT_MODE} --manifest ${OUTPUT}"
