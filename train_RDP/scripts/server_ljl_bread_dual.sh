#!/usr/bin/env bash
set -euo pipefail

# /home/ljl server entry point for the dual-arm Bread task.
# Combines bread_01, bread_02 and bread_03 into one PCA30 RDP dataset/model.

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
# Only BREAD_* path overrides are accepted here. Generic variables such as
# DATASET_PATH and PCA_PATH are deliberately ignored because operators often
# export them while running another task in the same shell.
RDP_CODE_DIR=${BREAD_RDP_CODE_DIR:-/home/ljl/FRS_Tact/train_RDP}
cd "${RDP_CODE_DIR}"

STAGE=${1:-help}
GPU_ID=${GPU_ID:-0}

LEROBOT_ROOT=${BREAD_LEROBOT_ROOT:-/DATA/ljl/substage/lerobot_v21/KaiyueChen}
WORK_ROOT=${BREAD_WORK_ROOT:-/DATA/ljl/substage/rdp_bread_dual}
TACTILE_CACHE_ROOT=${BREAD_TACTILE_CACHE_ROOT:-${WORK_ROOT}/tactile_embeddings_encoder0824}
PCA_PATH=${BREAD_PCA_PATH:-${WORK_ROOT}/pca/tactile_pca_bread_01_02_03_encoder0824_2x15.npz}
DATASET_PATH=${BREAD_DATASET_PATH:-${WORK_ROOT}/datasets/bread_01_02_03_pca30_dual_arm_rdp_zarr}
OUTPUT_ROOT=${BREAD_OUTPUT_ROOT:-${WORK_ROOT}/outputs/bread_01_02_03}

# Reuse the already-installed environments and encoder. This script never
# creates environments or downloads packages/models.
PYTHON_BIN=${BREAD_PYTHON_BIN:-/home/ljl/RDP_vitamin/.venv/bin/python}
ACCELERATE_BIN=${BREAD_ACCELERATE_BIN:-/home/ljl/RDP_vitamin/.venv/bin/accelerate}
JAX_PYTHON=${BREAD_JAX_PYTHON:-/home/ljl/RDP_vitamin/.venv-jax/bin/python}
ENCODER_DIR=${BREAD_ENCODER_DIR:-/home/ljl/RDP_vitamin/data/encoder_ckpt_0824}

DATASETS=(bread_01 bread_02 bread_03)
TASK_TAG=bread_01_02_03
PRECOMPUTE_BATCH=${BREAD_PRECOMPUTE_BATCH:-512}
PRECOMPUTE_WORKERS=${BREAD_PRECOMPUTE_WORKERS:-32}
CONVERT_WORKERS=${BREAD_CONVERT_WORKERS:-32}
AT_BATCH=${BREAD_AT_BATCH:-512}
# LDP is substantially heavier than AT; 64 is the stable physical batch.
LDP_BATCH=${BREAD_LDP_BATCH:-64}
NUM_WORKERS=${BREAD_NUM_WORKERS:-32}
AT_EPOCHS=${BREAD_AT_EPOCHS:-20}
LDP_EPOCHS=${BREAD_LDP_EPOCHS:-10}
AT_CHECKPOINT_EVERY=${BREAD_AT_CHECKPOINT_EVERY:-2}
LDP_CHECKPOINT_EVERY=${BREAD_LDP_CHECKPOINT_EVERY:-2}
AT_CHECKPOINT_KEEP=${BREAD_AT_CHECKPOINT_KEEP:-20}
LDP_CHECKPOINT_KEEP=${BREAD_LDP_CHECKPOINT_KEEP:-20}
MIXED_PRECISION=${BREAD_MIXED_PRECISION:-bf16}
LOGGING_MODE=${BREAD_LOGGING_MODE:-offline}
EXPERIMENT_ID=${BREAD_EXPERIMENT_ID:-$(date +%Y%m%d_%H%M%S)}
RESUME=${BREAD_RESUME:-true}
FORCE_PREPARE=${BREAD_FORCE_PREPARE:-0}
OVERWRITE_TACTILE=${BREAD_OVERWRITE_TACTILE:-0}
REPAIR_ORPHAN_CACHE=${BREAD_REPAIR_ORPHAN_CACHE:-1}
DRY_RUN=${DRY_RUN:-0}

usage() {
  cat <<'USAGE'
用法：
  bash scripts/server_ljl_bread_dual.sh doctor
  bash scripts/server_ljl_bread_dual.sh precompute
  bash scripts/server_ljl_bread_dual.sh prepare
  bash scripts/server_ljl_bread_dual.sh train
  bash scripts/server_ljl_bread_dual.sh all

数据：bread_01 + bread_02 + bread_03（合并训练一个双臂模型）

阶段：
  doctor     检查 GPU、现有环境、encoder 和三个数据集合同
  precompute 只生成四路触觉 embedding
  prepare    生成/续算 embedding、PCA30 和双臂 RDP Zarr
  train      使用已有 Zarr 训练双臂 AT -> LDP
  all        prepare + train；不会安装或下载环境

默认路径：
  代码       /home/ljl/FRS_Tact/train_RDP
  原始数据   /DATA/ljl/substage/lerobot_v21/KaiyueChen
  工作目录   /DATA/ljl/substage/rdp_bread_dual
  PyTorch    /home/ljl/RDP_vitamin/.venv
  JAX        /home/ljl/RDP_vitamin/.venv-jax
  encoder    /home/ljl/RDP_vitamin/data/encoder_ckpt_0824

默认参数：
  GPU_ID=0 BREAD_PRECOMPUTE_BATCH=512 BREAD_PRECOMPUTE_WORKERS=32
  BREAD_AT_BATCH=512 BREAD_LDP_BATCH=64 BREAD_NUM_WORKERS=32
  BREAD_AT_EPOCHS=20 BREAD_LDP_EPOCHS=10 BREAD_MIXED_PRECISION=bf16

路径和训练参数只接受 BREAD_* 覆盖，避免被其他任务导出的 DATASET_PATH、
PCA_PATH、WORK_ROOT 等变量污染。

示例：
  GPU_ID=1 bash scripts/server_ljl_bread_dual.sh all
  GPU_ID=1 BREAD_EXPERIMENT_ID=bread_v1 bash scripts/server_ljl_bread_dual.sh train
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

check_executable() {
  local path=$1
  local label=$2
  if [[ ! -x "${path}" ]]; then
    echo "${label} is not executable: ${path}" >&2
    exit 1
  fi
}

create_roots() {
  run mkdir -p "${TACTILE_CACHE_ROOT}" "$(dirname -- "${PCA_PATH}")" \
    "$(dirname -- "${DATASET_PATH}")" "${OUTPUT_ROOT}"
}

verify_runtime() {
  if [[ "${DRY_RUN}" == "1" ]]; then
    echo "Dry run：跳过 CUDA 运行时检查"
    return
  fi
  check_executable "${PYTHON_BIN}" "RDP Python"
  check_executable "${ACCELERATE_BIN}" "Accelerate"
  check_executable "${JAX_PYTHON}" "JAX Python"
  env "CUDA_VISIBLE_DEVICES=${GPU_ID}" "${PYTHON_BIN}" - <<'PY'
import torch

if not torch.cuda.is_available():
    raise SystemExit("RDP PyTorch 环境没有检测到 CUDA")
print(f"RDP torch={torch.__version__}, gpu={torch.cuda.get_device_name(0)}")
PY
  env -u LD_LIBRARY_PATH "CUDA_VISIBLE_DEVICES=${GPU_ID}" \
    XLA_PYTHON_CLIENT_PREALLOCATE=false "${JAX_PYTHON}" - <<'PY'
import jax

if jax.__version__ != "0.8.3":
    raise SystemExit(f"JAX version mismatch: expected 0.8.3, got {jax.__version__}")
if not any(device.platform == "gpu" for device in jax.devices()):
    raise SystemExit(f"JAX 环境没有检测到 GPU：{jax.devices()}")
print(f"JAX={jax.__version__}, devices={jax.devices()}")
PY
}

check_encoder() {
  if [[ ! -f "${ENCODER_DIR}/checkpoint.json" ]]; then
    if [[ "${DRY_RUN}" == "1" ]]; then
      echo "Dry run：将复用 tactile encoder：${ENCODER_DIR}"
      return
    fi
    echo "Tactile encoder metadata not found: ${ENCODER_DIR}/checkpoint.json" >&2
    exit 1
  fi
  local params
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

check_dataset_contract() {
  if [[ "${DRY_RUN}" == "1" ]]; then
    printf 'Dry run：将检查 Bread 双臂数据合同：%s\n' "${DATASETS[*]}"
    return
  fi
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
    dataset_root = root / dataset
    info_path = dataset_root / "meta" / "info.json"
    episodes_path = dataset_root / "meta" / "episodes.jsonl"
    if not info_path.is_file() or not episodes_path.is_file():
        raise SystemExit(f"{dataset}: 缺少 meta/info.json 或 meta/episodes.jsonl")
    info = json.loads(info_path.read_text())
    if info.get("codebase_version") != "v2.1":
        raise SystemExit(f"{dataset}: expected LeRobot v2.1, got {info.get('codebase_version')}")
    if int(info.get("fps", -1)) != 30:
        raise SystemExit(f"{dataset}: expected 30 fps, got {info.get('fps')}")
    features = info.get("features", {})
    for key, shape in required.items():
        actual = features.get(key, {}).get("shape")
        if actual != shape:
            raise SystemExit(f"{dataset}: {key} expected {shape}, got {actual}")
    print(f"contract OK: {dataset}, episodes={info['total_episodes']}, frames={info['total_frames']}")
PY
}

repair_orphan_tactile_cache() {
  local dataset cache_dir output_path metadata_path progress_path backup_path
  for dataset in "${DATASETS[@]}"; do
    cache_dir="${TACTILE_CACHE_ROOT}/KaiyueChen/${dataset}"
    output_path="${cache_dir}/embeddings.npy"
    metadata_path="${cache_dir}/metadata.json"
    progress_path="${cache_dir}/progress.json"
    if [[ -f "${output_path}" && ! -f "${metadata_path}" && ! -f "${progress_path}" ]]; then
      if [[ "${REPAIR_ORPHAN_CACHE}" != "1" ]]; then
        echo "Orphan tactile cache found: ${output_path}" >&2
        echo "Set BREAD_REPAIR_ORPHAN_CACHE=1 to quarantine and rebuild it." >&2
        exit 1
      fi
      backup_path="${output_path}.orphan.$(date +%Y%m%d_%H%M%S)"
      echo "检测到无进度标记的触觉半成品，将隔离后重新计算："
      echo "  ${output_path}"
      echo "  -> ${backup_path}"
      run mv "${output_path}" "${backup_path}"
    fi
  done
}

precompute_tactile() {
  repair_orphan_tactile_cache
  local args=(
    env -u LD_LIBRARY_PATH
    "CUDA_VISIBLE_DEVICES=${GPU_ID}"
    XLA_PYTHON_CLIENT_PREALLOCATE=false
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
    local cache_dir="${TACTILE_CACHE_ROOT}/KaiyueChen/${dataset}"
    if [[ ! -f "${cache_dir}/embeddings.npy" || ! -f "${cache_dir}/metadata.json" ]]; then
      if [[ "${DRY_RUN}" == "1" ]]; then
        echo "Dry run：${dataset} 的 embedding 将由 prepare 生成"
      else
        echo "Tactile embedding incomplete: ${cache_dir}" >&2
        exit 1
      fi
    fi
  done
}

prepare_dataset() {
  precompute_tactile
  check_tactile_cache

  if [[ "${FORCE_PREPARE}" == "1" || ! -f "${PCA_PATH}" ]]; then
    run "${PYTHON_BIN}" fit_pick_tube_tactile_pca.py \
      --tactile-cache-root "${TACTILE_CACHE_ROOT}" \
      --output "${PCA_PATH}" \
      --components-per-arm 15 \
      --datasets "${DATASETS[@]}"
  else
    echo "复用已有 PCA：${PCA_PATH}"
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
      --state-action-profile dual-arm-20x20
      --num-workers "${CONVERT_WORKERS}"
    )
    if [[ "${FORCE_PREPARE}" == "1" ]]; then
      args+=(--overwrite)
    fi
    run "${args[@]}"
  else
    echo "复用已有 RDP Zarr：${DATASET_PATH}"
  fi

  if [[ "${DRY_RUN}" != "1" ]]; then
    run env "PYTHON_BIN=${PYTHON_BIN}" "DATASET_PATH=${DATASET_PATH}" \
      bash scripts/setup_pick_tube_data.sh validate
  fi
}

train_model() {
  if [[ ! -d "${DATASET_PATH}/replay_buffer.zarr" && "${DRY_RUN}" != "1" ]]; then
    echo "RDP dataset not found: ${DATASET_PATH}/replay_buffer.zarr" >&2
    echo "请先运行：bash scripts/server_ljl_bread_dual.sh prepare" >&2
    exit 1
  fi
  run env \
    "PYTHON_BIN=${PYTHON_BIN}" \
    "ACCELERATE_BIN=${ACCELERATE_BIN}" \
    "DATASET_PATH=${DATASET_PATH}" \
    "OUTPUT_ROOT=${OUTPUT_ROOT}" \
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

case "${STAGE}" in
  doctor|precompute|prepare|train|all)
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

if [[ ! "${GPU_ID}" =~ ^[0-9]+$ ]]; then
  echo "GPU_ID must be a non-negative integer, got: ${GPU_ID}" >&2
  exit 2
fi

create_roots
case "${STAGE}" in
  doctor)
    check_dataset_contract
    check_encoder
    verify_runtime
    ;;
  precompute)
    check_dataset_contract
    check_encoder
    verify_runtime
    precompute_tactile
    ;;
  prepare)
    check_dataset_contract
    check_encoder
    verify_runtime
    prepare_dataset
    ;;
  train)
    verify_runtime
    train_model
    ;;
  all)
    check_dataset_contract
    check_encoder
    verify_runtime
    prepare_dataset
    train_model
    ;;
esac

printf '\nBread 双臂任务完成阶段：%s\n' "${STAGE}"
printf 'RDP dataset: %s\n' "${DATASET_PATH}"
printf 'AT/LDP output: %s\n' "${OUTPUT_ROOT}"
