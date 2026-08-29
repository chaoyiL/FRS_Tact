#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
RDP_DIR=${RDP_DIR:-/home/ljl/FRS_Tact/train_RDP}
cd "${RDP_DIR}"

STAGE=${1:-help}
TASK_SELECTION=${2:-both}

LEROBOT_ROOT=${LEROBOT_ROOT:-/DATA/ljl/substage/lerobot_v21/KaiyueChen}
WORK_ROOT=${WORK_ROOT:-/DATA/ljl/substage/rdp_single_right}
TACTILE_CACHE_ROOT=${TACTILE_CACHE_ROOT:-${WORK_ROOT}/tactile_embeddings_encoder0824}
PCA_ROOT=${PCA_ROOT:-${WORK_ROOT}/pca}
DATA_ROOT=${DATA_ROOT:-${WORK_ROOT}/datasets}
OUTPUT_ROOT=${OUTPUT_ROOT:-${WORK_ROOT}/outputs}
ENCODER_DIR=${ENCODER_DIR:-${WORK_ROOT}/encoder_ckpt_0824}

PYTHON_BIN=${PYTHON_BIN:-${RDP_DIR}/.venv/bin/python}
ACCELERATE_BIN=${ACCELERATE_BIN:-${RDP_DIR}/.venv/bin/accelerate}
JAX_PYTHON=${JAX_PYTHON:-${RDP_DIR}/.venv-jax/bin/python}
GPU_ID=${GPU_ID:-0}
MIXED_PRECISION=${MIXED_PRECISION:-bf16}
LOGGING_MODE=${LOGGING_MODE:-offline}
AT_BATCH=${AT_BATCH:-64}
LDP_BATCH=${LDP_BATCH:-64}
NUM_WORKERS=${NUM_WORKERS:-8}
AT_EPOCHS=${AT_EPOCHS:-20}
LDP_EPOCHS=${LDP_EPOCHS:-10}
AT_CHECKPOINT_EVERY=${AT_CHECKPOINT_EVERY:-2}
LDP_CHECKPOINT_EVERY=${LDP_CHECKPOINT_EVERY:-2}
AT_CHECKPOINT_KEEP=${AT_CHECKPOINT_KEEP:-20}
LDP_CHECKPOINT_KEEP=${LDP_CHECKPOINT_KEEP:-20}
PRECOMPUTE_BATCH=${PRECOMPUTE_BATCH:-64}
PRECOMPUTE_WORKERS=${PRECOMPUTE_WORKERS:-4}
EXPERIMENT_ID=${EXPERIMENT_ID:-$(date +%Y%m%d_%H%M%S)}
RESUME=${RESUME:-true}
FORCE_PREPARE=${FORCE_PREPARE:-0}
OVERWRITE_TACTILE=${OVERWRITE_TACTILE:-0}
HF_ENDPOINT=${HF_ENDPOINT:-https://alpha.hf-mirror.com}
HF_HUB_DISABLE_XET=${HF_HUB_DISABLE_XET:-1}
HF_HUB_DOWNLOAD_TIMEOUT=${HF_HUB_DOWNLOAD_TIMEOUT:-120}
HF_MAX_WORKERS=${HF_MAX_WORKERS:-8}
ENCODER_REVISION=${ENCODER_REVISION:-1a432a8b9c1a38f75cb40c58a4924a401486d832}
ENCODER_PARAMS_FILE=${ENCODER_PARAMS_FILE:-params-179d095d9e11460a99ce28062c5a7f4a.npz}
PYPI_INDEX_URL=${PYPI_INDEX_URL:-https://pypi.tuna.tsinghua.edu.cn/simple}
TORCH_INDEX_URL=${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu128}
JAX_VERSION=${JAX_VERSION:-0.8.3}
FLAX_VERSION=${FLAX_VERSION:-0.12.6}
DRY_RUN=${DRY_RUN:-0}

usage() {
  cat <<"USAGE"
用法：
  bash scripts/server_ljl_single_right.sh setup
  bash scripts/server_ljl_single_right.sh doctor
  bash scripts/server_ljl_single_right.sh prepare insert
  bash scripts/server_ljl_single_right.sh train insert
  bash scripts/server_ljl_single_right.sh all press
  bash scripts/server_ljl_single_right.sh all both

阶段：
  setup      创建 .venv 和 .venv-jax，并从 HF 镜像下载 encoder_ckpt_0824
  doctor     检查 GPU、环境、encoder 和所选数据合同
  precompute 只生成四路触觉 embedding
  prepare    生成 embedding、独立 PCA30 和 RDP Zarr
  train      使用已有 Zarr 训练 AT -> LDP
  all        自动 setup、prepare、train

任务：
  insert     合并 insert_01 + insert_02，训练一个模型
  press      使用 press_01，训练另一个模型
  both       顺序执行 insert 和 press

固定默认路径：
  代码       /home/ljl/FRS_Tact/train_RDP
  原始数据   /DATA/ljl/substage/lerobot_v21/KaiyueChen
  中间产物   /DATA/ljl/substage/rdp_single_right
  encoder    /DATA/ljl/substage/rdp_single_right/encoder_ckpt_0824
  输出       /DATA/ljl/substage/rdp_single_right/outputs

常用可选覆盖：
  GPU_ID=0 NUM_WORKERS=8 AT_BATCH=64 LDP_BATCH=64
  AT_EPOCHS=20 LDP_EPOCHS=10 LOGGING_MODE=offline
  EXPERIMENT_ID=v1 RESUME=true DRY_RUN=1
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

create_roots() {
  run mkdir -p "${WORK_ROOT}" "${TACTILE_CACHE_ROOT}" "${PCA_ROOT}" \
    "${DATA_ROOT}" "${OUTPUT_ROOT}" "${ENCODER_DIR}"
}

resolve_uv() {
  if command -v uv >/dev/null 2>&1; then
    command -v uv
  elif [[ -x "${HOME}/.local/bin/uv" ]]; then
    printf "%s\n" "${HOME}/.local/bin/uv"
  else
    printf "%s\n" "${HOME}/.local/bin/uv"
  fi
}

setup_environments() {
  create_roots
  if [[ ! -x "${PYTHON_BIN}" || ! -x "${ACCELERATE_BIN}" || ! -x "${RDP_DIR}/.venv/bin/hf" ]]; then
    run env \
      "VENV_DIR=${RDP_DIR}/.venv" \
      "PYPI_INDEX_URL=${PYPI_INDEX_URL}" \
      "TORCH_INDEX_URL=${TORCH_INDEX_URL}" \
      "DRY_RUN=${DRY_RUN}" \
      bash scripts/install_pick_tube_training_env.sh
  else
    echo "复用 RDP 训练环境：${RDP_DIR}/.venv"
  fi

  if [[ ! -x "${JAX_PYTHON}" ]]; then
    local uv_bin
    uv_bin=$(resolve_uv)
    if [[ "${DRY_RUN}" != "1" && ! -x "${uv_bin}" ]]; then
      echo "找不到 uv；RDP 环境安装脚本应先安装 uv：${uv_bin}" >&2
      exit 1
    fi
    run "${uv_bin}" venv --python 3.12 "${RDP_DIR}/.venv-jax"
    run "${uv_bin}" pip install --python "${JAX_PYTHON}" \
      --index-url "${PYPI_INDEX_URL}" \
      "jax[cuda12]==${JAX_VERSION}" \
      "flax==${FLAX_VERSION}" \
      "numpy==2.2.6" \
      "pyarrow==25.0.1" \
      "pillow==12.2.0" \
      "pyyaml==6.0.3" \
      "tqdm==4.70.0"
  else
    echo "复用 JAX 触觉编码环境：${RDP_DIR}/.venv-jax"
  fi

  verify_runtime
}

verify_runtime() {
  if [[ "${DRY_RUN}" == "1" ]]; then
    echo "Dry run：跳过 CUDA 运行时检查"
    return
  fi
  check_executable "${PYTHON_BIN}" "RDP Python"
  check_executable "${ACCELERATE_BIN}" "Accelerate"
  check_executable "${JAX_PYTHON}" "JAX Python"
  "${PYTHON_BIN}" - <<"PY"
import accelerate
import diffusers
import hydra
import numpy
import pyarrow
import sklearn
import torch
import zarr

if not torch.cuda.is_available():
    raise SystemExit("RDP PyTorch 环境没有检测到 CUDA")
print(f"RDP torch={torch.__version__}, gpu={torch.cuda.get_device_name(0)}")
PY
  env -u LD_LIBRARY_PATH XLA_PYTHON_CLIENT_PREALLOCATE=false \
    "${JAX_PYTHON}" - <<"PY"
import flax
import jax
import numpy
import pyarrow
import yaml
from PIL import Image

if jax.__version__ != "0.8.3":
    raise SystemExit(f"JAX version mismatch: expected 0.8.3, got {jax.__version__}")
if not any(device.platform == "gpu" for device in jax.devices()):
    raise SystemExit(f"JAX 环境没有检测到 GPU：{jax.devices()}")
print(f"JAX={jax.__version__}, devices={jax.devices()}")
PY
}

download_encoder() {
  create_roots
  if [[ -f "${ENCODER_DIR}/checkpoint.json" ]] && \
      compgen -G "${ENCODER_DIR}/params-*.npz" >/dev/null; then
    echo "复用 tactile encoder：${ENCODER_DIR}"
    return
  fi
  local hf_bin=${RDP_DIR}/.venv/bin/hf
  if [[ "${DRY_RUN}" != "1" && ! -x "${hf_bin}" ]]; then
    echo "找不到 hf CLI；请先运行 setup：${hf_bin}" >&2
    exit 1
  fi
  run env \
    "HF_ENDPOINT=${HF_ENDPOINT}" \
    "HF_HUB_DISABLE_XET=${HF_HUB_DISABLE_XET}" \
    "HF_HUB_DOWNLOAD_TIMEOUT=${HF_HUB_DOWNLOAD_TIMEOUT}" \
    "${hf_bin}" download KaiyueChen/encoder_ckpt_0824 \
      checkpoint.json "${ENCODER_PARAMS_FILE}" \
      --revision "${ENCODER_REVISION}" \
      --local-dir "${ENCODER_DIR}" \
      --max-workers "${HF_MAX_WORKERS}"
}

select_task() {
  local task=$1
  case "${task}" in
    insert)
      TASK_TAG=insert_01_02
      DATASETS=(insert_01 insert_02)
      ;;
    press)
      TASK_TAG=press_01
      DATASETS=(press_01)
      ;;
    *)
      echo "未知任务：${task}；只能使用 insert、press 或 both" >&2
      exit 2
      ;;
  esac
  PCA_PATH=${PCA_ROOT}/tactile_pca_${TASK_TAG}_encoder0824_2x15.npz
  DATASET_PATH=${DATA_ROOT}/${TASK_TAG}_pca30_single_right_rdp_zarr
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
  if [[ "${DRY_RUN}" == "1" ]]; then
    printf "Dry run：将检查数据合同：%s；数据集：%s\n" "${LEROBOT_ROOT}" "${DATASETS[*]}"
    return
  fi
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
    "observation.state": [7],
    "actions": [10],
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
    if [[ "${DRY_RUN}" == "1" ]]; then
      echo "Dry run：将使用 tactile encoder：${ENCODER_DIR}"
      return
    fi
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
  if [[ "${DRY_RUN}" != "1" ]]; then
    check_executable "${JAX_PYTHON}" "JAX Python"
  fi
  check_encoder
  local args=(
    env -u LD_LIBRARY_PATH
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
      --state-action-profile single-right-arm-7x10
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
    bash scripts/train_pick_tube_single_right_gpu.sh all
}

run_task() {
  local task=$1
  select_task "${task}"
  case "${STAGE}" in
    doctor)
      check_dataset_contract
      check_encoder
      verify_runtime
      ;;
    precompute)
      check_dataset_contract
      precompute_task
      ;;
    prepare)
      check_dataset_contract
      precompute_task
      prepare_task
      ;;
    train)
      if [[ "${DRY_RUN}" != "1" ]]; then
        check_executable "${PYTHON_BIN}" "RDP Python"
        check_executable "${ACCELERATE_BIN}" "Accelerate"
      fi
      train_task
      ;;
    all)
      check_dataset_contract
      precompute_task
      prepare_task
      train_task
      ;;
  esac
  printf "\nCompleted stage %s for task %s\n" "${STAGE}" "${TASK_TAG}"
}

case "${STAGE}" in
  setup)
    setup_environments
    download_encoder
    echo "环境与 tactile encoder 已准备完成。"
    exit 0
    ;;
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

if [[ "${STAGE}" == "all" ]]; then
  setup_environments
  download_encoder
elif [[ "${STAGE}" == "doctor" || "${STAGE}" == "precompute" || "${STAGE}" == "prepare" ]]; then
  verify_runtime
  check_encoder
fi

case "${TASK_SELECTION}" in
  insert|press)
    run_task "${TASK_SELECTION}"
    ;;
  both)
    run_task insert
    run_task press
    ;;
  *)
    usage >&2
    exit 2
    ;;
esac
