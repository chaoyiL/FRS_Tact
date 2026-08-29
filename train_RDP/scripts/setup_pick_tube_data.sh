#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
RDP_DIR=$(cd -- "${SCRIPT_DIR}/.." && pwd)
cd "${RDP_DIR}"

COMMAND=${1:-validate}
if [[ $# -gt 0 ]]; then
  shift
fi
PYTHON_BIN=${PYTHON_BIN:-${RDP_DIR}/.venv/bin/python}
HF_BIN=${HF_BIN:-${RDP_DIR}/.venv/bin/hf}
HF_ENDPOINT=${HF_ENDPOINT:-https://alpha.hf-mirror.com}
HF_HUB_DISABLE_XET=${HF_HUB_DISABLE_XET:-1}
HF_HUB_DOWNLOAD_TIMEOUT=${HF_HUB_DOWNLOAD_TIMEOUT:-60}
HF_MAX_WORKERS=${HF_MAX_WORKERS:-4}
RDP_MODEL_REPO=${RDP_MODEL_REPO:-wjstx/rdp}
RDP_MODEL_REVISION=${RDP_MODEL_REVISION:-main}
RDP_WEIGHTS_DIR=${RDP_WEIGHTS_DIR:-${RDP_DIR}/data/weights/wjstx_rdp}
LEROBOT_ROOT=${LEROBOT_ROOT:-/home/hillbot/datasets}
TACTILE_CACHE_ROOT=${TACTILE_CACHE_ROOT:-${RDP_DIR}/data/tactile_embeddings_encoder0809}
TACTILE_PCA_PATH=${TACTILE_PCA_PATH:-${RDP_DIR}/data/PCA_Transform_PickTube/tactile_pca_2x15.npz}
DATASET_PATH=${DATASET_PATH:-${RDP_DIR}/data/pick_tube_01_06_pca30_rdp_zarr}
SMOKE_DATASET_PATH=${SMOKE_DATASET_PATH:-${RDP_DIR}/data/pick_tube_01_06_pca30_smoke_rdp_zarr}
ENCODER_DIR=${ENCODER_DIR:-${RDP_DIR}/data/encoder_ckpt_0809}
ENCODER_RELEASE_BASE=${ENCODER_RELEASE_BASE:-https://alpha.hf-mirror.com/KaiyueChen/encoder_ckpt_0809/resolve/main}
ENCODER_PARAMS_FILE=params-235cb754d17b461b8be2d652c96fc169.npz
JAX_PYTHON=${JAX_PYTHON:-${RDP_DIR}/.venv-jax/bin/python}
OVERWRITE=${OVERWRITE:-0}
SMOKE_EPISODES=${SMOKE_EPISODES:-1}

usage() {
  cat <<USAGE
Usage: $0 <datasets|encoder|weights|precompute|pca|validate|convert|smoke>

  datasets   Download LeRobot pick_tube_01..06 from the HF mirror.
  encoder    Download the inference-only encoder from the HF mirror.
  weights    Download latest AT/LDP deployment checkpoints from wjstx/rdp.
  precompute Build tactile embeddings using a separate JAX CUDA environment.
  pca        Fit two arm-wise 1024-to-15 PCA projections.
  validate   Validate an existing RDP Zarr and load one AT/LDP batch.
  convert    Convert all LeRobot pick_tube_01..06 episodes, then validate.
  smoke      Convert a small subset to SMOKE_DATASET_PATH, then validate.

Environment variables:
  LEROBOT_ROOT        LeRobot root containing pick_tube_01..06
                      (default: /home/hillbot/datasets)
  HF_ENDPOINT         Hugging Face endpoint (default: alpha.hf-mirror.com)
  HF_HUB_DISABLE_XET  Disable Xet CAS downloads (default: 1)
  HF_HUB_DOWNLOAD_TIMEOUT  Per-file HTTP timeout in seconds (default: 60)
  HF_MAX_WORKERS      Parallel downloads per repository (default: 4)
  RDP_MODEL_REPO      RDP checkpoint repository (default: wjstx/rdp)
  RDP_MODEL_REVISION  Repository revision (default: main)
  RDP_WEIGHTS_DIR     Local checkpoint directory
  TACTILE_CACHE_ROOT  Root containing KaiyueChen/pick_tube_XX/embeddings.npy
  TACTILE_PCA_PATH    Two-arm PCA artifact used by conversion and deployment
  DATASET_PATH        Full RDP Zarr output/input directory
  JAX_PYTHON          Python from a separate CUDA-enabled JAX environment
  ENCODER_DIR         Downloaded encoder checkpoint directory
  ENCODER_RELEASE_BASE  Encoder download URL prefix
  OVERWRITE=1         Replace an existing conversion target
USAGE
}

download_datasets() {
  mkdir -p "${LEROBOT_ROOT}"
  for dataset_name in pick_tube_01 pick_tube_02 pick_tube_03 pick_tube_04 pick_tube_05 pick_tube_06; do
    echo "Downloading KaiyueChen/${dataset_name} to ${LEROBOT_ROOT}/${dataset_name}"
    HF_ENDPOINT="${HF_ENDPOINT}" \
      HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET}" \
      HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT}" \
      "${HF_BIN}" download \
      "KaiyueChen/${dataset_name}" \
      --repo-type dataset \
      --local-dir "${LEROBOT_ROOT}/${dataset_name}" \
      --max-workers "${HF_MAX_WORKERS}"
    test -f "${LEROBOT_ROOT}/${dataset_name}/meta/info.json"
  done
}

download_encoder() {
  mkdir -p "${ENCODER_DIR}"

  download_encoder_file() {
    local filename=$1
    local expected_sha256=$2
    local output_path="${ENCODER_DIR}/${filename}"

    if [[ -f "${output_path}" ]] && printf '%s  %s\n' "${expected_sha256}" "${output_path}" | sha256sum --check --status; then
      echo "Encoder file already verified: ${output_path}"
      return
    fi

    curl --fail --location --retry 5 --retry-all-errors \
      "${ENCODER_RELEASE_BASE}/${filename}" \
      --output "${output_path}.part"
    printf '%s  %s\n' "${expected_sha256}" "${output_path}.part" | sha256sum --check --status
    mv "${output_path}.part" "${output_path}"
    echo "Downloaded and verified: ${output_path}"
  }

  download_encoder_file checkpoint.json \
    aacfdbf6754f749e3516f39cf9c947e77277d35772c5b590d2f285d6c5c477c1
  download_encoder_file "${ENCODER_PARAMS_FILE}" \
    a070b4ca6f5fa6a22a73aeae06ce18c22a473bc621d09a5892659cf5c4891797
}

download_rdp_weights() {
  mkdir -p "${RDP_WEIGHTS_DIR}"
  HF_ENDPOINT="${HF_ENDPOINT}" \
    HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET}" \
    HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT}" \
    "${HF_BIN}" download \
    "${RDP_MODEL_REPO}" \
    --revision "${RDP_MODEL_REVISION}" \
    --include '*/checkpoints/latest.ckpt' \
    --include '*/.hydra/*.yaml' \
    --include '*/normalizer.pkl' \
    --local-dir "${RDP_WEIGHTS_DIR}" \
    --max-workers "${HF_MAX_WORKERS}"
  echo "Downloaded deployment checkpoints:"
  find "${RDP_WEIGHTS_DIR}" -path '*/checkpoints/latest.ckpt' -type f -print
}

precompute_tactile() {
  if [[ -z "${JAX_PYTHON}" || ! -x "${JAX_PYTHON}" ]]; then
    echo "Set JAX_PYTHON to a CUDA-enabled JAX environment." >&2
    exit 1
  fi
  if [[ ! -f "${ENCODER_DIR}/checkpoint.json" ]]; then
    download_encoder
  fi
  XLA_PYTHON_CLIENT_PREALLOCATE=false \
    "${JAX_PYTHON}" precompute_pick_tube_v21_tactile_embeddings.py \
      --dataset-root "${LEROBOT_ROOT}" \
      --cache-root "${TACTILE_CACHE_ROOT}" \
      --encoder-path "${ENCODER_DIR}" \
      "$@"
}

fit_tactile_pca() {
  "${PYTHON_BIN}" fit_pick_tube_tactile_pca.py \
    --tactile-cache-root "${TACTILE_CACHE_ROOT}" \
    --output "${TACTILE_PCA_PATH}" \
    "$@"
}

validate_dataset() {
  local dataset_path=$1
  if [[ ! -d "${dataset_path}/replay_buffer.zarr" ]]; then
    echo "RDP Zarr not found: ${dataset_path}/replay_buffer.zarr" >&2
    exit 1
  fi

  "${PYTHON_BIN}" - "${dataset_path}" <<'PY'
from pathlib import Path
import sys
import zarr

path = Path(sys.argv[1]) / "replay_buffer.zarr"
root = zarr.open_group(str(path), mode="r")
tactile_embedding_dim = int(root["data"]["tactile_embedding"].shape[1])
expected = {
    "camera1": (224, 224, 3),
    "camera2": (224, 224, 3),
    "observation_state": (20,),
    "tactile_embedding": (tactile_embedding_dim,),
    "action": (20,),
}
for key, tail in expected.items():
    shape = root["data"][key].shape
    if shape[1:] != tail:
        raise SystemExit(f"{key}: expected [N,{tail}], got {shape}")
lengths = {root["data"][key].shape[0] for key in expected}
if len(lengths) != 1:
    raise SystemExit(f"frame counts disagree: {lengths}")
episode_ends = root["meta"]["episode_ends"][:]
if len(episode_ends) == 0 or int(episode_ends[-1]) != next(iter(lengths)):
    raise SystemExit("episode_ends does not match frame count")
episode_repeats = root["meta"]["episode_repeats"][:]
if episode_repeats.shape != episode_ends.shape or (episode_repeats < 1).any():
    raise SystemExit("episode_repeats must be positive and aligned with episode_ends")
starts = [0, *episode_ends[:-1]]
effective_frames = sum(
    (int(end) - int(start)) * int(repeat)
    for start, end, repeat in zip(starts, episode_ends, episode_repeats)
)
print(
    f"contract OK: episodes={len(episode_ends)}, physical_frames={int(episode_ends[-1])}, "
    f"effective_training_frames={effective_frames}"
)
PY
  "${PYTHON_BIN}" validate_pick_tube_batch.py "${dataset_path}" --mode at --batch-size 2
  "${PYTHON_BIN}" validate_pick_tube_batch.py "${dataset_path}" --mode ldp --batch-size 2
}

convert_dataset() {
  local output_path=$1
  shift
  if [[ ! -f "${TACTILE_PCA_PATH}" ]]; then
    echo "Tactile PCA not found: ${TACTILE_PCA_PATH}; run '$0 pca' first." >&2
    exit 1
  fi
  local args=(
    --dataset-root "${LEROBOT_ROOT}"
    --tactile-cache-root "${TACTILE_CACHE_ROOT}"
    --output-dir "${output_path}"
    --tactile-pca-path "${TACTILE_PCA_PATH}"
  )
  if [[ "${OVERWRITE}" == "1" ]]; then
    args+=(--overwrite)
  fi
  args+=("$@")
  "${PYTHON_BIN}" convert_pick_tube_lerobot_to_rdp_zarr.py "${args[@]}"
  validate_dataset "${output_path}"
}

case "${COMMAND}" in
  datasets)
    download_datasets
    ;;
  encoder)
    download_encoder
    ;;
  weights)
    download_rdp_weights
    ;;
  precompute)
    precompute_tactile "$@"
    ;;
  validate)
    validate_dataset "${DATASET_PATH}"
    ;;
  pca)
    fit_tactile_pca "$@"
    ;;
  convert)
    convert_dataset "${DATASET_PATH}" "$@"
    ;;
  smoke)
    convert_dataset "${SMOKE_DATASET_PATH}" --max-episodes-per-dataset "${SMOKE_EPISODES}" "$@"
    ;;
  -h|--help|help)
    usage
    ;;
  *)
    usage >&2
    exit 2
    ;;
esac
