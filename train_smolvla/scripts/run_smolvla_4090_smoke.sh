#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
SMOKE_CONFIG="${PROJECT_ROOT}/train_smolvla/configs/train_smolvla_4090_smoke.yaml"
SMOKE_OUTPUT_PREFIX="/workspace/outputs/smolvla_4090_smoke"
SMOKE_STAGE="all"

log() {
    echo "[smolvla-smoke] $*"
}

fail() {
    echo "[smolvla-smoke] ERROR: $*" >&2
    exit 1
}

usage() {
    cat <<'EOF'
Usage: bash train_smolvla/scripts/run_smolvla_4090_smoke.sh [--stage STAGE]

Run a real single-RTX-4090 SmolVLA acceptance pipeline with two_tubes_04.

Stages:
  env         Install and validate the isolated SmolVLA environment and GPU
  data        Download/convert two_tubes_04 to the local LeRobot v3 dataset
  sample      Validate metadata and decode one real two-camera video sample
  preflight   Validate and print the exact official LeRobot training command
  train       Run five real forward/backward optimization steps and save
  checkpoint  Validate weights, optimizer state, step, dimensions, and cameras
  all         Run all stages in order (default)

Compatibility switches:
  SMOLVLA_SMOKE_SKIP_SETUP=1       Skip installation but still validate the env
  SMOLVLA_SMOKE_SKIP_DOWNLOAD=1    Skip download but still validate/decode data
  SMOLVLA_SMOKE_ALLOW_NON_4090=1   Allow another CUDA GPU for development
EOF
}

while (($#)); do
    case "$1" in
        --stage)
            (($# >= 2)) || fail "--stage requires a value"
            SMOKE_STAGE="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            fail "unknown argument: $1"
            ;;
    esac
done

case "${SMOKE_STAGE}" in
    all|env|data|sample|preflight|train|checkpoint) ;;
    *) fail "invalid stage '${SMOKE_STAGE}'; run with --help" ;;
esac

should_run() {
    [[ "${SMOKE_STAGE}" == "all" || "${SMOKE_STAGE}" == "$1" ]]
}

load_environment() {
    local env_file="${PROJECT_ROOT}/env_path"
    local previous_env_file="${PROJECT_ROOT}/environment_paths.sh"
    local legacy_env_file="${PROJECT_ROOT}/.env.frs"
    if [[ ! -f "${env_file}" ]]; then
        if [[ -f "${previous_env_file}" ]]; then
            env_file="${previous_env_file}"
        elif [[ -f "${legacy_env_file}" ]]; then
            env_file="${legacy_env_file}"
        fi
    fi
    [[ -f "${env_file}" ]] || fail \
        "missing env_path; run: bash scripts/setup_env.sh --smolvla"
    # shellcheck disable=SC1090
    source "${env_file}"
    [[ -x "${SMOLVLA_TORCH_PYTHON:-}" ]] || fail \
        "SMOLVLA_TORCH_PYTHON is not executable: ${SMOLVLA_TORCH_PYTHON:-unset}"
    log "environment index: ${env_file}"
    log "training Python: ${SMOLVLA_TORCH_PYTHON}"
}

configure_local_runtime_cache() {
    if [[ "${SMOLVLA_USE_LOCAL_ARROW_CACHE:-1}" == "1" ]]; then
        local cache_root="${SMOLVLA_LOCAL_CACHE_ROOT:-/tmp/frs_tact_smolvla}"
        export HF_DATASETS_CACHE="${cache_root}/datasets_arrow"
        export TMPDIR="${cache_root}/tmp"
        mkdir -p "${HF_DATASETS_CACHE}" "${TMPDIR}"
        log "local Arrow cache: ${HF_DATASETS_CACHE}"
        log "local temp directory: ${TMPDIR}"
    fi
}

validate_environment() {
    command -v nvidia-smi >/dev/null 2>&1 || fail "nvidia-smi is unavailable"
    [[ "${PROJECT_ROOT}" == /workspace/* ]] || fail \
        "the RunPod acceptance test expects the repository under /workspace, got ${PROJECT_ROOT}"
    [[ "${WORKSPACE_ROOT:-}" == "/workspace" ]] || fail \
        "WORKSPACE_ROOT must be /workspace, got ${WORKSPACE_ROOT:-unset}"
    [[ "${SMOLVLA_TORCH_PYTHON}" == /workspace/* ]] || fail \
        "the training environment must be stored under /workspace"
    [[ "${HF_HOME:-}" == /workspace/* ]] || fail \
        "HF_HOME must be stored under /workspace, got ${HF_HOME:-unset}"
    [[ "${HF_LEROBOT_HOME:-}" == /workspace/* ]] || fail \
        "HF_LEROBOT_HOME must be stored under /workspace, got ${HF_LEROBOT_HOME:-unset}"
    log "persistent model/data cache: ${HF_HOME}"
    log "persistent LeRobot cache: ${HF_LEROBOT_HOME}"
    log "persistent training output: $(dirname -- "${SMOKE_OUTPUT_PREFIX}")"
    nvidia-smi --query-gpu=index,name,driver_version,memory.total --format=csv,noheader
    "${SMOLVLA_TORCH_PYTHON}" - <<'PY'
from importlib.metadata import version
import os
import platform

from packaging.version import Version
import torch

expected = {
    "lerobot": "0.6.1",
    "torchcodec": "0.5.0",
}
actual = {name: version(name) for name in expected}
for name, wanted in expected.items():
    if Version(actual[name]) != Version(wanted):
        raise RuntimeError(f"{name} must be {wanted}, got {actual[name]}")
if not version("torch").startswith("2.7.1"):
    raise RuntimeError(f"torch must be 2.7.1, got {version('torch')}")
if not torch.cuda.is_available():
    raise RuntimeError("PyTorch cannot access CUDA")
if torch.cuda.device_count() != 1:
    raise RuntimeError(
        "the 4090 acceptance run must expose exactly one GPU; "
        f"PyTorch sees {torch.cuda.device_count()}"
    )
name = torch.cuda.get_device_name(0)
if "4090" not in name and os.environ.get("SMOLVLA_SMOKE_ALLOW_NON_4090") != "1":
    raise RuntimeError(
        f"expected an RTX 4090, got {name!r}; set "
        "SMOLVLA_SMOKE_ALLOW_NON_4090=1 only for development"
    )
if not torch.cuda.is_bf16_supported():
    raise RuntimeError(f"{name} does not report bf16 support")
print(f"[smolvla-smoke] Python: {platform.python_version()}")
print(
    "[smolvla-smoke] packages: "
    f"lerobot={actual['lerobot']} torch={version('torch')} "
    f"torchcodec={actual['torchcodec']} accelerate={version('accelerate')}"
)
print(f"[smolvla-smoke] CUDA device: {name}")
print("[smolvla-smoke] bf16: supported")
PY
    df -h / /workspace /tmp 2>/dev/null || true
}

validate_and_decode_sample() {
    export SMOLVLA_SMOKE_CONFIG="${SMOKE_CONFIG}"
    "${SMOLVLA_TORCH_PYTHON}" - <<'PY'
import json
import os
from pathlib import Path

from train_smolvla.torch_train import (
    _import_official_lerobot,
    _load,
    _select_dataset_cameras,
    dataset_sources,
    validate_dataset_contract,
)

config = _load(Path(os.environ["SMOLVLA_SMOKE_CONFIG"]))
validate_dataset_contract(config)
source = dataset_sources(config)[0]
root = Path(source["root"])
info = json.loads((root / "meta" / "info.json").read_text(encoding="utf-8"))
if info.get("codebase_version") != "v3.0":
    raise RuntimeError(
        f"expected LeRobot v3.0, got {info.get('codebase_version')!r}"
    )
parquets = list(root.glob("data/**/*.parquet"))
videos = list(root.glob("videos/**/*.mp4"))
if not parquets:
    raise RuntimeError(f"no parquet files found under {root / 'data'}")
for key in config["dataset"]["image_keys"]:
    if not any(key in str(path) for path in videos):
        raise RuntimeError(f"no mp4 files found for required camera {key}")

_import_official_lerobot()
from lerobot.datasets.lerobot_dataset import LeRobotDataset

dataset = LeRobotDataset(
    repo_id=source["repo_id"],
    root=root,
    revision=source.get("revision"),
    download_videos=False,
    video_backend="torchcodec",
)
selected = set(config["dataset"]["image_keys"])
_select_dataset_cameras(dataset, selected)
sample = dataset[0]
if tuple(sample["observation.state"].shape) != (20,):
    raise RuntimeError(
        f"decoded observation.state must be (20,), got {sample['observation.state'].shape}"
    )
if tuple(sample["action"].shape) != (20,):
    raise RuntimeError(f"decoded action must be (20,), got {sample['action'].shape}")
decoded_images = {key for key in sample if key.startswith("observation.images.")}
if decoded_images != selected:
    raise RuntimeError(
        f"decoded cameras must be {sorted(selected)}, got {sorted(decoded_images)}"
    )
for key in sorted(selected):
    image = sample[key]
    if image.ndim != 3 or image.shape[0] != 3:
        raise RuntimeError(f"decoded {key} must be CHW RGB, got {tuple(image.shape)}")
    if not image.isfinite().all():
        raise RuntimeError(f"decoded {key} contains non-finite values")

print(
    f"[smolvla-smoke] dataset: version=v3.0 fps={info['fps']} "
    f"episodes={info.get('total_episodes')} frames={len(dataset)}"
)
print(
    f"[smolvla-smoke] files: parquet={len(parquets)} mp4={len(videos)}; "
    "TorchCodec decoded sample 0"
)
print(
    "[smolvla-smoke] sample contract: state=20D action=20D cameras="
    + str(sorted(decoded_images))
)
PY
}

find_latest_checkpoint() {
    local latest_output checkpoint_dir
    latest_output="$({
        find "$(dirname -- "${SMOKE_OUTPUT_PREFIX}")" -maxdepth 1 -type d \
            -name "$(basename -- "${SMOKE_OUTPUT_PREFIX}")*" \
            -printf '%T@ %p\n' 2>/dev/null || true
    } | sort -nr | head -n 1 | cut -d' ' -f2-)"
    [[ -n "${latest_output}" && -d "${latest_output}" ]] || \
        fail "no smoke-test output directory found"
    checkpoint_dir="$(find "${latest_output}/checkpoints" -mindepth 1 -maxdepth 1 \
        -type d -name '[0-9]*' -print 2>/dev/null | sort | tail -n 1)"
    [[ -n "${checkpoint_dir}" ]] || fail "no numbered checkpoint found in ${latest_output}"
    printf '%s\n%s\n' "${latest_output}" "${checkpoint_dir}"
}

validate_checkpoint() {
    local paths latest_output checkpoint_dir
    mapfile -t paths < <(find_latest_checkpoint)
    ((${#paths[@]} == 2)) || fail "could not resolve the latest checkpoint"
    latest_output="${paths[0]}"
    checkpoint_dir="${paths[1]}"
    export SMOLVLA_SMOKE_CHECKPOINT_DIR="${checkpoint_dir}"
    "${SMOLVLA_TORCH_PYTHON}" - <<'PY'
import json
import os
from pathlib import Path

checkpoint = Path(os.environ["SMOLVLA_SMOKE_CHECKPOINT_DIR"])
model_dir = checkpoint / "pretrained_model"
state_dir = checkpoint / "training_state"
required_model_files = ["config.json", "train_config.json"]
required_state_files = [
    "training_step.json",
    "optimizer_param_groups.json",
    "optimizer_state.safetensors",
    "rng_state.safetensors",
    "scheduler_state.json",
]
missing = [
    str(path)
    for path in (
        *(model_dir / name for name in required_model_files),
        *(state_dir / name for name in required_state_files),
    )
    if not path.is_file()
]
if missing:
    raise RuntimeError(f"checkpoint is incomplete; missing: {missing}")
weights = list(model_dir.glob("*.safetensors")) + list(model_dir.glob("*.bin"))
if not weights:
    raise RuntimeError(f"no model/adapter weights found in {model_dir}")

step_state = json.loads((state_dir / "training_step.json").read_text(encoding="utf-8"))
if step_state.get("step") != 5:
    raise RuntimeError(f"checkpoint step must be 5, got {step_state}")
if step_state.get("num_processes") not in (None, 1):
    raise RuntimeError(f"checkpoint must be single-process, got {step_state}")
if step_state.get("batch_size") not in (None, 1):
    raise RuntimeError(f"checkpoint batch size must be 1, got {step_state}")

policy = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
inputs = policy.get("input_features", {})
outputs = policy.get("output_features", {})
state_shape = inputs.get("observation.state", {}).get("shape")
action_shape = outputs.get("action", {}).get("shape")
cameras = {key for key in inputs if key.startswith("observation.images.")}
expected_cameras = {
    "observation.images.camera1",
    "observation.images.camera2",
}
if state_shape != [20] or action_shape != [20] or cameras != expected_cameras:
    raise RuntimeError(
        "exported policy contract mismatch: "
        f"state={state_shape}, action={action_shape}, cameras={sorted(cameras)}"
    )
print(f"[smolvla-smoke] checkpoint step: {step_state}")
print(
    "[smolvla-smoke] exported contract: state=20D action=20D cameras="
    + str(sorted(cameras))
)
print("[smolvla-smoke] weight files: " + ", ".join(path.name for path in weights))
PY
    log "output directory: ${latest_output}"
    log "checkpoint: ${checkpoint_dir}"
}

cd "${PROJECT_ROOT}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

if should_run env; then
    log "[1/6] environment setup and RTX 4090 validation"
    if [[ "${SMOLVLA_SMOKE_SKIP_SETUP:-0}" != "1" ]]; then
        bash scripts/setup_env.sh --smolvla
    else
        log "environment installation skipped; validating the existing environment"
    fi
    load_environment
    configure_local_runtime_cache
    validate_environment
fi

if should_run data; then
    log "[2/6] download/convert two_tubes_04"
    if [[ "${SMOLVLA_SMOKE_SKIP_DOWNLOAD:-0}" != "1" ]]; then
        bash scripts/download_data.sh --dataset two_tubes_04
    else
        log "download skipped; the sample stage will still validate existing data"
    fi
fi

if should_run sample; then
    log "[3/6] metadata contract and real TorchCodec frame decode"
    load_environment
    configure_local_runtime_cache
    validate_and_decode_sample
fi

if should_run preflight; then
    log "[4/6] official LeRobot command preflight"
    load_environment
    configure_local_runtime_cache
    "${SMOLVLA_TORCH_PYTHON}" -m train_smolvla.torch_train \
        --config "${SMOKE_CONFIG}" --dry-run
fi

if should_run train; then
    log "[5/6] five real forward/backward optimization steps"
    load_environment
    configure_local_runtime_cache
    SMOLVLA_TRAIN_CONFIG="${SMOKE_CONFIG}" \
        bash train_smolvla/scripts/start_smolvla_train.sh
fi

if should_run checkpoint; then
    log "[6/6] checkpoint and exported policy contract"
    load_environment
    validate_checkpoint
fi

if [[ "${SMOKE_STAGE}" == "all" ]]; then
    log "PASS: environment, data, real decode, preflight, training, and checkpoint all passed"
else
    log "PASS: stage '${SMOKE_STAGE}' passed"
fi
