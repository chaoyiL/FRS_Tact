#!/usr/bin/env bash

set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
CONFIG_PATH="${1:-${PROJECT_ROOT}/train_frs/configs/train_frs.yaml}"
ENV_FILE="${PROJECT_ROOT}/.env.frs"
TMUX_SESSION="${FRS_TMUX_SESSION:-frs_pick_tube}"

log() { echo "[frs] $*"; }
warn() { echo "[frs] 警告：$*" >&2; }
fail() { echo "[frs] 错误：$*" >&2; exit 1; }
trap 'status=$?; echo "[frs] 训练链路失败，退出码 ${status}" >&2; exit "${status}"' ERR

if [[ -f "${ENV_FILE}" ]]; then
    # shellcheck disable=SC1090
    source "${ENV_FILE}"
fi
# Hugging Face sends HF_TOKEN in an HTTP header, which only accepts ASCII.
# Ignore placeholders such as "你的token" instead of crashing deep inside urllib3.
if [[ -n "${HF_TOKEN:-}" && ! "${HF_TOKEN}" =~ ^hf_[A-Za-z0-9]+$ ]]; then
    warn "忽略格式无效的 HF_TOKEN；请使用 hf_ 开头的 ASCII token，或执行 unset HF_TOKEN"
    unset HF_TOKEN
fi
# Progress is piped through tee below.  Disable Python's block buffering so
# long JAX stages and per-batch cache progress remain visible in real time.
export PYTHONUNBUFFERED=1
# Multiprocessing and XLA compiler scratch files must live on a local filesystem.
# The workspace volume can reject pymp cleanup with EBUSY and severely delay JIT.
RUNTIME_TMPDIR="${FRS_RUNTIME_TMPDIR:-/tmp/frs_tact-${UID}}"
mkdir -p "${RUNTIME_TMPDIR}"
export TMPDIR="${RUNTIME_TMPDIR}"
if command -v uv >/dev/null 2>&1; then
    UV_BIN="$(command -v uv)"
elif [[ -x "${HOME}/.local/bin/uv" ]]; then
    UV_BIN="${HOME}/.local/bin/uv"
else
    fail "找不到 uv；请先运行 scripts/setup_env.sh"
fi

# setup_env.sh may have been interrupted after writing .env.frs but before
# syncing its external virtualenv.  Prefer that environment when usable;
# otherwise fall back to a complete project-local .venv.
CONFIGURED_PYTHON="${UV_PROJECT_ENVIRONMENT:-}/bin/python"
LOCAL_VENV="${PROJECT_ROOT}/.venv"
if [[ -n "${UV_PROJECT_ENVIRONMENT:-}" ]] \
    && { [[ ! -x "${CONFIGURED_PYTHON}" ]] \
        || ! "${CONFIGURED_PYTHON}" -c 'import yaml, jax, flax, torch' >/dev/null 2>&1; }; then
    if [[ -x "${LOCAL_VENV}/bin/python" ]] \
        && "${LOCAL_VENV}/bin/python" -c 'import yaml, jax, flax, torch' >/dev/null 2>&1; then
        warn "配置环境 ${UV_PROJECT_ENVIRONMENT} 不完整，自动改用 ${LOCAL_VENV}"
        export UV_PROJECT_ENVIRONMENT="${LOCAL_VENV}"
    else
        fail "配置环境 ${UV_PROJECT_ENVIRONMENT} 不完整，项目 .venv 也不可用；请运行 scripts/setup_env.sh"
    fi
fi

[[ -f "${CONFIG_PATH}" ]] || fail "配置不存在：${CONFIG_PATH}"

mapfile -t SETTINGS < <(
    "${UV_BIN}" run --no-sync python - "${CONFIG_PATH}" <<'PY'
from pathlib import Path
import sys
import yaml

with Path(sys.argv[1]).open(encoding="utf-8") as file:
    cfg = yaml.safe_load(file) or {}
merge = cfg.get("checkpoint_merge") or {}
training = cfg.get("frs_training") or {}
checkpoint = Path(str(cfg.get("checkpoint", ""))).expanduser().resolve()
merge_output = Path(str(merge.get("output", cfg.get("checkpoint", "")))).expanduser().resolve()
if checkpoint != merge_output:
    raise ValueError(
        f"checkpoint_merge.output must equal checkpoint: {merge_output} != {checkpoint}"
    )
print(merge.get("adapter", ""))
print(merge.get("base", "lerobot/smolvla_base"))
print(merge.get("output", cfg.get("checkpoint", "")))
print("1" if merge.get("allow_download", True) else "0")
print(training.get("output", ""))
print((cfg.get("model") or {}).get("tactile_encoder_path", ""))
for source in cfg.get("datasets") or []:
    print("DATASET=" + str(source.get("root", "")))
PY
)
((${#SETTINGS[@]} >= 6)) || fail "无法解析 FRS 配置"
ADAPTER_ID="${SETTINGS[0]}"
BASE_ID="${SETTINGS[1]}"
MERGED_CHECKPOINT="${SETTINGS[2]}"
ALLOW_DOWNLOAD="${SETTINGS[3]}"
OUTPUT_DIR="${SETTINGS[4]}"
ENCODER_DIR="${SETTINGS[5]}"
[[ -n "${ADAPTER_ID}" && -n "${MERGED_CHECKPOINT}" && -n "${OUTPUT_DIR}" ]] || fail "配置缺少 checkpoint/output"
[[ -d "${ENCODER_DIR}" ]] || fail "触觉 encoder 不存在：${ENCODER_DIR}"
for setting in "${SETTINGS[@]:6}"; do
    [[ "${setting}" == DATASET=* ]] || continue
    dataset_root="${setting#DATASET=}"
    [[ -d "${dataset_root}" ]] || fail "数据集目录不存在：${dataset_root}"
done

if [[ "${FRS_FOREGROUND:-0}" != "1" && -z "${TMUX:-}" ]] && command -v tmux >/dev/null 2>&1; then
    if tmux has-session -t "${TMUX_SESSION}" 2>/dev/null; then
        fail "tmux session 已存在：${TMUX_SESSION}"
    fi
    printf -v inner 'FRS_FOREGROUND=1 bash %q %q' "$0" "${CONFIG_PATH}"
    tmux new-session -d -s "${TMUX_SESSION}" -c "${PROJECT_ROOT}" "${inner}"
    log "FRS 管线已在 tmux 后台启动：${TMUX_SESSION}"
    log "查看：tmux attach -t ${TMUX_SESSION}"
    exit 0
fi

cd "${PROJECT_ROOT}"
JAX_CACHE_DIR="${FRS_JAX_COMPILATION_CACHE_DIR:-${OUTPUT_DIR}/jax_compilation_cache}"
mkdir -p "${OUTPUT_DIR}" "${JAX_CACHE_DIR}"
export JAX_COMPILATION_CACHE_DIR="${JAX_CACHE_DIR}"
timestamp="$(date +%Y%m%d_%H%M%S)"
pipeline_log="${OUTPUT_DIR}/pipeline_${timestamp}.log"
exec > >(tee -a "${pipeline_log}") 2>&1

"${UV_BIN}" run --no-sync python - <<'PY'
import jax
devices = jax.devices()
print(f"JAX devices={devices}")
if not any(device.platform == "gpu" for device in devices):
    raise RuntimeError("JAX 没有识别到 GPU，拒绝启动 FRS 正式管线")
PY

download_flag="--allow-download"
if [[ "${ALLOW_DOWNLOAD}" != "1" ]]; then
    download_flag="--no-allow-download"
fi
log "合并/检查 SmolVLA PEFT checkpoint"
"${UV_BIN}" run --no-sync python tools/merge_smolvla_peft_to_jax.py \
    --adapter "${ADAPTER_ID}" \
    --base "${BASE_ID}" \
    --output "${MERGED_CHECKPOINT}" \
    "${download_flag}"

log "小样本 A/B 检查 FireFlow 与 SlerpFlow 反向积分"
"${UV_BIN}" run --no-sync python -m train_frs.compare_frs_reverse_solvers --config "${CONFIG_PATH}"

log "预计算/补齐全部配置数据集的 tactile embeddings"
"${UV_BIN}" run --no-sync python tools/precompute_tactile_embeddings.py --config "${CONFIG_PATH}"

log "生成/补齐全部配置数据集的 SmolVLA action caches"
# JAX/XLA 0.8.3's generic Triton GEMM emitter cannot tile one SmolVLA prefix
# projection on H100.  Use the cuBLAS GEMM path for cache preparation only.
# Training starts as a separate command below with its normal XLA configuration.
PREPARE_XLA_FLAGS="${FRS_PREPARE_XLA_FLAGS:---xla_gpu_enable_triton_gemm=false}"
log "action-cache XLA_FLAGS=${PREPARE_XLA_FLAGS}"
XLA_FLAGS="${PREPARE_XLA_FLAGS}" \
    "${UV_BIN}" run --no-sync python -m train_frs.prepare_frs_caches --config "${CONFIG_PATH}"

log "开始 multi-dataset tactile FRS 训练"
"${UV_BIN}" run --no-sync python -m train_frs.train_frs --config "${CONFIG_PATH}"
