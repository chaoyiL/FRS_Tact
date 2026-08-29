#!/usr/bin/env bash

# Leave-one-out ablation of the four optional gated losses.
# Each run turns off one of aux_decode / low_gate_safety / rank / repair
# and keeps the other three on. Shared caches are prepared once from the
# official YAML; checkpoints go to <repo>/checkpoints/frs/no_<loss>.
#
#   bash train_smolvla_frs/scripts/start_frs_loss_ablation.sh
#   bash train_smolvla_frs/scripts/start_frs_loss_ablation.sh train_smolvla_frs/configs/train_frs.yaml
#
# Skip merge/cache prep when they already exist:
#   FRS_SKIP_PREP=1 bash train_smolvla_frs/scripts/start_frs_loss_ablation.sh

set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
CONFIG_PATH="${1:-${PROJECT_ROOT}/train_smolvla_frs/configs/train_frs.yaml}"
OUTPUT_ROOT="${FRS_ABLATION_OUTPUT_ROOT:-${PROJECT_ROOT}/checkpoints/frs}"
ENV_FILE="${PROJECT_ROOT}/env_path"
PREVIOUS_ENV_FILE="${PROJECT_ROOT}/environment_paths.sh"
LEGACY_ENV_FILE="${PROJECT_ROOT}/.env.frs"
if [[ ! -f "${ENV_FILE}" ]]; then
    if [[ -f "${PREVIOUS_ENV_FILE}" ]]; then
        ENV_FILE="${PREVIOUS_ENV_FILE}"
    elif [[ -f "${LEGACY_ENV_FILE}" ]]; then
        ENV_FILE="${LEGACY_ENV_FILE}"
    fi
fi
TMUX_SESSION="${FRS_TMUX_SESSION:-frs_loss_ablation}"

log() { echo "[frs-ablation] $*"; }
warn() { echo "[frs-ablation] 警告：$*" >&2; }
fail() { echo "[frs-ablation] 错误：$*" >&2; exit 1; }
trap 'status=$?; echo "[frs-ablation] 训练链路失败，退出码 ${status}" >&2; exit "${status}"' ERR

if [[ -f "${ENV_FILE}" ]]; then
    # shellcheck disable=SC1090
    source "${ENV_FILE}"
fi
if [[ -n "${HF_TOKEN:-}" && ! "${HF_TOKEN}" =~ ^hf_[A-Za-z0-9]+$ ]]; then
    warn "忽略格式无效的 HF_TOKEN；请使用 hf_ 开头的 ASCII token，或执行 unset HF_TOKEN"
    unset HF_TOKEN
fi
export PYTHONUNBUFFERED=1
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
checkpoint = Path(str(cfg.get("checkpoint", ""))).expanduser().resolve()
print(merge.get("adapter", ""))
print(merge.get("base", "lerobot/smolvla_base"))
print(str(checkpoint))
print("1" if merge.get("allow_download", True) else "0")
print((cfg.get("model") or {}).get("tactile_encoder_path", ""))
for source in cfg.get("datasets") or []:
    print("DATASET=" + str(source.get("root", "")))
PY
)
((${#SETTINGS[@]} >= 5)) || fail "无法解析 FRS 配置"
ADAPTER_ID="${SETTINGS[0]}"
BASE_ID="${SETTINGS[1]}"
MERGED_CHECKPOINT="${SETTINGS[2]}"
ALLOW_DOWNLOAD="${SETTINGS[3]}"
ENCODER_DIR="${SETTINGS[4]}"
[[ -n "${ADAPTER_ID}" && -n "${MERGED_CHECKPOINT}" ]] || fail "配置缺少 checkpoint"
[[ -d "${ENCODER_DIR}" ]] || fail "触觉 encoder 不存在：${ENCODER_DIR}"
for setting in "${SETTINGS[@]:5}"; do
    [[ "${setting}" == DATASET=* ]] || continue
    dataset_root="${setting#DATASET=}"
    [[ -d "${dataset_root}" ]] || fail "数据集目录不存在：${dataset_root}"
done

if [[ "${FRS_FOREGROUND:-0}" != "1" && -z "${TMUX:-}" ]] && command -v tmux >/dev/null 2>&1; then
    if tmux has-session -t "${TMUX_SESSION}" 2>/dev/null; then
        fail "tmux session 已存在：${TMUX_SESSION}"
    fi
    printf -v inner 'FRS_FOREGROUND=1 FRS_ABLATION_OUTPUT_ROOT=%q bash %q %q' \
        "${OUTPUT_ROOT}" "$0" "${CONFIG_PATH}"
    tmux new-session -d -s "${TMUX_SESSION}" -c "${PROJECT_ROOT}" "${inner}"
    log "loss ablation 已在 tmux 后台启动：${TMUX_SESSION}"
    log "查看：tmux attach -t ${TMUX_SESSION}"
    exit 0
fi

cd "${PROJECT_ROOT}"
mkdir -p "${OUTPUT_ROOT}"
JAX_CACHE_DIR="${FRS_JAX_COMPILATION_CACHE_DIR:-${OUTPUT_ROOT}/jax_compilation_cache}"
mkdir -p "${JAX_CACHE_DIR}"
export JAX_COMPILATION_CACHE_DIR="${JAX_CACHE_DIR}"
timestamp="$(date +%Y%m%d_%H%M%S)"
pipeline_log="${OUTPUT_ROOT}/pipeline_${timestamp}.log"
exec > >(tee -a "${pipeline_log}") 2>&1

"${UV_BIN}" run --no-sync python - <<'PY'
import jax
devices = jax.devices()
print(f"JAX devices={devices}")
if not any(device.platform == "gpu" for device in devices):
    raise RuntimeError("JAX 没有识别到 GPU，拒绝启动 FRS ablation 管线")
PY

if [[ "${FRS_SKIP_PREP:-0}" != "1" ]]; then
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
    "${UV_BIN}" run --no-sync python -m train_smolvla_frs.compare_frs_reverse_solvers --config "${CONFIG_PATH}"

    log "生成/补齐 tactile embedding caches 与 SmolVLA action caches"
    PREPARE_XLA_FLAGS="${FRS_PREPARE_XLA_FLAGS:---xla_gpu_enable_triton_gemm=false}"
    log "action-cache XLA_FLAGS=${PREPARE_XLA_FLAGS}"
    XLA_FLAGS="${PREPARE_XLA_FLAGS}" \
        "${UV_BIN}" run --no-sync python -m train_smolvla_frs.prepare_frs_caches --config "${CONFIG_PATH}"
else
    log "跳过 checkpoint 合并与 cache 准备（FRS_SKIP_PREP=1）"
fi

log "写出 leave-one-out 配置到 ${OUTPUT_ROOT}"
mapfile -t ABLATION_RUNS < <(
    "${UV_BIN}" run --no-sync python -m train_smolvla_frs.utils.loss_ablation \
        --config "${CONFIG_PATH}" \
        --output-root "${OUTPUT_ROOT}"
)
((${#ABLATION_RUNS[@]} == 4)) || fail "期望 4 组 ablation 配置，实际 ${#ABLATION_RUNS[@]}"

for entry in "${ABLATION_RUNS[@]}"; do
    run_name="${entry%%$'\t'*}"
    run_config="${entry#*$'\t'}"
    run_dir="${OUTPUT_ROOT}/${run_name}"
    if [[ -f "${run_dir}/best/checkpoint.json" ]]; then
        log "跳过已完成：${run_name}（已有 best checkpoint）"
        continue
    fi
    log "开始训练 ${run_name}  config=${run_config}"
    "${UV_BIN}" run --no-sync python -m train_smolvla_frs.train_frs --config "${run_config}"
    log "完成 ${run_name}  output=${run_dir}"
done

log "四组 ablation 全部结束，结果目录：${OUTPUT_ROOT}"
log "  ${OUTPUT_ROOT}/no_aux_decode"
log "  ${OUTPUT_ROOT}/no_low_gate_safety"
log "  ${OUTPUT_ROOT}/no_rank"
log "  ${OUTPUT_ROOT}/no_repair"
