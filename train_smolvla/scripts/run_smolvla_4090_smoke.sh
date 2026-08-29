#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
SMOKE_CONFIG="${PROJECT_ROOT}/train_smolvla/configs/train_smolvla_4090_smoke.yaml"
SMOKE_OUTPUT_PREFIX="/workspace/outputs/smolvla_4090_smoke"

log() {
    echo "[smolvla-smoke] $*"
}

cd "${PROJECT_ROOT}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

if [[ "${SMOLVLA_SMOKE_SKIP_SETUP:-0}" != "1" ]]; then
    log "1/3 配置 SmolVLA 环境"
    bash scripts/setup_env.sh --smolvla
else
    log "1/3 跳过环境配置"
fi

if [[ "${SMOLVLA_SMOKE_SKIP_DOWNLOAD:-0}" != "1" ]]; then
    log "2/3 下载/升级 two_tubes_04 到 LeRobot v3.0"
    bash scripts/download_data.sh --dataset two_tubes_04
else
    log "2/3 跳过数据下载"
fi

[[ -f "${PROJECT_ROOT}/.env.frs" ]] || {
    echo "[smolvla-smoke] 缺少 ${PROJECT_ROOT}/.env.frs" >&2
    exit 1
}
# shellcheck disable=SC1091
source "${PROJECT_ROOT}/.env.frs"
[[ -x "${SMOLVLA_TORCH_PYTHON:-}" ]] || {
    echo "[smolvla-smoke] SmolVLA Python 不可执行：${SMOLVLA_TORCH_PYTHON:-unset}" >&2
    exit 1
}

"${SMOLVLA_TORCH_PYTHON}" - <<'PY'
import torch

count = torch.cuda.device_count()
if not torch.cuda.is_available() or count != 1:
    raise RuntimeError(f"冒烟测试必须只暴露一张 GPU，当前 PyTorch 检测到 {count} 张")
print(f"[smolvla-smoke] GPU: {torch.cuda.get_device_name(0)}")
PY

log "3/3 运行 5-step 单卡训练并保存 checkpoint"
SMOLVLA_TRAIN_CONFIG="${SMOKE_CONFIG}" \
    bash train_smolvla/scripts/start_smolvla_train.sh

latest_output="$({
    find "$(dirname -- "${SMOKE_OUTPUT_PREFIX}")" -maxdepth 1 -type d \
        -name "$(basename -- "${SMOKE_OUTPUT_PREFIX}")*" \
        -printf '%T@ %p\n' 2>/dev/null || true
} | sort -nr | head -n 1 | cut -d' ' -f2-)"
[[ -n "${latest_output}" && -d "${latest_output}" ]] || {
    echo "[smolvla-smoke] 未找到训练输出目录" >&2
    exit 1
}

checkpoint_dir="$(find "${latest_output}/checkpoints" -mindepth 1 -maxdepth 1 \
    -type d -print 2>/dev/null | sort | tail -n 1)"
[[ -n "${checkpoint_dir}" && -d "${checkpoint_dir}/pretrained_model" ]] || {
    echo "[smolvla-smoke] 未找到最终 checkpoint/pretrained_model" >&2
    exit 1
}
weight_file="$(find "${checkpoint_dir}/pretrained_model" -type f \
    \( -name '*.safetensors' -o -name '*.bin' \) -print -quit)"
[[ -n "${weight_file}" ]] || {
    echo "[smolvla-smoke] checkpoint 中没有模型权重" >&2
    exit 1
}

log "PASS：环境、two_tubes_04、5-step 训练和 checkpoint 链路全部通过"
log "输出目录：${latest_output}"
log "checkpoint：${checkpoint_dir}"
