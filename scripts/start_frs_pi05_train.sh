#!/usr/bin/env bash

# pi0.5 version of start_frs_train.sh. Same three-stage pipeline (tactile embeddings ->
# action caches -> FRS training), with two differences:
#   * no PEFT-merge stage: pi0.5 loads the official checkpoint directly, there is no adapter to
#     merge into a base (that stage exists only because LeRobot ships SmolVLA as a PEFT adapter).
#   * action caches come from tools/prepare_frs_pi05_cache.py, not tools/prepare_frs_caches.py.
# Stages 2 and 3 reuse the exact same base-model-agnostic tools as the SmolVLA pipeline.
#
# Environment setup and official checkpoint loading were verified on Linux with two H100 GPUs.
# The full three-stage run still requires the configured datasets and tactile encoder checkpoint.

set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
CONFIG_PATH="${1:-${PROJECT_ROOT}/configs/train_pi05_frs.yaml}"
ENV_FILE="${PROJECT_ROOT}/.env.frs"
TMUX_SESSION="${FRS_TMUX_SESSION:-frs_pick_tube_pi05}"

log() { echo "[frs-pi05] $*"; }
fail() { echo "[frs-pi05] 错误：$*" >&2; exit 1; }
trap 'status=$?; echo "[frs-pi05] 训练链路失败，退出码 ${status}" >&2; exit "${status}"' ERR

if [[ -f "${ENV_FILE}" ]]; then
    # shellcheck disable=SC1090
    source "${ENV_FILE}"
fi
if command -v uv >/dev/null 2>&1; then
    UV_BIN="$(command -v uv)"
elif [[ -x "${HOME}/.local/bin/uv" ]]; then
    UV_BIN="${HOME}/.local/bin/uv"
else
    fail "找不到 uv；请先运行 scripts/setup_env.sh"
fi
[[ -f "${CONFIG_PATH}" ]] || fail "配置不存在：${CONFIG_PATH}"

mapfile -t SETTINGS < <(
    "${UV_BIN}" run --no-sync python - "${CONFIG_PATH}" <<'PY'
from pathlib import Path
import sys
import urllib.parse
import yaml

with Path(sys.argv[1]).open(encoding="utf-8") as file:
    cfg = yaml.safe_load(file) or {}
model = cfg.get("model") or {}
training = cfg.get("frs_training") or {}
norm_stats = cfg.get("norm_stats") or {}

checkpoint = str(cfg.get("checkpoint", ""))
# Mirrors prepare_pi05.py:_is_local_path -- a gs:// (or any URL) checkpoint cannot be checked
# for existence on disk, and must not be fed to `[[ -d ... ]]`.
is_url = urllib.parse.urlparse(checkpoint).scheme != ""

print(checkpoint)
print("1" if is_url else "0")
print(training.get("output", ""))
print(model.get("tactile_encoder_path", ""))
print(str(norm_stats.get("dir", "")))
print(str(norm_stats.get("asset_id", "")))
# camera_map is required by tools/prepare_frs_pi05_cache.py; fail early here rather than after
# the (long) tactile-embedding stage.
print(str(len(model.get("camera_map") or {})))
for source in cfg.get("datasets") or []:
    print("DATASET=" + str(source.get("root", "")))
PY
)
((${#SETTINGS[@]} >= 7)) || fail "无法解析 FRS pi0.5 配置"
CHECKPOINT="${SETTINGS[0]}"
CHECKPOINT_IS_URL="${SETTINGS[1]}"
OUTPUT_DIR="${SETTINGS[2]}"
ENCODER_DIR="${SETTINGS[3]}"
NORM_STATS_DIR="${SETTINGS[4]}"
NORM_STATS_ASSET_ID="${SETTINGS[5]}"
CAMERA_MAP_SIZE="${SETTINGS[6]}"

[[ -n "${CHECKPOINT}" ]] || fail "配置缺少 checkpoint"
[[ -n "${OUTPUT_DIR}" ]] || fail "配置缺少 frs_training.output"
[[ -n "${NORM_STATS_DIR}" && -n "${NORM_STATS_ASSET_ID}" ]] \
    || fail "配置缺少 norm_stats.dir / norm_stats.asset_id（没有默认值，见 pi05_frs_plan.md）"
[[ "${CAMERA_MAP_SIZE}" != "0" ]] \
    || fail "配置缺少 model.camera_map（pi0.5 相机槽位 -> 数据集 observation key）"
[[ -d "${ENCODER_DIR}" ]] || fail "触觉 encoder 不存在：${ENCODER_DIR}"
if [[ "${CHECKPOINT_IS_URL}" == "1" ]]; then
    log "checkpoint 是远端 URL，首次运行会下载并缓存到 ${OPENPI_DATA_HOME:-${HOME}/.cache/openpi}：${CHECKPOINT}"
else
    [[ -d "${CHECKPOINT}" ]] || fail "本地 checkpoint 目录不存在：${CHECKPOINT}"
    [[ -d "${CHECKPOINT}/params" ]] \
        || fail "checkpoint 缺少 params/ 子目录（应传 checkpoint 根目录，不要带 /params）：${CHECKPOINT}"
fi
for setting in "${SETTINGS[@]:7}"; do
    [[ "${setting}" == DATASET=* ]] || continue
    dataset_root="${setting#DATASET=}"
    [[ -d "${dataset_root}" ]] || fail "数据集目录不存在：${dataset_root}"
    info_path="${dataset_root}/meta/info.json"
    [[ -f "${info_path}" ]] || fail "数据集缺少 meta/info.json：${dataset_root}"
    dataset_version="$(
        "${UV_BIN}" run --no-sync python -c \
            'import json, sys; print(json.load(open(sys.argv[1], encoding="utf-8")).get("codebase_version", ""))' \
            "${info_path}"
    )"
    [[ "${dataset_version}" == "v3.0" ]] \
        || fail "数据集必须是 LeRobot v3.0，当前为 ${dataset_version:-unknown}：${dataset_root}"
done

if [[ "${FRS_FOREGROUND:-0}" != "1" && -z "${TMUX:-}" ]] && command -v tmux >/dev/null 2>&1; then
    if tmux has-session -t "${TMUX_SESSION}" 2>/dev/null; then
        fail "tmux session 已存在：${TMUX_SESSION}"
    fi
    printf -v inner 'FRS_FOREGROUND=1 bash %q %q' "$0" "${CONFIG_PATH}"
    tmux new-session -d -s "${TMUX_SESSION}" -c "${PROJECT_ROOT}" "${inner}"
    log "FRS pi0.5 管线已在 tmux 后台启动：${TMUX_SESSION}"
    log "查看：tmux attach -t ${TMUX_SESSION}"
    exit 0
fi

cd "${PROJECT_ROOT}"
mkdir -p "${OUTPUT_DIR}"
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

# First real exercise of the vendored pi0.5 code: load the checkpoint and confirm the restored
# params match Pi0Config(pi05=True). This is verification step 2 from
# src/lerobot/policies/pi05_jax/README.md, and it is cheap compared to the stages below -- far
# better to fail here than after the tactile-embedding stage.
log "校验 pi0.5 checkpoint 能否加载（首次会下载权重，可能较慢）"
"${UV_BIN}" run --no-sync python - "${CONFIG_PATH}" <<'PY'
from pathlib import Path
import sys
import yaml

from lerobot.policies.pi05_jax import Pi0Config, load_pi0

with Path(sys.argv[1]).open(encoding="utf-8") as file:
    cfg = yaml.safe_load(file) or {}
model_cfg = cfg.get("model") or {}
paligemma_variant = str(model_cfg.get("paligemma_variant", "gemma_2b"))
action_expert_variant = str(model_cfg.get("action_expert_variant", "gemma_300m"))
config = Pi0Config(
    pi05=True,
    action_dim=int(model_cfg.get("action_dim", 32)),
    action_horizon=int(model_cfg.get("action_horizon", 50)),
    paligemma_variant=paligemma_variant,
    action_expert_variant=action_expert_variant,
)
# load_pi0 raises if the checkpoint carries parameters this config has no slot for -- the
# case that matters is a LoRA fine-tune loaded with the default (non-LoRA) variants, where
# the LoRA weights would otherwise be dropped without a word.
model = load_pi0(str(cfg["checkpoint"]), config=config)
print(
    f"pi0.5 loaded: action_dim={model.action_dim} action_horizon={model.action_horizon} "
    f"max_token_len={model.max_token_len} pi05={model.pi05} "
    f"variants={paligemma_variant}/{action_expert_variant}"
)
PY

log "预计算/补齐四数据集 tactile embeddings"
"${UV_BIN}" run --no-sync python tools/precompute_tactile_embeddings.py --config "${CONFIG_PATH}"

log "生成/补齐四数据集 pi0.5 action caches"
"${UV_BIN}" run --no-sync python tools/prepare_frs_pi05_cache.py --config "${CONFIG_PATH}"

log "开始 multi-dataset tactile FRS 训练（pi0.5 base）"
"${UV_BIN}" run --no-sync python tools/train_frs.py --config "${CONFIG_PATH}"
