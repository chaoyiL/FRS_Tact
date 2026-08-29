#!/usr/bin/env bash
# DECO Stage2: RTX PRO 6000 single-GPU / 16-vCPU server workflow.
set -Eeuo pipefail

WORKSPACE_ROOT="/workspace"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
CODE_ROOT="${CODE_ROOT:-$(cd -- "${SCRIPT_DIR}/../.." && pwd -P)}"
VENV_PATH="${VENV_PATH:-/workspace/venvs/deco-stage2}"
DATA_ROOT="${DATA_ROOT:-/workspace/data/lerobot/KaiyueChen}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-/workspace/checkpoints}"
STAGE1_ROOT="${STAGE1_ROOT:-${CHECKPOINT_ROOT}/deco/stage1}"
ENCODER_ROOT="${ENCODER_ROOT:-${CHECKPOINT_ROOT}/encoder/encoder_ckpt_0824}"
TACTILE_CACHE="${TACTILE_CACHE:-${CHECKPOINT_ROOT}/deco/tactile_encoder_cache}"
MANIFEST_ROOT="${MANIFEST_ROOT:-/workspace/manifests/deco-stage2}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/workspace/outputs/deco-stage2}"
STATE_ROOT="${STATE_ROOT:-/workspace/state/deco-stage2}"
LOG_ROOT="${LOG_ROOT:-/workspace/logs/deco-stage2}"
HF_HOME="${HF_HOME:-/workspace/huggingface}"
HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
WANDB_DIR="${WANDB_DIR:-/workspace/wandb}"
WANDB_CACHE_DIR="${WANDB_CACHE_DIR:-${WANDB_DIR}/cache}"
WANDB_CONFIG_DIR="${WANDB_CONFIG_DIR:-${WANDB_DIR}/config}"
WANDB_DATA_DIR="${WANDB_DATA_DIR:-${WANDB_DIR}/data}"

DECO_REVISION="ebce57a643f108e0f52a408e4bf2721861f78baf"
ENCODER_REVISION="1a432a8b9c1a38f75cb40c58a4924a401486d832"
INSERT_01_REVISION="deead6367f0a2d817306a28bcaecc089b5cfe653"
INSERT_02_REVISION="babbcf401b84640b599c084a9121e80561944e8f"
BREAD_01_REVISION="5f6613a7137d4827b3e3def106ac576f759fd10f"
BREAD_02_REVISION="41198c2da2c23951b4141fdb97b799530d72d658"
BREAD_03_REVISION="f32e32b936e656338c96a164d8896e5156e66811"

INSERT_MANIFEST="${MANIFEST_ROOT}/insert_01_02.json"
BREAD_MANIFEST="${MANIFEST_ROOT}/bread_01_03.json"
INSERT_STAGE1="${INSERT_STAGE1_CHECKPOINT:-${STAGE1_ROOT}/insert/deco_stage1_latest.pt}"
BREAD_STAGE1="${BREAD_STAGE1_CHECKPOINT:-${STAGE1_ROOT}/bread/deco_stage1_latest.pt}"
PYTHON_BIN="${VENV_PATH}/bin/python"
HF_BIN="${VENV_PATH}/bin/hf"
SERVER_ENV_FILE="${SERVER_ENV_FILE:-/workspace/secrets/deco-stage2.env}"

export HF_HOME HF_HUB_CACHE HF_DATASETS_CACHE
export WANDB_DIR WANDB_CACHE_DIR WANDB_CONFIG_DIR WANDB_DATA_DIR
export TOKENIZERS_PARALLELISM=false
if [[ -f "${SERVER_ENV_FILE}" ]]; then
    set -a
    source "${SERVER_ENV_FILE}"
    set +a
fi

log() { printf '[deco-stage2-server] %s\n' "$*"; }
warn() { printf '[deco-stage2-server] 警告：%s\n' "$*" >&2; }
fail() { printf '[deco-stage2-server] 错误：%s\n' "$*" >&2; exit 1; }

usage() {
    cat <<'EOF'
用法：
  bash train_deco/scripts/server_stage2_rtxpro6000.sh setup
  bash train_deco/scripts/server_stage2_rtxpro6000.sh download
  bash train_deco/scripts/server_stage2_rtxpro6000.sh prepare
  bash train_deco/scripts/server_stage2_rtxpro6000.sh doctor
  bash train_deco/scripts/server_stage2_rtxpro6000.sh train  insert|bread
  bash train_deco/scripts/server_stage2_rtxpro6000.sh upload insert|bread [RUN_ID]
  bash train_deco/scripts/server_stage2_rtxpro6000.sh run    insert|bread
  bash train_deco/scripts/server_stage2_rtxpro6000.sh all

all：配置环境、下载、准备 manifest，然后依次训练并上传 Insert 01~02 和
Bread 01~03。bread_04 不下载、不使用。默认单卡物理 batch size=512；
workers 使用 vCPU 的 75%，所以 16 vCPU 对应 12 workers；每个 worker
只预取 1 个 batch，以控制大 batch 的主机内存峰值。
EOF
}

validate_paths() {
    local value
    for value in \
        "${CODE_ROOT}" "${VENV_PATH}" "${DATA_ROOT}" "${CHECKPOINT_ROOT}" \
        "${MANIFEST_ROOT}" "${OUTPUT_ROOT}" "${STATE_ROOT}" "${LOG_ROOT}" \
        "${HF_HOME}" "${WANDB_DIR}"; do
        case "${value}" in
            /workspace|/workspace/*) ;;
            *) fail "所有路径必须位于 /workspace：${value}" ;;
        esac
    done
    [[ -f "${CODE_ROOT}/train_deco/train.py" ]] || fail \
        "代码必须放在 /workspace 下，且 CODE_ROOT 必须指向仓库根目录"
}

create_roots() {
    mkdir -p \
        "${VENV_PATH%/*}" "${DATA_ROOT}" "${STAGE1_ROOT}" "${ENCODER_ROOT}" \
        "${TACTILE_CACHE}" "${MANIFEST_ROOT}" "${OUTPUT_ROOT}" "${STATE_ROOT}" \
        "${LOG_ROOT}" "${HF_HUB_CACHE}" "${HF_DATASETS_CACHE}" \
        "${WANDB_CACHE_DIR}" "${WANDB_CONFIG_DIR}" "${WANDB_DATA_DIR}"
}

require_runtime() {
    [[ -x "${PYTHON_BIN}" && -x "${HF_BIN}" ]] || fail "请先运行 setup"
}

wandb_enabled() {
    case "${WANDB_ENABLED:-1}" in
        1|true|TRUE|yes|YES|on|ON) return 0 ;;
        0|false|FALSE|no|NO|off|OFF) return 1 ;;
        *) fail "WANDB_ENABLED 必须是布尔值，当前为：${WANDB_ENABLED}" ;;
    esac
}

require_hf_token() {
    [[ -n "${HF_TOKEN:-}" ]] || fail "未设置 HF_TOKEN：${SERVER_ENV_FILE}"
}

require_training_tokens() {
    require_hf_token
    if wandb_enabled && [[ "${WANDB_MODE:-online}" == "online" ]]; then
        [[ -n "${WANDB_API_KEY:-}" ]] || fail \
            "W&B online 模式未设置 WANDB_API_KEY；不用 W&B 时请设置 WANDB_ENABLED=0"
    fi
}

detect_workers() {
    local workers=$(( $(nproc) * 3 / 4 ))
    ((workers > 0)) || workers=1
    printf '%s\n' "${workers}"
}

setup_environment() {
    create_roots
    command -v python3 >/dev/null || fail "找不到 python3"
    command -v nvidia-smi >/dev/null || fail "找不到 nvidia-smi"
    VENV_PATH="${VENV_PATH}" \
    PYPI_INDEX="${PYPI_INDEX:-https://pypi.org/simple}" \
    PYTORCH_INDEX="${PYTORCH_INDEX:-https://download.pytorch.org/whl/cu128}" \
        bash "${CODE_ROOT}/train_deco/setup_environment.sh" \
        --python "${PYTHON_COMMAND:-python3}" --venv "${VENV_PATH}"
    "${HF_BIN}" version
    "${VENV_PATH}/bin/wandb" --version
}

check_disk() {
    local free_kb free_gb minimum_gb="${MIN_FREE_GB:-90}"
    free_kb="$(df -Pk /workspace | awk 'NR == 2 {print $4}')"
    free_gb=$((free_kb / 1024 / 1024))
    log "/workspace 可用空间：${free_gb} GiB"
    if ((free_gb < minimum_gb)) && [[ "${ALLOW_LOW_DISK:-0}" != "1" ]]; then
        fail "建议至少 ${minimum_gb} GiB；确认已有缓存后可设置 ALLOW_LOW_DISK=1"
    fi
}

download_dataset() {
    local name="$1" revision="$2"
    "${HF_BIN}" download "KaiyueChen/${name}" --type dataset \
        --revision "${revision}" --local-dir "${DATA_ROOT}/${name}" \
        --max-workers "${HF_DOWNLOAD_WORKERS:-8}"
}

download_assets() {
    require_runtime
    require_hf_token
    create_roots
    check_disk
    "${HF_BIN}" auth whoami
    download_dataset insert_01 "${INSERT_01_REVISION}"
    download_dataset insert_02 "${INSERT_02_REVISION}"
    download_dataset bread_01 "${BREAD_01_REVISION}"
    download_dataset bread_02 "${BREAD_02_REVISION}"
    download_dataset bread_03 "${BREAD_03_REVISION}"

    "${HF_BIN}" download wjstx/deco_0829 \
        bread/config.json bread/dataset_stats.json bread/deco_stage1_latest.pt \
        insert/config.json insert/dataset_stats.json insert/deco_stage1_latest.pt \
        --revision "${DECO_REVISION}" --local-dir "${STAGE1_ROOT}" \
        --max-workers "${HF_DOWNLOAD_WORKERS:-8}"
    "${HF_BIN}" download KaiyueChen/encoder_ckpt_0824 \
        --revision "${ENCODER_REVISION}" --local-dir "${ENCODER_ROOT}" \
        --max-workers "${HF_DOWNLOAD_WORKERS:-8}"
}

prepare_manifests() {
    require_runtime
    create_roots
    local prepare="${CODE_ROOT}/train_deco/scripts/prepare_data.sh"
    PYTHON_BIN="${PYTHON_BIN}" bash "${prepare}" --mode server \
        --root "${DATA_ROOT}/insert_01" --root "${DATA_ROOT}/insert_02" \
        --output "${INSERT_MANIFEST}" --dataset-id insert_01_02 \
        --state-action-profile single-right-arm-7x10 --require-black-camera0
    PYTHON_BIN="${PYTHON_BIN}" bash "${prepare}" --mode server \
        --root "${DATA_ROOT}/bread_01" --root "${DATA_ROOT}/bread_02" \
        --root "${DATA_ROOT}/bread_03" \
        --output "${BREAD_MANIFEST}" --dataset-id bread_01_02_03 \
        --state-action-profile dual-arm-20x20
}

set_task() {
    local task="$1" timestamp="$2"
    case "${task}" in
        insert)
            TASK_MANIFEST="${INSERT_MANIFEST}"
            TASK_STAGE1="${INSERT_STAGE1}"
            TASK_RUN_ID="${INSERT_RUN_ID:-insert-stage2-rtxpro6000-${timestamp}}"
            TASK_REPO="${HF_OUTPUT_INSERT_REPO:-}"
            ;;
        bread)
            TASK_MANIFEST="${BREAD_MANIFEST}"
            TASK_STAGE1="${BREAD_STAGE1}"
            TASK_RUN_ID="${BREAD_RUN_ID:-bread-stage2-rtxpro6000-${timestamp}}"
            TASK_REPO="${HF_OUTPUT_BREAD_REPO:-}"
            ;;
        *) fail "task 只能是 insert 或 bread" ;;
    esac
}

train_task() {
    local task="$1" timestamp workers batch_size run_dir
    timestamp="$(date +%Y%m%d_%H%M%S)"
    set_task "${task}" "${timestamp}"
    require_runtime
    require_training_tokens
    create_roots
    [[ -f "${TASK_MANIFEST}" ]] || fail "缺少 manifest：${TASK_MANIFEST}"
    [[ -f "${TASK_STAGE1}" ]] || fail "缺少 Stage1：${TASK_STAGE1}"
    [[ -f "${ENCODER_ROOT}/checkpoint.json" ]] || fail "缺少 encoder：${ENCODER_ROOT}"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

    workers="${WORKERS:-$(detect_workers)}"
    batch_size="${BATCH_SIZE:-512}"
    run_dir="${OUTPUT_ROOT}/${TASK_RUN_ID}"
    log "task=${task} run=${TASK_RUN_ID} batch=${batch_size} workers=${workers}"
    warn "BATCH_SIZE 是单卡物理 batch；如 CUDA OOM，请显式降低 BATCH_SIZE"
    printf '%s\n' "${TASK_RUN_ID}" > "${STATE_ROOT}/last_${task}_run_id"

    export WANDB_ENABLED="${WANDB_ENABLED:-1}"
    export WANDB_PROJECT="${WANDB_PROJECT:-deco-stage2}"
    export WANDB_GROUP="${WANDB_GROUP:-deco-stage2-rtxpro6000}"
    export WANDB_TAGS="${WANDB_TAGS:-stage2,rtxpro6000,${task}}"
    export WANDB_MODE="${WANDB_MODE:-online}"
    export WANDB_RUN_ID="${TASK_RUN_ID}"
    export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
    export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
    export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
    export DATALOADER_PREFETCH_FACTOR="${DATALOADER_PREFETCH_FACTOR:-1}"

    OUTPUT_DIR="${OUTPUT_ROOT}" RUN_ID="${TASK_RUN_ID}" \
    BATCH_SIZE="${batch_size}" WORKERS="${workers}" EPOCHS="${EPOCHS:-50}" \
    SAVE_EVERY="${SAVE_EVERY:-10}" LOG_EVERY_STEPS="${LOG_EVERY_STEPS:-10}" \
    TACTILE_ENCODER_CACHE="${TACTILE_CACHE}" VENV_PATH="${VENV_PATH}" \
    PYTHON_BIN="${PYTHON_BIN}" \
        bash "${CODE_ROOT}/train_deco/scripts/train.sh" \
        --mode local-stage2 --manifest "${TASK_MANIFEST}" \
        --run-id "${TASK_RUN_ID}" --stage1-checkpoint "${TASK_STAGE1}" \
        --tactile-encoder-checkpoint "${ENCODER_ROOT}" \
        2>&1 | tee "${LOG_ROOT}/${TASK_RUN_ID}.log"

    [[ -f "${run_dir}/deco_stage2_best.pt" ]] || fail "缺少 best checkpoint"
    [[ -f "${run_dir}/deco_stage2_best.ts" ]] || fail "缺少 best TorchScript"
}

last_run_id() {
    local task="$1" explicit="${2:-}" marker value
    [[ -n "${explicit}" ]] && { printf '%s\n' "${explicit}"; return; }
    marker="${STATE_ROOT}/last_${task}_run_id"
    [[ -f "${marker}" ]] || fail "未找到最近 RUN_ID，请显式传入"
    value="$(<"${marker}")"
    [[ -n "${value}" ]] || fail "RUN_ID 记录为空"
    printf '%s\n' "${value}"
}

upload_task() {
    local task="$1" run_id run_dir visibility
    run_id="$(last_run_id "${task}" "${2:-}")"
    set_task "${task}" unused
    require_runtime
    require_hf_token
    [[ -n "${TASK_REPO}" ]] || fail "请设置 HF_OUTPUT_${task^^}_REPO=owner/repo"
    run_dir="${OUTPUT_ROOT}/${run_id}"
    [[ -f "${run_dir}/deco_stage2_best.pt" ]] || fail "缺少训练产物：${run_dir}"
    visibility="--private"
    [[ "${HF_REPO_PRIVATE:-1}" == "1" ]] || visibility="--public"
    "${HF_BIN}" repos create "${TASK_REPO}" --type model \
        "${visibility}" --exist-ok
    "${HF_BIN}" upload "${TASK_REPO}" "${run_dir}" . --type model \
        --exclude "deco_stage2_epoch_*" --exclude "wandb/**" \
        --commit-message "Upload DECO Stage2 ${task} run ${run_id}"
    log "上传完成：https://huggingface.co/${TASK_REPO}"
}

doctor() {
    require_runtime
    create_roots
    log "code=${CODE_ROOT}"
    log "python=$("${PYTHON_BIN}" --version 2>&1)"
    log "cpu=$(nproc) workers=${WORKERS:-$(detect_workers)} batch=${BATCH_SIZE:-512} prefetch=${DATALOADER_PREFETCH_FACTOR:-1}"
    df -h /workspace
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
    "${PYTHON_BIN}" -c \
        'import torch, wandb, huggingface_hub; print(torch.__version__, torch.version.cuda, wandb.__version__, huggingface_hub.__version__)'
    [[ -n "${HF_TOKEN:-}" ]] && log "HF_TOKEN=已设置" || warn "HF_TOKEN=未设置"
    if wandb_enabled; then
        [[ -n "${WANDB_API_KEY:-}" ]] && log "WANDB_API_KEY=已设置" || warn "WANDB_API_KEY=未设置"
    else
        log "W&B=已禁用"
    fi
}

main() {
    local command="${1:-}"
    case "${command}" in
        -h|--help|help) usage; return ;;
        "") usage >&2; return 2 ;;
    esac
    validate_paths
    case "${command}" in
        setup) setup_environment ;;
        download) download_assets ;;
        prepare) prepare_manifests ;;
        doctor) doctor ;;
        train) [[ $# -eq 2 ]] || fail "train 需要 task"; train_task "$2" ;;
        upload) [[ $# -eq 2 || $# -eq 3 ]] || fail "upload 参数错误"; upload_task "$2" "${3:-}" ;;
        run) [[ $# -eq 2 ]] || fail "run 需要 task"; train_task "$2"; upload_task "$2" ;;
        all)
            [[ $# -eq 1 ]] || fail "all 不接受额外参数"
            setup_environment
            download_assets
            prepare_manifests
            train_task insert
            upload_task insert
            unset WANDB_TAGS WANDB_RUN_ID
            train_task bread
            upload_task bread
            ;;
        *) fail "未知命令：${command}" ;;
    esac
}
main "$@"
