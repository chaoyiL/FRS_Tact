#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
TRAIN_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
PROJECT_ROOT="$(cd -- "${TRAIN_ROOT}/.." && pwd)"
CONFIG="${TRAIN_ROOT}/configs/test_pi05_4090_insert01.yaml"
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
MAX_NORM_FRAMES="${PI05_SMOKE_NORM_FRAMES:-512}"
RUN_SETUP=0
SKIP_DOWNLOAD=0

usage() {
    cat <<'EOF'
用法：bash train_pi05/scripts/test_pi05_4090_insert01.sh [--setup] [--skip-download]

--setup          新服务器首次运行：安装 Python 3.12 数据环境和 Python 3.11 训练环境。
--skip-download  已有 /workspace/lerobot_v30/KaiyueChen/insert_01 时跳过下载/转换。

可选环境变量：
  PI05_SMOKE_NORM_FRAMES  计算测试归一化统计时抽样的最大帧数，默认 512。
  CUDA_VISIBLE_DEVICES    要测试的 GPU，默认 0，并且只允许暴露一张卡。
EOF
}

while (($#)); do
    case "$1" in
        --setup) RUN_SETUP=1 ;;
        --skip-download) SKIP_DOWNLOAD=1 ;;
        -h|--help) usage; exit 0 ;;
        *) printf '[pi05-smoke] 未知参数：%s\n' "$1" >&2; usage >&2; exit 2 ;;
    esac
    shift
done

log() { printf '[pi05-smoke] %s\n' "$*"; }
fail() { printf '[pi05-smoke] 错误：%s\n' "$*" >&2; exit 1; }

cd "${PROJECT_ROOT}"

if ((RUN_SETUP)); then
    log "配置两套隔离环境：Python 3.12 数据工具 + Python 3.11 Pi0.5 JAX 训练"
    bash scripts/setup_env.sh --pi05_train
fi

[[ -f "${ENV_FILE}" ]] || fail "缺少 ${ENV_FILE}；新服务器请加 --setup"
# shellcheck disable=SC1090
source "${ENV_FILE}"

[[ -x "${DATA_TOOL_PYTHON:-}" ]] || fail "DATA_TOOL_PYTHON 不可执行；请重新运行 --setup"
[[ -x "${TRAIN_PI05_PYTHON:-}" ]] || fail "TRAIN_PI05_PYTHON 不可执行；请重新运行 --setup"
[[ -f "${CONFIG}" ]] || fail "缺少测试配置：${CONFIG}"
command -v nvidia-smi >/dev/null 2>&1 || fail "未找到 nvidia-smi，无法进行 GPU 训练链路测试"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
[[ "${CUDA_VISIBLE_DEVICES}" != *,* ]] || fail "本测试只允许暴露一张 GPU，当前为 ${CUDA_VISIBLE_DEVICES}"
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.90}"
export PYTHONPATH="${TRAIN_ROOT}/src:${TRAIN_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONSAFEPATH=1

log "系统 GPU"
nvidia-smi --query-gpu=index,name,driver_version,memory.total --format=csv,noheader

log "验证两个解释器及 JAX GPU"
"${DATA_TOOL_PYTHON}" -c 'import platform; assert platform.python_version_tuple()[:2] == ("3", "12"); print("data python=" + platform.python_version())'
"${TRAIN_PI05_PYTHON}" - <<'PY'
import platform
import jax

devices = jax.devices()
if platform.python_version_tuple()[:2] != ("3", "11"):
    raise RuntimeError(f"Pi0.5 training requires Python 3.11, got {platform.python_version()}")
if len(devices) != 1 or devices[0].platform != "gpu":
    raise RuntimeError(f"Expected exactly one JAX GPU, got: {devices}")
print(f"train python={platform.python_version()}; jax={jax.__version__}; devices={devices}")
PY

if ((!SKIP_DOWNLOAD)); then
    log "下载并转换 KaiyueChen/insert_01"
    DATA_TOOL_PYTHON="${DATA_TOOL_PYTHON}" \
        bash scripts/download_data.sh --dataset insert_01
fi

DATASET_ROOT="/workspace/lerobot_v30/KaiyueChen/insert_01"
[[ -f "${DATASET_ROOT}/meta/info.json" ]] || fail "缺少转换后的 insert_01：${DATASET_ROOT}"

log "检查右手单臂数据合同：7D state、10D action、camera1"
bash train_pi05/scripts/start_pi05_train.sh --check "${CONFIG}"

log "抽样 ${MAX_NORM_FRAMES} 帧计算测试专用归一化统计"
"${TRAIN_PI05_PYTHON}" train_pi05/tools/compute_norm_stats.py \
    --config-name "${CONFIG}" --max-frames "${MAX_NORM_FRAMES}"

NORM_STATS="${TRAIN_ROOT}/assets/insert_01_4090_smoke/norm_stats.json"
[[ -s "${NORM_STATS}" ]] || fail "归一化统计没有生成：${NORM_STATS}"

log "开始 2 步真实 JAX 训练；首次运行需要下载 pi05_base 并完成 XLA 编译"
TRAIN_LOG="/workspace/outputs/pi05_4090_insert01_smoke.log"
mkdir -p "$(dirname -- "${TRAIN_LOG}")"
PI05_FOREGROUND=1 bash train_pi05/scripts/start_pi05_train.sh "${CONFIG}" 2>&1 | tee "${TRAIN_LOG}"

grep -Eq 'Step [0-9]+: .*loss=' "${TRAIN_LOG}" || fail "训练日志中没有找到 loss，真实更新步骤可能未执行"
if grep -Eiq 'loss=(nan|[+-]?inf)' "${TRAIN_LOG}"; then
    fail "训练产生非有限 loss；请检查数据和归一化统计：${TRAIN_LOG}"
fi

OUTPUT_ROOT="/workspace/outputs/pi05_4090_insert01_smoke"
CHECKPOINT_PARAMS="$(find "${OUTPUT_ROOT}" -mindepth 2 -maxdepth 2 -type d -name params -print -quit 2>/dev/null || true)"
[[ -n "${CHECKPOINT_PARAMS}" ]] || fail "训练结束但未找到 checkpoint params：${OUTPUT_ROOT}"

log "完整链路验证通过"
log "归一化统计：${NORM_STATS}"
log "训练日志：${TRAIN_LOG}"
log "Checkpoint：${CHECKPOINT_PARAMS}"
