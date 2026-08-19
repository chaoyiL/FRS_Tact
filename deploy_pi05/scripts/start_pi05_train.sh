#!/usr/bin/env bash
#
# Launch a pi0.5 fine-tune. Training configs live in Python, not YAML:
# src/lerobot/policies/pi05_jax/training/config.py's _CONFIGS (openpi's TrainConfig + tyro), so
# this script takes a config *name* and an experiment name rather than a config path.
#
#   scripts/start_pi05_train.sh                            # pi05_pick_tube, auto-named run
#   scripts/start_pi05_train.sh pi05_pick_tube my_run
#   scripts/start_pi05_train.sh pi05_pick_tube my_run --resume
#   PI05_FOREGROUND=1 scripts/start_pi05_train.sh debug smoke --overwrite
#
# Norm stats are computed on first use (openpi requires them before training; see
# tools/compute_pi05_norm_stats.py). Set PI05_SKIP_NORM_STATS=1 to skip that check.

set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
CONFIG_NAME="${1:-pi05_pick_tube}"
EXP_NAME="${2:-${CONFIG_NAME}_$(date +%Y%m%d_%H%M%S)}"
EXTRA_ARGS=("${@:3}")
ENV_FILE="${PROJECT_ROOT}/.env.frs"
TMUX_SESSION="${PI05_TMUX_SESSION:-pi05_${CONFIG_NAME}_train}"

if [[ -f "${ENV_FILE}" ]]; then
    # shellcheck disable=SC1090
    source "${ENV_FILE}"
fi
if command -v uv >/dev/null 2>&1; then
    UV_BIN="$(command -v uv)"
elif [[ -x "${HOME}/.local/bin/uv" ]]; then
    UV_BIN="${HOME}/.local/bin/uv"
else
    echo "找不到 uv；请先运行 scripts/setup_env.sh" >&2
    exit 1
fi

if [[ "${PI05_FOREGROUND:-0}" != "1" && -z "${TMUX:-}" ]] && command -v tmux >/dev/null 2>&1; then
    tmux has-session -t "${TMUX_SESSION}" 2>/dev/null && {
        echo "tmux session 已存在：${TMUX_SESSION}" >&2
        exit 1
    }
    printf -v inner 'PI05_FOREGROUND=1 bash %q %q %q' "$0" "${CONFIG_NAME}" "${EXP_NAME}"
    for arg in "${EXTRA_ARGS[@]}"; do
        printf -v quoted_arg ' %q' "${arg}"
        inner+="${quoted_arg}"
    done
    tmux new-session -d -s "${TMUX_SESSION}" -c "${PROJECT_ROOT}" "${inner}"
    echo "pi0.5 训练已在 tmux 后台启动：${TMUX_SESSION}（config=${CONFIG_NAME} exp=${EXP_NAME}）"
    echo "查看：tmux attach -t ${TMUX_SESSION}"
    exit 0
fi

cd "${PROJECT_ROOT}"

if [[ "${PI05_SKIP_NORM_STATS:-0}" != "1" ]]; then
    # Cheap compared to training, and openpi's data pipeline hard-fails without it -- better to
    # find out here than after the model has been built and the first batch requested.
    needs_stats=$("${UV_BIN}" run --no-sync python - "${CONFIG_NAME}" <<'PY'
import sys
from lerobot.policies.pi05_jax.training import config as _config

config = _config.get_config(sys.argv[1])
data_config = config.data.create(config.assets_dirs, config.model)
print("0" if data_config.repo_id == "fake" or data_config.norm_stats is not None else "1")
PY
)
    if [[ "${needs_stats}" == "1" ]]; then
        echo "norm stats 缺失，先运行 tools/compute_pi05_norm_stats.py --config-name=${CONFIG_NAME}"
        "${UV_BIN}" run --no-sync python tools/compute_pi05_norm_stats.py --config-name="${CONFIG_NAME}"
    fi
fi

exec "${UV_BIN}" run --no-sync python tools/train_pi05_jax.py \
    "${CONFIG_NAME}" --exp_name="${EXP_NAME}" "${EXTRA_ARGS[@]}"
