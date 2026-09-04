#!/usr/bin/env bash
set -Eeuo pipefail

# Pi0.5 FRS v3 learned-residual Gate，右手单臂训练。
# 复用现有 action cache 与 tactile embedding cache，只新建训练输出目录。

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
TRAIN_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
CONFIG_PATH="${TRAIN_ROOT}/configs/train_pi05_frs_right_v3.yaml"

export FRS_TMUX_SESSION="${FRS_TMUX_SESSION:-frs_pi05_right_v3_train}"
exec bash "${SCRIPT_DIR}/start_frs_pi05_train.sh" "$@" "${CONFIG_PATH}"
