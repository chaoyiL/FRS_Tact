#!/usr/bin/env bash
set -Eeuo pipefail

# Pi0.5 FRS 右手单臂训练（7D state / 10D action / 两路右手触觉）：
#   bash train_pi05_frs/scripts/start_frs_pi05_right_train.sh
# 只检查配置、数据、checkpoint 和统计量，不开始训练：
#   bash train_pi05_frs/scripts/start_frs_pi05_right_train.sh --check

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
TRAIN_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
CONFIG_PATH="${TRAIN_ROOT}/configs/train_pi05_frs_right.yaml"

export FRS_TMUX_SESSION="${FRS_TMUX_SESSION:-frs_pi05_right_train}"
exec bash "${SCRIPT_DIR}/start_frs_pi05_train.sh" "$@" "${CONFIG_PATH}"
