#!/usr/bin/env bash
set -Eeuo pipefail

# Task4 press: right-hand Pi0.5 FRS v3 learned-residual Gate training.
# Prepare press-specific tactile/action caches, then train in a separate output directory.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
TRAIN_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
CONFIG_PATH="${TRAIN_ROOT}/configs/train_pi05_frs_right_press_v3.yaml"

export FRS_TMUX_SESSION="${FRS_TMUX_SESSION:-frs_pi05_press_v3_train}"
exec bash "${SCRIPT_DIR}/start_frs_pi05_train.sh" "$@" "${CONFIG_PATH}"
