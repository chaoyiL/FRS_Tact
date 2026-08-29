#!/usr/bin/env bash
# 一键启动纯视觉 Pi0.5 右手单臂训练。
# 实际的环境检查、数据预检和 tmux 启动由通用入口统一完成。
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
TRAIN_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
RIGHT_CONFIG="${TRAIN_ROOT}/configs/train_pi05_right.yaml"

export PI05_TMUX_SESSION="${PI05_RIGHT_TMUX_SESSION:-pi05_right_train}"
exec bash "${SCRIPT_DIR}/start_pi05_train.sh" "$@" "${RIGHT_CONFIG}"
