#!/usr/bin/env bash

set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
CONFIG_PATH="${PROJECT_ROOT}/configs/train_vtsmolvla_jax_visual_k0.yaml"
export FRS_TMUX_SESSION="${FRS_TMUX_SESSION:-smolvla_k0}"
exec bash "${SCRIPT_DIR}/start_vtsmolvla_train.sh" --config "${CONFIG_PATH}" "$@"
