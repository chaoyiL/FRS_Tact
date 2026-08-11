#!/usr/bin/env bash

set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
CONFIG_PATH="${PROJECT_ROOT}/configs/train_smolvla_jax_colleague_matched.yaml"

export FRS_TMUX_SESSION="${FRS_TMUX_SESSION:-smolvla_colleague_matched}"

exec bash "${SCRIPT_DIR}/start_vtsmolvla_train.sh" --config "${CONFIG_PATH}" "$@"
