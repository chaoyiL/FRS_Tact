#!/usr/bin/env bash
set -euo pipefail

# Stage 2 delegates the base DECO environment setup, then adds only the
# conversion-time CPU dependencies.  JAX is never imported by the trainer.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_PATH="${VENV_PATH:-${SCRIPT_DIR}/.venv}"
ARGS=("$@")
for ((index = 0; index < ${#ARGS[@]}; index++)); do
  if [[ "${ARGS[index]}" == "--venv" ]]; then
    VENV_PATH="${ARGS[index + 1]:?--venv requires a path}"
  fi
done

bash "${SCRIPT_DIR}/scripts/setup_env.sh" "${ARGS[@]}"
"${VENV_PATH}/bin/python" -m pip install \
  'safetensors>=0.5,<1' \
  'jax[cpu]>=0.4.30,<0.6' \
  'flax>=0.10,<0.12'
