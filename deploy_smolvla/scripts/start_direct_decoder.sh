#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CONFIG="${DIRECT_DECODER_CONFIG:-${ROOT}/deploy_smolvla/configs/deploy_direct_decoder.yaml}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTHONUNBUFFERED=1
exec bash "${ROOT}/deploy_smolvla/scripts/start_remote_client.sh" \
    --config "${CONFIG}" "$@"
