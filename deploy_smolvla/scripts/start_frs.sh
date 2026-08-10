#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CONFIG="${FRS_DEPLOY_CONFIG:-${ROOT}/deploy_smolvla/configs/deploy_frs.yaml}"
exec bash "${ROOT}/deploy_smolvla/scripts/start_remote_client.sh" \
    --config "${CONFIG}" "$@"
