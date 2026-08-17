#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CONFIG="${PI05_FRS_DEPLOY_CONFIG:-${ROOT}/deploy_pi05_frs/configs/deploy_pi05_frs.yaml}"
exec bash "${ROOT}/deploy_pi05_frs/scripts/start_remote_client.sh" \
    --config "${CONFIG}" "$@"
