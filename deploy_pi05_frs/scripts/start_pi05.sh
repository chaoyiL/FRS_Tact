#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CONFIG="${PI05_DEPLOY_CONFIG:-${ROOT}/deploy_pi05_frs/configs/deploy_pi05.yaml}"
exec bash "${ROOT}/deploy_pi05_frs/scripts/start_remote_client.sh" \
    --mode pi05 --config "${CONFIG}" "$@"
