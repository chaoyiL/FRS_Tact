#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DEPLOY_ROOT="${ROOT}/deploy_pi05"
CONFIG="${PI05_DEPLOY_CONFIG:-${DEPLOY_ROOT}/configs/deploy_pi05.yaml}"
exec bash "${DEPLOY_ROOT}/scripts/start_remote_client.sh" \
    "$@" --mode pi05 --config "${CONFIG}"
