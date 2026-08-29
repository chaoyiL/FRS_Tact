#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export PI05_DEPLOY_CONFIG="${PI05_DEPLOY_CONFIG:-${ROOT}/deploy_pi05/configs/deploy_pi05_right.yaml}"
exec bash "${ROOT}/deploy_pi05/scripts/start_pi05.sh" "$@"
