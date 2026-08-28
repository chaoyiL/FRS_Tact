#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PACKAGE_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
export CONFIG="${CONFIG:-${PACKAGE_ROOT}/configs/deploy_deco_right.yaml}"
exec bash "${SCRIPT_DIR}/start_deco.sh" "$@"
