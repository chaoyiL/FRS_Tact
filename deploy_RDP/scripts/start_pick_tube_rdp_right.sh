#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
RDP_DIR=$(cd -- "${SCRIPT_DIR}/.." && pwd)
export RDP_DEPLOY_CONFIG="${RDP_DEPLOY_CONFIG:-${RDP_DIR}/configs/deploy_pick_tube_rdp_right.yaml}"

exec bash "${SCRIPT_DIR}/start_pick_tube_rdp_client.sh" "$@"
