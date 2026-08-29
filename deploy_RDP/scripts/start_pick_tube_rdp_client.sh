#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
RDP_DIR=$(cd -- "${SCRIPT_DIR}/.." && pwd)
PYTHON_BIN=${PYTHON_BIN:-${RDP_DIR}/.venv/bin/python}
CONFIG=${RDP_DEPLOY_CONFIG:-${RDP_DIR}/configs/deploy_pick_tube_rdp.yaml}

cd "${RDP_DIR}"
exec "${PYTHON_BIN}" deploy_pick_tube_rdp.py --config "${CONFIG}" "$@"
