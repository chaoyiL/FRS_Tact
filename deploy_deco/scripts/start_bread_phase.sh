#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-${PROJECT_ROOT}/deploy_deco/.venv/bin/python}"
CONFIG="${CONFIG:-${PROJECT_ROOT}/deploy_deco/configs/deploy_deco_bread_phase.yaml}"

cd "${PROJECT_ROOT}"
exec "${PYTHON_BIN}" -m deploy_deco.bread_phase_client --config "${CONFIG}" "$@"
