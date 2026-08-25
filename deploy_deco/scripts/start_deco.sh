#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
PACKAGE_ROOT="${PACKAGE_ROOT:-${PROJECT_ROOT}/deploy_deco}"
VENV_PATH="${VENV_PATH:-${PACKAGE_ROOT}/.venv}"
PYTHON_BIN="${PYTHON_BIN:-${VENV_PATH}/bin/python}"
CONFIG="${CONFIG:-${PACKAGE_ROOT}/configs/deploy_deco.yaml}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "找不到 deploy_deco Python：${PYTHON_BIN}" >&2
  echo "请在 deploy_deco 中创建环境，或设置 PYTHON_BIN。" >&2
  exit 1
fi

cd "${PROJECT_ROOT}"
exec "${PYTHON_BIN}" -m deploy_deco.remote_client --config "${CONFIG}" "$@"
