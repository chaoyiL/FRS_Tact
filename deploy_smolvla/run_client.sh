#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -n "${FRS_PYTHON:-}" ]]; then
    PYTHON_BIN="${FRS_PYTHON}"
elif [[ -n "${VB3_PYTHON:-}" ]]; then
    PYTHON_BIN="${VB3_PYTHON}"
elif [[ -x "${ROOT}/.venv/bin/python" ]]; then
    PYTHON_BIN="${ROOT}/.venv/bin/python"
else
    PYTHON_BIN="python3"
fi
args=()
if (( $# > 0 )); then
    args=(--config "$1")
fi
export PYTHONUNBUFFERED=1
export PYTHONPATH="${ROOT}/src:${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
exec "${PYTHON_BIN}" -m deploy_smolvla.remote_client "${args[@]}"
