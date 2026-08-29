#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
TRAIN_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
TRAIN_VENV="${TRAIN_PI05_VENV:-${TRAIN_ROOT}/.venv}"
UV_BIN="${UV_BIN:-uv}"

case "${1:-}" in
    --check)
        printf 'project: %s\nenvironment: %s\npython: 3.11\n' \
            "${TRAIN_ROOT}" "${TRAIN_VENV}"
        ;;
    "")
        UV_PROJECT_ENVIRONMENT="${TRAIN_VENV}" "${UV_BIN}" sync \
            --python 3.11 --project "${TRAIN_ROOT}"
        ;;
    *)
        printf 'usage: %s [--check]\n' "$0" >&2
        exit 2
        ;;
esac

