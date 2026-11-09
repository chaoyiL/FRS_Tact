#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
PROJECT_DIR="$(cd -- "$SCRIPT_DIR/.." && pwd -P)"
REPO_ROOT="$(cd -- "$PROJECT_DIR/.." && pwd -P)"

if ! command -v uv >/dev/null 2>&1; then
  echo "uv is required; install uv before running this script." >&2
  exit 127
fi

if [[ -n "${VIRTUAL_ENV:-}" ]]; then
  active_venv="$(readlink -f -- "$VIRTUAL_ENV")"
  case "$active_venv" in
    "$REPO_ROOT/.venv"|"$REPO_ROOT/deploy_pi05/.venv"|"$REPO_ROOT/train_pi05_frs/.venv")
      echo "refusing root, deploy, or train_pi05_frs virtual environment: $active_venv" >&2
      exit 2
      ;;
  esac
fi

VIRTUAL_ENV= UV_PROJECT_ENVIRONMENT="$PROJECT_DIR/.venv" \
  uv sync --frozen --project "$PROJECT_DIR" --python 3.12
