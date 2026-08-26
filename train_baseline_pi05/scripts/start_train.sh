#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
PROJECT_DIR="$(cd -- "$SCRIPT_DIR/.." && pwd -P)"
REPO_ROOT="$(cd -- "$PROJECT_DIR/.." && pwd -P)"
PYTHON="$PROJECT_DIR/.venv/bin/python"

if [[ ! -x "$PYTHON" ]]; then
  echo "standalone environment is missing: run $PROJECT_DIR/scripts/setup_env.sh" >&2
  exit 1
fi

export PYTHONSAFEPATH=1
export PYTHONPATH="$PROJECT_DIR/src:$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false

exec "$PYTHON" -m train_baseline_pi05.pipeline "$@"
