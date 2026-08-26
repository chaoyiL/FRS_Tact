#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
PROJECT_DIR="$(cd -- "$SCRIPT_DIR/.." && pwd -P)"
REPO_ROOT="$(cd -- "$PROJECT_DIR/.." && pwd -P)"
PYTHON="$PROJECT_DIR/.venv/bin/python"
CONFIG="$PROJECT_DIR/configs/deploy_baseline_pi05.yaml"
CHECK_ONLY=false
MAX_ITERATIONS=""

usage() {
  cat <<'EOF'
Usage: bash deploy_baseline_pi05/scripts/start_baseline_pi05.sh [--config PATH] [--check] [--max-iterations N]

Robot authentication may be supplied with VB_ROBOT_TOKEN or a VB3_TOKEN_FILE.
The --check path validates the deployment config and local assets without requiring either token.
EOF
}

while (( $# > 0 )); do
  case "$1" in
    --config)
      (( $# >= 2 )) || { echo "--config requires a path" >&2; exit 2; }
      CONFIG="$2"
      shift 2
      ;;
    --check)
      CHECK_ONLY=true
      shift
      ;;
    --max-iterations)
      (( $# >= 2 )) || { echo "--max-iterations requires a value" >&2; exit 2; }
      MAX_ITERATIONS="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ -n "$MAX_ITERATIONS" && ! "$MAX_ITERATIONS" =~ ^[0-9]+$ ]]; then
  echo "--max-iterations must be a non-negative integer" >&2
  exit 2
fi
if [[ ! -x "$PYTHON" ]]; then
  echo "standalone environment is missing: $PYTHON" >&2
  echo "run: uv sync --project $PROJECT_DIR" >&2
  exit 1
fi
if [[ ! -r "$CONFIG" ]]; then
  echo "deployment config is not readable: $CONFIG" >&2
  exit 2
fi

CONFIG="$(cd -- "$(dirname -- "$CONFIG")" && pwd -P)/$(basename -- "$CONFIG")"

if [[ "$CHECK_ONLY" != true && -z "${VB_ROBOT_TOKEN:-}" && -n "${VB3_TOKEN_FILE:-}" ]]; then
  if [[ ! -r "$VB3_TOKEN_FILE" ]]; then
    echo "token file is not readable: $VB3_TOKEN_FILE" >&2
    exit 2
  fi
  while IFS= read -r candidate || [[ -n "$candidate" ]]; do
    if [[ -n "$candidate" ]]; then
      export VB_ROBOT_TOKEN="$candidate"
      break
    fi
  done < "$VB3_TOKEN_FILE"
  if [[ -z "${VB_ROBOT_TOKEN:-}" ]]; then
    echo "token file is empty: $VB3_TOKEN_FILE" >&2
    exit 2
  fi
fi

export PYTHONSAFEPATH=1
export PYTHONUNBUFFERED=1
export PYTHONPATH="$PROJECT_DIR/src:$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false

PYTHON_ARGS=(-m deploy_baseline_pi05.remote_client --config "$CONFIG")
if [[ "$CHECK_ONLY" == true ]]; then
  PYTHON_ARGS+=(--check)
fi
if [[ -n "$MAX_ITERATIONS" ]]; then
  PYTHON_ARGS+=(--max-iterations "$MAX_ITERATIONS")
fi

cd "$REPO_ROOT"
exec "$PYTHON" "${PYTHON_ARGS[@]}"
