#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
PROJECT_DIR="$(cd -- "$SCRIPT_DIR/.." && pwd -P)"

# Right-arm task3 by default. An explicit --config overrides this default.
exec bash "$SCRIPT_DIR/start_baseline_pi05.sh" \
  --config "$PROJECT_DIR/configs/deploy_baseline_pi05_task3.yaml" "$@"
