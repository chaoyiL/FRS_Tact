#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
PROJECT_DIR="$(cd -- "$SCRIPT_DIR/.." && pwd -P)"

# Bimanual weights and cameras are configured in the bimanual YAML.
exec bash "$SCRIPT_DIR/start_baseline_pi05.sh" \
  --config "$PROJECT_DIR/configs/deploy_baseline_pi05_bimanual.yaml" "$@"
