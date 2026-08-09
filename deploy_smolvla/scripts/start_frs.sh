#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
exec "${ROOT}/deploy_smolvla/scripts/start_vtsmolvla.sh" \
    --config "${ROOT}/deploy_smolvla/configs/deploy_frs.yaml" "$@"
