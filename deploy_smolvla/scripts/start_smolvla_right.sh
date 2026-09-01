#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CONFIG="${ROOT}/deploy_smolvla/configs/deploy_smolvla_pytorch_right.yaml"

exec bash "${ROOT}/deploy_smolvla/scripts/start_remote_client.sh" \
    "$@" --config "${CONFIG}"
