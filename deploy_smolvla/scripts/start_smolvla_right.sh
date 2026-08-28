#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export SMOLVLA_VISION_CONFIG="${SMOLVLA_VISION_CONFIG:-${ROOT}/deploy_smolvla/configs/deploy_smolvla_pytorch_right.yaml}"
export SMOLVLA_FRS_CONFIG="${SMOLVLA_FRS_CONFIG:-${ROOT}/deploy_smolvla/configs/deploy_frs_right.yaml}"

exec bash "${ROOT}/deploy_smolvla/scripts/start_smolvla.sh" "$@"
