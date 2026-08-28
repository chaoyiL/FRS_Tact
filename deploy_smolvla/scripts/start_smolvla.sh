#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
MODE="vision"
CONFIG=""
PASSTHROUGH=()

usage() {
    cat <<'EOF'
Usage: bash deploy_smolvla/scripts/start_smolvla.sh --mode vision|frs [--config PATH] [--check]

Modes:
  vision  PyTorch SmolVLA deployment (default)
  frs     JAX SmolVLA + FRS deployment
EOF
}

while (( $# > 0 )); do
    case "$1" in
        --mode)
            [[ $# -ge 2 ]] || { echo "--mode requires vision or frs" >&2; exit 2; }
            MODE="$2"
            shift 2
            ;;
        --config)
            [[ $# -ge 2 ]] || { echo "--config requires a path" >&2; exit 2; }
            CONFIG="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            PASSTHROUGH+=("$1")
            shift
            ;;
    esac
done

case "${MODE}" in
    vision)
        CONFIG="${CONFIG:-${SMOLVLA_VISION_CONFIG:-${ROOT}/deploy_smolvla/configs/deploy_smolvla_pytorch.yaml}}"
        ;;
    frs)
        CONFIG="${CONFIG:-${SMOLVLA_FRS_CONFIG:-${ROOT}/deploy_smolvla/configs/deploy_frs.yaml}}"
        ;;
    *)
        echo "Unsupported mode: ${MODE}; expected vision or frs" >&2
        exit 2
        ;;
esac

echo "mode=${MODE}"
exec bash "${ROOT}/deploy_smolvla/scripts/start_remote_client.sh" \
    --config "${CONFIG}" "${PASSTHROUGH[@]}"
