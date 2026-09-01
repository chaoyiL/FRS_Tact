#!/usr/bin/env bash
set -euo pipefail

# Run the server-side insert, press, and Bread RDP pipelines as one serial job.
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
DEFAULT_RDP_CODE_DIR=$(cd -- "${SCRIPT_DIR}/.." && pwd)
RDP_CODE_DIR=${RDP_SEQUENCE_CODE_DIR:-${DEFAULT_RDP_CODE_DIR}}
STAGE=${1:-train}
RDP_SEQUENCE_ID=${RDP_SEQUENCE_ID:-$(date +%Y%m%d_%H%M%S)}
EXPERIMENT_ID=${EXPERIMENT_ID:-${RDP_SEQUENCE_ID}}
BREAD_EXPERIMENT_ID=${BREAD_EXPERIMENT_ID:-${RDP_SEQUENCE_ID}}
BREAD_RESUME=${BREAD_RESUME:-${RESUME:-true}}

usage() {
  cat <<'USAGE'
Usage: bash scripts/server_ljl_rdp_sequence.sh <train|all|doctor>

Runs insert, press, then Bread in strict serial order. The default stage is train.
RDP_SEQUENCE_ID defaults to a timestamp and is passed to both task launchers
unless EXPERIMENT_ID or BREAD_EXPERIMENT_ID is explicitly set.
USAGE
}

case "${STAGE}" in
  train|all|doctor)
    ;;
  -h|--help|help)
    usage
    exit 0
    ;;
  *)
    usage >&2
    exit 2
    ;;
esac

run_single() {
  local task=$1
  env -u BASELINE_JSON \
    "RDP_DIR=${RDP_CODE_DIR}" \
    "EXPERIMENT_ID=${EXPERIMENT_ID}" \
    bash "${RDP_CODE_DIR}/scripts/server_ljl_single_right.sh" "${STAGE}" "${task}"
}

run_bread() {
  env -u BASELINE_JSON \
    "BREAD_RDP_CODE_DIR=${RDP_CODE_DIR}" \
    "BREAD_EXPERIMENT_ID=${BREAD_EXPERIMENT_ID}" \
    "BREAD_RESUME=${BREAD_RESUME}" \
    bash "${RDP_CODE_DIR}/scripts/server_ljl_bread_dual.sh" "${STAGE}"
}

run_single insert
run_single press
run_bread

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  printf '\nRDP sequence dry run completed.\n'
else
  printf '\nRDP sequence completed.\n'
fi
printf 'stage: %s\n' "${STAGE}"
printf 'sequence id: %s\n' "${RDP_SEQUENCE_ID}"
