#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
RDP_DIR="$(cd -- "$SCRIPT_DIR/.." && pwd)"
export RDP_BASELINE_REPO_ROOT="$(cd -- "$RDP_DIR/.." && pwd)"
stage="${1:---help}"
case "$stage" in
  at|ldp|all) shift ;;
  -h|--help)
    cat <<'HELP'
Usage: bash train_RDP/scripts/train_rdp_baseline.sh {at|ldp|all} [Hydra overrides...]
Environment: RUN_ID, OUTPUT_ROOT, PYTHON_BIN, DATASET_PATH, TACTILE_CACHE_PATH,
TACTILE_PCA_PATH, AT_CKPT, AT_EPOCHS=601, LDP_EPOCHS=401, AT_BATCH=64,
LDP_BATCH=64, NUM_WORKERS=4, DEVICE=cuda:0, MIXED_PRECISION=no (LDP),
WANDB_MODE=offline, DRY_RUN=1 (print commands only).
For a separate LDP invocation, set AT_CKPT to the baseline AT's latest.ckpt.
HELP
    exit 0 ;;
  *) echo "Unknown stage: $stage" >&2; exit 2 ;;
esac
PYTHON_BIN="${PYTHON_BIN:-$RDP_DIR/.venv/bin/python}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$RDP_DIR/data/outputs/rdp_baseline}"
AT_DIR="$OUTPUT_ROOT/$RUN_ID/at"
LDP_DIR="$OUTPUT_ROOT/$RUN_ID/ldp"
AT_CKPT="${AT_CKPT:-$AT_DIR/checkpoints/latest.ckpt}"
common=("training.device=${DEVICE:-cuda:0}" "training.resume=false"
  "dataloader.num_workers=${NUM_WORKERS:-4}" "val_dataloader.num_workers=${NUM_WORKERS:-4}"
  "logging.mode=${WANDB_MODE:-offline}")
[[ -z "${DATASET_PATH:-}" ]] || common+=("dataset_path=$DATASET_PATH")
[[ -z "${TACTILE_CACHE_PATH:-}" ]] || common+=("tactile_cache_path=$TACTILE_CACHE_PATH")
[[ -z "${TACTILE_PCA_PATH:-}" ]] || common+=("tactile_pca_path=$TACTILE_PCA_PATH")
run() {
  printf '%q ' "$@"
  printf '\n'
  if [[ "${DRY_RUN:-0}" != 1 ]]; then "$@"; fi
}
if [[ "$stage" == at || "$stage" == all ]]; then
  run "$PYTHON_BIN" "$RDP_DIR/train_baseline.py" --config-name=train_at \
    "hydra.run.dir=$AT_DIR" "training.num_epochs=${AT_EPOCHS:-601}" \
    "dataloader.batch_size=${AT_BATCH:-64}" "val_dataloader.batch_size=${AT_BATCH:-64}" \
    "${common[@]}" "$@"
fi
if [[ "$stage" == ldp || "$stage" == all ]]; then
  run "$PYTHON_BIN" "$RDP_DIR/train_baseline.py" --config-name=train_ldp \
    "hydra.run.dir=$LDP_DIR" "at_load_dir=$AT_CKPT" \
    "training.mixed_precision=${MIXED_PRECISION:-no}" "training.num_epochs=${LDP_EPOCHS:-401}" \
    "dataloader.batch_size=${LDP_BATCH:-64}" "val_dataloader.batch_size=${LDP_BATCH:-64}" \
    "${common[@]}" "$@"
fi
