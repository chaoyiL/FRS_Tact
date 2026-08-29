#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN=${PYTHON_BIN:-python3}
DATASET_ROOT=${DATASET_ROOT:-/home/hillbot/datasets}
TACTILE_CACHE_ROOT=${TACTILE_CACHE_ROOT:-data/tactile_embeddings_encoder0809}
OUTPUT_DIR=${OUTPUT_DIR:-data/pick_tube_01_04_rdp_zarr}

"${PYTHON_BIN}" convert_pick_tube_lerobot_to_rdp_zarr.py \
  --dataset-root "${DATASET_ROOT}" \
  --tactile-cache-root "${TACTILE_CACHE_ROOT}" \
  --output-dir "${OUTPUT_DIR}" \
  "$@"

"${PYTHON_BIN}" validate_pick_tube_batch.py "${OUTPUT_DIR}" --mode at
"${PYTHON_BIN}" validate_pick_tube_batch.py "${OUTPUT_DIR}" --mode ldp
