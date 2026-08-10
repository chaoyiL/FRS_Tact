#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

# Edit these pinned deployment inputs deliberately when updating the asset set.
CHECKPOINT_ROOT="${FRS_CHECKPOINT_ROOT:-${ROOT}/checkpoints}"
BASE_REPO="lerobot/smolvla_base"
BASE_REVISION="c83c3163b8ca9b7e67c509fffd9121e66cb96205"
ADAPTER_REPO="KaiyueChen/pick_tube_02_3w"
ADAPTER_REVISION="31d819d8844de98174ede123f894adbf7b4372ef"
FRS_REPO="KaiyueChen/frs_0809_02"
FRS_REVISION="7e23f3e8c308dc5ba3a4df7634c68dac28572897"
ENCODER_REPO="KaiyueChen/encoder_ckpt_0809"
ENCODER_REVISION="450aa60963cde9540bd6c8047bf2529eff1def37"
BASE_DIR="${CHECKPOINT_ROOT}/model/pick_tube_02_3w_jax"
FRS_DIR="${CHECKPOINT_ROOT}/frs/frs_0809_02"
ENCODER_DIR="${CHECKPOINT_ROOT}/encoder/encoder_ckpt_0809"

if [[ -n "${FRS_DOWNLOAD_UV:-}" ]]; then
    UV_BIN="${FRS_DOWNLOAD_UV}"
elif command -v uv >/dev/null 2>&1; then
    UV_BIN="$(command -v uv)"
elif [[ -x "${HOME:-}/.local/bin/uv" ]]; then
    UV_BIN="${HOME}/.local/bin/uv"
else
    echo "uv was not found; set FRS_DOWNLOAD_UV or install uv" >&2
    exit 1
fi

if [[ -n "${FRS_DOWNLOAD_PYTHON:-}" ]]; then
    PYTHON_CMD=("${FRS_DOWNLOAD_PYTHON}")
else
    PYTHON_CMD=("${UV_BIN}" run --no-sync python)
fi

base_complete() {
    "${PYTHON_CMD[@]}" - "${BASE_DIR}" "${BASE_REPO}" "${BASE_REVISION}" \
        "${ADAPTER_REPO}" "${ADAPTER_REVISION}" <<'PY'
import json
from pathlib import Path
import sys

directory = Path(sys.argv[1])
base_repo, base_revision, adapter_repo, adapter_revision = sys.argv[2:]
required = (
    "model.safetensors",
    "config.json",
    "train_config.json",
    "policy_preprocessor.json",
    "policy_postprocessor.json",
    "policy_preprocessor_step_5_normalizer_processor.safetensors",
    "policy_postprocessor_step_0_unnormalizer_processor.safetensors",
)
if any(not (directory / name).is_file() or (directory / name).stat().st_size == 0 for name in required):
    raise SystemExit(1)
try:
    with (directory / "conversion_manifest.json").open(encoding="utf-8") as file:
        manifest = json.load(file)
except (OSError, ValueError, TypeError):
    raise SystemExit(1)
expected = {
    "source_base": base_repo,
    "base_revision": base_revision,
    "source_adapter": adapter_repo,
    "adapter_revision": adapter_revision,
}
if not isinstance(manifest, dict) or any(manifest.get(key) != value for key, value in expected.items()):
    raise SystemExit(1)
PY
}

checkpoint_complete() {
    "${PYTHON_CMD[@]}" - "$1" "$2" "$3" "$4" "${ROOT}" <<'PY'
import json
from pathlib import Path
import sys

import numpy as np

directory = Path(sys.argv[1]).resolve()
repo_id, revision, checkpoint_kind, project_root = sys.argv[2:]
try:
    with (directory / ".download-provenance.json").open(encoding="utf-8") as file:
        provenance = json.load(file)
    expected_provenance = {
        "format_version": 1,
        "repo_id": repo_id,
        "revision": revision,
    }
    if provenance != expected_provenance:
        raise ValueError("checkpoint provenance mismatch")
    if checkpoint_kind == "encoder":
        sys.path.insert(0, project_root)
        from deploy_smolvla.src.download_ckpt import verify_checkpoint

        verify_checkpoint(directory)
        raise SystemExit(0)
    with (directory / "checkpoint.json").open(encoding="utf-8") as file:
        metadata = json.load(file)
    if not isinstance(metadata, dict):
        raise ValueError("invalid checkpoint metadata")
    params_name = metadata.get("params_file", "params.npz")
    if not isinstance(params_name, str) or not params_name:
        raise ValueError("invalid checkpoint metadata")
    params_path = (directory / params_name).resolve()
    if not params_path.is_relative_to(directory):
        raise ValueError("params archive is outside checkpoint directory")
    if not params_path.is_file() or params_path.stat().st_size == 0:
        raise ValueError("params archive is missing or empty")
    with np.load(params_path, allow_pickle=False) as archive:
        if not archive.files:
            raise ValueError("params archive is empty")
except (ImportError, OSError, ValueError, TypeError, EOFError, KeyError):
    raise SystemExit(1)
PY
}

frs_complete() {
    checkpoint_complete "${FRS_DIR}" "${FRS_REPO}" "${FRS_REVISION}" frs
}

encoder_complete() {
    checkpoint_complete "${ENCODER_DIR}" "${ENCODER_REPO}" "${ENCODER_REVISION}" encoder
}

write_provenance() {
    "${PYTHON_CMD[@]}" - "$1" "$2" "$3" <<'PY'
import json
from pathlib import Path
import sys

directory = Path(sys.argv[1])
directory.mkdir(parents=True, exist_ok=True)
destination = directory / ".download-provenance.json"
temporary = directory / ".download-provenance.json.tmp"
temporary.write_text(
    json.dumps(
        {"format_version": 1, "repo_id": sys.argv[2], "revision": sys.argv[3]},
        sort_keys=True,
    ) + "\n",
    encoding="utf-8",
)
temporary.replace(destination)
PY
}

if base_complete; then
    echo "skip: base checkpoint: ${BASE_DIR}"
else
    case "${BASE_DIR}" in
        "${CHECKPOINT_ROOT}/model/"*) ;;
        *)
            echo "refusing to overwrite base directory outside ${CHECKPOINT_ROOT}/model: ${BASE_DIR}" >&2
            exit 1
            ;;
    esac
    if ! "${UV_BIN}" run --no-sync python "${ROOT}/tools/merge_smolvla_peft_to_jax.py" \
        --adapter "${ADAPTER_REPO}" --adapter-revision "${ADAPTER_REVISION}" \
        --base "${BASE_REPO}" --base-revision "${BASE_REVISION}" \
        --output "${BASE_DIR}" --allow-download --overwrite; then
        echo "base checkpoint merge failed: ${BASE_DIR}" >&2
        exit 1
    fi
    base_complete || {
        echo "base checkpoint failed validation after merge: ${BASE_DIR}" >&2
        exit 1
    }
fi

if frs_complete; then
    echo "skip: FRS checkpoint: ${FRS_DIR}"
else
    if ! "${UV_BIN}" run --no-sync hf download "${FRS_REPO}" \
        --revision "${FRS_REVISION}" --include checkpoint.json 'params-*.npz' \
        --local-dir "${FRS_DIR}"; then
        echo "FRS checkpoint download failed: ${FRS_DIR}" >&2
        exit 1
    fi
    write_provenance "${FRS_DIR}" "${FRS_REPO}" "${FRS_REVISION}"
    frs_complete || {
        echo "FRS checkpoint failed validation after download: ${FRS_DIR}" >&2
        exit 1
    }
fi

if encoder_complete; then
    echo "skip: tactile encoder checkpoint: ${ENCODER_DIR}"
else
    if ! "${UV_BIN}" run --no-sync python "${ROOT}/deploy_smolvla/src/download_ckpt.py" \
        --minimal --repo-id "${ENCODER_REPO}" --revision "${ENCODER_REVISION}" \
        --output-dir "${ENCODER_DIR}"; then
        echo "tactile encoder download failed: ${ENCODER_DIR}" >&2
        exit 1
    fi
    write_provenance "${ENCODER_DIR}" "${ENCODER_REPO}" "${ENCODER_REVISION}"
    encoder_complete || {
        echo "encoder checkpoint failed validation after download: ${ENCODER_DIR}" >&2
        exit 1
    }
fi

echo "deployment checkpoints ready:"
echo "  checkpoint: ${BASE_DIR}"
echo "  frs.checkpoint: ${FRS_DIR}"
echo "  frs.tactile_encoder_checkpoint: ${ENCODER_DIR}"
