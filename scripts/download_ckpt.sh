#!/usr/bin/env bash
set -euo pipefail

# Checkpoint/model assets only. Training datasets are owned by scripts/download_data.sh.
#
# 使用方法（在项目根目录执行）：
#   1. 下载普通 PyTorch 模型或 checkpoint：
#      bash scripts/download_ckpt.sh OWNER/REPO
#      示例：bash scripts/download_ckpt.sh lerobot/smolvla_base
#
#   2. 下载 SmolVLA FRS 部署所需的完整资源：
#      bash scripts/download_ckpt.sh --smolvla-frs
#      这会下载并检查 JAX SmolVLA、FRS checkpoint、触觉编码器和 tokenizer。
#
#   3. 不传参数时，默认执行 --smolvla-frs：
#      bash scripts/download_ckpt.sh
#
#   4. 查看帮助：
#      bash scripts/download_ckpt.sh --help
#
# 可选环境变量：
#   FRS_CHECKPOINT_ROOT  修改 checkpoint 保存根目录（默认：<项目>/checkpoints）
#   FRS_DOWNLOAD_UV     指定 uv 可执行文件
#   FRS_DOWNLOAD_PYTHON 指定用于校验 checkpoint 的 Python 可执行文件
# 脚本也会自动读取项目根目录中的 .env.frs（如果存在）。

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DOWNLOAD_MODE="smolvla-frs"
GENERIC_REPO=""

usage() {
    cat <<'EOF'
Usage:
  bash scripts/download_ckpt.sh OWNER/REPO
  bash scripts/download_ckpt.sh --smolvla-frs

OWNER/REPO       Download one complete PyTorch/model repository.
--smolvla-frs    Download and validate the configured JAX SmolVLA, FRS,
                 tactile encoder, and tokenizer deployment asset set.
With no arguments, --smolvla-frs is used for backward compatibility.
EOF
}

case "$#" in
    0) ;;
    1)
        case "$1" in
            -h|--help)
                usage
                exit 0
                ;;
            --smolvla-frs)
                DOWNLOAD_MODE="smolvla-frs"
                ;;
            *)
                DOWNLOAD_MODE="model"
                GENERIC_REPO="${1#--}"
                ;;
        esac
        ;;
    *)
        usage >&2
        exit 2
        ;;
esac

ENV_FILE="${ROOT}/.env.frs"
if [[ -f "${ENV_FILE}" ]]; then
    # shellcheck disable=SC1090
    source "${ENV_FILE}"
fi

# Edit only these Hugging Face repository IDs when updating the asset set.
CHECKPOINT_ROOT="${FRS_CHECKPOINT_ROOT:-${ROOT}/checkpoints}"
BASE_REPO="lerobot/smolvla_base"
BASE_REVISION="c83c3163b8ca9b7e67c509fffd9121e66cb96205"
SMOLVLA_REPO="KaiyueChen/pick_tube_01"
FRS_REPO="KaiyueChen/smolvla_frs_pick_tube_05_bimanual_best"
ENCODER_REPO="KaiyueChen/encoder_ckpt_0809"
TOKENIZER_REPO="HuggingFaceTB/SmolVLM2-500M-Video-Instruct"
TOKENIZER_REVISION="7b375e1b73b11138ff12fe22c8f2822d8fe03467"
TOKENIZER_FILES=(
    config.json
    tokenizer_config.json
    tokenizer.json
    special_tokens_map.json
    added_tokens.json
    chat_template.json
    merges.txt
    vocab.json
)
TOKENIZER_CACHE_ROOT="${CHECKPOINT_ROOT}/model"
TOKENIZER_REPO_CACHE="${TOKENIZER_CACHE_ROOT}/models--HuggingFaceTB--SmolVLM2-500M-Video-Instruct"

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

validate_repo_id() {
    "${PYTHON_CMD[@]}" - "$1" <<'PY'
from huggingface_hub.utils import validate_repo_id
import sys

try:
    validate_repo_id(sys.argv[1])
except ValueError:
    raise SystemExit(1)
PY
}

derive_repo_basename() {
    local repo_id="$1"
    local basename="${repo_id##*/}"

    case "${basename}" in
        ''|.|..|*/*|*\\*)
            return 1
            ;;
    esac
    printf '%s\n' "${basename}"
}

resolve_revision() {
    local repo_id="$1"
    local revision

    revision="$("${PYTHON_CMD[@]}" - "${repo_id}" <<'PY'
from huggingface_hub import HfApi
import sys

print(HfApi().model_info(sys.argv[1]).sha)
PY
)" || {
        echo "could not resolve revision for ${repo_id}" >&2
        return 1
    }

    if [[ ! "${revision}" =~ ^[0-9a-f]{40}$ ]]; then
        echo "could not resolve revision for ${repo_id}: invalid revision" >&2
        return 1
    fi
    printf '%s\n' "${revision}"
}

download_model_repo() {
    local repo_id="$1"
    local repo_name=""
    local model_root="${CHECKPOINT_ROOT}/model"
    local metadata_root="${model_root}/.frs_hf_repos"
    local output_dir=""
    local repo_marker=""
    local owned_repo_id=""

    if ! validate_repo_id "${repo_id}"; then
        echo "invalid Hugging Face repository ID: ${repo_id}" >&2
        exit 2
    fi
    repo_name="$(derive_repo_basename "${repo_id}")"
    output_dir="${model_root}/${repo_name}"
    repo_marker="${metadata_root}/${repo_name}.repo-id"

    if [[ -L "${CHECKPOINT_ROOT}" || -L "${model_root}" || -L "${metadata_root}" ]]; then
        echo "checkpoint download roots must not be symbolic links" >&2
        exit 1
    fi
    mkdir -p "${metadata_root}"
    if [[ -L "${output_dir}" || ( -e "${output_dir}" && ! -d "${output_dir}" ) ]]; then
        echo "invalid checkpoint output directory: ${output_dir}" >&2
        exit 1
    fi
    if [[ -f "${repo_marker}" ]]; then
        owned_repo_id="$(<"${repo_marker}")"
        if [[ "${owned_repo_id}" != "${repo_id}" ]]; then
            echo "${output_dir} belongs to ${owned_repo_id}; refusing ${repo_id}" >&2
            exit 1
        fi
    elif [[ -d "${output_dir}" && -n "$(find "${output_dir}" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
        echo "existing checkpoint directory lacks repository ownership marker: ${output_dir}" >&2
        exit 1
    else
        mkdir -p "${output_dir}"
        printf '%s\n' "${repo_id}" >"${repo_marker}"
    fi

    echo "download model: ${repo_id} -> ${output_dir}"
    "${UV_BIN}" run --no-sync hf download "${repo_id}" \
        --repo-type model \
        --local-dir "${output_dir}"
}

if [[ "${DOWNLOAD_MODE}" == "model" ]]; then
    download_model_repo "${GENERIC_REPO}"
    exit 0
fi

for repo_id in "${SMOLVLA_REPO}" "${FRS_REPO}" "${ENCODER_REPO}"; do
    if ! validate_repo_id "${repo_id}"; then
        echo "invalid Hugging Face repository ID: ${repo_id}" >&2
        exit 1
    fi
done

if ! SMOLVLA_BASENAME="$(derive_repo_basename "${SMOLVLA_REPO}")" || \
    ! FRS_BASENAME="$(derive_repo_basename "${FRS_REPO}")" || \
    ! ENCODER_BASENAME="$(derive_repo_basename "${ENCODER_REPO}")"; then
    echo "invalid Hugging Face repository basename" >&2
    exit 1
fi
BASE_DIR="${CHECKPOINT_ROOT}/model/${SMOLVLA_BASENAME}_jax"
FRS_DIR="${CHECKPOINT_ROOT}/frs/${FRS_BASENAME}"
ENCODER_DIR="${CHECKPOINT_ROOT}/encoder/${ENCODER_BASENAME}"

if ! SMOLVLA_REVISION="$(resolve_revision "${SMOLVLA_REPO}")"; then
    exit 1
fi
if ! FRS_REVISION="$(resolve_revision "${FRS_REPO}")"; then
    exit 1
fi
if ! ENCODER_REVISION="$(resolve_revision "${ENCODER_REPO}")"; then
    exit 1
fi

echo "resolved: ${SMOLVLA_REPO} @ ${SMOLVLA_REVISION} -> ${BASE_DIR}"
echo "resolved: ${FRS_REPO} @ ${FRS_REVISION} -> ${FRS_DIR}"
echo "resolved: ${ENCODER_REPO} @ ${ENCODER_REVISION} -> ${ENCODER_DIR}"

validate_checkpoint_targets() {
    "${PYTHON_CMD[@]}" - "${CHECKPOINT_ROOT}" \
        "${BASE_DIR}" "${FRS_DIR}" "${ENCODER_DIR}" <<'PY'
from pathlib import Path
import sys

checkpoint_root = Path(sys.argv[1]).resolve(strict=False)
targets = (
    ("base", Path(sys.argv[2]), checkpoint_root / "model"),
    ("FRS", Path(sys.argv[3]), checkpoint_root / "frs"),
    ("encoder", Path(sys.argv[4]), checkpoint_root / "encoder"),
)
for label, target, allowed_root in targets:
    canonical_target = target.resolve(strict=False)
    canonical_root = allowed_root.resolve(strict=False)
    if not canonical_root.is_relative_to(checkpoint_root):
        print(
            f"refusing checkpoint category root outside checkpoint root: "
            f"{label}: {canonical_root} (checkpoint root: {checkpoint_root})",
            file=sys.stderr,
        )
        raise SystemExit(1)
    if not canonical_target.is_relative_to(canonical_root):
        print(
            f"refusing checkpoint target outside its allowed root: "
            f"{label}: {canonical_target} (allowed: {canonical_root})",
            file=sys.stderr,
        )
        raise SystemExit(1)
PY
}

validate_checkpoint_targets

base_complete() {
    "${PYTHON_CMD[@]}" - "${BASE_DIR}" "${BASE_REPO}" "${BASE_REVISION}" \
        "${SMOLVLA_REPO}" "${SMOLVLA_REVISION}" <<'PY'
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

tokenizer_complete() {
    "${PYTHON_CMD[@]}" - "${TOKENIZER_CACHE_ROOT}" "${TOKENIZER_REPO_CACHE}" \
        "${TOKENIZER_REPO}" "${TOKENIZER_REVISION}" "${TOKENIZER_FILES[@]}" <<'PY'
import os
from pathlib import Path
import sys

cache_root = Path(sys.argv[1]).resolve()
repo_cache = Path(sys.argv[2]).resolve()
repo_id = sys.argv[3]
revision = sys.argv[4]
required = tuple(sys.argv[5:])
try:
    if (repo_cache / "refs/main").read_text(encoding="utf-8") != revision:
        raise ValueError("tokenizer revision mismatch")
    snapshot = repo_cache / "snapshots" / revision
    if any(
        not (snapshot / name).is_file() or (snapshot / name).stat().st_size == 0
        for name in required
    ):
        raise ValueError("tokenizer cache is incomplete")
    for variable in (
        "TRANSFORMERS_CACHE",
        "PYTORCH_TRANSFORMERS_CACHE",
        "PYTORCH_PRETRAINED_BERT_CACHE",
    ):
        os.environ.pop(variable, None)
    os.environ["HF_HUB_CACHE"] = str(cache_root)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    from transformers import AutoTokenizer

    AutoTokenizer.from_pretrained(
        repo_id,
        local_files_only=True,
        cache_dir=str(cache_root),
    )
except (ImportError, OSError, ValueError, TypeError):
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

write_tokenizer_ref() {
    "${PYTHON_CMD[@]}" - "${TOKENIZER_REPO_CACHE}" "${TOKENIZER_REVISION}" <<'PY'
from pathlib import Path
import sys

repo_cache = Path(sys.argv[1])
revision = sys.argv[2]
refs = repo_cache / "refs"
refs.mkdir(parents=True, exist_ok=True)
destination = refs / "main"
temporary = refs / "main.tmp"
temporary.write_text(revision, encoding="utf-8")
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
        --adapter "${SMOLVLA_REPO}" --adapter-revision "${SMOLVLA_REVISION}" \
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

if tokenizer_complete; then
    echo "skip: tokenizer cache: ${TOKENIZER_CACHE_ROOT}"
else
    if ! "${UV_BIN}" run --no-sync hf download "${TOKENIZER_REPO}" \
        --revision "${TOKENIZER_REVISION}" --include "${TOKENIZER_FILES[@]}" \
        --cache-dir "${TOKENIZER_CACHE_ROOT}" --force-download; then
        echo "tokenizer download failed: ${TOKENIZER_REPO} -> ${TOKENIZER_CACHE_ROOT}" >&2
        exit 1
    fi
    write_tokenizer_ref
    tokenizer_complete || {
        echo "tokenizer cache failed validation after download: ${TOKENIZER_CACHE_ROOT}" >&2
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
echo "  HF_HUB_CACHE: ${TOKENIZER_CACHE_ROOT}"
