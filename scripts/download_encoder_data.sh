#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
    cat <<'EOF'
用法：bash scripts/download_encoder_data.sh [参数]
按 train_encoder/configs/train.yaml 下载 pick_01、insert_01、bread_03、two_tubes_01。
每个任务默认 300 条训练 + 35 条独立验证 episode；仅下载选中的 Parquet。
  --dry-run            只读取 metadata 并显示抽样计划，不下载 episode 文件
  --output-dir PATH    下载目录（默认 .cache/encoder_data/train300_val35）
  --config PATH        训练 YAML；任务、数量和 split_seed 均从这里读取
  --max-workers N      并发下载数（默认 4）
  --revision REV       首次下载的 Hub revision（默认 main，随后固定 commit）
可用 DATA_TOOL_PYTHON=/path/to/python 指定已安装数据依赖的解释器。
EOF
    exit 0
fi

PYTHON_OVERRIDE="${DATA_TOOL_PYTHON:-}"
for ENV_FILE in "${PROJECT_ROOT}/env_path" "${PROJECT_ROOT}/environment_paths.sh" "${PROJECT_ROOT}/.env.frs"; do
    if [[ -f "${ENV_FILE}" ]]; then
        # shellcheck disable=SC1090
        source "${ENV_FILE}"
        break
    fi
done

cd "${PROJECT_ROOT}"
for CANDIDATE in "${PYTHON_OVERRIDE}" "${DATA_TOOL_PYTHON:-}" "${FRS_PYTHON:-}" \
    "${SMOLVLA_TORCH_PYTHON:-}" "${PROJECT_ROOT}/.venv/bin/python" \
    "${PROJECT_ROOT}/.venv-smolvla-torch/bin/python"; do
    if [[ -n "${CANDIDATE}" && -x "${CANDIDATE}" ]]; then
        exec "${CANDIDATE}" -m train_encoder.download_data "$@"
    fi
done
if command -v uv >/dev/null 2>&1; then
    # 下载仅依赖轻量数据包，不同步项目的 CUDA/PyTorch 训练依赖。
    exec uv run --no-project --with 'huggingface-hub>=0.34,<1' \
        --with 'numpy>=2,<2.3' --with 'pyarrow>=21,<30' --with 'PyYAML>=6,<7' \
        python -m train_encoder.download_data "$@"
fi
echo '找不到数据工具解释器或 uv；请设置 DATA_TOOL_PYTHON，或安装 uv。' >&2
exit 1
