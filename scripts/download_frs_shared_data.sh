#!/usr/bin/env bash
set -Eeuo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
    cat <<'EOF'
用法：bash scripts/download_frs_shared_data.sh [--dry-run] [--config PATH] [--max-workers N]
按共享 FRS 配置下载四任务各 300 训练 + 35 验证所需的 335 条 episode。
仅下载选中的 v2.1 Parquet，然后保留全部模态转换为可用于 Pi0.5 的完整 v3 子集。
默认保存到 /workspace/frs_shared_four_tasks/datasets/{raw,v30}。
需要先运行 bash scripts/setup_env.sh --pi05_frs_train。
EOF
    exit 0
fi
PYTHON_OVERRIDE="${DATA_TOOL_PYTHON:-}"
if [[ -f "${REPO_ROOT}/env_path" ]]; then
    source "${REPO_ROOT}/env_path"
fi
if [[ "${REPO_ROOT}" == /workspace/* ]]; then
    DEFAULT_PYTHON="${FRS_WORKSPACE_ROOT:-/workspace}/venvs/lerobot_data_tools/bin/python"
else
    DEFAULT_PYTHON="${REPO_ROOT}/data_tools/.venv/bin/python"
fi
DATA_PYTHON="${PYTHON_OVERRIDE:-${DATA_TOOL_PYTHON:-${DEFAULT_PYTHON}}}"
[[ -x "${DATA_PYTHON}" ]] || { echo '请先运行 bash scripts/setup_env.sh --pi05_frs_train' >&2; exit 1; }
# Server images may enable this optional backend for a different Python environment.
# Disable it before huggingface_hub reads its environment at import time.
case "${HF_HUB_ENABLE_HF_TRANSFER:-0}" in
    1|[Oo][Nn]|[Yy][Ee][Ss]|[Tt][Rr][Uu][Ee])
        if ! "${DATA_PYTHON}" -c 'import hf_transfer' >/dev/null 2>&1; then
            export HF_HUB_ENABLE_HF_TRANSFER=0
            echo '当前数据工具环境没有可用的 hf_transfer，已切换为普通 Hugging Face 下载。' >&2
        fi
        ;;
esac
# Conversion uses the root data-tools LeRobot; never the private FRS reader package.
export PYTHONPATH="${REPO_ROOT}"
cd "${REPO_ROOT}"
exec "${DATA_PYTHON}" -m train_pi05_frs.tools.prepare_shared_data "$@"
