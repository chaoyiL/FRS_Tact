#!/usr/bin/env bash
set -Eeuo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
    cat <<'EOF'
用法：bash scripts/download_frs_shared_ckpts.sh [--dry-run] [--config PATH] [--max-workers N]
下载共享 FRS 配置中的 insert、press、bread、pick 四个冻结 Pi0.5 权重。
固定仓库 commit，仅下载 Orbax params 和各任务 norm_stats，不下载 optimizer。
默认保存到 /workspace/FRS_Tact/checkpoints/model/，支持中断后重复运行。
触觉 encoder 使用 SHARED_FOUR_TASKS.md 中的独立下载命令。
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
export PYTHONPATH="${REPO_ROOT}"
cd "${REPO_ROOT}"
exec "${DATA_PYTHON}" -m train_pi05_frs.tools.download_shared_checkpoints "$@"
