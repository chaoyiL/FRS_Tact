#!/usr/bin/env bash
set -Eeuo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
    cat <<'EOF'
用法：bash scripts/upload_frs_pi05_shared_ckpt.sh OWNER/REPO [--dry-run] [--private] [--config PATH] [--checkpoint PATH]
默认上传 4090 配置对应的 best checkpoint；可用 --config 或 --checkpoint 指定。
先解析 best 符号链接、校验完整快照，再上传该版的全部清单文件到 HF model 仓库根目录。
--dry-run 只做本地校验，不访问 HF。上传需要目标仓库的写入权限。
默认新仓库为 public；传 --private 创建 private 新仓库。已有仓库的可见性保持原样。
使用专用 Pi0.5 FRS Python 环境，校验在 CPU 上运行，不要求训练图片。
EOF
    exit 0
fi
PYTHON_OVERRIDE="${TRAIN_PI05_FRS_PYTHON:-}"
if [[ -f "${REPO_ROOT}/env_path" ]]; then
    source "${REPO_ROOT}/env_path"
fi
if [[ "${REPO_ROOT}" == /workspace/* ]]; then
    DEFAULT_PYTHON="${FRS_WORKSPACE_ROOT:-/workspace}/venvs/pi05_frs_train/bin/python"
else
    DEFAULT_PYTHON="${REPO_ROOT}/train_pi05_frs/.venv/bin/python"
fi
UPLOAD_PYTHON="${PYTHON_OVERRIDE:-${TRAIN_PI05_FRS_PYTHON:-${DEFAULT_PYTHON}}}"
[[ -x "${UPLOAD_PYTHON}" ]] || { echo '请先运行 bash scripts/setup_env.sh --pi05_frs_train' >&2; exit 1; }
export PYTHONPATH="${REPO_ROOT}/train_pi05_frs/src:${REPO_ROOT}"
export PYTHONSAFEPATH=1
export JAX_PLATFORMS=cpu
export HF_HUB_ENABLE_HF_TRANSFER=0
cd "${REPO_ROOT}"
exec "${UPLOAD_PYTHON}" -m train_pi05_frs.tools.upload_shared_frs "$@"
