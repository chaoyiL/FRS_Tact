#!/usr/bin/env bash
set -Eeuo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
TRAIN_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd -- "${TRAIN_ROOT}/.." && pwd)"
TRAIN_PYTHON_OVERRIDE="${TRAIN_PI05_FRS_PYTHON:-}"
if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
    cat <<'EOF'
用法：bash train_pi05_frs/scripts/start_frs_pi05_shared_train.sh [参数]
四个冻结 Pi0.5 -> 四任务缓存 -> 一个共享 20D FRS。
  --config PATH         默认 configs/train_pi05_frs_shared_four_tasks.yaml
  --check               仅检查四个任务配置和输入，不加载 GPU/创建缓存
  --prepare-task insert 仅生成指定任务的冻结缓存，暂不训练 FRS
  --train-only          使用已有四任务缓存，直接训练共享 FRS
  --resume              恢复同一共享 FRS 的 last checkpoint
  --evaluate-only --task insert 评估共享 best checkpoint 在 insert 验证集上的表现
默认在 tmux 的 frs_pi05_shared_train 会话中执行；FRS_FOREGROUND=1 使用当前终端。
EOF
    exit 0
fi
if [[ -f "${REPO_ROOT}/env_path" ]]; then
    source "${REPO_ROOT}/env_path"
fi
if [[ "${REPO_ROOT}" == /workspace/* ]]; then
    DEFAULT_PYTHON="${FRS_WORKSPACE_ROOT:-/workspace}/venvs/pi05_frs_train/bin/python"
else
    DEFAULT_PYTHON="${TRAIN_ROOT}/.venv/bin/python"
fi
TRAIN_PYTHON="${TRAIN_PYTHON_OVERRIDE:-${TRAIN_PI05_FRS_PYTHON:-${DEFAULT_PYTHON}}}"
[[ -x "${TRAIN_PYTHON}" ]] || { echo '请先运行 bash scripts/setup_env.sh --pi05_frs_train' >&2; exit 1; }
export PYTHONPATH="${TRAIN_ROOT}/src:${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONSAFEPATH=1
export PYTHONUNBUFFERED=1
cd "${TRAIN_ROOT}"
for ARG in "$@"; do
    if [[ "${ARG}" == "--check" ]]; then
        exec "${TRAIN_PYTHON}" -m train_pi05_frs.tools.train_shared_frs "$@"
    fi
done
if [[ "${FRS_FOREGROUND:-0}" != "1" && -z "${TMUX:-}" ]] && command -v tmux >/dev/null 2>&1; then
    "${TRAIN_PYTHON}" -m train_pi05_frs.tools.train_shared_frs "$@" --check
    SESSION="${FRS_TMUX_SESSION:-frs_pi05_shared_train}"
    if tmux has-session -t "${SESSION}" 2>/dev/null; then
        echo "tmux 会话已存在：${SESSION}" >&2
        exit 1
    fi
    printf -v INNER '%q ' env FRS_FOREGROUND=1 "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}" \
        "FRS_ARROW_TMPDIR=${FRS_ARROW_TMPDIR:-}" \
        bash "${SCRIPT_DIR}/start_frs_pi05_shared_train.sh" "$@"
    tmux new-session -d -s "${SESSION}" -c "${TRAIN_ROOT}" "${INNER}"
    echo "共享 FRS 已启动：tmux attach -t ${SESSION}"
    exit 0
fi
exec "${TRAIN_PYTHON}" -m train_pi05_frs.tools.train_shared_frs "$@"
