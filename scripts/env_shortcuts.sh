#!/usr/bin/env bash
# 由 env_path source；为多套隔离环境提供不歧义的快捷命令。

_frs_run_python() {
    local executable="$1"
    local setup_hint="$2"
    shift 2
    if [[ -z "${executable}" || ! -x "${executable}" ]]; then
        printf '[env] Python 不可执行：%s；请运行 %s\n' "${executable:-unset}" "${setup_hint}" >&2
        return 127
    fi
    "${executable}" "$@"
}

data-python() {
    _frs_run_python "${DATA_TOOL_PYTHON:-}" "bash scripts/setup_env.sh --pi05_train" "$@"
}

pi05-python() {
    _frs_run_python "${TRAIN_PI05_PYTHON:-}" "bash scripts/setup_env.sh --pi05_train" "$@"
}

pi05-deploy-python() {
    _frs_run_python "${PI05_PYTHON:-}" "bash scripts/setup_env.sh --pi05_deploy" "$@"
}

pi05-frs-python() {
    _frs_run_python "${TRAIN_PI05_FRS_PYTHON:-}" \
        "bash scripts/setup_env.sh --pi05_frs_train" "$@"
}

smolvla-python() {
    _frs_run_python "${SMOLVLA_TORCH_PYTHON:-}" "bash scripts/setup_env.sh --smolvla" "$@"
}

hf() {
    local python_path=""
    local cli=""
    for python_path in \
        "${DATA_TOOL_PYTHON:-}" \
        "${FRS_PYTHON:-}" \
        "${PI05_PYTHON:-}" \
        "${TRAIN_PI05_PYTHON:-}" \
        "${TRAIN_PI05_FRS_PYTHON:-}"
    do
        [[ -n "${python_path}" ]] || continue
        cli="$(dirname -- "${python_path}")/hf"
        if [[ -x "${cli}" ]]; then
            "${cli}" "$@"
            return $?
        fi
    done
    printf '[env] 找不到 hf；请先运行对应的 setup_env.sh 环境配置\n' >&2
    return 127
}

wandb() {
    local python_path=""
    local cli=""
    for python_path in "${TRAIN_PI05_PYTHON:-}" "${FRS_PYTHON:-}"
    do
        [[ -n "${python_path}" ]] || continue
        cli="$(dirname -- "${python_path}")/wandb"
        if [[ -x "${cli}" ]]; then
            "${cli}" "$@"
            return $?
        fi
    done
    printf '[env] 找不到 wandb；请先运行 --pi05_train 或 --smolvla 环境配置\n' >&2
    return 127
}
