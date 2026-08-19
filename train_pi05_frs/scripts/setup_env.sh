#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
TRAIN_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd -- "${TRAIN_ROOT}/.." && pwd)"
TRAIN_VENV="${TRAIN_PI05_FRS_VENV:-${TRAIN_ROOT}/.venv}"
UV_BIN="${UV_BIN:-uv}"
TRAIN_PI05_FRS_PYTHON="${TRAIN_PI05_FRS_PYTHON:-${TRAIN_VENV}/bin/python}"

fail() {
    printf '%s\n' "$*" >&2
    return 1
}

validate_environment_targets() {
    TRAIN_VENV="$(realpath -m -- "${TRAIN_VENV}")"
    local root_venv deploy_venv
    root_venv="$(realpath -m -- "${REPO_ROOT}/.venv")"
    deploy_venv="$(realpath -m -- "${REPO_ROOT}/deploy_pi05/.venv")"
    [[ "${TRAIN_VENV}" != "${root_venv}" && "${TRAIN_VENV}" != "${deploy_venv}" ]] \
        || fail "Pi0.5 FRS 训练必须使用独立虚拟环境：${TRAIN_VENV}"
    TRAIN_PI05_FRS_PYTHON="${TRAIN_PI05_FRS_PYTHON:-${TRAIN_VENV}/bin/python}"
}

sync_environment() {
    validate_environment_targets
    UV_PROJECT_ENVIRONMENT="${TRAIN_VENV}" "${UV_BIN}" sync \
        --frozen --python 3.12 --project "${TRAIN_ROOT}"
}

check_environment() {
    validate_environment_targets
    printf 'project: %s\n' "${TRAIN_ROOT}"
    printf 'environment: %s\n' "${TRAIN_VENV}"
    printf 'python: 3.12\n'
    printf 'python executable: %s\n' "${TRAIN_PI05_FRS_PYTHON}"
    printf 'entrypoints: setup_env.sh, start_frs_pi05_train.sh\n'
}

main() {
    case "${1:-}" in
        "") sync_environment ;;
        --check) check_environment ;;
        *) fail "usage: $0 [--check]" ;;
    esac
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    main "$@"
fi
