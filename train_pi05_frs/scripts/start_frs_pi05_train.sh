#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
TRAIN_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd -- "${TRAIN_ROOT}/.." && pwd)"
TRAIN_PYTHON="${TRAIN_PI05_FRS_PYTHON:-${TRAIN_ROOT}/.venv/bin/python}"
export PYTHONPATH="${TRAIN_ROOT}/src:${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
# The standalone directory contains ``utils/`` as part of ``train_pi05_frs``.
# Never let the working directory expose it as a top-level package that shadows
# the protected repository's ``utils`` imports used by ``train_encoder``.
export PYTHONSAFEPATH=1
TMUX_SESSION="${FRS_TMUX_SESSION:-frs_pi05_train}"

log() {
    printf '[frs-pi05] %s\n' "$*"
}

fail() {
    printf '[frs-pi05] error: %s\n' "$*" >&2
    return 1
}

trap 'status=$?; printf "[frs-pi05] pipeline failed with exit code %s\n" "${status}" >&2; exit "${status}"' ERR

usage() {
    printf 'usage: %s [--check] [CONFIG]\n' "$0" >&2
}

run_pipeline() {
    local config_path="$1"
    local output_dir="$2"
    local timestamp pipeline_log

    cd "${TRAIN_ROOT}"
    mkdir -p -- "${output_dir}"
    timestamp="$(date +%Y%m%d_%H%M%S)"
    pipeline_log="${output_dir}/pipeline_${timestamp}.log"
    export FRS_PIPELINE_LOG="${pipeline_log}"
    exec > >(tee -a "${pipeline_log}") 2>&1
    log "pipeline log: ${pipeline_log}"

    log "smoke-checking Pi0.5 checkpoint shapes and JAX GPU"
    FRS_PIPELINE_STAGE=checkpoint-smoke "${TRAIN_PYTHON}" \
        -m train_pi05_frs.tools.prepare_frs_pi05_cache \
        --config "${config_path}" --checkpoint-smoke

    log "precomputing tactile embeddings"
    FRS_PIPELINE_STAGE=precompute-tactile "${TRAIN_PYTHON}" \
        -m train_pi05_frs.tools.precompute_tactile_embeddings \
        --config "${config_path}"

    log "preparing Pi0.5 action caches"
    FRS_PIPELINE_STAGE=prepare-pi05-cache "${TRAIN_PYTHON}" \
        -m train_pi05_frs.tools.prepare_frs_pi05_cache \
        --config "${config_path}"

    log "training FRS decoder"
    FRS_PIPELINE_STAGE=train-frs "${TRAIN_PYTHON}" \
        -m train_pi05_frs.tools.train_frs \
        --config "${config_path}"

    log "pipeline completed"
}

main() {
    local check_only=0
    local config_path=""
    local output_dir inner

    while (($#)); do
        case "$1" in
            --check)
                check_only=1
                ;;
            -*)
                usage
                return 2
                ;;
            *)
                [[ -z "${config_path}" ]] || { usage; return 2; }
                config_path="$1"
                ;;
        esac
        shift
    done
    config_path="${config_path:-${TRAIN_ROOT}/configs/train_pi05_frs.yaml}"

    [[ -x "${TRAIN_PYTHON}" ]] || fail "training Python is not executable: ${TRAIN_PYTHON}"
    # Preserve the venv's python symlink: resolving its target would bypass
    # pyvenv.cfg and run the system interpreter without the training packages.
    TRAIN_PYTHON="$(realpath -s -- "${TRAIN_PYTHON}")"
    export TRAIN_PI05_FRS_PYTHON="${TRAIN_PYTHON}"
    [[ -f "${config_path}" ]] || fail "configuration does not exist: ${config_path}"
    config_path="$(realpath -- "${config_path}")"

    # Running from the standalone project keeps its package and private lerobot ahead
    # of similarly named packages at repository root.
    cd "${TRAIN_ROOT}"

    # Dependency-light schema/input validation happens before mkdir, JAX/model/GPU, or tmux.
    output_dir="$(
        FRS_PIPELINE_STAGE=validate "${TRAIN_PYTHON}" \
            -m train_pi05_frs.tools.train_frs \
            --config "${config_path}" --check --print-output
    )"
    [[ -n "${output_dir}" ]] || fail "configuration did not provide frs_training.output"

    if [[ "${check_only}" == "1" ]]; then
        log "configuration, schema, and input paths are valid: ${config_path}"
        return 0
    fi

    if [[ "${FRS_FOREGROUND:-0}" != "1" && -z "${TMUX:-}" ]] \
        && command -v tmux >/dev/null 2>&1; then
        if tmux has-session -t "${TMUX_SESSION}" 2>/dev/null; then
            fail "tmux session already exists: ${TMUX_SESSION}"
        fi
        printf -v inner 'source %q; run_pipeline %q %q' \
            "${BASH_SOURCE[0]}" "${config_path}" "${output_dir}"
        tmux new-session -d -s "${TMUX_SESSION}" -c "${TRAIN_ROOT}" "${inner}"
        log "pipeline started in tmux session ${TMUX_SESSION}"
        log "attach with: tmux attach -t ${TMUX_SESSION}"
        return 0
    fi

    run_pipeline "${config_path}" "${output_dir}"
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    main "$@"
fi
