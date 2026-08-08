#!/usr/bin/env bash

set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
ENV_FILE="${PROJECT_ROOT}/.env.frs"

HF_NAMESPACE="KaiyueChen"
DATASETS=(pick_tube_01 pick_tube_02 pick_tube_03 pick_tube_04)
CLEANUP_SOURCE=0
PROJECT_PYTHON=""
UV_BIN=""
DATASET_ALREADY_VALIDATED=0

log() {
    echo "[download-data] $*" >&2
}

fail() {
    echo "[download-data] 错误：$*" >&2
    exit 1
}

usage() {
    printf '%s\n' \
        "用法：scripts/download_data.sh [--cleanup-source]" \
        "" \
        "下载并验证四个 pick_tube v3 数据集，然后准备最小 encoder_05。" \
        "  --cleanup-source    全部验证成功后删除四个源 snapshot cache 和转换残留" \
        "  --help              显示本帮助"
}

parse_arguments() {
    while (($#)); do
        case "$1" in
            --cleanup-source)
                CLEANUP_SOURCE=1
                ;;
            -h|--help)
                usage
                exit 0
                ;;
            *)
                usage >&2
                fail "未知参数：$1"
                ;;
        esac
        shift
    done
}

require_environment() {
    [[ -f "${ENV_FILE}" ]] || fail "缺少 ${ENV_FILE}；请先运行 scripts/setup_env.sh"
    # shellcheck disable=SC1090
    source "${ENV_FILE}"

    local variable
    for variable in \
        FRS_STORAGE_ROOT FRS_VENV_DIR UV_PROJECT_ENVIRONMENT \
        HF_HOME HF_HUB_CACHE HF_DATASETS_CACHE HF_LEROBOT_HOME TMPDIR
    do
        [[ -n "${!variable:-}" ]] || fail ".env.frs 缺少 ${variable}"
    done
    [[ "${UV_PROJECT_ENVIRONMENT}" == "${FRS_VENV_DIR}" ]] || \
        fail ".env.frs 中 UV_PROJECT_ENVIRONMENT 必须等于 FRS_VENV_DIR"
    PROJECT_PYTHON="${FRS_VENV_DIR}/bin/python"
    [[ -x "${PROJECT_PYTHON}" ]] || \
        fail "FRS_VENV_DIR 指向的环境不存在或没有 Python：${FRS_VENV_DIR}"

    command -v uv >/dev/null 2>&1 || fail "找不到 uv；请先运行 scripts/setup_env.sh"
    UV_BIN="$(command -v uv)"
}

configure_paths() {
    HF_DATASET_CACHE_DIR="${HF_HOME}/datasets"
    V30_ROOT="${FRS_STORAGE_ROOT}/lerobot_v30"
    V30_WORK_ROOT="${FRS_STORAGE_ROOT}/lerobot_v30_work"
    LOCK_ROOT="${FRS_STORAGE_ROOT}/.locks"
    mkdir -p \
        "${HF_DATASET_CACHE_DIR}" \
        "${V30_ROOT}/${HF_NAMESPACE}" \
        "${V30_WORK_ROOT}/${HF_NAMESPACE}" \
        "${LOCK_ROOT}"
}

acquire_download_lock() {
    command -v flock >/dev/null 2>&1 || fail "找不到 flock（util-linux）"
    exec 9>"${LOCK_ROOT}/frs-download-data.lock"
    flock -n 9 || fail "另一个 FRS 数据下载正在运行"
}

repo_id_for() {
    printf '%s/%s\n' "${HF_NAMESPACE}" "$1"
}

source_cache_for() {
    printf '%s/datasets--%s--%s\n' "${HF_DATASET_CACHE_DIR}" "${HF_NAMESPACE}" "$1"
}

final_root_for() {
    printf '%s/%s/%s\n' "${V30_ROOT}" "${HF_NAMESPACE}" "$1"
}

work_root_for() {
    printf '%s/%s/%s\n' "${V30_WORK_ROOT}" "${HF_NAMESPACE}" "$1"
}

dataset_version() {
    "${PROJECT_PYTHON}" - "$1" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1]) / "meta" / "info.json"
if not path.is_file():
    print("unknown")
else:
    try:
        print(json.loads(path.read_text(encoding="utf-8")).get("codebase_version", "unknown"))
    except (OSError, ValueError, TypeError):
        print("unknown")
PY
}

latest_snapshot_for() {
    local dataset_name="$1"
    local snapshots_root
    snapshots_root="$(source_cache_for "${dataset_name}")/snapshots"
    [[ -d "${snapshots_root}" ]] || return 0
    find "${snapshots_root}" -mindepth 1 -maxdepth 1 -type d -printf '%T@ %p\n' \
        | sort -nr \
        | head -n 1 \
        | cut -d' ' -f2-
}

clear_conversion_paths() {
    local dataset_name="$1"
    local work_root
    work_root="$(work_root_for "${dataset_name}")"
    # These are fixed derived paths for one of the four known repositories.
    rm -rf -- "${work_root}" "${work_root}_old" "${work_root}_v30"
}

copy_snapshot_to_work() {
    local dataset_name="$1"
    local snapshot_root="$2"
    local work_root
    work_root="$(work_root_for "${dataset_name}")"
    clear_conversion_paths "${dataset_name}"
    mkdir -p "${work_root}"
    cp -a -- "${snapshot_root}/." "${work_root}/"
}

convert_v21_to_v30() {
    local dataset_name="$1"
    local repo_id
    local work_root
    repo_id="$(repo_id_for "${dataset_name}")"
    work_root="$(work_root_for "${dataset_name}")"

    log "使用官方 LeRobot converter：${repo_id}"
    if ! "${UV_BIN}" run --no-sync python \
        -m lerobot.datasets.v30.convert_dataset_v21_to_v30 \
        --repo-id="${repo_id}" \
        --root="${work_root}" \
        --push-to-hub=false \
        --force-conversion
    then
        fail "${repo_id} v2.1 → v3.0 转换失败；源 snapshot 已保留"
    fi
}

promote_work_to_final() {
    local dataset_name="$1"
    local work_root
    local final_root
    work_root="$(work_root_for "${dataset_name}")"
    final_root="$(final_root_for "${dataset_name}")"
    [[ "$(dataset_version "${work_root}")" == "v3.0" ]] || \
        fail "转换结果不是 v3.0：${work_root}"

    # final_root is a fixed derived repository root, never a caller-provided path.
    rm -rf -- "${final_root}"
    mv -- "${work_root}" "${final_root}"
}

download_and_prepare_dataset() {
    local dataset_name="$1"
    local repo_id
    local final_root
    local snapshot_root
    local source_version
    repo_id="$(repo_id_for "${dataset_name}")"
    final_root="$(final_root_for "${dataset_name}")"
    DATASET_ALREADY_VALIDATED=0

    if [[ -d "${final_root}" && ! -L "${final_root}" ]] \
        && [[ "$(dataset_version "${final_root}")" == "v3.0" ]]
    then
        if validate_dataset "${dataset_name}"; then
            DATASET_ALREADY_VALIDATED=1
            log "复用现有且验证有效的 v3 数据集：${repo_id}"
            return 0
        fi
        log "现有 v3 数据集验证失败，将从保留的 snapshot 重建：${repo_id}"
    fi

    log "下载数据集：${repo_id}"
    "${UV_BIN}" run --no-sync hf download "${repo_id}" \
        --repo-type dataset \
        --cache-dir "${HF_DATASET_CACHE_DIR}"

    snapshot_root="$(latest_snapshot_for "${dataset_name}")"
    [[ -n "${snapshot_root}" && -d "${snapshot_root}" ]] || \
        fail "下载后未找到 ${repo_id} snapshot"
    source_version="$(dataset_version "${snapshot_root}")"
    case "${source_version}" in
        v3.0)
            copy_snapshot_to_work "${dataset_name}" "${snapshot_root}"
            ;;
        v2.1)
            copy_snapshot_to_work "${dataset_name}" "${snapshot_root}"
            convert_v21_to_v30 "${dataset_name}"
            ;;
        *)
            fail "${repo_id} 版本为 ${source_version@Q}；只支持 v2.1 或 v3.0"
            ;;
    esac
    promote_work_to_final "${dataset_name}"
}

validate_dataset() {
    local dataset_name="$1"
    local repo_id
    local final_root
    repo_id="$(repo_id_for "${dataset_name}")"
    final_root="$(final_root_for "${dataset_name}")"

    "${PROJECT_PYTHON}" - "${repo_id}" "${final_root}" <<'PY' || return 1
import json
import math
import sys
from pathlib import Path

from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata


repo_id = sys.argv[1]
root = Path(sys.argv[2])


def fail(message: str) -> None:
    raise ValueError(f"{repo_id}: {message}")


def numeric_leaves(value):
    if isinstance(value, list):
        for item in value:
            yield from numeric_leaves(item)
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        yield float(value)
    else:
        fail(f"统计值包含非数值：{value!r}")


info_path = root / "meta" / "info.json"
stats_path = root / "meta" / "stats.json"
if not info_path.is_file():
    fail("缺少 meta/info.json")
if not stats_path.is_file():
    fail("缺少 meta/stats.json")
info = json.loads(info_path.read_text(encoding="utf-8"))
stats = json.loads(stats_path.read_text(encoding="utf-8"))

if info.get("codebase_version") != "v3.0":
    fail(f"codebase_version 必须为 v3.0，实际为 {info.get('codebase_version')!r}")
if not isinstance(info.get("total_episodes"), int) or info["total_episodes"] < 1:
    fail("total_episodes 必须为正整数")
if not isinstance(info.get("total_frames"), int) or info["total_frames"] < 1:
    fail("total_frames 必须为正整数")

features = info.get("features")
if not isinstance(features, dict):
    fail("features 必须为对象")


def require_feature(key: str, *, width: int | None = None, visual: bool = False) -> None:
    feature = features.get(key)
    if not isinstance(feature, dict):
        fail(f"缺少 feature {key}")
    if visual and feature.get("dtype") not in {"image", "video"}:
        fail(f"{key} 必须是 image/video")
    if width is not None and feature.get("shape") != [width]:
        fail(f"{key} shape 必须为 [{width}]，实际为 {feature.get('shape')!r}")


# The training rename map is camera0 -> camera1 and camera1 -> camera2.
for key in ("observation.images.camera0", "observation.images.camera1"):
    require_feature(key, visual=True)
for key in (
    "observation.images.tactile_left_0",
    "observation.images.tactile_right_0",
    "observation.images.tactile_left_1",
    "observation.images.tactile_right_1",
):
    require_feature(key, visual=True)
require_feature("observation.state", width=20)
require_feature("actions", width=20)
for key in ("index", "frame_index", "episode_index"):
    require_feature(key, width=1)

for key in ("observation.state", "actions"):
    feature_stats = stats.get(key)
    if not isinstance(feature_stats, dict):
        fail(f"缺少 {key} normalization stats")
    for stat_name in ("min", "max", "mean", "std", "count"):
        if stat_name not in feature_stats:
            fail(f"{key} stats 缺少 {stat_name}")
        values = list(numeric_leaves(feature_stats[stat_name]))
        if not values or not all(math.isfinite(value) for value in values):
            fail(f"{key}.{stat_name} 必须为有限数值")
        if stat_name in {"min", "max", "mean", "std"} and len(values) != 20:
            fail(f"{key}.{stat_name} 宽度必须为 20")
        if stat_name == "count" and any(value <= 0 for value in values):
            fail(f"{key}.count 必须为正数")

metadata = LeRobotDatasetMetadata(repo_id=repo_id, root=root)
if metadata.total_episodes < 1:
    fail("LeRobotDatasetMetadata 未解析到 episode")
dataset = LeRobotDataset(repo_id=repo_id, root=root, episodes=[0])
sample = dataset[0]
for key in ("observation.state", "actions", "index", "frame_index", "episode_index"):
    if key not in sample:
        fail(f"sample zero 缺少 {key}")
PY
    log "数据集验证通过：${repo_id}"
}

resolve_encoder_path() {
    "${PROJECT_PYTHON}" - \
        "${PROJECT_ROOT}/configs/train_vtsmolvla_jax_tactile16.yaml" \
        "${PROJECT_ROOT}/configs/train_vtsmolvla_jax_tactile32.yaml" <<'PY'
import ast
import sys
from pathlib import Path


paths = []
for config_path in map(Path, sys.argv[1:]):
    in_model = False
    raw_path = None
    for line in config_path.read_text(encoding="utf-8").splitlines():
        content = line.split("#", 1)[0].rstrip()
        if not content:
            continue
        indentation = len(content) - len(content.lstrip())
        stripped = content.strip()
        if indentation == 0:
            in_model = stripped == "model:"
            continue
        if in_model and indentation > 0 and stripped.startswith("tactile_encoder_path:"):
            raw_path = stripped.split(":", 1)[1].strip()
            if raw_path[:1] in {'"', "'"}:
                raw_path = ast.literal_eval(raw_path)
            break
    if raw_path is None:
        raise ValueError(f"{config_path} 缺少 model.tactile_encoder_path")
    path = Path(str(raw_path)).expanduser()
    if not path.is_absolute():
        raise ValueError(f"{config_path} 的 tactile_encoder_path 必须是绝对路径")
    paths.append(path)
if len(set(paths)) != 1:
    raise ValueError(f"K8/K21 tactile_encoder_path 不一致：{paths}")
print(paths[0])
PY
}

validate_encoder() {
    local encoder_root="$1"
    "${PROJECT_PYTHON}" - "${encoder_root}" <<'PY'
import json
import sys
from pathlib import Path


root = Path(sys.argv[1])
checkpoint_path = root / "checkpoint.json"
if not checkpoint_path.is_file():
    raise FileNotFoundError(f"缺少 {checkpoint_path}")
metadata = json.loads(checkpoint_path.read_text(encoding="utf-8"))
params_file = metadata.get("params_file")
if not isinstance(params_file, str) or not params_file:
    raise ValueError(f"{checkpoint_path} 缺少 params_file")
relative = Path(params_file)
if relative.is_absolute() or ".." in relative.parts:
    raise ValueError(f"params_file 必须位于 encoder 目录内：{params_file!r}")
params_path = root / relative
if not params_path.is_file():
    raise FileNotFoundError(f"缺少 checkpoint 声明的参数文件：{params_path}")
PY
}

cleanup_successful_sources() {
    local dataset_name
    local source_cache
    local work_root
    for dataset_name in "${DATASETS[@]}"; do
        source_cache="$(source_cache_for "${dataset_name}")"
        work_root="$(work_root_for "${dataset_name}")"
        rm -rf -- \
            "${source_cache}" \
            "${work_root}" \
            "${work_root}_old" \
            "${work_root}_v30"
    done
    log "已清理四个已验证数据集的源 snapshot cache 和转换残留"
}

main() {
    parse_arguments "$@"
    require_environment
    configure_paths
    acquire_download_lock
    cd "${PROJECT_ROOT}"

    local dataset_name
    local encoder_root
    for dataset_name in "${DATASETS[@]}"; do
        download_and_prepare_dataset "${dataset_name}"
        if ((!DATASET_ALREADY_VALIDATED)); then
            validate_dataset "${dataset_name}"
        fi
    done

    encoder_root="$(resolve_encoder_path)"
    bash "${PROJECT_ROOT}/scripts/download_ckpt.sh"
    validate_encoder "${encoder_root}"
    log "最小 tactile encoder 验证通过：${encoder_root}"

    if ((CLEANUP_SOURCE)); then
        cleanup_successful_sources
    fi
    log "四个 v3 数据集与 encoder_05 均已准备完成"
}

main "$@"
