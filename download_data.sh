#!/usr/bin/env bash

set -euo pipefail

# ===================== 配置区域 =====================
HF_NAMESPACE="KaiyueChen"
# LeRobot 本地缓存 / v3.0 转换路径用的命名空间（原从 openpi config.py 的 DATASET_REPO_NAMESPACE 读取）
# LEROBOT_NAMESPACE="${LEROBOT_NAMESPACE:-${HF_NAMESPACE}}"
LEROBOT_NAMESPACE="chaoyi"
# HF_DATASET_CACHE_DIR="${HF_DATASET_CACHE_DIR:-${HOME}/.cache/huggingface/dataset}" # for server
HF_DATASET_CACHE_DIR="${HF_DATASET_CACHE_DIR:-/workspace}" # for runpods
if [[ "${BASH_SOURCE[0]}" == "$0" && -n "${1:-}" ]]; then
    HF_DATASET_CACHE_DIR="$1"
fi
# ====================================================

DATASETS=(
    # white_smash_01
    # white_smash_03
    # white_smash_04
    # white_smash_05
    # white_smash_06
    # white_smash_07
    # yellow_smash_01
    # yellow_smash_02
    # yellow_smash_03
    # yellow_smash_05
    # yellow_smash_06
    # yellow_smash_07
    # black_smash_01
    # black_smash_02
    # black_smash_03
    # black_smash_04
    # black_smash_06
    # black_smash_07
    # tactile_test_02
    tactile_test_03
)

log() {
    # 打到 stderr，避免被 $(...) 命令替换吞掉
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" >&2
}

load_lerobot_config() {
    LEROBOT_CACHE_DIR="${HOME}/.cache/huggingface/lerobot/${LEROBOT_NAMESPACE}"
    # 最终 v3.0 产物目录（可在 /workspace 网络盘上，供软链接指向）
    V30_CONVERT_ROOT="${HF_DATASET_CACHE_DIR}/lerobot_v30"
    # 转换工作目录：默认放容器本地盘，避免 MooseFS 上 to_parquet close 失败
    # 可通过环境变量覆盖，例如: V30_CONVERT_WORK_ROOT=/tmp/lerobot_v30_work
    V30_CONVERT_WORK_ROOT="${V30_CONVERT_WORK_ROOT:-${HOME}/.cache/lerobot_v30_work}"
}

check_deps() {
    if ! command -v uv &>/dev/null; then
        echo "=========================================="
        echo " 未检测到 uv，请先安装 uv:"
        echo "   curl -LsSf https://astral.sh/uv/install.sh | sh"
        echo ""
        echo " 然后在项目环境中安装依赖:"
        echo "   uv add huggingface_hub lerobot"
        echo "=========================================="
        exit 1
    fi

    if ! uv run hf version &>/dev/null; then
        echo "=========================================="
        echo " uv 环境中未检测到 hf 命令，请执行:"
        echo "   uv add huggingface_hub"
        echo ""
        echo " 安装后如需登录，请执行:"
        echo "   uv run hf auth login"
        echo "=========================================="
        exit 1
    fi

    if ! uv run python -c "import lerobot" &>/dev/null; then
        echo "=========================================="
        echo " uv 环境中未检测到 lerobot，请执行:"
        echo "   uv add lerobot"
        echo "=========================================="
        exit 1
    fi
}

get_dataset_version() {
    local dataset_dir="$1"
    local info_json="${dataset_dir}/meta/info.json"

    if [[ ! -f "$info_json" ]]; then
        echo "unknown"
        return 0
    fi

    python - "$info_json" <<'PY'
import json
import sys

with open(sys.argv[1], "r", encoding="utf-8") as f:
    info = json.load(f)
print(info.get("codebase_version", "unknown"))
PY
}

create_lerobot_symlink() {
    local dataset_name="$1"
    local source_dir="$2"
    local link_path="${LEROBOT_CACHE_DIR}/${dataset_name}"
    local source_abs

    if [[ -z "$source_dir" || "$source_dir" == *$'\n'* ]]; then
        log "错误: 软链接源路径非法: ${source_dir@Q}"
        return 1
    fi
    if [[ ! -e "$source_dir" ]]; then
        log "错误: 软链接源不存在: ${source_dir}"
        return 1
    fi

    source_abs="$(realpath "$source_dir")"

    mkdir -p "$LEROBOT_CACHE_DIR"

    if [[ -L "$link_path" ]]; then
        rm -f "$link_path"
    elif [[ -e "$link_path" ]]; then
        log "警告: 目标已存在且不是软链接，跳过: $link_path"
        return 0
    fi

    ln -s "$source_abs" "$link_path"
    log "已创建软链接 ${link_path} -> ${source_abs}"
}

get_snapshot_dir() {
    local dataset_name="$1"
    local repo_cache_dir="${HF_DATASET_CACHE_DIR}/datasets--${HF_NAMESPACE//\//--}--${dataset_name}"
    local snapshots_dir="${repo_cache_dir}/snapshots"
    local latest_snapshot=""

    if [[ ! -d "$snapshots_dir" ]]; then
        echo ""
        return 0
    fi

    latest_snapshot="$(ls -1dt "${snapshots_dir}"/* 2>/dev/null | head -n 1 || true)"
    echo "$latest_snapshot"
}

get_v30_dataset_dir() {
    local dataset_name="$1"
    echo "${V30_CONVERT_ROOT}/${LEROBOT_NAMESPACE}/${dataset_name}"
}

get_v30_work_dataset_dir() {
    local dataset_name="$1"
    echo "${V30_CONVERT_WORK_ROOT}/${LEROBOT_NAMESPACE}/${dataset_name}"
}

get_hf_repo_cache_dir() {
    local dataset_name="$1"
    echo "${HF_DATASET_CACHE_DIR}/datasets--${HF_NAMESPACE//\//--}--${dataset_name}"
}

# 搬迁成功后清理本地盘上的转换残留（工作目录 / _old / _v30）
cleanup_local_convert_work() {
    local dataset_name="$1"
    local work_dir
    local old_backup
    local leftover_v30
    local ns_dir
    local before_kb=""
    local after_kb=""

    work_dir="$(get_v30_work_dataset_dir "$dataset_name")"
    old_backup="${V30_CONVERT_WORK_ROOT}/${LEROBOT_NAMESPACE}/${dataset_name}_old"
    leftover_v30="${V30_CONVERT_WORK_ROOT}/${LEROBOT_NAMESPACE}/${dataset_name}_v30"
    ns_dir="${V30_CONVERT_WORK_ROOT}/${LEROBOT_NAMESPACE}"

    before_kb="$(df -Pk "$V30_CONVERT_WORK_ROOT" 2>/dev/null | awk 'NR==2{print $4}')"

    log "清理本地盘转换残留:"
    for p in "$work_dir" "$old_backup" "$leftover_v30"; do
        if [[ -e "$p" || -L "$p" ]]; then
            log "  删除: ${p}"
            rm -rf "$p"
        fi
    done

    # 命名空间目录若已空则一并去掉
    if [[ -d "$ns_dir" ]] && [[ -z "$(ls -A "$ns_dir" 2>/dev/null)" ]]; then
        rmdir "$ns_dir" 2>/dev/null || true
    fi
    if [[ -d "$V30_CONVERT_WORK_ROOT" ]] && [[ -z "$(ls -A "$V30_CONVERT_WORK_ROOT" 2>/dev/null)" ]]; then
        rmdir "$V30_CONVERT_WORK_ROOT" 2>/dev/null || true
    fi

    after_kb="$(df -Pk / 2>/dev/null | awk 'NR==2{print $4}')"
    if [[ -n "$before_kb" && -n "$after_kb" && "$after_kb" -ge "$before_kb" ]]; then
        log "本地盘清理完成，可用空间约 $((after_kb / 1024 / 1024))G（清理前约 $((before_kb / 1024 / 1024))G）"
    else
        log "本地盘清理完成"
    fi
}

# 按文件拷贝到 MooseFS：单文件超时则杀进程重试，避免整次 rsync 卡死
rsync_to_workspace_resilient() {
    local src_dir="$1"
    local dst_dir="$2"
    local timeout_sec="${3:-120}"
    local retries="${4:-3}"
    local rel=""
    local src_file=""
    local dst_file=""
    local attempt=0
    local n_total=0
    local n_done=0

    mkdir -p "$dst_dir"
    mapfile -t files < <(cd "$src_dir" && find . -type f | sed 's|^\./||' | sort)
    n_total=${#files[@]}
    log "开始按文件 rsync 到 workspace（共 ${n_total} 个，单文件超时 ${timeout_sec}s，重试 ${retries} 次）"

    for rel in "${files[@]}"; do
        src_file="${src_dir}/${rel}"
        dst_file="${dst_dir}/${rel}"
        mkdir -p "$(dirname "$dst_file")"

        # 已存在且大小一致则跳过（支持断点续传）
        if [[ -f "$dst_file" ]] && [[ "$(stat -c%s "$dst_file")" == "$(stat -c%s "$src_file")" ]]; then
            n_done=$((n_done + 1))
            continue
        fi

        attempt=0
        while true; do
            attempt=$((attempt + 1))
            rm -f "$dst_file"
            if timeout "${timeout_sec}s" rsync -a --partial "$src_file" "$dst_file"; then
                if [[ -f "$dst_file" ]] && [[ "$(stat -c%s "$dst_file")" == "$(stat -c%s "$src_file")" ]]; then
                    break
                fi
                log "警告: ${rel} 拷贝后大小不一致，重试 (${attempt}/${retries})"
            else
                log "警告: ${rel} 拷贝超时/失败，重试 (${attempt}/${retries})"
                # 杀掉可能残留的卡住 rsync
                pkill -9 -f "rsync -a --partial ${src_file}" 2>/dev/null || true
                rm -f "$dst_file"
            fi
            if [[ "$attempt" -ge "$retries" ]]; then
                log "错误: ${rel} 连续 ${retries} 次失败，中止搬迁（已完成 ${n_done}/${n_total}）"
                return 1
            fi
            sleep 2
        done

        n_done=$((n_done + 1))
        if (( n_done % 10 == 0 || n_done == n_total )); then
            log "搬迁进度: ${n_done}/${n_total}"
        fi
    done

    # 同步空目录结构（若有）
    (cd "$src_dir" && find . -type d | sed 's|^\./||') | while read -r d; do
        [[ -z "$d" || "$d" == "." ]] && continue
        mkdir -p "${dst_dir}/${d}"
    done

    log "按文件 rsync 完成: ${n_done}/${n_total}"
}

# 本地转换成功后：先删 workspace 旧版，再把 v3.0 挪过去，避免新旧各占一份爆盘
# 注意：跨盘禁止用 mv / 整包 rsync（MooseFS 易卡在 request_wait_answer）
promote_v30_to_workspace() {
    local dataset_name="$1"
    local work_dir
    local final_dir
    local hf_cache_dir
    work_dir="$(get_v30_work_dataset_dir "$dataset_name")"
    final_dir="$(get_v30_dataset_dir "$dataset_name")"
    hf_cache_dir="$(get_hf_repo_cache_dir "$dataset_name")"

    # 1) 确认本地工作目录已是独立的 v3.0（不再依赖 snapshot 软链）
    if [[ -L "$work_dir" ]]; then
        log "错误: 工作目录仍是软链，不能删除 workspace 旧版: ${work_dir}"
        return 1
    fi
    if [[ "$(get_dataset_version "$work_dir")" != "v3.0" ]]; then
        log "错误: 工作目录不是 v3.0，拒绝删旧搬迁: ${work_dir}"
        return 1
    fi

    # 先链到本地，保证即使搬迁失败也能训练
    create_lerobot_symlink "$dataset_name" "$work_dir"

    # 2) 删除 workspace 上的旧版本（HF 下载缓存）；最终目录若完整 v3 则跳过搬迁
    if [[ -e "$hf_cache_dir" ]]; then
        log "删除 workspace 旧版 HF cache: ${hf_cache_dir}"
        rm -rf "$hf_cache_dir"
    fi
    rm -rf "${final_dir}_old" "${final_dir}_v30" "${final_dir}_xfer" 2>/dev/null || true

    if [[ -d "$final_dir" && ! -L "$final_dir" && "$(get_dataset_version "$final_dir")" == "v3.0" ]]; then
        log "workspace 最终目录已是 v3.0，跳过搬迁: ${final_dir}"
        create_lerobot_symlink "$dataset_name" "$final_dir"
        cleanup_local_convert_work "$dataset_name"
        return 0
    fi

    if ! command -v rsync &>/dev/null; then
        log "错误: 未找到 rsync"
        return 1
    fi
    if ! command -v timeout &>/dev/null; then
        log "错误: 未找到 timeout（coreutils）"
        return 1
    fi

    # 3) 按文件拷到 workspace（可续传）；成功后再切软链、清理本地盘
    local xfer_dir="${final_dir}_xfer"
    mkdir -p "$(dirname "$final_dir")"
    log "将 v3.0 按文件搬迁到 workspace:"
    log "  ${work_dir}/ -> ${xfer_dir}/"
    if ! rsync_to_workspace_resilient "$work_dir" "$xfer_dir" 180 3; then
        log "错误: 搬迁失败。本地 v3.0 仍保留，软链仍指向: ${work_dir}"
        return 1
    fi

    if [[ "$(get_dataset_version "$xfer_dir")" != "v3.0" ]]; then
        log "错误: 搬迁后版本异常: ${xfer_dir}"
        return 1
    fi

    rm -rf "$final_dir"
    mv "$xfer_dir" "$final_dir"
    create_lerobot_symlink "$dataset_name" "$final_dir"

    # 搬迁成功后清理本地硬盘上的转换产物
    cleanup_local_convert_work "$dataset_name"
    log "已搬迁到最终目录: ${final_dir}"
}

link_snapshot_for_convert() {
    local snapshot_dir="$1"
    local target_dir="$2"
    local snapshot_abs

    snapshot_abs="$(realpath "$snapshot_dir")"
    mkdir -p "$(dirname "$target_dir")"
    rm -rf "$target_dir"
    # 不整包复制：软链到 HF snapshot，转换脚本读这里、把 v3.0 写到旁路目录
    ln -s "$snapshot_abs" "$target_dir"
}

# 检查 snapshot 内 parquet 是否完整；返回损坏文件列表（每行一个绝对路径）
find_corrupt_parquets() {
    local dataset_dir="$1"
    python - "$dataset_dir" <<'PY'
from pathlib import Path
import sys

root = Path(sys.argv[1])
bad = []
for path in sorted((root / "data").glob("*/*.parquet")):
    try:
        size = path.stat().st_size
        if size < 8:
            bad.append(path.resolve())
            continue
        with open(path, "rb") as f:
            head = f.read(4)
            f.seek(-4, 2)
            tail = f.read(4)
        if head != b"PAR1" or tail != b"PAR1":
            bad.append(path.resolve())
    except OSError:
        bad.append(path.resolve())
for p in bad:
    print(p)
PY
}

repair_corrupt_parquets() {
    local dataset_name="$1"
    local snapshot_dir="$2"
    local repo_id="${HF_NAMESPACE}/${dataset_name}"
    local bad_files=()
    local blob_path=""
    local rel_path=""
    local line=""

    mapfile -t bad_files < <(find_corrupt_parquets "$snapshot_dir" || true)
    if [[ ${#bad_files[@]} -eq 0 ]]; then
        return 0
    fi

    log "发现 ${#bad_files[@]} 个损坏的 parquet（常见于下载中断），将删除后强制重下:"
    for line in "${bad_files[@]}"; do
        [[ -z "$line" ]] && continue
        log "  - ${line} ($(stat -c%s "$line" 2>/dev/null || echo '?') bytes)"
        blob_path="$(readlink -f "$line")"
        # 删 blob，让 hf download 重新拉取
        rm -f "$blob_path" "$line"
    done

    log "重新下载缺失文件: ${repo_id}"
    # stdout 重定向，避免污染外层 $(...) 路径捕获
    uv run hf download "$repo_id" \
        --repo-type dataset \
        --cache-dir "$HF_DATASET_CACHE_DIR" >&2

    mapfile -t bad_files < <(find_corrupt_parquets "$snapshot_dir" || true)
    if [[ ${#bad_files[@]} -gt 0 ]]; then
        log "错误: 重下后仍有损坏 parquet:"
        printf '  %s\n' "${bad_files[@]}" >&2
        return 1
    fi
    log "损坏 parquet 已修复"
}

upgrade_dataset_to_v30() {
    local dataset_name="$1"
    local snapshot_dir="$2"
    local repo_id="${LEROBOT_NAMESPACE}/${dataset_name}"
    local final_dir
    local work_dir
    local old_backup
    local leftover_v30
    local version=""
    local avail_kb=""

    final_dir="$(get_v30_dataset_dir "$dataset_name")"
    work_dir="$(get_v30_work_dataset_dir "$dataset_name")"
    old_backup="${V30_CONVERT_WORK_ROOT}/${LEROBOT_NAMESPACE}/${dataset_name}_old"
    leftover_v30="${V30_CONVERT_WORK_ROOT}/${LEROBOT_NAMESPACE}/${dataset_name}_v30"

    # 最终目录已是真实 v3.0 则跳过
    if [[ -d "$final_dir" && ! -L "$final_dir" ]]; then
        version="$(get_dataset_version "$final_dir")"
        if [[ "$version" == "v3.0" ]]; then
            log "本地已存在 v3.0 数据集，跳过升级: ${final_dir}"
            cleanup_local_convert_work "$dataset_name"
            UPGRADED_DATASET_DIR="$final_dir"
            return 0
        fi
    fi

    version="$(get_dataset_version "$snapshot_dir")"
    if [[ "$version" == "v3.0" ]]; then
        log "下载产物已是 v3.0，直接作为升级结果: ${snapshot_dir}"
        rm -rf "$old_backup" "$leftover_v30" "$work_dir"
        UPGRADED_DATASET_DIR="$snapshot_dir"
        return 0
    fi

    if [[ "$version" != "v2.1" ]]; then
        log "错误: 期望 v2.1 数据集以升级到 v3.0，实际版本为 '${version}': ${snapshot_dir}"
        return 1
    fi

    repair_corrupt_parquets "$dataset_name" "$snapshot_dir"

    # 容器本地盘至少留约 50G，避免写到一半空间不足
    avail_kb="$(df -Pk "$V30_CONVERT_WORK_ROOT" 2>/dev/null | awk 'NR==2{print $4}')"
    if [[ -n "$avail_kb" && "$avail_kb" -lt 52428800 ]]; then
        log "警告: 转换工作盘可用空间约 $((avail_kb / 1024 / 1024))G，建议 >= 50G（目录: ${V30_CONVERT_WORK_ROOT}）"
    fi

    log "准备本地升级到 v3.0: ${repo_id}"
    log "  源 snapshot: ${snapshot_dir}"
    log "  转换工作目录(本地盘): ${work_dir}"
    log "  最终目录: ${final_dir}"

    # 清理残留；在本地盘工作目录软链接入 snapshot，转换输出也写本地盘
    mkdir -p "${V30_CONVERT_WORK_ROOT}/${LEROBOT_NAMESPACE}"
    rm -rf "$old_backup" "$leftover_v30" "$work_dir"
    link_snapshot_for_convert "$snapshot_dir" "$work_dir"

    # 仅本地升级：指定 --root，跳过 hub 上 v3.0 检索；不上传
    # 写在本地盘，规避 /workspace(MooseFS) 上 ParquetWriter.close OSError
    # convert 的 print 走 stderr，避免污染路径变量
    if ! uv run python -m lerobot.datasets.v30.convert_dataset_v21_to_v30 \
        --repo-id="$repo_id" \
        --root="$work_dir" \
        --push-to-hub=false \
        --force-conversion >&2
    then
        log "错误: v2.1 → v3.0 转换失败（数据集仍是原来的 v2.1，并未升级成功）"
        rm -rf "$leftover_v30"
        if [[ -L "$work_dir" ]]; then
            rm -f "$work_dir"
        fi
        return 1
    fi

    # *_old 此时只是指向 snapshot 的软链，删掉不影响 HF cache
    if [[ -e "$old_backup" ]]; then
        log "删除旧版本入口: ${old_backup}"
        rm -rf "$old_backup"
    fi
    rm -rf "$leftover_v30"

    version="$(get_dataset_version "$work_dir")"
    if [[ "$version" != "v3.0" ]]; then
        log "错误: 转换命令已返回成功，但目录版本为 '${version}'（期望 v3.0）: ${work_dir}"
        return 1
    fi

    # 本地转完 → 删 workspace 旧版 → 再挪到 workspace
    promote_v30_to_workspace "$dataset_name"

    version="$(get_dataset_version "$final_dir")"
    if [[ "$version" != "v3.0" ]]; then
        log "错误: 搬迁到最终目录后版本为 '${version}'（期望 v3.0）: ${final_dir}"
        return 1
    fi

    log "升级完成: ${final_dir} (v3.0)"
    UPGRADED_DATASET_DIR="$final_dir"
}

download_dataset() {
    local dataset_name="$1"
    local repo_id="${HF_NAMESPACE}/${dataset_name}"
    local snapshot_dir=""
    local upgraded_dir=""
    local final_dir=""
    local version=""

    final_dir="$(get_v30_dataset_dir "$dataset_name")"
    if [[ -d "$final_dir" && ! -L "$final_dir" ]]; then
        version="$(get_dataset_version "$final_dir")"
        if [[ "$version" == "v3.0" ]]; then
            log "最终目录已是 v3.0，跳过下载与转换: ${final_dir}"
            create_lerobot_symlink "$dataset_name" "$final_dir"
            return 0
        fi
    fi

    log "开始下载 ${repo_id} (cache: ${HF_DATASET_CACHE_DIR})"

    uv run hf download "$repo_id" \
        --repo-type dataset \
        --cache-dir "$HF_DATASET_CACHE_DIR" >&2

    snapshot_dir="$(get_snapshot_dir "$dataset_name")"
    if [[ -z "$snapshot_dir" ]]; then
        log "错误: 未找到 snapshot 目录，期望路径: ${HF_DATASET_CACHE_DIR}/datasets--${HF_NAMESPACE//\//--}--${dataset_name}/snapshots/*"
        return 1
    fi

    # 通过 UPGRADED_DATASET_DIR 传回路径，避免 $(...) 吞掉 convert/hf 的 stdout
    UPGRADED_DATASET_DIR=""
    upgrade_dataset_to_v30 "$dataset_name" "$snapshot_dir"
    upgraded_dir="${UPGRADED_DATASET_DIR}"
    if [[ -z "$upgraded_dir" || ! -e "$upgraded_dir" ]]; then
        log "错误: 升级后未得到有效目录"
        return 1
    fi
    create_lerobot_symlink "$dataset_name" "$upgraded_dir"
    log "完成下载与升级 ${repo_id}"
}

main() {
    check_deps
    load_lerobot_config
    mkdir -p "$HF_DATASET_CACHE_DIR"
    mkdir -p "$V30_CONVERT_ROOT"
    mkdir -p "$V30_CONVERT_WORK_ROOT"

    log "============================================"
    log "Hugging Face smash 数据集下载脚本启动"
    log "命名空间: ${HF_NAMESPACE}"
    log "HF dataset cache: ${HF_DATASET_CACHE_DIR}"
    log "LeRobot namespace: ${LEROBOT_NAMESPACE}"
    log "LeRobot 链接目录: ${LEROBOT_CACHE_DIR}"
    log "v3.0 最终目录: ${V30_CONVERT_ROOT}"
    log "v3.0 转换工作目录(本地盘): ${V30_CONVERT_WORK_ROOT}"
    log "数据集数量: ${#DATASETS[@]}"
    log "============================================"

    for dataset_name in "${DATASETS[@]}"; do
        download_dataset "$dataset_name"
    done

    log "全部数据集下载并升级完成"
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    main "$@"
fi
