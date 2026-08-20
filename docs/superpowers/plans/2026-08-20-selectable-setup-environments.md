# Selectable Setup Environments Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `--root` and `--pi05_deploy` selectors to `scripts/setup_env.sh` so callers can install and verify one uv environment, while no arguments keep installing both.

**Architecture:** Parse and validate the selector before any side effect, store it in `SETUP_MODE`, and use two small predicates to gate root and Pi0.5 sync/verification work. Keep path selection in the existing `FRS_VENV_DIR` and `PI05_VENV_DIR` variables, retain the complete `.env.frs` contract, and split the current combined synchronization and GPU checks into target-specific functions.

**Tech Stack:** Bash 5, uv, pytest, Python 3.12.

## Global Constraints

- `bash scripts/setup_env.sh` must remain equivalent to installing root plus Pi0.5 deployment.
- `bash scripts/setup_env.sh --root` must install and verify only the root environment.
- `bash scripts/setup_env.sh --pi05_deploy` must install and verify only the Pi0.5 deployment environment.
- `-h` and `--help` must print usage and exit zero without side effects.
- Unknown arguments and multiple selectors must fail before any installation or filesystem mutation.
- `FRS_VENV_DIR`, `PI05_VENV_DIR`, platform defaults, relative-path resolution, and distinct-directory validation must remain compatible.
- Root synchronization must use the repository `uv.lock`; Pi0.5 synchronization must use `deploy_pi05/uv.lock` with `--project deploy_pi05`.
- `.env.frs` must continue recording both configured environment paths.
- Preserve the user's unrelated modification to `train_smolvla_frs/configs/train_frs_bimanual_gated.yaml`.

---

### Task 1: Add selector parsing and target-specific setup execution

**Files:**
- Modify: `tests/test_setup_env_dual_environment.py`
- Modify: `scripts/setup_env.sh:27-342`

**Interfaces:**
- Consumes: Existing `VENV_DIR`, `PI05_VENV_DIR`, `UV_BIN`, `ENV_FILE`, and verification functions in `scripts/setup_env.sh`.
- Produces: `SETUP_MODE=all|root|pi05_deploy`, `parse_args "$@"`, `should_setup_root`, `should_setup_pi05`, `sync_root_environment`, `sync_pi05_environment`, `check_root_gpu`, and `check_pi05_gpu`.

- [ ] **Step 1: Add failing entry-point tests for selectors, compatibility, help, and errors**

Append helpers and tests to `tests/test_setup_env_dual_environment.py`. The helper sources the real script, replaces every side-effecting function used by `main` with an event logger, and invokes `main` in a fresh Bash process:

```python
import shlex


def run_stubbed_main(tmp_path: Path, *args: str) -> subprocess.CompletedProcess[str]:
    events = tmp_path / "events.log"
    quoted_args = " ".join(shlex.quote(arg) for arg in args)
    command = f"""
set -euo pipefail
source {SETUP}
EVENTS={events}
record() {{ printf '%s\n' "$1" >>"$EVENTS"; }}
install_system_dependencies() {{ record system; }}
install_uv() {{ UV_BIN=uv; record uv; }}
persist_uv_path() {{ record path; }}
validate_environment_targets() {{ record validate-targets; }}
validate_selected_projects() {{ record validate-projects; }}
configure_uv_storage() {{ export UV_PROJECT_ENVIRONMENT="$VENV_DIR"; export UV_CACHE_DIR=/tmp/uv; record uv-storage; }}
configure_runtime_storage() {{ export HF_HOME=/tmp/hf; export HF_HUB_CACHE=/tmp/hf/hub; export HF_DATASETS_CACHE=/tmp/hf/data; export HF_LEROBOT_HOME=/tmp/hf/lerobot; export TMPDIR=/tmp/frs; record runtime-storage; }}
install_python() {{ record python; }}
sync_root_environment() {{ record sync-root; }}
sync_pi05_environment() {{ record sync-pi05; }}
write_environment_file() {{ record env-file; }}
verify_python_environment() {{ record verify-root; }}
verify_pi05_environment() {{ record verify-pi05; }}
check_root_gpu() {{ record gpu-root; }}
check_pi05_gpu() {{ record gpu-pi05; }}
print_summary() {{ record summary; }}
main {quoted_args}
"""
    return subprocess.run(
        ["bash", "-c", command],
        cwd=tmp_path,
        env=os.environ.copy(),
        text=True,
        capture_output=True,
        check=False,
    )


def read_events(tmp_path: Path) -> list[str]:
    events = tmp_path / "events.log"
    return events.read_text(encoding="utf-8").splitlines() if events.exists() else []


def test_main_without_selector_preserves_dual_environment_setup(tmp_path: Path) -> None:
    completed = run_stubbed_main(tmp_path)
    assert completed.returncode == 0, completed.stderr
    events = read_events(tmp_path)
    assert "sync-root" in events
    assert "sync-pi05" in events
    assert "verify-root" in events
    assert "verify-pi05" in events
    assert "gpu-root" in events
    assert "gpu-pi05" in events


def test_root_selector_skips_pi05_setup(tmp_path: Path) -> None:
    completed = run_stubbed_main(tmp_path, "--root")
    assert completed.returncode == 0, completed.stderr
    events = read_events(tmp_path)
    assert "sync-root" in events
    assert "verify-root" in events
    assert "gpu-root" in events
    assert not any("pi05" in event for event in events)


def test_pi05_deploy_selector_skips_root_setup(tmp_path: Path) -> None:
    completed = run_stubbed_main(tmp_path, "--pi05_deploy")
    assert completed.returncode == 0, completed.stderr
    events = read_events(tmp_path)
    assert "sync-pi05" in events
    assert "verify-pi05" in events
    assert "gpu-pi05" in events
    assert not any(event.endswith("root") for event in events)


def test_help_has_no_side_effects(tmp_path: Path) -> None:
    completed = run_stubbed_main(tmp_path, "--help")
    assert completed.returncode == 0
    assert "--pi05_deploy" in completed.stdout
    assert read_events(tmp_path) == []


def test_invalid_or_conflicting_selectors_fail_before_side_effects(tmp_path: Path) -> None:
    for args in [("--unknown",), ("--root", "--pi05_deploy")]:
        case_dir = tmp_path / args[0].removeprefix("-").replace("-", "_")
        case_dir.mkdir(exist_ok=True)
        completed = run_stubbed_main(case_dir, *args)
        assert completed.returncode != 0
        assert "用法" in completed.stderr
        assert read_events(case_dir) == []
```

- [ ] **Step 2: Run the selector tests and verify RED**

Run:

```bash
uv run --no-sync pytest \
  tests/test_setup_env_dual_environment.py::test_main_without_selector_preserves_dual_environment_setup \
  tests/test_setup_env_dual_environment.py::test_root_selector_skips_pi05_setup \
  tests/test_setup_env_dual_environment.py::test_pi05_deploy_selector_skips_root_setup \
  tests/test_setup_env_dual_environment.py::test_help_has_no_side_effects \
  tests/test_setup_env_dual_environment.py::test_invalid_or_conflicting_selectors_fail_before_side_effects -v
```

Expected: FAIL because `main` ignores arguments and the target-specific sync/GPU functions do not exist.

- [ ] **Step 3: Implement argument parsing before side effects**

Add after `UV_BIN=""` in `scripts/setup_env.sh`:

```bash
SETUP_MODE="all"
SHOW_HELP=0

usage() {
    cat <<'EOF'
用法: bash scripts/setup_env.sh [--root | --pi05_deploy]

不传参数       安装根环境和 Pi0.5 部署环境
--root         只安装根 SmolVLA/FRS 环境
--pi05_deploy  只安装 Pi0.5 部署环境
-h, --help     显示帮助
EOF
}

usage_error() {
    echo "[setup] 错误：$*" >&2
    usage >&2
    return 2
}

parse_args() {
    local selected=""
    SHOW_HELP=0
    while (($#)); do
        case "$1" in
            -h|--help)
                if (($# != 1)); then
                    usage_error "--help 不能与其他参数一起使用"
                    return $?
                fi
                SHOW_HELP=1
                return 0
                ;;
            --root)
                if [[ -n "${selected}" ]]; then
                    usage_error "一次只能选择一个安装环境"
                    return $?
                fi
                selected="root"
                ;;
            --pi05_deploy)
                if [[ -n "${selected}" ]]; then
                    usage_error "一次只能选择一个安装环境"
                    return $?
                fi
                selected="pi05_deploy"
                ;;
            *)
                usage_error "未知参数：$1"
                return $?
                ;;
        esac
        shift
    done
    SETUP_MODE="${selected:-all}"
}

should_setup_root() {
    [[ "${SETUP_MODE}" == "all" || "${SETUP_MODE}" == "root" ]]
}

should_setup_pi05() {
    [[ "${SETUP_MODE}" == "all" || "${SETUP_MODE}" == "pi05_deploy" ]]
}
```

- [ ] **Step 4: Split synchronization without changing uv commands**

Extract the body of `sync_environments` into these functions:

```bash
validate_selected_projects() {
    if should_setup_root; then
        [[ -f "${PROJECT_ROOT}/pyproject.toml" ]] || fail "缺少根项目：${PROJECT_ROOT}/pyproject.toml"
        [[ -f "${PROJECT_ROOT}/uv.lock" ]] || fail "缺少根项目锁文件：${PROJECT_ROOT}/uv.lock"
    fi
    if should_setup_pi05; then
        [[ -f "${PI05_PROJECT_ROOT}/pyproject.toml" ]] || \
            fail "缺少 Pi0.5 部署项目：${PI05_PROJECT_ROOT}/pyproject.toml"
        [[ -f "${PI05_PROJECT_ROOT}/uv.lock" ]] || \
            fail "缺少 Pi0.5 部署锁文件：${PI05_PROJECT_ROOT}/uv.lock"
    fi
}

install_python() {
    cd "${PROJECT_ROOT}"
    check_existing_uv_processes
    log "安装 Python ${PYTHON_VERSION}"
    "${UV_BIN}" python install "${PYTHON_VERSION}"
}

sync_root_environment() {
    cd "${PROJECT_ROOT}"
    log "根项目环境目录：${VENV_DIR}"
    log "uv cache：${UV_CACHE_DIR}"
    log "按照根目录 uv.lock 同步 SmolVLA/训练环境"
    UV_PROJECT_ENVIRONMENT="${VENV_DIR}" \
        "${UV_BIN}" sync --frozen --python "${PYTHON_VERSION}"
}

sync_pi05_environment() {
    log "Pi0.5 部署环境目录：${PI05_VENV_DIR}"
    log "按照 deploy_pi05/uv.lock 同步独立 Pi0.5 环境"
    UV_PROJECT_ENVIRONMENT="${PI05_VENV_DIR}" \
        "${UV_BIN}" sync --frozen --python "${PYTHON_VERSION}" \
        --project "${PI05_PROJECT_ROOT}"
}

# Preserve the source-level helper used by existing tests and shell consumers.
sync_environments() {
    validate_environment_targets
    validate_selected_projects
    install_python
    sync_root_environment
    sync_pi05_environment
    write_environment_file
}
```

- [ ] **Step 5: Split GPU verification and route `main` by mode**

Replace `check_gpu` with the following two target-specific checks. Each independently computes
`expect_gpu`, prints the NVIDIA device when available, and retains the current Python assertions:

```bash
check_root_gpu() {
    cd "${PROJECT_ROOT}"
    log "检查根环境 NVIDIA、PyTorch 和 JAX 设备"
    local expect_gpu=0
    if command -v nvidia-smi >/dev/null 2>&1; then
        expect_gpu=1
        nvidia-smi --query-gpu=index,name,driver_version,memory.total --format=csv,noheader
    else
        warn "没有找到 nvidia-smi；本机只能做 CPU 开发，不能用于正式训练"
    fi
    FRS_EXPECT_GPU="${expect_gpu}" UV_PROJECT_ENVIRONMENT="${VENV_DIR}" \
        "${UV_BIN}" run --no-sync python - <<'PY'
import os

import jax
import torch

devices = jax.devices()
print(f"JAX devices: {devices}")
print(f"PyTorch CUDA available: {torch.cuda.is_available()}")
if os.environ.get("FRS_EXPECT_GPU") == "1":
    if not torch.cuda.is_available():
        raise RuntimeError("nvidia-smi 可用，但 PyTorch 没有识别到 CUDA")
    if not any(device.platform == "gpu" for device in devices):
        raise RuntimeError("nvidia-smi 可用，但 JAX 没有识别到 GPU")
PY
}

check_pi05_gpu() {
    log "检查 Pi0.5 环境 NVIDIA 和 JAX 设备"
    local expect_gpu=0
    if command -v nvidia-smi >/dev/null 2>&1; then
        expect_gpu=1
        nvidia-smi --query-gpu=index,name,driver_version,memory.total --format=csv,noheader
    else
        warn "没有找到 nvidia-smi；本机只能做 CPU 开发，不能用于正式训练"
    fi
    (
        cd "${PI05_PROJECT_ROOT}"
        FRS_EXPECT_GPU="${expect_gpu}" \
        PYTHONDONTWRITEBYTECODE=1 \
        PYTHONPATH="${PI05_PROJECT_ROOT}/src:${PI05_PROJECT_ROOT}:${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" \
            "${PI05_VENV_DIR}/bin/python" - <<'PY'
import os

import jax

devices = jax.devices()
print(f"Pi0.5 JAX devices: {devices}")
if os.environ.get("FRS_EXPECT_GPU") == "1" and not any(
    device.platform == "gpu" for device in devices
):
    raise RuntimeError("nvidia-smi 可用，但 Pi0.5 JAX 没有识别到 GPU")
PY
    )
}
```

Then replace `main` and the final summary with this control flow:

```bash
print_summary() {
    log "环境安装完成：${SETUP_MODE}"
    echo
    if should_setup_root; then
        echo "根项目环境（已安装）：${VENV_DIR}"
    else
        echo "根项目环境（本次未安装）：${VENV_DIR}"
    fi
    if should_setup_pi05; then
        echo "Pi0.5 部署环境（已安装）：${PI05_VENV_DIR}"
    else
        echo "Pi0.5 部署环境（本次未安装）：${PI05_VENV_DIR}"
    fi
    echo "环境变量：${ENV_FILE}"
    echo "Hugging Face 缓存：${HF_HOME}"
    echo "Arrow 数据缓存：${HF_DATASETS_CACHE}"
    echo
    echo "首次使用时登录："
    echo "  cd ${PROJECT_ROOT}"
    echo "  source ${ENV_FILE}"
    echo "  ${UV_BIN} run --no-sync hf auth login"
    echo "  ${UV_BIN} run --no-sync wandb login"
    echo
    echo "一键启动视觉 SmolVLA："
    echo "  bash ${PROJECT_ROOT}/train_smolvla/scripts/train.sh"
    echo "一键启动 VT-SmolVLA："
    echo "  bash ${PROJECT_ROOT}/train_vtsmolvla/scripts/train.sh"
    echo "一键启动纯视觉 Pi0.5："
    echo "  bash ${PROJECT_ROOT}/deploy_pi05/scripts/start_pi05.sh"
    echo "一键启动 Pi0.5 + FRS："
    echo "  bash ${PROJECT_ROOT}/deploy_pi05/scripts/start_pi05_frs.sh"
}

main() {
    parse_args "$@" || return $?
    if ((SHOW_HELP)); then
        usage
        return 0
    fi

    validate_environment_targets
    validate_selected_projects
    install_system_dependencies
    install_uv
    persist_uv_path
    configure_uv_storage
    configure_runtime_storage
    install_python
    should_setup_root && sync_root_environment
    should_setup_pi05 && sync_pi05_environment
    write_environment_file
    should_setup_root && verify_python_environment
    should_setup_pi05 && verify_pi05_environment
    should_setup_root && check_root_gpu
    should_setup_pi05 && check_pi05_gpu
    print_summary
}
```

- [ ] **Step 6: Run the selector tests and verify GREEN**

Run:

```bash
uv run --no-sync pytest tests/test_setup_env_dual_environment.py -v
```

Expected: all tests PASS, including the existing exact uv-lock assertions and relative-path checks.

- [ ] **Step 7: Add a real-shell help smoke test**

Run:

```bash
bash scripts/setup_env.sh --help
bash -n scripts/setup_env.sh
```

Expected: help lists `--root` and `--pi05_deploy`, exits zero without setup logs, and Bash syntax
validation exits zero.

- [ ] **Step 8: Commit the tested feature**

```bash
git add scripts/setup_env.sh tests/test_setup_env_dual_environment.py
git commit -m "feat: select setup environment"
```

### Task 2: Document target-specific installation and run regression verification

**Files:**
- Modify: `train_smolvla/README.md:7-14`
- Modify: `train_vtsmolvla/README.md:7-14`
- Modify: `deploy_pi05/README.md:24-46`

**Interfaces:**
- Consumes: The `--root` and `--pi05_deploy` CLI implemented in Task 1.
- Produces: Copy-pasteable installation commands for root training users and Pi0.5 deployment users.

- [ ] **Step 1: Update the SmolVLA and VT-SmolVLA setup commands**

In `train_smolvla/README.md` and `train_vtsmolvla/README.md`, replace the setup command in the
training preparation snippets with:

```bash
bash scripts/setup_env.sh --root
```

Immediately after each snippet, add:

```text
省略 `--root` 时，统一安装脚本会保持兼容行为，同时安装根环境和 Pi0.5 部署环境。
```

- [ ] **Step 2: Document Pi0.5-only setup**

Replace the installation example in `deploy_pi05/README.md` with:

```bash
cd /home/typhon/FRS_Tact
bash scripts/setup_env.sh --pi05_deploy
```

Add this exact explanation below it:

```text
省略 `--pi05_deploy` 时，统一脚本会保持旧行为，同时安装根项目环境和 Pi0.5
部署环境。只部署 Pi0.5 时使用上述选项，可跳过根 SmolVLA/FRS 依赖同步和验证。
```

Keep the existing `PI05_VENV_DIR` override and `.env.frs` sourcing instructions.

- [ ] **Step 3: Run focused root-environment regression checks**

In the repository root environment, run the installer suite and root package-boundary checks:

```bash
JAX_PLATFORMS=cpu uv run --no-sync pytest \
  tests/test_setup_env_dual_environment.py \
  tests/train_smolvla/test_package_boundary.py \
  tests/train_vtsmolvla/test_package_boundary.py -v
```

Expected: all selected root-environment checks PASS. The isolated `deploy_pi05` and
`train_pi05_frs` project tests require their owning environments and are not part of this root
environment command. In particular, do not use the Pi0.5 protected-path test as an all-pass gate
for this feature: it intentionally treats the documentation paths updated here as protected.

- [ ] **Step 4: Check formatting, shell syntax, and worktree scope**

Run:

```bash
bash -n scripts/setup_env.sh
git diff --check
git status --short
```

Expected: Bash and diff checks exit zero. Status contains only the intended script/test/docs changes
plus the user's pre-existing modification to
`train_smolvla_frs/configs/train_frs_bimanual_gated.yaml`.

- [ ] **Step 5: Commit documentation**

```bash
git add train_smolvla/README.md train_vtsmolvla/README.md deploy_pi05/README.md
git commit -m "docs: explain selective environment setup"
```
