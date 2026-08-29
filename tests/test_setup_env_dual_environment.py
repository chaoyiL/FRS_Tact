from __future__ import annotations

import os
from pathlib import Path
import shlex
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[1]
SETUP = ROOT / "scripts/setup_env.sh"
PI05_TRAIN_LAUNCHER = ROOT / "train_pi05/scripts/start_pi05_train.sh"


def test_setup_env_help_is_safe_to_execute_directly() -> None:
    source = SETUP.read_text(encoding="utf-8")
    assert "```" not in source, "shell 脚本不能包含 Markdown 代码围栏"

    completed = subprocess.run(
        ["bash", str(SETUP), "--help"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        timeout=5,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert "--smolvla" in completed.stdout
    assert "--pi05_deploy" in completed.stdout
    assert "--pi05_train" in completed.stdout


def test_setup_env_declares_three_distinct_environment_targets() -> None:
    source = SETUP.read_text(encoding="utf-8")
    assert 'PI05_PROJECT_ROOT="${PROJECT_ROOT}/deploy_pi05"' in source
    assert 'SMOLVLA_TORCH_VENV_DIR="${SMOLVLA_TORCH_VENV_DIR:-${DEFAULT_SMOLVLA_TORCH_VENV_DIR}}"' in source
    assert 'export SMOLVLA_TORCH_PYTHON=%q' in source
    assert '"torchcodec==${SMOLVLA_TORCHCODEC_VERSION}"' in source
    assert 'PI05_VENV_DIR="${PI05_VENV_DIR:-${DEFAULT_PI05_VENV_DIR}}"' in source
    assert 'UV_PROJECT_ENVIRONMENT="${PI05_VENV_DIR}"' in source
    assert '--project "${PI05_PROJECT_ROOT}"' in source
    assert 'PI05_TRAIN_PROJECT_ROOT="${PROJECT_ROOT}/train_pi05"' in source
    assert 'PI05_TRAIN_VENV_DIR="${PI05_TRAIN_VENV_DIR:-${DEFAULT_PI05_TRAIN_VENV_DIR}}"' in source
    assert 'UV_PROJECT_ENVIRONMENT="${PI05_TRAIN_VENV_DIR}"' in source
    assert '--project "${PI05_TRAIN_PROJECT_ROOT}"' in source
    assert 'UV_HTTP_TIMEOUT="${UV_HTTP_TIMEOUT:-120}"' in source
    assert 'UV_HTTP_RETRIES="${UV_HTTP_RETRIES:-8}"' in source
    assert 'UV_DEFAULT_INDEX="${FRS_PYPI_MIRROR}"' in source
    assert 'pytorch-cpu=${FRS_PYTORCH_INDEX}' in source
    assert "sync_environments" in source


def test_pi05_train_launcher_loads_canonical_environment_file() -> None:
    source = PI05_TRAIN_LAUNCHER.read_text(encoding="utf-8")
    assert 'ENV_FILE="${PROJECT_ROOT}/.env.frs"' in source
    assert 'source "${ENV_FILE}"' in source
    assert 'TRAIN_PYTHON_OVERRIDE="${TRAIN_PI05_PYTHON:-}"' in source
    assert (
        'TRAIN_PYTHON="${TRAIN_PYTHON_OVERRIDE:-${TRAIN_PI05_PYTHON:-${TRAIN_ROOT}/.venv/bin/python}}"'
        in source
    )


def test_sync_environments_uses_each_project_lock(tmp_path: Path) -> None:
    fake_uv = tmp_path / "uv"
    log = tmp_path / "uv.log"
    fake_uv.write_text(
        '#!/usr/bin/env bash\nprintf \'%s\\t%s\\n\' "${UV_PROJECT_ENVIRONMENT:-}" "$*" >>"${UV_LOG}"\n',
        encoding="utf-8",
    )
    fake_uv.chmod(0o755)
    env_file = tmp_path / ".env.frs"
    root_venv = tmp_path / "root-venv"
    pi05_venv = tmp_path / "pi05-venv"
    command = f"""
set -euo pipefail
source {SETUP}
UV_BIN={fake_uv}
UV_LOG={log}
export UV_LOG
VENV_DIR={root_venv}
PI05_VENV_DIR={pi05_venv}
UV_PROJECT_ENVIRONMENT={root_venv}
UV_CACHE_DIR={tmp_path / 'uv-cache'}
HF_HOME={tmp_path / 'hf'}
HF_HUB_CACHE={tmp_path / 'hf/hub'}
HF_DATASETS_CACHE={tmp_path / 'hf/datasets'}
HF_LEROBOT_HOME={tmp_path / 'hf/lerobot'}
TMPDIR={tmp_path / 'tmp'}
ENV_FILE={env_file}
check_existing_uv_processes() {{ :; }}
sync_environments
"""
    completed = subprocess.run(
        ["bash", "-c", command],
        cwd=tmp_path,
        env=os.environ.copy(),
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    calls = log.read_text(encoding="utf-8").splitlines()
    assert f"{root_venv}\tsync --frozen --python 3.12" in calls
    assert (
        f"{pi05_venv}\tsync --frozen --python 3.12 --project {ROOT / 'deploy_pi05'}"
        in calls
    )
    env_text = env_file.read_text(encoding="utf-8")
    assert f"export PI05_PYTHON={pi05_venv}/bin/python" in env_text
    assert f"export PI05_FRS_PYTHON={pi05_venv}/bin/python" in env_text


def test_sync_environments_rejects_a_shared_environment_target(tmp_path: Path) -> None:
    fake_uv = tmp_path / "uv"
    log = tmp_path / "uv.log"
    fake_uv.write_text(
        '#!/usr/bin/env bash\nprintf \'called\\n\' >>"${UV_LOG}"\n',
        encoding="utf-8",
    )
    fake_uv.chmod(0o755)
    shared_venv = tmp_path / "shared-venv"
    command = f"""
set -euo pipefail
source {SETUP}
UV_BIN={fake_uv}
UV_LOG={log}
export UV_LOG
VENV_DIR={shared_venv}
PI05_VENV_DIR={shared_venv}
check_existing_uv_processes() {{ :; }}
sync_environments
"""
    completed = subprocess.run(
        ["bash", "-c", command],
        cwd=tmp_path,
        env=os.environ.copy(),
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode != 0
    assert "必须使用不同的虚拟环境目录" in completed.stderr
    assert not log.exists(), "任何 uv 命令之前就必须拒绝共用环境"


def test_relative_environment_overrides_are_resolved_from_project_root(
    tmp_path: Path,
) -> None:
    command = f"""
set -euo pipefail
source {SETUP}
VENV_DIR=.venv-root-custom
PI05_VENV_DIR=deploy_pi05/.venv-custom
validate_environment_targets
printf '%s\\n%s\\n' "$VENV_DIR" "$PI05_VENV_DIR"
"""
    completed = subprocess.run(
        ["bash", "-c", command],
        cwd=tmp_path,
        env=os.environ.copy(),
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.splitlines() == [
        str(ROOT / ".venv-root-custom"),
        str(ROOT / "deploy_pi05/.venv-custom"),
    ]


def run_stubbed_main(tmp_path: Path, *args: str) -> subprocess.CompletedProcess[str]:
    events = tmp_path / "events.log"
    quoted_args = " ".join(shlex.quote(arg) for arg in args)
    command = f"""
set -euo pipefail
source {SETUP}
EVENTS={events}
record() {{ printf '%s\\n' "$1" >>"$EVENTS"; }}
install_system_dependencies() {{ record system; }}
install_uv() {{ UV_BIN=uv; record uv; }}
persist_uv_path() {{ record path; }}
validate_environment_targets() {{ record validate-targets; }}
validate_selected_projects() {{ record validate-projects; }}
configure_uv_storage() {{ export UV_PROJECT_ENVIRONMENT="$VENV_DIR"; export UV_CACHE_DIR=/tmp/uv; record uv-storage; }}
configure_runtime_storage() {{ export HF_HOME=/tmp/hf; export HF_HUB_CACHE=/tmp/hf/hub; export HF_DATASETS_CACHE=/tmp/hf/data; export HF_LEROBOT_HOME=/tmp/hf/lerobot; export TMPDIR=/tmp/frs; record runtime-storage; }}
install_python() {{ record python; }}
sync_root_environment() {{ record sync-root; }}
sync_smolvla_torch_environment() {{ record sync-smolvla-torch; }}
sync_pi05_environment() {{ record sync-pi05; }}
sync_pi05_train_environment() {{ record sync-pi05-train; }}
write_environment_file() {{ record env-file; }}
verify_python_environment() {{ record verify-root; }}
verify_smolvla_torch_environment() {{ record verify-smolvla-torch; }}
verify_pi05_environment() {{ record verify-pi05; }}
verify_pi05_train_environment() {{ record verify-pi05-train; }}
check_root_gpu() {{ record gpu-root; }}
check_pi05_gpu() {{ record gpu-pi05; }}
check_pi05_train_gpu() {{ record gpu-pi05-train; }}
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
    assert "sync-smolvla-torch" in events
    assert "sync-pi05" in events
    assert "verify-root" in events
    assert "verify-smolvla-torch" in events
    assert "verify-pi05" in events
    assert "gpu-root" in events
    assert "gpu-pi05" in events


def test_smolvla_selector_skips_pi05_setup(tmp_path: Path) -> None:
    completed = run_stubbed_main(tmp_path, "--smolvla")
    assert completed.returncode == 0, completed.stderr
    events = read_events(tmp_path)
    assert "sync-root" in events
    assert "sync-smolvla-torch" in events
    assert "verify-root" in events
    assert "verify-smolvla-torch" in events
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


def test_pi05_train_selector_only_sets_up_training_environment(tmp_path: Path) -> None:
    completed = run_stubbed_main(tmp_path, "--pi05_train")
    assert completed.returncode == 0, completed.stderr
    events = read_events(tmp_path)
    assert "sync-pi05-train" in events
    assert "verify-pi05-train" in events
    assert "gpu-pi05-train" in events
    assert "sync-root" not in events
    assert "sync-pi05" not in events


def test_help_has_no_side_effects(tmp_path: Path) -> None:
    completed = run_stubbed_main(tmp_path, "--help")
    assert completed.returncode == 0
    assert "--pi05_deploy" in completed.stdout
    assert read_events(tmp_path) == []


@pytest.mark.parametrize(
    ("args", "case_name"),
    [
        ((selector, help_flag), f"{selector.removeprefix('--')}_{help_flag.removeprefix('-')}")
        for selector in ("--smolvla", "--pi05_deploy", "--pi05_train")
        for help_flag in ("-h", "--help")
    ]
    + [
        ((help_flag, selector), f"{help_flag.removeprefix('-')}_{selector.removeprefix('--')}")
        for selector in ("--smolvla", "--pi05_deploy", "--pi05_train")
        for help_flag in ("-h", "--help")
    ],
)
def test_help_cannot_be_combined_with_a_selector_in_either_order(
    tmp_path: Path, args: tuple[str, str], case_name: str
) -> None:
    case_dir = tmp_path / case_name
    case_dir.mkdir()

    completed = run_stubbed_main(case_dir, *args)

    assert completed.returncode != 0
    assert "用法" in completed.stderr
    assert read_events(case_dir) == []


def test_invalid_or_conflicting_selectors_fail_before_side_effects(tmp_path: Path) -> None:
    for args in [("--unknown",), ("--smolvla", "--pi05_deploy")]:
        case_dir = tmp_path / args[0].removeprefix("-").replace("-", "_")
        case_dir.mkdir(exist_ok=True)
        completed = run_stubbed_main(case_dir, *args)
        assert completed.returncode != 0
        assert "用法" in completed.stderr
        assert read_events(case_dir) == []


def test_smolvla_storage_setup_does_not_create_unselected_pi05_parent(
    tmp_path: Path,
) -> None:
    root_venv = tmp_path / "root" / ".venv"
    pi05_venv = tmp_path / "absent-pi05" / ".venv"
    command = f"""
set -euo pipefail
source {SETUP}
SETUP_MODE=smolvla
VENV_DIR={root_venv}
PI05_VENV_DIR={pi05_venv}
UV_CACHE_DIR_VALUE={tmp_path / 'uv-cache'}
configure_uv_storage
"""
    completed = subprocess.run(
        ["bash", "-c", command],
        cwd=tmp_path,
        env=os.environ.copy(),
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert root_venv.parent.is_dir()
    assert not pi05_venv.parent.exists()


def test_pi05_storage_setup_does_not_create_unselected_root_parent(
    tmp_path: Path,
) -> None:
    root_venv = tmp_path / "absent-root" / ".venv"
    pi05_venv = tmp_path / "pi05" / ".venv"
    command = f"""
set -euo pipefail
source {SETUP}
SETUP_MODE=pi05_deploy
VENV_DIR={root_venv}
PI05_VENV_DIR={pi05_venv}
UV_CACHE_DIR_VALUE={tmp_path / 'uv-cache'}
configure_uv_storage
"""
    completed = subprocess.run(
        ["bash", "-c", command],
        cwd=tmp_path,
        env=os.environ.copy(),
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert pi05_venv.parent.is_dir()
    assert not root_venv.parent.exists()
