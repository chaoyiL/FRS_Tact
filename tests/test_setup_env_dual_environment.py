from __future__ import annotations

import os
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]
SETUP = ROOT / "scripts/setup_env.sh"


def test_setup_env_declares_two_distinct_environment_targets() -> None:
    source = SETUP.read_text(encoding="utf-8")
    assert 'PI05_PROJECT_ROOT="${PROJECT_ROOT}/deploy_pi05"' in source
    assert 'PI05_VENV_DIR="${PI05_VENV_DIR:-${PI05_PROJECT_ROOT}/.venv}"' in source
    assert 'UV_PROJECT_ENVIRONMENT="${PI05_VENV_DIR}"' in source
    assert '--project "${PI05_PROJECT_ROOT}"' in source
    assert "sync_environments" in source


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
