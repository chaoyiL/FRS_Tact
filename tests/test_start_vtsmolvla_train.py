from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


def _write_fake_uv(path: Path, call_log: Path) -> None:
    path.write_text(
        "#!/usr/bin/env bash\n"
        "set -eu\n"
        f"printf '%s\\n' \"$*\" >> {call_log!s}\n"
        "if [[ \"${4:-}\" == \"-\" ]]; then\n"
        "  printf '%s\\n' /tmp/vtsmolvla-output 0 ''\n"
        "fi\n",
        encoding="utf-8",
    )
    path.chmod(0o755)


def _prepare_launcher_project(tmp_path: Path) -> tuple[Path, Path, Path]:
    project = tmp_path / "project"
    scripts = project / "scripts"
    configs = project / "configs"
    fake_bin = tmp_path / "bin"
    scripts.mkdir(parents=True)
    configs.mkdir()
    fake_bin.mkdir()
    shutil.copy2(ROOT / "scripts" / "start_vtsmolvla_train.sh", scripts)
    (project / ".env.frs").write_text("UV_PROJECT_ENVIRONMENT=/tmp/test-venv\n")
    (configs / "train_vtsmolvla_jax.yaml").write_text("output: /tmp/unused\n")
    return project, fake_bin, fake_bin / "uv-calls.log"


@pytest.mark.parametrize(
    ("arguments", "config_name"),
    [
        ([], "train_vtsmolvla_jax.yaml"),
        (["--config", "configs/explicit.yaml"], "explicit.yaml"),
        (["--config=configs/equal form.yaml"], "equal form.yaml"),
    ],
)
def test_launcher_selects_default_and_explicit_config_paths(
    tmp_path: Path, arguments: list[str], config_name: str
) -> None:
    project, fake_bin, call_log = _prepare_launcher_project(tmp_path)
    config_path = project / "configs" / config_name
    config_path.write_text("output: /tmp/unused\n")
    _write_fake_uv(fake_bin / "uv", call_log)
    env = os.environ | {"PATH": f"{fake_bin}:{os.environ['PATH']}", "FRS_FOREGROUND": "1"}

    result = subprocess.run(
        ["bash", "scripts/start_vtsmolvla_train.sh", *arguments],
        cwd=project,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert f"[vtsmolvla] config={config_path}" in result.stdout


@pytest.mark.parametrize(
    "arguments",
    [
        ["--unknown"],
        ["--config"],
        ["--config", "first.yaml", "--config=second.yaml"],
    ],
)
def test_launcher_rejects_invalid_config_arguments_before_preflight(
    tmp_path: Path, arguments: list[str]
) -> None:
    project, fake_bin, call_log = _prepare_launcher_project(tmp_path)
    _write_fake_uv(fake_bin / "uv", call_log)
    env = os.environ | {"PATH": f"{fake_bin}:{os.environ['PATH']}", "FRS_FOREGROUND": "1"}

    result = subprocess.run(
        ["bash", "scripts/start_vtsmolvla_train.sh", *arguments],
        cwd=project,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode != 0
    assert not call_log.exists(), "invalid arguments must fail before the JAX preflight"


def test_launcher_tmux_forwards_config_path_with_spaces(tmp_path: Path) -> None:
    project, fake_bin, call_log = _prepare_launcher_project(tmp_path)
    config_path = project / "configs" / "paper config.yaml"
    config_path.write_text("output: /tmp/unused\n")
    _write_fake_uv(fake_bin / "uv", call_log)
    tmux_log = fake_bin / "tmux-command.log"
    tmux = fake_bin / "tmux"
    tmux.write_text(
        "#!/usr/bin/env bash\n"
        "if [[ \"$1\" == \"has-session\" ]]; then exit 1; fi\n"
        f"printf '%s\\n' \"${{@: -1}}\" > {tmux_log!s}\n",
        encoding="utf-8",
    )
    tmux.chmod(0o755)
    env = os.environ | {"PATH": f"{fake_bin}:{os.environ['PATH']}"}

    result = subprocess.run(
        ["bash", "scripts/start_vtsmolvla_train.sh", "--config", str(config_path)],
        cwd=project,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    inner_command = tmux_log.read_text(encoding="utf-8")
    assert "--config" in inner_command
    assert "paper\\ config.yaml" in inner_command
