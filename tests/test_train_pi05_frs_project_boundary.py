from __future__ import annotations

import os
import subprocess
import textwrap
import tomllib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TRAIN_ROOT = ROOT / "train_pi05_frs"
SETUP_SCRIPT = TRAIN_ROOT / "scripts" / "setup_env.sh"


def run_sourced_setup(
    tmp_path: Path,
    *,
    train_venv: Path,
    function: str,
    extra_environment: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    fake_uv = tmp_path / "uv"
    fake_uv.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        "printf '%s\\n%s\\n' \"${UV_PROJECT_ENVIRONMENT:?}\" \"$*\" > \"${FAKE_UV_LOG:?}\"\n",
        encoding="utf-8",
    )
    fake_uv.chmod(0o755)
    environment = {
        **os.environ,
        "PATH": f"{tmp_path}:{os.environ['PATH']}",
        "UV_BIN": str(fake_uv),
        "FAKE_UV_LOG": str(tmp_path / "uv.called"),
        "TRAIN_PI05_FRS_VENV": str(train_venv),
        **(extra_environment or {}),
    }
    return subprocess.run(
        [
            "bash",
            "-c",
            textwrap.dedent(
                f"""
                set -euo pipefail
                source {SETUP_SCRIPT}
                {function}
                """
            ),
        ],
        text=True,
        capture_output=True,
        env=environment,
        check=False,
    )


def test_setup_rejects_root_or_deploy_environment_before_uv(tmp_path: Path) -> None:
    for forbidden in (ROOT / ".venv", ROOT / "deploy_pi05/.venv"):
        result = run_sourced_setup(
            tmp_path,
            train_venv=forbidden,
            function="validate_environment_targets",
        )
        assert result.returncode != 0
        assert "独立虚拟环境" in result.stderr
        assert not (tmp_path / "uv.called").exists()


def test_setup_syncs_only_the_train_project_environment(tmp_path: Path) -> None:
    train_venv = TRAIN_ROOT / ".venv"
    result = run_sourced_setup(
        tmp_path,
        train_venv=train_venv,
        function="sync_environment",
    )

    assert result.returncode == 0, result.stderr
    assert (tmp_path / "uv.called").read_text(encoding="utf-8").splitlines() == [
        str(train_venv),
        f"sync --frozen --python 3.12 --project {TRAIN_ROOT}",
    ]


def test_setup_derives_python_from_the_canonicalized_train_environment(tmp_path: Path) -> None:
    train_venv_alias = TRAIN_ROOT / "alias" / ".." / ".venv"
    result = run_sourced_setup(
        tmp_path,
        train_venv=train_venv_alias,
        function='validate_environment_targets; printf "%s" "${TRAIN_PI05_FRS_PYTHON}"',
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout == str(TRAIN_ROOT / ".venv" / "bin" / "python")
    assert not (tmp_path / "uv.called").exists()


def test_setup_rejects_root_or_deploy_python_before_uv(tmp_path: Path) -> None:
    for forbidden in (ROOT / ".venv/bin/python", ROOT / "deploy_pi05/.venv/bin/python"):
        result = run_sourced_setup(
            tmp_path,
            train_venv=TRAIN_ROOT / ".venv",
            function="sync_environment",
            extra_environment={"TRAIN_PI05_FRS_PYTHON": str(forbidden)},
        )

        assert result.returncode != 0
        assert "独立虚拟环境" in result.stderr
        assert not (tmp_path / "uv.called").exists()


def test_sourcing_setup_exposes_default_python_without_running_main(tmp_path: Path) -> None:
    fake_uv = tmp_path / "uv"
    fake_uv.write_text(
        "#!/usr/bin/env bash\n"
        "touch \"${FAKE_UV_LOG:?}\"\n",
        encoding="utf-8",
    )
    fake_uv.chmod(0o755)
    result = subprocess.run(
        [
            "bash",
            "-c",
            f"source {SETUP_SCRIPT}; printf '%s' \"${{TRAIN_PI05_FRS_PYTHON}}\"",
        ],
        text=True,
        capture_output=True,
        env={
            **os.environ,
            "UV_BIN": str(fake_uv),
            "FAKE_UV_LOG": str(tmp_path / "uv.called"),
        },
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout == str(TRAIN_ROOT / ".venv" / "bin" / "python")
    assert not (tmp_path / "uv.called").exists()


def test_setup_check_is_dependency_free_and_reports_boundary(tmp_path: Path) -> None:
    fake_uv = tmp_path / "uv"
    fake_uv.write_text("#!/usr/bin/env bash\nexit 99\n", encoding="utf-8")
    fake_uv.chmod(0o755)
    environment = {
        **os.environ,
        "PATH": f"{tmp_path}:{os.environ['PATH']}",
        "UV_BIN": str(fake_uv),
    }
    result = subprocess.run(
        ["bash", str(SETUP_SCRIPT), "--check"],
        text=True,
        capture_output=True,
        env=environment,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert f"project: {TRAIN_ROOT}" in result.stdout
    assert f"environment: {TRAIN_ROOT / '.venv'}" in result.stdout
    assert "python: 3.12" in result.stdout
    assert "entrypoints:" in result.stdout


def test_standalone_metadata_and_ignore_rules_define_a_private_boundary() -> None:
    metadata = tomllib.loads((TRAIN_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert (TRAIN_ROOT / "uv.lock").is_file()
    assert metadata["project"]["name"] == "pi05-frs-training"
    assert metadata["project"]["requires-python"] == ">=3.12,<3.13"
    assert metadata["tool"]["setuptools"]["packages"]["find"]["where"] == ["src", ".."]
    assert metadata["tool"]["setuptools"]["packages"]["find"]["include"] == [
        "train_pi05_frs*",
        "lerobot*",
    ]

    dependencies = metadata["project"]["dependencies"]
    for pinned_dependency in (
        "jax[cuda12-local]==0.5.3",
        "jaxlib==0.5.3",
        "flax==0.10.2",
        "orbax-checkpoint==0.11.13",
        "transformers==4.53.2",
        "ml-dtypes==0.4.1",
    ):
        assert pinned_dependency in dependencies
    assert any(dependency.startswith("numpy") and "<2.3" in dependency for dependency in dependencies)

    ignore_rules = (ROOT / ".gitignore").read_text(encoding="utf-8")
    for rule in (
        "/train_pi05_frs/.venv/",
        "/train_pi05_frs/.cache/",
        "/train_pi05_frs/outputs/",
    ):
        assert rule in ignore_rules
    assert "/train_pi05_frs/src/" not in ignore_rules
    assert "/train_pi05_frs/configs/" not in ignore_rules
    assert "/train_pi05_frs/tests/" not in ignore_rules
