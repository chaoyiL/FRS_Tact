"""Regression checks for the self-contained Pi0.5 migration boundary."""

from __future__ import annotations

from pathlib import Path
import stat
import subprocess
import tomllib


DEPLOY_ROOT = Path(__file__).resolve().parents[1]
TARGET_ROOT = DEPLOY_ROOT.parent
OLD_PACKAGE = "deploy_pi05_frs"


def _implementation_files() -> list[Path]:
    """Return implementation metadata, Python, and shell files, excluding tests."""
    files = [DEPLOY_ROOT / "pyproject.toml"]
    for suffix in ("*.py", "*.sh"):
        files.extend(path for path in DEPLOY_ROOT.rglob(suffix) if "tests" not in path.parts)
    return sorted(files)


def test_implementation_has_no_legacy_package_import_or_entrypoint() -> None:
    """Only the FRS config filename may retain the historical suffix."""
    offenders: list[str] = []
    for path in _implementation_files():
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if OLD_PACKAGE in line and f"{OLD_PACKAGE}.yaml" not in line:
                offenders.append(f"{path.relative_to(DEPLOY_ROOT)}:{line_number}: {line}")
    assert not offenders, "\n".join(offenders)


def test_migrated_implementation_is_self_contained_and_target_dependencies_are_clean() -> None:
    """Pi0.5 code must not spill into target-root dependency packages."""
    assert not (TARGET_ROOT / OLD_PACKAGE).exists()

    protected_paths = (
        "lerobot",
        "utils",
        "train_encoder",
        "train_smolvla_frs",
        "pyproject.toml",
        "uv.lock",
    )
    result = subprocess.run(
        ["git", "diff", "--quiet", "HEAD", "--", *protected_paths],
        cwd=TARGET_ROOT,
        check=False,
    )
    assert result.returncode == 0, "target-root dependency files were modified"

    status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=all", "--", *protected_paths],
        cwd=TARGET_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert not status.stdout, f"target-root dependency files were added or modified:\n{status.stdout}"


def test_tool_default_configs_are_present() -> None:
    """Copied Pi0.5 preparation tools keep their checked-in default configs."""
    assert (DEPLOY_ROOT / "configs" / "train_pi05_frs.yaml").is_file()
    assert (DEPLOY_ROOT / "configs" / "train_tactile_encoder.yaml").is_file()


def test_frs_training_config_has_its_advertised_launcher() -> None:
    """The copied FRS training config must not advertise a missing script."""
    assert (DEPLOY_ROOT / "scripts" / "start_frs_pi05_train.sh").is_file()


def test_training_and_setup_support_files_are_complete_and_executable() -> None:
    """Every local preparation/training entrypoint and its referenced plan is present."""
    executable_scripts = (
        "scripts/setup_env.sh",
        "scripts/start_pi05_train.sh",
        "scripts/start_frs_pi05_train.sh",
    )
    for relative_path in executable_scripts:
        path = DEPLOY_ROOT / relative_path
        assert path.is_file(), f"missing migrated support file: {relative_path}"
        assert stat.S_IMODE(path.stat().st_mode) & 0o111 == 0o111

    for relative_path in (
        "scripts/start_pi05.sh",
        "scripts/start_pi05_frs.sh",
        "scripts/start_remote_client.sh",
    ):
        assert stat.S_IMODE((DEPLOY_ROOT / relative_path).stat().st_mode) == 0o664

    assert (DEPLOY_ROOT / "pi05_frs_plan.md").is_file()


def test_current_local_documented_paths_exist() -> None:
    """Actionable local paths named by migrated docs/config/scripts must resolve."""
    relative_paths = (
        "configs/deploy_pi05.yaml",
        "configs/deploy_pi05_frs.yaml",
        "configs/train_pi05_frs.yaml",
        "configs/train_tactile_encoder.yaml",
        "scripts/setup_env.sh",
        "scripts/start_pi05.sh",
        "scripts/start_pi05_frs.sh",
        "scripts/start_remote_client.sh",
        "scripts/start_pi05_train.sh",
        "scripts/start_frs_pi05_train.sh",
        "src/lerobot/policies/pi05_jax/README.md",
        "tools/compute_pi05_norm_stats.py",
        "tools/precompute_tactile_embeddings.py",
        "tools/prepare_frs_pi05_cache.py",
        "tools/train_frs.py",
        "tools/train_pi05_jax.py",
        "pi05_frs_plan.md",
    )
    missing = [relative_path for relative_path in relative_paths if not (DEPLOY_ROOT / relative_path).exists()]
    assert not missing, f"missing locally referenced paths: {missing}"


def test_packaging_discovers_private_runtime_packages() -> None:
    """A normal frozen sync must install the private Pi0.5 runtime editable."""
    payload = tomllib.loads((DEPLOY_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    discovery = payload["tool"]["setuptools"]["packages"]["find"]
    assert discovery["where"] == ["src", "."]
    includes = set(discovery["include"])
    assert {"lerobot*", "tactile_encoder*", "train_pi05_frs*", "utils*"} <= includes


def test_readme_documents_isolated_mode_specific_deployment() -> None:
    """Operators need the target path, local environment, and both client modes."""
    readme = (DEPLOY_ROOT / "README.md").read_text(encoding="utf-8")
    for required_text in (
        "/home/typhon/FRS_Tact/deploy_pi05",
        "cd /home/typhon/FRS_Tact/deploy_pi05 && uv sync --frozen",
        "configs/deploy_pi05.yaml",
        "configs/deploy_pi05_frs.yaml",
        "bash deploy_pi05/scripts/start_pi05.sh --check",
        "bash deploy_pi05/scripts/start_pi05_frs.sh --check",
        "vb3_robot_server",
        "frs_steering_v1",
        "/home/typhon/FRS_Tact/deploy_pi05/outputs",
    ):
        assert required_text in readme
    assert "--no-install-project" not in readme
