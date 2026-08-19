from __future__ import annotations

import os
import hashlib
import subprocess
import textwrap
import tomllib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TRAIN_ROOT = ROOT / "train_pi05_frs"
SETUP_SCRIPT = TRAIN_ROOT / "scripts" / "setup_env.sh"
SOURCE_ROOT = Path("/home/typhon/FRS_Tact-pi05-frs-jax")
SOURCE_MANIFEST = TRAIN_ROOT / "source_manifest.sha256"
APPROVED_ADAPTATIONS = {
    "prepare_pi05.py": "train_pi05_frs/pi05_cache/prepare.py",
    "utils/pi05_source_model.py": "train_pi05_frs/pi05_cache/source_model.py",
    "utils/integration.py": "train_pi05_frs/pi05_cache/source_model.py",
    "utils/flow_matching.py": "train_pi05_frs/pi05_cache/source_model.py",
    "src/lerobot/datasets/__init__.py": "train_pi05_frs/src/lerobot/datasets/__init__.py",
    "src/lerobot/datasets/tactile_cache.py": (
        "train_pi05_frs/src/lerobot/datasets/tactile_cache.py"
    ),
}


def read_source_manifest() -> tuple[dict[str, str], dict[str, str]]:
    mappings: dict[str, str] = {}
    checksums: dict[str, str] = {}
    for line in SOURCE_MANIFEST.read_text(encoding="utf-8").splitlines():
        if line.startswith("# ") and " -> " in line:
            source, target = line[2:].split(" -> ", 1)
            mappings[source] = target
        elif line and not line.startswith("#"):
            digest, target = line.split("  ", 1)
            checksums[target] = digest
    return mappings, checksums


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


def test_training_environment_imports_checkpoint_with_private_lerobot() -> None:
    """Integration coverage for the Task 3 checkpoint module and Task 2 private runtime."""
    result = subprocess.run(
        [
            str(TRAIN_ROOT / ".venv" / "bin" / "python"),
            "-c",
            "from train_pi05_frs.utils.checkpoint import load_checkpoint",
        ],
        text=True,
        capture_output=True,
        cwd=ROOT,
        env={
            **os.environ,
            "PYTHONPATH": f"{TRAIN_ROOT / 'src'}:{ROOT}",
        },
        check=False,
    )

    assert result.returncode == 0, result.stderr


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


def test_source_manifest_maps_and_verifies_unchanged_private_files() -> None:
    mappings, checksums = read_source_manifest()

    assert mappings
    assert set(mappings.values()) == set(checksums)
    for source, target in mappings.items():
        source_path = SOURCE_ROOT / source
        target_path = ROOT / target
        assert source_path.is_file(), source
        assert target_path.is_file(), target
        assert hashlib.sha256(source_path.read_bytes()).hexdigest() == checksums[target]
        assert hashlib.sha256(target_path.read_bytes()).hexdigest() == checksums[target]


def test_private_closure_contains_only_approved_pi05_and_dataset_runtime() -> None:
    private_root = TRAIN_ROOT / "src" / "lerobot"
    files = {
        path.relative_to(private_root).as_posix()
        for path in private_root.rglob("*")
        if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc"
    }

    forbidden_fragments = (
        "smolvla",
        "deploy",
        "encoder",
        "modalities_eval",
        "train_smolvla",
        "train_vtsmolvla",
        "dataset_writer.py",
        "compute_stats.py",
    )
    assert not {
        path for path in files if any(fragment in path.lower() for fragment in forbidden_fragments)
    }
    mappings, _ = read_source_manifest()
    mapped_private_files = {
        Path(target).relative_to("train_pi05_frs/src/lerobot").as_posix()
        for target in mappings.values()
        if target.startswith("train_pi05_frs/src/lerobot/")
    }
    adapted_private_files = {
        Path(target).relative_to("train_pi05_frs/src/lerobot").as_posix()
        for target in APPROVED_ADAPTATIONS.values()
        if target.startswith("train_pi05_frs/src/lerobot/")
    }
    assert files == mapped_private_files | adapted_private_files


def test_private_dataset_init_exports_readers_without_writer_imports() -> None:
    source = (TRAIN_ROOT / "src" / "lerobot" / "datasets" / "__init__.py").read_text(
        encoding="utf-8"
    )

    assert "LeRobotDatasetMetadata" in source
    assert "LeRobotDataset" in source
    assert "compute_stats" not in source
    assert "dataset_writer" not in source


def test_root_level_pi05_cache_import_bootstraps_private_lerobot() -> None:
    result = subprocess.run(
        [
            str(SOURCE_ROOT / ".venv" / "bin" / "python"),
            "-c",
            (
                "from pathlib import Path; "
                "from train_pi05_frs.pi05_cache import prepare_cache; "
                "import lerobot; print(Path(lerobot.__file__).resolve())"
            ),
        ],
        cwd=ROOT,
        env={
            **os.environ,
            "PYTHONPATH": f"{TRAIN_ROOT / 'src'}:{ROOT}",
            "PYTHONDONTWRITEBYTECODE": "1",
        },
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert TRAIN_ROOT / "src" in Path(result.stdout.strip()).parents


def test_standalone_cwd_directly_exports_private_pi05_model_api() -> None:
    result = subprocess.run(
        [
            str(SOURCE_ROOT / ".venv" / "bin" / "python"),
            "-c",
            (
                "from pathlib import Path; "
                "from lerobot.policies.pi05_jax import Pi0Config, load_pi0; "
                "import lerobot; print(Path(lerobot.__file__).resolve())"
            ),
        ],
        cwd=TRAIN_ROOT,
        env={
            **os.environ,
            "PYTHONPATH": f"{TRAIN_ROOT / 'src'}:{ROOT}",
            "PYTHONDONTWRITEBYTECODE": "1",
        },
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert TRAIN_ROOT / "src" in Path(result.stdout.strip()).parents


def test_approved_adaptations_have_explicit_mapping_tests() -> None:
    test_source = (TRAIN_ROOT / "tests" / "test_pi05_cache.py").read_text(encoding="utf-8")
    unchanged_mappings, _ = read_source_manifest()

    assert set(unchanged_mappings.values()).isdisjoint(APPROVED_ADAPTATIONS.values())
    assert all((SOURCE_ROOT / source).is_file() for source in APPROVED_ADAPTATIONS)
    assert all((ROOT / target).is_file() for target in APPROVED_ADAPTATIONS.values())
    for required_behavior in (
        "test_record_selection_is_episode_disjoint_trimmed_and_strided",
        "test_twenty_dimensional_actions_are_padded_to_model_dimension",
        "test_camera_map_rejects_unknown_pi05_slot_before_dataset_access",
        "test_norm_stats_reject_dimensions_wider_than_dataset",
        "test_inference_noise_is_deterministic_per_seed_and_dataset_index",
        "test_inversion_mse_matches_per_sample_squared_error",
        "test_reverse_solvers_preserve_shape_and_finiteness",
        "test_prepare_cache_records_provenance_resumes_and_skips_completed_cache",
        "test_tactile_fingerprint_metadata_and_reader_support_both_checkpoint_formats",
        "test_tactile_fingerprint_rejects_missing_params_file",
        "test_tactile_fingerprint_rejects_invalid_params_file",
        "test_tactile_fingerprint_rejects_params_path_escape",
        "test_tactile_fingerprint_requires_params_to_be_regular_file",
    ):
        assert required_behavior in test_source
