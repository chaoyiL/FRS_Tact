from __future__ import annotations

import os
import hashlib
import subprocess
import sys
import textwrap
import tomllib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TRAIN_ROOT = ROOT / "train_pi05_frs"
SETUP_SCRIPT = TRAIN_ROOT / "scripts" / "setup_env.sh"
SOURCE_ROOT = Path("/home/typhon/FRS_Tact-pi05-frs-jax")
SOURCE_MANIFEST = TRAIN_ROOT / "source_manifest.sha256"
DESIGN_COMMIT = "9a321e6"
TRAINING_SOURCE_MAPPINGS = {
    f"train_pi05_frs/{relative}": f"train_pi05_frs/{relative}"
    for relative in (
        "__init__.py",
        "evaluate.py",
        "plot_history.py",
        "tests/__init__.py",
        "tests/test_data.py",
        "tests/test_model.py",
        "train.py",
        "utils/__init__.py",
        "utils/checkpoint.py",
        "utils/data.py",
        "utils/history_plot.py",
        "utils/integration.py",
        "utils/metrics.py",
        "utils/model.py",
        "utils/mp_batches.py",
        "utils/visualize.py",
        "utils/window_io.py",
    )
}
PIPELINE_SOURCE_MAPPINGS = {
    "configs/train_pi05_frs.yaml": "train_pi05_frs/configs/train_pi05_frs.yaml",
    "scripts/start_frs_pi05_train.sh": (
        "train_pi05_frs/scripts/start_frs_pi05_train.sh"
    ),
    "tools/precompute_tactile_embeddings.py": (
        "train_pi05_frs/tools/precompute_tactile_embeddings.py"
    ),
    "tools/prepare_frs_pi05_cache.py": (
        "train_pi05_frs/tools/prepare_frs_pi05_cache.py"
    ),
    "tools/train_frs.py": "train_pi05_frs/tools/train_frs.py",
}
PROTECTED = (
    "pyproject.toml",
    "uv.lock",
    "lerobot",
    "train_encoder",
    "utils",
    "deploy_pi05",
    "train_smolvla",
    "train_smolvla_frs",
    "train_vtsmolvla",
)
APPROVED_ADAPTATIONS = {
    "prepare_pi05.py": "train_pi05_frs/pi05_cache/prepare.py",
    "utils/cache.py": "train_pi05_frs/pi05_cache/cache.py",
    "utils/pi05_source_model.py": "train_pi05_frs/pi05_cache/source_model.py",
    "utils/integration.py": "train_pi05_frs/pi05_cache/source_model.py",
    "utils/flow_matching.py": "train_pi05_frs/pi05_cache/source_model.py",
    "src/lerobot/datasets/__init__.py": "train_pi05_frs/src/lerobot/datasets/__init__.py",
    "src/lerobot/datasets/tactile_cache.py": (
        "train_pi05_frs/src/lerobot/datasets/tactile_cache.py"
    ),
    "src/lerobot/policies/pi05_jax/__init__.py": (
        "train_pi05_frs/src/lerobot/policies/pi05_jax/__init__.py"
    ),
    "src/lerobot/policies/pi05_jax/training/__init__.py": (
        "train_pi05_frs/src/lerobot/policies/pi05_jax/training/__init__.py"
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
    assert "find" not in metadata["tool"]["setuptools"].get("packages", {})

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


def test_setuptools_resolves_every_package_to_the_standalone_project() -> None:
    """Exercise setuptools' expanded configuration, not just TOML literals."""
    from setuptools import Distribution
    from setuptools.command.build_py import build_py
    from setuptools.config import pyprojecttoml

    distribution = pyprojecttoml.apply_configuration(
        Distribution(), TRAIN_ROOT / "pyproject.toml"
    )
    command = build_py(distribution)
    command.ensure_finalized()
    packages = list(distribution.packages or ())

    assert packages
    assert len(packages) == len(set(packages))
    assert "lerobot.processor" not in packages
    assert "train_pi05_frs.tools" in packages
    for package in packages:
        package_dir = (TRAIN_ROOT / command.get_package_dir(package)).resolve()
        if package == "lerobot" or package.startswith("lerobot."):
            expected_root = (TRAIN_ROOT / "src" / "lerobot").resolve()
        elif package == "train_pi05_frs" or package.startswith("train_pi05_frs."):
            expected_root = TRAIN_ROOT.resolve()
        else:
            raise AssertionError(f"unexpected package in standalone build: {package}")
        assert package_dir == expected_root or expected_root in package_dir.parents

    private_init = (TRAIN_ROOT / command.get_package_dir("lerobot") / "__init__.py").resolve()
    assert private_init.read_bytes() == (TRAIN_ROOT / "src/lerobot/__init__.py").read_bytes()


def test_vendored_package_docs_describe_the_trimmed_private_copy() -> None:
    package_doc = (
        TRAIN_ROOT / "src/lerobot/policies/pi05_jax/__init__.py"
    ).read_text(encoding="utf-8")
    training_doc = (
        TRAIN_ROOT / "src/lerobot/policies/pi05_jax/training/__init__.py"
    ).read_text(encoding="utf-8")

    assert "README.md in this directory" not in package_doc
    assert "selected" in package_doc.lower() and "private" in package_doc
    assert "Mirrors upstream module-for-module" not in training_doc
    assert "sharding" in training_doc and "only" in training_doc


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
        "test_load_manifest_rejects_inconsistent_progress",
        "test_tactile_fingerprint_metadata_and_reader_support_both_checkpoint_formats",
        "test_tactile_fingerprint_rejects_missing_params_file",
        "test_tactile_fingerprint_rejects_invalid_params_file",
        "test_tactile_fingerprint_rejects_params_path_escape",
        "test_tactile_fingerprint_requires_params_to_be_regular_file",
    ):
        assert required_behavior in test_source


def _git_lines(repository: Path, *arguments: str) -> set[str]:
    result = subprocess.run(
        ["git", "-C", str(repository), *arguments],
        check=True,
        text=True,
        capture_output=True,
    )
    return {line for line in result.stdout.splitlines() if line}


def test_all_source_training_python_paths_have_explicit_target_mappings() -> None:
    source_python = {
        path
        for path in _git_lines(SOURCE_ROOT, "ls-files", "train_pi05_frs")
        if path.endswith(".py")
    }

    assert len(source_python) == 17
    assert source_python == set(TRAINING_SOURCE_MAPPINGS)
    assert all((ROOT / target).is_file() for target in TRAINING_SOURCE_MAPPINGS.values())
    assert set(TRAINING_SOURCE_MAPPINGS.values()) <= _git_lines(
        ROOT, "ls-files", "train_pi05_frs"
    )


def test_all_source_pipeline_entries_have_explicit_target_mappings() -> None:
    assert all((SOURCE_ROOT / source).is_file() for source in PIPELINE_SOURCE_MAPPINGS)
    assert all((ROOT / target).is_file() for target in PIPELINE_SOURCE_MAPPINGS.values())
    assert set(PIPELINE_SOURCE_MAPPINGS.values()) <= _git_lines(
        ROOT, "ls-files", "train_pi05_frs"
    )


def test_training_project_tracks_no_forbidden_package_or_generated_artifact() -> None:
    tracked = _git_lines(ROOT, "ls-files", "train_pi05_frs")
    forbidden_packages = {
        "deploy_pi05_frs",
        "tactile_encoder",
        "modalities_eval",
        "train_smolvla",
        "train_smolvla_frs",
        "train_vtsmolvla",
    }
    generated_parts = {
        ".venv",
        ".cache",
        ".pytest_cache",
        "__pycache__",
        "action-cache",
        "action_cache",
        "best",
        "cache",
        "caches",
        "checkpoints",
        "last",
        "outputs",
        "tactile-cache",
        "tactile_cache",
        "tactile-embeddings",
        "tactile_embeddings",
    }
    generated_suffixes = {
        ".ckpt",
        ".npy",
        ".npz",
        ".pt",
        ".pth",
        ".pyc",
        ".pyo",
        ".safetensors",
    }
    generated_metadata = {"checkpoint.json", "manifest.json", "metadata.json"}

    def is_generated_artifact(path: str) -> bool:
        candidate = Path(path)
        return bool(
            generated_parts.intersection(candidate.parts)
            or candidate.suffix in generated_suffixes
            or candidate.name in generated_metadata
        )

    assert not {
        path
        for path in tracked
        if forbidden_packages.intersection(Path(path).parts)
    }
    assert not {
        path
        for path in tracked
        if is_generated_artifact(path)
    }
    for generated_path in (
        "train_pi05_frs/.pytest_cache/v/cache/nodeids",
        "train_pi05_frs/action_cache/demo/manifest.json",
        "train_pi05_frs/tactile_embeddings/demo/embeddings.npy",
        "train_pi05_frs/run/best/checkpoint.json",
        "train_pi05_frs/run/last/params.npz",
        "train_pi05_frs/run/model.safetensors",
    ):
        assert is_generated_artifact(generated_path)
    assert not is_generated_artifact("train_pi05_frs/pi05_cache/cache.py")
    assert not is_generated_artifact("train_pi05_frs/utils/checkpoint.py")


def test_protected_root_paths_have_no_branch_or_worktree_diff() -> None:
    changed = _git_lines(ROOT, "diff", "--name-only", DESIGN_COMMIT, "--", *PROTECTED)

    assert changed == set()


def test_rejected_pi05_cache_import_does_not_pollute_sys_path() -> None:
    script = textwrap.dedent(
        f"""
        import json
        import sys

        private_src = {str(TRAIN_ROOT / 'src')!r}
        sys.path[:] = [entry for entry in sys.path if entry != private_src]
        import lerobot
        before = list(sys.path)
        try:
            import train_pi05_frs.pi05_cache
        except RuntimeError:
            pass
        else:
            raise AssertionError("foreign lerobot import was not rejected")
        if sys.path != before:
            raise AssertionError(json.dumps({{"before": before, "after": sys.path}}))
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=ROOT,
        env={
            **os.environ,
            "PYTHONPATH": str(ROOT),
            "PYTHONDONTWRITEBYTECODE": "1",
        },
        check=False,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 0, result.stderr


def test_failed_pi05_cache_submodule_import_restores_sys_path() -> None:
    init_path = TRAIN_ROOT / "pi05_cache" / "__init__.py"
    script = textwrap.dedent(
        f"""
        import importlib.util
        import json
        import sys

        blocked_name = "train_pi05_frs.pi05_cache_probe.prepare"

        class BlockPrepare:
            def find_spec(self, fullname, path=None, target=None):
                if fullname == blocked_name:
                    raise ModuleNotFoundError("injected prepare import failure", name=fullname)
                return None

        private_src = {str(TRAIN_ROOT / 'src')!r}
        sys.path[:] = [entry for entry in sys.path if entry != private_src]
        before = list(sys.path)
        sys.meta_path.insert(0, BlockPrepare())
        spec = importlib.util.spec_from_file_location(
            "train_pi05_frs.pi05_cache_probe",
            {str(init_path)!r},
            submodule_search_locations=[{str(init_path.parent)!r}],
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        try:
            spec.loader.exec_module(module)
        except ModuleNotFoundError as exc:
            if exc.name != blocked_name:
                raise
        else:
            raise AssertionError("probe import unexpectedly succeeded")
        if sys.path != before:
            raise AssertionError(json.dumps({{"before": before, "after": sys.path}}))
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=ROOT,
        env={
            **os.environ,
            "PYTHONPATH": str(ROOT),
            "PYTHONDONTWRITEBYTECODE": "1",
        },
        check=False,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 0, result.stderr
