from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import stat
import subprocess
import sys
import tomllib
import zipfile

from train_smolvla_frs.utils.bimanual_schema import BIMANUAL_LOSS_MODE
from train_smolvla_frs.utils.bimanual_schema import BIMANUAL_OBJECTIVE_VERSION
from train_smolvla_frs.utils.bimanual_schema import LEFT_ACTION_SLICE
from train_smolvla_frs.utils.bimanual_schema import LEFT_WRIST_TOKEN_INDICES
from train_smolvla_frs.utils.bimanual_schema import RIGHT_ACTION_SLICE
from train_smolvla_frs.utils.bimanual_schema import RIGHT_WRIST_TOKEN_INDICES
from train_smolvla_frs.utils.bimanual_schema import validate_bimanual_objective_metadata


ROOT = Path(__file__).resolve().parents[2]


def test_bimanual_schema_has_fixed_contract_and_validates_metadata() -> None:
    assert BIMANUAL_LOSS_MODE == "bimanual_gated"
    assert BIMANUAL_OBJECTIVE_VERSION == 2
    assert LEFT_ACTION_SLICE == slice(0, 10)
    assert RIGHT_ACTION_SLICE == slice(10, 20)
    assert LEFT_WRIST_TOKEN_INDICES == (0, 1)
    assert RIGHT_WRIST_TOKEN_INDICES == (2, 3)
    validate_bimanual_objective_metadata(
        {
            "loss_mode": "bimanual_gated",
            "loss_objective_version": 2,
            "action_slices": {"left": [0, 10], "right": [10, 20]},
            "wrist_token_indices": {"left": [0, 1], "right": [2, 3]},
        }
    )


def test_core_frs_files_live_in_train_smolvla_frs() -> None:
    expected = {
        "train_frs.py",
        "evaluate.py",
        "plot_history.py",
        "utils/checkpoint.py",
        "utils/bimanual_schema.py",
        "utils/data.py",
        "utils/history_plot.py",
        "utils/integration.py",
        "utils/metrics.py",
        "utils/model.py",
        "utils/mp_batches.py",
        "utils/visualize.py",
        "utils/window_io.py",
    }
    missing = sorted(path for path in expected if not (ROOT / "train_smolvla_frs" / path).is_file())
    assert missing == []


def test_new_core_modules_import_and_old_package_is_gone() -> None:
    assert importlib.util.find_spec("train_smolvla_frs.train_frs") is not None
    assert importlib.util.find_spec("train_smolvla_frs.utils.model") is not None
    assert importlib.util.find_spec("tactile_flow_steering") is None


def test_frs_entrypoints_live_in_package() -> None:
    expected = {
        "train_frs.py",
        "prepare_frs_caches.py",
        "compare_frs_reverse_solvers.py",
    }
    missing = sorted(path for path in expected if not (ROOT / "train_smolvla_frs" / path).is_file())
    assert missing == []
    assert not (ROOT / "tools" / "train_frs.py").exists()
    assert not (ROOT / "tools" / "prepare_frs_caches.py").exists()
    assert not (ROOT / "tools" / "compare_frs_reverse_solvers.py").exists()
    assert not (ROOT / "prepare.py").exists()
    assert not (ROOT / "train_smolvla_frs" / "prepare.py").exists()
    assert not (ROOT / "train_smolvla_frs" / "train.py").exists()


def test_migrated_cli_files_remain_executable() -> None:
    for relative_path in (
        "train_frs.py",
        "prepare_frs_caches.py",
        "compare_frs_reverse_solvers.py",
    ):
        mode = (ROOT / "train_smolvla_frs" / relative_path).stat().st_mode
        assert mode & stat.S_IXUSR, relative_path


def test_train_smolvla_frs_module_help() -> None:
    completed = subprocess.run(
        [sys.executable, "-m", "train_smolvla_frs.train_frs", "--help"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert "--config" in completed.stdout


def test_config_and_launcher_live_in_train_smolvla_frs() -> None:
    assert (ROOT / "train_smolvla_frs" / "configs" / "train_frs.yaml").is_file()
    assert (ROOT / "train_smolvla_frs" / "scripts" / "start_frs_train.sh").is_file()
    assert not (ROOT / "configs" / "train_frs.yaml").exists()
    assert not (ROOT / "scripts" / "start_frs_train.sh").exists()


def test_train_smolvla_frs_is_discovered_by_setuptools() -> None:
    with (ROOT / "pyproject.toml").open("rb") as file:
        project = tomllib.load(file)
    includes = project["tool"]["setuptools"]["packages"]["find"]["include"]
    assert "train_smolvla_frs*" in includes


def test_wheel_contains_train_smolvla_frs_runtime_resources(tmp_path: Path) -> None:
    completed = subprocess.run(
        [
            "uv",
            "build",
            "--wheel",
            "--no-build-isolation",
            "--no-create-gitignore",
            "--cache-dir",
            str(tmp_path / "uv-cache"),
            "--out-dir",
            str(tmp_path),
            str(ROOT),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr

    wheels = list(tmp_path.glob("*.whl"))
    assert len(wheels) == 1
    with zipfile.ZipFile(wheels[0]) as wheel:
        contents = set(wheel.namelist())
    assert {
        "modalities_eval/__init__.py",
        "modalities_eval/utils.py",
        "train_smolvla_frs/README.md",
        "train_smolvla_frs/configs/train_frs.yaml",
        "train_smolvla_frs/scripts/start_frs_train.sh",
    } <= contents

    installed = tmp_path / "installed"
    with zipfile.ZipFile(wheels[0]) as wheel:
        wheel.extractall(installed)
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(installed)
    outside_repo = tmp_path / "outside-repo"
    outside_repo.mkdir()
    for module in (
        "train_smolvla_frs.train_frs",
        "train_smolvla_frs.prepare_frs_caches",
        "train_smolvla_frs.compare_frs_reverse_solvers",
    ):
        help_result = subprocess.run(
            [sys.executable, "-m", module, "--help"],
            cwd=outside_repo,
            env=environment,
            text=True,
            capture_output=True,
            check=False,
        )
        assert help_result.returncode == 0, f"{module}: {help_result.stderr}"


def test_pyyaml_is_a_direct_runtime_dependency() -> None:
    with (ROOT / "pyproject.toml").open("rb") as file:
        project = tomllib.load(file)
    dependencies = project["project"]["dependencies"]
    assert any(dependency.lower().startswith("pyyaml") for dependency in dependencies)


def test_launcher_uses_new_module_paths() -> None:
    launcher = (ROOT / "train_smolvla_frs" / "scripts" / "start_frs_train.sh").read_text()
    assert "python -m train_smolvla_frs.compare_frs_reverse_solvers" in launcher
    assert "python -m train_smolvla_frs.prepare_frs_caches" in launcher
    assert "python -m train_smolvla_frs.train_frs" in launcher
    assert "tools/merge_smolvla_peft_to_jax.py" in launcher


def test_loss_ablation_launcher_trains_four_leave_one_out_runs() -> None:
    launcher_path = ROOT / "train_smolvla_frs" / "scripts" / "start_frs_loss_ablation.sh"
    launcher = launcher_path.read_text()
    assert launcher_path.stat().st_mode & stat.S_IXUSR
    assert "python -m train_smolvla_frs.utils.loss_ablation" in launcher
    assert "python -m train_smolvla_frs.train_frs" in launcher
    assert "checkpoints/frs" in launcher
    for name in ("no_aux_decode", "no_low_gate_safety", "no_rank", "no_repair"):
        assert name in launcher
