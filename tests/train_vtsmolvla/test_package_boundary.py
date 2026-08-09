import os
from pathlib import Path
import subprocess
import sys
import tomllib
import zipfile


ROOT = Path(__file__).resolve().parents[2]


def test_vt_package_is_discoverable():
    import importlib.util

    assert importlib.util.find_spec("train_vtsmolvla") is not None


def test_legacy_mixed_package_and_lazy_exports_are_gone():
    import importlib.util

    assert importlib.util.find_spec("lerobot.policies.smolvla_jax") is None
    source = (ROOT / "src/lerobot/policies/__init__.py").read_text(encoding="utf-8")
    assert ".smolvla_jax" not in source


def test_vt_config_extends_visual_config():
    from train_smolvla import JaxSmolVLAConfig
    from train_vtsmolvla import VTSmolVLAConfig

    config = VTSmolVLAConfig(
        tactile_keys=("observation.images.tactile",),
        tactile_num_tokens=1,
    )

    assert isinstance(config, JaxSmolVLAConfig)


def test_vt_runtime_types_extend_visual_primitives():
    from train_smolvla.data import LeRobotJaxDataLoader
    from train_smolvla.modeling import JaxSmolVLA
    from train_smolvla.policy import JaxSmolVLAPolicy
    from train_smolvla.preprocessing import JaxSmolVLAPreprocessor
    from train_smolvla.training import JaxSmolVLATrainer
    from train_vtsmolvla import (
        VTJaxSmolVLA,
        VTJaxSmolVLAPolicy,
        VTJaxSmolVLAPreprocessor,
        VTJaxSmolVLATrainer,
        VTLeRobotJaxDataLoader,
    )

    assert issubclass(VTJaxSmolVLA, JaxSmolVLA)
    assert issubclass(VTJaxSmolVLAPolicy, JaxSmolVLAPolicy)
    assert issubclass(VTJaxSmolVLATrainer, JaxSmolVLATrainer)
    assert issubclass(VTLeRobotJaxDataLoader, LeRobotJaxDataLoader)
    assert issubclass(VTJaxSmolVLAPreprocessor, JaxSmolVLAPreprocessor)


def test_vt_model_preserves_explicit_tactile_call_interfaces():
    import inspect

    from train_vtsmolvla import VTJaxSmolVLA

    for method_name in (
        "embed_prefix",
        "flow_velocity",
        "build_prefix_context",
        "sample_actions",
    ):
        parameters = inspect.signature(getattr(VTJaxSmolVLA, method_name)).parameters
        assert "tactile_images" in parameters, method_name
        assert "tactile_embeddings" in parameters, method_name
        assert "tactile_masks" in parameters, method_name
    sample_parameters = inspect.signature(VTJaxSmolVLA.sample_actions).parameters
    for parameter in (
        "noise",
        "num_steps",
        "previous_chunk",
        "inference_delay",
        "execution_horizon",
    ):
        assert parameter in sample_parameters


def test_vt_runtime_assets_replace_old_entrypoints():
    for relative in ("README.md", "configs/train.yaml", "scripts/train.sh", "launcher.py", "train.py"):
        assert (ROOT / "train_vtsmolvla" / relative).is_file(), relative
    for old in (
        "tools/train_vtsmolvla_jax.py",
        "configs/train_vtsmolvla_jax.yaml",
        "scripts/start_vtsmolvla_train.sh",
    ):
        assert not (ROOT / old).exists(), old
    assert "bash ${PROJECT_ROOT}/train_vtsmolvla/scripts/train.sh" in (
        ROOT / "scripts/setup_env.sh"
    ).read_text(encoding="utf-8")


def test_vt_runtime_assets_are_packaged_and_outputs_are_ignored():
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    package_data = project["tool"]["setuptools"]["package-data"]["train_vtsmolvla"]
    assert {"README.md", "configs/*.yaml", "scripts/*.sh"} <= set(package_data)
    ignored = subprocess.run(
        ["git", "check-ignore", "-q", "train_vtsmolvla/outputs/logs/example.log"],
        cwd=ROOT,
        check=False,
    )
    assert ignored.returncode == 0


def test_vt_shell_is_thin_and_valid():
    shell = ROOT / "train_vtsmolvla/scripts/train.sh"
    completed = subprocess.run(["bash", "-n", str(shell)], check=False)
    assert completed.returncode == 0
    source = shell.read_text(encoding="utf-8")
    assert "python -m train_vtsmolvla.launcher" in source
    for forbidden in ("batch_size", "optimizer_lr", "save_freq", "steps:"):
        assert forbidden not in source


def test_vt_cache_instructions_defer_precompute_to_the_launcher():
    config = (ROOT / "train_vtsmolvla/configs/train.yaml").read_text(encoding="utf-8")
    assert "首次训练前运行一次" not in config
    assert "launcher 自动检查并执行" in config
    assert "bash train_vtsmolvla/scripts/train.sh" in config


def test_vt_readme_documents_the_clean_sdist_to_wheel_release_path():
    readme = (ROOT / "train_vtsmolvla/README.md").read_text(encoding="utf-8")

    assert "uv build --sdist" in readme
    assert "uv build --wheel" in readme


def test_wheel_contains_vt_runtime_assets_and_help_runs_outside_repo(tmp_path):
    sdist_dir = tmp_path / "sdist"
    completed = subprocess.run(
        [
            "uv",
            "build",
            "--sdist",
            "--no-build-isolation",
            "--no-create-gitignore",
            "--cache-dir",
            str(tmp_path / "uv-cache"),
            "--out-dir",
            str(sdist_dir),
            str(ROOT),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    (sdist_path,) = sdist_dir.glob("*.tar.gz")
    wheel_dir = tmp_path / "wheel"
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
            str(wheel_dir),
            str(sdist_path),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    (wheel_path,) = wheel_dir.glob("*.whl")
    with zipfile.ZipFile(wheel_path) as wheel:
        names = set(wheel.namelist())
        assert {
            "train_vtsmolvla/README.md",
            "train_vtsmolvla/configs/train.yaml",
            "train_vtsmolvla/scripts/train.sh",
            "train_vtsmolvla/precompute.py",
        } <= names
        assert not any(name.startswith("lerobot/policies/smolvla_jax/") for name in names)
        installed = tmp_path / "installed"
        wheel.extractall(installed)
    outside = tmp_path / "outside"
    outside.mkdir()
    environment = {**os.environ, "PYTHONPATH": str(installed)}
    legacy_lookup = subprocess.run(
        [
            sys.executable,
            "-c",
            "import importlib.util; assert importlib.util.find_spec('lerobot.policies.smolvla_jax') is None",
        ],
        cwd=outside,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )
    assert legacy_lookup.returncode == 0, legacy_lookup.stderr
    for module in (
        "train_vtsmolvla.train",
        "train_vtsmolvla.launcher",
        "train_vtsmolvla.precompute",
    ):
        result = subprocess.run(
            [sys.executable, "-m", module, "--help"],
            cwd=outside,
            env=environment,
            text=True,
            capture_output=True,
            check=False,
        )
        assert result.returncode == 0, f"{module}: {result.stderr}"

    direct_tool = subprocess.run(
        [sys.executable, str(ROOT / "tools/precompute_tactile_embeddings.py"), "--help"],
        cwd=outside,
        text=True,
        capture_output=True,
        check=False,
    )
    assert direct_tool.returncode == 0, direct_tool.stderr
