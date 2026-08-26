from __future__ import annotations

import ast
import os
import re
import shutil
import subprocess
import sys
import tomllib
from pathlib import Path

import torch

from deploy_baseline_pi05.direct_decoder import DirectDecoderConfig as DeployDecoderConfig
from deploy_baseline_pi05.direct_decoder import DirectTactileActionDecoder as DeployDecoder
from train_baseline_pi05.model import DirectDecoderConfig as TrainDecoderConfig
from train_baseline_pi05.model import DirectTactileActionDecoder as TrainDecoder


REPO_ROOT = Path(__file__).resolve().parents[2]
DEPLOY_ROOT = REPO_ROOT / "deploy_baseline_pi05"
CONFIG = DEPLOY_ROOT / "configs" / "deploy_baseline_pi05.yaml"
LAUNCHER = DEPLOY_ROOT / "scripts" / "start_baseline_pi05.sh"
MANIFEST = DEPLOY_ROOT / "pyproject.toml"
README = DEPLOY_ROOT / "README.md"


def _production_sources() -> list[Path]:
    return sorted(
        path
        for path in DEPLOY_ROOT.rglob("*.py")
        if "tests" not in path.parts and "__pycache__" not in path.parts
    )


def test_deploy_package_has_no_original_or_frs_refinement_runtime_imports() -> None:
    forbidden_modules = (
        "deploy_pi05",
        "deploy_smolvla",
        "train_frs",
        "train_smolvla_frs",
        "train_baseline_pi05",
    )
    forbidden_symbols = {
        "frs_runtime",
        "reverse_integrate",
        "forward_integrate",
        "sample_and_reverse",
        "x_base",
        "residual_action",
        "action_residual",
        "flow_matching_decoder",
        "flow_matching_refinement",
    }

    for path in _production_sources():
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(path))
        imported = [
            node.module or ""
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom)
        ]
        imported.extend(
            alias.name
            for node in ast.walk(tree)
            if isinstance(node, ast.Import)
            for alias in node.names
        )
        assert not any(
            module == prefix or module.startswith(f"{prefix}.")
            for module in imported
            for prefix in forbidden_modules
        ), path

        identifiers = {
            node.id for node in ast.walk(tree) if isinstance(node, ast.Name)
        } | {
            node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)
        }
        assert identifiers.isdisjoint(forbidden_symbols), path
        assert not any(re.search(rf"\b{re.escape(symbol)}\b", source) for symbol in forbidden_symbols), path


def test_train_and_deploy_decoder_are_bit_exact_on_cpu() -> None:
    train_config = TrainDecoderConfig()
    deploy_config = DeployDecoderConfig()
    assert train_config.to_primitive() == deploy_config.to_primitive()

    torch.manual_seed(7)
    train_model = TrainDecoder(train_config).eval()
    deploy_model = DeployDecoder(deploy_config).eval()
    train_state = train_model.state_dict()
    deploy_state = deploy_model.state_dict()
    assert list(train_state) == list(deploy_state)
    assert [value.shape for value in train_state.values()] == [
        value.shape for value in deploy_state.values()
    ]
    deploy_model.load_state_dict(train_state, strict=True)

    coarse = torch.randn(2, 50, 20)
    tactile = torch.randn(2, 4, 512)
    with torch.inference_mode():
        train_output = train_model(coarse, tactile)
        deploy_output = deploy_model(coarse, tactile)
    assert torch.equal(train_output, deploy_output)


def test_remote_check_is_a_fresh_dependency_light_process() -> None:
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join(
        (str(DEPLOY_ROOT / "src"), str(REPO_ROOT))
    )
    environment["PYTHONPROFILEIMPORTTIME"] = "1"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "deploy_baseline_pi05.remote_client",
            "--config",
            str(CONFIG),
            "--check",
        ],
        cwd=REPO_ROOT,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "deploy config sha256" in result.stdout
    imported = result.stderr.lower()
    assert not re.search(r"import time:.*\b(jax|torch|websockets)(?:\.|\b)", imported)


def _fake_launcher_project(tmp_path: Path) -> tuple[Path, Path, Path]:
    project = tmp_path / "deploy_baseline_pi05"
    script = project / "scripts" / LAUNCHER.name
    script.parent.mkdir(parents=True)
    shutil.copy2(LAUNCHER, script)
    config = project / "config.yaml"
    config.write_text("test: true\n", encoding="utf-8")
    python = project / ".venv" / "bin" / "python"
    python.parent.mkdir(parents=True)
    python.write_text(
        "#!/usr/bin/env bash\n"
        "{\n"
        "  printf 'ARG=%s\\n' \"$@\"\n"
        "  printf 'PYTHONSAFEPATH=%s\\n' \"${PYTHONSAFEPATH-}\"\n"
        "  printf 'PYTHONUNBUFFERED=%s\\n' \"${PYTHONUNBUFFERED-}\"\n"
        "  printf 'PYTHONPATH=%s\\n' \"${PYTHONPATH-}\"\n"
        "  printf 'XLA=%s\\n' \"${XLA_PYTHON_CLIENT_PREALLOCATE-}\"\n"
        "  printf 'TOKEN=%s\\n' \"${VB_ROBOT_TOKEN-}\"\n"
        "} > \"$CAPTURE_FILE\"\n",
        encoding="utf-8",
    )
    python.chmod(0o755)
    return script, config, project


def test_launcher_uses_only_project_python_and_forwards_real_check(tmp_path: Path) -> None:
    script, config, project = _fake_launcher_project(tmp_path)
    capture = tmp_path / "capture.txt"
    environment = os.environ.copy()
    environment.update(
        {
            "CAPTURE_FILE": str(capture),
            "PYTHONPATH": "incoming-path",
        }
    )
    environment.pop("VB_ROBOT_TOKEN", None)
    environment.pop("VB3_TOKEN_FILE", None)
    result = subprocess.run(
        [
            "bash",
            str(script),
            "--config",
            str(config),
            "--check",
            "--max-iterations",
            "3",
        ],
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    lines = capture.read_text(encoding="utf-8").splitlines()
    assert lines[:6] == [
        "ARG=-m",
        "ARG=deploy_baseline_pi05.remote_client",
        "ARG=--config",
        f"ARG={config.resolve()}",
        "ARG=--check",
        "ARG=--max-iterations",
    ]
    assert "ARG=3" in lines
    assert "PYTHONSAFEPATH=1" in lines
    assert "PYTHONUNBUFFERED=1" in lines
    assert "XLA=false" in lines
    pythonpath = next(line.removeprefix("PYTHONPATH=") for line in lines if line.startswith("PYTHONPATH="))
    entries = pythonpath.split(os.pathsep)
    assert entries[:2] == [str(project / "src"), str(project.parent)]
    assert entries[-1] == "incoming-path"
    assert "TOKEN=" in lines


def test_launcher_loads_optional_token_file_for_real_run(tmp_path: Path) -> None:
    script, config, _project = _fake_launcher_project(tmp_path)
    capture = tmp_path / "capture.txt"
    token_file = tmp_path / "robot.token"
    token_file.write_text("\nserver-secret\nignored\n", encoding="utf-8")
    environment = os.environ.copy()
    environment.update(
        {
            "CAPTURE_FILE": str(capture),
            "VB3_TOKEN_FILE": str(token_file),
        }
    )
    environment.pop("VB_ROBOT_TOKEN", None)
    result = subprocess.run(
        ["bash", str(script), "--config", str(config), "--max-iterations", "1"],
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "TOKEN=server-secret" in capture.read_text(encoding="utf-8").splitlines()


def test_launcher_rejects_a_missing_project_environment(tmp_path: Path) -> None:
    project = tmp_path / "deploy_baseline_pi05"
    script = project / "scripts" / LAUNCHER.name
    script.parent.mkdir(parents=True)
    shutil.copy2(LAUNCHER, script)
    result = subprocess.run(
        ["bash", str(script), "--check"],
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode != 0
    assert str(project / ".venv" / "bin" / "python") in result.stderr


def test_manifest_is_python312_runtime_only_and_packages_both_modules() -> None:
    manifest = tomllib.loads(MANIFEST.read_text(encoding="utf-8"))
    project = manifest["project"]
    assert project["requires-python"] == ">=3.12,<3.13"
    dependencies = "\n".join(project["dependencies"]).lower()
    for required in (
        "torch",
        "jax",
        "flax",
        "orbax-checkpoint",
        "websockets",
        "msgpack",
        "pyyaml",
        "opencv-python-headless",
        "safetensors",
    ):
        assert re.search(rf"(?m)^{re.escape(required)}(?:\[|[<>=!~;])", dependencies)
    for excluded in ("datasets", "pandas", "pyarrow", "matplotlib", "optax"):
        assert not re.search(rf"(?m)^{excluded}(?:\[|[<>=!~;])", dependencies)

    setuptools = manifest["tool"]["setuptools"]
    assert setuptools["package-dir"]["deploy_baseline_pi05"] == "."
    assert setuptools["package-dir"]["lerobot"] == "src/lerobot"
    packages = set(setuptools["packages"])
    assert "deploy_baseline_pi05" in packages
    assert "deploy_baseline_pi05.tactile_runtime" in packages
    assert "lerobot.policies.pi05_jax" in packages


def test_readme_documents_the_fixed_direct_decoder_and_server_handoff() -> None:
    text = README.read_text(encoding="utf-8")
    compact = text.replace(" ", "")
    for required in (
        "[B,50,20]",
        "[B,4,512]",
        "left0",
        "right0",
        "left1",
        "right1",
        "2层",
        "frs_steering_v1",
        "vitac",
        "--check",
        "--max-iterations",
        "fail-stop",
        "trace",
        "YAML",
        "GPU",
        "robot",
    ):
        assert required.lower() in compact.lower()
    for forbidden_behavior in ("无FRS", "无两次flowmatching积分", "无residual", "无x_base"):
        assert forbidden_behavior.lower() in compact.lower()
