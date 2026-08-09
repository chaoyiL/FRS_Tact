from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


def _write_executable(path: Path, body: str) -> None:
    path.write_text("#!/usr/bin/env bash\nset -Eeuo pipefail\n" + body, encoding="utf-8")
    path.chmod(0o755)


def _prepare_setup_project(tmp_path: Path) -> tuple[Path, dict[str, str], Path, Path]:
    project = tmp_path / "project"
    scripts = project / "scripts"
    fake_bin = tmp_path / "bin"
    storage = tmp_path / "storage"
    scripts.mkdir(parents=True)
    fake_bin.mkdir()
    shutil.copy2(ROOT / "scripts" / "setup_env.sh", scripts)

    command_log = tmp_path / "commands.log"
    program_log = tmp_path / "verification-programs.log"
    operation_log = tmp_path / "device-operations.log"
    fake_modules = tmp_path / "fake-modules"
    (fake_modules / "jax" / "_src").mkdir(parents=True)
    (fake_modules / "jax" / "__init__.py").write_text(
        """import os
from pathlib import Path

import numpy as np


class Device:
    platform = "gpu"


def devices():
    return [Device() for _ in range(int(os.environ.get("FAKE_JAX_DEVICE_COUNT", "4")))]


def _record(name):
    with Path(os.environ["FAKE_DEVICE_OPERATION_LOG"]).open("a", encoding="utf-8") as file:
        file.write(name + "\\n")


def device_put(value, sharding):
    _record("device_put")
    return np.asarray(value)


def device_get(value):
    _record("device_get")
    return value
""",
        encoding="utf-8",
    )
    (fake_modules / "jax" / "numpy.py").write_text(
        """import os
from pathlib import Path

import numpy as np


def sum(value, axis=None):
    with Path(os.environ["FAKE_DEVICE_OPERATION_LOG"]).open("a", encoding="utf-8") as file:
        file.write("sharded_sum\\n")
    return np.sum(value, axis=axis) + float(os.environ.get("FAKE_SHARDED_SUM_OFFSET", "0"))
""",
        encoding="utf-8",
    )
    (fake_modules / "jax" / "sharding.py").write_text(
        """class Mesh:
    def __init__(self, devices, axis_names):
        self.devices = devices
        self.axis_names = axis_names


class NamedSharding:
    def __init__(self, mesh, spec):
        self.mesh = mesh
        self.spec = spec


class PartitionSpec:
    def __init__(self, *partitions):
        self.partitions = partitions
""",
        encoding="utf-8",
    )
    (fake_modules / "jax" / "_src" / "__init__.py").write_text("", encoding="utf-8")
    (fake_modules / "jax" / "_src" / "lib.py").write_text(
        """import os


class _CudaVersions:
    @staticmethod
    def cuda_runtime_get_version():
        return int(os.environ.get("FAKE_CUDA_VERSION", "12080"))

    @staticmethod
    def cudnn_get_version():
        return int(os.environ.get("FAKE_CUDNN_VERSION", "91900"))


cuda_versions = _CudaVersions()
""",
        encoding="utf-8",
    )
    (fake_modules / "torch.py").write_text(
        """import os


class _Cuda:
    @staticmethod
    def is_available():
        return os.environ.get("FAKE_TORCH_CUDA_AVAILABLE", "1") == "1"

    @staticmethod
    def device_count():
        return int(os.environ.get("FAKE_TORCH_DEVICE_COUNT", "4"))


cuda = _Cuda()
""",
        encoding="utf-8",
    )
    _write_executable(
        fake_bin / "uv",
        f"""
printf 'uv %s\\n' "$*" >> {command_log}
if [[ "${{1:-}}" == "--version" ]]; then
    echo 'uv 0.8.0'
    exit 0
fi
if [[ "${{1:-}} ${{2:-}} ${{3:-}} ${{4:-}}" == "run --no-sync python -" ]]; then
    program="$(mktemp)"
    cat >"${{program}}"
    printf '\\n### PROGRAM ###\\n' >> {program_log}
    cat "${{program}}" >> {program_log}
    if grep -q 'JAX devices' "${{program}}"; then
        PYTHONPATH={fake_modules} {sys.executable} "${{program}}"
    fi
    exit 0
fi
exit 0
""",
    )
    _write_executable(
        fake_bin / "nvidia-smi",
        f"""
printf 'nvidia-smi %s\\n' "$*" >> {command_log}
if [[ "$*" == *'name,driver_version'* ]]; then
    printf '%b' "${{FAKE_GPU_ROWS:-NVIDIA RTX PRO 6000 Blackwell Server Edition, 595.84\\nNVIDIA RTX PRO 6000 Blackwell Server Edition, 595.84\\nNVIDIA RTX PRO 6000 Blackwell Server Edition, 595.84\\nNVIDIA RTX PRO 6000 Blackwell Server Edition, 595.84\\n}}"
else
    printf '%b' "${{FAKE_GPU_ROWS:-NVIDIA RTX PRO 6000 Blackwell Server Edition, 595.84\\nNVIDIA RTX PRO 6000 Blackwell Server Edition, 595.84\\nNVIDIA RTX PRO 6000 Blackwell Server Edition, 595.84\\nNVIDIA RTX PRO 6000 Blackwell Server Edition, 595.84\\n}}"
fi
""",
    )
    for command in ("mktemp", "chmod", "mv"):
        _write_executable(
            fake_bin / command,
            f"printf '{command} %s\\n' \"$*\" >> {command_log}\n"
            f'exec /usr/bin/{command} "$@"\n',
        )

    home = tmp_path / "home"
    home.mkdir()
    (home / ".bashrc").write_text("# keep me unchanged\n", encoding="utf-8")
    env = os.environ | {
        "PATH": f"{fake_bin}:{os.environ['PATH']}",
        "HOME": str(home),
        "FRS_STORAGE_ROOT": str(storage),
        "FRS_VENV_DIR": str(storage / ".venvs" / "frs_tact"),
        "FRS_UV_CACHE_DIR": str(storage / ".cache" / "uv"),
        "FAKE_TORCH_DEVICE_COUNT": "4",
        "FAKE_JAX_DEVICE_COUNT": "4",
        "FAKE_DEVICE_OPERATION_LOG": str(operation_log),
    }
    return project, env, command_log, program_log


def _run_setup(project: Path, env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", "scripts/setup_env.sh"],
        cwd=project,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


def _parse_environment_file(path: Path) -> dict[str, str]:
    keys = (
        "FRS_STORAGE_ROOT",
        "FRS_VENV_DIR",
        "UV_PROJECT_ENVIRONMENT",
        "UV_CACHE_DIR",
        "HF_HOME",
        "HF_HUB_CACHE",
        "HF_DATASETS_CACHE",
        "HF_LEROBOT_HOME",
        "TMPDIR",
        "UV_DEFAULT_INDEX",
        "UV_HTTP_TIMEOUT",
        "HF_ENDPOINT",
    )
    command = "source \"$1\"; printf '%s\\0' " + " ".join(f'\"${{{key}}}\"' for key in keys)
    result = subprocess.run(
        ["bash", "-c", command, "bash", str(path)],
        text=False,
        capture_output=True,
        check=True,
    )
    values = result.stdout.decode().split("\0")[:-1]
    return dict(zip(keys, values, strict=True))


def test_default_environment_contract_uses_server_storage_not_checkout_venv(tmp_path: Path) -> None:
    project = tmp_path / "project"
    scripts = project / "scripts"
    scripts.mkdir(parents=True)
    script = (ROOT / "scripts" / "setup_env.sh").read_text(encoding="utf-8")
    setup_library = scripts / "setup_env_library.sh"
    setup_library.write_text(
        script.rsplit('\nmain "$@"', maxsplit=1)[0]
        + "\nexport UV_CACHE_DIR=\"${UV_CACHE_DIR_VALUE}\"\n"
        + "export UV_DEFAULT_INDEX=\"${UV_DEFAULT_INDEX_VALUE}\"\n"
        + "export UV_HTTP_TIMEOUT=\"${UV_HTTP_TIMEOUT_VALUE}\"\n"
        + "export HF_HOME=\"${HF_HOME_VALUE}\"\n"
        + "export HF_HUB_CACHE=\"${HF_HUB_CACHE_VALUE}\"\n"
        + "export HF_DATASETS_CACHE=\"${HF_DATASETS_CACHE_VALUE}\"\n"
        + "export HF_LEROBOT_HOME=\"${HF_LEROBOT_HOME_VALUE}\"\n"
        + "export HF_ENDPOINT=\"${HF_ENDPOINT_VALUE}\"\n"
        + "export TMPDIR=\"${TMPDIR_VALUE}\"\n"
        + "write_environment_file\n",
        encoding="utf-8",
    )
    env = {
        key: value
        for key, value in os.environ.items()
        if key not in {"FRS_STORAGE_ROOT", "FRS_VENV_DIR", "FRS_UV_CACHE_DIR", "UV_CACHE_DIR"}
    }

    result = subprocess.run(
        ["bash", str(setup_library)],
        cwd=project,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    parsed_env = _parse_environment_file(project / ".env.frs")
    assert parsed_env["FRS_STORAGE_ROOT"] == "/DATA/ljl/substage"
    assert parsed_env["FRS_VENV_DIR"] == "/home/ljl/.venvs/frs_tact"
    assert parsed_env["UV_PROJECT_ENVIRONMENT"] == parsed_env["FRS_VENV_DIR"]
    assert parsed_env["UV_CACHE_DIR"] == "/DATA/ljl/substage/.cache/uv"
    assert parsed_env["HF_HOME"] == "/DATA/ljl/substage/huggingface"
    assert parsed_env["TMPDIR"] == "/DATA/ljl/substage/tmp"
    assert parsed_env["FRS_VENV_DIR"] != str(project / ".venv")


def test_setup_persists_authoritative_environment_atomically(tmp_path: Path) -> None:
    project, env, command_log, _ = _prepare_setup_project(tmp_path)

    result = _run_setup(project, env)

    assert result.returncode == 0, result.stderr
    storage = Path(env["FRS_STORAGE_ROOT"])
    parsed_env = _parse_environment_file(project / ".env.frs")
    assert parsed_env == {
        "FRS_STORAGE_ROOT": str(storage),
        "FRS_VENV_DIR": str(storage / ".venvs" / "frs_tact"),
        "UV_PROJECT_ENVIRONMENT": str(storage / ".venvs" / "frs_tact"),
        "UV_CACHE_DIR": str(storage / ".cache" / "uv"),
        "HF_HOME": str(storage / "huggingface"),
        "HF_HUB_CACHE": str(storage / "huggingface" / "hub"),
        "HF_DATASETS_CACHE": str(storage / "huggingface" / "datasets_arrow"),
        "HF_LEROBOT_HOME": str(storage / "huggingface" / "lerobot"),
        "TMPDIR": str(storage / "tmp"),
        "UV_DEFAULT_INDEX": "https://pypi.tuna.tsinghua.edu.cn/simple",
        "UV_HTTP_TIMEOUT": "300",
        "HF_ENDPOINT": "https://hf-mirror.com",
    }
    calls = command_log.read_text(encoding="utf-8")
    assert f"mktemp --tmpdir={project} .env.frs.XXXXXX" in calls
    assert "chmod 600 " in calls
    assert f"mv " in calls and f" {project / '.env.frs'}" in calls
    assert not list(project.glob(".env.frs.*"))
    assert (Path(env["HOME"]) / ".bashrc").read_text(encoding="utf-8") == "# keep me unchanged\n"


def test_setup_uses_project_lock_instead_of_global_process_scan() -> None:
    script = (ROOT / "scripts" / "setup_env.sh").read_text(encoding="utf-8")

    assert 'exec 9>"${STORAGE_ROOT}/.locks/frs-setup.lock"' in script
    assert 'flock -n 9 || fail "另一个 FRS 环境安装正在运行"' in script
    assert 'command -v flock >/dev/null 2>&1 || fail "找不到 flock（util-linux）"' in script
    assert "ps -eo" not in script
    assert ".bashrc" not in script


def test_dependency_mirror_contract_keeps_cuda_wheels_on_official_index() -> None:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    indexes = {item["name"]: item for item in project["tool"]["uv"]["index"]}

    assert indexes["tsinghua-pypi"] == {
        "name": "tsinghua-pypi",
        "url": "https://pypi.tuna.tsinghua.edu.cn/simple",
        "default": True,
    }
    assert indexes["pytorch-cu128"] == {
        "name": "pytorch-cu128",
        "url": "https://download.pytorch.org/whl/cu128",
        "explicit": True,
    }
    lock = (ROOT / "uv.lock").read_text(encoding="utf-8")
    assert 'registry = "https://pypi.tuna.tsinghua.edu.cn/simple"' in lock
    assert 'index = "https://download.pytorch.org/whl/cu128"' in lock


@pytest.mark.parametrize(
    ("gpu_rows", "should_pass"),
    [
        (("NVIDIA RTX PRO 6000 Blackwell Server Edition, 595.84\n") * 4, True),
        (("NVIDIA RTX PRO 6000 Blackwell Server Edition, 570.85\n") + ("NVIDIA RTX PRO 6000 Blackwell Server Edition, 595.84\n") * 3, False),
        (("NVIDIA RTX PRO 6000 Blackwell Server Edition, 595.84\n") * 3, False),
        (("NVIDIA RTX PRO 6000 Blackwell Server Edition, 595.84\n") * 3 + "NVIDIA H100 80GB HBM3, 595.84\n", False),
    ],
)
def test_setup_requires_exactly_four_rtx_pro_6000s_and_minimum_driver(
    tmp_path: Path, gpu_rows: str, should_pass: bool
) -> None:
    project, env, _, _ = _prepare_setup_project(tmp_path)
    env["FAKE_GPU_ROWS"] = gpu_rows

    result = _run_setup(project, env)

    assert (result.returncode == 0) is should_pass, result.stdout + result.stderr


@pytest.mark.parametrize("count_variable", ["FAKE_TORCH_DEVICE_COUNT", "FAKE_JAX_DEVICE_COUNT"])
def test_setup_rejects_non_four_device_framework_runtime(
    tmp_path: Path, count_variable: str
) -> None:
    project, env, _, _ = _prepare_setup_project(tmp_path)
    env[count_variable] = "1"

    result = _run_setup(project, env)

    assert result.returncode != 0


def test_setup_rejects_incorrect_sharded_sum(tmp_path: Path) -> None:
    project, env, _, _ = _prepare_setup_project(tmp_path)
    env["FAKE_SHARDED_SUM_OFFSET"] = "1"

    result = _run_setup(project, env)

    assert result.returncode != 0
    assert "Not equal to tolerance" in result.stderr


def test_setup_lock_contention_fails_before_uv(tmp_path: Path) -> None:
    import fcntl

    project, env, command_log, _ = _prepare_setup_project(tmp_path)
    lock_path = Path(env["FRS_STORAGE_ROOT"]) / ".locks" / "frs-setup.lock"
    lock_path.parent.mkdir(parents=True)
    with lock_path.open("w", encoding="utf-8") as lock_file:
        fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
        result = _run_setup(project, env)

    assert result.returncode != 0
    assert "另一个 FRS 环境安装正在运行" in result.stderr
    assert not command_log.exists(), "setup must acquire its project lock before invoking uv"


def test_setup_verification_program_checks_cuda_stack_and_sharded_sum(tmp_path: Path) -> None:
    project, env, _, program_log = _prepare_setup_project(tmp_path)

    result = _run_setup(project, env)

    assert result.returncode == 0, result.stderr
    program = program_log.read_text(encoding="utf-8")
    assert "cuda_versions" in program
    assert "cudnn" in program.lower()
    assert "nccl" in program.lower()
    assert "libdevice.10.bc" in program
    assert "NamedSharding" in program
    assert "PartitionSpec" in program
    assert "jax.device_put" in program
    assert "assert_allclose" in program
    operations = Path(env["FAKE_DEVICE_OPERATION_LOG"]).read_text(encoding="utf-8").splitlines()
    assert operations == ["device_put", "sharded_sum", "device_get"]
