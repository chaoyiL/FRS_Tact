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
        "args=\"$*\"\n"
        "if [[ \"$args\" == *\"tools/precompute_tactile_embeddings.py\"* ]]; then\n"
        f"  printf 'precompute %s\\n' \"${{@: -1}}\" >> {call_log!s}\n"
        "  exit 0\n"
        "fi\n"
        "if [[ \"$args\" == *\"tools/train_vtsmolvla_jax.py\"* ]]; then\n"
        f"  printf 'train %s\\n' \"${{@: -1}}\" >> {call_log!s}\n"
        "  if [[ \"${FAKE_FAIL_TRAIN_K8:-0}\" == 1 && \"$args\" == *\"tactile16.yaml\"* ]]; then exit 17; fi\n"
        "  exit 0\n"
        "fi\n"
        "if [[ \"$args\" == *\" python - \"* ]]; then\n"
        "  [[ \"${CUDA_VISIBLE_DEVICES:-}\" == \"${FAKE_EXPECTED_GPUS:-0,1}\" ]] || exit 19\n"
        "  case \"$args\" in\n"
        "    *tactile16.yaml*) printf '%s\\n' /tmp/vtsmolvla-k8-output 1 '';;\n"
        "    *tactile32.yaml*) printf '%s\\n' /tmp/vtsmolvla-k21-output 1 '';;\n"
        "    *) printf '%s\\n' /tmp/vtsmolvla-output 1 '';;\n"
        "  esac\n"
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
    storage = project / ".cache"
    (project / ".env.frs").write_text(
        "\n".join(
            [
                f"export FRS_STORAGE_ROOT={storage}",
                "export FRS_VENV_DIR=/tmp/test-venv",
                "export UV_PROJECT_ENVIRONMENT=/tmp/test-venv",
                f"export UV_CACHE_DIR={storage / '.cache' / 'uv'}",
                f"export HF_HOME={storage / 'huggingface'}",
                f"export HF_HUB_CACHE={storage / 'huggingface' / 'hub'}",
                f"export HF_DATASETS_CACHE={storage / 'huggingface' / 'datasets_arrow'}",
                f"export HF_LEROBOT_HOME={storage / 'huggingface' / 'lerobot'}",
                f"export TMPDIR={storage / 'tmp'}",
                "",
            ]
        ),
        encoding="utf-8",
    )
    for name in (
        "train_vtsmolvla_jax.yaml",
        "train_vtsmolvla_jax_tactile16.yaml",
        "train_vtsmolvla_jax_tactile32.yaml",
    ):
        (configs / name).write_text("output: /tmp/unused\n", encoding="utf-8")
    return project, fake_bin, fake_bin / "uv-calls.log"


def _run_launcher(
    project: Path, fake_bin: Path, *arguments: str, **extra_env: str
) -> subprocess.CompletedProcess[str]:
    env = os.environ | {
        "PATH": f"{fake_bin}:{os.environ['PATH']}",
        "FRS_FOREGROUND": "1",
        **extra_env,
    }
    return subprocess.run(
        ["bash", "scripts/start_vtsmolvla_train.sh", *arguments],
        cwd=project,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


def test_default_precomputes_once_then_trains_k8_and_k21(tmp_path: Path) -> None:
    project, fake_bin, call_log = _prepare_launcher_project(tmp_path)
    _write_fake_uv(fake_bin / "uv", call_log)

    result = _run_launcher(project, fake_bin)

    assert result.returncode == 0, result.stderr
    assert call_log.read_text(encoding="utf-8").splitlines() == [
        "precompute " + str(project / "configs/train_vtsmolvla_jax_tactile16.yaml"),
        "train " + str(project / "configs/train_vtsmolvla_jax_tactile16.yaml"),
        "train " + str(project / "configs/train_vtsmolvla_jax_tactile32.yaml"),
    ]


def test_k8_failure_does_not_start_k21(tmp_path: Path) -> None:
    project, fake_bin, call_log = _prepare_launcher_project(tmp_path)
    _write_fake_uv(fake_bin / "uv", call_log)

    result = _run_launcher(project, fake_bin, FAKE_FAIL_TRAIN_K8="1")

    assert result.returncode == 17
    assert call_log.read_text(encoding="utf-8").splitlines() == [
        "precompute " + str(project / "configs/train_vtsmolvla_jax_tactile16.yaml"),
        "train " + str(project / "configs/train_vtsmolvla_jax_tactile16.yaml"),
    ]


@pytest.mark.parametrize(
    ("arguments", "config_name"),
    [
        (["--config", "configs/explicit.yaml"], "explicit.yaml"),
        (["--config=configs/equal form.yaml"], "equal form.yaml"),
        (["--experiment", "k8"], "train_vtsmolvla_jax_tactile16.yaml"),
        (["--experiment=k21"], "train_vtsmolvla_jax_tactile32.yaml"),
    ],
)
def test_launcher_selects_legacy_or_requested_single_config(
    tmp_path: Path, arguments: list[str], config_name: str
) -> None:
    project, fake_bin, call_log = _prepare_launcher_project(tmp_path)
    config_path = project / "configs" / config_name
    config_path.write_text("output: /tmp/unused\n", encoding="utf-8")
    _write_fake_uv(fake_bin / "uv", call_log)

    result = _run_launcher(project, fake_bin, *arguments)

    assert result.returncode == 0, result.stderr
    assert call_log.read_text(encoding="utf-8").splitlines() == [
        "precompute " + str(config_path),
        "train " + str(config_path),
    ]


@pytest.mark.parametrize(
    "arguments",
    [
        ["--unknown"],
        ["--config"],
        ["--config", "first.yaml", "--config=second.yaml"],
        ["--experiment", "bad"],
        ["--experiment", "both", "--config", "configs/explicit.yaml"],
        ["--gpus", "0"],
    ],
)
def test_launcher_rejects_invalid_arguments_before_preflight(
    tmp_path: Path, arguments: list[str]
) -> None:
    project, fake_bin, call_log = _prepare_launcher_project(tmp_path)
    _write_fake_uv(fake_bin / "uv", call_log)

    result = _run_launcher(project, fake_bin, *arguments)

    assert result.returncode != 0
    assert not call_log.exists(), "invalid arguments must fail before the JAX preflight"


def test_launcher_sets_two_visible_gpus_before_jax_preflight(tmp_path: Path) -> None:
    project, fake_bin, call_log = _prepare_launcher_project(tmp_path)
    _write_fake_uv(fake_bin / "uv", call_log)

    result = _run_launcher(project, fake_bin, "--experiment", "k8", "--gpus", "3,5", FAKE_EXPECTED_GPUS="3,5")

    assert result.returncode == 0, result.stderr


def test_launcher_tmux_forwards_the_original_argument_vector(tmp_path: Path) -> None:
    project, fake_bin, call_log = _prepare_launcher_project(tmp_path)
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
        [
            "bash",
            "scripts/start_vtsmolvla_train.sh",
            "--experiment",
            "k8",
            "--gpus",
            "3,5",
            "--session",
            "my session",
        ],
        cwd=project,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    inner_command = tmux_log.read_text(encoding="utf-8")
    assert "--experiment k8" in inner_command
    assert "--gpus 3\\,5" in inner_command
    assert "--session my\\ session" in inner_command
