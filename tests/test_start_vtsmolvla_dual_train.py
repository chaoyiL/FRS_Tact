from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _write_fake_commands(fake_bin: Path, event_log: Path) -> None:
    (fake_bin / "uv").write_text(
        "#!/usr/bin/env bash\n"
        "set -eu\n"
        "args=\"$*\"\n"
        "if [[ \"$args\" == *\"precompute_tactile_embeddings.py\"* ]]; then\n"
        f"  printf 'tactile gpu=%s\\n' \"${{CUDA_VISIBLE_DEVICES:-}}\" >> {event_log!s}; exit 0\n"
        "fi\n"
        "if [[ \"$args\" == *\"precompute_smolvla_training_cache.py\"* ]]; then\n"
        "  for ((i = 1; i <= $#; i++)); do [[ \"${!i}\" == \"--dataset-index\" ]] && { next=$((i + 1)); index=\"${!next}\"; }; done\n"
        f"  printf 'offline index=%s gpu=%s\\n' \"$index\" \"${{CUDA_VISIBLE_DEVICES:-}}\" >> {event_log!s}\n"
        "  [[ \"${FAKE_FAIL_OFFLINE_INDEX:-}\" != \"$index\" ]]; exit\n"
        "fi\n"
        "if [[ \"$args\" == *\" python - \"* ]]; then\n"
        f"  printf 'preflight gpu=%s\\n' \"${{CUDA_VISIBLE_DEVICES:-}}\" >> {event_log!s}; exit 0\n"
        "fi\n"
        "if [[ \"$args\" == *\"tools/train_vtsmolvla_jax.py\"* ]]; then\n"
        "  case \"$args\" in *tactile16.yaml*) name=k8;; *tactile32.yaml*) name=k21;; esac\n"
        f"  printf 'train name=%s gpu=%s\\n' \"$name\" \"${{CUDA_VISIBLE_DEVICES:-}}\" >> {event_log!s}\n"
        "  [[ \"${FAKE_FAIL_TRAIN:-}\" != \"$name\" ]]; exit\n"
        "fi\n",
        encoding="utf-8",
    )
    (fake_bin / "uv").chmod(0o755)
    (fake_bin / "nvidia-smi").write_text(
        "#!/usr/bin/env bash\n"
        "printf '%s\\n' 'NVIDIA RTX PRO 6000 Blackwell Server Edition, 595.84' 'NVIDIA RTX PRO 6000 Blackwell Server Edition, 595.84' 'NVIDIA RTX PRO 6000 Blackwell Server Edition, 595.84' 'NVIDIA RTX PRO 6000 Blackwell Server Edition, 595.84'\n",
        encoding="utf-8",
    )
    (fake_bin / "nvidia-smi").chmod(0o755)


def _project(tmp_path: Path) -> tuple[Path, Path, Path]:
    project = tmp_path / "project"
    scripts = project / "scripts"
    configs = project / "configs"
    fake_bin = tmp_path / "bin"
    scripts.mkdir(parents=True)
    configs.mkdir()
    fake_bin.mkdir()
    for script in ("precompute_smolvla_training_cache.sh", "start_vtsmolvla_dual_train.sh"):
        shutil.copy2(ROOT / "scripts" / script, scripts)
    for config in ("train_vtsmolvla_jax_tactile16.yaml", "train_vtsmolvla_jax_tactile32.yaml"):
        (configs / config).write_text("output: /tmp/output\n")
    (project / ".env.frs").write_text("export FRS_STORAGE_ROOT=/tmp/frs-storage\n")
    return project, fake_bin, fake_bin / "events.log"


def _run(project: Path, fake_bin: Path, *args: str, **extra_env: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", "scripts/start_vtsmolvla_dual_train.sh", "--foreground", *args],
        cwd=project,
        env=os.environ | {"PATH": f"{fake_bin}:{os.environ['PATH']}", **extra_env},
        text=True,
        capture_output=True,
        check=False,
    )


def test_foreground_prepares_four_datasets_before_starting_independent_gpu_pairs(tmp_path: Path) -> None:
    project, fake_bin, event_log = _project(tmp_path)
    _write_fake_commands(fake_bin, event_log)

    result = _run(project, fake_bin)

    assert result.returncode == 0, result.stderr
    events = event_log.read_text(encoding="utf-8").splitlines()
    assert events[0] == "tactile gpu=0"
    assert {event for event in events if event.startswith("offline")} == {
        f"offline index={index} gpu={index}" for index in range(4)
    }
    train_events = [event for event in events if event.startswith("train")]
    assert set(train_events) == {"train name=k8 gpu=0,1", "train name=k21 gpu=2,3"}
    assert min(events.index(event) for event in train_events) > max(
        index for index, event in enumerate(events) if event.startswith("offline")
    )


def test_precompute_failure_prevents_both_training_jobs(tmp_path: Path) -> None:
    project, fake_bin, event_log = _project(tmp_path)
    _write_fake_commands(fake_bin, event_log)

    result = _run(project, fake_bin, FAKE_FAIL_OFFLINE_INDEX="1")

    assert result.returncode != 0
    assert "preparation failed" in result.stderr
    assert not any(line.startswith("train") for line in event_log.read_text(encoding="utf-8").splitlines())


def test_foreground_reports_both_statuses_when_one_training_fails(tmp_path: Path) -> None:
    project, fake_bin, event_log = _project(tmp_path)
    _write_fake_commands(fake_bin, event_log)

    result = _run(project, fake_bin, FAKE_FAIL_TRAIN="k8")

    assert result.returncode != 0
    assert "K8 status=1" in result.stderr
    assert "K21 status=0" in result.stderr
    assert {line for line in event_log.read_text(encoding="utf-8").splitlines() if line.startswith("train")} == {
        "train name=k8 gpu=0,1",
        "train name=k21 gpu=2,3",
    }


def test_existing_interactive_session_is_not_overwritten(tmp_path: Path) -> None:
    project, fake_bin, _ = _project(tmp_path)
    _write_fake_commands(fake_bin, fake_bin / "events.log")
    (fake_bin / "tmux").write_text(
        "#!/usr/bin/env bash\n"
        "[[ \"$1\" == has-session ]] && exit 0\n"
        "exit 99\n",
        encoding="utf-8",
    )
    (fake_bin / "tmux").chmod(0o755)

    result = subprocess.run(
        ["bash", "scripts/start_vtsmolvla_dual_train.sh"],
        cwd=project,
        env=os.environ | {"PATH": f"{fake_bin}:{os.environ['PATH']}"},
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode != 0
    assert "vtsmolvla_k8" in result.stderr


def test_coordinator_tmux_forwards_original_arguments(tmp_path: Path) -> None:
    project, fake_bin, _ = _project(tmp_path)
    _write_fake_commands(fake_bin, fake_bin / "events.log")
    tmux_log = fake_bin / "tmux.log"
    (fake_bin / "tmux").write_text(
        "#!/usr/bin/env bash\n"
        "if [[ \"$1\" == has-session ]]; then exit 1; fi\n"
        f"printf '%s\\n' \"${{@: -1}}\" > {tmux_log!s}\n",
        encoding="utf-8",
    )
    (fake_bin / "tmux").chmod(0o755)

    result = subprocess.run(
        ["bash", "scripts/start_vtsmolvla_dual_train.sh", "--log-root", "logs with spaces"],
        cwd=project,
        env=os.environ | {"PATH": f"{fake_bin}:{os.environ['PATH']}"},
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    command = tmux_log.read_text(encoding="utf-8")
    assert "--coordinator" in command
    assert "--log-root logs\\ with\\ spaces" in command
