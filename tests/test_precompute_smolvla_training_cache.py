from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _write_fake_uv(path: Path, event_log: Path) -> None:
    path.write_text(
        "#!/usr/bin/env bash\n"
        "set -eu\n"
        "args=\"$*\"\n"
        "if [[ \"$args\" == *\"precompute_tactile_embeddings.py\"* ]]; then\n"
        f"  printf 'tactile gpu=%s\\n' \"${{CUDA_VISIBLE_DEVICES:-}}\" >> {event_log!s}\n"
        "  exit 0\n"
        "fi\n"
        "if [[ \"$args\" == *\"precompute_smolvla_training_cache.py\"* ]]; then\n"
        "  for ((i = 1; i <= $#; i++)); do\n"
        "    if [[ \"${!i}\" == \"--dataset-index\" ]]; then next=$((i + 1)); index=\"${!next}\"; fi\n"
        "  done\n"
        f"  printf 'offline-start index=%s gpu=%s\\n' \"$index\" \"${{CUDA_VISIBLE_DEVICES:-}}\" >> {event_log!s}\n"
        "  sleep 0.05\n"
        f"  printf 'offline-end index=%s gpu=%s\\n' \"$index\" \"${{CUDA_VISIBLE_DEVICES:-}}\" >> {event_log!s}\n"
        "  [[ \"${FAKE_FAIL_OFFLINE_INDEX:-}\" != \"$index\" ]]\n"
        "  exit\n"
        "fi\n",
        encoding="utf-8",
    )
    path.chmod(0o755)


def _project(tmp_path: Path) -> tuple[Path, Path, Path]:
    project = tmp_path / "project"
    scripts = project / "scripts"
    configs = project / "configs"
    fake_bin = tmp_path / "bin"
    scripts.mkdir(parents=True)
    configs.mkdir()
    fake_bin.mkdir()
    shutil.copy2(ROOT / "scripts" / "precompute_smolvla_training_cache.sh", scripts)
    (configs / "train_vtsmolvla_jax_tactile16.yaml").write_text("output: /tmp/output\n")
    (project / ".env.frs").write_text("export FRS_STORAGE_ROOT=/tmp/frs-storage\n")
    return project, fake_bin, fake_bin / "events.log"


def _run(project: Path, fake_bin: Path, **extra_env: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", "scripts/precompute_smolvla_training_cache.sh"],
        cwd=project,
        env=os.environ | {"PATH": f"{fake_bin}:{os.environ['PATH']}", **extra_env},
        text=True,
        capture_output=True,
        check=False,
    )


def test_precompute_runs_six_datasets_in_two_four_gpu_waves(tmp_path: Path) -> None:
    project, fake_bin, event_log = _project(tmp_path)
    _write_fake_uv(fake_bin / "uv", event_log)

    result = _run(project, fake_bin)

    assert result.returncode == 0, result.stderr
    events = event_log.read_text(encoding="utf-8").splitlines()
    assert events[0] == "tactile gpu=0"
    assert {event for event in events if event.startswith("offline-start")} == {
        f"offline-start index={index} gpu={index % 4}" for index in range(6)
    }
    assert {event for event in events if event.startswith("offline-end")} == {
        f"offline-end index={index} gpu={index % 4}" for index in range(6)
    }
    first_wave_end = max(events.index(f"offline-end index={index} gpu={index}") for index in range(4))
    second_wave_start = min(events.index(f"offline-start index={index} gpu={index % 4}") for index in (4, 5))
    assert first_wave_end < second_wave_start


def test_precompute_waits_for_every_job_and_fails_when_any_dataset_fails(tmp_path: Path) -> None:
    project, fake_bin, event_log = _project(tmp_path)
    _write_fake_uv(fake_bin / "uv", event_log)

    result = _run(project, fake_bin, FAKE_FAIL_OFFLINE_INDEX="2")

    assert result.returncode != 0
    assert "offline dataset 2 failed" in result.stderr
    events = event_log.read_text(encoding="utf-8").splitlines()
    assert {event for event in events if event.startswith("offline-end")} == {
        f"offline-end index={index} gpu={index}" for index in range(4)
    }
    assert not any("index=4" in event or "index=5" in event for event in events)
