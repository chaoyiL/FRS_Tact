"""Black-box coverage for the rerunnable FRS deployment downloader."""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import stat
import subprocess
import sys

import numpy as np


SCRIPT = Path(__file__).resolve().parents[1] / "deploy_smolvla/scripts/download.sh"
BASE_DIR = Path("checkpoints/model/pick_tube_02_3w_jax")
FRS_DIR = Path("checkpoints/frs/frs_0809_02")
ENCODER_DIR = Path("checkpoints/encoder/encoder_ckpt_0809")
BASE_SIDECARS = (
    "config.json",
    "train_config.json",
    "policy_preprocessor.json",
    "policy_postprocessor.json",
    "policy_preprocessor_step_5_normalizer_processor.safetensors",
    "policy_postprocessor_step_0_unnormalizer_processor.safetensors",
)


def make_project(tmp_path: Path) -> tuple[Path, Path, dict[str, str]]:
    project = tmp_path / "project"
    script_path = project / "deploy_smolvla/scripts/download.sh"
    script_path.parent.mkdir(parents=True)
    shutil.copy2(SCRIPT, script_path)
    (project / "tools").mkdir()
    (project / "tools/merge_smolvla_peft_to_jax.py").touch()
    (project / "deploy_smolvla/src").mkdir(parents=True)
    (project / "deploy_smolvla/src/download_ckpt.py").touch()

    bin_dir = project / "bin"
    bin_dir.mkdir()
    log_path = project / "calls.log"
    log_path.touch()
    fake_uv = bin_dir / "uv"
    fake_uv.write_text(
        """#!/usr/bin/env bash
set -euo pipefail

log_path="${FRS_TEST_LOG:?}"
printf '%s\\n' "$*" >> "$log_path"

output=""
for ((index = 1; index <= $#; index++)); do
    if [[ "${!index}" == "--output" || "${!index}" == "--local-dir" || "${!index}" == "--output-dir" ]]; then
        next=$((index + 1))
        output="${!next}"
    fi
done

if [[ "$*" == *"merge_smolvla_peft_to_jax.py"* ]]; then
    mkdir -p "$output"
    printf 'model' > "$output/model.safetensors"
    for sidecar in config.json train_config.json policy_preprocessor.json policy_postprocessor.json policy_preprocessor_step_5_normalizer_processor.safetensors policy_postprocessor_step_0_unnormalizer_processor.safetensors; do
        printf '{}' > "$output/$sidecar"
    done
    printf '%s' '{"source_base":"lerobot/smolvla_base","source_adapter":"KaiyueChen/pick_tube_02_3w","adapter_revision":"31d819d8844de98174ede123f894adbf7b4372ef","base_revision":"c83c3163b8ca9b7e67c509fffd9121e66cb96205"}' > "$output/conversion_manifest.json"
elif [[ "$*" == *"hf download KaiyueChen/frs_0809_02"* || "$*" == *"download_ckpt.py"* ]]; then
    mkdir -p "$output"
    printf '%s' '{"params_file":"params.npz"}' > "$output/checkpoint.json"
    "$FRS_DOWNLOAD_PYTHON" - "$output/params.npz" <<'PY'
import numpy as np
import sys
np.savez(sys.argv[1], value=np.array([1]))
PY
fi
""",
        encoding="utf-8",
    )
    fake_uv.chmod(fake_uv.stat().st_mode | stat.S_IXUSR)
    env = {
        **os.environ,
        "FRS_DOWNLOAD_UV": str(fake_uv),
        "FRS_DOWNLOAD_PYTHON": sys.executable,
        "FRS_TEST_LOG": str(log_path),
    }
    return project, log_path, env


def run_download(project: Path, env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", "deploy_smolvla/scripts/download.sh"],
        cwd=project,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


def write_complete_base(project: Path) -> None:
    directory = project / BASE_DIR
    directory.mkdir(parents=True)
    (directory / "model.safetensors").write_bytes(b"model")
    for filename in BASE_SIDECARS:
        (directory / filename).write_text("{}", encoding="utf-8")
    (directory / "conversion_manifest.json").write_text(
        json.dumps(
            {
                "source_base": "lerobot/smolvla_base",
                "source_adapter": "KaiyueChen/pick_tube_02_3w",
                "adapter_revision": "31d819d8844de98174ede123f894adbf7b4372ef",
                "base_revision": "c83c3163b8ca9b7e67c509fffd9121e66cb96205",
            }
        ),
        encoding="utf-8",
    )


def write_complete_checkpoint(directory: Path) -> None:
    directory.mkdir(parents=True)
    (directory / "checkpoint.json").write_text(
        json.dumps({"params_file": "params.npz"}), encoding="utf-8"
    )
    np.savez(directory / "params.npz", value=np.array([1]))


def write_complete_frs(project: Path) -> None:
    write_complete_checkpoint(project / FRS_DIR)


def write_complete_encoder(project: Path) -> None:
    write_complete_checkpoint(project / ENCODER_DIR)


def write_incomplete_frs(project: Path) -> None:
    directory = project / FRS_DIR
    directory.mkdir(parents=True)
    (directory / "checkpoint.json").write_text(
        json.dumps({"params_file": "params-missing.npz"}), encoding="utf-8"
    )


def test_downloads_all_missing_assets(tmp_path: Path) -> None:
    project, log_path, env = make_project(tmp_path)
    result = run_download(project, env)

    assert result.returncode == 0, result.stderr
    calls = log_path.read_text(encoding="utf-8")
    assert "merge_smolvla_peft_to_jax.py" in calls
    assert "hf download KaiyueChen/frs_0809_02" in calls
    assert "download_ckpt.py" in calls
    assert (project / "checkpoints/model/pick_tube_02_3w_jax/model.safetensors").is_file()
    assert (project / "checkpoints/frs/frs_0809_02/checkpoint.json").is_file()
    assert (project / "checkpoints/encoder/encoder_ckpt_0809/checkpoint.json").is_file()


def test_skips_complete_assets_and_downloads_missing_frs(tmp_path: Path) -> None:
    project, log_path, env = make_project(tmp_path)
    write_complete_base(project)
    write_complete_encoder(project)
    result = run_download(project, env)

    assert result.returncode == 0, result.stderr
    calls = log_path.read_text(encoding="utf-8")
    assert "hf download KaiyueChen/frs_0809_02" in calls
    assert "merge_smolvla_peft_to_jax.py" not in calls
    assert "download_ckpt.py" not in calls


def test_repairs_frs_metadata_with_missing_params(tmp_path: Path) -> None:
    project, log_path, env = make_project(tmp_path)
    write_complete_base(project)
    write_incomplete_frs(project)
    write_complete_encoder(project)
    result = run_download(project, env)

    assert result.returncode == 0, result.stderr
    assert "hf download KaiyueChen/frs_0809_02" in log_path.read_text(encoding="utf-8")


def test_skips_all_complete_assets(tmp_path: Path) -> None:
    project, log_path, env = make_project(tmp_path)
    write_complete_base(project)
    write_complete_frs(project)
    write_complete_encoder(project)
    result = run_download(project, env)

    assert result.returncode == 0, result.stderr
    assert log_path.read_text(encoding="utf-8") == ""


def test_is_executable_and_supports_checkpoint_roots_with_spaces(tmp_path: Path) -> None:
    project, _log_path, env = make_project(tmp_path)
    checkpoint_root = project / "checkpoint root"
    env["FRS_CHECKPOINT_ROOT"] = str(checkpoint_root)
    result = run_download(project, env)

    assert SCRIPT.stat().st_mode & stat.S_IXUSR
    assert result.returncode == 0, result.stderr
    assert (checkpoint_root / "model/pick_tube_02_3w_jax/model.safetensors").is_file()
