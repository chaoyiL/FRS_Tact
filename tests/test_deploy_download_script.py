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
import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "deploy_smolvla/scripts/download.sh"
BASE_DIR = Path("checkpoints/model/pick_tube_02_3w_jax")
FRS_DIR = Path("checkpoints/frs/frs_0809_02")
ENCODER_DIR = Path("checkpoints/encoder/encoder_ckpt_0809")
PROVENANCE_FILE = ".download-provenance.json"
BASE_REPO = "lerobot/smolvla_base"
BASE_REVISION = "c83c3163b8ca9b7e67c509fffd9121e66cb96205"
ADAPTER_REPO = "KaiyueChen/pick_tube_02_3w"
ADAPTER_REVISION = "31d819d8844de98174ede123f894adbf7b4372ef"
FRS_REPO = "KaiyueChen/frs_0809_02"
FRS_REVISION = "7e23f3e8c308dc5ba3a4df7634c68dac28572897"
ENCODER_REPO = "KaiyueChen/encoder_ckpt_0809"
ENCODER_REVISION = "450aa60963cde9540bd6c8047bf2529eff1def37"
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
    (project / "deploy_smolvla/__init__.py").touch()
    (project / "deploy_smolvla/src").mkdir(parents=True)
    shutil.copy2(
        ROOT / "deploy_smolvla/src/download_ckpt.py",
        project / "deploy_smolvla/src/download_ckpt.py",
    )

    bin_dir = project / "bin"
    bin_dir.mkdir()
    log_path = project / "calls.log"
    log_path.touch()
    fake_uv = bin_dir / "uv"
    fake_uv.write_text(
        """#!/usr/bin/env bash
set -euo pipefail

log_path="${FRS_TEST_LOG:?}"
"$FRS_DOWNLOAD_PYTHON" - "$log_path" "$@" <<'PY'
import json
import sys
with open(sys.argv[1], "a", encoding="utf-8") as file:
    file.write(json.dumps(sys.argv[2:]) + "\\n")
PY

output=""
for ((index = 1; index <= $#; index++)); do
    if [[ "${!index}" == "--output" || "${!index}" == "--local-dir" || "${!index}" == "--output-dir" ]]; then
        next=$((index + 1))
        output="${!next}"
    fi
done

asset=""
if [[ "$*" == *"merge_smolvla_peft_to_jax.py"* ]]; then
    asset="base"
elif [[ "$*" == *"hf download KaiyueChen/frs_0809_02"* ]]; then
    asset="frs"
elif [[ "$*" == *"download_ckpt.py"* ]]; then
    asset="encoder"
fi
if [[ "${FRS_TEST_FAIL_ASSET:-}" == "$asset" ]]; then
    printf 'simulated %s failure\\n' "$asset" >&2
    exit 23
fi

if [[ "$asset" == "base" ]]; then
    mkdir -p "$output"
    printf 'model' > "$output/model.safetensors"
    for sidecar in config.json train_config.json policy_preprocessor.json policy_postprocessor.json policy_preprocessor_step_5_normalizer_processor.safetensors policy_postprocessor_step_0_unnormalizer_processor.safetensors; do
        printf '{}' > "$output/$sidecar"
    done
    printf '%s' '{"source_base":"lerobot/smolvla_base","source_adapter":"KaiyueChen/pick_tube_02_3w","adapter_revision":"31d819d8844de98174ede123f894adbf7b4372ef","base_revision":"c83c3163b8ca9b7e67c509fffd9121e66cb96205"}' > "$output/conversion_manifest.json"
elif [[ "$asset" == "frs" ]]; then
    mkdir -p "$output"
    printf '%s' '{"params_file":"params.npz"}' > "$output/checkpoint.json"
    "$FRS_DOWNLOAD_PYTHON" - "$output/params.npz" <<'PY'
import numpy as np
import sys
np.savez(sys.argv[1], value=np.array([1]))
PY
elif [[ "$asset" == "encoder" ]]; then
    mkdir -p "$output"
    printf '%s' '{"params_file":"params.npz","parameter_paths":["tactile_resnet/kernel"],"tactile_clip_config":{"embedding_dim":32,"tactile_image_size":224,"tactile_history":1}}' > "$output/checkpoint.json"
    "$FRS_DOWNLOAD_PYTHON" - "$output/params.npz" <<'PY'
import numpy as np
import sys
np.savez(sys.argv[1], **{"tactile_resnet/kernel": np.array([1.0])})
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


def read_calls(log_path: Path) -> list[list[str]]:
    return [
        json.loads(line)
        for line in log_path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def expected_calls(project: Path) -> list[list[str]]:
    return [
        [
            "run",
            "--no-sync",
            "python",
            str(project / "tools/merge_smolvla_peft_to_jax.py"),
            "--adapter",
            ADAPTER_REPO,
            "--adapter-revision",
            ADAPTER_REVISION,
            "--base",
            BASE_REPO,
            "--base-revision",
            BASE_REVISION,
            "--output",
            str(project / BASE_DIR),
            "--allow-download",
            "--overwrite",
        ],
        [
            "run",
            "--no-sync",
            "hf",
            "download",
            FRS_REPO,
            "--revision",
            FRS_REVISION,
            "--include",
            "checkpoint.json",
            "params-*.npz",
            "--local-dir",
            str(project / FRS_DIR),
        ],
        [
            "run",
            "--no-sync",
            "python",
            str(project / "deploy_smolvla/src/download_ckpt.py"),
            "--minimal",
            "--repo-id",
            ENCODER_REPO,
            "--revision",
            ENCODER_REVISION,
            "--output-dir",
            str(project / ENCODER_DIR),
        ],
    ]


def write_provenance(directory: Path, repo_id: str, revision: str) -> None:
    (directory / PROVENANCE_FILE).write_text(
        json.dumps({"format_version": 1, "repo_id": repo_id, "revision": revision}),
        encoding="utf-8",
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
                "source_base": BASE_REPO,
                "source_adapter": ADAPTER_REPO,
                "adapter_revision": ADAPTER_REVISION,
                "base_revision": BASE_REVISION,
            }
        ),
        encoding="utf-8",
    )


def write_complete_frs(project: Path, *, provenance: bool = True) -> None:
    directory = project / FRS_DIR
    directory.mkdir(parents=True)
    (directory / "checkpoint.json").write_text(
        json.dumps({"params_file": "params.npz"}), encoding="utf-8"
    )
    np.savez(directory / "params.npz", value=np.array([1]))
    if provenance:
        write_provenance(directory, FRS_REPO, FRS_REVISION)


def write_complete_encoder(project: Path, *, provenance: bool = True) -> None:
    directory = project / ENCODER_DIR
    directory.mkdir(parents=True)
    metadata = {
        "params_file": "params.npz",
        "parameter_paths": ["tactile_resnet/kernel"],
        "tactile_clip_config": {
            "embedding_dim": 32,
            "tactile_image_size": 224,
            "tactile_history": 1,
        },
    }
    (directory / "checkpoint.json").write_text(json.dumps(metadata), encoding="utf-8")
    np.savez(directory / "params.npz", **{"tactile_resnet/kernel": np.array([1.0])})
    if provenance:
        write_provenance(directory, ENCODER_REPO, ENCODER_REVISION)


def write_incomplete_frs(project: Path) -> None:
    directory = project / FRS_DIR
    directory.mkdir(parents=True)
    (directory / "checkpoint.json").write_text(
        json.dumps({"params_file": "params-missing.npz"}), encoding="utf-8"
    )
    write_provenance(directory, FRS_REPO, FRS_REVISION)


def test_downloads_all_missing_assets_with_exact_pinned_commands(tmp_path: Path) -> None:
    project, log_path, env = make_project(tmp_path)
    result = run_download(project, env)

    assert result.returncode == 0, result.stderr
    assert read_calls(log_path) == expected_calls(project)
    assert (project / BASE_DIR / "model.safetensors").is_file()
    assert (project / FRS_DIR / "checkpoint.json").is_file()
    assert (project / ENCODER_DIR / "checkpoint.json").is_file()
    assert json.loads((project / FRS_DIR / PROVENANCE_FILE).read_text()) == {
        "format_version": 1,
        "repo_id": FRS_REPO,
        "revision": FRS_REVISION,
    }
    assert json.loads((project / ENCODER_DIR / PROVENANCE_FILE).read_text()) == {
        "format_version": 1,
        "repo_id": ENCODER_REPO,
        "revision": ENCODER_REVISION,
    }


def test_skips_complete_assets_and_downloads_missing_frs(tmp_path: Path) -> None:
    project, log_path, env = make_project(tmp_path)
    write_complete_base(project)
    write_complete_encoder(project)
    result = run_download(project, env)

    assert result.returncode == 0, result.stderr
    assert read_calls(log_path) == [expected_calls(project)[1]]


def test_repairs_frs_metadata_with_missing_params(tmp_path: Path) -> None:
    project, log_path, env = make_project(tmp_path)
    write_complete_base(project)
    write_incomplete_frs(project)
    write_complete_encoder(project)
    result = run_download(project, env)

    assert result.returncode == 0, result.stderr
    assert read_calls(log_path) == [expected_calls(project)[1]]


def test_repairs_malformed_frs_archive(tmp_path: Path) -> None:
    project, log_path, env = make_project(tmp_path)
    write_complete_base(project)
    write_complete_frs(project)
    (project / FRS_DIR / "params.npz").write_bytes(b"not an npz")
    write_complete_encoder(project)

    result = run_download(project, env)

    assert result.returncode == 0, result.stderr
    assert read_calls(log_path) == [expected_calls(project)[1]]


def test_incompatible_encoder_is_refreshed_with_project_verifier(tmp_path: Path) -> None:
    project, log_path, env = make_project(tmp_path)
    write_complete_base(project)
    write_complete_frs(project)
    directory = project / ENCODER_DIR
    directory.mkdir(parents=True)
    (directory / "checkpoint.json").write_text(
        json.dumps({"params_file": "params.npz"}), encoding="utf-8"
    )
    np.savez(directory / "params.npz", value=np.array([1]))
    write_provenance(directory, ENCODER_REPO, ENCODER_REVISION)

    result = run_download(project, env)

    assert result.returncode == 0, result.stderr
    assert read_calls(log_path) == [expected_calls(project)[2]]


@pytest.mark.parametrize("asset", ["frs", "encoder"])
def test_source_mismatch_refreshes_asset(tmp_path: Path, asset: str) -> None:
    project, log_path, env = make_project(tmp_path)
    write_complete_base(project)
    write_complete_frs(project)
    write_complete_encoder(project)
    directory = project / (FRS_DIR if asset == "frs" else ENCODER_DIR)
    expected_repo = FRS_REPO if asset == "frs" else ENCODER_REPO
    write_provenance(directory, expected_repo, "wrong-revision")

    result = run_download(project, env)

    assert result.returncode == 0, result.stderr
    expected_index = 1 if asset == "frs" else 2
    assert read_calls(log_path) == [expected_calls(project)[expected_index]]


@pytest.mark.parametrize("asset", ["frs", "encoder"])
def test_legacy_asset_without_provenance_is_refreshed_once(
    tmp_path: Path, asset: str
) -> None:
    project, log_path, env = make_project(tmp_path)
    write_complete_base(project)
    write_complete_frs(project, provenance=asset != "frs")
    write_complete_encoder(project, provenance=asset != "encoder")

    first = run_download(project, env)

    assert first.returncode == 0, first.stderr
    expected_index = 1 if asset == "frs" else 2
    assert read_calls(log_path) == [expected_calls(project)[expected_index]]
    log_path.write_text("", encoding="utf-8")

    second = run_download(project, env)

    assert second.returncode == 0, second.stderr
    assert read_calls(log_path) == []


def test_skips_all_complete_assets_with_explicit_messages_and_summary(tmp_path: Path) -> None:
    project, log_path, env = make_project(tmp_path)
    write_complete_base(project)
    write_complete_frs(project)
    write_complete_encoder(project)
    result = run_download(project, env)

    assert result.returncode == 0, result.stderr
    assert read_calls(log_path) == []
    assert f"skip: base checkpoint: {project / BASE_DIR}" in result.stdout
    assert f"skip: FRS checkpoint: {project / FRS_DIR}" in result.stdout
    assert f"skip: tactile encoder checkpoint: {project / ENCODER_DIR}" in result.stdout
    assert f"checkpoint: {project / BASE_DIR}" in result.stdout
    assert f"frs.checkpoint: {project / FRS_DIR}" in result.stdout
    assert (
        f"frs.tactile_encoder_checkpoint: {project / ENCODER_DIR}" in result.stdout
    )


@pytest.mark.parametrize(
    ("asset", "label", "directory"),
    [
        ("base", "base checkpoint merge", BASE_DIR),
        ("frs", "FRS checkpoint download", FRS_DIR),
        ("encoder", "tactile encoder download", ENCODER_DIR),
    ],
)
def test_delegated_failure_names_asset_and_destination(
    tmp_path: Path, asset: str, label: str, directory: Path
) -> None:
    project, log_path, env = make_project(tmp_path)
    if asset != "base":
        write_complete_base(project)
    if asset == "encoder":
        write_complete_frs(project)
    env["FRS_TEST_FAIL_ASSET"] = asset

    result = run_download(project, env)

    assert result.returncode != 0
    assert f"{label} failed: {project / directory}" in result.stderr
    assert len(read_calls(log_path)) == 1


def test_refuses_unguarded_base_overwrite(tmp_path: Path) -> None:
    project, log_path, env = make_project(tmp_path)
    script = project / "deploy_smolvla/scripts/download.sh"
    content = script.read_text(encoding="utf-8")
    content = content.replace(
        'BASE_DIR="${CHECKPOINT_ROOT}/model/pick_tube_02_3w_jax"',
        'BASE_DIR="${CHECKPOINT_ROOT}/../outside"',
    )
    script.write_text(content, encoding="utf-8")

    result = run_download(project, env)

    assert result.returncode != 0
    assert "refusing to overwrite base directory outside" in result.stderr
    assert read_calls(log_path) == []


def test_is_executable_and_supports_checkpoint_roots_with_spaces(tmp_path: Path) -> None:
    project, _log_path, env = make_project(tmp_path)
    checkpoint_root = project / "checkpoint root"
    env["FRS_CHECKPOINT_ROOT"] = str(checkpoint_root)
    result = run_download(project, env)

    assert SCRIPT.stat().st_mode & stat.S_IXUSR
    assert result.returncode == 0, result.stderr
    assert (checkpoint_root / "model/pick_tube_02_3w_jax/model.safetensors").is_file()
