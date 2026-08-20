from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "upload_frs_ckpt.sh"


def _make_checkpoint(
    path: Path,
    *,
    params_file: str = "params-test.npz",
    create_params: bool = True,
) -> Path:
    path.mkdir(parents=True)
    if create_params:
        params_path = path / params_file
        params_path.parent.mkdir(parents=True, exist_ok=True)
        params_path.write_bytes(b"params")
    (path / "checkpoint.json").write_text(
        json.dumps({"params_file": params_file}),
        encoding="utf-8",
    )
    return path


def _run_script(
    tmp_path: Path,
    repo_id: str,
    checkpoint: Path,
    *extra: str | Path,
) -> tuple[subprocess.CompletedProcess[str], list[list[str]]]:
    fake_bin = tmp_path / "fake-bin"
    fake_bin.mkdir(exist_ok=True)
    call_log = tmp_path / "uv-calls.log"
    uv = fake_bin / "uv"
    uv.write_text(
        "#!/usr/bin/env bash\n"
        "if [[ \"$1\" == run && \"$2\" == --no-sync && \"$3\" == python ]]; then\n"
        "    shift 3\n"
        "    exec python \"$@\"\n"
        "fi\n"
        "printf 'CALL' >>\"$UV_CALL_LOG\"\n"
        "printf '\\t%s' \"$@\" >>\"$UV_CALL_LOG\"\n"
        "printf '\\n' >>\"$UV_CALL_LOG\"\n",
        encoding="utf-8",
    )
    uv.chmod(0o755)
    env = os.environ.copy()
    env["PATH"] = f"{fake_bin}:{env['PATH']}"
    env["UV_CALL_LOG"] = str(call_log)
    result = subprocess.run(
        ["bash", str(SCRIPT), repo_id, str(checkpoint), *(str(value) for value in extra)],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    calls = []
    if call_log.exists():
        calls = [line.split("\t")[1:] for line in call_log.read_text().splitlines()]
    return result, calls


def test_uploads_selected_checkpoint_and_explicit_figures(tmp_path: Path) -> None:
    checkpoint = _make_checkpoint(tmp_path / "run" / "best")
    figures = tmp_path / "plots"
    figures.mkdir()
    (figures / "training_overview.png").write_bytes(b"png")

    result, calls = _run_script(
        tmp_path,
        "KaiyueChen/frs-best",
        checkpoint,
        "--figures-dir",
        figures,
    )

    assert result.returncode == 0, result.stderr
    assert calls == [
        ["run", "--no-sync", "hf", "auth", "whoami"],
        [
            "run",
            "--no-sync",
            "hf",
            "repo",
            "create",
            "KaiyueChen/frs-best",
            "--repo-type",
            "model",
            "--exist-ok",
        ],
        [
            "run",
            "--no-sync",
            "hf",
            "upload",
            "KaiyueChen/frs-best",
            str(checkpoint.resolve()),
            ".",
            "--repo-type",
            "model",
            "--commit-message",
            "Upload FRS checkpoint",
        ],
        [
            "run",
            "--no-sync",
            "hf",
            "upload",
            "KaiyueChen/frs-best",
            str(figures.resolve()),
            "figures",
            "--repo-type",
            "model",
            "--include",
            "*.png",
            "--commit-message",
            "Upload FRS training figures",
        ],
    ]
    assert all("--private" not in call for call in calls)


def test_defaults_figures_to_checkpoint_parent(tmp_path: Path) -> None:
    checkpoint = _make_checkpoint(tmp_path / "run" / "best")
    (checkpoint.parent / "gate_diagnostics.png").write_bytes(b"png")

    result, calls = _run_script(tmp_path, "KaiyueChen/frs-best", checkpoint)

    assert result.returncode == 0, result.stderr
    assert calls[-1][5] == str(checkpoint.parent.resolve())


def test_help_does_not_require_uv() -> None:
    result = subprocess.run(
        ["bash", str(SCRIPT), "--help"],
        cwd=ROOT,
        env={"PATH": "/usr/bin:/bin"},
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0
    assert "OWNER/REPO" in result.stdout


def test_rejects_unsafe_repo_id_before_upload(tmp_path: Path) -> None:
    checkpoint = _make_checkpoint(tmp_path / "run" / "best")
    (checkpoint.parent / "plot.png").write_bytes(b"png")

    result, calls = _run_script(tmp_path, "../outside", checkpoint)

    assert result.returncode != 0
    assert "仓库 ID" in result.stderr
    assert calls == []


def test_rejects_missing_parameter_file_before_upload(tmp_path: Path) -> None:
    checkpoint = _make_checkpoint(
        tmp_path / "run" / "best",
        create_params=False,
    )
    (checkpoint.parent / "plot.png").write_bytes(b"png")

    result, calls = _run_script(tmp_path, "KaiyueChen/frs-best", checkpoint)

    assert result.returncode != 0
    assert "params_file" in result.stderr
    assert calls == []


@pytest.mark.parametrize("params_file", ["../outside.npz", "/tmp/outside.npz"])
def test_rejects_unsafe_params_file(tmp_path: Path, params_file: str) -> None:
    checkpoint = _make_checkpoint(
        tmp_path / "run" / "best",
        params_file=params_file,
        create_params=False,
    )
    (checkpoint.parent / "plot.png").write_bytes(b"png")

    result, calls = _run_script(tmp_path, "KaiyueChen/frs-best", checkpoint)

    assert result.returncode != 0
    assert "params_file" in result.stderr
    assert calls == []


def test_rejects_checkpoint_directory_symlink(tmp_path: Path) -> None:
    real_checkpoint = _make_checkpoint(tmp_path / "real" / "best")
    checkpoint = tmp_path / "linked-best"
    checkpoint.symlink_to(real_checkpoint, target_is_directory=True)
    (tmp_path / "plot.png").write_bytes(b"png")

    result, calls = _run_script(tmp_path, "KaiyueChen/frs-best", checkpoint)

    assert result.returncode != 0
    assert "符号链接" in result.stderr
    assert calls == []


def test_rejects_figures_directory_symlink(tmp_path: Path) -> None:
    checkpoint = _make_checkpoint(tmp_path / "run" / "best")
    real_figures = tmp_path / "real-figures"
    real_figures.mkdir()
    (real_figures / "plot.png").write_bytes(b"png")
    figures = tmp_path / "linked-figures"
    figures.symlink_to(real_figures, target_is_directory=True)

    result, calls = _run_script(
        tmp_path,
        "KaiyueChen/frs-best",
        checkpoint,
        "--figures-dir",
        figures,
    )

    assert result.returncode != 0
    assert "符号链接" in result.stderr
    assert calls == []


def test_rejects_missing_top_level_png(tmp_path: Path) -> None:
    checkpoint = _make_checkpoint(tmp_path / "run" / "best")

    result, calls = _run_script(tmp_path, "KaiyueChen/frs-best", checkpoint)

    assert result.returncode != 0
    assert "PNG" in result.stderr
    assert calls == []
