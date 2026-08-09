import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

from deploy_smolvla.src import download_ckpt


ROOT = Path(__file__).resolve().parents[1]


def test_default_encoder_output_matches_vt_server_contract() -> None:
    assert download_ckpt.DEFAULT_REPO_ID == "liuchaoyi/encoder_ckpt_05"
    assert download_ckpt.DEFAULT_OUTPUT_DIR == Path("/DATA/ljl/substage/checkpoints/encoder_ckpt_05")


def test_output_dir_override_is_preserved(tmp_path: Path) -> None:
    args = download_ckpt.parse_args(["--output-dir", str(tmp_path / "custom")])
    assert args.output_dir == tmp_path / "custom"


def test_shell_wrapper_reports_defaults_and_forwards_arguments(tmp_path: Path) -> None:
    project = tmp_path / "project"
    scripts = project / "scripts"
    download_src = project / "deploy_smolvla" / "src"
    fake_bin = tmp_path / "bin"
    scripts.mkdir(parents=True)
    download_src.mkdir(parents=True)
    fake_bin.mkdir()
    wrapper = scripts / "download_ckpt.sh"
    shutil.copy2(ROOT / "scripts" / "download_ckpt.sh", wrapper)
    shutil.copy2(ROOT / "deploy_smolvla" / "src" / "download_ckpt.py", download_src / "download_ckpt.py")
    uv = fake_bin / "uv"
    uv.write_text("#!/usr/bin/env bash\nprintf '%s\\n' \"$@\"\n", encoding="utf-8")
    uv.chmod(0o755)
    (project / ".env.frs").write_text(
        "export FRS_STORAGE_ROOT=/DATA/ljl/substage\n"
        "export FRS_VENV_DIR=/home/ljl/.venvs/frs_tact\n",
        encoding="utf-8",
    )
    custom_output = tmp_path / "custom output"
    env = os.environ.copy()
    env["PATH"] = f"{fake_bin}:{env['PATH']}"

    result = subprocess.run(
        ["bash", str(wrapper), "--output-dir", str(custom_output)],
        cwd=project,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "默认仓库：liuchaoyi/encoder_ckpt_05" in result.stdout
    assert "默认目录：/DATA/ljl/substage/checkpoints/encoder_ckpt_05" in result.stdout
    output_lines = result.stdout.splitlines()
    assert output_lines[-7:] == [
        "run",
        "--no-sync",
        "python",
        str(download_src / "download_ckpt.py"),
        "--minimal",
        "--output-dir",
        str(custom_output),
    ]


def test_main_downloads_and_verifies_custom_output_offline(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    output_dir = tmp_path / "custom" / "encoder"
    snapshot_call: dict[str, object] = {}

    class FakeApi:
        def __init__(self, *, token: str | None) -> None:
            self.token = token

        def model_info(self, repo_id: str, *, revision: str) -> SimpleNamespace:
            assert repo_id == download_ckpt.DEFAULT_REPO_ID
            assert revision == "main"
            return SimpleNamespace(sha="offline-revision")

    def fake_snapshot_download(**kwargs: object) -> None:
        snapshot_call.update(kwargs)
        local_dir = kwargs["local_dir"]
        assert isinstance(local_dir, Path)
        assert local_dir.is_dir()
        metadata = {
            "epoch": 6,
            "params_file": "params.npz",
            "parameter_paths": ["tactile_resnet/kernel"],
            "tactile_backbone": "resnet18",
            "tactile_clip_config": {
                "embedding_dim": 32,
                "tactile_image_size": 224,
                "tactile_history": 1,
            },
        }
        (local_dir / "checkpoint.json").write_text(
            json.dumps(metadata), encoding="utf-8"
        )
        download_ckpt.np.savez(
            local_dir / "params.npz",
            **{"tactile_resnet/kernel": download_ckpt.np.array([1.0])},
        )

    monkeypatch.setattr(download_ckpt, "HfApi", FakeApi)
    monkeypatch.setattr(download_ckpt, "snapshot_download", fake_snapshot_download)
    monkeypatch.setattr(
        sys,
        "argv",
        ["download_ckpt.py", "--output-dir", str(output_dir)],
    )

    download_ckpt.main()

    assert snapshot_call["local_dir"] == output_dir.resolve()
    assert snapshot_call["repo_id"] == download_ckpt.DEFAULT_REPO_ID
    assert download_ckpt.verify_checkpoint(output_dir)["epoch"] == 6
    assert "校验通过" in capsys.readouterr().out
