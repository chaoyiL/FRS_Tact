import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

from deploy_smolvla.src import download_ckpt

ROOT = Path(__file__).resolve().parents[1]


def test_default_encoder_output_is_project_local() -> None:
    assert download_ckpt.DEFAULT_OUTPUT_DIR == ROOT / "checkpoints" / "encoder" / "encoder_ckpt_06"


def test_output_dir_override_is_preserved(tmp_path: Path) -> None:
    args = download_ckpt.parse_args(["--output-dir", str(tmp_path / "custom")])
    assert args.output_dir == tmp_path / "custom"


def test_encoder_shell_wrapper_reports_defaults_and_forwards_arguments(tmp_path: Path) -> None:
    project = tmp_path / "project"
    scripts = project / "scripts"
    download_src = project / "deploy_smolvla" / "src"
    fake_bin = tmp_path / "bin"
    scripts.mkdir(parents=True)
    download_src.mkdir(parents=True)
    fake_bin.mkdir()
    wrapper = scripts / "download_encoder.sh"
    shutil.copy2(ROOT / "scripts" / "download_encoder.sh", wrapper)
    shutil.copy2(ROOT / "deploy_smolvla" / "src" / "download_ckpt.py", download_src / "download_ckpt.py")
    uv = fake_bin / "uv"
    uv.write_text("#!/usr/bin/env bash\nprintf '%s\\n' \"$@\"\n", encoding="utf-8")
    uv.chmod(0o755)
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
    assert "默认仓库：liuchaoyi/encoder_ckpt_06" in result.stdout
    assert (
        f"默认目录：{project / 'checkpoints' / 'encoder' / 'encoder_ckpt_06'}"
        in result.stdout
    )
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


def test_model_shell_wrapper_downloads_repo_to_project_model_dir(tmp_path: Path) -> None:
    project = tmp_path / "project"
    scripts = project / "scripts"
    fake_bin = tmp_path / "bin"
    scripts.mkdir(parents=True)
    fake_bin.mkdir()
    wrapper = scripts / "download_ckpt.sh"
    shutil.copy2(ROOT / "scripts" / "download_ckpt.sh", wrapper)
    uv = fake_bin / "uv"
    uv.write_text("#!/usr/bin/env bash\nprintf '%s\\n' \"$@\"\n", encoding="utf-8")
    uv.chmod(0o755)
    env = os.environ.copy()
    env["PATH"] = f"{fake_bin}:{env['PATH']}"

    result = subprocess.run(
        ["bash", str(wrapper), "--Aether258/pi05_bi_two_tubes_all_step8000"],
        cwd=project,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    output_dir = project / "checkpoints" / "model" / "pi05_bi_two_tubes_all_step8000"
    assert result.returncode == 0, result.stderr
    assert "模型仓库：Aether258/pi05_bi_two_tubes_all_step8000" in result.stdout
    assert f"下载目录：{output_dir}" in result.stdout
    assert result.stdout.splitlines()[-9:] == [
        "run",
        "--no-sync",
        "hf",
        "download",
        "Aether258/pi05_bi_two_tubes_all_step8000",
        "--repo-type",
        "model",
        "--local-dir",
        str(output_dir),
    ]


def test_model_shell_wrapper_accepts_standard_positional_repo(tmp_path: Path) -> None:
    project = tmp_path / "project"
    scripts = project / "scripts"
    fake_bin = tmp_path / "bin"
    scripts.mkdir(parents=True)
    fake_bin.mkdir()
    wrapper = scripts / "download_ckpt.sh"
    shutil.copy2(ROOT / "scripts" / "download_ckpt.sh", wrapper)
    uv = fake_bin / "uv"
    uv.write_text("#!/usr/bin/env bash\nprintf '%s\\n' \"$@\"\n", encoding="utf-8")
    uv.chmod(0o755)
    env = os.environ.copy()
    env["PATH"] = f"{fake_bin}:{env['PATH']}"

    result = subprocess.run(
        ["bash", str(wrapper), "Aether258/pi05_bi_two_tubes_all_step8000"],
        cwd=project,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "Aether258/pi05_bi_two_tubes_all_step8000" in result.stdout


def test_model_shell_wrapper_rejects_unsafe_repo_id(tmp_path: Path) -> None:
    project = tmp_path / "project"
    scripts = project / "scripts"
    scripts.mkdir(parents=True)
    wrapper = scripts / "download_ckpt.sh"
    shutil.copy2(ROOT / "scripts" / "download_ckpt.sh", wrapper)

    result = subprocess.run(
        ["bash", str(wrapper), "--../../outside"],
        cwd=project,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode != 0
    assert "仓库 ID" in result.stderr


def test_model_shell_wrapper_reuses_owned_dir_and_rejects_same_name_from_other_owner(
    tmp_path: Path,
) -> None:
    project = tmp_path / "project"
    scripts = project / "scripts"
    fake_bin = tmp_path / "bin"
    scripts.mkdir(parents=True)
    fake_bin.mkdir()
    wrapper = scripts / "download_ckpt.sh"
    shutil.copy2(ROOT / "scripts" / "download_ckpt.sh", wrapper)
    call_log = tmp_path / "uv-calls.log"
    uv = fake_bin / "uv"
    uv.write_text(
        "#!/usr/bin/env bash\nprintf 'called\\n' >>\"$UV_CALL_LOG\"\n",
        encoding="utf-8",
    )
    uv.chmod(0o755)
    env = os.environ.copy()
    env["PATH"] = f"{fake_bin}:{env['PATH']}"
    env["UV_CALL_LOG"] = str(call_log)

    first = subprocess.run(
        ["bash", str(wrapper), "alice/shared-model"],
        cwd=project,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    resumed = subprocess.run(
        ["bash", str(wrapper), "alice/shared-model"],
        cwd=project,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    conflict = subprocess.run(
        ["bash", str(wrapper), "bob/shared-model"],
        cwd=project,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    output_dir = project / "checkpoints" / "model" / "shared-model"
    repo_marker = (
        project
        / "checkpoints"
        / "model"
        / ".frs_hf_repos"
        / "shared-model.repo-id"
    )
    assert first.returncode == 0, first.stderr
    assert resumed.returncode == 0, resumed.stderr
    assert output_dir.is_dir()
    assert repo_marker.read_text(encoding="utf-8").strip() == "alice/shared-model"
    assert call_log.read_text(encoding="utf-8").splitlines() == ["called", "called"]
    assert conflict.returncode != 0
    assert "已经属于 alice/shared-model" in conflict.stderr


def test_model_shell_wrapper_rejects_unowned_nonempty_or_symlink_target(tmp_path: Path) -> None:
    project = tmp_path / "project"
    scripts = project / "scripts"
    fake_bin = tmp_path / "bin"
    scripts.mkdir(parents=True)
    fake_bin.mkdir()
    wrapper = scripts / "download_ckpt.sh"
    shutil.copy2(ROOT / "scripts" / "download_ckpt.sh", wrapper)
    uv = fake_bin / "uv"
    uv.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
    uv.chmod(0o755)
    env = os.environ.copy()
    env["PATH"] = f"{fake_bin}:{env['PATH']}"
    model_root = project / "checkpoints" / "model"
    unowned = model_root / "unowned"
    unowned.mkdir(parents=True)
    (unowned / "weights.bin").write_bytes(b"existing")
    outside = tmp_path / "outside"
    outside.mkdir()
    (model_root / "linked").symlink_to(outside, target_is_directory=True)

    unowned_result = subprocess.run(
        ["bash", str(wrapper), "owner/unowned"],
        cwd=project,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    symlink_result = subprocess.run(
        ["bash", str(wrapper), "owner/linked"],
        cwd=project,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert unowned_result.returncode != 0
    assert "缺少仓库归属标记" in unowned_result.stderr
    assert symlink_result.returncode != 0
    assert "符号链接" in symlink_result.stderr
    assert list(outside.iterdir()) == []


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
