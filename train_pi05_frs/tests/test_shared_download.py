from __future__ import annotations

import json
import os
import shlex
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from train_pi05_frs.tools import download_shared_checkpoints as download
from train_pi05_frs.tools.train_shared_frs import DEFAULT_CONFIG, load_shared_config


def remote_files(source):
    width = source["native_action_dim"]
    stats = {
        key: {field: [0.0] * size for field in ("mean", "std", "q01", "q99")}
        for key, size in (("state", 7 if width == 10 else 20), ("actions", width))
    }
    return {
        "_CHECKPOINT_METADATA": b"{}",
        "params/_METADATA": b"{}",
        "params/_sharding": b"sharding",
        "params/manifest.ocdbt": b"manifest",
        "params/d/metadata": b"index",
        "params/ocdbt.process_0/d/shard": b"parameter shard",
        f"assets/{source['norm_stats_asset_id']}/norm_stats.json": json.dumps({"norm_stats": stats}).encode(),
        "train_state/d/optimizer": b"unneeded optimizer",
    }


@pytest.fixture
def fake_api():
    class API:
        def __init__(self, source):
            self.source = source

        def model_info(self, repo, *, revision, files_metadata):
            assert repo == self.source["checkpoint_repo"]
            assert revision == self.source["checkpoint_revision"]
            assert files_metadata
            return SimpleNamespace(
                sha=revision,
                siblings=[
                    SimpleNamespace(rfilename=k, size=len(v)) for k, v in remote_files(self.source).items()
                ],
            )

    return API


def test_all_four_download_pinned_inference_files_and_resume(tmp_path, monkeypatch, fake_api):
    config = load_shared_config(DEFAULT_CONFIG)
    assert [s["checkpoint_repo"] for s in config["sources"]] == [
        "KaiyueChen/pi05_task3_0830_1w",
        "KaiyueChen/pi05_task4_0830_6k",
        "KaiyueChen/pi05_task2_0902_12k",
        "KaiyueChen/pi05_task1_0902_8k",
    ]
    calls = []
    for source in config["sources"]:
        source = {**source, "checkpoint": str(tmp_path / source["name"])}
        files = remote_files(source)
        plan = download.download_plan(source, fake_api(source))

        def snapshot(source=source, files=files, **kwargs):
            calls.append(kwargs)
            assert kwargs["revision"] == source["checkpoint_revision"]
            assert all(not name.startswith("train_state/") for name in kwargs["allow_patterns"])
            assert "params/ocdbt.process_0/d/shard" in kwargs["allow_patterns"]
            for name in kwargs["allow_patterns"]:
                path = Path(kwargs["local_dir"]) / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(files[name])

        monkeypatch.setattr(download, "snapshot_download", snapshot)
        download.download_one(plan, max_workers=2)
        download.download_one(plan, max_workers=2)
        download.validate_local(plan)
        assert not (plan["target"] / "train_state").exists()
        # A truncated shard must be detected after an interrupted transfer.
        (plan["target"] / "params/ocdbt.process_0/d/shard").write_bytes(b"bad")
        with pytest.raises(ValueError, match="incomplete"):
            download.validate_local(plan)
        download.download_one(plan, max_workers=2)
    assert len(calls) == 12


def test_download_refuses_unowned_or_different_revision(tmp_path, fake_api):
    source = {**load_shared_config(DEFAULT_CONFIG)["sources"][0], "checkpoint": str(tmp_path / "existing")}
    plan = download.download_plan(source, fake_api(source))
    plan["target"].mkdir()
    (plan["target"] / "unrelated.bin").write_bytes(b"keep")
    with pytest.raises(ValueError, match="no .frs_pi05_source"):
        download.download_one(plan, max_workers=1)
    (plan["target"] / download.MARKER).write_text('{"revision": "other"}')
    with pytest.raises(ValueError, match="different repository/revision"):
        download.download_one(plan, max_workers=1)
    assert (plan["target"] / "unrelated.bin").read_bytes() == b"keep"


def test_download_dry_run_never_writes_targets(tmp_path, monkeypatch, capsys, fake_api):
    config = load_shared_config(DEFAULT_CONFIG)
    for source in config["sources"]:
        source["checkpoint"] = str(tmp_path / source["name"])
    by_repo = {s["checkpoint_repo"]: s for s in config["sources"]}

    class API:
        def model_info(self, repo, **kwargs):
            return fake_api(by_repo[repo]).model_info(repo, **kwargs)

    monkeypatch.setattr(download, "HfApi", API)
    monkeypatch.setattr(download, "load_shared_config", lambda _: config)
    monkeypatch.setattr(download, "snapshot_download", lambda **kw: pytest.fail("dry-run downloaded weights"))
    download.main(["--dry-run"])
    assert "Total:" in capsys.readouterr().out
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("script", ["download_frs_shared_ckpts.sh", "download_frs_shared_data.sh"])
@pytest.mark.parametrize(
    "enabled,available,expected", [("1", False, False), ("1", True, True), ("0", False, False)]
)
def test_download_launchers_configure_transfer_before_hub_import(
    tmp_path, script, enabled, available, expected
):
    root = Path(__file__).resolve().parents[2]
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    launcher = scripts / script
    launcher.write_text((root / "scripts" / script).read_text())
    # Simulate an image/env_path enabling the optional backend for another venv.
    (tmp_path / "env_path").write_text(f"export HF_HUB_ENABLE_HF_TRANSFER={enabled}\n")
    python = tmp_path / "data-python"
    python.write_text(
        "#!/usr/bin/env bash\n"
        'if [[ "$1" == "-c" ]]; then\n'
        f"  exit {0 if available else 1}\n"
        "fi\n"
        f"exec {shlex.quote(sys.executable)} -c "
        "'from huggingface_hub import constants; print(constants.HF_HUB_ENABLE_HF_TRANSFER)'\n"
    )
    python.chmod(0o755)
    result = subprocess.run(
        ["bash", str(launcher)],
        env={**os.environ, "DATA_TOOL_PYTHON": str(python), "HF_HUB_ENABLE_HF_TRANSFER": "1"},
        capture_output=True,
        text=True,
        check=True,
        timeout=10,
    )
    assert result.stdout.strip() == str(expected)
    assert ("已切换为普通" in result.stderr) == (enabled == "1" and not available)


@pytest.mark.parametrize("requested,expected", [(None, True), ("0", False)])
def test_data_launcher_disables_xet_by_default_and_honors_explicit_override(tmp_path, requested, expected):
    root = Path(__file__).resolve().parents[2]
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    launcher = scripts / "download_frs_shared_data.sh"
    launcher.write_text((root / "scripts/download_frs_shared_data.sh").read_text())
    python = tmp_path / "data-python"
    python.write_text(
        "#!/usr/bin/env bash\n"
        f"exec {shlex.quote(sys.executable)} -c "
        "'from huggingface_hub import constants; print(constants.HF_HUB_DISABLE_XET)'\n"
    )
    python.chmod(0o755)
    environment = {**os.environ, "DATA_TOOL_PYTHON": str(python), "HF_HUB_ENABLE_HF_TRANSFER": "0"}
    environment.pop("HF_HUB_DISABLE_XET", None)
    if requested is not None:
        environment["HF_HUB_DISABLE_XET"] = requested
    result = subprocess.run(
        ["bash", str(launcher)], env=environment, capture_output=True, text=True, check=True, timeout=10
    )
    assert result.stdout.strip() == str(expected)
