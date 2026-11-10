from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest
from flax import nnx

from train_pi05_frs.tools import upload_shared_frs as upload
from train_pi05_frs.utils.checkpoint import load_checkpoint, save_checkpoint
from train_pi05_frs.utils.model import DecoderConfig, TactileConditionedFlowDecoder, make_optimizer


@pytest.fixture
def checkpoint(tmp_path):
    model = TactileConditionedFlowDecoder(
        DecoderConfig(
            action_dim=20,
            action_horizon=2,
            tactile_window=2,
            resnet_embedding_dim=4,
            gru_hidden_dim=4,
            model_dim=8,
            depth=1,
            num_heads=2,
            num_tactile_tokens=4,
            tactile_token_identity=True,
        ),
        rngs=nnx.Rngs(0),
    )
    optimizer = make_optimizer(model, learning_rate=1e-3, weight_decay=0, total_steps=4)
    alias = tmp_path / "run/best"
    save_checkpoint(
        alias,
        model,
        epoch=5,
        metrics={"val_mse": 0.1},
        optimizer=optimizer,
        extra_metadata={"shared_frs_interface_version": 1},
    )
    return alias, model, optimizer


def test_upload_pins_best_generation_and_round_trips_complete_checkpoint(tmp_path, checkpoint, monkeypatch):
    import huggingface_hub

    alias, model, optimizer = checkpoint
    original = alias.resolve()
    received = tmp_path / "downloaded"
    calls = []

    class API:
        def create_repo(self, **kwargs):
            calls.append(kwargs)
            # Training publishing a newer best cannot change this upload's snapshot.
            save_checkpoint(
                alias,
                model,
                epoch=10,
                metrics={"val_mse": 0.05},
                optimizer=optimizer,
                extra_metadata={"shared_frs_interface_version": 1},
            )

        def upload_folder(self, **kwargs):
            calls.append(kwargs)
            assert Path(kwargs["folder_path"]) == original
            assert set(kwargs["allow_patterns"]) == {
                "checkpoint.json",
                "params.npz",
                "opt_state.npz",
                "opt_state.treedef.pkl",
            }
            received.mkdir()
            for name in kwargs["allow_patterns"]:
                shutil.copyfile(original / name, received / name)
            return "test-commit"

    monkeypatch.setattr(huggingface_hub, "HfApi", API)
    upload.main(["org/shared-frs", "--checkpoint", str(alias), "--private"])
    assert calls[0] == {"repo_id": "org/shared-frs", "repo_type": "model", "private": True, "exist_ok": True}
    _, metadata = load_checkpoint(received)
    assert metadata["epoch"] == 5 and metadata["has_opt_state"]
    assert json.loads((alias / "checkpoint.json").read_text())["epoch"] == 10


def test_dry_run_validates_without_hub_requests(checkpoint, monkeypatch, capsys):
    import huggingface_hub

    monkeypatch.setattr(huggingface_hub, "HfApi", lambda: pytest.fail("dry-run contacted Hub"))
    upload.main(["org/shared-frs", "--checkpoint", str(checkpoint[0]), "--dry-run"])
    assert "Validation passed" in capsys.readouterr().out


def test_bad_checkpoint_fails_before_repo_creation(checkpoint, monkeypatch):
    import huggingface_hub

    alias = checkpoint[0]
    with (alias / "params.npz").open("ab") as file:
        file.write(b"corrupt")
    monkeypatch.setattr(huggingface_hub, "HfApi", lambda: pytest.fail("invalid checkpoint contacted Hub"))
    with pytest.raises(ValueError, match="size mismatch"):
        upload.main(["org/shared-frs", "--checkpoint", str(alias)])


def test_upload_launcher_help_and_syntax():
    script = Path(__file__).resolve().parents[2] / "scripts/upload_frs_pi05_shared_ckpt.sh"
    subprocess.run(["bash", "-n", str(script)], check=True)
    result = subprocess.run(["bash", str(script), "--help"], capture_output=True, text=True, check=True)
    assert "--dry-run" in result.stdout and "OWNER/REPO" in result.stdout
