from __future__ import annotations

import importlib
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from train_baseline_pi05.checkpoint import save_best_checkpoint, save_last_checkpoint
from train_baseline_pi05.model import DirectDecoderConfig as TrainDecoderConfig
from train_baseline_pi05.model import DirectTactileActionDecoder as TrainDecoder


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "deploy_baseline_pi05.yaml"


def _expected_source(tmp_path: Path) -> dict[str, object]:
    checkpoint = (tmp_path / "pi05").resolve()
    norm = (checkpoint / "assets").resolve()
    encoder = (tmp_path / "encoder_ckpt_0824").resolve()
    return {
        "pi": {
            "checkpoint": str(checkpoint),
            "norm_stats_dir": str(norm),
            "norm_stats_asset_id": "two_tubes_0102",
            "variant": {"paligemma": "gemma_2b_lora", "action_expert": "gemma_300m_lora"},
            "model_action_width": 20,
            "sample_steps": 10,
        },
        "encoder": {
            "checkpoint": str(encoder),
            "key_order": [
                "observation.images.tactile_left_0",
                "observation.images.tactile_right_0",
                "observation.images.tactile_left_1",
                "observation.images.tactile_right_1",
            ],
        },
    }


def _training_checkpoint(tmp_path: Path) -> tuple[Path, dict[str, object], TrainDecoder]:
    expected = _expected_source(tmp_path)
    model = TrainDecoder(TrainDecoderConfig()).eval()
    path = save_best_checkpoint(
        tmp_path,
        model,
        model.config,
        epoch=1,
        global_step=3,
        metrics={"validation_loss": 0.1},
        source_contract=expected,
    )
    return path, expected, model


def test_default_config_locks_direct_pi05_contract():
    from deploy_baseline_pi05.deployment import TACTILE_KEYS, load_deployment_config

    config = load_deployment_config(CONFIG)

    assert config.model.action_horizon == 50
    assert config.model.action_dim == 20
    assert config.model.state_dim == 20
    assert config.direct_decoder.tactile_keys == TACTILE_KEYS
    assert config.direct_decoder.num_layers == 2
    assert config.direct_decoder.d_model == 128
    assert config.direct_decoder.nhead == 4
    assert config.direct_decoder.dim_feedforward == 256
    assert config.direct_decoder.dropout == 0.1
    assert config.observation.data_type == "vitac"
    assert len(config.model.camera_map) == 2


def test_config_import_does_not_import_torch_or_jax():
    code = (
        "import sys; import deploy_baseline_pi05.deployment; "
        "assert 'torch' not in sys.modules; assert 'jax' not in sys.modules"
    )
    completed = subprocess.run(
        [sys.executable, "-c", code],
        check=False,
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": str(Path.cwd())},
    )

    assert completed.returncode == 0, completed.stderr


def test_training_checkpoint_loads_strictly_and_matches_training_forward(tmp_path: Path):
    from deploy_baseline_pi05.checkpoint import load_decoder

    path, expected, training = _training_checkpoint(tmp_path)
    deployment = load_decoder(path, device="cpu", expected_source=expected)
    for parameter in training.parameters():
        parameter.requires_grad_(False)
    coarse = torch.randn(1, 50, 20)
    tactile = torch.randn(1, 4, 512)

    assert deployment.training is False
    assert all(not parameter.requires_grad for parameter in deployment.parameters())
    torch.testing.assert_close(deployment(coarse, tactile), training(coarse, tactile), rtol=0, atol=0)


@pytest.mark.parametrize(
    ("target", "value", "message"),
    [
        ("run_kind", "informal", "run contract"),
        ("mode", "other", "run contract"),
        ("decoder_config.num_layers", 3, "num_layers"),
        ("decoder_config.action_horizon", 20, "action_horizon"),
        ("decoder_config.tactile_keys", ["wrong"] * 4, "tactile_keys"),
        ("source_contract.pi.model_action_width", 32, "source_contract"),
    ],
)
def test_loader_rejects_wrong_checkpoint_contract(
    tmp_path: Path, target: str, value: object, message: str
):
    from deploy_baseline_pi05.checkpoint import load_decoder

    path, expected, _ = _training_checkpoint(tmp_path)
    raw = torch.load(path, weights_only=True)
    target_dict: dict[str, object] = raw
    pieces = target.split(".")
    for piece in pieces[:-1]:
        target_dict = target_dict[piece]  # type: ignore[assignment,index]
    target_dict[pieces[-1]] = value
    torch.save(raw, path)

    with pytest.raises(ValueError, match=message):
        load_decoder(path, device="cpu", expected_source=expected)


@pytest.mark.parametrize("field", ["checkpoint", "norm_stats_dir", "variant", "model_action_width"])
def test_loader_rejects_expected_source_mismatch(tmp_path: Path, field: str):
    from deploy_baseline_pi05.checkpoint import load_decoder

    path, expected, _ = _training_checkpoint(tmp_path)
    changed = _expected_source(tmp_path)
    pi = changed["pi"]
    assert isinstance(pi, dict)
    if field == "variant":
        pi[field] = {"paligemma": "other", "action_expert": "gemma_300m_lora"}
    elif field == "model_action_width":
        pi[field] = 32
    else:
        pi[field] = str(tmp_path / "different")

    with pytest.raises(ValueError, match="source_contract"):
        load_decoder(path, device="cpu", expected_source=changed)


def test_loader_rejects_resume_checkpoint_even_when_the_decoder_state_is_valid(tmp_path: Path):
    from deploy_baseline_pi05.checkpoint import load_decoder

    _, expected, model = _training_checkpoint(tmp_path)
    path = save_last_checkpoint(
        tmp_path,
        model,
        model.config,
        epoch=1,
        global_step=3,
        metrics={"validation_loss": 0.1},
        source_contract=expected,
        best_state={"validation_loss": 0.1},
    )

    with pytest.raises(ValueError, match="invalid schema"):
        load_decoder(path, device="cpu", expected_source=expected)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("epoch", True),
        ("global_step", -1),
        ("metrics", {"validation_loss": float("nan")}),
        ("metrics", {1: 0.1}),
        ("metrics", {"validation_loss": False}),
    ],
)
def test_loader_rejects_invalid_best_checkpoint_metadata(
    tmp_path: Path, field: str, value: object
):
    from deploy_baseline_pi05.checkpoint import load_decoder

    path, expected, _ = _training_checkpoint(tmp_path)
    raw = torch.load(path, weights_only=True)
    raw[field] = value
    torch.save(raw, path)

    with pytest.raises(ValueError, match=field):
        load_decoder(path, device="cpu", expected_source=expected)


@pytest.mark.parametrize(
    ("group", "field", "actual_value", "expected_value"),
    [
        ("pi", "norm_stats_asset_id", "./asset", str((Path.cwd() / "asset").resolve())),
        ("pi", "variant", {"paligemma": "./variant", "action_expert": "gemma_300m_lora"}, {"paligemma": str((Path.cwd() / "variant").resolve()), "action_expert": "gemma_300m_lora"}),
        ("encoder", "key_order", ["./tactilekey", "observation.images.tactile_right_0", "observation.images.tactile_left_1", "observation.images.tactile_right_1"], [str((Path.cwd() / "tactilekey").resolve()), "observation.images.tactile_right_0", "observation.images.tactile_left_1", "observation.images.tactile_right_1"]),
    ],
)
def test_loader_does_not_canonicalize_nonpath_source_contract_values(
    tmp_path: Path, group: str, field: str, actual_value: object, expected_value: object
):
    from deploy_baseline_pi05.checkpoint import load_decoder

    path, expected, _ = _training_checkpoint(tmp_path)
    expected_group = expected[group]
    assert isinstance(expected_group, dict)
    expected_group[field] = expected_value
    raw = torch.load(path, weights_only=True)
    raw_group = raw["source_contract"][group]
    raw_group[field] = actual_value
    torch.save(raw, path)

    with pytest.raises(ValueError, match="source_contract"):
        load_decoder(path, device="cpu", expected_source=expected)
