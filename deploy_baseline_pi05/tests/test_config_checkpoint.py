from __future__ import annotations

import importlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch
import yaml

from train_baseline_pi05.checkpoint import save_best_checkpoint, save_last_checkpoint
from train_baseline_pi05.model import DirectDecoderConfig as TrainDecoderConfig
from train_baseline_pi05.model import DirectTactileActionDecoder as TrainDecoder


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "deploy_baseline_pi05.yaml"
TRAINED_PROMPT = (
    "Firstly, use the left hand to pick up the blue tube, and then use the right hand to "
    "pick up the green tube. Next, use the left hand to place the blue tube back firstly, "
    "and then use the right hand to place the green tube back."
)


def _fake_asset_config(tmp_path: Path, *, missing: str | None = None) -> tuple[Path, dict[str, Path]]:
    config_dir = tmp_path / "config"
    assets = tmp_path / "assets"
    config_dir.mkdir()
    paths = {
        "source": assets / "pi05",
        "params": assets / "pi05" / "params",
        "norm_stats": assets / "norm" / "two_tubes_0102" / "norm_stats.json",
        "encoder_metadata": assets / "encoder" / "checkpoint.json",
        "encoder_params": assets / "encoder" / "custom-params.npz",
        "decoder": assets / "decoder" / "best.pt",
    }
    if missing != "source":
        paths["source"].mkdir(parents=True)
    if missing != "params" and missing != "source":
        paths["params"].mkdir()
    if missing != "norm_stats":
        paths["norm_stats"].parent.mkdir(parents=True)
        paths["norm_stats"].write_text("{}\n", encoding="utf-8")
    if missing != "encoder_metadata":
        paths["encoder_metadata"].parent.mkdir(parents=True)
        paths["encoder_metadata"].write_text(
            json.dumps({"params_file": paths["encoder_params"].name}), encoding="utf-8"
        )
    if missing != "encoder_params":
        paths["encoder_params"].parent.mkdir(parents=True, exist_ok=True)
        paths["encoder_params"].write_bytes(b"fake encoder params")
    if missing != "decoder":
        paths["decoder"].parent.mkdir(parents=True)
        paths["decoder"].write_bytes(b"fake decoder")

    raw = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    raw["source"]["checkpoint"] = "../assets/pi05"
    raw["norm_stats"]["dir"] = "../assets/norm"
    raw["tactile_encoder"]["checkpoint"] = "../assets/encoder"
    raw["direct_decoder"]["checkpoint"] = "../assets/decoder/best.pt"
    config_path = config_dir / "deploy.yaml"
    config_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    return config_path, paths


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
    assert config.observation.language_prompt == TRAINED_PROMPT
    assert len(config.model.camera_map) == 2


def test_relative_asset_paths_resolve_once_against_yaml_directory_from_any_cwd(
    tmp_path: Path, monkeypatch
) -> None:
    from deploy_baseline_pi05.deployment import expected_source_contract, load_deployment_config
    from deploy_baseline_pi05.remote_client import _trace_identity

    config_path, paths = _fake_asset_config(tmp_path)
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)

    config = load_deployment_config(config_path)
    assert config.source.checkpoint == paths["source"].resolve()
    assert config.norm_stats.directory == (tmp_path / "assets/norm").resolve()
    assert config.tactile_encoder.checkpoint == paths["encoder_metadata"].parent.resolve()
    assert config.direct_decoder.checkpoint == paths["decoder"].resolve()
    expected = expected_source_contract(config)
    identity = _trace_identity(config)
    assert expected["pi"]["checkpoint"] == identity["source_checkpoint"]
    assert expected["encoder"]["checkpoint"] == identity["tactile_encoder_checkpoint"]
    assert identity["direct_decoder_checkpoint"] == str(paths["decoder"].resolve())

    captured: dict[str, object] = {}

    class Policy:
        def __init__(self, received):
            captured["policy_source"] = received.source.checkpoint

    class Encoder:
        def __init__(self, checkpoint, *, tactile_keys):
            captured["encoder"] = checkpoint
            self.tactile_keys = tactile_keys

    def load_decoder(checkpoint, *, device, expected_source):
        captured["decoder"] = checkpoint
        captured["expected_source"] = expected_source
        return object()

    class Runtime:
        def __init__(self, **kwargs):
            captured["runtime"] = kwargs

    monkeypatch.setitem(
        sys.modules,
        "deploy_baseline_pi05.policy",
        type("PolicyModule", (), {"Pi05VisualPolicy": Policy}),
    )
    monkeypatch.setitem(
        sys.modules,
        "deploy_baseline_pi05.tactile_encoder",
        type("EncoderModule", (), {"FrozenTactileEncoder": Encoder}),
    )
    monkeypatch.setitem(
        sys.modules,
        "deploy_baseline_pi05.checkpoint",
        type("CheckpointModule", (), {"load_decoder": staticmethod(load_decoder)}),
    )
    monkeypatch.setitem(
        sys.modules,
        "deploy_baseline_pi05.runtime",
        type("RuntimeModule", (), {"DirectDecoderRuntime": Runtime}),
    )
    from deploy_baseline_pi05.remote_client import _make_runtime

    _make_runtime(config)
    assert captured["policy_source"] == paths["source"].resolve()
    assert captured["encoder"] == paths["encoder_metadata"].parent.resolve()
    assert captured["decoder"] == paths["decoder"].resolve()
    assert captured["expected_source"] == expected


def test_check_accepts_a_complete_lightweight_asset_tree(tmp_path: Path) -> None:
    from deploy_baseline_pi05.remote_client import check

    config_path, _paths = _fake_asset_config(tmp_path)
    config = check(config_path)
    assert config.config_path == config_path.resolve()


@pytest.mark.parametrize(
    ("missing", "expected"),
    [
        ("source", "source checkpoint"),
        ("params", "params"),
        ("norm_stats", "norm_stats.json"),
        ("encoder_metadata", "checkpoint.json"),
        ("encoder_params", "custom-params.npz"),
        ("decoder", "best.pt"),
    ],
)
def test_check_cli_fails_for_each_missing_asset_without_heavy_imports(
    tmp_path: Path, missing: str, expected: str
) -> None:
    config_path, _paths = _fake_asset_config(tmp_path, missing=missing)
    environment = {
        **os.environ,
        "PYTHONPATH": os.pathsep.join((str(ROOT / "src"), str(ROOT.parent))),
        "PYTHONPROFILEIMPORTTIME": "1",
    }
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "deploy_baseline_pi05.remote_client",
            "--config",
            str(config_path),
            "--check",
        ],
        cwd=tmp_path,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode != 0
    assert expected in result.stderr
    imported = result.stderr.lower()
    assert not any(f"import time:" in line and module in line for line in imported.splitlines() for module in ("jax", "torch", "websockets"))


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
