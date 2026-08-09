from __future__ import annotations

import argparse
import ast
import subprocess
import sys
from pathlib import Path

import pytest
import yaml


from train_vtsmolvla import train as vt_train
from train_vtsmolvla.checkpoint import initialize_tactile_fusion_params

ROOT = Path(__file__).resolve().parents[2]


def _run_module_help(module: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", module, "--help"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


def test_vt_cli_has_no_parameter_overrides() -> None:
    result = _run_module_help("train_vtsmolvla.train")

    assert result.returncode == 0, result.stderr
    assert "--config" in result.stdout
    for forbidden in ("--batch-size", "--steps", "--output", "--checkpoint"):
        assert forbidden not in result.stdout


def test_vt_yaml_is_the_complete_parameter_source() -> None:
    cfg = yaml.safe_load(
        (ROOT / "train_vtsmolvla/configs/train.yaml").read_text(encoding="utf-8")
    )

    assert {
        "checkpoint",
        "datasets",
        "output",
        "steps",
        "batch_size",
        "num_workers",
        "log_freq",
        "validation",
        "image_transforms",
        "modality_dropout",
        "wandb",
        "model",
        "resume",
        "launcher",
        "tactile_embedding_cache",
    } <= cfg.keys()
    assert cfg["model"]["use_tactile_encoder"] is True


def test_vt_entrypoint_uses_explicit_components_without_argv_mutation() -> None:
    source_path = ROOT / "train_vtsmolvla/train.py"
    tree = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))

    assert "sys.argv" not in source_path.read_text(encoding="utf-8")
    imported_modules = {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module is not None
    }
    assert "tools.train_smolvla_jax" not in imported_modules
    assert "train_smolvla_jax" not in imported_modules
    for module in (
        "train_vtsmolvla.configuration",
        "train_vtsmolvla.modeling",
        "train_vtsmolvla.data",
        "train_vtsmolvla.training",
        "train_vtsmolvla.checkpoint",
        "train_vtsmolvla.validation",
    ):
        assert module in imported_modules


def test_vt_main_passes_the_complete_vt_component_bundle(monkeypatch, tmp_path):
    config_path = tmp_path / "train.yaml"
    config_path.write_text(
        "model:\n"
        "  use_tactile_encoder: true\n"
        "  tactile_encoder_path: /encoder\n"
        "  freeze_tactile_encoder: true\n"
        "  tactile_keys: [left]\n"
        "  tactile_embedding_dim: 512\n"
        "  tactile_num_tokens: 1\n",
        encoding="utf-8",
    )
    recorded = []
    monkeypatch.setattr(
        vt_train,
        "parse_args",
        lambda argv, **kwargs: argparse.Namespace(config=config_path),
    )
    monkeypatch.setattr(
        vt_train,
        "run_training",
        lambda path, *, components: recorded.append((path, components)),
    )

    vt_train.main([])

    assert recorded == [(config_path, vt_train.VT_COMPONENTS)]
    assert vt_train.VT_COMPONENTS.prepare_params is initialize_tactile_fusion_params


def test_vt_config_rejects_unfrozen_tactile_encoder_before_training(tmp_path):
    config_path = tmp_path / "train.yaml"
    config_path.write_text(
        "model:\n"
        "  use_tactile_encoder: true\n"
        "  tactile_encoder_path: /encoder\n"
        "  freeze_tactile_encoder: false\n"
        "  tactile_keys: [left]\n"
        "  tactile_embedding_dim: 512\n"
        "  tactile_num_tokens: 1\n",
        encoding="utf-8",
    )

    with pytest.raises(NotImplementedError, match="freeze_tactile_encoder=True"):
        vt_train._validate_vt_config(config_path)
