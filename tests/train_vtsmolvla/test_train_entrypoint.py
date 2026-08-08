from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

import yaml


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
