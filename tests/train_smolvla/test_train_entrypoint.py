from __future__ import annotations

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


def _load_training_yaml(relative_path: str) -> dict:
    return yaml.safe_load((ROOT / relative_path).read_text(encoding="utf-8"))


def test_visual_cli_has_no_parameter_overrides() -> None:
    result = _run_module_help("train_smolvla.train")

    assert result.returncode == 0, result.stderr
    assert "--config" in result.stdout
    for forbidden in ("--batch-size", "--steps", "--output", "--checkpoint"):
        assert forbidden not in result.stdout


def test_visual_yaml_is_the_complete_parameter_source() -> None:
    cfg = _load_training_yaml("train_smolvla/configs/train.yaml")

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
    } <= cfg.keys()
    assert cfg["launcher"] == {
        "tmux_session": "smolvla_train",
        "foreground": False,
        "logs_dir": "train_smolvla/outputs/logs",
    }


def test_visual_yaml_has_no_tactile_settings_or_examples() -> None:
    config_path = ROOT / "train_smolvla/configs/train.yaml"
    cfg = _load_training_yaml("train_smolvla/configs/train.yaml")

    assert "tactile_embedding_cache" not in cfg
    assert not any("tactile" in key for key in cfg.get("model", {}))
    assert "tactile" not in config_path.read_text(encoding="utf-8").lower()
