from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from train_frs.train_frs import RUN_CONFIG_NAME, save_run_config


def test_default_frs_config_enables_balanced_training_and_early_stop() -> None:
    config_path = Path(__file__).parents[1] / "train_frs" / "configs" / "train_frs.yaml"
    training = yaml.safe_load(config_path.read_text(encoding="utf-8"))["frs_training"]
    assert training["dataset_balanced_sampling"] is True
    assert training["dataset_balanced_loss"] is True
    assert training["early_stop_patience"] == 5
    assert training["early_stop_min_evals"] == 5
    assert training["resume"] is False


def test_save_run_config_writes_effective_yaml(tmp_path: Path) -> None:
    config = {
        "datasets": [{"repo_id": "owner/data"}],
        "frs_training": {"output": str(tmp_path), "batch_size": 128},
    }

    written = save_run_config(config, output_dir=tmp_path)

    assert written == tmp_path / RUN_CONFIG_NAME
    assert yaml.safe_load(written.read_text(encoding="utf-8")) == config
    assert save_run_config(config, output_dir=tmp_path) == written


def test_save_run_config_rejects_different_parameters(tmp_path: Path) -> None:
    original = {"frs_training": {"output": str(tmp_path), "batch_size": 128}}
    changed = {"frs_training": {"output": str(tmp_path), "batch_size": 256}}
    save_run_config(original, output_dir=tmp_path)

    with pytest.raises(ValueError, match="different train_config.yaml"):
        save_run_config(changed, output_dir=tmp_path)
