from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from train_frs.train_frs import RUN_CONFIG_NAME, save_run_config


def test_default_frs_config_is_single_dataset_absolute_repair_stage3() -> None:
    config_path = Path(__file__).parents[1] / "train_frs" / "configs" / "train_frs.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    assert [source["repo_id"] for source in config["datasets"]] == [
        "KaiyueChen/pick_tube_01"
    ]
    training = config["frs_training"]
    assert training["dataset_balanced_sampling"] is False
    assert training["dataset_balanced_loss"] is False
    assert training["early_stop_patience"] == 5
    assert training["early_stop_min_evals"] == 5
    assert training["output"].endswith("frs_0813_single_01_stage3")
    assert training["init_from"].endswith("frs_0813_single_01/best_rank")
    assert training["learning_rate"] == 1.0e-5
    assert training["aux_decode_weight"] == 4.0
    assert training["repair_weight"] == 4.0
    assert training["repair_margin"] == 0.0
    assert training["rank_weight"] == 1.0
    assert training["rank_margin"] == 0.0
    assert training["high_gate_rank_aggregation"] == "balanced_mean"
    assert training["high_gate_rank_hard_fraction"] == 1.0
    assert training["high_gate_rank_worst_beta"] == 20.0
    assert training["high_gate_rank_source_weights"] == {}
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
