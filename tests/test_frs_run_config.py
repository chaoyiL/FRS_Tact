from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from train_smolvla_frs.train_frs import resolve_decode_solver
from train_smolvla_frs.train_frs import resolve_optional_loss_weight
from train_smolvla_frs.train_frs import RUN_CONFIG_NAME, save_run_config
from train_smolvla_frs.utils.loss_ablation import (
    DEFAULT_ABLATION_REPAIR_WEIGHT,
    LOSS_ABLATION_SWITCHES,
    build_loss_ablation_runs,
    write_loss_ablation_configs,
)


def test_default_frs_config_is_pick05_state_conditioned_asymmetric_objective() -> None:
    config_path = Path(__file__).parents[1] / "train_smolvla_frs" / "configs" / "train_frs.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    assert [source["repo_id"] for source in config["datasets"]] == [
        "KaiyueChen/pick_tube_05"
    ]
    training = config["frs_training"]
    assert training["dataset_balanced_sampling"] is False
    assert training["dataset_balanced_loss"] is False
    assert training["early_stop_patience"] == 5
    assert training["early_stop_min_evals"] == 5
    assert training["output"].endswith("frs_0815_01_state")
    assert training["init_from"] is None
    model = config["model"]
    assert model["state_conditioning"] is True
    assert model["state_dropout_rate"] == 0.1
    assert config["action_cache"]["root"].endswith("action_cache_slerpflow_state_v3")
    assert training["learning_rate"] == 1.0e-4
    assert training["gate_lambda"] == 0.25
    assert training["aux_decode_weight"] == 4.0
    assert training["aux_decode_solver"] == "fireflow"
    assert training["low_gate_safety_weight"] == 0.5
    assert training["low_gate_safety_margin"] == 0.03
    assert training["best_max_low_gate_unsafe_frac"] == 0.1
    assert training["repair_weight"] == 2.0
    assert training["repair_margin"] == 0.0
    assert training["rank_weight"] == 2.0
    assert training["rank_margin"] == 0.0
    assert training["high_gate_rank_aggregation"] == "balanced_mean"
    assert training["high_gate_rank_hard_fraction"] == 1.0
    assert training["high_gate_rank_worst_beta"] == 20.0
    assert training["high_gate_rank_source_weights"] == {}
    assert training["resume"] is False


def test_resolve_decode_solver_accepts_fireflow() -> None:
    assert resolve_decode_solver("FireFlow") == "fireflow"
    assert resolve_decode_solver(None) == "euler"
    with pytest.raises(ValueError, match="decode solver"):
        resolve_decode_solver("slerpflow")


def test_resolve_optional_loss_weight_is_the_on_off_switch() -> None:
    assert resolve_optional_loss_weight(True, 4.0) == 4.0
    assert resolve_optional_loss_weight(False, 4.0) == 0.0
    assert resolve_optional_loss_weight(None, 0.5) == 0.5
    with pytest.raises(ValueError, match="must be >= 0"):
        resolve_optional_loss_weight(True, -0.1)


def test_train_smolvla_frs_yaml_exposes_optional_loss_switches() -> None:
    config_path = Path(__file__).parents[1] / "train_smolvla_frs" / "configs" / "train_frs.yaml"
    training = yaml.safe_load(config_path.read_text(encoding="utf-8"))["frs_training"]
    assert training["aux_decode"] is True
    assert training["low_gate_safety"] is True
    assert training["rank"] is True
    assert training["repair"] is True
    assert training["aux_decode_weight"] == 4.0
    assert training["low_gate_safety_weight"] == 0.5
    assert training["rank_weight"] == 2.0
    assert training["repair_weight"] == 0.0


def test_loss_ablation_closes_one_switch_and_keeps_the_other_three(tmp_path: Path) -> None:
    config_path = Path(__file__).parents[1] / "train_smolvla_frs" / "configs" / "train_frs.yaml"
    base = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    output_root = tmp_path / "checkpoints" / "frs"

    runs = build_loss_ablation_runs(base, output_root=output_root)

    assert [name for name, _config in runs] == [f"no_{name}" for name in LOSS_ABLATION_SWITCHES]
    for disabled, (_name, config) in zip(LOSS_ABLATION_SWITCHES, runs, strict=True):
        training = config["frs_training"]
        for switch in LOSS_ABLATION_SWITCHES:
            assert training[switch] is (switch != disabled)
        assert training["output"] == str((output_root / f"no_{disabled}").resolve())
        if training["repair"]:
            assert training["repair_weight"] == DEFAULT_ABLATION_REPAIR_WEIGHT
        else:
            assert training["repair_weight"] == 0.0

    written = write_loss_ablation_configs(base, output_root=output_root)
    assert [name for name, _path in written] == [name for name, _config in runs]
    for name, path in written:
        assert path.is_file()
        dumped = yaml.safe_load(path.read_text(encoding="utf-8"))
        assert dumped["frs_training"]["output"].endswith(f"/{name}")


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
