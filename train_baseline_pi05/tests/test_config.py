from __future__ import annotations

from pathlib import Path
import subprocess
import sys

import pytest
import yaml

from train_baseline_pi05.config import load_config


ROOT = Path(__file__).resolve().parents[2]


def test_default_yaml_locks_direct_decoder_contract():
    config = load_config(ROOT / "train_baseline_pi05/configs/train_baseline_pi05.yaml")
    assert config.source.action_horizon == 50
    assert config.source.action_dim == 20
    assert config.decoder.num_layers == 2
    assert config.decoder.d_model == 128
    assert config.decoder.tactile_keys == (
        "observation.images.tactile_left_0",
        "observation.images.tactile_right_0",
        "observation.images.tactile_left_1",
        "observation.images.tactile_right_1",
    )


def test_config_import_does_not_import_heavy_runtimes():
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            "import train_baseline_pi05.config; "
            "import sys; assert 'jax' not in sys.modules; assert 'torch' not in sys.modules",
        ],
        cwd=ROOT,
        check=False,
    )
    assert completed.returncode == 0


def test_load_config_rejects_contract_violations(tmp_path: Path):
    default_path = ROOT / "train_baseline_pi05/configs/train_baseline_pi05.yaml"
    raw = yaml.safe_load(default_path.read_text(encoding="utf-8"))
    raw["source"]["action_horizon"] = 49
    bad_path = tmp_path / "bad.yaml"
    bad_path.write_text(yaml.safe_dump(raw), encoding="utf-8")

    with pytest.raises(ValueError, match="action_horizon"):
        load_config(bad_path)
