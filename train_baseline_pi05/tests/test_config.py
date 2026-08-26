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


@pytest.mark.parametrize(
    ("case", "expected_error"),
    [
        ("quoted_boolean", "boolean"),
        ("nonpositive_dimension", "d_model"),
        ("wrong_decoder_layer_count", "num_layers"),
        ("wrong_action_dimension", "action_dim"),
        ("wrong_action_horizon", "action_horizon"),
        ("wrong_tactile_dimension", "embedding_dim"),
        ("duplicate_tactile_key", "tactile_keys"),
        ("reordered_tactile_keys", "tactile_keys"),
        ("invalid_split_sum", "split fractions"),
        ("output_overlaps_input", "overlaps an input asset root"),
    ],
)
def test_load_config_rejects_strict_contract_violations(
    tmp_path: Path, case: str, expected_error: str
):
    default_path = ROOT / "train_baseline_pi05/configs/train_baseline_pi05.yaml"
    raw = yaml.safe_load(default_path.read_text(encoding="utf-8"))
    if case == "quoted_boolean":
        raw["tactile"]["freeze_encoder"] = "true"
    elif case == "nonpositive_dimension":
        raw["decoder"]["d_model"] = 0
    elif case == "wrong_decoder_layer_count":
        raw["decoder"]["num_layers"] = 3
    elif case == "wrong_action_dimension":
        raw["source"]["action_dim"] = 19
    elif case == "wrong_action_horizon":
        raw["source"]["action_horizon"] = 49
    elif case == "wrong_tactile_dimension":
        raw["tactile"]["embedding_dim"] = 256
    elif case == "duplicate_tactile_key":
        raw["decoder"]["tactile_keys"][1] = raw["decoder"]["tactile_keys"][0]
    elif case == "reordered_tactile_keys":
        raw["decoder"]["tactile_keys"][0], raw["decoder"]["tactile_keys"][1] = (
            raw["decoder"]["tactile_keys"][1],
            raw["decoder"]["tactile_keys"][0],
        )
    elif case == "invalid_split_sum":
        raw["dataset"]["test_fraction"] = 0.2
    elif case == "output_overlaps_input":
        raw["decoder"]["output"] = raw["dataset"]["root"]
    else:
        raise AssertionError(f"Unhandled test case: {case}")

    bad_path = tmp_path / "bad.yaml"
    bad_path.write_text(yaml.safe_dump(raw), encoding="utf-8")

    with pytest.raises(ValueError, match=expected_error):
        load_config(bad_path)
