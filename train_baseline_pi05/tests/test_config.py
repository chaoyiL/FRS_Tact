from __future__ import annotations

from pathlib import Path
import subprocess
import sys

import pytest
import yaml

from train_baseline_pi05.config import DecoderTrainConfig, load_config


ROOT = Path(__file__).resolve().parents[2]


def test_default_yaml_locks_direct_decoder_contract():
    config = load_config(ROOT / "train_baseline_pi05/configs/train_baseline_pi05.yaml")
    assert config.source.action_horizon == 50
    assert config.source.model_action_dim == 20
    assert config.source.paligemma_variant == "gemma_2b_lora"
    assert config.source.action_expert_variant == "gemma_300m_lora"
    assert config.source.use_quantile_norm is True
    assert config.dataset.revision is None
    assert config.dataset.rename_map == {
        "observation.images.camera0": "observation.images.camera1",
        "observation.images.camera1": "observation.images.camera2",
    }
    assert config.dataset.camera_map == {
        "left_wrist_0_rgb": "observation.images.camera1",
        "right_wrist_0_rgb": "observation.images.camera2",
    }
    assert config.dataset.frame_stride == 5
    assert config.cache.action_batch_size == 4
    assert config.decoder.num_layers == 2
    assert config.decoder.d_model == 128
    assert config.decoder.tactile_keys == (
        "observation.images.tactile_left_0",
        "observation.images.tactile_right_0",
        "observation.images.tactile_left_1",
        "observation.images.tactile_right_1",
    )


def test_decoder_defaults_to_cuda(tmp_path: Path):
    assert DecoderTrainConfig(output=tmp_path).device == "cuda"
    assert DecoderTrainConfig.from_mapping({"output": str(tmp_path)}).device == "cuda"


def test_action_prefetch_is_opt_in_and_requires_boolean(tmp_path: Path):
    from train_baseline_pi05.config import CacheConfig

    raw = {"action_root": str(tmp_path / "actions"), "tactile_root": str(tmp_path / "tactile")}
    assert CacheConfig.from_mapping(raw).action_prefetch is False
    assert CacheConfig.from_mapping({**raw, "action_prefetch": True}).action_prefetch is True
    with pytest.raises(ValueError, match="boolean"):
        CacheConfig.from_mapping({**raw, "action_prefetch": "false"})


@pytest.mark.parametrize("action_dim", [10, 20])
@pytest.mark.parametrize("right_only", [True, False])
def test_load_config_accepts_supported_action_and_tactile_contracts(tmp_path: Path, action_dim: int, right_only: bool):
    raw = yaml.safe_load((ROOT / "train_baseline_pi05/configs/train_baseline_pi05.yaml").read_text())
    raw["source"]["model_action_dim"] = action_dim
    raw["decoder"]["action_dim"] = action_dim
    if right_only:
        raw["decoder"]["tactile_keys"] = [
            "observation.images.tactile_left_1", "observation.images.tactile_right_1",
        ]
    path = tmp_path / "supported.yaml"
    path.write_text(yaml.safe_dump(raw))

    config = load_config(path)

    assert config.decoder.action_dim == action_dim
    assert len(config.decoder.tactile_keys) == (2 if right_only else 4)


def test_config_rejects_old_cross_arm_tactile_pair(tmp_path):
    with pytest.raises(ValueError, match="tactile_keys"):
        DecoderTrainConfig.from_mapping({
            "output": str(tmp_path),
            "tactile_keys": ["observation.images.tactile_right_0", "observation.images.tactile_right_1"],
        })


@pytest.mark.parametrize("name", ["task3", "task3_5080", "task4"])
def test_right_arm_profiles_use_camera1_faces_and_fresh_outputs(name):
    config = load_config(ROOT / f"train_baseline_pi05/configs/train_baseline_pi05_{name}.yaml")
    assert config.decoder.tactile_keys == ("observation.images.tactile_left_1", "observation.images.tactile_right_1")
    task = name.split("_")[0]
    assert config.cache.action_root.parent.name == task
    assert config.cache.tactile_root.parent.name == f"{task}_right_two_face"
    assert config.decoder.output.parent.name == f"{task}_right_two_face"
    assert config.decoder.resume is False


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
        raw["source"]["model_action_dim"] = 19
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


@pytest.mark.parametrize(
    ("field", "invalid_value"),
    [
        ("action_horizon", 49),
        ("action_dim", 19),
        ("tactile_dim", 256),
        ("d_model", 64),
        ("nhead", 3),
        ("num_layers", 3),
        ("dim_feedforward", 128),
        ("dropout", 0.2),
    ],
)
def test_load_config_rejects_any_noncanonical_decoder_architecture(
    tmp_path: Path, field: str, invalid_value: int | float
) -> None:
    default_path = ROOT / "train_baseline_pi05/configs/train_baseline_pi05.yaml"
    raw = yaml.safe_load(default_path.read_text(encoding="utf-8"))
    raw["decoder"][field] = invalid_value
    bad_path = tmp_path / f"bad-{field}.yaml"
    bad_path.write_text(yaml.safe_dump(raw), encoding="utf-8")

    with pytest.raises(ValueError, match=rf"decoder\.{field}"):
        load_config(bad_path)
