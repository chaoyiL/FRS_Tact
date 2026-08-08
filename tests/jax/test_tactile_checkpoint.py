from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from lerobot.policies.smolvla_jax.checkpoint import write_effective_config
from lerobot.policies.smolvla_jax.configuration import JaxSmolVLAConfig


def test_tactile_repeat_factor_defaults_validates_and_derives_effective_tokens() -> None:
    legacy = JaxSmolVLAConfig(
        use_tactile_encoder=True,
        tactile_encoder_path="encoder",
        tactile_keys=("t0", "t1", "t2", "t3"),
        tactile_num_tokens=4,
    )
    assert legacy.tactile_token_repeat_factor == 1
    assert legacy.effective_tactile_num_tokens == 4

    expanded = legacy.with_overrides({"tactile_token_repeat_factor": 8})
    assert expanded.tactile_token_repeat_factor == 8
    assert expanded.effective_tactile_num_tokens == 32

    for invalid in (0, -1, 1.5, True, "8"):
        with pytest.raises(ValueError, match="tactile_token_repeat_factor"):
            legacy.with_overrides({"tactile_token_repeat_factor": invalid})


def test_effective_config_persists_tactile_repeat_factor(tmp_path: Path) -> None:
    config = replace(
        JaxSmolVLAConfig(),
        use_tactile_encoder=True,
        tactile_encoder_path="encoder",
        tactile_keys=("t0", "t1", "t2", "t3"),
        tactile_num_tokens=4,
        tactile_token_repeat_factor=21,
    )
    write_effective_config(tmp_path, config)
    raw = json.loads((tmp_path / "config.json").read_text())
    assert raw["tactile_token_repeat_factor"] == 21
    assert JaxSmolVLAConfig.from_pretrained(tmp_path).tactile_token_repeat_factor == 21


def test_from_pretrained_rejects_invalid_tactile_repeat_factor(tmp_path: Path) -> None:
    (tmp_path / "config.json").write_text(
        json.dumps({"tactile_token_repeat_factor": 0})
    )
    with pytest.raises(ValueError, match="tactile_token_repeat_factor"):
        JaxSmolVLAConfig.from_pretrained(tmp_path)


@pytest.mark.parametrize("invalid", (0, -1, 1.5, True, "8"))
def test_direct_config_rejects_invalid_tactile_repeat_factor(invalid: object) -> None:
    with pytest.raises(ValueError, match="tactile_token_repeat_factor"):
        JaxSmolVLAConfig(tactile_token_repeat_factor=invalid)  # type: ignore[arg-type]


@pytest.mark.parametrize("invalid", (0, -1, 1.5, True, "8"))
def test_replace_rejects_invalid_tactile_repeat_factor(invalid: object) -> None:
    with pytest.raises(ValueError, match="tactile_token_repeat_factor"):
        replace(JaxSmolVLAConfig(), tactile_token_repeat_factor=invalid)


def test_from_pretrained_defaults_missing_tactile_repeat_factor_to_one(tmp_path: Path) -> None:
    (tmp_path / "config.json").write_text(json.dumps({}))
    assert JaxSmolVLAConfig.from_pretrained(tmp_path).tactile_token_repeat_factor == 1


def test_trainable_compute_dtype_defaults_validates_and_persists(tmp_path: Path) -> None:
    legacy = JaxSmolVLAConfig()
    assert legacy.trainable_compute_dtype == "bfloat16"

    write_effective_config(tmp_path, legacy)
    raw = json.loads((tmp_path / "config.json").read_text())
    assert raw["trainable_compute_dtype"] == "bfloat16"
    assert JaxSmolVLAConfig.from_pretrained(tmp_path).trainable_compute_dtype == "bfloat16"


@pytest.mark.parametrize("invalid", ("float32", "float16", "BF16", "", None, 16))
def test_trainable_compute_dtype_rejects_invalid_explicit_values(invalid: object) -> None:
    with pytest.raises(ValueError, match="trainable_compute_dtype"):
        JaxSmolVLAConfig(trainable_compute_dtype=invalid)  # type: ignore[arg-type]


def test_from_pretrained_legacy_missing_trainable_compute_dtype_defaults_bfloat16(
    tmp_path: Path,
) -> None:
    (tmp_path / "config.json").write_text(json.dumps({}))
    assert JaxSmolVLAConfig.from_pretrained(tmp_path).trainable_compute_dtype == "bfloat16"


def test_from_pretrained_rejects_invalid_trainable_compute_dtype(tmp_path: Path) -> None:
    (tmp_path / "config.json").write_text(
        json.dumps({"trainable_compute_dtype": "float32"})
    )
    with pytest.raises(ValueError, match="trainable_compute_dtype"):
        JaxSmolVLAConfig.from_pretrained(tmp_path)
