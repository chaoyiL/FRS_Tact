from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("jax")

from lerobot.policies.smolvla_jax.checkpoint import (  # noqa: E402
    load_safetensors_params,
    parameter_summary,
    save_portable_params,
    write_effective_config,
)
from lerobot.policies.smolvla_jax.configuration import JaxSmolVLAConfig  # noqa: E402


def test_portable_round_trip_and_manifest(tmp_path: Path) -> None:
    params = {
        "float.weight": np.arange(12, dtype=np.float32).reshape(3, 4),
        "int.value": np.arange(3, dtype=np.int32),
    }
    output = save_portable_params(params, tmp_path / "checkpoint")
    restored = load_safetensors_params(output)
    np.testing.assert_array_equal(restored["float.weight"], params["float.weight"])
    np.testing.assert_array_equal(restored["int.value"], params["int.value"])
    manifest = json.loads((output / "conversion_manifest.json").read_text())
    assert manifest["tensor_count"] == 2
    assert manifest["parameter_count"] == 15
    assert parameter_summary(restored)["layout"] == "pytorch_source_layout"


def test_effective_config_persists_dimensions_and_lora_settings(tmp_path: Path) -> None:
    config = replace(
        JaxSmolVLAConfig(),
        action_dim=20,
        state_dim=20,
        module_modes={
            "vision": "lora",
            "connector": "frozen",
            "vlm_text": "frozen",
            "expert": "full",
            "action": "full",
            "state_proj": "full",
        },
        lora_rank=4,
        lora_alpha=8.0,
    )
    write_effective_config(tmp_path, config)
    raw = json.loads((tmp_path / "config.json").read_text())
    assert raw["input_features"]["observation.state"]["shape"] == [20]
    assert raw["output_features"]["action"]["shape"] == [20]
    assert raw["module_modes"]["vision"] == "lora"
    assert raw["lora_rank"] == 4
    assert raw["lora_alpha"] == 8.0
