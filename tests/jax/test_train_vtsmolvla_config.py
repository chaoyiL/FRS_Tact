from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest
import yaml

from tools.train_vtsmolvla_jax import _validate_vt_config


ROOT = Path(__file__).resolve().parents[2]


def _load_repo_yaml(name: str) -> dict:
    return yaml.safe_load((ROOT / "configs" / name).read_text())


def _scientific_config(config: dict) -> dict:
    normalized = deepcopy(config)
    normalized.pop("output", None)
    wandb = normalized.get("wandb") or {}
    wandb.pop("name", None)
    wandb.pop("tags", None)
    normalized["wandb"] = wandb
    normalized["model"] = dict(normalized["model"])
    normalized["model"].pop("tactile_token_repeat_factor", None)
    return normalized


def test_paper_ratio_configs_only_change_factor_and_output_identity() -> None:
    base = _load_repo_yaml("train_vtsmolvla_jax.yaml")
    tactile16 = _load_repo_yaml("train_vtsmolvla_jax_tactile16.yaml")
    tactile32 = _load_repo_yaml("train_vtsmolvla_jax_tactile32.yaml")

    assert base["model"]["tactile_token_repeat_factor"] == 1
    assert tactile16["model"]["tactile_token_repeat_factor"] == 8
    assert tactile32["model"]["tactile_token_repeat_factor"] == 21
    assert _scientific_config(base) == _scientific_config(tactile16)
    assert _scientific_config(base) == _scientific_config(tactile32)
    assert len({base["output"], tactile16["output"], tactile32["output"]}) == 3


def _valid_config() -> dict:
    return {
        "model": {
            "use_tactile_encoder": True,
            "tactile_encoder_path": "encoder",
            "tactile_keys": ["t0", "t1", "t2", "t3"],
            "tactile_embedding_dim": 512,
            "tactile_num_tokens": 4,
            "tactile_token_repeat_factor": 8,
            "image_keys": ["camera1", "camera2"],
        },
        "tactile_embedding_cache": {"enabled": False},
    }


def _write(path: Path, config: dict) -> Path:
    path.write_text(yaml.safe_dump(config, sort_keys=False))
    return path


def test_vt_launcher_accepts_legacy_default_and_paper_factors(tmp_path: Path) -> None:
    config = _valid_config()
    for factor in (1, 8, 21):
        config["model"]["tactile_token_repeat_factor"] = factor
        _validate_vt_config(_write(tmp_path / f"k{factor}.yaml", config))

    del config["model"]["tactile_token_repeat_factor"]
    _validate_vt_config(_write(tmp_path / "legacy.yaml", config))


@pytest.mark.parametrize("invalid", [0, -1, 1.5, True, "8"])
def test_vt_launcher_rejects_invalid_repeat_factor(tmp_path: Path, invalid: object) -> None:
    config = deepcopy(_valid_config())
    config["model"]["tactile_token_repeat_factor"] = invalid
    with pytest.raises(ValueError, match="tactile_token_repeat_factor"):
        _validate_vt_config(_write(tmp_path / "invalid.yaml", config))
