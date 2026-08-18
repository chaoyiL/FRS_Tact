from __future__ import annotations

import pytest

from deploy_pi05_frs.frs_config import validate_frs_config_section


def _config() -> dict[str, object]:
    return {
        "observation": {"data_type": "vitac"},
        "control": {"steps_per_inference": 50, "action_horizon": 50},
        "frs": {
            "enabled": True,
            "checkpoint": "/models/frs/best",
            "tactile_encoder_checkpoint": "/models/tactile_encoder",
            "tactile_keys": ["observation.images.tactile_left_0"],
            "tactile_window_divisor": 5,
            "reverse_steps": 50,
            "reverse_solver": "slerpflow",
            "decode_steps": 10,
            "decode_solver": "fireflow",
            "verify_source_checkpoint_fingerprint": False,
        },
    }


def test_real_frs_config_validator_accepts_valid_config() -> None:
    validate_frs_config_section(_config())


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("enabled", "true"),
        ("verify_source_checkpoint_fingerprint", "false"),
    ],
)
def test_real_frs_config_validator_rejects_pseudo_booleans(key, value) -> None:
    config = _config()
    config["frs"][key] = value

    with pytest.raises(ValueError, match=rf"frs\.{key} must be a boolean"):
        validate_frs_config_section(config)


def test_real_frs_config_validator_rejects_boolean_integer() -> None:
    config = _config()
    config["frs"]["tactile_window_divisor"] = True

    with pytest.raises(ValueError, match=r"frs\.tactile_window_divisor must be an integer"):
        validate_frs_config_section(config)
