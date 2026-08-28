import pytest

from dataclasses import asdict
import os
from pathlib import Path
import subprocess
from types import SimpleNamespace

from train_deco.train import (
    action_mode_config_fields,
    augmentation_config_from_args,
    build_argument_parser,
    resolve_dataset_action_mode,
    training_dataset_source,
)
from train_deco.input_adapter import augmentation_preset
from train_deco.resume import validate_resume_config


def test_accepts_precomputed_tcp_delta_with_absolute_gripper():
    contract = {"action_mode": "tcp_delta_absolute_gripper"}
    assert resolve_dataset_action_mode(contract) == "tcp_delta_absolute_gripper"


def test_legacy_delta_flag_must_not_describe_mixed_action_mode():
    contract = {
        "action_mode": "tcp_delta_absolute_gripper",
        "use_delta_action": True,
    }
    with pytest.raises(ValueError, match="use_delta_action"):
        resolve_dataset_action_mode(contract)


def test_mixed_action_checkpoint_omits_legacy_delta_flag():
    assert action_mode_config_fields("tcp_delta_absolute_gripper") == {}
    assert action_mode_config_fields("delta") == {"use_delta_action": True}
    assert action_mode_config_fields("absolute") == {"use_delta_action": False}


def test_lerobot_training_accepts_multiroot_manifest():
    args = SimpleNamespace(
        dataset_format="lerobot-v21",
        dataset_dir=None,
        dataset_manifest="/tmp/pick-tube-01-06.json",
    )
    assert training_dataset_source(args) == "/tmp/pick-tube-01-06.json"


def test_preprocessed_training_rejects_multiroot_manifest():
    args = SimpleNamespace(
        dataset_format="preprocessed",
        dataset_dir="/tmp/preprocessed",
        dataset_manifest="/tmp/pick-tube-01-06.json",
    )
    with pytest.raises(ValueError, match="dataset-manifest"):
        training_dataset_source(args)


def _augmentation_args(**overrides):
    values = {
        "augmentation_preset": None,
        "augmentation_enabled": True,
        "augmentation_identity_probability": 0.25,
        "augmentation_low_light_probability": 0.55,
        "augmentation_mild_probability": 0.20,
        "augmentation_exposure_probability": 0.5,
        "augmentation_exposure_range": (0.58, 0.90),
        "augmentation_gamma_range": (1.10, 1.50),
        "augmentation_mild_brightness_range": (0.90, 1.10),
        "augmentation_contrast_range": (0.85, 1.10),
        "augmentation_saturation_range": (0.90, 1.10),
        "augmentation_blur_probability": 0.20,
        "augmentation_blur_kernel_sizes": (3, 5),
        "augmentation_blur_sigma_range": (0.1, 1.0),
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_training_arguments_build_approved_low_light_config():
    args = _augmentation_args()

    config = asdict(augmentation_config_from_args(args))

    assert config["version"] == "low-light-v1"
    assert config["enabled"] is True
    assert config["exposure_range"] == (0.58, 0.90)
    assert config["gamma_range"] == (1.10, 1.50)
    assert config["shared_across_cameras"] is True


def test_training_arguments_resolve_balanced_light_v2_atomically():
    args = _augmentation_args(augmentation_preset="balanced-light-v2")

    config = asdict(augmentation_config_from_args(args))

    assert config["version"] == "balanced-light-v2"
    assert config["identity_probability"] == 0.25
    assert config["low_light_probability"] == 0.0
    assert config["mild_probability"] == 0.75
    assert config["mild_brightness_range"] == (0.90, 1.20)


def test_named_preset_rejects_fine_grained_conflicts():
    args = _augmentation_args(
        augmentation_preset="balanced-light-v2",
        augmentation_mild_probability=0.50,
    )

    with pytest.raises(ValueError, match="cannot be combined"):
        augmentation_config_from_args(args)


def test_python_entrypoint_defaults_to_legacy_augmentation_path():
    args = build_argument_parser().parse_args([])

    assert args.augmentation_preset is None
    assert augmentation_config_from_args(args).version == "low-light-v1"


def test_exact_resume_rejects_changed_augmentation_config():
    checkpoint = {
        "training_state_version": 2,
        "augmentation": {"version": "low-light-v1", "exposure_range": (0.58, 0.90)},
    }
    current = {
        **checkpoint,
        "augmentation": {"version": "low-light-v1", "exposure_range": (0.70, 0.90)},
    }

    with pytest.raises(ValueError, match="augmentation"):
        validate_resume_config(checkpoint, current, resume_mode="exact")


def test_exact_resume_rejects_cross_preset_configuration():
    checkpoint = {
        "training_state_version": 2,
        "augmentation": asdict(augmentation_preset("low-light-v1")),
    }
    current = {
        **checkpoint,
        "augmentation": asdict(augmentation_preset("balanced-light-v2")),
    }

    with pytest.raises(ValueError, match="augmentation"):
        validate_resume_config(checkpoint, current, resume_mode="exact")


@pytest.mark.parametrize("preset", ("low-light-v1", "balanced-light-v2"))
def test_exact_resume_accepts_same_preset_augmentation(preset):
    augmentation = asdict(augmentation_preset(preset))
    checkpoint = {
        "training_state_version": 2,
        "augmentation": augmentation,
    }

    validate_resume_config(
        checkpoint,
        {**checkpoint, "augmentation": dict(augmentation)},
        resume_mode="exact",
    )


def _launcher_dry_run(**environment):
    launcher = Path(__file__).parents[1] / "scripts" / "train.sh"
    return subprocess.run(
        ["bash", str(launcher), "--mode", "local-smoke", "--dry-run"],
        cwd=Path(__file__).parents[2],
        env={**os.environ, **environment},
        text=True,
        capture_output=True,
        check=True,
    ).stdout


def test_train_launcher_defaults_new_runs_to_balanced_light_v2():
    output = _launcher_dry_run(RUN_ID="balanced-light-test")

    assert "--augmentation-preset balanced-light-v2" in output
    assert "--run-id balanced-light-test" in output


def test_train_launcher_can_explicitly_select_low_light_v1():
    output = _launcher_dry_run(AUGMENTATION_PRESET="low-light-v1")

    assert "--augmentation-preset low-light-v1" in output


def test_train_launcher_preserves_run_id_and_accepts_yes_for_augmentation():
    launcher = Path(__file__).parents[1] / "scripts" / "train.sh"

    result = subprocess.run(
        ["bash", str(launcher), "--mode", "local-smoke", "--dry-run"],
        cwd=Path(__file__).parents[2],
        env={
            **os.environ,
            "RUN_ID": "deco_low_light_test",
            "AUGMENTATION_ENABLED": "yes",
        },
        text=True,
        capture_output=True,
        check=True,
    )

    assert "--run-id deco_low_light_test" in result.stdout
    assert "--augmentation-enabled" in result.stdout
    assert "--no-augmentation-enabled" not in result.stdout
