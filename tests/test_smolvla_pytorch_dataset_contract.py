from __future__ import annotations

from datetime import timedelta
import json
import os
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest
import yaml

from train_smolvla.torch_train import (
    _accelerate_command,
    _configure_single_gpu_precision,
    _effective_output_dir,
    _install_accelerate_timeout,
    _prepare_output_dir,
    _select_dataset_cameras,
    build_command,
    resolve_wandb_mode,
    validate_constructed_policy,
    validate_dataset_contract,
)


def _config(root: Path) -> dict:
    return {
        "dataset": {
            "expected_fps": 30,
            "state_dim": 20,
            "action_dim": 20,
            "image_keys": [
                "observation.images.camera0",
                "observation.images.camera1",
            ],
        },
        "datasets": [{"repo_id": "test/dataset", "root": str(root)}],
    }


def test_visual_contract_allows_and_prunes_tactile_cameras(tmp_path: Path) -> None:
    features = {
        "observation.state": {"dtype": "float32", "shape": [20]},
        "action": {"dtype": "float32", "shape": [20]},
        "observation.images.camera0": {"dtype": "video", "shape": [3, 480, 640]},
        "observation.images.camera1": {"dtype": "video", "shape": [3, 480, 640]},
        "observation.images.tactile_left_0": {
            "dtype": "video",
            "shape": [3, 224, 224],
        },
    }
    info = {"codebase_version": "v3.0", "fps": 30, "features": features}
    info_path = tmp_path / "meta/info.json"
    info_path.parent.mkdir(parents=True)
    info_path.write_text(json.dumps(info), encoding="utf-8")

    validate_dataset_contract(_config(tmp_path))

    metadata = SimpleNamespace(
        features=features,
        info={"features": dict(features)},
        stats={key: {"mean": [0.0]} for key in features},
    )
    dataset = SimpleNamespace(
        meta=metadata,
        delta_timestamps={key: [0.0] for key in features},
        reader=SimpleNamespace(delta_indices={key: [0] for key in features}),
    )
    _select_dataset_cameras(
        dataset,
        {"observation.images.camera0", "observation.images.camera1"},
    )
    assert "observation.images.tactile_left_0" not in metadata.info["features"]
    assert "observation.images.tactile_left_0" not in metadata.stats
    assert "observation.images.tactile_left_0" not in dataset.delta_timestamps
    assert "observation.images.tactile_left_0" not in dataset.reader.delta_indices


def test_wandb_auto_is_offline_without_credentials(monkeypatch) -> None:
    config = {"wandb": {"enable": True, "mode": "auto"}}
    monkeypatch.delenv("WANDB_MODE", raising=False)
    monkeypatch.setattr("train_smolvla.torch_train._wandb_has_credentials", lambda: False)

    assert resolve_wandb_mode(config) == "offline"


def test_wandb_auto_honors_explicit_environment_mode(monkeypatch) -> None:
    config = {"wandb": {"enable": True, "mode": "auto"}}
    monkeypatch.setenv("WANDB_MODE", "disabled")

    assert resolve_wandb_mode(config) == "disabled"


def test_wandb_auto_rejects_unauthenticated_online_environment(monkeypatch) -> None:
    config = {"wandb": {"enable": True, "mode": "auto"}}
    monkeypatch.setenv("WANDB_MODE", "online")
    monkeypatch.setattr("train_smolvla.torch_train._wandb_has_credentials", lambda: False)

    assert resolve_wandb_mode(config) == "offline"


def test_default_command_resolves_wandb_and_accelerate_defaults(monkeypatch) -> None:
    config_path = Path(__file__).parents[1] / "train_smolvla/configs/train_smolvla.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    monkeypatch.delenv("WANDB_MODE", raising=False)
    monkeypatch.setattr("train_smolvla.torch_train._wandb_has_credentials", lambda: False)

    assert "--wandb.mode=offline" in build_command(config)
    assert "--dynamo_backend=no" in _accelerate_command(
        config_path, config, dry_run=True
    )


def test_constructed_policy_matches_dual_arm_contract() -> None:
    def feature(size: int) -> SimpleNamespace:
        return SimpleNamespace(shape=(size,))

    policy = SimpleNamespace(
        config=SimpleNamespace(
            input_features={
                "observation.state": feature(20),
                "observation.images.camera1": feature(3),
                "observation.images.camera2": feature(3),
            },
            output_features={"action": feature(20)},
            chunk_size=20,
            n_action_steps=10,
            num_vlm_layers=16,
            num_expert_layers=16,
        )
    )
    config = {
        "dataset": {"state_dim": 20, "action_dim": 20},
        "policy": {
            "chunk_size": 20,
            "n_action_steps": 10,
            "num_vlm_layers": 16,
            "num_expert_layers": 16,
        },
    }

    validate_constructed_policy(
        policy,
        config,
        ("observation.images.camera1", "observation.images.camera2"),
    )


def test_constructed_policy_rejects_pretrained_six_dimensional_defaults() -> None:
    policy = SimpleNamespace(
        config=SimpleNamespace(
            input_features={
                "observation.state": SimpleNamespace(shape=(6,)),
                "observation.images.camera1": SimpleNamespace(shape=(3, 256, 256)),
                "observation.images.camera2": SimpleNamespace(shape=(3, 256, 256)),
                "observation.images.camera3": SimpleNamespace(shape=(3, 256, 256)),
            },
            output_features={"action": SimpleNamespace(shape=(6,))},
            chunk_size=20,
            n_action_steps=10,
            num_vlm_layers=16,
            num_expert_layers=16,
        )
    )
    config = {
        "dataset": {"state_dim": 20, "action_dim": 20},
        "policy": {
            "chunk_size": 20,
            "n_action_steps": 10,
            "num_vlm_layers": 16,
            "num_expert_layers": 16,
        },
    }

    with pytest.raises(ValueError, match="constructed SmolVLA inputs"):
        validate_constructed_policy(
            policy,
            config,
            ("observation.images.camera1", "observation.images.camera2"),
        )


def test_accelerate_process_group_timeout_is_injected(monkeypatch) -> None:
    class FakeInitProcessGroupKwargs:
        def __init__(self, *, timeout) -> None:
            self.timeout = timeout

    received: dict = {}

    class FakeAccelerator:
        def __init__(self, *args, **kwargs) -> None:
            received["args"] = args
            received["kwargs"] = kwargs

    accelerate_module = ModuleType("accelerate")
    accelerate_module.Accelerator = FakeAccelerator
    accelerate_utils_module = ModuleType("accelerate.utils")
    accelerate_utils_module.InitProcessGroupKwargs = FakeInitProcessGroupKwargs
    monkeypatch.setitem(sys.modules, "accelerate", accelerate_module)
    monkeypatch.setitem(sys.modules, "accelerate.utils", accelerate_utils_module)

    restore = _install_accelerate_timeout(
        {"distributed": {"timeout_seconds": 7200}}
    )
    try:
        accelerate_module.Accelerator(kwargs_handlers=["ddp"])
    finally:
        restore()

    handlers = received["kwargs"]["kwargs_handlers"]
    assert handlers[0] == "ddp"
    assert handlers[1].timeout == timedelta(seconds=7200)
    assert accelerate_module.Accelerator is FakeAccelerator


def test_existing_output_directory_is_preserved_and_incremented(
    tmp_path: Path, monkeypatch
) -> None:
    output_dir = tmp_path / "smolvla_task1"
    output_dir.mkdir()
    marker = output_dir / "failed-run.log"
    marker.write_text("preserve me", encoding="utf-8")
    config = {
        "training": {
            "output_dir": str(output_dir),
            "existing_output": "increment",
            "resume_from": None,
        }
    }
    monkeypatch.delenv("FRS_SMOLVLA_OUTPUT_DIR", raising=False)

    selected = _prepare_output_dir(config)

    assert selected != output_dir
    assert selected.parent == output_dir.parent
    assert selected.name.startswith("smolvla_task1-")
    assert not selected.exists()
    assert marker.read_text(encoding="utf-8") == "preserve me"
    assert _effective_output_dir(config) == selected


@pytest.mark.parametrize(
    "launcher_name",
    ["start_smolvla_train.sh", "start_smolvla_right_train.sh"],
)
def test_launcher_uses_local_regenerable_arrow_cache(launcher_name: str) -> None:
    launcher = (
        Path(__file__).parents[1] / "train_smolvla" / "scripts" / launcher_name
    ).read_text(encoding="utf-8")

    assert 'SMOLVLA_USE_LOCAL_ARROW_CACHE:-1' in launcher
    assert '/tmp/frs_tact_smolvla' in launcher
    assert 'export HF_DATASETS_CACHE="${SMOLVLA_LOCAL_CACHE_ROOT}/datasets_arrow"' in launcher
    assert 'export TMPDIR="${SMOLVLA_LOCAL_CACHE_ROOT}/tmp"' in launcher


def test_training_configs_use_smolvla_names() -> None:
    config_dir = Path(__file__).parents[1] / "train_smolvla/configs"

    assert (config_dir / "train_smolvla.yaml").is_file()
    assert (config_dir / "train_smolvla_right.yaml").is_file()
    assert not (config_dir / "train_pytorch.yaml").exists()
    assert not (config_dir / "train_pytorch_right.yaml").exists()


def test_4090_smoke_config_keeps_contract_and_minimizes_training() -> None:
    config_path = (
        Path(__file__).parents[1]
        / "train_smolvla/configs/train_smolvla_4090_smoke.yaml"
    )
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    assert config["datasets"] == [
        {
            "repo_id": "KaiyueChen/two_tubes_04",
            "root": "/workspace/lerobot_v30/KaiyueChen/two_tubes_04",
        }
    ]
    assert config["dataset"]["state_dim"] == 20
    assert config["dataset"]["action_dim"] == 20
    assert len(config["dataset"]["image_keys"]) == 2
    assert config["augmentation"] == {
        "preset": "balanced-light-v2",
        "enabled": True,
    }
    assert config["training"]["batch_size"] == 1
    assert config["training"]["steps"] == 5
    assert config["training"]["save_freq"] == 5
    assert config["training"]["eval_steps"] == 0
    assert config["distributed"]["num_gpus"] == 1
    assert config["wandb"]["enable"] is False


def test_4090_smoke_launcher_has_real_staged_acceptance_checks() -> None:
    launcher = (
        Path(__file__).parents[1]
        / "train_smolvla/scripts/run_smolvla_4090_smoke.sh"
    ).read_text(encoding="utf-8")

    for stage in ("env", "data", "sample", "preflight", "train", "checkpoint"):
        assert f"should_run {stage}" in launcher
    assert 'video_backend="torchcodec"' in launcher
    assert "sample = dataset[0]" in launcher
    assert "five real forward/backward optimization steps" in launcher
    assert '"optimizer_state.safetensors"' in launcher
    assert 'step_state.get("step") != 5' in launcher
    assert '"observation.images.camera1"' in launcher
    assert '"observation.images.camera2"' in launcher


def test_single_gpu_precision_uses_yaml_value(monkeypatch) -> None:
    monkeypatch.delenv("ACCELERATE_MIXED_PRECISION", raising=False)

    _configure_single_gpu_precision(
        {"distributed": {"mixed_precision": "bf16"}}
    )

    assert os.environ["ACCELERATE_MIXED_PRECISION"] == "bf16"
