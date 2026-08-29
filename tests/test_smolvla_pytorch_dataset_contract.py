from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from train_smolvla.torch_train import (
    _accelerate_command,
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
    config_path = Path(__file__).parents[1] / "train_smolvla/configs/train_pytorch.yaml"
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
