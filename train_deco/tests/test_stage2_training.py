from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from train_deco.checkpoint import atomic_torch_save, capture_rng_state, load_checkpoint
from train_deco.lerobot_vision_dataset import TACTILE_NAMES
from train_deco.model_factory import STAGE2_MODEL_TYPE
from train_deco.stage2_initialization import configure_stage2_trainability
from train_deco.tactile_encoder_conversion import ResolvedTactileEncoder
from train_deco.train import (
    STAGE2_CHECKPOINT_SCHEMA_VERSION,
    build_stage2_checkpoint_metadata,
    create_training_datasets,
    resolve_tactile_encoder_distributed,
    restore_stage2_training_state,
    run_epoch,
    validate_stage2_resume_checkpoint,
)
from train_deco.training_utils import (
    constant_lr_scheduler,
    stage2_gradient_diagnostics,
    stage2_optimizer_parameter_groups,
)


class _Stage2Block(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.tactile_key = nn.Linear(2, 2)
        self.tactile_value = nn.Linear(2, 2)
        self.tactile_gate = nn.Parameter(torch.tensor(0.25))
        self.img_qkv_pi = nn.Linear(2, 2)


class _BoundaryModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.tactile_image_mode = True
        self.stage1_weight = nn.Parameter(torch.ones(2, 2))
        self.tactile_encoder = nn.Linear(2, 2)
        self.sensor_embeddings = nn.Embedding(4, 2)
        self.mmattn = nn.ModuleList([_Stage2Block()])


class _TrainPolicy(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.img_encoder = nn.BatchNorm2d(3)
        self.tactile_encoder = nn.BatchNorm2d(3)
        self.weight = nn.Parameter(torch.tensor(0.5))
        self.frozen = nn.Parameter(torch.tensor(3.0), requires_grad=False)
        self.seen_tactile: list[torch.Tensor] = []

    def forward(
        self,
        image0,
        image1,
        *,
        obs,
        act=None,
        task_idx=None,
        training=True,
        tactile_images=None,
    ):
        del image0, image1, obs, task_idx
        assert tactile_images is not None
        self.seen_tactile.append(tactile_images.detach().clone())
        if training:
            noise = torch.zeros_like(act)
            return act * self.weight, noise
        return torch.zeros(
            tactile_images.shape[0], 2, 1, device=tactile_images.device
        ) + self.weight * 0


def _batch() -> dict[str, torch.Tensor]:
    return {
        "observation": torch.zeros(1, 1),
        "images": torch.rand(1, 2, 3, 4, 6),
        "tactile_images": torch.rand(1, 4, 3, 3, 5),
        "action": torch.ones(1, 2, 1),
        "is_pad": torch.zeros(1, 2, dtype=torch.bool),
        "task_index": torch.zeros(1, dtype=torch.long),
    }


def _optimizer(model: nn.Module):
    optimizer = torch.optim.AdamW(
        [{"group_name": "policy_decay", "params": [model.weight], "lr": 0.1}],
        betas=(0.95, 0.999),
    )
    return optimizer, constant_lr_scheduler(optimizer)


def _artifact(tmp_path: Path) -> ResolvedTactileEncoder:
    weights = tmp_path / "encoder.safetensors"
    weights.write_bytes(b"resolved tactile weights")
    weights_sha256 = hashlib.sha256(weights.read_bytes()).hexdigest()
    metadata = tmp_path / "encoder.json"
    metadata.write_text(
        json.dumps({"weights_sha256": weights_sha256}), encoding="utf-8"
    )
    return ResolvedTactileEncoder(
        weights_path=weights,
        metadata_path=metadata,
        source_sha256="source-digest",
        architecture="resnet18",
        embedding_dim=512,
    )


def test_stage2_dataset_mode_requests_tactile_streams_only_for_stage2(monkeypatch) -> None:
    calls: list[bool] = []

    def builder(*args, **kwargs):
        del args
        calls.append(kwargs["include_tactile"])
        return object(), object()

    monkeypatch.setattr(
        "train_deco.lerobot_vision_dataset.build_lerobot_vision_datasets", builder
    )
    base = dict(
        dataset_format="lerobot-v21",
        dataset_manifest="manifest.json",
        dataset_dir=None,
        action_chunk_size=2,
        validation_ratio=0.1,
        episode_split_seed=42,
        limit_samples=None,
    )

    create_training_datasets(SimpleNamespace(stage=1, **base))
    create_training_datasets(SimpleNamespace(stage=2, **base))

    assert calls == [False, True]


class _Broadcast:
    def __init__(self, incoming=None) -> None:
        self.incoming = incoming
        self.broadcasts = []
        self.barriers = 0

    def broadcast_object_list(self, values, src, device=None) -> None:
        del src, device
        if self.incoming is not None:
            values[0] = copy.deepcopy(self.incoming)
        self.broadcasts.append(copy.deepcopy(values[0]))

    def barrier(self) -> None:
        self.barriers += 1


def test_rank0_resolves_once_then_broadcasts_the_exact_artifact(monkeypatch, tmp_path) -> None:
    artifact = _artifact(tmp_path)
    calls = []
    fake_dist = _Broadcast()
    monkeypatch.setattr("train_deco.train.dist", fake_dist)

    resolved = resolve_tactile_encoder_distributed(
        "encoder-source",
        "cache",
        rank=0,
        world_size=2,
        device=torch.device("cpu"),
        resolver=lambda source, cache: calls.append((source, cache)) or artifact,
    )

    assert calls == [("encoder-source", "cache")]
    assert resolved == artifact
    assert fake_dist.broadcasts[0]["weights_path"] == str(artifact.weights_path)
    assert fake_dist.barriers == 1


def test_nonzero_rank_never_calls_converter_and_loads_broadcast_artifact(
    monkeypatch, tmp_path
) -> None:
    artifact = _artifact(tmp_path)
    payload = {
        "ok": True,
        "weights_path": str(artifact.weights_path),
        "metadata_path": str(artifact.metadata_path),
        "source_sha256": artifact.source_sha256,
        "architecture": artifact.architecture,
        "embedding_dim": artifact.embedding_dim,
    }
    fake_dist = _Broadcast(payload)
    monkeypatch.setattr("train_deco.train.dist", fake_dist)

    resolved = resolve_tactile_encoder_distributed(
        "must-not-be-read",
        "cache",
        rank=1,
        world_size=2,
        device=torch.device("cpu"),
        resolver=lambda *_: pytest.fail("nonzero rank imported/called converter"),
    )

    assert resolved == artifact
    assert fake_dist.barriers == 1


def test_rank0_conversion_error_is_broadcast_before_all_ranks_fail(monkeypatch) -> None:
    fake_dist = _Broadcast()
    monkeypatch.setattr("train_deco.train.dist", fake_dist)

    with pytest.raises(RuntimeError, match="ValueError: invalid encoder"):
        resolve_tactile_encoder_distributed(
            "bad",
            "cache",
            rank=0,
            world_size=2,
            device=torch.device("cpu"),
            resolver=lambda *_: (_ for _ in ()).throw(ValueError("invalid encoder")),
        )

    assert fake_dist.broadcasts == [
        {"ok": False, "error": "ValueError: invalid encoder"}
    ]


def test_stage2_optimizer_contains_every_trainable_and_no_frozen_parameter() -> None:
    model = _BoundaryModel()
    report = configure_stage2_trainability(model)

    groups = stage2_optimizer_parameter_groups(model, learning_rate=1e-3, weight_decay=1e-6)
    optimized = {
        id(parameter) for group in groups for parameter in group["params"]
    }
    trainable = {
        id(parameter) for parameter in model.parameters() if parameter.requires_grad
    }
    frozen = {
        id(parameter) for parameter in model.parameters() if not parameter.requires_grad
    }

    assert optimized == trainable
    assert optimized.isdisjoint(frozen)
    assert report.trainable_parameters < report.total_parameters


def test_stage2_diagnostics_report_gates_and_trainable_frozen_gradient_norms() -> None:
    model = _BoundaryModel()
    report = configure_stage2_trainability(model)
    next(parameter for parameter in model.parameters() if parameter.requires_grad).grad = torch.ones(4, 2)

    diagnostics = stage2_gradient_diagnostics(model, report)

    assert diagnostics["gate_values"] == {"mmattn.0.tactile_gate": 0.25}
    assert diagnostics["gradient_norms"]["trainable"] > 0
    assert diagnostics["gradient_norms"]["frozen"] == 0


@pytest.mark.parametrize("train", [True, False])
def test_run_epoch_preprocesses_and_passes_tactile_images_in_both_paths(train) -> None:
    model = _TrainPolicy()
    optimizer, scheduler = _optimizer(model)
    scaler = torch.amp.GradScaler("cuda", enabled=False)

    _, step = run_epoch(
        model=model,
        loader=[_batch()],
        device=torch.device("cpu"),
        optimizer=optimizer if train else None,
        scheduler=scheduler if train else None,
        scaler=scaler,
        observation_index=torch.tensor([0]),
        image_size=4,
        use_task_condition=False,
        train=train,
        world_size=1,
        stage=2,
    )

    assert model.seen_tactile[0].shape == (1, 4, 3, 224, 224)
    assert model.seen_tactile[0].min() >= 0
    assert model.seen_tactile[0].max() <= 1
    assert step == int(train)
    if train:
        assert model.img_encoder.training is False
        assert model.tactile_encoder.training is False


def test_stage2_metadata_records_provenance_categories_and_gate_values(tmp_path) -> None:
    model = _BoundaryModel()
    report = configure_stage2_trainability(model)
    stage1 = tmp_path / "stage1.pt"
    stage1.write_bytes(b"stage one")
    artifact = _artifact(tmp_path)

    metadata = build_stage2_checkpoint_metadata(
        model, report, stage1_checkpoint=stage1, tactile_artifact=artifact,
        tactile_adapter_rank=32,
    )

    assert metadata["model_type"] == STAGE2_MODEL_TYPE
    assert metadata["tactile_field_order"] == list(TACTILE_NAMES)
    assert metadata["tactile_encoder"]["source_sha256"] == "source-digest"
    assert metadata["tactile_encoder"]["artifact_path"] == str(artifact.weights_path.resolve())
    assert metadata["tactile_encoder"]["artifact_sha256"] == hashlib.sha256(
        artifact.weights_path.read_bytes()
    ).hexdigest()
    assert metadata["tactile_adapter_rank"] == 32
    assert metadata["gate_values"] == {"mmattn.0.tactile_gate": 0.25}
    assert metadata["parameter_categories"]["trainable"]
    assert metadata["parameter_categories"]["frozen"]
    assert metadata["stage1_checkpoint"]["sha256"] == hashlib.sha256(b"stage one").hexdigest()


def test_stage2_resume_rejects_stage1_checkpoint() -> None:
    with pytest.raises(ValueError, match="Stage2.*resume.*Stage1"):
        validate_stage2_resume_checkpoint(
            {"config": {"model_type": "upstream-deco-stage1"}}
        )


def test_cpu_synthetic_step_checkpoint_and_exact_resume_continue_state(tmp_path) -> None:
    torch.manual_seed(17)
    model = _TrainPolicy()
    frozen_before = model.frozen.detach().clone()
    weight_before = model.weight.detach().clone()
    optimizer, scheduler = _optimizer(model)
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    _, global_step = run_epoch(
        model=model,
        loader=[_batch()],
        device=torch.device("cpu"),
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        observation_index=torch.tensor([0]),
        image_size=4,
        use_task_condition=False,
        train=True,
        world_size=1,
        stage=2,
    )
    assert not torch.equal(model.weight, weight_before)
    assert torch.equal(model.frozen, frozen_before)

    config = {
        "training_state_version": 3,
        "model_type": STAGE2_MODEL_TYPE,
        "stage": 2,
    }
    payload = {
        "checkpoint_schema_version": STAGE2_CHECKPOINT_SCHEMA_VERSION,
        "stage": 2,
        "model_type": STAGE2_MODEL_TYPE,
        "stage2_metadata": {
            "model_type": STAGE2_MODEL_TYPE,
            "tactile_field_order": list(TACTILE_NAMES),
            "stage1_checkpoint": {"path": "stage1.pt", "sha256": "s1"},
            "tactile_encoder": {
                "source_sha256": "source",
                "artifact_sha256": "artifact",
                "artifact_path": "encoder.safetensors",
            },
            "tactile_adapter_rank": 32,
            "gate_values": {},
            "parameter_categories": {"trainable": {}, "frozen": {}},
        },
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(),
        "epoch": 1,
        "global_step": global_step,
        "best_val": 0.4,
        "patience_best_val": 0.4,
        "stale_epochs": 0,
        "stats": {"action_mean": [0.0]},
        "config": config,
        "rng_states": [capture_rng_state()],
    }
    checkpoint_path = tmp_path / "deco_stage2_latest.pt"
    atomic_torch_save(payload, checkpoint_path)

    resumed_model = _TrainPolicy()
    resumed_optimizer, resumed_scheduler = _optimizer(resumed_model)
    resumed_scaler = torch.amp.GradScaler("cuda", enabled=False)
    state = restore_stage2_training_state(
        load_checkpoint(checkpoint_path, "cpu"),
        model=resumed_model,
        optimizer=resumed_optimizer,
        scheduler=resumed_scheduler,
        scaler=resumed_scaler,
        current_config=config,
        world_size=1,
        rank=0,
    )

    assert state["epoch"] == 1
    assert state["global_step"] == 1
    assert state["stats"] == payload["stats"]
    assert state["config"] == config
    assert torch.equal(resumed_model.weight, model.weight)
    assert resumed_scheduler.state_dict() == scheduler.state_dict()
    resumed_optimizer_state = next(iter(resumed_optimizer.state.values()))
    original_optimizer_state = next(iter(optimizer.state.values()))
    assert torch.equal(
        resumed_optimizer_state["exp_avg"], original_optimizer_state["exp_avg"]
    )

    _, next_step = run_epoch(
        model=resumed_model,
        loader=[_batch()],
        device=torch.device("cpu"),
        optimizer=resumed_optimizer,
        scheduler=resumed_scheduler,
        scaler=resumed_scaler,
        observation_index=torch.tensor([0]),
        image_size=4,
        use_task_condition=False,
        train=True,
        world_size=1,
        stage=2,
        global_step=state["global_step"],
    )
    assert next_step == 2
    assert int(next(iter(resumed_optimizer.state.values()))["step"].item()) == 2

