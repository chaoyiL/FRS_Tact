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
    build_argument_parser,
    STAGE2_CHECKPOINT_SCHEMA_VERSION,
    build_stage2_checkpoint_metadata,
    export_stage2_torchscript_artifacts,
    create_training_datasets,
    apply_restored_dataset_stats,
    resolve_tactile_encoder_distributed,
    restore_stage2_training_state,
    restore_stage2_resume_arguments,
    run_epoch,
    stage2_config_from_resume_checkpoint,
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
        for encoder in (self.img_encoder, self.tactile_encoder):
            for parameter in encoder.parameters():
                parameter.requires_grad_(False)

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

class _AblationBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.tactile_gate = nn.Parameter(torch.tensor(0.75))


class _AblationPolicy(_TrainPolicy):
    def __init__(self) -> None:
        super().__init__()
        self.mmattn = nn.ModuleList([_AblationBlock()])
        self.seen_gate_values: list[float] = []

    def forward(self, *args, **kwargs):
        self.seen_gate_values.append(float(self.mmattn[0].tactile_gate.detach()))
        return super().forward(*args, **kwargs)



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




class _RecordingGradScaler:
    def __init__(self, scale: float = 8.0) -> None:
        self.scale_value = scale
        self.unscale_calls = 0

    def scale(self, loss):
        return loss * self.scale_value

    def unscale_(self, optimizer) -> None:
        self.unscale_calls += 1
        for group in optimizer.param_groups:
            for parameter in group["params"]:
                if parameter.grad is not None:
                    parameter.grad.div_(self.scale_value)

    def step(self, optimizer) -> None:
        optimizer.step()

    def update(self) -> None:
        return None

    def get_scale(self) -> float:
        return self.scale_value

    def is_enabled(self) -> bool:
        return True
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




def _valid_resume_checkpoint(adapter_rank: int = 32) -> dict:
    return {
        "checkpoint_schema_version": STAGE2_CHECKPOINT_SCHEMA_VERSION,
        "stage": 2,
        "model_type": STAGE2_MODEL_TYPE,
        "config": {
            "training_state_version": 3,
            "model_type": STAGE2_MODEL_TYPE,
            "stage": 2,
            "hidden_dim": 64,
            "layers": 2,
            "heads": 4,
            "image_size": 64,
            "inference_steps": 3,
            "rope_height": 64,
            "rope_width": 64,
            "use_task_condition": False,
            "tactile_adapter_rank": adapter_rank,
        },
        "stage2_metadata": {
            "model_type": STAGE2_MODEL_TYPE,
            "tactile_field_order": list(TACTILE_NAMES),
            "stage1_checkpoint": {
                "path": "stage1.pt",
                "sha256": "1" * 64,
            },
            "tactile_encoder": {
                "source_sha256": "2" * 64,
                "artifact_sha256": "3" * 64,
                "artifact_path": "encoder.safetensors",
                "metadata_path": "encoder.json",
                "architecture": "resnet18",
                "embedding_dim": 512,
            },
            "tactile_adapter_rank": adapter_rank,
            "gate_values": {},
            "parameter_categories": {
                "trainable": {"test": ["weight"]},
                "frozen": {
                    "test": [
                        "frozen",
                        "img_encoder.weight",
                        "img_encoder.bias",
                        "tactile_encoder.weight",
                        "tactile_encoder.bias",
                    ]
                },
            },
            "parameter_counts": {"total": 14, "trainable": 1},
        },
        "stats": {
            "observation_mean": [0.0],
            "observation_std": [1.0],
            "action_mean": [0.0],
            "action_std": [1.0],
        },
    }
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





def test_stage2_tactile_disabled_validation_temporarily_zeros_and_restores_gates() -> None:
    model = _AblationPolicy()
    scaler = torch.amp.GradScaler("cuda", enabled=False)

    run_epoch(
        model=model, loader=[_batch()], device=torch.device("cpu"),
        optimizer=None, scheduler=None, scaler=scaler,
        observation_index=torch.tensor([0]), image_size=4,
        use_task_condition=False, train=False, world_size=1, stage=2,
        tactile_ablation="disabled",
    )

    assert model.seen_gate_values == [0.0]
    assert model.mmattn[0].tactile_gate.item() == pytest.approx(0.75)


def test_stage2_shuffled_tactile_validation_rolls_batch_without_consuming_rng() -> None:
    batch = {
        key: value.repeat(2, *([1] * (value.ndim - 1)))
        for key, value in _batch().items()
    }
    batch["tactile_images"][0].zero_()
    batch["tactile_images"][1].fill_(1.0)
    normal = _AblationPolicy()
    shuffled = _AblationPolicy()
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    common = dict(
        loader=[batch], device=torch.device("cpu"), optimizer=None,
        scheduler=None, scaler=scaler, observation_index=torch.tensor([0]),
        image_size=4, use_task_condition=False, train=False, world_size=1,
        stage=2, validation_seed=17,
    )
    run_epoch(model=normal, **common)
    rng_before = torch.get_rng_state().clone()
    run_epoch(model=shuffled, tactile_ablation="shuffled", **common)

    assert torch.equal(torch.get_rng_state(), rng_before)
    assert torch.equal(
        shuffled.seen_tactile[0], torch.roll(normal.seen_tactile[0], 1, dims=0)
    )

def test_stage2_gradient_diagnostics_are_recorded_after_amp_unscale() -> None:
    model = _TrainPolicy()
    optimizer, scheduler = _optimizer(model)
    scaler = _RecordingGradScaler(scale=8.0)
    report = SimpleNamespace(
        trainable_by_category={"test": ("weight",)},
        frozen_by_category={"test": ("frozen",)},
    )

    metrics, _ = run_epoch(
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
        stage2_parameter_report=report,
    )

    assert scaler.unscale_calls == 1
    assert metrics["gradient_norms"]["trainable"] == pytest.approx(3.0)
    assert metrics["gradient_norms"]["frozen"] == 0.0
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

def test_stage2_post_save_export_creates_epoch_latest_best_and_sidecars(tmp_path) -> None:
    pt = tmp_path / "deco_stage2_epoch_3.pt"
    pt.write_bytes(b"durable checkpoint")

    def fake_exporter(policy, stats, config, output, *args, **kwargs):
        del policy, stats, config, args, kwargs
        output.write_bytes(b"torchscript")
        output.with_suffix(output.suffix + ".json").write_text(
            json.dumps({"output_path": str(output)}), encoding="utf-8"
        )
        return {"output_path": str(output), "format": "fixture"}

    events = export_stage2_torchscript_artifacts(
        policy=object(), stats={}, config={}, stage2_metadata={},
        output_dir=tmp_path, epoch=3, val_loss=0.2,
        image_height=8, image_width=12, periodic=True, improved=True,
        exporter=fake_exporter,
    )

    assert pt.read_bytes() == b"durable checkpoint"
    for name in (
        "deco_stage2_epoch_3.ts", "deco_stage2_latest.ts",
        "deco_stage2_best.ts",
    ):
        assert (tmp_path / name).is_file()
        assert (tmp_path / f"{name}.json").is_file()
    assert events[0]["event"] == "torchscript_saved"


def test_stage2_post_save_export_failure_preserves_pt_and_returns_explicit_event(tmp_path) -> None:
    pt = tmp_path / "deco_stage2_latest.pt"
    pt.write_bytes(b"durable checkpoint")

    events = export_stage2_torchscript_artifacts(
        policy=object(), stats={}, config={}, stage2_metadata={},
        output_dir=tmp_path, epoch=4, val_loss=0.3,
        image_height=8, image_width=12, periodic=True, improved=False,
        exporter=lambda *args, **kwargs: (_ for _ in ()).throw(
            RuntimeError("trace failed")
        ),
    )

    assert pt.read_bytes() == b"durable checkpoint"
    assert events == [{
        "event": "torchscript_export_failed",
        "stage": 2,
        "epoch": 4,
        "error": "RuntimeError: trace failed",
    }]





def test_stage2_resume_model_config_is_checkpoint_driven_for_nondefault_rank() -> None:
    checkpoint = _valid_resume_checkpoint(adapter_rank=64)
    current = {
        key: 999
        for key in (
            "hidden_dim", "layers", "heads", "image_size", "inference_steps",
            "rope_height", "rope_width", "tactile_adapter_rank",
        )
    }
    current["use_task_condition"] = True

    resolved = stage2_config_from_resume_checkpoint(checkpoint, current)

    assert resolved["tactile_adapter_rank"] == 64
    assert resolved["hidden_dim"] == checkpoint["config"]["hidden_dim"]
    assert resolved["use_task_condition"] is False


def test_stage2_resume_restores_state_arguments_and_keeps_only_runtime_overrides() -> None:
    checkpoint = _valid_resume_checkpoint(adapter_rank=64)
    checkpoint["config"].update({
        "dataset_manifest": "/saved/nondefault.json",
        "dataset_format": "lerobot-v21",
        "action_chunk_size": 17,
        "batch_size": 3,
        "lr": 7e-5,
        "lr_final": 7e-5,
        "lr_scheduler": "constant",
        "weight_decay": 9e-6,
        "warmup_epochs": 0,
        "cosine_t_max_epochs": 23,
        "seed": 91,
        "augmentation_enabled": False,
        "stage1_checkpoint": "/missing/stage1.pt",
        "tactile_encoder_checkpoint": "/missing/encoder",
    })
    args = build_argument_parser().parse_args([
        "--stage", "2", "--resume", "/runtime/stage2.pt",
        "--output-dir", "/runtime/output", "--run-id", "runtime-run",
        "--epochs", "99", "--workers", "6", "--validation-seed", "444",
    ])

    restored = restore_stage2_resume_arguments(
        args, checkpoint_loader=lambda path, device: checkpoint
    )

    assert restored is checkpoint
    assert args.dataset_manifest == "/saved/nondefault.json"
    assert args.action_chunk_size == 17
    assert args.batch_size == 3
    assert args.lr == 7e-5
    assert args.lr_scheduler == "constant"
    assert args.weight_decay == 9e-6
    assert args.seed == 91
    assert args.augmentation_enabled is False
    assert args.tactile_adapter_rank == 64
    assert args.stage1_checkpoint is None
    assert args.tactile_encoder_checkpoint is None
    assert args.output_dir == "/runtime/output"
    assert args.run_id == "runtime-run"
    assert args.epochs == 99
    assert args.workers == 6
    assert args.validation_seed == 444


@pytest.mark.parametrize(
    ("path", "bad_value", "message"),
    [
        (("stage2_metadata", "tactile_adapter_rank"), -1, "positive integer"),
        (("stage2_metadata", "gate_values"), {"bad": "string"}, "gate_values"),
        (("stage2_metadata", "stage1_checkpoint", "path"), "", "non-empty path"),
        (("stage2_metadata", "tactile_encoder", "source_sha256"), "bad", "SHA256"),
        (("stage2_metadata", "parameter_categories", "trainable"), [], "mapping"),
    ],
)
def test_stage2_resume_rejects_corrupt_nested_metadata(
    path, bad_value, message
) -> None:
    checkpoint = _valid_resume_checkpoint()
    target = checkpoint
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = bad_value
    if path[-1] == "tactile_adapter_rank":
        checkpoint["config"]["tactile_adapter_rank"] = bad_value

    with pytest.raises(ValueError, match=message):
        validate_stage2_resume_checkpoint(checkpoint)


def test_restored_stats_replace_all_dataset_normalization_state() -> None:
    train_dataset = SimpleNamespace(stats={"action_mean": torch.tensor([99.0])})
    val_dataset = SimpleNamespace(stats={"action_mean": torch.tensor([88.0])})
    checkpoint_stats = _valid_resume_checkpoint()["stats"]

    restored = apply_restored_dataset_stats(
        checkpoint_stats, train_dataset, val_dataset
    )

    assert restored["action_mean"].tolist() == [0.0]
    assert train_dataset.stats["action_mean"].tolist() == [0.0]
    assert val_dataset.stats["action_mean"].tolist() == [0.0]
    assert train_dataset.stats["action_mean"] is not val_dataset.stats["action_mean"]


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

    payload = _valid_resume_checkpoint()
    config = payload["config"]
    payload.update(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "epoch": 1,
            "global_step": global_step,
            "best_val": 0.4,
            "patience_best_val": 0.4,
            "stale_epochs": 0,
            "rng_states": [capture_rng_state()],
        }
    )
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

