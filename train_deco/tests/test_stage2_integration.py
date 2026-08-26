from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from PIL import Image
from safetensors.torch import save_file
from torch import nn

import train_deco.export_torchscript as export_module
from train_deco.checkpoint import atomic_torch_save, capture_rng_state, load_checkpoint
from train_deco.export_torchscript import STAGE2_EXPORT_FORMAT, export_checkpoint
from train_deco.input_adapter import letterbox_and_normalize, letterbox_tactile_images
from train_deco.lerobot_vision_dataset import (
    CAMERA_NAMES,
    TACTILE_NAMES,
    build_lerobot_vision_datasets,
)
from train_deco.model_factory import STAGE2_MODEL_TYPE, build_model, build_stage2_model
from train_deco.stage2_initialization import initialize_stage2_from_stage1
from train_deco.tactile_encoder_conversion import ResolvedTactileEncoder
from train_deco.train import (
    STAGE2_CHECKPOINT_SCHEMA_VERSION,
    build_stage2_checkpoint_metadata,
    jsonable_stats,
    restore_stage2_training_state,
)
from train_deco.training_utils import (
    constant_lr_scheduler,
    stage2_gradient_diagnostics,
    stage2_optimizer_parameter_groups,
)


class _TinyImageEncoder(nn.Module):
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        pooled = images.mean(dim=(1, 2, 3), keepdim=True)
        return pooled.reshape(-1, 1, 1, 1).expand(-1, 512, 2, 2)


class _TinyTactileEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.projection = nn.Linear(3, 512, bias=False)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.projection(images.mean(dim=(-2, -1)))


def _config(dataset_id: str) -> dict:
    return {
        "model_type": STAGE2_MODEL_TYPE,
        "stage": 2,
        "source_obs_dim": 20,
        "obs_dim": 20,
        "action_dim": 20,
        "chunk_size": 2,
        "observation_indices": list(range(20)),
        "hidden_dim": 32,
        "layers": 1,
        "heads": 4,
        "image_size": 32,
        "inference_steps": 2,
        "rope_height": 32,
        "rope_width": 32,
        "use_task_condition": False,
        "num_tasks": 1,
        "dataset_id": dataset_id,
        "action_mode": "tcp_delta_absolute_gripper",
        "camera_names": list(CAMERA_NAMES),
        "tactile_field_order": list(TACTILE_NAMES),
        "tactile_adapter_rank": 4,
        "training_state_version": 3,
        "world_size": 1,
        "batch_size": 1,
        "seed": 17,
        "train_samples": 2,
        "steps_per_epoch": 1,
        "objective_version": "masked-flow-mse-v1",
        "validation_metric_version": "no-repeat-masked-element-mean-v1",
        "validation_seed": 12345,
        "lr": 1e-3,
        "lr_final": 1e-3,
        "backbone_lr": 1e-4,
        "backbone_lr_final": 1e-4,
        "weight_decay": 1e-6,
        "warmup_steps": 0,
        "cosine_t_max_steps": 1,
        "backbone_freeze_steps": 0,
        "backbone_bn_eval": True,
        "scheduler_type": "per-step-constant-v1",
        "optimizer_group_names": ["policy_decay", "policy_no_decay"],
        "rank_seed_scheme": "base-plus-rank-v1",
        "augmentation": {"version": "disabled-test-fixture"},
        "early_stopping_min_delta": 0.0,
    }


def _jpeg_bytes(rgb: tuple[int, int, int]) -> bytes:
    encoded = io.BytesIO()
    Image.new("RGB", (5, 3), color=rgb).save(
        encoded, format="JPEG", quality=100, subsampling=0
    )
    return encoded.getvalue()


def _write_lerobot_fixture(root: Path) -> None:
    (root / "meta").mkdir(parents=True)
    features = {
        "observation.state": {"dtype": "float32", "shape": [20]},
        "actions": {"dtype": "float32", "shape": [20]},
        **{
            name: {"dtype": "image", "shape": [3, 5, 3]}
            for name in (*CAMERA_NAMES, *TACTILE_NAMES)
        },
    }
    info = {
        "codebase_version": "v2.1",
        "video_path": None,
        "total_videos": 0,
        "fps": 30,
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "chunks_size": 1000,
        "total_episodes": 2,
        "total_frames": 6,
        "total_tasks": 1,
        "features": features,
    }
    (root / "meta/info.json").write_text(json.dumps(info), encoding="utf-8")
    (root / "meta/tasks.jsonl").write_text(
        json.dumps({"task_index": 0, "task": "pick tube"}) + "\n",
        encoding="utf-8",
    )

    tactile_colors = ((255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0))
    episodes = []
    for episode_index in range(2):
        episodes.append({"episode_index": episode_index, "length": 3, "tasks": [0]})
        columns = {
            "observation.state": [
                np.full(20, row, dtype=np.float32) for row in range(3)
            ],
            "actions": [
                np.full(20, 1.0, dtype=np.float32),
                np.full(20, 2.0, dtype=np.float32),
                np.zeros(20, dtype=np.float32),
            ],
            "frame_index": [0, 1, 2],
            "episode_index": [episode_index] * 3,
            "task_index": [0] * 3,
        }
        for name in CAMERA_NAMES:
            columns[name] = [{"bytes": _jpeg_bytes((16, 32, 64))}] * 3
        for name, color in zip(TACTILE_NAMES, tactile_colors):
            columns[name] = [{"bytes": _jpeg_bytes(color)}] * 3
        path = root / "data/chunk-000" / f"episode_{episode_index:06d}.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(pa.table(columns), path)
    (root / "meta/episodes.jsonl").write_text(
        "".join(json.dumps(episode) + "\n" for episode in episodes),
        encoding="utf-8",
    )


def _build_tiny_stage2(config: dict) -> nn.Module:
    model = build_stage2_model(
        config,
        load_backbone=False,
        tactile_encoder=_TinyTactileEncoder(),
    )
    model.img_encoder = _TinyImageEncoder()
    return model


def _resolved_encoder_fixture(root: Path, model: nn.Module) -> ResolvedTactileEncoder:
    weights_path = root / "encoder.safetensors"
    save_file(model.tactile_encoder.state_dict(), str(weights_path))
    weights_sha256 = hashlib.sha256(weights_path.read_bytes()).hexdigest()
    metadata_path = root / "encoder.json"
    metadata_path.write_text(
        json.dumps({"weights_sha256": weights_sha256}), encoding="utf-8"
    )
    return ResolvedTactileEncoder(
        weights_path=weights_path,
        metadata_path=metadata_path,
        source_sha256="1" * 64,
        architecture="resnet18",
        embedding_dim=512,
    )


def test_stage2_dataset_to_export_contract(tmp_path: Path, monkeypatch) -> None:
    dataset_root = tmp_path / "lerobot-v21"
    _write_lerobot_fixture(dataset_root)
    train_dataset, _ = build_lerobot_vision_datasets(
        dataset_root,
        action_chunk_size=2,
        validation_ratio=0.5,
        split_seed=0,
        include_tactile=True,
    )
    sample = train_dataset[0]

    assert train_dataset.metadata["tactile_names"] == list(TACTILE_NAMES)
    assert sample["images"].shape == (2, 3, 3, 5)
    assert sample["tactile_images"].shape == (4, 3, 3, 5)
    assert torch.equal(
        sample["tactile_images"].mean(dim=(2, 3)).argmax(dim=1),
        torch.tensor([0, 1, 2, 0]),
    )

    visual = letterbox_and_normalize(sample["images"].unsqueeze(0), 32)
    tactile = letterbox_tactile_images(
        sample["tactile_images"].unsqueeze(0), (224, 224)
    )
    assert visual.shape == (1, 2, 3, 32, 32)
    assert tactile.shape == (1, 4, 3, 224, 224)
    assert torch.count_nonzero(tactile[:, :, :, :44]) == 0
    assert tactile.min() >= 0 and tactile.max() <= 1

    config = _config(train_dataset.manifest["dataset_id"])
    stage1 = build_model(config, load_backbone=False)
    stage1.img_encoder = _TinyImageEncoder()
    # A Stage2 source is a trained Stage1 checkpoint.  The upstream constructor
    # deliberately zero-initializes the final head, which would block gradients
    # to every newly added module in this small fixture.
    with torch.no_grad():
        stage1.linear.weight.fill_(0.1)
        stage1.linear.bias.fill_(0.01)
    stage1_checkpoint = tmp_path / "deco_stage1_latest.pt"
    torch.save({"model": stage1.state_dict(), "config": config}, stage1_checkpoint)

    model = _build_tiny_stage2(config)
    initialization = initialize_stage2_from_stage1(model, stage1_checkpoint)
    report = initialization.parameters
    trainable_names = {
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    }
    reported_trainable = {
        name for names in report.trainable_by_category.values() for name in names
    }
    assert trainable_names == reported_trainable
    assert set(report.trainable_by_category) == {
        "sensor_embeddings",
        "tactile_attention",
        "tactile_gates",
        "pi_adapters",
    }
    assert all(
        not parameter.requires_grad
        for name, parameter in model.named_parameters()
        if name.startswith("tactile_encoder.") or name in stage1.state_dict()
    )

    with torch.no_grad():
        model.sensor_embeddings.weight.zero_()
        for block in model.mmattn:
            block.tactile_gate.fill_(0.25)
    frozen_before = {
        name: parameter.detach().clone()
        for name, parameter in model.named_parameters()
        if not parameter.requires_grad
    }
    trainable_before = {
        name: parameter.detach().clone()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    seen_tokens: list[torch.Tensor] = []
    hook = model.mmattn[0].tactile_key.register_forward_pre_hook(
        lambda _module, args: seen_tokens.append(args[0].detach().clone())
    )
    optimizer = torch.optim.AdamW(
        stage2_optimizer_parameter_groups(model, learning_rate=1e-3, weight_decay=1e-6),
        betas=(0.95, 0.999),
    )
    scheduler = constant_lr_scheduler(optimizer)
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    try:
        torch.manual_seed(23)
        prediction, noise = model(
            visual[:, 0],
            visual[:, 1],
            obs=sample["observation"].unsqueeze(0),
            act=sample["action"].unsqueeze(0),
            training=True,
            tactile_images=tactile,
        )
        target = noise - sample["action"].unsqueeze(0)
        valid = (~sample["is_pad"]).view(1, 2, 1).expand_as(prediction)
        loss = (prediction - target).square()[valid].mean()
        loss.backward()
        optimizer.step()
        scheduler.step()
    finally:
        hook.remove()

    assert prediction.shape == (1, 2, 20)
    assert seen_tokens[0].shape == (1, 4, 512)
    assert torch.isfinite(loss)
    assert all(
        parameter.grad is None
        for parameter in model.parameters()
        if not parameter.requires_grad
    )
    assert all(
        torch.equal(model.get_parameter(name), value)
        for name, value in frozen_before.items()
    )
    assert any(
        not torch.equal(model.get_parameter(name), value)
        for name, value in trainable_before.items()
    )

    artifact = _resolved_encoder_fixture(tmp_path, model)
    metadata = build_stage2_checkpoint_metadata(
        model,
        report,
        stage1_checkpoint=stage1_checkpoint,
        tactile_artifact=artifact,
        tactile_adapter_rank=4,
    )
    metadata["gate_values"] = stage2_gradient_diagnostics(model, report)["gate_values"]
    payload = {
        "model": model.state_dict(),
        "config": config,
        "stats": jsonable_stats(train_dataset.stats),
        "epoch": 1,
        "val_loss": 0.25,
        "best_val": 0.25,
        "patience_best_val": 0.25,
        "stale_epochs": 0,
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(),
        "rng_states": [capture_rng_state()],
        "run_id": "integration",
        "global_step": 1,
        "checkpoint_schema_version": STAGE2_CHECKPOINT_SCHEMA_VERSION,
        "stage": 2,
        "model_type": STAGE2_MODEL_TYPE,
        "stage2_metadata": metadata,
    }
    checkpoint_path = tmp_path / "deco_stage2_latest.pt"
    atomic_torch_save(payload, checkpoint_path)

    resumed = _build_tiny_stage2(config)
    resumed_report = initialize_stage2_from_stage1(resumed, stage1_checkpoint).parameters
    resumed_optimizer = torch.optim.AdamW(
        stage2_optimizer_parameter_groups(
            resumed, learning_rate=1e-3, weight_decay=1e-6
        ),
        betas=(0.95, 0.999),
    )
    resumed_scheduler = constant_lr_scheduler(resumed_optimizer)
    resumed_scaler = torch.amp.GradScaler("cuda", enabled=False)
    restored = restore_stage2_training_state(
        load_checkpoint(checkpoint_path, "cpu"),
        model=resumed,
        optimizer=resumed_optimizer,
        scheduler=resumed_scheduler,
        scaler=resumed_scaler,
        current_config=config,
        world_size=1,
        rank=0,
    )
    assert restored["epoch"] == 1 and restored["global_step"] == 1
    assert {
        name for names in resumed_report.trainable_by_category.values() for name in names
    } == trainable_names
    assert all(
        torch.equal(resumed.state_dict()[name], value)
        for name, value in model.state_dict().items()
    )

    monkeypatch.setattr(
        export_module,
        "build_stage2_model",
        lambda export_config, load_backbone=False: _build_tiny_stage2(export_config),
    )
    output_path = tmp_path / "deco_stage2_latest.ts"
    exported_metadata = export_checkpoint(checkpoint_path, output_path, 3, 5)
    exported = torch.jit.load(str(output_path), map_location="cpu").eval()
    with torch.inference_mode():
        torch.manual_seed(31)
        action = exported(
            sample["images"].unsqueeze(0),
            sample["tactile_images"].unsqueeze(0),
            torch.ones(1, 20),
        )

    assert exported_metadata["format"] == STAGE2_EXPORT_FORMAT
    assert exported_metadata["input"]["stream_order"] == [
        *CAMERA_NAMES,
        *TACTILE_NAMES,
    ]
    assert action.shape == (1, 2, 20)
    assert torch.isfinite(action).all()
