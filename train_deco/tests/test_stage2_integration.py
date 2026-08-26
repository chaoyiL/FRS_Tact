from __future__ import annotations

import gc
import hashlib
import io
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from PIL import Image
from safetensors.torch import save_file
from torch import nn

import train_deco.export_torchscript as export_module
from train_deco.checkpoint import (
    atomic_torch_save,
    capture_rng_state,
    load_checkpoint,
    restore_rng_state,
)
from train_deco.export_torchscript import STAGE2_EXPORT_FORMAT, export_checkpoint
from train_deco.input_adapter import letterbox_and_normalize, letterbox_tactile_images
from train_deco.lerobot_vision_dataset import (
    CAMERA_NAMES,
    TACTILE_NAMES,
    build_lerobot_vision_datasets,
)
from train_deco.model_factory import (
    MODEL_TYPE,
    STAGE2_MODEL_TYPE,
    build_model,
    build_stage2_model,
)
from train_deco.stage2_initialization import (
    configure_stage2_trainability,
    initialize_stage2_from_stage1,
)
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
        self.scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        code = images.mean(dim=(1, 2, 3), keepdim=False).unsqueeze(1)
        identity = torch.ones_like(code)
        padding = torch.zeros(
            code.shape[0], 510, dtype=code.dtype, device=code.device
        )
        return torch.cat((code, identity, padding), dim=1) * self.scale


def _config(dataset_id: str) -> dict:
    return {
        "model_type": STAGE2_MODEL_TYPE,
        "stage": 2,
        "source_obs_dim": 20,
        "obs_dim": 20,
        "action_dim": 20,
        "chunk_size": 32,
        "observation_indices": list(range(20)),
        "hidden_dim": 512,
        "layers": 6,
        "heads": 8,
        "image_size": 256,
        "inference_steps": 5,
        "rope_height": 256,
        "rope_width": 256,
        "use_task_condition": False,
        "num_tasks": 1,
        "dataset_id": dataset_id,
        "action_mode": "tcp_delta_absolute_gripper",
        "camera_names": list(CAMERA_NAMES),
        "tactile_field_order": list(TACTILE_NAMES),
        "tactile_adapter_rank": 32,
        "training_state_version": 3,
        "world_size": 1,
        "batch_size": 1,
        "seed": 17,
        "train_samples": 2,
        "steps_per_epoch": 1,
        "objective_version": "masked-flow-mse-v1",
        "validation_metric_version": "no-repeat-masked-element-mean-v1",
        "validation_seed": 12345,
        "lr": 1e-4,
        "lr_final": 1e-4,
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
    physical_tactile_order = tuple(reversed(TACTILE_NAMES))
    features = {
        "observation.state": {"dtype": "float32", "shape": [20]},
        "actions": {"dtype": "float32", "shape": [20]},
        **{
            name: {"dtype": "image", "shape": [3, 5, 3]}
            for name in (*CAMERA_NAMES, *physical_tactile_order)
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

    tactile_colors = {
        name: (level, level, level)
        for name, level in zip(TACTILE_NAMES, (32, 64, 128, 192))
    }
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
        for name in physical_tactile_order:
            columns[name] = [{"bytes": _jpeg_bytes(tactile_colors[name])}] * 3
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


def _category_parameters(model: nn.Module, report) -> dict[str, list[nn.Parameter]]:
    named = dict(model.named_parameters())
    return {
        category: [named[name] for name in names]
        for category, names in report.trainable_by_category.items()
    }


def _snapshot_parameters(
    model: nn.Module, names: tuple[str, ...] | set[str]
) -> dict[str, torch.Tensor]:
    named = dict(model.named_parameters())
    return {name: named[name].detach().clone() for name in names}


def _has_finite_nonzero_gradient(parameters: list[nn.Parameter]) -> bool:
    return any(
        parameter.grad is not None
        and torch.isfinite(parameter.grad).all()
        and torch.count_nonzero(parameter.grad) > 0
        for parameter in parameters
    )


def _all_gradients_absent_or_zero(parameters: list[nn.Parameter]) -> bool:
    return all(
        parameter.grad is None or torch.count_nonzero(parameter.grad) == 0
        for parameter in parameters
    )


def _backward_stage2_step(
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    visual: torch.Tensor,
    tactile: torch.Tensor,
    sample: dict[str, torch.Tensor],
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    optimizer.zero_grad(set_to_none=True)
    torch.manual_seed(seed)
    prediction, noise = model(
        visual[:, 0],
        visual[:, 1],
        obs=sample["observation"].unsqueeze(0),
        act=sample["action"].unsqueeze(0),
        training=True,
        tactile_images=tactile,
    )
    target = noise - sample["action"].unsqueeze(0)
    valid = (~sample["is_pad"]).view(1, 32, 1).expand_as(prediction)
    loss = (prediction - target).square()[valid].mean()
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    return prediction, loss


def _finish_stage2_step(
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: torch.amp.GradScaler,
) -> None:
    scaler.step(optimizer)
    scaler.update()
    scheduler.step()


def _assert_nested_equal(actual: Any, expected: Any) -> None:
    if isinstance(expected, torch.Tensor):
        assert isinstance(actual, torch.Tensor)
        assert torch.equal(actual, expected)
    elif isinstance(expected, dict):
        assert isinstance(actual, dict)
        assert actual.keys() == expected.keys()
        for key in expected:
            _assert_nested_equal(actual[key], expected[key])
    elif isinstance(expected, (list, tuple)):
        assert isinstance(actual, type(expected))
        assert len(actual) == len(expected)
        for actual_value, expected_value in zip(actual, expected):
            _assert_nested_equal(actual_value, expected_value)
    else:
        assert actual == expected


def _next_rng_values() -> tuple[float, float, float]:
    return random.random(), float(np.random.random()), float(torch.rand(()))


def test_stage2_dataset_to_export_contract(tmp_path: Path, monkeypatch) -> None:
    dataset_root = tmp_path / "lerobot-v21"
    _write_lerobot_fixture(dataset_root)
    train_dataset, _ = build_lerobot_vision_datasets(
        dataset_root,
        action_chunk_size=32,
        validation_ratio=0.5,
        split_seed=0,
        include_tactile=True,
    )
    sample = train_dataset[0]

    assert train_dataset.metadata["tactile_names"] == list(TACTILE_NAMES)
    assert sample["images"].shape == (2, 3, 3, 5)
    assert sample["tactile_images"].shape == (4, 3, 3, 5)
    assert sample["action"].shape == (32, 20)
    dataset_codes = sample["tactile_images"].mean(dim=(1, 2, 3))
    torch.testing.assert_close(
        dataset_codes,
        torch.tensor([32, 64, 128, 192], dtype=torch.float32) / 255,
        rtol=0,
        atol=1 / 255,
    )
    assert torch.all(dataset_codes[1:] > dataset_codes[:-1])

    visual = letterbox_and_normalize(sample["images"].unsqueeze(0), 256)
    tactile = letterbox_tactile_images(
        sample["tactile_images"].unsqueeze(0), (224, 224)
    )
    assert visual.shape == (1, 2, 3, 256, 256)
    assert tactile.shape == (1, 4, 3, 224, 224)
    assert torch.count_nonzero(tactile[:, :, :, :45]) == 0
    assert tactile.min() >= 0 and tactile.max() <= 1
    preprocessed_codes = tactile.mean(dim=(2, 3, 4)).squeeze(0)
    torch.testing.assert_close(
        preprocessed_codes,
        dataset_codes * (134 / 224),
        rtol=0,
        atol=1e-5,
    )
    assert torch.all(preprocessed_codes[1:] > preprocessed_codes[:-1])

    config = _config(train_dataset.manifest["dataset_id"])
    stage1_config = dict(config)
    stage1_config.update({"model_type": MODEL_TYPE, "stage": 1})
    stage1_config.pop("tactile_field_order")
    stage1_config.pop("tactile_adapter_rank")
    stage1 = build_model(stage1_config, load_backbone=False)
    stage1.img_encoder = _TinyImageEncoder()
    # A Stage2 source is a trained Stage1 checkpoint.  The upstream constructor
    # deliberately zero-initializes the final head, which would block gradients
    # to every newly added module in this small fixture.
    with torch.no_grad():
        stage1.linear.weight.fill_(0.1)
        stage1.linear.bias.fill_(0.01)
    stage1_checkpoint = tmp_path / "deco_stage1_latest.pt"
    torch.save(
        {"model": stage1.state_dict(), "config": stage1_config}, stage1_checkpoint
    )

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
    stage1_parameter_names = set(dict(stage1.named_parameters()))
    assert all(
        not parameter.requires_grad
        for name, parameter in model.named_parameters()
        if name.startswith("tactile_encoder.") or name in stage1_parameter_names
    )
    del stage1
    gc.collect()

    with torch.no_grad():
        model.sensor_embeddings.weight.zero_()
    assert all(
        torch.count_nonzero(block.tactile_gate) == 0 for block in model.mmattn
    )
    frozen_names = {
        name for names in report.frozen_by_category.values() for name in names
    }
    frozen_before = _snapshot_parameters(model, frozen_names)
    category_parameters = _category_parameters(model, report)
    zero_gate_before = {
        category: _snapshot_parameters(model, set(names))
        for category, names in report.trainable_by_category.items()
    }
    encoder_inputs: list[torch.Tensor] = []
    encoder_outputs: list[torch.Tensor] = []
    seen_tokens: list[torch.Tensor] = []
    input_hook = model.tactile_encoder.register_forward_pre_hook(
        lambda _module, args: encoder_inputs.append(args[0].detach().clone())
    )
    output_hook = model.tactile_encoder.register_forward_hook(
        lambda _module, _args, output: encoder_outputs.append(output.detach().clone())
    )
    token_hook = model.mmattn[0].tactile_key.register_forward_pre_hook(
        lambda _module, args: seen_tokens.append(args[0].detach().clone())
    )
    optimizer = torch.optim.AdamW(
        stage2_optimizer_parameter_groups(
            model, learning_rate=config["lr"], weight_decay=config["weight_decay"]
        ),
        betas=(0.95, 0.999),
    )
    scheduler = constant_lr_scheduler(optimizer)
    scaler = torch.amp.GradScaler("cpu", enabled=True, init_scale=128.0)
    try:
        prediction, loss = _backward_stage2_step(
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            visual=visual,
            tactile=tactile,
            sample=sample,
            seed=23,
        )
        assert _has_finite_nonzero_gradient(category_parameters["tactile_gates"])
        assert _has_finite_nonzero_gradient(category_parameters["pi_adapters"])
        assert _all_gradients_absent_or_zero(category_parameters["sensor_embeddings"])
        assert _all_gradients_absent_or_zero(category_parameters["tactile_attention"])
        _finish_stage2_step(optimizer, scheduler, scaler)

        for category in ("tactile_gates", "pi_adapters"):
            assert any(
                not torch.equal(model.get_parameter(name), value)
                for name, value in zero_gate_before[category].items()
            )
        for category in ("sensor_embeddings", "tactile_attention"):
            assert all(
                torch.equal(model.get_parameter(name), value)
                for name, value in zero_gate_before[category].items()
            )

        with torch.no_grad():
            for block in model.mmattn:
                block.tactile_gate.fill_(0.25)
        nonzero_gate_before = {
            category: _snapshot_parameters(model, set(names))
            for category, names in report.trainable_by_category.items()
        }
        _backward_stage2_step(
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            visual=visual,
            tactile=tactile,
            sample=sample,
            seed=29,
        )
        for category, parameters in category_parameters.items():
            assert _has_finite_nonzero_gradient(parameters), category
        _finish_stage2_step(optimizer, scheduler, scaler)
        for category, before in nonzero_gate_before.items():
            assert any(
                not torch.equal(model.get_parameter(name), value)
                for name, value in before.items()
            ), category
    finally:
        input_hook.remove()
        output_hook.remove()
        token_hook.remove()

    assert prediction.shape == (1, 32, 20)
    assert encoder_inputs[0].shape == (4, 3, 224, 224)
    torch.testing.assert_close(
        encoder_inputs[0].mean(dim=(1, 2, 3)), preprocessed_codes
    )
    assert encoder_outputs[0].shape == (4, 512)
    torch.testing.assert_close(encoder_outputs[0][:, 0], preprocessed_codes)
    torch.testing.assert_close(
        encoder_outputs[0][:, 1], torch.ones_like(preprocessed_codes)
    )
    assert seen_tokens[0].shape == (1, 4, 512)
    assert torch.all(seen_tokens[0][0, 1:, 0] > seen_tokens[0][0, :-1, 0])
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
    assert scheduler.last_epoch == 2
    del frozen_before
    gc.collect()

    artifact = _resolved_encoder_fixture(tmp_path, model)
    metadata = build_stage2_checkpoint_metadata(
        model,
        report,
        stage1_checkpoint=stage1_checkpoint,
        tactile_artifact=artifact,
        tactile_adapter_rank=32,
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
        "global_step": 2,
        "checkpoint_schema_version": STAGE2_CHECKPOINT_SCHEMA_VERSION,
        "stage": 2,
        "model_type": STAGE2_MODEL_TYPE,
        "stage2_metadata": metadata,
    }
    checkpoint_path = tmp_path / "deco_stage2_latest.pt"
    atomic_torch_save(payload, checkpoint_path)

    loaded_checkpoint = load_checkpoint(checkpoint_path, "cpu")
    restore_rng_state(loaded_checkpoint["rng_states"][0])
    expected_next_rng = _next_rng_values()
    random.seed(999)
    np.random.seed(999)
    torch.manual_seed(999)
    stage1_checkpoint.unlink()
    artifact.weights_path.unlink()
    artifact.metadata_path.unlink()
    assert not stage1_checkpoint.exists()
    assert not artifact.weights_path.exists()
    assert not artifact.metadata_path.exists()
    del payload, model, optimizer, scheduler, scaler
    gc.collect()

    resumed = _build_tiny_stage2(config)
    resumed_report = configure_stage2_trainability(resumed)
    resumed_optimizer = torch.optim.AdamW(
        stage2_optimizer_parameter_groups(
            resumed,
            learning_rate=config["lr"],
            weight_decay=config["weight_decay"],
        ),
        betas=(0.95, 0.999),
    )
    resumed_scheduler = constant_lr_scheduler(resumed_optimizer)
    resumed_scaler = torch.amp.GradScaler("cpu", enabled=True, init_scale=2.0)
    restored = restore_stage2_training_state(
        loaded_checkpoint,
        model=resumed,
        optimizer=resumed_optimizer,
        scheduler=resumed_scheduler,
        scaler=resumed_scaler,
        current_config=config,
        world_size=1,
        rank=0,
    )
    assert restored["epoch"] == 1 and restored["global_step"] == 2
    assert {
        name for names in resumed_report.trainable_by_category.values() for name in names
    } == trainable_names
    _assert_nested_equal(resumed.state_dict(), loaded_checkpoint["model"])
    _assert_nested_equal(resumed_optimizer.state_dict(), loaded_checkpoint["optimizer"])
    _assert_nested_equal(resumed_scheduler.state_dict(), loaded_checkpoint["scheduler"])
    _assert_nested_equal(resumed_scaler.state_dict(), loaded_checkpoint["scaler"])
    assert restored["stats"] == loaded_checkpoint["stats"]
    assert restored["stage2_metadata"] == loaded_checkpoint["stage2_metadata"]
    assert restored["config"] == loaded_checkpoint["config"]
    assert _next_rng_values() == expected_next_rng

    resumed_frozen_before = _snapshot_parameters(resumed, frozen_names)
    previous_optimizer_step = max(
        int(state["step"]) for state in resumed_optimizer.state_dict()["state"].values()
    )
    _backward_stage2_step(
        model=resumed,
        optimizer=resumed_optimizer,
        scaler=resumed_scaler,
        visual=visual,
        tactile=tactile,
        sample=sample,
        seed=37,
    )
    _finish_stage2_step(resumed_optimizer, resumed_scheduler, resumed_scaler)
    continued_global_step = restored["global_step"] + 1
    assert previous_optimizer_step == 2
    assert continued_global_step == 3
    assert resumed_scheduler.last_epoch == continued_global_step
    assert max(
        int(state["step"]) for state in resumed_optimizer.state_dict()["state"].values()
    ) == continued_global_step
    assert all(
        torch.equal(resumed.get_parameter(name), value)
        for name, value in resumed_frozen_before.items()
    )
    del loaded_checkpoint, resumed, resumed_optimizer, resumed_scheduler
    del resumed_scaler, resumed_frozen_before
    gc.collect()

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
    assert action.shape == (1, 32, 20)
    assert torch.isfinite(action).all()


def test_stage2_integration_fixture_uses_real_stage1_architecture() -> None:
    fixture_config = _config("fixture")

    assert {
        key: fixture_config[key]
        for key in (
            "action_dim",
            "chunk_size",
            "hidden_dim",
            "layers",
            "heads",
            "image_size",
            "inference_steps",
            "rope_height",
            "rope_width",
            "camera_names",
        )
    } == {
        "action_dim": 20,
        "chunk_size": 32,
        "hidden_dim": 512,
        "layers": 6,
        "heads": 8,
        "image_size": 256,
        "inference_steps": 5,
        "rope_height": 256,
        "rope_width": 256,
        "camera_names": list(CAMERA_NAMES),
    }


def test_stage2_tiny_encoder_preserves_four_sensor_codes() -> None:
    codes = torch.tensor([0.125, 0.25, 0.5, 0.75])
    images = codes.view(4, 1, 1, 1).expand(4, 3, 8, 8)

    output = _TinyTactileEncoder()(images)

    torch.testing.assert_close(output[:, 0], codes)
    torch.testing.assert_close(output[:, 1], torch.ones_like(codes))


def test_stage2_readme_describes_dynamic_tactile_shape_and_rank_loading() -> None:
    readme = Path("train_deco/README.md").read_text(encoding="utf-8")

    for contract in (
        "`[H, W, 3]`",
        "当前实际数据是 `[224, 224, 3]`",
        "`[4, 3, H, W]`",
        "`[B, 4, 3, H, W]`",
        "固定字段名",
        "不依赖 Parquet 的物理列顺序",
        "每个 rank 都 strict-load",
        "rank 0 校验转换 metadata 和 SHA256",
    ):
        assert contract in readme
