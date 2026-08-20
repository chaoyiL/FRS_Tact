from __future__ import annotations

import json
import os
from pathlib import Path
import re
import subprocess
import sys
from typing import Any
import zipfile

import numpy as np
import pytest
import yaml

from train_pi05_frs.utils.bimanual_schema import (
    BIMANUAL_LOSS_MODE,
    bimanual_objective_metadata,
    validate_bimanual_objective_metadata,
)
from train_pi05_frs.tools.train_frs import load_config, validate_config


REPO_ROOT = Path(__file__).resolve().parents[2]
TRAIN_ROOT = REPO_ROOT / "train_pi05_frs"
CONFIG_PATH = TRAIN_ROOT / "configs" / "train_pi05_frs.yaml"
LAUNCHER = TRAIN_ROOT / "scripts" / "start_frs_pi05_train.sh"


def _valid_config(tmp_path: Path) -> dict[str, Any]:
    checkpoint = tmp_path / "checkpoint"
    (checkpoint / "params").mkdir(parents=True)
    encoder = tmp_path / "encoder"
    encoder.mkdir()
    np.savez(encoder / "params-fake.npz", p00000=np.zeros((1,), dtype=np.float32))
    (encoder / "checkpoint.json").write_text(
        json.dumps(
            {
                "params_file": "params-fake.npz",
                "parameter_paths": ["tactile_resnet/example"],
                "tactile_clip_config": {
                    "embedding_dim": 512,
                    "tactile_image_size": 224,
                },
            }
        ),
        encoding="utf-8",
    )
    dataset = tmp_path / "dataset"
    (dataset / "meta").mkdir(parents=True)
    (dataset / "meta" / "info.json").write_text(
        json.dumps(
            {
                "codebase_version": "v3.0",
                "fps": 30,
                "total_episodes": 1,
                "total_frames": 2,
                "total_tasks": 1,
                "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
                "video_path": (
                    "videos/{video_key}/chunk-{chunk_index:03d}/"
                    "file-{file_index:03d}.mp4"
                ),
                "features": {
                    "timestamp": {"dtype": "float32", "shape": [1], "names": None},
                    "frame_index": {"dtype": "int64", "shape": [1], "names": None},
                    "episode_index": {"dtype": "int64", "shape": [1], "names": None},
                    "index": {"dtype": "int64", "shape": [1], "names": None},
                    "task_index": {"dtype": "int64", "shape": [1], "names": None},
                    "actions": {"dtype": "float32", "shape": [2]},
                    "observation.state": {"dtype": "float32", "shape": [3]},
                    "observation.images.camera0": {"dtype": "video", "shape": [3, 224, 224]},
                    "observation.images.camera1": {"dtype": "video", "shape": [3, 224, 224]},
                    "observation.images.tactile_left_0": {"dtype": "video", "shape": [3, 224, 224]},
                    "observation.images.tactile_right_0": {"dtype": "video", "shape": [3, 224, 224]},
                    "observation.images.tactile_left_1": {"dtype": "video", "shape": [3, 224, 224]},
                    "observation.images.tactile_right_1": {"dtype": "video", "shape": [3, 224, 224]},
                },
            }
        ),
        encoding="utf-8",
    )
    (dataset / "meta" / "stats.json").write_text(
        json.dumps(
            {
                "observation.state": {
                    "min": [0.0, 0.0, 0.0],
                    "max": [1.0, 1.0, 1.0],
                    "mean": [0.5, 0.5, 0.5],
                    "std": [1.0, 1.0, 1.0],
                    "count": [2],
                },
                "actions": {
                    "min": [0.0, 0.0],
                    "max": [1.0, 1.0],
                    "mean": [0.5, 0.5],
                    "std": [1.0, 1.0],
                    "count": [2],
                },
            }
        ),
        encoding="utf-8",
    )
    import pyarrow as pa
    import pyarrow.parquet as pq

    pq.write_table(
        pa.table({"task": ["demo"], "task_index": [0]}),
        dataset / "meta" / "tasks.parquet",
    )
    episodes = dataset / "meta" / "episodes" / "chunk-000"
    episodes.mkdir(parents=True)
    visual_keys = [
        "observation.images.camera0",
        "observation.images.camera1",
        "observation.images.tactile_left_0",
        "observation.images.tactile_right_0",
        "observation.images.tactile_left_1",
        "observation.images.tactile_right_1",
    ]
    episode_columns: dict[str, list[int | float]] = {
        "episode_index": [0],
        "dataset_from_index": [0],
        "dataset_to_index": [2],
        "data/chunk_index": [0],
        "data/file_index": [0],
    }
    for visual_key in visual_keys:
        episode_columns[f"videos/{visual_key}/chunk_index"] = [0]
        episode_columns[f"videos/{visual_key}/file_index"] = [0]
        episode_columns[f"videos/{visual_key}/from_timestamp"] = [0.0]
        episode_columns[f"videos/{visual_key}/to_timestamp"] = [1.0 / 30.0]
    pq.write_table(pa.table(episode_columns), episodes / "file-000.parquet")
    data_file = dataset / "data" / "chunk-000" / "file-000.parquet"
    data_file.parent.mkdir(parents=True)
    pq.write_table(
        pa.table(
            {
                "episode_index": [0, 0],
                "frame_index": [0, 1],
                "index": [0, 1],
                "timestamp": [0.0, 1.0 / 30.0],
                "task_index": [0, 0],
                "observation.state": [[0.0, 0.0, 0.0], [0.1, 0.1, 0.1]],
                "actions": [[0.0, 0.0], [0.1, 0.1]],
            }
        ),
        data_file,
    )
    for visual_key in visual_keys:
        video_file = (
            dataset
            / "videos"
            / visual_key
            / "chunk-000"
            / "file-000.mp4"
        )
        video_file.parent.mkdir(parents=True)
        video_file.write_bytes(b"synthetic-video-asset")
    norm_stats = tmp_path / "norm-stats" / "robot"
    norm_stats.mkdir(parents=True)
    (norm_stats / "norm_stats.json").write_text(
        json.dumps(
            {
                "norm_stats": {
                    "state": {
                        "mean": [0.0, 0.0, 0.0],
                        "std": [1.0, 1.0, 1.0],
                        "q01": [-1.0, -1.0, -1.0],
                        "q99": [1.0, 1.0, 1.0],
                    },
                    "actions": {
                        "mean": [0.0, 0.0],
                        "std": [1.0, 1.0],
                        "q01": [-1.0, -1.0],
                        "q99": [1.0, 1.0],
                    },
                }
            }
        ),
        encoding="utf-8",
    )
    return {
        "checkpoint": str(checkpoint),
        "allow_download": False,
        "datasets": [
            {
                "repo_id": "org/demo",
                "root": str(dataset),
                "action_key": "actions",
                "rename_map": {
                    "observation.images.camera0": "observation.images.camera1",
                    "observation.images.camera1": "observation.images.camera2",
                },
            }
        ],
        "action_cache": {
            "root": str(tmp_path / "action-cache"),
            "model_sample_steps": 2,
            "reverse_steps": 3,
            "reverse_solver": "slerpflow",
            "batch_size": 2,
            "load_workers": 1,
            "flush_every": 1,
            "inference_seed": 0,
            "split_seed": 1,
            "val_fraction": 0.1,
            "frame_stride": 1,
            "drop_tail_action_chunks": 0,
            "max_episodes": None,
            "max_samples": None,
        },
        "tactile_embedding_cache": {
            "enabled": True,
            "root": str(tmp_path / "tactile-cache"),
            "dtype": "float16",
            "precompute_batch_size": 2,
            "precompute_num_workers": 0,
            "precompute_prefetch_factor": 2,
            "precompute_video_backend": "torchcodec",
            "precompute_flush_every": 1,
        },
        "model": {
            "use_tactile_encoder": True,
            "tactile_encoder_path": str(encoder),
            "freeze_tactile_encoder": True,
            "tactile_keys": [
                "observation.images.tactile_left_0",
                "observation.images.tactile_right_0",
                "observation.images.tactile_left_1",
                "observation.images.tactile_right_1",
            ],
            "tactile_embedding_dim": 512,
            "tactile_num_tokens": 4,
            "tactile_image_size": 224,
            "state_conditioning": True,
            "state_dropout_rate": 0.1,
            "camera_map": {
                "left_wrist_0_rgb": "observation.images.camera1",
                "right_wrist_0_rgb": "observation.images.camera2",
            },
            "action_dim": 32,
            "action_horizon": 50,
            "paligemma_variant": "gemma_2b",
            "action_expert_variant": "gemma_300m",
        },
        "norm_stats": {
            "dir": str(tmp_path / "norm-stats"),
            "asset_id": "robot",
            "use_quantile_norm": True,
        },
        "frs_training": {
            "output": str(tmp_path / "training-output"),
            "tactile_window_divisor": 5,
            "history_stride": 3,
            "loss_mode": "gated",
            "gate_tau": 0.4,
            "gate_temperature": 0.1,
            "gate_lambda": 0.25,
            "aux_decode_weight": 4.0,
            "aux_decode_steps": 10,
            "aux_decode_solver": "fireflow",
            "low_gate_safety_weight": 0.5,
            "low_gate_safety_margin": 0.03,
            "rank_low_gate_threshold": 0.3,
            "rank_high_gate_threshold": 0.7,
            "rank_weight": 2.0,
            "rank_margin": 0.0,
            "repair_weight": 2.0,
            "repair_margin": 0.0,
            "best_max_low_gate_unsafe_frac": 0.1,
            "best_min_high_gate_gain": 0.0,
            "best_min_high_gate_rank_satisfied_frac": 0.8,
            "model_dim": 256,
            "depth": 6,
            "num_heads": 4,
            "mlp_ratio": 4,
            "learning_rate": 3.0e-4,
            "weight_decay": 1.0e-4,
            "grad_clip_norm": 1.0,
            "warmup_epochs": 2,
            "lr_reference_dim": 256,
            "min_lr_ratio": 0.1,
            "lr_schedule": "cosine",
            "batch_size": 2,
            "epochs": 1,
            "validation_steps": 1,
            "eval_every": 1,
            "seed": 42,
            "write_plots": True,
            "resume": False,
            "resume_from": None,
        },
    }


def _write_config(tmp_path: Path, config: dict[str, Any] | None = None) -> tuple[Path, dict[str, Any]]:
    value = config or _valid_config(tmp_path)
    path = tmp_path / "pipeline.yaml"
    path.write_text(yaml.safe_dump(value, sort_keys=False), encoding="utf-8")
    return path, value


def test_default_config_and_tools_live_inside_training_project() -> None:
    from train_pi05_frs.tools import prepare_frs_pi05_cache, train_frs

    assert CONFIG_PATH.is_file()
    assert prepare_frs_pi05_cache.DEFAULT_CONFIG == CONFIG_PATH
    assert train_frs.DEFAULT_CONFIG == CONFIG_PATH
    config = train_frs.load_config(CONFIG_PATH)
    assert config["checkpoint"] == "gs://openpi-assets/checkpoints/pi05_base"
    assert config["action_cache"]["reverse_solver"] == "slerpflow"
    assert config["frs_training"]["loss_mode"] == "gated"


def test_bimanual_config_is_independent_and_validated():
    path = TRAIN_ROOT / "configs" / "train_pi05_frs_bimanual_gated.yaml"
    config = load_config(path)
    assert config["frs_training"]["loss_mode"] == BIMANUAL_LOSS_MODE
    assert "gate_lambda" not in config["frs_training"]
    validate_config(config, check_paths=False)


def test_bimanual_metadata_supports_native_and_padded_action_widths():
    for action_dim in (20, 32):
        metadata = bimanual_objective_metadata(action_dim=action_dim)
        validate_bimanual_objective_metadata(metadata, action_dim=action_dim)
        assert metadata["steered_action_dim"] == 20
        assert metadata["padded_tail_policy"] == "vla_endpoint_masked"


def test_bimanual_metadata_rejects_wrong_tail_policy():
    metadata = bimanual_objective_metadata(action_dim=32)
    metadata["padded_tail_policy"] = "train_gt"
    with pytest.raises(ValueError, match="padded_tail_policy"):
        validate_bimanual_objective_metadata(metadata, action_dim=32)


def test_load_config_rejects_non_mapping_root(tmp_path: Path) -> None:
    from train_pi05_frs.tools.train_frs import load_config

    path = tmp_path / "invalid.yaml"
    path.write_text("- not\n- a\n- mapping\n", encoding="utf-8")
    with pytest.raises(ValueError, match="root.*mapping"):
        load_config(path)


@pytest.mark.parametrize(
    ("section", "key", "bad_value"),
    [
        (None, "allow_download", "false"),
        ("tactile_embedding_cache", "enabled", 1),
        ("model", "use_tactile_encoder", "true"),
        ("model", "freeze_tactile_encoder", 1),
        ("model", "state_conditioning", "yes"),
        ("norm_stats", "use_quantile_norm", 1),
        ("frs_training", "write_plots", "true"),
        ("frs_training", "resume", 0),
    ],
)
def test_schema_rejects_non_boolean_boolean_fields(
    tmp_path: Path, section: str | None, key: str, bad_value: object
) -> None:
    from train_pi05_frs.tools.train_frs import validate_config

    config = _valid_config(tmp_path)
    target = config if section is None else config[section]
    target[key] = bad_value
    with pytest.raises(ValueError, match=f"{key}.*boolean"):
        validate_config(config, check_paths=False)


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda cfg: cfg.update(datasets={"repo_id": "org/demo"}), "datasets.*list"),
        (lambda cfg: cfg.update(action_cache=[]), "action_cache.*mapping"),
        (lambda cfg: cfg["datasets"].append("org/not-a-mapping"), "datasets\\[1\\].*mapping"),
        (lambda cfg: cfg["model"].update(tactile_keys="one-key"), "tactile_keys.*list"),
        (lambda cfg: cfg["model"].update(camera_map=[]), "camera_map.*mapping"),
        (lambda cfg: cfg["datasets"][0].update(rename_map=[]), "rename_map.*mapping"),
    ],
)
def test_schema_rejects_wrong_mapping_and_list_types(
    tmp_path: Path, mutate: Any, message: str
) -> None:
    from train_pi05_frs.tools.train_frs import validate_config

    config = _valid_config(tmp_path)
    mutate(config)
    with pytest.raises(ValueError, match=message):
        validate_config(config, check_paths=False)


@pytest.mark.parametrize("key", ["episodes", "weight"])
def test_schema_rejects_dataset_controls_that_pipeline_would_ignore(
    tmp_path: Path, key: str
) -> None:
    from train_pi05_frs.tools.train_frs import validate_config

    config = _valid_config(tmp_path)
    config["datasets"][0][key] = [0] if key == "episodes" else 2.0

    with pytest.raises(
        ValueError,
        match=rf"config\.datasets\[0\]\.{key}.*not supported.*ignored",
    ):
        validate_config(config, check_paths=False)


@pytest.mark.parametrize(
    ("section", "unknown_key", "qualified_key"),
    [
        (None, "epohcs", "config.epohcs"),
        ("action_cache", "batch_szie", "config.action_cache.batch_szie"),
        (
            "tactile_embedding_cache",
            "precompute_batch_szie",
            "config.tactile_embedding_cache.precompute_batch_szie",
        ),
        ("model", "action_horizn", "config.model.action_horizn"),
        ("norm_stats", "asset", "config.norm_stats.asset"),
        ("frs_training", "epohcs", "config.frs_training.epohcs"),
        ("datasets.0", "repo", "config.datasets[0].repo"),
    ],
)
def test_schema_rejects_unknown_keys_with_qualified_diagnostic(
    tmp_path: Path, section: str | None, unknown_key: str, qualified_key: str
) -> None:
    from train_pi05_frs.tools.train_frs import validate_config

    config = _valid_config(tmp_path)
    if section is None:
        target = config
    elif section == "datasets.0":
        target = config["datasets"][0]
    else:
        target = config[section]
    target[unknown_key] = 1

    with pytest.raises(ValueError, match=rf"{re.escape(qualified_key)}.*unknown"):
        validate_config(config, check_paths=False)


@pytest.mark.parametrize(
    ("section", "key", "bad_value"),
    [
        ("action_cache", "inference_seed", "0"),
        ("action_cache", "split_seed", False),
        ("frs_training", "seed", "42"),
        ("frs_training", "learning_rate", "3e-4"),
        ("frs_training", "learning_rate", float("nan")),
        ("frs_training", "aux_decode_weight", float("inf")),
    ],
)
def test_schema_rejects_non_numeric_numeric_fields(
    tmp_path: Path, section: str, key: str, bad_value: object
) -> None:
    from train_pi05_frs.tools.train_frs import validate_config

    config = _valid_config(tmp_path)
    config[section][key] = bad_value
    with pytest.raises(ValueError, match=key):
        validate_config(config, check_paths=False)


@pytest.mark.parametrize(
    ("section", "key"),
    [
        ("action_cache", "root"),
        ("tactile_embedding_cache", "root"),
        ("frs_training", "output"),
    ],
)
def test_schema_rejects_url_output_paths(tmp_path: Path, section: str, key: str) -> None:
    from train_pi05_frs.tools.train_frs import validate_config

    config = _valid_config(tmp_path)
    config[section][key] = "gs://not-a-local-output"
    with pytest.raises(ValueError, match="local filesystem path"):
        validate_config(config, check_paths=False)


@pytest.mark.parametrize(
    ("section", "key", "protected"),
    [
        ("frs_training", "output", REPO_ROOT / "pyproject.toml"),
        ("action_cache", "root", REPO_ROOT / "uv.lock"),
        ("frs_training", "output", REPO_ROOT / "deploy_pi05" / "generated"),
        ("action_cache", "root", REPO_ROOT / "train_encoder" / "cache"),
        ("tactile_embedding_cache", "root", TRAIN_ROOT / "src" / "cache"),
        ("frs_training", "output", TRAIN_ROOT / "configs" / "generated"),
        ("action_cache", "root", TRAIN_ROOT / "tests" / "cache"),
    ],
)
def test_schema_rejects_protected_output_or_cache_descendants(
    tmp_path: Path, section: str, key: str, protected: Path
) -> None:
    from train_pi05_frs.tools.train_frs import validate_config

    config = _valid_config(tmp_path)
    config[section][key] = str(protected)

    with pytest.raises(ValueError, match=rf"config\.{re.escape(section)}\.{key}.*protected"):
        validate_config(config, check_paths=False)


def test_schema_rejects_symlink_alias_of_protected_source_tree(tmp_path: Path) -> None:
    from train_pi05_frs.tools.train_frs import validate_config

    config = _valid_config(tmp_path)
    alias = tmp_path / "source-alias"
    alias.symlink_to(TRAIN_ROOT / "utils", target_is_directory=True)
    config["frs_training"]["output"] = str(alias / "generated")

    with pytest.raises(ValueError, match=r"frs_training\.output.*protected"):
        validate_config(config, check_paths=False)


@pytest.mark.parametrize(
    ("section", "key", "protected"),
    [
        ("frs_training", "output", REPO_ROOT / ".venv" / "frs-output"),
        ("action_cache", "root", TRAIN_ROOT / ".venv" / "frs-cache"),
        ("tactile_embedding_cache", "root", TRAIN_ROOT / "new_module.py"),
    ],
)
def test_schema_rejects_repository_descendants_and_all_environment_trees(
    tmp_path: Path, section: str, key: str, protected: Path
) -> None:
    from train_pi05_frs.tools.train_frs import validate_config

    config = _valid_config(tmp_path)
    config[section][key] = str(protected)

    with pytest.raises(ValueError, match="protected"):
        validate_config(config, check_paths=False)


def test_schema_allows_only_designated_generated_roots_inside_standalone_project(
    tmp_path: Path,
) -> None:
    from train_pi05_frs.tools.train_frs import validate_config

    config = _valid_config(tmp_path)
    config["action_cache"]["root"] = str(TRAIN_ROOT / ".cache" / "action")
    config["tactile_embedding_cache"]["root"] = str(
        TRAIN_ROOT / ".cache" / "tactile"
    )
    config["frs_training"]["output"] = str(TRAIN_ROOT / "outputs" / "review")

    validate_config(config, check_paths=False)


def test_path_validator_rejects_symlinked_designated_generated_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from train_pi05_frs.utils import path_safety

    repo_root = tmp_path / "repo"
    train_root = repo_root / "train_pi05_frs"
    protected_venv = repo_root / ".venv"
    train_root.mkdir(parents=True)
    protected_venv.mkdir()
    (train_root / ".cache").symlink_to(protected_venv, target_is_directory=True)
    monkeypatch.setattr(path_safety, "REPO_ROOT", repo_root)
    monkeypatch.setattr(path_safety, "TRAIN_ROOT", train_root)
    monkeypatch.setattr(
        path_safety,
        "_ALLOWED_TRAIN_GENERATED",
        (train_root / ".cache", train_root / "outputs"),
    )
    monkeypatch.setattr(path_safety, "_EXACT_PROTECTED", (Path("/"),))
    monkeypatch.setattr(path_safety, "_PROTECTED_SUBTREES", ())

    with pytest.raises(ValueError, match="protected"):
        path_safety.validate_output_roots(
            {"config.action_cache.root": train_root / ".cache" / "inside-root-venv"}
        )


@pytest.mark.parametrize(
    ("left", "right"),
    [
        ("action_cache", "tactile_embedding_cache"),
        ("action_cache", "frs_training"),
        ("tactile_embedding_cache", "frs_training"),
    ],
)
@pytest.mark.parametrize("relation", ["same", "left_contains_right", "right_contains_left"])
def test_schema_rejects_overlapping_output_and_cache_roots(
    tmp_path: Path, left: str, right: str, relation: str
) -> None:
    from train_pi05_frs.tools.train_frs import validate_config

    config = _valid_config(tmp_path)
    key = {
        "action_cache": "root",
        "tactile_embedding_cache": "root",
        "frs_training": "output",
    }
    base = tmp_path / "collision"
    left_value, right_value = {
        "same": (base, base),
        "left_contains_right": (base, base / "nested"),
        "right_contains_left": (base / "nested", base),
    }[relation]
    config[left][key[left]] = str(left_value)
    config[right][key[right]] = str(right_value)

    with pytest.raises(ValueError, match="overlap"):
        validate_config(config, check_paths=False)


@pytest.mark.parametrize(
    "input_name", ["checkpoint", "encoder", "dataset", "norm", "resume"]
)
@pytest.mark.parametrize(
    "writable_name", ["action_cache", "tactile_embedding_cache", "frs_training"]
)
@pytest.mark.parametrize(
    "relation", ["same", "writable_contains_input", "input_contains_writable"]
)
def test_schema_rejects_output_overlap_with_every_read_only_asset_root(
    tmp_path: Path, input_name: str, writable_name: str, relation: str
) -> None:
    from train_pi05_frs.tools.train_frs import validate_config

    config = _valid_config(tmp_path / f"{input_name}-{relation}")
    base = tmp_path / f"collision-{input_name}-{relation}"
    writable, read_only = {
        "same": (base, base),
        "writable_contains_input": (base, base / "input"),
        "input_contains_writable": (base / "output", base),
    }[relation]
    writable_key = {
        "action_cache": "root",
        "tactile_embedding_cache": "root",
        "frs_training": "output",
    }[writable_name]
    config[writable_name][writable_key] = str(writable)
    if input_name == "checkpoint":
        config["checkpoint"] = str(read_only)
    elif input_name == "encoder":
        config["model"]["tactile_encoder_path"] = str(read_only)
    elif input_name == "dataset":
        config["datasets"][0]["root"] = str(read_only)
    elif input_name == "norm":
        config["norm_stats"]["dir"] = str(read_only.parent)
        config["norm_stats"]["asset_id"] = read_only.name
    else:
        config["frs_training"]["resume"] = True
        config["frs_training"]["resume_from"] = str(read_only)

    with pytest.raises(ValueError, match="read-only.*overlap"):
        validate_config(config, check_paths=False)


def test_schema_rejects_symlink_alias_between_output_and_input(
    tmp_path: Path,
) -> None:
    from train_pi05_frs.tools.train_frs import validate_config

    config = _valid_config(tmp_path / "config")
    shared = tmp_path / "shared"
    shared.mkdir()
    alias = tmp_path / "output-alias"
    alias.symlink_to(shared, target_is_directory=True)
    config["checkpoint"] = str(shared)
    config["frs_training"]["output"] = str(alias)

    with pytest.raises(ValueError, match="read-only.*overlap"):
        validate_config(config, check_paths=False)


def test_path_preflight_rejects_nonempty_fresh_output(tmp_path: Path) -> None:
    from train_pi05_frs.tools.train_frs import validate_config

    config = _valid_config(tmp_path)
    output = Path(config["frs_training"]["output"])
    output.mkdir(parents=True)
    (output / "stale.txt").write_text("stale", encoding="utf-8")

    with pytest.raises(FileExistsError, match="output directory is not empty"):
        validate_config(config, check_paths=True)


def test_path_preflight_owned_pipeline_log_does_not_mask_stale_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from train_pi05_frs.tools.train_frs import validate_config

    config = _valid_config(tmp_path)
    output = Path(config["frs_training"]["output"])
    output.mkdir(parents=True)
    pipeline_log = output / "pipeline_current.log"
    pipeline_log.write_text("current", encoding="utf-8")
    (output / "stale.txt").write_text("stale", encoding="utf-8")
    monkeypatch.setenv("FRS_PIPELINE_LOG", str(pipeline_log))

    with pytest.raises(FileExistsError, match="output directory is not empty"):
        validate_config(config, check_paths=True)


def test_path_preflight_does_not_trust_pipeline_log_outside_owned_stage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from train_pi05_frs.tools.train_frs import validate_config

    config = _valid_config(tmp_path)
    output = Path(config["frs_training"]["output"])
    output.mkdir(parents=True)
    pipeline_log = output / "pipeline_untrusted.log"
    pipeline_log.write_text("stale", encoding="utf-8")
    monkeypatch.setenv("FRS_PIPELINE_LOG", str(pipeline_log))
    monkeypatch.delenv("FRS_PIPELINE_STAGE", raising=False)

    with pytest.raises(FileExistsError, match="output directory is not empty"):
        validate_config(config, check_paths=True)


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda cfg: cfg["model"].update(
                camera_map={"not_a_pi05_camera": "observation.images.camera1"}
            ),
            "camera_map.*not_a_pi05_camera",
        ),
        (
            lambda cfg: cfg["model"].update(paligemma_variant="unknown"),
            "paligemma_variant",
        ),
        (
            lambda cfg: cfg["model"].update(tactile_num_tokens=3),
            "tactile_num_tokens",
        ),
        (
            lambda cfg: cfg["datasets"][0].update(episodes="0,1"),
            "episodes.*not supported",
        ),
        (
            lambda cfg: cfg["datasets"][0].update(weight=float("nan")),
            "weight",
        ),
    ],
)
def test_schema_rejects_model_contract_drift(
    tmp_path: Path, mutate: Any, message: str
) -> None:
    from train_pi05_frs.tools.train_frs import validate_config

    config = _valid_config(tmp_path)
    mutate(config)
    with pytest.raises(ValueError, match=message):
        validate_config(config, check_paths=False)


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda cfg: cfg["model"].update(action_horizon=51),
            "action_horizon.*tactile_window_divisor",
        ),
        (
            lambda cfg: cfg["frs_training"].update(model_dim=255),
            "model_dim.*num_heads",
        ),
        (
            lambda cfg: cfg["model"].update(
                tactile_keys=cfg["model"]["tactile_keys"][:3],
                tactile_num_tokens=3,
            ),
            "tactile_keys.*exactly 4",
        ),
        (
            lambda cfg: cfg["model"].update(
                tactile_keys=cfg["model"]["tactile_keys"] + ["observation.images.extra"],
                tactile_num_tokens=5,
            ),
            "tactile_num_tokens.*exactly 4",
        ),
    ],
)
def test_schema_rejects_early_shape_invariant_mismatches(
    tmp_path: Path, mutate: Any, message: str
) -> None:
    from train_pi05_frs.tools.train_frs import validate_config

    config = _valid_config(tmp_path)
    mutate(config)

    with pytest.raises(ValueError, match=message):
        validate_config(config, check_paths=False)


def test_schema_rejects_colliding_sanitized_dataset_cache_dirs(tmp_path: Path) -> None:
    from train_pi05_frs.tools.train_frs import validate_config

    config = _valid_config(tmp_path)
    config["datasets"] = [
        {"repo_id": "org/two", "root": config["datasets"][0]["root"]},
        {"repo_id": "org/../two", "root": config["datasets"][0]["root"]},
    ]

    with pytest.raises(ValueError, match="same sanitized action-cache directory"):
        validate_config(config, check_paths=False)


def test_path_preflight_rejects_missing_inputs_without_creating_outputs(tmp_path: Path) -> None:
    from train_pi05_frs.tools.train_frs import validate_config

    config = _valid_config(tmp_path)
    config["model"]["tactile_encoder_path"] = str(tmp_path / "missing-encoder")
    output = Path(config["frs_training"]["output"])
    with pytest.raises(FileNotFoundError, match="tactile encoder"):
        validate_config(config, check_paths=True)
    assert not output.exists()
    assert not Path(config["action_cache"]["root"]).exists()
    assert not Path(config["tactile_embedding_cache"]["root"]).exists()


def test_path_preflight_validates_encoder_and_local_norm_stats_files(tmp_path: Path) -> None:
    from train_pi05_frs.tools.train_frs import validate_config

    config = _valid_config(tmp_path)
    encoder = Path(config["model"]["tactile_encoder_path"])
    (encoder / "checkpoint.json").unlink()
    with pytest.raises(FileNotFoundError, match="encoder checkpoint"):
        validate_config(config, check_paths=True)

    config = _valid_config(tmp_path / "norm-case")
    stats_file = (
        Path(config["norm_stats"]["dir"])
        / config["norm_stats"]["asset_id"]
        / "norm_stats.json"
    )
    stats_file.unlink()
    with pytest.raises(FileNotFoundError, match="norm stats"):
        validate_config(config, check_paths=True)


def test_check_preflight_uses_pyproject_distribution_metadata_and_light_gpu_probe() -> None:
    from train_pi05_frs.tools import train_frs

    requirements = train_frs.production_runtime_requirements()
    calls: list[str] = []
    versions = {requirement.name: train_frs.importlib.metadata.version(requirement.name) for requirement in requirements}

    train_frs.preflight_environment(
        version_getter=lambda name: calls.append(name) or versions[name],
        gpu_probe=lambda: ["GPU 0: fake accelerator"],
        python_version=(3, 12),
    )

    assert calls == [requirement.name for requirement in requirements]
    required_names = {requirement.name.lower() for requirement in requirements}
    assert {
        "jax",
        "jax-cuda12-plugin",
        "jax-cuda12-pjrt",
        "optax",
        "torchvision",
        "torchcodec",
        "av",
        "pillow",
        "huggingface-hub",
        "sentencepiece",
        "safetensors",
    } <= required_names


def test_check_preflight_rejects_each_applicable_missing_dependency() -> None:
    from importlib.metadata import PackageNotFoundError
    from train_pi05_frs.tools import train_frs

    requirements = train_frs.production_runtime_requirements()
    versions = {requirement.name: train_frs.importlib.metadata.version(requirement.name) for requirement in requirements}
    for missing in requirements:
        def version_getter(name: str, *, missing_name: str = missing.name) -> str:
            if name == missing_name:
                raise PackageNotFoundError(name)
            return versions[name]

        with pytest.raises(RuntimeError, match=rf"dependency.*{re.escape(missing.name)}"):
            train_frs.preflight_environment(
                version_getter=version_getter,
                gpu_probe=lambda: ["GPU 0"],
                python_version=(3, 12),
            )


def test_check_preflight_rejects_unsatisfied_pyproject_specifier() -> None:
    from train_pi05_frs.tools import train_frs

    requirements = train_frs.production_runtime_requirements()
    versions = {requirement.name: train_frs.importlib.metadata.version(requirement.name) for requirement in requirements}

    def version_getter(name: str) -> str:
        if name == "jax":
            return "0.0.0"
        return versions[name]

    with pytest.raises(RuntimeError, match="jax.*==0.5.3"):
        train_frs.preflight_environment(
            version_getter=version_getter,
            gpu_probe=lambda: ["GPU 0"],
            python_version=(3, 12),
        )


@pytest.mark.parametrize("python_version", [(3, 11), (3, 13)])
def test_check_preflight_requires_exactly_python_3_12(
    python_version: tuple[int, int],
) -> None:
    from train_pi05_frs.tools import train_frs

    with pytest.raises(RuntimeError, match="Python 3.12"):
        train_frs.preflight_environment(
            version_getter=lambda name: pytest.fail(f"read metadata for {name}"),
            gpu_probe=lambda: pytest.fail("probed GPU"),
            python_version=python_version,
        )


def test_check_preflight_rejects_missing_gpu_without_importing_jax() -> None:
    from train_pi05_frs.tools import train_frs

    requirements = train_frs.production_runtime_requirements()
    versions = {requirement.name: train_frs.importlib.metadata.version(requirement.name) for requirement in requirements}
    sys.modules.pop("jax", None)

    with pytest.raises(RuntimeError, match="GPU"):
        train_frs.preflight_environment(
            version_getter=versions.__getitem__,
            gpu_probe=lambda: [],
            python_version=(3, 12),
        )
    assert "jax" not in sys.modules


def test_production_requirements_respect_platform_markers() -> None:
    from train_pi05_frs.tools import train_frs

    linux = {
        requirement.name.lower()
        for requirement in train_frs.production_runtime_requirements(
            marker_environment={"sys_platform": "linux", "platform_machine": "x86_64"}
        )
    }
    windows = {
        requirement.name.lower()
        for requirement in train_frs.production_runtime_requirements(
            marker_environment={"sys_platform": "win32", "platform_machine": "AMD64"}
        )
    }
    assert {"jax-cuda12-plugin", "jax-cuda12-pjrt", "nvidia-cuda-nvcc-cu12"} <= linux
    assert not {"jax-cuda12-plugin", "jax-cuda12-pjrt", "nvidia-cuda-nvcc-cu12"} & windows
    assert "torchcodec" in linux
    assert "torchcodec" in windows


def test_path_preflight_rejects_zip_that_is_not_a_numpy_archive(tmp_path: Path) -> None:
    from train_pi05_frs.tools.train_frs import validate_config

    config = _valid_config(tmp_path)
    params = Path(config["model"]["tactile_encoder_path"]) / "params-fake.npz"
    params.unlink()
    with zipfile.ZipFile(params, "w") as archive:
        archive.writestr("plain.txt", "not a numpy array")

    with pytest.raises(ValueError, match="valid npz.*array"):
        validate_config(config, check_paths=True)


@pytest.mark.parametrize("missing", ["tasks", "stats", "episodes"])
def test_path_preflight_requires_complete_lerobot_v3_metadata(
    tmp_path: Path, missing: str
) -> None:
    from train_pi05_frs.tools.train_frs import validate_config

    config = _valid_config(tmp_path)
    meta = Path(config["datasets"][0]["root"]) / "meta"
    target = {
        "tasks": meta / "tasks.parquet",
        "stats": meta / "stats.json",
        "episodes": meta / "episodes/chunk-000/file-000.parquet",
    }[missing]
    target.unlink()

    with pytest.raises((FileNotFoundError, ValueError), match=missing):
        validate_config(config, check_paths=True)


@pytest.mark.parametrize("missing", ["data", "video"])
def test_path_preflight_requires_referenced_lerobot_data_and_video_assets(
    tmp_path: Path, missing: str
) -> None:
    from train_pi05_frs.tools.train_frs import validate_config

    config = _valid_config(tmp_path)
    dataset = Path(config["datasets"][0]["root"])
    target = {
        "data": dataset / "data/chunk-000/file-000.parquet",
        "video": (
            dataset
            / "videos/observation.images.camera0/chunk-000/file-000.mp4"
        ),
    }[missing]
    target.unlink()

    with pytest.raises((FileNotFoundError, ValueError), match=missing):
        validate_config(config, check_paths=True)


@pytest.mark.parametrize(
    "missing_column",
    [
        "data/chunk_index",
        "data/file_index",
        "videos/observation.images.camera0/chunk_index",
        "videos/observation.images.camera0/file_index",
        "videos/observation.images.camera0/from_timestamp",
        "videos/observation.images.camera0/to_timestamp",
    ],
)
def test_path_preflight_requires_episode_asset_location_fields(
    tmp_path: Path, missing_column: str
) -> None:
    import pyarrow.parquet as pq
    from train_pi05_frs.tools.train_frs import validate_config

    config = _valid_config(tmp_path)
    episodes_path = (
        Path(config["datasets"][0]["root"])
        / "meta/episodes/chunk-000/file-000.parquet"
    )
    table = pq.read_table(episodes_path).drop([missing_column])
    pq.write_table(table, episodes_path)

    with pytest.raises(ValueError, match=re.escape(missing_column)):
        validate_config(config, check_paths=True)


@pytest.mark.parametrize("missing_column", ["index", "task_index"])
def test_path_preflight_requires_default_index_columns_in_data_parquet(
    tmp_path: Path, missing_column: str
) -> None:
    import pyarrow.parquet as pq
    from train_pi05_frs.tools.train_frs import validate_config

    config = _valid_config(tmp_path)
    data_path = Path(config["datasets"][0]["root"]) / "data/chunk-000/file-000.parquet"
    table = pq.read_table(data_path).drop([missing_column])
    pq.write_table(table, data_path)

    with pytest.raises(ValueError, match=missing_column):
        validate_config(config, check_paths=True)


@pytest.mark.parametrize(
    "missing_feature",
    ["timestamp", "frame_index", "episode_index", "index", "task_index"],
)
def test_path_preflight_requires_lerobot_v3_default_feature_declarations(
    tmp_path: Path, missing_feature: str
) -> None:
    from train_pi05_frs.tools.train_frs import validate_config

    config = _valid_config(tmp_path)
    info_path = Path(config["datasets"][0]["root"]) / "meta/info.json"
    info = json.loads(info_path.read_text(encoding="utf-8"))
    del info["features"][missing_feature]
    info_path.write_text(json.dumps(info), encoding="utf-8")

    with pytest.raises(ValueError, match=missing_feature):
        validate_config(config, check_paths=True)


@pytest.mark.parametrize(
    ("feature_name", "field", "bad_value"),
    [
        ("timestamp", "dtype", "float64"),
        ("frame_index", "shape", [2]),
        ("episode_index", "shape", [True]),
        ("task_index", "shape", [1.0]),
        ("index", "names", ["unexpected"]),
    ],
)
def test_path_preflight_requires_exact_lerobot_v3_default_feature_specs(
    tmp_path: Path, feature_name: str, field: str, bad_value: object
) -> None:
    from train_pi05_frs.tools.train_frs import validate_config

    config = _valid_config(tmp_path)
    info_path = Path(config["datasets"][0]["root"]) / "meta/info.json"
    info = json.loads(info_path.read_text(encoding="utf-8"))
    info["features"][feature_name][field] = bad_value
    info_path.write_text(json.dumps(info), encoding="utf-8")

    with pytest.raises(ValueError, match=feature_name):
        validate_config(config, check_paths=True)


def test_path_preflight_requires_default_feature_names_field(tmp_path: Path) -> None:
    from train_pi05_frs.tools.train_frs import validate_config

    config = _valid_config(tmp_path)
    info_path = Path(config["datasets"][0]["root"]) / "meta/info.json"
    info = json.loads(info_path.read_text(encoding="utf-8"))
    del info["features"]["timestamp"]["names"]
    info_path.write_text(json.dumps(info), encoding="utf-8")

    with pytest.raises(ValueError, match="timestamp.*names"):
        validate_config(config, check_paths=True)


def test_path_preflight_fixture_loads_through_real_lerobot_v3_reader_schema(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HF_DATASETS_CACHE", str(tmp_path / "hf-datasets-cache"))
    import datasets
    from lerobot.datasets.feature_utils import get_hf_features_from_features
    from lerobot.datasets.io_utils import load_nested_dataset
    from lerobot.datasets.utils import DatasetInfo
    from train_pi05_frs.tools.train_frs import validate_config

    monkeypatch.setattr(
        datasets.config, "HF_DATASETS_CACHE", str(tmp_path / "hf-datasets-cache")
    )

    config = _valid_config(tmp_path)
    validate_config(config, check_paths=True)
    dataset_root = Path(config["datasets"][0]["root"])
    info = json.loads((dataset_root / "meta/info.json").read_text(encoding="utf-8"))
    dataset_info = DatasetInfo.from_dict(info)

    loaded = load_nested_dataset(
        dataset_root / "data",
        features=get_hf_features_from_features(dataset_info.features),
    )

    assert len(loaded) == info["total_frames"]
    assert set(loaded.column_names) == {
        key
        for key, feature in dataset_info.features.items()
        if feature["dtype"] != "video"
    }


def test_path_preflight_requires_image_backed_feature_column(tmp_path: Path) -> None:
    from train_pi05_frs.tools.train_frs import validate_config

    config = _valid_config(tmp_path)
    info_path = Path(config["datasets"][0]["root"]) / "meta/info.json"
    info = json.loads(info_path.read_text(encoding="utf-8"))
    image_key = "observation.images.camera0"
    info["features"][image_key]["dtype"] = "image"
    info_path.write_text(json.dumps(info), encoding="utf-8")

    with pytest.raises(ValueError, match=re.escape(image_key)):
        validate_config(config, check_paths=True)


@pytest.mark.parametrize(
    ("stat_name", "field", "bad_value"),
    [
        ("state", "mean", [0.0, 0.0, 0.0, 0.0]),
        ("state", "std", [1.0, 1.0]),
        ("actions", "q99", [1.0, 1.0, 1.0]),
    ],
)
def test_path_preflight_validates_norm_stats_against_dataset_feature_width(
    tmp_path: Path, stat_name: str, field: str, bad_value: list[float]
) -> None:
    from train_pi05_frs.tools.train_frs import validate_config

    config = _valid_config(tmp_path)
    stats_path = (
        Path(config["norm_stats"]["dir"])
        / config["norm_stats"]["asset_id"]
        / "norm_stats.json"
    )
    payload = json.loads(stats_path.read_text(encoding="utf-8"))
    payload["norm_stats"][stat_name][field] = bad_value
    stats_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match=rf"norm stats {stat_name}.*{field}.*width"):
        validate_config(config, check_paths=True)


@pytest.mark.parametrize(
    ("stat_name", "field", "bad_value", "diagnostic"),
    [
        ("state", "mean", [[0.0, 0.0, 0.0]], "one-dimensional"),
        ("state", "std", [1.0, -0.1, 1.0], "std.*non-negative"),
        ("actions", "q99", [-1.0, 1.0], "q99.*greater.*q01"),
        ("actions", "q99", [-2.0, 1.0], "q99.*greater.*q01"),
    ],
)
def test_path_preflight_rejects_invalid_norm_stat_vector_semantics(
    tmp_path: Path,
    stat_name: str,
    field: str,
    bad_value: object,
    diagnostic: str,
) -> None:
    from train_pi05_frs.tools.train_frs import validate_config

    config = _valid_config(tmp_path)
    stats_path = (
        Path(config["norm_stats"]["dir"])
        / config["norm_stats"]["asset_id"]
        / "norm_stats.json"
    )
    payload = json.loads(stats_path.read_text(encoding="utf-8"))
    payload["norm_stats"][stat_name][field] = bad_value
    stats_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match=diagnostic):
        validate_config(config, check_paths=True)


def _set_encoder_params_file(config: dict[str, Any], params_file: str | None) -> Path:
    encoder = Path(config["model"]["tactile_encoder_path"])
    metadata_path = encoder / "checkpoint.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if params_file is None:
        metadata.pop("params_file", None)
    else:
        metadata["params_file"] = params_file
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    return encoder


@pytest.mark.parametrize("escape_kind", ["absolute", "parent", "symlink"])
def test_path_preflight_rejects_encoder_params_file_escape(
    tmp_path: Path, escape_kind: str
) -> None:
    from train_pi05_frs.tools.train_frs import validate_config

    config = _valid_config(tmp_path)
    external = tmp_path / "external-params.npz"
    np.savez(external, p00000=np.zeros((1,), dtype=np.float32))
    encoder = Path(config["model"]["tactile_encoder_path"])
    if escape_kind == "absolute":
        params_file = str(external)
    elif escape_kind == "parent":
        params_file = "../external-params.npz"
    else:
        params_file = "linked-params.npz"
        (encoder / params_file).symlink_to(external)
    _set_encoder_params_file(config, params_file)

    with pytest.raises(ValueError, match="params_file.*within.*checkpoint directory"):
        validate_config(config, check_paths=True)


def test_path_preflight_rejects_encoder_params_file_directory(tmp_path: Path) -> None:
    from train_pi05_frs.tools.train_frs import validate_config

    config = _valid_config(tmp_path)
    encoder = _set_encoder_params_file(config, "params-dir")
    (encoder / "params-dir").mkdir()

    with pytest.raises(ValueError, match="params_file.*regular file"):
        validate_config(config, check_paths=True)


def test_path_preflight_rejects_missing_encoder_params_file(tmp_path: Path) -> None:
    from train_pi05_frs.tools.train_frs import validate_config

    config = _valid_config(tmp_path)
    _set_encoder_params_file(config, "missing.npz")

    with pytest.raises(FileNotFoundError, match="params_file.*missing"):
        validate_config(config, check_paths=True)


def test_path_preflight_accepts_legacy_default_encoder_params_file(tmp_path: Path) -> None:
    from train_pi05_frs.tools.train_frs import validate_config

    config = _valid_config(tmp_path)
    encoder = _set_encoder_params_file(config, None)
    (encoder / "params-fake.npz").replace(encoder / "params.npz")

    validate_config(config, check_paths=True)


@pytest.mark.parametrize(
    ("metadata_key", "bad_value", "message"),
    [
        ("embedding_dim", 256, "embedding_dim.*tactile_embedding_dim"),
        ("tactile_image_size", 112, "tactile_image_size"),
    ],
)
def test_path_preflight_rejects_encoder_config_mismatch(
    tmp_path: Path, metadata_key: str, bad_value: int, message: str
) -> None:
    from train_pi05_frs.tools.train_frs import validate_config

    config = _valid_config(tmp_path)
    metadata_path = Path(config["model"]["tactile_encoder_path"]) / "checkpoint.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["tactile_clip_config"][metadata_key] = bad_value
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        validate_config(config, check_paths=True)


def test_path_preflight_resolves_camera_map_values_after_dataset_rename(tmp_path: Path) -> None:
    from train_pi05_frs.tools.train_frs import validate_config

    config = _valid_config(tmp_path)
    config["model"]["camera_map"]["left_wrist_0_rgb"] = "observation.images.typo"

    with pytest.raises(ValueError, match="camera_map.*typo"):
        validate_config(config, check_paths=True)


def test_prepare_reuses_one_model_preserves_url_and_sanitizes_each_cache_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from train_pi05_frs.tools import prepare_frs_pi05_cache as tool

    shared_model = object()
    load_calls: list[tuple[str, object]] = []
    prepare_calls: list[dict[str, object]] = []
    monkeypatch.setattr(
        tool,
        "load_pi0",
        lambda checkpoint, config: load_calls.append((checkpoint, config)) or shared_model,
    )
    monkeypatch.setattr(
        tool,
        "prepare_cache",
        lambda **kwargs: prepare_calls.append(kwargs) or Path(str(kwargs["cache_dir"])),
    )
    config = _valid_config(tmp_path)
    config["checkpoint"] = "gs://example/pi05"
    config["allow_download"] = True
    source = config["datasets"][0]
    config["datasets"] = [
        {**source, "repo_id": "org/one"},
        {**source, "repo_id": "org/../two"},
    ]

    outputs = tool.prepare_from_config(config)

    assert [call[0] for call in load_calls] == ["gs://example/pi05"]
    assert len(prepare_calls) == 2
    assert all(call["checkpoint_dir"] == "gs://example/pi05" for call in prepare_calls)
    assert all(call["loaded_model"] is shared_model for call in prepare_calls)
    cache_root = Path(config["action_cache"]["root"])
    assert outputs == [cache_root / "org" / "one", cache_root / "org" / "two"]


def test_prepare_resolves_local_checkpoint_and_norm_stats_from_repo_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from train_pi05_frs.tools import prepare_frs_pi05_cache as tool

    config = _valid_config(tmp_path)
    checkpoint = Path(config["checkpoint"])
    norm_stats = Path(config["norm_stats"]["dir"])
    config["checkpoint"] = os.path.relpath(checkpoint, REPO_ROOT)
    config["norm_stats"]["dir"] = os.path.relpath(norm_stats, REPO_ROOT)
    load_calls: list[str] = []
    prepare_calls: list[dict[str, object]] = []
    monkeypatch.setattr(
        tool,
        "load_pi0",
        lambda value, config: load_calls.append(value) or object(),
    )
    monkeypatch.setattr(
        tool,
        "prepare_cache",
        lambda **kwargs: prepare_calls.append(kwargs) or Path(str(kwargs["cache_dir"])),
    )

    tool.prepare_from_config(config)

    assert load_calls == [str(checkpoint)]
    assert prepare_calls[0]["checkpoint_dir"] == str(checkpoint)
    assert prepare_calls[0]["norm_stats_dir"] == str(norm_stats)


def test_precompute_resolves_dataset_roots_before_source_parsing(tmp_path: Path) -> None:
    from train_pi05_frs.tools import precompute_tactile_embeddings as tool

    config = _valid_config(tmp_path)
    dataset_root = Path(config["datasets"][0]["root"])
    config["datasets"][0]["root"] = os.path.relpath(dataset_root, REPO_ROOT)

    resolved = tool._resolved_source_config(config)

    assert resolved["datasets"][0]["root"] == str(dataset_root)
    assert config["datasets"][0]["root"] != str(dataset_root)


def test_complete_embedding_cache_must_have_valid_array(tmp_path: Path) -> None:
    from train_pi05_frs.tools import precompute_tactile_embeddings as tool

    path = tmp_path / "embeddings.npy"
    with pytest.raises(FileNotFoundError, match="embeddings"):
        tool._open_existing_embeddings(path, shape=(2, 4, 8), dtype="float16")

    wrong = np.lib.format.open_memmap(
        path, mode="w+", dtype=np.float32, shape=(2, 4, 7)
    )
    wrong.flush()
    del wrong
    with pytest.raises(ValueError, match="shape/dtype"):
        tool._open_existing_embeddings(path, shape=(2, 4, 8), dtype="float16")


def test_embedding_cache_rejects_unpaired_metadata_or_array(tmp_path: Path) -> None:
    from train_pi05_frs.tools import precompute_tactile_embeddings as tool

    metadata = tmp_path / "metadata.json"
    embeddings = tmp_path / "embeddings.npy"
    embeddings.write_bytes(b"partial")
    with pytest.raises(ValueError, match="inconsistent tactile cache files"):
        tool._validate_cache_file_pair(metadata, embeddings, overwrite=False)

    embeddings.unlink()
    metadata.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="inconsistent tactile cache files"):
        tool._validate_cache_file_pair(metadata, embeddings, overwrite=False)


@pytest.mark.parametrize(
    ("status", "completed", "total"),
    [("incomplete", 10, 10), ("complete", 9, 10), ("unknown", 1, 10)],
)
def test_embedding_cache_rejects_inconsistent_progress(
    status: str, completed: int, total: int
) -> None:
    from train_pi05_frs.tools import precompute_tactile_embeddings as tool

    with pytest.raises(ValueError, match="cache progress"):
        tool._validate_cache_progress(status=status, completed=completed, total=total)


def test_train_forwards_multi_dataset_caches_and_strict_booleans(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from train_pi05_frs.tools import train_frs

    config = _valid_config(tmp_path)
    dataset_root = Path(config["datasets"][0]["root"])
    config["datasets"][0]["root"] = os.path.relpath(dataset_root, REPO_ROOT)
    config["datasets"].append({**config["datasets"][0], "repo_id": "other/demo"})
    for repo_id in ("org/demo", "other/demo"):
        cache_dir = Path(config["action_cache"]["root"]).joinpath(*repo_id.split("/"))
        cache_dir.mkdir(parents=True)
        (cache_dir / "manifest.json").write_text("{}", encoding="utf-8")
    calls: list[dict[str, object]] = []
    monkeypatch.setattr(train_frs, "train_decoder", lambda **kwargs: calls.append(kwargs))

    train_frs.train_from_config(config)

    assert len(calls) == 1
    assert calls[0]["cache_dirs"] == [
        Path(config["action_cache"]["root"]) / "org" / "demo",
        Path(config["action_cache"]["root"]) / "other" / "demo",
    ]
    assert calls[0]["state_conditioning"] is True
    assert calls[0]["write_plots"] is True
    assert calls[0]["resume"] is False
    assert [source["root"] for source in calls[0]["dataset_sources"]] == [
        str(dataset_root),
        str(dataset_root),
    ]


def _fake_python(tmp_path: Path) -> Path:
    wrapper = tmp_path / "fake-python"
    wrapper.write_text(
        """#!/usr/bin/env bash
set -euo pipefail
if [[ -n "${FRS_EXPECT_PYTHON_INVOCATION:-}" && "$0" != "${FRS_EXPECT_PYTHON_INVOCATION}" ]]; then
    exit 42
fi
stage="${FRS_PIPELINE_STAGE:?}"
printf '%s\\n' "${stage}" >> "${FRS_EVENT_LOG:?}"
if [[ "${FRS_FAIL_STAGE:-}" == "${stage}" ]]; then
    exit 41
fi
if [[ "${stage}" == "validate" ]]; then
    exec "${FRS_REAL_PYTHON:?}" "$@"
fi
if [[ "${FRS_REVALIDATE_STAGES:-0}" == "1" ]]; then
    exec "${FRS_REAL_PYTHON:?}" -m train_pi05_frs.tools.train_frs \
        --config "${FRS_CONFIG_PATH:?}" --check
fi
exit 0
""",
        encoding="utf-8",
    )
    wrapper.chmod(0o755)
    return wrapper


def _run_launcher(
    tmp_path: Path,
    *,
    check: bool = False,
    fail_stage: str | None = None,
    block_jax: bool = False,
    relative_config: bool = False,
    foreground: bool = True,
    fake_tmux: bool = False,
    relative_python: bool = False,
    symlink_python: bool = False,
    revalidate_stages: bool = False,
) -> tuple[subprocess.CompletedProcess[str], list[str], dict[str, Any]]:
    config_path, config = _write_config(tmp_path)
    event_log = tmp_path / "events.txt"
    fake_python = _fake_python(tmp_path)
    fake_nvidia_smi = tmp_path / "nvidia-smi"
    fake_nvidia_smi.write_text(
        "#!/usr/bin/env bash\nprintf '%s\\n' 'GPU 0: test accelerator'\n",
        encoding="utf-8",
    )
    fake_nvidia_smi.chmod(0o755)
    selected_python = fake_python
    if symlink_python:
        selected_python = tmp_path / "venv" / "bin" / "python"
        selected_python.parent.mkdir(parents=True)
        selected_python.symlink_to(fake_python)
    environment = {
        **os.environ,
        "TRAIN_PI05_FRS_PYTHON": (
            os.path.relpath(selected_python, REPO_ROOT)
            if relative_python
            else str(selected_python)
        ),
        "FRS_EXPECT_PYTHON_INVOCATION": str(selected_python) if symlink_python else "",
        "FRS_REAL_PYTHON": sys.executable,
        "FRS_EVENT_LOG": str(event_log),
        "FRS_FAIL_STAGE": fail_stage or "",
        "FRS_NVIDIA_SMI": str(fake_nvidia_smi),
        "FRS_REVALIDATE_STAGES": "1" if revalidate_stages else "0",
        "FRS_CONFIG_PATH": str(config_path),
    }
    if foreground:
        environment["FRS_FOREGROUND"] = "1"
    else:
        environment.pop("FRS_FOREGROUND", None)
    if fake_tmux:
        fake_bin = tmp_path / "fake-bin"
        fake_bin.mkdir()
        tmux = fake_bin / "tmux"
        tmux.write_text(
            """#!/usr/bin/env bash
printf '%s\\n' "$*" >> "${FRS_TMUX_LOG:?}"
if [[ "${1:-}" == "has-session" ]]; then
    exit 1
fi
inner="${!#}"
bash -c "${inner}"
""",
            encoding="utf-8",
        )
        tmux.chmod(0o755)
        environment["FRS_TMUX_LOG"] = str(tmp_path / "tmux.txt")
        environment["PATH"] = f"{fake_bin}:{environment['PATH']}"
    if block_jax:
        blocker = tmp_path / "block-jax"
        blocker.mkdir()
        (blocker / "jax.py").write_text(
            "raise AssertionError('JAX imported during --check')\n", encoding="utf-8"
        )
        environment["PYTHONPATH"] = f"{blocker}:{environment.get('PYTHONPATH', '')}"
    command = ["bash", str(LAUNCHER)]
    if check:
        command.append("--check")
    command.append(os.path.relpath(config_path, REPO_ROOT) if relative_config else str(config_path))
    result = subprocess.run(
        command,
        cwd=REPO_ROOT,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )
    events = event_log.read_text(encoding="utf-8").splitlines() if event_log.exists() else []
    return result, events, config


def test_foreground_launcher_runs_exact_pipeline_order(tmp_path: Path) -> None:
    result, events, _ = _run_launcher(tmp_path)

    assert result.returncode == 0, result.stderr
    assert events == [
        "validate",
        "checkpoint-smoke",
        "precompute-tactile",
        "prepare-pi05-cache",
        "train-frs",
    ]


def test_foreground_launcher_allows_only_its_own_log_during_stage_revalidation(
    tmp_path: Path,
) -> None:
    result, events, config = _run_launcher(tmp_path, revalidate_stages=True)

    assert result.returncode == 0, result.stderr
    assert events == [
        "validate",
        "checkpoint-smoke",
        "precompute-tactile",
        "prepare-pi05-cache",
        "train-frs",
    ]
    output = Path(config["frs_training"]["output"])
    assert len(list(output.glob("pipeline_*.log"))) == 1


def test_schema_preserves_transactional_implicit_resume(tmp_path: Path) -> None:
    from train_pi05_frs.tools.train_frs import validate_config

    config = _valid_config(tmp_path)
    config["frs_training"]["resume"] = True

    validate_config(config, check_paths=False)


@pytest.mark.parametrize("target_name", ["action_cache", "tactile_cache"])
def test_schema_rejects_implicit_resume_symlink_to_writable_cache(
    tmp_path: Path, target_name: str
) -> None:
    from train_pi05_frs.tools.train_frs import validate_config

    config = _valid_config(tmp_path)
    config["frs_training"]["resume"] = True
    output = Path(config["frs_training"]["output"])
    output.mkdir(parents=True)
    target = Path(
        config[
            "action_cache" if target_name == "action_cache" else "tactile_embedding_cache"
        ]["root"]
    )
    target.mkdir(parents=True)
    (output / "last").symlink_to(target, target_is_directory=True)

    with pytest.raises(ValueError, match="implicit resume.*same output"):
        validate_config(config, check_paths=True)


def test_schema_rejects_symlinked_generation_container_escape(
    tmp_path: Path,
) -> None:
    from train_pi05_frs.tools.train_frs import validate_config

    config = _valid_config(tmp_path)
    config["frs_training"]["resume"] = True
    output = Path(config["frs_training"]["output"])
    output.mkdir(parents=True)
    cache_root = Path(config["action_cache"]["root"])
    generation = cache_root / "generation-a"
    generation.mkdir(parents=True)
    (output / ".checkpoint-generations").symlink_to(
        cache_root, target_is_directory=True
    )
    (output / "last").symlink_to(
        Path(".checkpoint-generations") / generation.name,
        target_is_directory=True,
    )

    with pytest.raises(ValueError, match="implicit resume.*same output"):
        validate_config(config, check_paths=True)


@pytest.mark.parametrize("layout", ["legacy", "generation"])
def test_schema_allows_controlled_implicit_resume_layout(
    tmp_path: Path, layout: str
) -> None:
    from train_pi05_frs.tools.train_frs import validate_config

    config = _valid_config(tmp_path)
    config["frs_training"]["resume"] = True
    output = Path(config["frs_training"]["output"])
    if layout == "legacy":
        (output / "last").mkdir(parents=True)
    else:
        generation = output / ".checkpoint-generations" / "generation-a"
        generation.mkdir(parents=True)
        (output / "last").symlink_to(
            Path(".checkpoint-generations") / generation.name,
            target_is_directory=True,
        )

    validate_config(config, check_paths=True)


@pytest.mark.parametrize(
    "failed_stage",
    [
        "validate",
        "checkpoint-smoke",
        "precompute-tactile",
        "prepare-pi05-cache",
        "train-frs",
    ],
)
def test_launcher_stops_after_each_failed_stage(tmp_path: Path, failed_stage: str) -> None:
    expected = [
        "validate",
        "checkpoint-smoke",
        "precompute-tactile",
        "prepare-pi05-cache",
        "train-frs",
    ]
    result, events, _ = _run_launcher(tmp_path, fail_stage=failed_stage)

    assert result.returncode != 0
    assert events == expected[: expected.index(failed_stage) + 1]


def test_check_is_dependency_light_and_creates_no_pipeline_artifacts(tmp_path: Path) -> None:
    result, events, config = _run_launcher(tmp_path, check=True, block_jax=True)

    assert result.returncode == 0, result.stderr
    assert events == ["validate"]
    assert not Path(config["frs_training"]["output"]).exists()
    assert not Path(config["action_cache"]["root"]).exists()
    assert not Path(config["tactile_embedding_cache"]["root"]).exists()


def test_launcher_resolves_config_before_changing_directory(tmp_path: Path) -> None:
    result, events, _ = _run_launcher(tmp_path, check=True, relative_config=True)

    assert result.returncode == 0, result.stderr
    assert events == ["validate"]


def test_launcher_preserves_virtualenv_python_symlink(tmp_path: Path) -> None:
    result, events, _ = _run_launcher(tmp_path, check=True, symlink_python=True)

    assert result.returncode == 0, result.stderr
    assert events == ["validate"]


def test_tmux_is_opt_in_after_preflight_and_before_output_creation(tmp_path: Path) -> None:
    result, events, config = _run_launcher(
        tmp_path, foreground=False, fake_tmux=True, relative_python=True
    )

    assert result.returncode == 0, result.stderr
    assert events == [
        "validate",
        "checkpoint-smoke",
        "precompute-tactile",
        "prepare-pi05-cache",
        "train-frs",
    ]
    tmux_calls = (tmp_path / "tmux.txt").read_text(encoding="utf-8").splitlines()
    assert tmux_calls[0].startswith("has-session -t ")
    assert tmux_calls[1].startswith("new-session -d -s ")
    output = Path(config["frs_training"]["output"])
    assert len(list(output.glob("pipeline_*.log"))) == 1


def test_check_exits_before_tmux_even_when_tmux_is_available(tmp_path: Path) -> None:
    result, events, _ = _run_launcher(
        tmp_path, check=True, foreground=False, fake_tmux=True
    )

    assert result.returncode == 0, result.stderr
    assert events == ["validate"]
    assert not (tmp_path / "tmux.txt").exists()


def test_launcher_uses_private_src_and_training_interpreter_without_uv() -> None:
    script = LAUNCHER.read_text(encoding="utf-8")

    assert 'TRAIN_PYTHON="${TRAIN_PI05_FRS_PYTHON:-${TRAIN_ROOT}/.venv/bin/python}"' in script
    assert 'export PYTHONPATH="${TRAIN_ROOT}/src:${REPO_ROOT}' in script
    assert "export PYTHONSAFEPATH=1" in script
    assert 'cd "${TRAIN_ROOT}"' in script
    assert "uv run" not in script


def test_launcher_python_path_does_not_shadow_protected_root_utils() -> None:
    result = subprocess.run(
        [
            "bash",
            "-c",
            (
                f'source "{LAUNCHER}"\n'
                'cd "${TRAIN_ROOT}"\n'
                '"${TRAIN_PYTHON}" -c \'import train_pi05_frs.utils.data; '
                'import utils; print(utils.__file__)\''
            ),
        ],
        cwd=REPO_ROOT,
        env={
            **os.environ,
            "JAX_PLATFORMS": "cpu",
            "PYTHONDONTWRITEBYTECODE": "1",
        },
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert Path(result.stdout.strip()).resolve() == REPO_ROOT / "utils/__init__.py"


def test_spawn_workers_complete_a_real_synthetic_batch(tmp_path: Path) -> None:
    stub_root = tmp_path / "stubs"
    image_dataset_module = stub_root / "train_encoder" / "utils" / "image_dataset.py"
    image_dataset_module.parent.mkdir(parents=True)
    (stub_root / "train_encoder" / "__init__.py").write_text("", encoding="utf-8")
    (stub_root / "train_encoder" / "utils" / "__init__.py").write_text("", encoding="utf-8")
    image_dataset_module.write_text(
        """
import numpy as np


class SyntheticDataset:
    def indices_for_episode(self, episode_index):
        assert episode_index == 0
        return (0, 1, 2)

    def get_images(self, frame_index, tactile_keys, *, as_float):
        assert not as_float
        return {
            key: np.full((2, 2, 3), frame_index, dtype=np.uint8)
            for key in tactile_keys
        }


class DatasetBundle:
    dataset = SyntheticDataset()


def create_image_dataset(repo_id, *, image_size, cache_size):
    assert repo_id == "org/synthetic"
    assert image_size == 2
    assert cache_size == 256
    return DatasetBundle()
""",
        encoding="utf-8",
    )
    driver = tmp_path / "spawn_driver.py"
    driver.write_text(
        """
import json

from train_pi05_frs.utils.mp_batches import MpTactileWindowLoader


def main():
    loader = MpTactileWindowLoader(
        repo_id="org/synthetic",
        image_size=2,
        image_cache_size=8,
        tactile_window=2,
        history_stride=1,
        num_workers=2,
        prefetch_batches=1,
        load_threads=1,
    )
    try:
        batches = list(loader.iter_image_batches([[(2, 0)]]))
    finally:
        loader.close()
    batch = batches[0]
    assert batch.shape == (1, 2, 4, 2, 2, 3)
    assert (batch[:, 0] == 1).all()
    assert (batch[:, 1] == 2).all()
    print(json.dumps({"shape": list(batch.shape), "dtype": str(batch.dtype)}))


if __name__ == "__main__":
    main()
""",
        encoding="utf-8",
    )
    result = subprocess.run(
        [str(TRAIN_ROOT / ".venv/bin/python"), str(driver)],
        cwd=REPO_ROOT,
        env={
            **os.environ,
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONPATH": ":".join(
                (str(stub_root), str(TRAIN_ROOT / "src"), str(REPO_ROOT))
            ),
        },
        check=False,
        text=True,
        capture_output=True,
        timeout=60,
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout.strip().splitlines()[-1])
    assert payload == {"shape": [1, 2, 4, 2, 2, 3], "dtype": "uint8"}


def test_readme_documents_training_boundary_and_operational_handoff() -> None:
    readme = (TRAIN_ROOT / "README.md").read_text(encoding="utf-8")

    for phrase in (
        "train_pi05_frs/.venv",
        "setup_env.sh",
        "start_frs_pi05_train.sh",
        "evaluate",
        "resume",
        "tmux",
        "deploy_pi05",
        "train_encoder",
        "modality",
    ):
        assert phrase in readme
    assert "/home/typhon/FRS_Tact-pi05-frs-jax" not in readme
    assert "--dataset-root" not in readme
    assert "dataset repository ID recorded in the action-cache manifest" in readme
    assert "Automated mock/CPU verification" in readme
    assert "Real GPU/data/checkpoint verification" in readme
    assert "has not been run" in readme
    assert "dereference" in readme and ".checkpoint-generations" in readme
    assert "data parquet" in readme and "video assets" in readme
    assert "never point" in readme and "mutable" in readme and "deployment" in readme
    assert "PINNED_DECODER_CHECKPOINT=" in readme
    assert "readlink -f" in readme
    assert '--checkpoint-dir "${PINNED_DECODER_CHECKPOINT}"' in readme
    for script in re.findall(r"```bash\n(.*?)\n```", readme, flags=re.DOTALL):
        result = subprocess.run(
            ["bash", "-n"], input=script, check=False, text=True, capture_output=True
        )
        assert result.returncode == 0, result.stderr
