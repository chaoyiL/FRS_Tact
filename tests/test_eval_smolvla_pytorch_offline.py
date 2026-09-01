import csv
import importlib.util
import io
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

MODULE_PATH = Path(__file__).resolve().parents[1] / "tools" / "eval_smolvla_pytorch_offline.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("eval_smolvla_pytorch_offline", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _rotation_6d_from_matrix(matrix: np.ndarray) -> np.ndarray:
    basis = np.asarray(matrix, dtype=np.float64)[:, :2]
    return np.concatenate((basis[:, 0], basis[:, 1]))


def _rotation_z_matrix(theta_degrees: float) -> np.ndarray:
    theta = np.deg2rad(theta_degrees)
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    return np.asarray(
        [
            [cos_theta, -sin_theta, 0.0],
            [sin_theta, cos_theta, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def _fill_identity_rotations(actions: np.ndarray) -> None:
    identity = _rotation_6d_from_matrix(np.eye(3, dtype=np.float64)).astype(np.float32)
    actions[..., 3:9] = identity
    actions[..., 13:19] = identity


def _png_bytes(color: tuple[int, int, int]) -> bytes:
    pil_image = pytest.importorskip("PIL.Image")
    buffer = io.BytesIO()
    pil_image.new("RGB", (2, 1), color).save(buffer, format="PNG")
    return buffer.getvalue()


def _write_raw_lerobot_parquet(path: Path, rows: list[dict[str, object]]) -> None:
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    table = pa.table(
        {
            "episode_index": pa.array([row["episode_index"] for row in rows], type=pa.int64()),
            "frame_index": pa.array([row["frame_index"] for row in rows], type=pa.int64()),
            "observation.state": pa.array(
                [row["observation.state"] for row in rows], type=pa.list_(pa.float32())
            ),
            "actions": pa.array([row["actions"] for row in rows], type=pa.list_(pa.float32())),
            "observation.images.camera0": pa.array(
                [row["observation.images.camera0"] for row in rows],
                type=pa.struct([("bytes", pa.binary()), ("path", pa.string())]),
            ),
        }
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path)


def _write_episode_meta(dataset_root: Path, episodes: list[dict[str, object]]) -> None:
    meta_path = dataset_root / "meta" / "episodes.jsonl"
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(
        "".join(json.dumps(episode, sort_keys=True) + "\n" for episode in episodes),
        encoding="utf-8",
    )


def test_build_gt_chunks_aligns_t_plus_h_and_masks_padding_and_terminal_actions():
    actions = np.asarray(
        [
            [1.0, 10.0],
            [2.0, 20.0],
            [3.0, 30.0],
            [0.0, 0.0],
        ],
        dtype=np.float32,
    )

    module = _load_module()
    gt, valid = module.build_gt_chunks(actions, horizon=3)

    expected_gt = np.asarray(
        [
            [[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]],
            [[2.0, 20.0], [3.0, 30.0], [0.0, 0.0]],
            [[3.0, 30.0], [0.0, 0.0], [0.0, 0.0]],
            [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
        ],
        dtype=np.float32,
    )
    expected_valid = np.asarray(
        [
            [True, True, True],
            [True, True, False],
            [True, False, False],
            [False, False, False],
        ]
    )

    np.testing.assert_allclose(gt, expected_gt)
    np.testing.assert_array_equal(valid, expected_valid)


def test_build_gt_chunks_stops_validity_after_first_terminal_sentinel():
    actions = np.asarray(
        [
            [1.0, 10.0],
            [0.0, 0.0],
            [2.0, 20.0],
        ],
        dtype=np.float32,
    )

    module = _load_module()
    gt, valid = module.build_gt_chunks(actions, horizon=3)

    np.testing.assert_allclose(
        gt[0],
        np.asarray(
            [
                [1.0, 10.0],
                [0.0, 0.0],
                [2.0, 20.0],
            ],
            dtype=np.float32,
        ),
    )
    np.testing.assert_array_equal(valid[0], np.asarray([True, False, False]))


def test_parse_episode_selection_accepts_csv_ranges_and_rejects_invalid_tokens():
    module = _load_module()

    assert module.parse_episode_selection(None) is None
    assert module.parse_episode_selection("") is None
    assert module.parse_episode_selection("4,2-3,3,1") == (1, 2, 3, 4)
    parser = module._build_arg_parser()
    args = parser.parse_args(["--config", "c.yaml", "--dataset-root", "data", "--output-dir", "out"])
    assert args.episodes == "202-211"

    with pytest.raises(ValueError, match="episode range"):
        module.parse_episode_selection("5-2")
    with pytest.raises(ValueError, match="episode"):
        module.parse_episode_selection("1,nope")


def test_decode_image_cell_reads_embedded_bytes_and_relative_path(tmp_path: Path):
    module = _load_module()
    image_path = tmp_path / "images" / "camera0" / "frame-000000.png"
    image_path.parent.mkdir(parents=True)
    image_path.write_bytes(_png_bytes((8, 9, 10)))

    embedded = module.decode_image_cell({"bytes": _png_bytes((1, 2, 3)), "path": ""}, dataset_root=tmp_path)
    from_path = module.decode_image_cell(
        {"bytes": None, "path": "images/camera0/frame-000000.png"},
        dataset_root=tmp_path,
    )

    assert embedded.shape == (1, 2, 3)
    assert embedded.dtype == np.uint8
    np.testing.assert_array_equal(embedded[0, 0], np.asarray([1, 2, 3], dtype=np.uint8))
    np.testing.assert_array_equal(from_path[0, 0], np.asarray([8, 9, 10], dtype=np.uint8))


def test_load_raw_lerobot_episodes_filters_sorts_and_requires_actions_key(tmp_path: Path):
    module = _load_module()
    dataset_root = tmp_path / "dataset"
    action0 = np.arange(20, dtype=np.float32).tolist()
    action1 = (np.arange(20, dtype=np.float32) + 100.0).tolist()
    action2 = (np.arange(20, dtype=np.float32) + 200.0).tolist()
    _write_episode_meta(dataset_root, [{"episode_index": 1, "episode_length": 1}, {"episode_index": 2, "episode_length": 2}])
    _write_raw_lerobot_parquet(
        dataset_root / "data" / "chunk-000" / "file-000.parquet",
        [
            {
                "episode_index": 2,
                "frame_index": 1,
                "observation.state": [4.0, 5.0],
                "actions": action1,
                "observation.images.camera0": {"bytes": _png_bytes((20, 21, 22)), "path": ""},
            },
            {
                "episode_index": 1,
                "frame_index": 0,
                "observation.state": [9.0, 10.0],
                "actions": action2,
                "observation.images.camera0": {"bytes": _png_bytes((30, 31, 32)), "path": ""},
            },
            {
                "episode_index": 2,
                "frame_index": 0,
                "observation.state": [1.0, 2.0],
                "actions": action0,
                "observation.images.camera0": {"bytes": _png_bytes((10, 11, 12)), "path": ""},
            },
        ],
    )

    episodes = module.load_raw_lerobot_episodes(
        dataset_root=dataset_root,
        episodes=(2,),
        image_keys=("observation.images.camera0",),
        state_dim=2,
        action_dim=20,
    )

    assert [episode.episode_index for episode in episodes] == [2]
    assert episodes[0].frame_indices.tolist() == [0, 1]
    np.testing.assert_allclose(episodes[0].states, np.asarray([[1.0, 2.0], [4.0, 5.0]], dtype=np.float32))
    np.testing.assert_allclose(episodes[0].actions, np.asarray([action0, action1], dtype=np.float32))
    np.testing.assert_array_equal(
        episodes[0].observations[0]["observation.images.camera0"][0, 0],
        np.asarray([10, 11, 12], dtype=np.uint8),
    )

    missing_action_root = tmp_path / "missing-action"
    _write_episode_meta(missing_action_root, [{"episode_index": 2, "episode_length": 1}])
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    table = pa.table(
        {
            "episode_index": pa.array([2], type=pa.int64()),
            "frame_index": pa.array([0], type=pa.int64()),
            "observation.state": pa.array([[1.0, 2.0]], type=pa.list_(pa.float32())),
            "action": pa.array([action0], type=pa.list_(pa.float32())),
            "observation.images.camera0": pa.array(
                [{"bytes": _png_bytes((10, 11, 12)), "path": ""}],
                type=pa.struct([("bytes", pa.binary()), ("path", pa.string())]),
            ),
        }
    )
    missing_action_path = missing_action_root / "data" / "chunk-000" / "file-000.parquet"
    missing_action_path.parent.mkdir(parents=True)
    pq.write_table(table, missing_action_path)
    with pytest.raises(KeyError, match="actions"):
        module.load_raw_lerobot_episodes(
            dataset_root=missing_action_root,
            episodes=(2,),
            image_keys=("observation.images.camera0",),
            state_dim=2,
            action_dim=20,
        )


def test_load_raw_lerobot_episodes_validates_requested_metadata_lengths_and_contiguous_frames(
    tmp_path: Path,
):
    module = _load_module()
    dataset_root = tmp_path / "dataset"
    _write_episode_meta(
        dataset_root,
        [
            {"episode_index": 202, "episode_length": 2},
            {"episode_index": 203, "length": 1},
            {"episode_index": 204, "dataset_from_index": 0, "dataset_to_index": 2},
        ],
    )
    action = np.ones(20, dtype=np.float32).tolist()
    _write_raw_lerobot_parquet(
        dataset_root / "data" / "chunk-000" / "file-000.parquet",
        [
            {
                "episode_index": 202,
                "frame_index": 0,
                "observation.state": np.ones(20, dtype=np.float32).tolist(),
                "actions": action,
                "observation.images.camera0": {"bytes": _png_bytes((1, 2, 3)), "path": ""},
            },
            {
                "episode_index": 202,
                "frame_index": 1,
                "observation.state": np.ones(20, dtype=np.float32).tolist(),
                "actions": action,
                "observation.images.camera0": {"bytes": _png_bytes((4, 5, 6)), "path": ""},
            },
            {
                "episode_index": 204,
                "frame_index": 0,
                "observation.state": np.ones(20, dtype=np.float32).tolist(),
                "actions": action,
                "observation.images.camera0": {"bytes": _png_bytes((7, 8, 9)), "path": ""},
            },
            {
                "episode_index": 204,
                "frame_index": 0,
                "observation.state": np.ones(20, dtype=np.float32).tolist(),
                "actions": action,
                "observation.images.camera0": {"bytes": _png_bytes((10, 11, 12)), "path": ""},
            },
        ],
    )

    valid = module.load_raw_lerobot_episodes(
        dataset_root=dataset_root,
        episodes=(202,),
        image_keys=("observation.images.camera0",),
        state_dim=20,
        action_dim=20,
    )
    assert [episode.episode_index for episode in valid] == [202]

    with pytest.raises(ValueError, match="missing requested episodes.*203"):
        module.load_raw_lerobot_episodes(
            dataset_root=dataset_root,
            episodes=(202, 203),
            image_keys=("observation.images.camera0",),
            state_dim=20,
            action_dim=20,
        )
    with pytest.raises(ValueError, match="duplicate frame_index.*204"):
        module.load_raw_lerobot_episodes(
            dataset_root=dataset_root,
            episodes=(204,),
            image_keys=("observation.images.camera0",),
            state_dim=20,
            action_dim=20,
        )


def test_load_raw_lerobot_episodes_requires_episode_metadata_for_complete_eval(tmp_path: Path):
    module = _load_module()
    dataset_root = tmp_path / "dataset"
    _write_raw_lerobot_parquet(
        dataset_root / "data" / "chunk-000" / "file-000.parquet",
        [
            {
                "episode_index": 202,
                "frame_index": 0,
                "observation.state": np.ones(20, dtype=np.float32).tolist(),
                "actions": np.ones(20, dtype=np.float32).tolist(),
                "observation.images.camera0": {"bytes": _png_bytes((1, 2, 3)), "path": ""},
            },
        ],
    )

    with pytest.raises(FileNotFoundError, match="meta/episodes.jsonl"):
        module.load_raw_lerobot_episodes(
            dataset_root=dataset_root,
            episodes=(202,),
            image_keys=("observation.images.camera0",),
            state_dim=20,
            action_dim=20,
        )


def test_load_eval_runtime_reuses_deploy_helpers_and_validates_checkpoint_contract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    module = _load_module()
    config_path = tmp_path / "deploy.yaml"
    config_path.write_text(
        "\n".join(
            [
                "backend: pytorch_smolvla",
                "checkpoint: checkpoints/policy",
                "device: cpu",
                "connection: {}",
                "observation:",
                "  data_type: vision",
                "  language_prompt: fold",
                "control:",
                "  action_horizon: 3",
                "runtime: {}",
                "rename_map:",
                "  observation.images.camera0: observation.images.front",
            ]
        ),
        encoding="utf-8",
    )
    calls: list[tuple[str, object]] = []

    class FakePolicy:
        def __init__(self, chunk_size: int):
            self.config = SimpleNamespace(chunk_size=chunk_size)

        def to(self, device):
            calls.append(("to", str(device)))
            return self

        def eval(self):
            calls.append(("eval", None))
            return self

        def reset(self):
            calls.append(("reset", None))

    fake_policy = FakePolicy(chunk_size=3)

    fake_runtime = SimpleNamespace(
        _load_config=lambda path: calls.append(("_load_config", path)) or {
            "checkpoint": "checkpoints/policy",
            "device": "cuda",
            "control": {"action_horizon": 3},
            "observation": {"language_prompt": "fold"},
            "rename_map": {"observation.images.camera0": "observation.images.front"},
            "revision": "main",
            "allow_download": True,
        },
        _resolve_checkpoint=lambda value, path: calls.append(("_resolve_checkpoint", value)) or "/abs/policy",
        _load_policy=lambda checkpoint, **kwargs: calls.append(("_load_policy", kwargs)) or fake_policy,
        _policy_contract=lambda policy: calls.append(("_policy_contract", policy)) or (
            2,
            20,
            ("observation.images.front",),
        ),
        make_pre_post_processors=lambda *args, **kwargs: calls.append(("make_pre_post_processors", kwargs))
        or ("pre", "post"),
        _prepare_frame=lambda observation, **kwargs: {"prepared": observation, **kwargs},
    )
    monkeypatch.setattr(module, "_smolvla_runtime", lambda: fake_runtime)

    runtime = module.load_eval_runtime(config_path, device="cpu")

    assert runtime.checkpoint == "/abs/policy"
    assert runtime.horizon == 3
    assert runtime.state_dim == 2
    assert runtime.action_dim == 20
    assert runtime.model_image_keys == ("observation.images.front",)
    assert runtime.dataset_image_keys == ("observation.images.camera0",)
    assert runtime.preprocess == "pre"
    assert runtime.postprocess == "post"
    assert ("_load_config", config_path) in calls
    assert ("_resolve_checkpoint", "checkpoints/policy") in calls
    assert any(name == "_load_policy" and value["revision"] == "main" for name, value in calls)
    assert any(name == "make_pre_post_processors" for name, _ in calls)

    fake_policy.config.chunk_size = 4
    with pytest.raises(ValueError, match="chunk_size"):
        module.load_eval_runtime(config_path, device="cpu")


def test_episode_shards_include_metadata_and_only_resume_exact_matches(tmp_path: Path):
    module = _load_module()
    pred = np.zeros((2, 3, 20), dtype=np.float32)
    gt = np.ones((2, 3, 20), dtype=np.float32)
    valid = np.ones((2, 3), dtype=bool)
    metadata = module.episode_shard_metadata(
        config_path=Path("/configs/eval.yaml"),
        checkpoint="/checkpoints/policy",
        dataset_root=Path("/datasets/raw"),
        seed=123,
        device="cpu",
        episode_index=7,
        frame_count=2,
        horizon=3,
        state_dim=2,
        action_dim=20,
        image_keys=("observation.images.camera0",),
    )

    path = module.save_episode_shard(
        output_dir=tmp_path,
        episode_index=7,
        pred=pred,
        gt=gt,
        valid=valid,
        frame_indices=np.asarray([0, 1], dtype=np.int64),
        metadata=metadata,
    )
    loaded = module.load_matching_episode_shard(tmp_path, episode_index=7, expected_metadata=metadata)

    assert path.name == "episode_000007.npz"
    assert loaded is not None
    assert loaded["metadata"]["complete"] is True
    assert loaded["metadata"]["seed"] == 123
    np.testing.assert_array_equal(loaded["pred"], pred)
    np.testing.assert_array_equal(loaded["frame_indices"], np.asarray([0, 1], dtype=np.int64))

    mismatched = dict(metadata)
    mismatched["seed"] = 456
    assert module.load_matching_episode_shard(tmp_path, episode_index=7, expected_metadata=mismatched) is None


def test_reproducibility_fingerprints_changed_config_checkpoint_dataset_and_revision(
    tmp_path: Path,
):
    module = _load_module()
    config_path = tmp_path / "deploy.yaml"
    config_path.write_text("prompt: old\n", encoding="utf-8")
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "adapter_config.json").write_text('{"base_model_name_or_path": "base"}', encoding="utf-8")
    (checkpoint / "adapter_model.safetensors").write_bytes(b"adapter")
    (checkpoint / "policy_preprocessor.json").write_text("policy-pre", encoding="utf-8")
    (checkpoint / "policy_postprocessor.json").write_text("policy-post", encoding="utf-8")
    (checkpoint / "policy_preprocessor_step_0_processor.safetensors").write_bytes(b"pre-step")
    (checkpoint / "policy_postprocessor_step_0_processor.safetensors").write_bytes(b"post-step")
    (checkpoint / "preprocessor_config.json").write_text("pre", encoding="utf-8")
    (checkpoint / "postprocessor_config.json").write_text("post", encoding="utf-8")
    dataset_root = tmp_path / "dataset"
    parquet_path = dataset_root / "data" / "chunk-000" / "file-000.parquet"
    _write_raw_lerobot_parquet(
        parquet_path,
        [
            {
                "episode_index": 202,
                "frame_index": 0,
                "observation.state": np.ones(20, dtype=np.float32).tolist(),
                "actions": np.ones(20, dtype=np.float32).tolist(),
                "observation.images.camera0": {"bytes": _png_bytes((1, 2, 3)), "path": ""},
            },
        ],
    )
    tree_path = dataset_root / ".cache" / "huggingface" / "trees" / "main.json"
    tree_path.parent.mkdir(parents=True)
    tree_path.write_text('{"commit_hash": "abc123"}', encoding="utf-8")
    config = {
        "observation": {"language_prompt": "fold"},
        "rename_map": {"observation.images.camera0": "observation.images.front"},
        "gripper": {"left_close_threshold": 0.4, "right_close_threshold": 0.6},
    }

    metadata = module.build_reproducibility_metadata(
        config_path=config_path,
        dataset_root=dataset_root,
        parquet_paths=(parquet_path,),
        checkpoint=checkpoint,
        config=config,
        selected_episodes=(202,),
        device="cpu",
        seed=99,
        versions={"torch": "2.fake", "lerobot": "0.fake", "peft": "1.fake"},
    )

    assert metadata["config"]["sha256"] == module.file_sha256(config_path)
    assert metadata["deployment"]["prompt"] == "fold"
    assert metadata["deployment"]["rename_map"] == config["rename_map"]
    assert metadata["deployment"]["gripper_thresholds"] == {"left": 0.4, "right": 0.6}
    assert metadata["dataset"]["revision"] == "abc123"
    assert metadata["dataset"]["parquet_files"][0]["sha256"] == module.file_sha256(parquet_path)
    assert metadata["checkpoint"]["files"]["adapter_config.json"]["sha256"] == module.file_sha256(
        checkpoint / "adapter_config.json"
    )
    assert metadata["checkpoint"]["files"]["preprocessor_config.json"]["sha256"] == module.file_sha256(
        checkpoint / "preprocessor_config.json"
    )
    assert metadata["checkpoint"]["files"]["policy_preprocessor.json"]["sha256"] == module.file_sha256(
        checkpoint / "policy_preprocessor.json"
    )
    assert metadata["checkpoint"]["files"]["policy_postprocessor.json"]["sha256"] == module.file_sha256(
        checkpoint / "policy_postprocessor.json"
    )
    assert metadata["checkpoint"]["files"]["policy_preprocessor_step_0_processor.safetensors"][
        "sha256"
    ] == module.file_sha256(checkpoint / "policy_preprocessor_step_0_processor.safetensors")
    assert metadata["checkpoint"]["files"]["policy_postprocessor_step_0_processor.safetensors"][
        "sha256"
    ] == module.file_sha256(checkpoint / "policy_postprocessor_step_0_processor.safetensors")

    shard_metadata = module.episode_shard_metadata(
        config_path=config_path,
        checkpoint=str(checkpoint),
        dataset_root=dataset_root,
        seed=99,
        device="cpu",
        episode_index=202,
        frame_count=1,
        horizon=20,
        state_dim=20,
        action_dim=20,
        image_keys=("observation.images.camera0",),
        reproducibility=metadata,
    )
    module.save_episode_shard(
        output_dir=tmp_path / "out",
        episode_index=202,
        pred=np.zeros((1, 20, 20), dtype=np.float32),
        gt=np.zeros((1, 20, 20), dtype=np.float32),
        valid=np.ones((1, 20), dtype=bool),
        frame_indices=np.asarray([0], dtype=np.int64),
        metadata=shard_metadata,
    )

    config_path.write_text("prompt: changed\n", encoding="utf-8")
    changed = module.build_reproducibility_metadata(
        config_path=config_path,
        dataset_root=dataset_root,
        parquet_paths=(parquet_path,),
        checkpoint=checkpoint,
        config=config,
        selected_episodes=(202,),
        device="cpu",
        seed=99,
        versions={"torch": "2.fake", "lerobot": "0.fake", "peft": "1.fake"},
    )
    changed_shard_metadata = module.episode_shard_metadata(
        config_path=config_path,
        checkpoint=str(checkpoint),
        dataset_root=dataset_root,
        seed=99,
        device="cpu",
        episode_index=202,
        frame_count=1,
        horizon=20,
        state_dim=20,
        action_dim=20,
        image_keys=("observation.images.camera0",),
        reproducibility=changed,
    )
    assert module.load_matching_episode_shard(
        tmp_path / "out", episode_index=202, expected_metadata=changed_shard_metadata
    ) is None


def test_dataset_revision_falls_back_to_huggingface_tree_filename(tmp_path: Path):
    module = _load_module()
    dataset_root = tmp_path / "dataset"
    tree_path = dataset_root / ".cache" / "huggingface" / "trees" / (
        "1234567890abcdef1234567890abcdef12345678.json"
    )
    tree_path.parent.mkdir(parents=True)
    tree_path.write_text('{"siblings": []}', encoding="utf-8")

    assert module._dataset_revision(dataset_root) == "1234567890abcdef1234567890abcdef12345678"


def test_write_reports_persists_reproducibility_metadata_to_json_and_npz(tmp_path: Path):
    module = _load_module()
    pred = np.zeros((1, 2, 20), dtype=np.float32)
    gt = np.zeros((1, 2, 20), dtype=np.float32)
    valid = np.asarray([[True, False]])
    _fill_identity_rotations(pred)
    _fill_identity_rotations(gt)
    pred[:, :, [9, 19]] = 1.0
    gt[:, :, [9, 19]] = 1.0
    metadata = {
        "dataset": {"revision": "abc123", "fingerprint": "dataset-fp"},
        "checkpoint": {"path": "/checkpoint", "fingerprint": "checkpoint-fp"},
        "config": {"sha256": "config-digest"},
        "deployment": {
            "prompt": "fold",
            "rename_map": {"observation.images.camera0": "observation.images.front"},
            "gripper_thresholds": {"left": 0.5, "right": 0.6},
        },
        "selected_episodes": [202],
        "device": "cpu",
        "seed": 123,
        "versions": {"torch": "2.fake", "lerobot": "0.fake", "peft": "1.fake"},
        "fps": 30.0,
    }

    paths = module.write_reports(
        output_dir=tmp_path,
        pred=pred,
        gt=gt,
        valid=valid,
        episode_indices=np.asarray([202], dtype=np.int64),
        frame_indices=np.asarray([0], dtype=np.int64),
        close_thresholds={"left": 0.5, "right": 0.6},
        metadata=metadata,
    )

    metrics = json.loads(paths["metrics"].read_text(encoding="utf-8"))
    assert metrics["run_metadata"] == metadata
    assert metrics["gripper_timeline"]["left"]["state"]["precision"] is None
    with np.load(paths["predictions"], allow_pickle=False) as archive:
        assert json.loads(str(archive["metadata_json"].item())) == metadata


def test_timeline_gripper_metrics_use_only_lead0_sequence_per_episode():
    module = _load_module()
    pred = np.ones((4, 3, 20), dtype=np.float32)
    gt = np.ones((4, 3, 20), dtype=np.float32)
    valid = np.ones((4, 3), dtype=bool)
    _fill_identity_rotations(pred)
    _fill_identity_rotations(gt)
    episode_indices = np.asarray([202, 202, 202, 202], dtype=np.int64)
    frame_indices = np.asarray([0, 1, 2, 3], dtype=np.int64)
    gt[:, 0, 9] = np.asarray([0.9, 0.4, 0.4, 0.4], dtype=np.float32)
    pred[:, 0, 9] = np.asarray([0.9, 0.9, 0.3, 0.3], dtype=np.float32)
    gt[:, 1:, 9] = 0.1
    pred[:, 1:, 9] = 0.1

    metrics = module.compute_timeline_gripper_metrics(
        pred=pred,
        gt=gt,
        valid=valid,
        episode_indices=episode_indices,
        frame_indices=frame_indices,
        close_thresholds={"left": 0.5, "right": 0.5},
        fps=30.0,
    )

    left = metrics["left"]
    assert left["state"]["precision"] == pytest.approx(1.0)
    assert left["state"]["recall"] == pytest.approx(2.0 / 3.0)
    assert left["state"]["f1"] == pytest.approx(0.8)
    assert left["close_event"]["support"] == 1
    assert left["close_event"]["predicted_positive_count"] == 1
    assert left["close_event"]["onset_timing_mae_frames"] == pytest.approx(1.0)
    assert left["close_event"]["onset_timing_mae_seconds"] == pytest.approx(1.0 / 30.0)

    chunk_metrics = module.compute_metrics(pred, gt, valid, close_thresholds={"left": 0.5, "right": 0.5})
    assert "chunk_forecast" in chunk_metrics["gripper"]["left"]
    assert "close_event" not in chunk_metrics["gripper"]["left"]


def test_predict_chunk_uses_torch_inference_mode_without_breaking_fake_runtime():
    module = _load_module()
    events: list[str] = []

    class InferenceMode:
        def __enter__(self):
            events.append("enter")

        def __exit__(self, exc_type, exc, traceback):
            events.append("exit")

    def predict(frame):
        events.append("predict")
        return frame

    runtime = SimpleNamespace(
        policy=SimpleNamespace(
            config=SimpleNamespace(chunk_size=2, action_feature={"shape": [20]}),
            predict_action_chunk=predict,
        ),
        preprocess=lambda frame: frame,
        postprocess=lambda frame: np.zeros((1, 2, 20), dtype=np.float32),
        prepare_frame=lambda observation: observation,
        torch=SimpleNamespace(inference_mode=lambda: InferenceMode()),
    )

    result = module._predict_chunk(runtime, {"observation.state": np.zeros(20, dtype=np.float32)})

    assert events == ["enter", "predict", "exit"]
    assert result.shape == (2, 20)


def test_run_offline_eval_rejects_non_20_20_20_contract_before_loading_dataset(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    module = _load_module()
    load_called = False
    runtime = SimpleNamespace(
        config_path=tmp_path / "deploy.yaml",
        config={"control": {"action_horizon": 20}, "observation": {"language_prompt": "fold"}},
        checkpoint="/checkpoint",
        device="cpu",
        state_dim=7,
        action_dim=20,
        horizon=20,
        dataset_image_keys=("observation.images.camera0",),
    )

    def load_raw(**kwargs):
        nonlocal load_called
        load_called = True
        return []

    monkeypatch.setattr(module, "load_eval_runtime", lambda *args, **kwargs: runtime)
    monkeypatch.setattr(module, "load_raw_lerobot_episodes", load_raw)

    with pytest.raises(ValueError, match="20/20/20"):
        module.run_offline_eval(
            config_path=tmp_path / "deploy.yaml",
            dataset_root=tmp_path / "dataset",
            episodes="202",
            output_dir=tmp_path / "out",
            device="cpu",
            seed=1,
        )
    assert load_called is False


def test_run_episode_sets_deterministic_torch_seed_per_frame(monkeypatch: pytest.MonkeyPatch):
    module = _load_module()
    seeds: list[int] = []
    prepared_frames: list[dict[str, object]] = []

    runtime = SimpleNamespace(
        policy=SimpleNamespace(
            config=SimpleNamespace(chunk_size=2, action_feature={"shape": [20]}),
            predict_action_chunk=lambda frame: frame,
        ),
        preprocess=lambda frame: frame,
        postprocess=lambda frame: SimpleNamespace(
            detach=lambda: SimpleNamespace(
                cpu=lambda: SimpleNamespace(
                    numpy=lambda: np.full((1, 2, 20), frame["observation.state"][0], dtype=np.float32)
                )
            )
        ),
        prepare_frame=lambda observation: prepared_frames.append(observation)
        or {"observation.state": observation["observation.state"]},
        torch=SimpleNamespace(manual_seed=lambda seed: seeds.append(int(seed))),
    )
    episode = module.RawEpisode(
        episode_index=3,
        frame_indices=np.asarray([4, 5], dtype=np.int64),
        states=np.asarray([[10.0, 0.0], [20.0, 0.0]], dtype=np.float32),
        actions=np.ones((2, 20), dtype=np.float32),
        observations=[
            {"observation.state": np.asarray([10.0, 0.0], dtype=np.float32)},
            {"observation.state": np.asarray([20.0, 0.0], dtype=np.float32)},
        ],
    )

    result = module.run_episode_inference(episode, runtime=runtime, seed=1000)

    assert seeds == [
        module.frame_seed(1000, episode_index=3, frame_index=4),
        module.frame_seed(1000, episode_index=3, frame_index=5),
    ]
    assert len(prepared_frames) == 2
    assert result.pred.shape == (2, 2, 20)
    np.testing.assert_allclose(result.pred[:, 0, 0], np.asarray([10.0, 20.0], dtype=np.float32))
    np.testing.assert_array_equal(result.valid, np.asarray([[True, True], [True, False]]))


def test_compute_metrics_uses_element_weighted_errors_and_window_slices():
    pred = np.zeros((2, 3, 20), dtype=np.float32)
    gt = np.zeros((2, 3, 20), dtype=np.float32)
    valid = np.asarray([[True, True, False], [True, False, False]])
    _fill_identity_rotations(pred)
    _fill_identity_rotations(gt)

    pred[0, 0, 0] = 1.0
    pred[0, 0, 1] = 2.0
    pred[0, 1, 0] = 3.0
    pred[1, 0, 19] = 4.0

    module = _load_module()
    metrics = module.compute_metrics(pred, gt, valid, close_thresholds={"left": 0.5, "right": 0.5})

    assert metrics["counts"]["valid_steps"] == 3
    assert metrics["counts"]["valid_action_elements"] == 60
    assert metrics["windows"]["full"]["mae"] == pytest.approx(10.0 / 60.0)
    assert metrics["windows"]["full"]["rmse"] == pytest.approx(np.sqrt(30.0 / 60.0))
    assert metrics["windows"]["first_1"]["mae"] == pytest.approx(7.0 / 40.0)
    assert metrics["windows"]["first_10"]["mae"] == pytest.approx(10.0 / 60.0)
    assert metrics["per_horizon"][0]["mae"] == pytest.approx(7.0 / 40.0)
    assert metrics["per_horizon"][1]["mae"] == pytest.approx(3.0 / 20.0)
    assert metrics["per_horizon"][2]["valid_steps"] == 0
    assert metrics["per_horizon"][0]["left_translation_mae"] == pytest.approx(3.0 / 6.0)
    assert metrics["per_horizon"][0]["right_gripper_mae"] == pytest.approx(4.0 / 2.0)


def test_compute_metrics_reports_rotation_6d_geodesic_error_in_degrees():
    pred = np.zeros((1, 1, 20), dtype=np.float32)
    gt = np.zeros((1, 1, 20), dtype=np.float32)
    valid = np.asarray([[True]])
    _fill_identity_rotations(pred)
    _fill_identity_rotations(gt)

    identity = _rotation_6d_from_matrix(np.eye(3, dtype=np.float64))
    yaw_ninety = _rotation_6d_from_matrix(_rotation_z_matrix(90.0))
    pred[0, 0, 3:9] = yaw_ninety
    pred[0, 0, 13:19] = identity
    gt[0, 0, 3:9] = identity
    gt[0, 0, 13:19] = identity

    module = _load_module()
    metrics = module.compute_metrics(pred, gt, valid, close_thresholds={"left": 0.5, "right": 0.5})

    assert metrics["windows"]["full"]["left_rotation_geodesic_deg"] == pytest.approx(90.0)
    assert metrics["windows"]["full"]["right_rotation_geodesic_deg"] == pytest.approx(0.0)
    assert metrics["per_horizon"][0]["left_rotation_geodesic_deg"] == pytest.approx(90.0)


def test_compute_metrics_reports_gripper_close_state_and_event_metrics():
    pred = np.zeros((4, 4, 20), dtype=np.float32)
    gt = np.zeros((4, 4, 20), dtype=np.float32)
    valid = np.ones((4, 4), dtype=bool)
    _fill_identity_rotations(pred)
    _fill_identity_rotations(gt)

    gt[:, :, 9] = np.asarray(
        [
            [0.9, 0.4, 0.3, 0.3],
            [0.9, 0.8, 0.7, 0.6],
            [0.9, 0.4, 0.4, 0.4],
            [0.9, 0.4, 0.4, 0.4],
        ],
        dtype=np.float32,
    )
    pred[:, :, 9] = np.asarray(
        [
            [0.9, 0.2, 0.2, 0.2],
            [0.9, 0.4, 0.4, 0.4],
            [0.9, 0.7, 0.4, 0.4],
            [0.8, 0.7, 0.7, 0.7],
        ],
        dtype=np.float32,
    )
    gt[:, :, 19] = 1.0
    pred[:, :, 19] = 1.0

    module = _load_module()
    metrics = module.compute_metrics(pred, gt, valid, close_thresholds={"left": 0.5, "right": 0.5})

    state_metrics = metrics["gripper"]["left"]["state"]
    assert state_metrics["precision"] == pytest.approx(5.0 / 8.0)
    assert state_metrics["recall"] == pytest.approx(5.0 / 9.0)
    assert state_metrics["f1"] == pytest.approx(10.0 / 17.0)

    event_metrics = metrics["gripper"]["left"]["chunk_forecast"]["close_event"]
    assert event_metrics["precision"] == pytest.approx(2.0 / 3.0)
    assert event_metrics["recall"] == pytest.approx(2.0 / 3.0)
    assert event_metrics["f1"] == pytest.approx(2.0 / 3.0)
    assert event_metrics["onset_timing_mae"] == pytest.approx(0.5)


def test_compute_metrics_persists_first_10_gripper_forecast_events():
    pred = np.ones((2, 12, 20), dtype=np.float32)
    gt = np.ones((2, 12, 20), dtype=np.float32)
    valid = np.ones((2, 12), dtype=bool)
    _fill_identity_rotations(pred)
    _fill_identity_rotations(gt)

    # Row 0 closes inside the deployed first-10 window. Row 1 closes only
    # outside that window, while the prediction closes one step too early.
    gt[0, 8:, 9] = 0.2
    pred[0, 9:, 9] = 0.2
    gt[1, 10:, 9] = 0.2
    pred[1, 9:, 9] = 0.2

    module = _load_module()
    metrics = module.compute_metrics(pred, gt, valid, close_thresholds={"left": 0.5, "right": 0.5})

    first_10 = metrics["gripper"]["left"]["first_10_forecast"]["close_event"]
    assert first_10["support"] == 1
    assert first_10["predicted_positive_count"] == 2
    assert first_10["precision"] == pytest.approx(0.5)
    assert first_10["recall"] == pytest.approx(1.0)
    assert first_10["f1"] == pytest.approx(2.0 / 3.0)


def test_timeline_lead0_slice_excludes_invalid_terminal_rows():
    module = _load_module()
    pred = np.zeros((3, 2, 20), dtype=np.float32)
    gt = np.zeros((3, 2, 20), dtype=np.float32)
    valid = np.asarray([[True, True], [True, False], [False, False]])
    frames = np.asarray([7, 8, 9], dtype=np.int64)
    pred[:, 0, 0] = np.asarray([0.1, 0.2, 99.0])
    gt[:, 0, 0] = np.asarray([0.1, 0.2, 0.0])

    timeline_pred, timeline_gt, timeline_frames = module._timeline_lead0_slice(
        pred, gt, valid, frames
    )

    np.testing.assert_array_equal(timeline_frames, np.asarray([7, 8]))
    np.testing.assert_allclose(timeline_pred[:, 0, 0], np.asarray([0.1, 0.2]))
    np.testing.assert_allclose(timeline_gt[:, 0, 0], np.asarray([0.1, 0.2]))


def test_write_reports_writes_npz_json_csv_and_plot_artifacts(tmp_path: Path):
    pred = np.zeros((2, 3, 20), dtype=np.float32)
    gt = np.zeros((2, 3, 20), dtype=np.float32)
    valid = np.asarray([[True, True, False], [True, False, False]])
    _fill_identity_rotations(pred)
    _fill_identity_rotations(gt)
    pred[0, 0, 0] = 1.0
    pred[0, 0, 9] = 0.2
    gt[0, 0, 9] = 0.8

    module = _load_module()
    paths = module.write_reports(
        output_dir=tmp_path,
        pred=pred,
        gt=gt,
        valid=valid,
        episode_indices=np.asarray([202, 202], dtype=np.int64),
        frame_indices=np.asarray([0, 1], dtype=np.int64),
        close_thresholds={"left": 0.5, "right": 0.5},
    )

    assert set(paths) == {
        "predictions",
        "metrics",
        "per_horizon",
        "episode_metrics",
        "action_error_heatmap",
        "group_error_by_horizon",
        "gripper_timelines",
        "action_timelines",
    }
    assert all(path.exists() for key, value in paths.items() for path in (value if isinstance(value, list) else [value]))

    archive = np.load(paths["predictions"])
    np.testing.assert_array_equal(archive["episode_indices"], np.asarray([202, 202]))
    np.testing.assert_array_equal(archive["frame_indices"], np.asarray([0, 1]))
    np.testing.assert_array_equal(archive["valid"], valid)

    metrics = json.loads(paths["metrics"].read_text(encoding="utf-8"))
    assert metrics["counts"]["valid_steps"] == 3
    assert "per_episode" in metrics
    assert "full" in metrics["windows"]

    with paths["per_horizon"].open(newline="", encoding="utf-8") as file:
        per_horizon_rows = list(csv.DictReader(file))
    assert per_horizon_rows[0].keys() == {
        "lead_step",
        "valid_steps",
        "mae",
        "rmse",
        "left_translation_mae",
        "right_translation_mae",
        "left_rotation_geodesic_deg",
        "right_rotation_geodesic_deg",
        "left_gripper_mae",
        "right_gripper_mae",
    }
    assert per_horizon_rows[0]["lead_step"] == "0"

    with paths["episode_metrics"].open(newline="", encoding="utf-8") as file:
        episode_rows = list(csv.DictReader(file))
    assert len(episode_rows) == 1
    assert episode_rows[0]["episode_index"] == "202"

    assert [path.name for path in paths["gripper_timelines"]] == [
        "gripper_timeline_episode_202.png"
    ]
    assert [path.name for path in paths["action_timelines"]] == [
        "action_timeline_episode_202.png"
    ]
