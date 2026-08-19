from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
TRAIN_ROOT = ROOT / "train_pi05_frs"

import jax.numpy as jnp
import numpy as np
import pytest

from train_pi05_frs.pi05_cache import cache, policy_inputs, prepare
from train_pi05_frs.pi05_cache import source_model
from lerobot.policies.pi05_jax import transforms
from lerobot.policies.pi05_jax.normalize import NormStats
from lerobot.datasets import tactile_cache


class FakeMetadata:
    total_episodes = 4
    episodes = [
        {"dataset_from_index": 0, "dataset_to_index": 8},
        {"dataset_from_index": 8, "dataset_to_index": 16},
        {"dataset_from_index": 16, "dataset_to_index": 24},
        {"dataset_from_index": 24, "dataset_to_index": 32},
    ]


def make_stats(dim: int) -> NormStats:
    return NormStats(
        mean=np.zeros(dim, dtype=np.float32),
        std=np.ones(dim, dtype=np.float32),
        q01=-np.ones(dim, dtype=np.float32),
        q99=np.ones(dim, dtype=np.float32),
    )


def expected_encoder_fingerprint(checkpoint_dir: Path, params_name: str) -> str:
    digest = hashlib.sha256()
    for relative_path in ("checkpoint.json", params_name):
        path = checkpoint_dir / relative_path
        digest.update(relative_path.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


@pytest.mark.parametrize("checkpoint_format", ["legacy", "metadata_params_file"])
def test_tactile_fingerprint_metadata_and_reader_support_both_checkpoint_formats(
    tmp_path: Path,
    checkpoint_format: str,
) -> None:
    encoder_dir = tmp_path / "encoder"
    encoder_dir.mkdir()
    if checkpoint_format == "legacy":
        params_name = "params.npz"
        metadata_payload: dict[str, object] = {
            "version": 3,
            "parameter_paths": ["encoder/kernel"],
        }
        (encoder_dir / "checkpoint.json").write_text(
            json.dumps(metadata_payload, sort_keys=True), encoding="utf-8"
        )
        np.savez(encoder_dir / params_name, p00000=np.arange(4, dtype=np.float32))
    else:
        from train_encoder.utils.checkpoint import save_checkpoint
        from train_encoder.utils.model import TactileClipConfig

        save_checkpoint(
            encoder_dir,
            {"encoder": {"kernel": jnp.arange(4, dtype=jnp.float32)}},
            epoch=1,
            metrics={"loss": 0.5},
            model_id="unit-test",
            config=TactileClipConfig(),
        )
        metadata_payload = json.loads((encoder_dir / "checkpoint.json").read_text(encoding="utf-8"))
        params_name = metadata_payload["params_file"]
        assert isinstance(params_name, str) and params_name.startswith("params-")

    fingerprint = tactile_cache.tactile_encoder_fingerprint(encoder_dir)

    assert fingerprint == expected_encoder_fingerprint(encoder_dir, params_name)
    cache_dir = tmp_path / "tactile-cache"
    cache_dir.mkdir()
    metadata = tactile_cache.create_tactile_cache_metadata(
        repo_id="org/demo",
        revision="rev-1",
        dataset_root=tmp_path / "dataset",
        total_frames=2,
        tactile_keys=("left", "right"),
        source_tactile_keys=("observation.left", "observation.right"),
        embedding_dim=3,
        image_size=224,
        dtype="float32",
        encoder_path=encoder_dir,
        completed_frames=2,
        status="complete",
    )
    assert metadata["encoder_sha256"] == fingerprint
    tactile_cache.atomic_write_json(cache_dir / tactile_cache.TACTILE_METADATA_NAME, metadata)
    expected_embeddings = np.arange(12, dtype=np.float32).reshape(2, 2, 3)
    np.save(cache_dir / tactile_cache.TACTILE_EMBEDDINGS_NAME, expected_embeddings)

    reader = tactile_cache.TactileEmbeddingCache(
        cache_dir,
        repo_id="org/demo",
        revision="rev-1",
        total_frames=2,
        tactile_keys=("left", "right"),
        source_tactile_keys=("observation.left", "observation.right"),
        embedding_dim=3,
        image_size=224,
        encoder_path=encoder_dir,
    )

    np.testing.assert_array_equal(reader.get_many([1, 0]), expected_embeddings[[1, 0]])


@pytest.mark.parametrize(
    ("metadata_payload", "missing_name"),
    [
        ({"version": 3}, "params.npz"),
        ({"version": 4, "params_file": "params-missing.npz"}, "params-missing.npz"),
    ],
)
def test_tactile_fingerprint_rejects_missing_params_file(
    tmp_path: Path,
    metadata_payload: dict[str, object],
    missing_name: str,
) -> None:
    encoder_dir = tmp_path / "encoder"
    encoder_dir.mkdir()
    (encoder_dir / "checkpoint.json").write_text(json.dumps(metadata_payload), encoding="utf-8")

    with pytest.raises(FileNotFoundError, match=missing_name):
        tactile_cache.tactile_encoder_fingerprint(encoder_dir)


@pytest.mark.parametrize("bad_params_file", [None, 7, [], "", "   "])
def test_tactile_fingerprint_rejects_invalid_params_file(
    tmp_path: Path,
    bad_params_file: object,
) -> None:
    encoder_dir = tmp_path / "encoder"
    encoder_dir.mkdir()
    (encoder_dir / "checkpoint.json").write_text(
        json.dumps({"version": 4, "params_file": bad_params_file}), encoding="utf-8"
    )

    with pytest.raises(ValueError, match="params_file.*non-empty relative path"):
        tactile_cache.tactile_encoder_fingerprint(encoder_dir)


def test_tactile_fingerprint_rejects_params_path_escape(tmp_path: Path) -> None:
    encoder_dir = tmp_path / "encoder"
    encoder_dir.mkdir()
    outside = tmp_path / "outside.npz"
    np.savez(outside, p00000=np.ones(1, dtype=np.float32))
    symlink = encoder_dir / "params-link.npz"
    symlink.symlink_to(outside)

    for params_file in ("../outside.npz", str(outside), symlink.name):
        (encoder_dir / "checkpoint.json").write_text(
            json.dumps({"version": 4, "params_file": params_file}), encoding="utf-8"
        )
        tactile_cache.tactile_encoder_fingerprint.cache_clear()
        with pytest.raises(ValueError, match="params_file.*checkpoint directory"):
            tactile_cache.tactile_encoder_fingerprint(encoder_dir)


def test_tactile_fingerprint_requires_params_to_be_regular_file(tmp_path: Path) -> None:
    encoder_dir = tmp_path / "encoder"
    encoder_dir.mkdir()
    (encoder_dir / "params-directory").mkdir()
    (encoder_dir / "checkpoint.json").write_text(
        json.dumps({"version": 4, "params_file": "params-directory"}), encoding="utf-8"
    )

    with pytest.raises(ValueError, match="params_file.*regular file"):
        tactile_cache.tactile_encoder_fingerprint(encoder_dir)


def test_training_lerobot_is_private() -> None:
    import lerobot

    path = Path(lerobot.__file__).resolve()
    assert TRAIN_ROOT / "src" in path.parents


def test_runtime_modules_resolve_only_inside_standalone_project() -> None:
    import lerobot.datasets.dataset_metadata as dataset_metadata
    import lerobot.policies.pi05_jax.policy_config as policy_config
    import train_pi05_frs.pi05_cache.cache as cache_module
    import train_pi05_frs.pi05_cache.prepare as prepare_module

    source_checkout = Path("/home/typhon/FRS_Tact-pi05-frs-jax")
    for module in (dataset_metadata, policy_config, cache_module, prepare_module):
        path = Path(module.__file__).resolve()
        assert TRAIN_ROOT in path.parents
        assert source_checkout not in path.parents


def test_record_selection_is_episode_disjoint_trimmed_and_strided() -> None:
    records, train_episodes, val_episodes = cache.build_records(
        FakeMetadata(),
        val_fraction=0.25,
        split_seed=7,
        frame_stride=2,
        max_episodes=None,
        max_samples=None,
        action_horizon=2,
        drop_tail_action_chunks=1,
    )

    assert set(train_episodes).isdisjoint(val_episodes)
    assert {record.episode_index for record in records if record.split == "train"} == set(
        train_episodes
    )
    assert {record.episode_index for record in records if record.split == "val"} == set(
        val_episodes
    )
    assert [record.dataset_index for record in records] == [0, 2, 4, 8, 10, 12, 16, 18, 20, 24, 26, 28]


def test_twenty_dimensional_actions_are_padded_to_model_dimension() -> None:
    action = np.arange(3 * 20, dtype=np.float32).reshape(3, 20)
    result = transforms.PadStatesAndActions(32)(
        {"state": np.arange(20, dtype=np.float32), "actions": action}
    )

    assert result["state"].shape == (32,)
    assert result["actions"].shape == (3, 32)
    np.testing.assert_array_equal(result["actions"][:, :20], action)
    np.testing.assert_array_equal(result["actions"][:, 20:], 0)


def test_camera_map_rejects_unknown_pi05_slot_before_dataset_access() -> None:
    with pytest.raises(ValueError, match="camera_map keys"):
        policy_inputs.Pi05SampleProcessor(
            dataset_repo_id="unused/dataset",
            camera_map={"not_a_pi05_camera": "observation.images.camera0"},
            state_stats=make_stats(20),
            action_stats=make_stats(20),
        )


def test_norm_stats_reject_dimensions_wider_than_dataset() -> None:
    with pytest.raises(ValueError, match="wider"):
        policy_inputs._match_norm_stats_dim(make_stats(21), 20, label="action")


def test_norm_stats_pad_missing_dimensions_as_identity() -> None:
    padded = policy_inputs._match_norm_stats_dim(make_stats(14), 20, label="state")

    np.testing.assert_array_equal(padded.mean[14:], 0)
    np.testing.assert_array_equal(padded.std[14:], 1)
    np.testing.assert_array_equal(padded.q01[14:], -1)
    np.testing.assert_array_equal(padded.q99[14:], 1)


def test_inference_noise_is_deterministic_per_seed_and_dataset_index() -> None:
    first = prepare.deterministic_noise([11, 12], (3, 4), seed=17)
    repeated = prepare.deterministic_noise([11, 12], (3, 4), seed=17)
    other_seed = prepare.deterministic_noise([11, 12], (3, 4), seed=18)

    np.testing.assert_array_equal(first, repeated)
    assert not np.array_equal(first, other_seed)


def test_inversion_mse_matches_per_sample_squared_error() -> None:
    x_base = jnp.asarray([[[0.0, 2.0]], [[3.0, 5.0]]], dtype=jnp.float32)
    noise = jnp.asarray([[[1.0, 0.0]], [[1.0, 1.0]]], dtype=jnp.float32)

    result = source_model.inversion_mse(x_base, noise)

    np.testing.assert_allclose(result, np.asarray([2.5, 10.0], dtype=np.float32))


@pytest.mark.parametrize(
    "integrate",
    [
        source_model.euler_integrate_velocity,
        source_model.fireflow_integrate_velocity,
        source_model.slerpflow_integrate_velocity,
    ],
)
def test_reverse_solvers_preserve_shape_and_finiteness(integrate: object) -> None:
    initial = jnp.arange(1, 25, dtype=jnp.float32).reshape(2, 3, 4) / 24.0

    result = integrate(
        lambda value, time: 0.1 * value + time[:, None, None], initial, num_steps=4
    )

    assert result.shape == initial.shape
    assert bool(jnp.all(jnp.isfinite(result)))


def test_slerpflow_integrates_constant_radial_velocity() -> None:
    initial = jnp.ones((2, 3, 4), dtype=jnp.float32)
    direction = initial / jnp.linalg.norm(initial, axis=(1, 2), keepdims=True)

    result = source_model.slerpflow_integrate_velocity(
        lambda value, time: direction,
        initial,
        num_steps=10,
    )

    np.testing.assert_allclose(result, initial + direction, atol=1e-5)


def test_prepare_cache_signature_and_return_type_are_standalone_contract() -> None:
    signature = inspect.signature(prepare.prepare_cache)

    assert all(parameter.kind is inspect.Parameter.KEYWORD_ONLY for parameter in signature.parameters.values())
    assert signature.return_annotation in (Path, "pathlib.Path")
    assert signature.parameters["checkpoint_dir"].annotation in (str, "str")


def test_prepare_cache_records_provenance_resumes_and_skips_completed_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint = tmp_path / "checkpoint"
    (checkpoint / "params").mkdir(parents=True)
    (checkpoint / "params" / "weights").write_bytes(b"weights")
    cache_dir = tmp_path / "cache"
    records = [
        cache.SampleRecord(dataset_index=0, episode_index=0, split="train"),
        cache.SampleRecord(dataset_index=1, episode_index=1, split="val"),
    ]

    class FakeEvalModel:
        dataset_repo_id = "org/demo"
        dataset_root = tmp_path / "dataset"
        dataset_revision = "rev-1"
        action_key = "action"
        action_horizon = 2
        action_dim = 4
        state_stats = make_stats(3)
        model = object()

        def __init__(self, checkpoint_dir: str, **kwargs: object) -> None:
            assert checkpoint_dir == str(checkpoint)

    class RuntimeMetadata:
        camera_keys = ("observation.images.camera0",)
        total_episodes = 2
        episodes = [
            {"dataset_from_index": 0, "dataset_to_index": 1},
            {"dataset_from_index": 1, "dataset_to_index": 2},
        ]

        def __init__(self, *args: object, **kwargs: object) -> None:
            pass

    class FakeDataset:
        def __init__(self, *args: object, **kwargs: object) -> None:
            pass

        def select_columns(self, action_key: str) -> object:
            assert action_key == "action"
            return object()

    monkeypatch.setattr(prepare, "load_norm_stats_or_raise", lambda *_: (make_stats(3), make_stats(3)))
    monkeypatch.setattr(prepare, "Pi05EvalModel", FakeEvalModel)
    monkeypatch.setattr(prepare, "LeRobotDatasetMetadata", RuntimeMetadata)
    monkeypatch.setattr(prepare, "LeRobotDataset", FakeDataset)
    monkeypatch.setattr(
        prepare,
        "build_records",
        lambda *_args, **_kwargs: (records, (0,), (1,)),
    )

    arrays = cache.create_cache_arrays(
        cache_dir,
        records,
        action_horizon=2,
        action_dim=4,
        state_dim=3,
    )
    arrays["target"][0] = 11.0
    arrays["x_base"][0] = 12.0
    arrays["gt_action"][0] = 13.0
    arrays["state"][0] = 14.0
    arrays["inversion_mse"][:] = np.asarray([0.0, 0.0], dtype=np.float32)
    cache.flush_arrays(arrays)

    monkeypatch.setattr(
        prepare,
        "_load_observation_batch",
        lambda *_args, **_kwargs: (
            SimpleNamespace(state=jnp.asarray([[1.0, 2.0, 3.0]], dtype=jnp.float32)),
            jnp.full((1, 2, 4), 23.0, dtype=jnp.float32),
        ),
    )
    sample_calls: list[np.ndarray] = []

    def fake_sample_and_reverse(*args: object, **kwargs: object) -> tuple[object, object]:
        sample_calls.append(np.asarray(args[2]))
        return (
            jnp.full((1, 2, 4), 21.0, dtype=jnp.float32),
            jnp.full((1, 2, 4), 22.0, dtype=jnp.float32),
        )

    monkeypatch.setattr(prepare, "sample_and_reverse", fake_sample_and_reverse)

    expected_configuration = {
        "checkpoint_dir": str(checkpoint.resolve()),
        "checkpoint_fingerprint": prepare._checkpoint_fingerprint(checkpoint),
        "dataset_repo_id": "org/demo",
        "dataset_root": str((tmp_path / "dataset").resolve()),
        "dataset_revision": "rev-1",
        "action_key": "action",
        "rename_map": None,
        "camera_map": {"base_0_rgb": "observation.images.camera0"},
        "norm_stats_dir": str(tmp_path / "stats"),
        "norm_stats_asset_id": "asset",
        "use_quantile_norm": True,
        "base_model": "pi0.5",
        "action_dim": 4,
        "action_horizon": 2,
        "paligemma_variant": "gemma_2b",
        "action_expert_variant": "gemma_300m",
        "model_sample_steps": 2,
        "reverse_steps": 3,
        "reverse_solver": "euler",
        "inference_seed": 5,
        "split_seed": 7,
        "val_fraction": 0.5,
        "frame_stride": 1,
        "max_episodes": None,
        "max_samples": None,
        "drop_tail_action_chunks": 0,
    }
    manifest = {
        "version": cache.CACHE_VERSION,
        "status": "incomplete",
        "completed_samples": 1,
        "sample_count": 2,
        "action_horizon": 2,
        "action_dim": 4,
        "state_dim": 3,
        "configuration": expected_configuration,
        "records_sha256": cache.records_digest(records),
    }
    cache.atomic_write_json(cache_dir / cache.MANIFEST_NAME, manifest)

    kwargs = dict(
        checkpoint_dir=str(checkpoint),
        cache_dir=cache_dir,
        dataset_repo_id="org/demo",
        dataset_root=tmp_path / "dataset",
        dataset_revision="rev-1",
        action_key="action",
        rename_map=None,
        camera_map={"base_0_rgb": "observation.images.camera0"},
        norm_stats_dir=tmp_path / "stats",
        norm_stats_asset_id="asset",
        use_quantile_norm=True,
        action_dim=4,
        action_horizon=2,
        model_sample_steps=2,
        reverse_steps=3,
        reverse_solver="euler",
        batch_size=1,
        load_workers=1,
        inference_seed=5,
        split_seed=7,
        val_fraction=0.5,
        frame_stride=1,
        max_episodes=None,
        max_samples=None,
        drop_tail_action_chunks=0,
    )

    assert prepare.prepare_cache(**kwargs) == cache_dir
    assert cache.load_manifest(cache_dir)["status"] == "complete"
    resumed = cache.open_cache_arrays(cache_dir)
    np.testing.assert_array_equal(resumed["target"][0], 11.0)
    np.testing.assert_array_equal(resumed["x_base"][0], 12.0)
    np.testing.assert_array_equal(resumed["gt_action"][0], 13.0)
    np.testing.assert_array_equal(resumed["state"][0], 14.0)
    np.testing.assert_array_equal(resumed["target"][1], 21.0)
    np.testing.assert_array_equal(resumed["x_base"][1], 22.0)
    np.testing.assert_array_equal(resumed["gt_action"][1], 23.0)
    np.testing.assert_array_equal(resumed["state"][1], [1.0, 2.0, 3.0])
    assert len(sample_calls) == 1
    assert prepare.prepare_cache(**kwargs) == cache_dir
    assert len(sample_calls) == 1
    with pytest.raises(ValueError, match="different inputs"):
        prepare.prepare_cache(**{**kwargs, "inference_seed": 99})


def _prepare_provenance_case(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    status: str,
) -> tuple[dict[str, object], dict[str, object]]:
    checkpoint = tmp_path / "checkpoint"
    (checkpoint / "params").mkdir(parents=True)
    (checkpoint / "params" / "weights").write_bytes(b"weights")
    cache_dir = tmp_path / "cache"
    records = [
        cache.SampleRecord(dataset_index=0, episode_index=0, split="train"),
        cache.SampleRecord(dataset_index=1, episode_index=1, split="val"),
    ]

    class FakeEvalModel:
        dataset_repo_id = "org/demo"
        dataset_root = tmp_path / "dataset"
        dataset_revision = "rev-1"
        action_key = "action"
        state_stats = make_stats(3)
        model = object()

        def __init__(
            self,
            checkpoint_dir: str,
            *,
            action_dim: int,
            action_horizon: int,
            **kwargs: object,
        ) -> None:
            assert checkpoint_dir == str(checkpoint)
            self.action_dim = action_dim
            self.action_horizon = action_horizon

    class RuntimeMetadata:
        camera_keys = ("observation.images.camera0",)
        total_episodes = 2
        episodes = [
            {"dataset_from_index": 0, "dataset_to_index": 1},
            {"dataset_from_index": 1, "dataset_to_index": 2},
        ]

        def __init__(self, *args: object, **kwargs: object) -> None:
            pass

    class MustNotReadDataset:
        def __init__(self, *args: object, **kwargs: object) -> None:
            pass

        def select_columns(self, action_key: str) -> object:
            raise AssertionError("provenance mismatch must fail before dataset access")

    monkeypatch.setattr(
        prepare, "load_norm_stats_or_raise", lambda *_: (make_stats(3), make_stats(3))
    )
    monkeypatch.setattr(prepare, "Pi05EvalModel", FakeEvalModel)
    monkeypatch.setattr(prepare, "LeRobotDatasetMetadata", RuntimeMetadata)
    monkeypatch.setattr(prepare, "LeRobotDataset", MustNotReadDataset)
    monkeypatch.setattr(
        prepare,
        "build_records",
        lambda *_args, **_kwargs: (records, (0,), (1,)),
    )

    arrays = cache.create_cache_arrays(
        cache_dir,
        records,
        action_horizon=2,
        action_dim=4,
        state_dim=3,
    )
    cache.flush_arrays(arrays)
    kwargs: dict[str, object] = {
        "checkpoint_dir": str(checkpoint),
        "cache_dir": cache_dir,
        "dataset_repo_id": "org/demo",
        "dataset_root": tmp_path / "dataset",
        "dataset_revision": "rev-1",
        "action_key": "action",
        "rename_map": None,
        "camera_map": {"base_0_rgb": "observation.images.camera0"},
        "norm_stats_dir": tmp_path / "stats",
        "norm_stats_asset_id": "asset",
        "use_quantile_norm": True,
        "action_dim": 4,
        "action_horizon": 2,
        "paligemma_variant": "gemma_2b",
        "action_expert_variant": "gemma_300m",
        "model_sample_steps": 2,
        "reverse_steps": 3,
        "reverse_solver": "euler",
        "batch_size": 1,
        "load_workers": 1,
        "inference_seed": 5,
        "split_seed": 7,
        "val_fraction": 0.5,
        "frame_stride": 1,
        "max_episodes": None,
        "max_samples": None,
        "drop_tail_action_chunks": 0,
    }
    configuration = {
        "checkpoint_dir": str(checkpoint.resolve()),
        "checkpoint_fingerprint": prepare._checkpoint_fingerprint(checkpoint),
        "dataset_repo_id": "org/demo",
        "dataset_root": str((tmp_path / "dataset").resolve()),
        "dataset_revision": "rev-1",
        "action_key": "action",
        "rename_map": None,
        "camera_map": {"base_0_rgb": "observation.images.camera0"},
        "norm_stats_dir": str(tmp_path / "stats"),
        "norm_stats_asset_id": "asset",
        "use_quantile_norm": True,
        "base_model": "pi0.5",
        "action_dim": 4,
        "action_horizon": 2,
        "paligemma_variant": "gemma_2b",
        "action_expert_variant": "gemma_300m",
        "model_sample_steps": 2,
        "reverse_steps": 3,
        "reverse_solver": "euler",
        "inference_seed": 5,
        "split_seed": 7,
        "val_fraction": 0.5,
        "frame_stride": 1,
        "max_episodes": None,
        "max_samples": None,
        "drop_tail_action_chunks": 0,
    }
    manifest: dict[str, object] = {
        "version": cache.CACHE_VERSION,
        "status": status,
        "completed_samples": len(records) if status == "complete" else 1,
        "sample_count": len(records),
        "action_horizon": 2,
        "action_dim": 4,
        "state_dim": 3,
        "configuration": configuration,
        "records_sha256": cache.records_digest(records),
    }
    return kwargs, manifest


@pytest.mark.parametrize("status", ["complete", "incomplete"])
@pytest.mark.parametrize(
    ("field_path", "bad_value", "diagnostic"),
    [
        (("configuration", "action_dim"), 5, "configuration.action_dim"),
        (("configuration", "action_horizon"), 3, "configuration.action_horizon"),
        (("configuration", "paligemma_variant"), "dummy", "configuration.paligemma_variant"),
        (("configuration", "action_expert_variant"), "dummy", "configuration.action_expert_variant"),
        (("action_dim",), 5, "action_dim"),
        (("action_horizon",), 3, "action_horizon"),
        (("state_dim",), 4, "state_dim"),
    ],
)
def test_prepare_cache_rejects_every_output_provenance_mismatch_before_skip_or_resume(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    status: str,
    field_path: tuple[str, ...],
    bad_value: object,
    diagnostic: str,
) -> None:
    kwargs, manifest = _prepare_provenance_case(tmp_path, monkeypatch, status=status)
    target: dict[str, object] = manifest
    for part in field_path[:-1]:
        target = target[part]  # type: ignore[assignment]
    target[field_path[-1]] = bad_value
    cache.atomic_write_json(Path(kwargs["cache_dir"]) / cache.MANIFEST_NAME, manifest)

    with pytest.raises(ValueError, match=diagnostic.replace(".", r"\.")):
        prepare.prepare_cache(**kwargs)  # type: ignore[arg-type]


@pytest.mark.parametrize("status", ["complete", "incomplete"])
@pytest.mark.parametrize("array_name", sorted(cache.ARRAY_FILENAMES))
def test_prepare_cache_rejects_every_cache_array_shape_before_skip_or_resume(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    status: str,
    array_name: str,
) -> None:
    kwargs, manifest = _prepare_provenance_case(tmp_path, monkeypatch, status=status)
    cache_dir = Path(kwargs["cache_dir"])
    filename = cache.ARRAY_FILENAMES[array_name]
    with (cache_dir / filename).open("wb") as file:
        np.save(file, np.zeros((1,), dtype=np.float32), allow_pickle=False)
    cache.atomic_write_json(cache_dir / cache.MANIFEST_NAME, manifest)

    with pytest.raises(ValueError, match=rf"arrays\.{array_name}\.shape"):
        prepare.prepare_cache(**kwargs)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("status", "completed_samples", "sample_count", "diagnostic"),
    [
        ("unknown", 1, 2, "status"),
        ("complete", 1, 2, "completed_samples"),
        ("incomplete", 2, 2, "completed_samples"),
        ("incomplete", -1, 2, "completed_samples"),
        ("incomplete", 3, 2, "completed_samples"),
        ("incomplete", 1.0, 2, "completed_samples"),
        ("complete", 2, 3, "sample_count"),
        ("complete", 2, True, "sample_count"),
    ],
)
def test_prepare_cache_rejects_inconsistent_progress_before_skip_or_resume(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    status: str,
    completed_samples: object,
    sample_count: object,
    diagnostic: str,
) -> None:
    kwargs, manifest = _prepare_provenance_case(tmp_path, monkeypatch, status="complete")
    manifest.update(
        {
            "status": status,
            "completed_samples": completed_samples,
            "sample_count": sample_count,
        }
    )
    cache.atomic_write_json(Path(kwargs["cache_dir"]) / cache.MANIFEST_NAME, manifest)

    with pytest.raises(ValueError, match=diagnostic):
        prepare.prepare_cache(**kwargs)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("status", "completed_samples", "sample_count", "diagnostic"),
    [
        ("unknown", 1, 2, "status"),
        ("complete", 1, 2, "completed_samples"),
        ("incomplete", 2, 2, "completed_samples"),
        ("incomplete", -1, 2, "completed_samples"),
        ("incomplete", 3, 2, "completed_samples"),
        ("incomplete", 1.0, 2, "completed_samples"),
        ("complete", 2, -1, "sample_count"),
        ("complete", 2, True, "sample_count"),
    ],
)
def test_load_manifest_rejects_inconsistent_progress(
    tmp_path: Path,
    status: str,
    completed_samples: object,
    sample_count: object,
    diagnostic: str,
) -> None:
    cache.atomic_write_json(
        tmp_path / cache.MANIFEST_NAME,
        {
            "version": cache.CACHE_VERSION,
            "status": status,
            "completed_samples": completed_samples,
            "sample_count": sample_count,
        },
    )

    with pytest.raises(ValueError, match=diagnostic):
        cache.load_manifest(tmp_path, require_complete=False)


def test_cache_arrays_preserve_names_and_dtypes(tmp_path: Path) -> None:
    records = [
        cache.SampleRecord(3, 0, "train"),
        cache.SampleRecord(8, 1, "val"),
    ]
    arrays = cache.create_cache_arrays(
        tmp_path,
        records,
        action_horizon=2,
        action_dim=4,
        state_dim=3,
    )

    assert (tmp_path / "predicted_actions.npy").is_file()
    assert arrays["x_base"].dtype == np.float32
    assert arrays["dataset_index"].dtype == np.int64
    assert arrays["split"].dtype == np.uint8
