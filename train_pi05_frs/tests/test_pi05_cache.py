from __future__ import annotations

import inspect
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
