import io
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pytest
import torch
import torchvision
import zarr
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from PIL import Image

from convert_pick_tube_lerobot_to_rdp_zarr import (
    DEFAULT_DATASETS,
    DEFAULT_DATASET_REPEATS,
    create_output,
    decode_images,
    parse_dataset_repeats,
    sha256_file_cached,
)
from reactive_diffusion_policy.common.replay_buffer import ReplayBuffer
from reactive_diffusion_policy.common.artifact_manifest import (
    build_normalizer_cache_signature,
)
from reactive_diffusion_policy.common.sampler import SequenceSampler
from reactive_diffusion_policy.dataset.real_image_tactile_latent_diffusion_dataset import (
    RealImageTactileLatentDiffusionDataset,
)
from reactive_diffusion_policy.dataset.real_image_tactile_dataset import RealImageTactileDataset
from reactive_diffusion_policy.model.common.normalizer import (
    LinearNormalizer,
    SingleFieldLinearNormalizer,
)
from reactive_diffusion_policy.model.vision.multi_image_obs_encoder import TrainOnlyTransform


def test_pick_tube_conversion_defaults_include_new_diverse_data():
    assert DEFAULT_DATASETS[-2:] == ("pick_tube_05", "pick_tube_06")
    assert parse_dataset_repeats(list(DEFAULT_DATASET_REPEATS)) == {
        "pick_tube_05": 2,
        "pick_tube_06": 2,
    }


def test_parallel_rgb_decode_preserves_row_order(tmp_path):
    payloads = []
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (12, 34, 56)]
    for color in colors:
        buffer = io.BytesIO()
        Image.new("RGB", (4, 3), color=color).save(buffer, format="PNG")
        payloads.append({"bytes": buffer.getvalue()})

    serial = decode_images(payloads, tmp_path)
    with ThreadPoolExecutor(max_workers=3) as executor:
        parallel = decode_images(payloads, tmp_path, executor)

    np.testing.assert_array_equal(parallel, serial)
    assert parallel.shape == (len(colors), 3, 4, 3)
    np.testing.assert_array_equal(parallel[:, 0, 0], np.asarray(colors, dtype=np.uint8))


def test_converter_preallocates_final_zarr_shapes(tmp_path):
    _, arrays = create_output(
        tmp_path / "replay_buffer.zarr",
        tactile_embedding_dim=30,
        frame_count=101,
        rgb_chunk_frames=64,
        compressor_name="zstd",
        compression_level=1,
    )

    assert arrays["camera1"].shape == (101, 224, 224, 3)
    assert arrays["camera1"].chunks == (64, 224, 224, 3)
    assert arrays["tactile_embedding"].shape == (101, 30)
    assert arrays["action"].shape == (101, 20)


def test_source_cache_hash_is_reused_when_identity_is_unchanged(tmp_path):
    path = tmp_path / "embeddings.npy"
    path.write_bytes(b"stable tactile cache")

    first = sha256_file_cached(path)
    second = sha256_file_cached(path)

    assert first == second
    payload = json.loads(
        path.with_name("embeddings.npy.sha256.json").read_text(encoding="utf-8")
    )
    assert payload["sha256"] == first


def test_sequence_sampler_repeats_selected_episodes_only():
    replay_buffer = ReplayBuffer.create_empty_numpy()
    replay_buffer.add_episode({"value": np.arange(3, dtype=np.float32)[:, None]})
    replay_buffer.add_episode({"value": np.arange(2, dtype=np.float32)[:, None]})

    sampler = SequenceSampler(
        replay_buffer=replay_buffer,
        sequence_length=1,
        episode_mask=np.array([True, True]),
        episode_repeats=np.array([1, 2]),
    )

    assert sampler.indices[:, 0].tolist() == [0, 1, 2, 3, 4, 3, 4]


def test_sequence_sampler_uses_canonical_noop_for_action_suffix():
    replay_buffer = ReplayBuffer.create_empty_numpy()
    action = np.zeros((2, 20), dtype=np.float32)
    action[:, 0] = [0.001, 0.002]
    action[:, 3] = 1
    action[:, 7] = 1
    action[:, 9] = [0.02, 0.03]
    action[:, 10] = [0.004, 0.005]
    action[:, 13] = 1
    action[:, 17] = 1
    action[:, 19] = [0.03, 0.04]
    replay_buffer.add_episode(
        {
            "action": action,
            "action_valid": np.array([True, False]),
            "idle_arm_mask": np.array([[True, False], [False, True]]),
            "value": np.arange(2, dtype=np.float32)[:, None],
        }
    )
    sampler = SequenceSampler(
        replay_buffer=replay_buffer,
        sequence_length=4,
        pad_after=2,
        canonical_action_padding=True,
    )

    sample = sampler.sample_sequence(0)

    np.testing.assert_allclose(sample["action"][2:, :3], 0.0)
    np.testing.assert_allclose(
        sample["action"][2:, 3:9], np.tile([1, 0, 0, 0, 1, 0], (2, 1))
    )
    np.testing.assert_allclose(sample["action"][2:, 9], 0.03)
    np.testing.assert_allclose(sample["action"][2:, 10:13], 0.0)
    np.testing.assert_allclose(
        sample["action"][2:, 13:19], np.tile([1, 0, 0, 0, 1, 0], (2, 1))
    )
    np.testing.assert_allclose(sample["action"][2:, 19], 0.04)
    assert not sample["action_valid"][2:].any()
    assert not sample["idle_arm_mask"][2:].any()
    np.testing.assert_array_equal(sample["value"][2:], [[1], [1]])


def test_sequence_sampler_uses_canonical_noop_for_action_prefix():
    replay_buffer = ReplayBuffer.create_empty_numpy()
    action = np.zeros((2, 20), dtype=np.float32)
    action[:, 0] = [0.001, 0.002]
    action[:, 3] = 1
    action[:, 7] = 1
    action[:, 9] = [0.02, 0.03]
    action[:, 10] = [0.004, 0.005]
    action[:, 13] = 1
    action[:, 17] = 1
    action[:, 19] = [0.03, 0.04]
    replay_buffer.add_episode({"action": action})
    sampler = SequenceSampler(
        replay_buffer=replay_buffer,
        sequence_length=4,
        pad_before=2,
        canonical_action_padding=True,
    )

    sample = sampler.sample_sequence(0)

    np.testing.assert_allclose(sample["action"][:2, :3], 0.0)
    np.testing.assert_allclose(
        sample["action"][:2, 3:9], np.tile([1, 0, 0, 0, 1, 0], (2, 1))
    )
    np.testing.assert_allclose(sample["action"][:2, 9], 0.02)
    np.testing.assert_allclose(sample["action"][:2, 10:13], 0.0)
    np.testing.assert_allclose(
        sample["action"][:2, 13:19], np.tile([1, 0, 0, 0, 1, 0], (2, 1))
    )
    np.testing.assert_allclose(sample["action"][:2, 19], 0.03)


def test_latent_normalizer_uses_canonical_noop_for_both_padding_edges(monkeypatch):
    replay_buffer = ReplayBuffer.create_empty_numpy()
    action = np.zeros((2, 20), dtype=np.float32)
    action[:, 0] = [0.001, 0.002]
    action[:, 3] = 1
    action[:, 7] = 1
    action[:, 9] = [0.02, 0.03]
    action[:, 10] = [0.004, 0.005]
    action[:, 13] = 1
    action[:, 17] = 1
    action[:, 19] = [0.03, 0.04]
    replay_buffer.add_episode({"action": action})

    normalizer = LinearNormalizer()
    normalizer["action"] = SingleFieldLinearNormalizer.create_identity()
    monkeypatch.setattr(
        RealImageTactileDataset,
        "get_normalizer",
        lambda self, **kwargs: normalizer,
    )

    class CaptureAT:
        device = torch.device("cpu")
        act_scale = 1.0
        use_vq = True
        use_conv_encoder = False

        def preprocess(self, value):
            self.actions = value.detach().clone()
            return value.flatten(1)

        def encoder(self, value):
            return value

    dataset = object.__new__(RealImageTactileLatentDiffusionDataset)
    dataset.shape_meta = {"action": {"shape": [20]}}
    dataset.replay_buffer = replay_buffer
    dataset.sampler = SequenceSampler(
        replay_buffer=replay_buffer,
        sequence_length=4,
        pad_before=2,
        pad_after=2,
        canonical_action_padding=True,
    )
    dataset.at = CaptureAT()
    dataset.use_latent_action_before_vq = True
    dataset.has_v2_action_contract = True

    dataset.get_normalizer(latent_batch_size=64)

    expected_prefix = np.zeros(20, dtype=np.float32)
    expected_prefix[[3, 7, 13, 17]] = 1
    expected_prefix[[9, 19]] = action[0, [9, 19]]
    expected_suffix = expected_prefix.copy()
    expected_suffix[[9, 19]] = action[-1, [9, 19]]
    np.testing.assert_allclose(
        dataset.at.actions[0, :2], np.tile(expected_prefix, (2, 1))
    )
    np.testing.assert_allclose(
        dataset.at.actions[-1, 2:], np.tile(expected_suffix, (2, 1))
    )


def test_sequence_sampler_legacy_20d_action_padding_repeats_last_action():
    replay_buffer = ReplayBuffer.create_empty_numpy()
    action = np.arange(40, dtype=np.float32).reshape(2, 20)
    replay_buffer.add_episode({"action": action})
    sampler = SequenceSampler(
        replay_buffer=replay_buffer,
        sequence_length=4,
        pad_after=2,
    )

    sample = sampler.sample_sequence(0)

    np.testing.assert_array_equal(sample["action"][2:], np.tile(action[-1], (2, 1)))


def _write_minimal_action_dataset(path, include_v2_arrays):
    replay_buffer = ReplayBuffer.create_empty_numpy()
    episode = {"action": np.zeros((2, 20), dtype=np.float32)}
    episode["action"][:, 3] = 1
    episode["action"][:, 7] = 1
    episode["action"][:, 13] = 1
    episode["action"][:, 17] = 1
    if include_v2_arrays:
        episode["action_valid"] = np.array([True, False])
        episode["idle_arm_mask"] = np.array([[True, False], [False, False]])
    replay_buffer.add_episode(episode)
    path.mkdir()
    replay_buffer.save_to_path(path / "replay_buffer.zarr")


def _minimal_dataset(path, **kwargs):
    return RealImageTactileDataset(
        shape_meta={"obs": {}, "action": {"shape": [20]}},
        dataset_path=str(path),
        horizon=2,
        use_episode_repeats=False,
        **kwargs,
    )


def test_dataset_exposes_v2_masks_at_top_level(tmp_path):
    dataset_path = tmp_path / "v2"
    _write_minimal_action_dataset(dataset_path, include_v2_arrays=True)

    dataset = _minimal_dataset(dataset_path)
    sample = dataset[0]

    assert dataset.sampler.canonical_action_padding is True
    assert sample["valid_mask"].dtype == torch.bool
    assert sample["idle_arm_mask"].dtype == torch.bool
    torch.testing.assert_close(sample["valid_mask"], torch.tensor([True, False]))
    torch.testing.assert_close(
        sample["idle_arm_mask"], torch.tensor([[True, False], [False, False]])
    )


@pytest.mark.parametrize("load_to_memory", [False, True])
def test_dataset_persists_v2_manifest_for_artifact_signatures(
    tmp_path, load_to_memory
):
    dataset_path = tmp_path / f"v2-manifest-{load_to_memory}"
    _write_minimal_action_dataset(dataset_path, include_v2_arrays=True)
    manifest = {
        "action_representation_version": 2,
        "action_contract": "bimanual_relative_pose20d_v2",
        "normalizer_version": "zero_centered_v2",
        "dataset_digest": "d" * 64,
        "pca_sha256": "p" * 64,
        "tactile_cache_sha256": "t" * 64,
        "converter_git_commit": "converter-commit",
    }
    raw_manifest = json.dumps(manifest, sort_keys=True)
    root = zarr.open_group(str(dataset_path / "replay_buffer.zarr"), mode="a")
    root["meta"].attrs["v2_manifest_json"] = raw_manifest

    dataset = _minimal_dataset(
        dataset_path,
        load_to_memory=load_to_memory,
        action_normalizer_version="zero_centered_v2",
    )
    cfg = OmegaConf.create(
        {
            "artifact_git_commit": "training-commit",
            "task": {
                "dataset": {
                    "action_normalizer_version": "zero_centered_v2",
                    "seed": 42,
                    "val_ratio": 0.0,
                    "max_train_episodes": None,
                }
            },
        }
    )

    assert dataset.artifact_manifest_raw == raw_manifest
    assert dataset.artifact_manifest == manifest
    signature = build_normalizer_cache_signature(cfg, dataset, None)
    assert signature["dataset"]["digest"] == "d" * 64


def test_dataset_requires_explicit_legacy_action_contract_opt_in(tmp_path):
    dataset_path = tmp_path / "legacy"
    _write_minimal_action_dataset(dataset_path, include_v2_arrays=False)

    with pytest.raises(ValueError, match="allow_legacy_action_contract"):
        _minimal_dataset(dataset_path)

    dataset = _minimal_dataset(dataset_path, allow_legacy_action_contract=True)
    sample = dataset[0]
    assert dataset.sampler.canonical_action_padding is False
    torch.testing.assert_close(sample["valid_mask"], torch.ones(2, dtype=torch.bool))
    torch.testing.assert_close(sample["idle_arm_mask"], torch.zeros((2, 2), dtype=torch.bool))


def test_color_jitter_is_bypassed_in_eval_mode():
    transform = TrainOnlyTransform(torchvision.transforms.ColorJitter(brightness=0.5))
    value = torch.rand(2, 3, 16, 16)
    transform.eval()

    output = transform(value)

    assert output is value


def test_validation_sampler_keeps_observation_read_limit_without_oversampling():
    replay_buffer = ReplayBuffer.create_empty_numpy()
    replay_buffer.add_episode({"value": np.arange(3, dtype=np.float32)[:, None]})
    replay_buffer.add_episode({"value": np.arange(2, dtype=np.float32)[:, None]})

    dataset = object.__new__(RealImageTactileDataset)
    dataset.replay_buffer = replay_buffer
    dataset.horizon = 1
    dataset.n_latency_steps = 0
    dataset.pad_before = 0
    dataset.pad_after = 0
    dataset.val_mask = np.array([False, True])
    dataset.key_first_k = {"value": 1}

    val_dataset = dataset.get_validation_dataset()

    assert val_dataset.sampler.key_first_k == {"value": 1}
    assert val_dataset.sampler.indices[:, 0].tolist() == [3, 4]


def test_pick_tube_configs_match_official_rdp_temporal_and_model_defaults():
    OmegaConf.register_new_resolver("eval", eval, replace=True)
    config_dir = str(
        (Path(__file__).resolve().parents[1] / "reactive_diffusion_policy" / "config")
    )
    with initialize_config_dir(version_base=None, config_dir=config_dir):
        at_cfg = compose(config_name="train_pick_tube_at_workspace")
        ldp_cfg = compose(config_name="train_pick_tube_ldp_workspace")
        at_cfg_16 = compose(
            config_name="train_pick_tube_at_workspace",
            overrides=["task.tactile_embedding_dim=16"],
        )
        ldp_cfg_60 = compose(
            config_name="train_pick_tube_ldp_workspace",
            overrides=["task.tactile_embedding_dim=60"],
        )

    for cfg in (at_cfg, ldp_cfg):
        assert cfg.horizon == 32
        assert cfg.n_obs_steps == 2
        assert cfg.dataset_obs_steps == 4
        assert cfg.dataset_obs_temporal_downsample_ratio == 2
        assert cfg.n_action_steps == 29
        assert cfg.task.dataset.pad_before == 3
        assert cfg.task.dataset.pad_after == 28
        assert cfg.task.dataset.val_ratio == 0.1
        assert cfg.task.dataset.use_episode_repeats is False
        assert cfg.checkpoint.topk.monitor_key == "val_idle_score"
        assert cfg.training.val_every == 1
        assert list(cfg.task.shape_meta.obs.tactile_embedding.shape) == [30]
        assert list(cfg.task.shape_meta.extended_obs.tactile_embedding.shape) == [30]
        assert "pca30" in cfg.task.dataset_path

    assert list(at_cfg_16.task.shape_meta.obs.tactile_embedding.shape) == [16]
    assert list(at_cfg_16.task.shape_meta.extended_obs.tactile_embedding.shape) == [16]
    assert list(ldp_cfg_60.task.shape_meta.obs.tactile_embedding.shape) == [60]
    assert list(ldp_cfg_60.task.shape_meta.extended_obs.tactile_embedding.shape) == [60]

    assert at_cfg.at.policy.n_latent_dims == 16
    assert at_cfg.at.policy.conv_latent_dims == 32
    assert at_cfg.at.policy.rnn_latent_dims == 64
    assert at_cfg.at.policy.n_embed == 32
    assert ldp_cfg.policy.at.n_latent_dims == 16
    assert ldp_cfg.policy.at.conv_latent_dims == 32
    assert ldp_cfg.policy.at.rnn_latent_dims == 64
    assert ldp_cfg.policy.at.n_embed == 32
    assert ldp_cfg.task.dataset.at.n_latent_dims == 16
    assert ldp_cfg.task.dataset.at.conv_latent_dims == 32
    assert ldp_cfg.task.dataset.at.rnn_latent_dims == 64
    assert ldp_cfg.task.dataset.at.n_embed == 32
    assert at_cfg.dataloader.batch_size == 64
    assert at_cfg.training.num_epochs == 20
    assert at_cfg.training.checkpoint_every == 10

    assert list(ldp_cfg.policy.down_dims) == [512, 1024, 2048]
    assert len(ldp_cfg.policy.obs_encoder.random_transforms) == 1
    assert ldp_cfg.policy.obs_encoder.random_transforms[0].type == "RandomCrop"
    assert list(ldp_cfg.policy.obs_encoder.resize_shape) == [224, 224]
    assert ldp_cfg.dataloader.batch_size == 64
    assert ldp_cfg.dataloader.num_workers == 8
    assert ldp_cfg.training.num_epochs == 10
    assert ldp_cfg.training.checkpoint_every == 2
