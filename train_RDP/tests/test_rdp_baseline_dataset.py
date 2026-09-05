"""Baseline data contract tests; CPU-only synthetic replay with nonzero rotations."""
import importlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
import zarr
from scipy.spatial.transform import Rotation


class BaselineDatasetTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        n = 16
        self.state = np.zeros((n, 7), np.float32)
        self.state[:, 0] = np.arange(n) * .0001
        self.state[:, 3:6] = [.2, -.4, .7]
        self.state[:, 6] = np.linspace(.12, .08, n)
        rotation = Rotation.from_rotvec(self.state[:, 3:6]).as_matrix()
        raw = np.zeros((n, 10), np.float32)
        raw[:, 3:9] = [1, 0, 0, 0, 1, 0]
        raw[:, :3] = np.einsum('nji,nj->ni', rotation, np.tile([.0001, 0, 0], (n, 1)))
        raw[:, 9] = self.state[:, 6]
        raw[[7, 15], :9] = 0  # Invalid original terminal rows must not be decoded.
        self.raw = raw
        replay = zarr.open_group(str(self.root / 'replay_buffer.zarr'), mode='w')
        data = replay.create_group('data')
        for key, value in {
            'observation_state': self.state, 'action_raw': raw,
            'action': np.zeros_like(raw),  # Deliberately corrupt canonical actions.
            'camera2': np.arange(n, dtype=np.uint8)[:, None, None, None] * np.ones((n, 2, 2, 3), np.uint8),
            'tactile_embedding': np.full((n, 30), 999, np.float32),
        }.items():
            data.create_dataset(key, data=value)
        replay.create_group('meta').create_dataset('episode_ends', data=np.array([8, 16]))
        cache = np.zeros((n, 4, 512), np.float16)
        cache[:, 2, :15] = np.arange(n)[:, None]
        np.save(self.root / 'embeddings.npy', cache)
        from reactive_diffusion_policy.model.tactile_pca import save_tactile_pca
        components = np.zeros((2, 15, 1024), np.float32)
        components[:, np.arange(15), np.arange(15)] = 1
        save_tactile_pca(self.root / 'pca.npz', means=np.zeros((2, 1024), np.float32),
                        components=components, explained_variance_ratio=np.ones((2, 15), np.float32), sample_count=n)

    def dataset(self, **kwargs):
        try:
            cls = importlib.import_module('rdp_baseline.dataset').SingleRightChunkRelativeDataset
        except ModuleNotFoundError:
            self.fail('Baseline dataset implementation is missing')
        args = dict(dataset_path=str(self.root), tactile_cache_path=str(self.root / 'embeddings.npy'),
                    tactile_pca_path=str(self.root / 'pca.npz'), horizon=4, n_obs_steps=2,
                    obs_temporal_downsample_ratio=1, pad_before=1, pad_after=2, val_ratio=0)
        args.update(kwargs)
        return cls(**args)

    def test_chunk_uses_last_observation_base_and_keeps_micro_motion(self):
        ds = self.dataset()
        sample = ds[1]  # Unpadded window frames 0,1,2,3; base is frame 1.
        self.assertEqual(set(sample), {'obs', 'extended_obs', 'action'})
        r = Rotation.from_rotvec(self.state[1, 3:6]).as_matrix()
        expected = np.array([r.T @ np.array([x, 0, 0]) for x in (0, .0001, .0002, .0003)])
        np.testing.assert_allclose(sample['action'][:, :3], expected, atol=1e-9)
        np.testing.assert_allclose(sample['action'][:, 9], self.raw[:4, 9])
        np.testing.assert_allclose(sample['obs']['right_robot_tcp_pose'][-1], [0,0,0,1,0,0,0,1,0], atol=1e-7)
        self.assertEqual(ds.action_contract, 'single_right_chunk_relative10d_v1')

    def test_right_raw_tactile_projection_and_rgb_are_aligned(self):
        sample = self.dataset()[1]
        np.testing.assert_array_equal(sample['extended_obs']['tactile_embedding'][:, 0], [0,1,2,3])
        np.testing.assert_array_equal(sample['obs']['tactile_embedding'][:, 0], [0,1])
        np.testing.assert_allclose(sample['obs']['camera2'][:, 0, 0, 0], [0, 1/255])

    def test_padding_repeats_absolute_edge_target_and_terminal_holds(self):
        ds = self.dataset()
        start = ds[0]
        np.testing.assert_allclose(start['action'][0], start['action'][1])
        self.assertGreater(float(torch.linalg.vector_norm(start['action'][0, :3])), 0)
        last = ds[6]  # Frames 5,6,7,7; terminal absolute target is state[7].
        np.testing.assert_allclose(last['action'][1, :9], last['action'][2, :9], atol=1e-7)
        np.testing.assert_allclose(last['action'][2], last['action'][3])
        self.assertTrue(torch.isfinite(last['action']).all())

    def test_normalizer_matches_original_range_and_identity_without_images(self):
        ds = self.dataset()
        # If normalization attempts to load RGB, this raises.
        ds._read_rgb = lambda *args: (_ for _ in ()).throw(AssertionError('normalizer read RGB'))
        normalizer = ds.get_normalizer(batch_size=3)
        actions = ds.get_lowdim_batch(np.arange(len(ds)))['action'].reshape(-1, 10)
        from reactive_diffusion_policy.common.normalize_util import get_action_normalizer
        original = get_action_normalizer(actions, version='legacy_v1')
        np.testing.assert_allclose(normalizer['action'].normalize(actions).detach(), original.normalize(actions).detach(), atol=2e-6)
        np.testing.assert_allclose(normalizer['action'].normalize(actions)[:, 3:9].detach(), actions[:, 3:9])
        self.assertIs(normalizer, ds.get_normalizer())

    def test_validation_is_episode_disjoint_and_no_camera_shape_omits_image_reads(self):
        shape = {'obs': {'right_robot_tcp_pose': {'shape':[9]}, 'right_robot_gripper_width': {'shape':[1]},
                         'tactile_embedding': {'shape':[15]}}, 'action': {'shape':[10]}}
        ds = self.dataset(val_ratio=.5, shape_meta=shape)
        val = ds.get_validation_dataset()
        self.assertFalse(set(ds.indices[:, 0]) & set(val.indices[:, 0]))
        self.assertNotIn('camera2', ds[0]['obs'])

    def test_rejects_canonical_only_replay(self):
        replay = zarr.open_group(str(self.root / 'replay_buffer.zarr'), mode='a')
        del replay['data/action_raw']
        with self.assertRaisesRegex(ValueError, 'action_raw'):
            self.dataset()

    def test_noncommuting_rotation_target_and_fixed_base_are_preserved(self):
        from rdp_baseline.geometry import state7_to_matrix, pose10_to_matrix, matrix_to_pose9, relative_to_base
        base = state7_to_matrix(np.array([[.4, -.2, .6, .2, -.4, .7, .12]]))
        delta = np.eye(4)
        delta[:3, :3] = Rotation.from_rotvec([.09, -.02, .04]).as_matrix()
        delta[:3, 3] = [.0001, -.0003, .0002]
        raw = np.r_[delta[:3, 3], delta[:3, :2].T.reshape(-1), .11][None]
        target = base @ pose10_to_matrix(raw)
        expected = base[0] @ delta
        np.testing.assert_allclose(target[0], expected, atol=1e-12)
        recovered = relative_to_base(target[:, None], base)
        np.testing.assert_allclose(recovered[0, 0], delta, atol=1e-12)
        np.testing.assert_allclose(matrix_to_pose9(recovered)[0, 0], raw[0, :9], atol=1e-8)

    def test_latency_drops_actions_after_selecting_last_downsampled_observation(self):
        ds = self.dataset(n_latency_steps=1, n_obs_steps=3, obs_temporal_downsample_ratio=2)
        sample = ds[1]  # Raw frames 0..4, observed [0,2], outputs targets1..4.
        np.testing.assert_allclose(sample['action'][0, :3], 0, atol=1e-9)
        np.testing.assert_array_equal(sample['obs']['tactile_embedding'][:, 0], [0,2])
        np.testing.assert_array_equal(sample['extended_obs']['tactile_embedding'][:, 0], [0,1,2,3,4])
        self.assertEqual(tuple(sample['action'].shape), (4,10))

    def test_manifest_cache_preserves_source_order_without_copying_raw(self):
        full = np.load(self.root / 'embeddings.npy')
        np.save(self.root / 'first.npy', full[:8])
        np.save(self.root / 'second.npy', full[8:])
        manifest = {'version': 1, 'total_frames': 16, 'shards': [
            {'path': 'first.npy', 'start': 0, 'stop': 8},
            {'path': 'second.npy', 'start': 0, 'stop': 8}]}
        path = self.root / 'raw_tactile_manifest.json'
        path.write_text(json.dumps(manifest))
        ds = self.dataset(tactile_cache_path=path)
        np.testing.assert_array_equal(ds.tactile[:, 0], np.arange(16))

    def test_both_arms_use_independent_fixed_bases_and_hold_terminal_widths(self):
        replay = zarr.open_group(str(self.root / 'replay_buffer.zarr'), mode='a')
        state = np.concatenate((self.state.copy(), self.state.copy(), np.zeros((16, 6))), axis=1).astype(np.float32)
        state[:, 7:10] += [1, 2, 3]
        state[:, 13] = .03
        raw = np.concatenate((self.raw, self.raw), axis=1)
        raw[:, 19] = .02
        del replay['data/observation_state'], replay['data/action_raw']
        replay['data'].create_dataset('observation_state', data=state)
        replay['data'].create_dataset('action_raw', data=raw)
        replay['data'].create_dataset('camera1', data=replay['data/camera2'][:] + 20)
        cache = np.load(self.root / 'embeddings.npy')
        cache[:, 0, :15] = np.arange(16)[:, None] + 100
        np.save(self.root / 'embeddings.npy', cache)
        module = importlib.import_module('rdp_baseline.dataset')
        self.assertTrue(hasattr(module, 'ChunkRelativeDataset'), 'Multi-arm baseline adapter missing')
        ds = module.ChunkRelativeDataset(
            self.root, self.root / 'embeddings.npy', self.root / 'pca.npz', arms='both',
            horizon=4, n_obs_steps=2, obs_temporal_downsample_ratio=1, pad_before=1, pad_after=2, val_ratio=0)
        sample = ds[1]
        self.assertEqual(sample['action'].shape, (4, 20))
        self.assertEqual(ds.action_contract, 'dual_arm_chunk_relative20d_v1')
        np.testing.assert_allclose(sample['action'][:, :9], sample['action'][:, 10:19], atol=3e-7)
        for arm in ('left', 'right'):
            np.testing.assert_allclose(sample['obs'][arm + '_robot_tcp_pose'][-1], [0,0,0,1,0,0,0,1,0], atol=1e-7)
        np.testing.assert_array_equal(sample['extended_obs']['tactile_embedding'][:, [0, 15]], [[100,0],[101,1],[102,2],[103,3]])
        self.assertIn('camera1', sample['obs'])
        np.testing.assert_allclose(ds[6]['action'][-1, [9, 19]], [self.state[7, 6], .03])
        normalizer = ds.get_normalizer(batch_size=3)
        actions = ds.get_lowdim_batch(np.arange(len(ds)))['action']
        normalized = normalizer['action'].normalize(actions).detach().numpy()
        np.testing.assert_allclose(normalized[..., 3:9], actions[..., 3:9])
        np.testing.assert_allclose(normalized[..., 13:19], actions[..., 13:19])


if __name__ == '__main__':
    unittest.main()
