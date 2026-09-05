"""Original RDP chunk-relative sampling for single-right and dual-arm replays.

Source action_raw[t] is a LOCAL increment from measured T[t] to next target,
so A[t] = T[t] @ delta[t]. It is NOT an original-RDP chunk-relative label.
Each sampled chunk returns inv(T[last observation]) @ A[t]. Source terminal
rows have no valid next target: hold the final measured pose and width. Padding
repeats absolute edge targets before the fixed-base conversion, as upstream.
Dual-arm state20 stores left state7, right state7, then six unused source fields;
its action20 stores the two local pose10 increments in left/right order.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import torch
import zarr

from reactive_diffusion_policy.dataset.base_dataset import BaseImageDataset
from reactive_diffusion_policy.model.common.normalizer import LinearNormalizer
from reactive_diffusion_policy.common.normalize_util import (
    concatenate_normalizer, get_range_normalizer_from_stat,
    get_identity_normalizer_from_stat, get_image_range_normalizer,
)
from reactive_diffusion_policy.model.tactile_pca import BimanualTactilePCA
from .geometry import state7_to_matrix, pose10_to_matrix, matrix_to_pose9, relative_to_base


class _StreamingStats:
    def __init__(self):
        self.count = 0

    def update(self, value):
        value = np.asarray(value).reshape(-1, value.shape[-1]).astype(np.float64)
        if not len(value):
            return
        if self.count == 0:
            self.minimum, self.maximum = value.min(0), value.max(0)
            self.total, self.squares = value.sum(0), (value * value).sum(0)
        else:
            self.minimum = np.minimum(self.minimum, value.min(0))
            self.maximum = np.maximum(self.maximum, value.max(0))
            self.total += value.sum(0)
            self.squares += (value * value).sum(0)
        self.count += len(value)

    def finish(self):
        if not self.count:
            raise ValueError('Cannot normalize an empty training split')
        mean = self.total / self.count
        return {key: value.astype(np.float32) for key, value in dict(
            min=self.minimum, max=self.maximum, mean=mean,
            std=np.sqrt(np.maximum(0, self.squares / self.count - mean * mean))).items()}


def _pose_normalizer(stat):
    if len(stat['min']) == 20:
        return concatenate_normalizer([
            _pose_normalizer({k: v[start:start+10].copy() for k, v in stat.items()})
            for start in (0, 10)])
    # Exactly upstream get_action_normalizer's D9/D10 rule: minmax xyz/grip,
    # untouched rotation (not an identity-residual representation).
    parts = [get_range_normalizer_from_stat({k: v[:3].copy() for k, v in stat.items()}),
             get_identity_normalizer_from_stat({k: v[3:9].copy() for k, v in stat.items()})]
    if len(stat['min']) == 10:
        parts.append(get_range_normalizer_from_stat({k: v[9:].copy() for k, v in stat.items()}))
    return concatenate_normalizer(parts)


def _raw_tactile_shards(path, total_frames):
    """Map source caches in replay order without concatenating their raw arrays."""
    path = Path(path)
    if path.suffix == '.json':
        manifest = json.loads(path.read_text())
        if manifest.get('version') != 1 or manifest.get('total_frames') != total_frames:
            raise ValueError('Raw tactile manifest version/frame count mismatch')
        entries = manifest['shards']
    else:
        entries = [{'path': str(path.resolve()), 'start': 0, 'stop': total_frames}]
    offset = 0
    for entry in entries:
        source = Path(entry['path'])
        if not source.is_absolute():
            source = path.parent / source
        values = np.load(source, mmap_mode='r', allow_pickle=False)
        start, stop = int(entry['start']), int(entry['stop'])
        if values.ndim != 3 or values.shape[1:] != (4, 512) or not 0 <= start < stop <= len(values):
            raise ValueError(f'Invalid raw tactile shard {source}: {values.shape}, [{start}:{stop}]')
        if path.suffix != '.json' and len(values) != total_frames:
            raise ValueError('Raw tactile cache must align one-to-one with replay')
        yield offset, values[start:stop]
        offset += stop-start
    if offset != total_frames:
        raise ValueError('Raw tactile shards must align one-to-one with replay')


class ChunkRelativeDataset(BaseImageDataset):
    action_contract = 'single_right_chunk_relative10d_v1'

    def __init__(self, dataset_path, tactile_cache_path, tactile_pca_path,
                 horizon=32, n_obs_steps=4, obs_temporal_downsample_ratio=2,
                 n_latency_steps=0, pad_before=3, pad_after=28, val_ratio=.1,
                 seed=42, max_train_episodes=None, episode_limit=None, shape_meta=None,
                 arms='right'):
        if arms not in ('right', 'both'):
            raise ValueError('arms must be right or both')
        self.arms = ('left', 'right') if arms == 'both' else ('right',)
        self.action_contract = ('dual_arm_chunk_relative20d_v1' if arms == 'both'
                                else 'single_right_chunk_relative10d_v1')
        path = Path(dataset_path)
        if path.name != 'replay_buffer.zarr':
            path = path / 'replay_buffer.zarr'
        self.dataset_path = str(path)
        self.replay = zarr.open_group(str(path), mode='r')
        data = self.replay['data']
        if 'action_raw' not in data:
            raise ValueError('Baseline requires source action_raw; canonical action cannot be used')
        ends = np.asarray(self.replay['meta/episode_ends'][:], dtype=np.int64)
        if not len(ends) or np.any(np.diff(np.r_[0, ends]) <= 0):
            raise ValueError('Replay must contain nonempty episodes')
        total_frames = int(ends[-1])
        state_dim, action_dim = (20, 20) if arms == 'both' else (7, 10)
        if data['observation_state'].shape != (total_frames, state_dim) or data['action_raw'].shape != (total_frames, action_dim):
            raise ValueError(f'Baseline arms={arms} expects state{state_dim} and action_raw{action_dim}')
        if episode_limit is not None:
            if int(episode_limit) < 1:
                raise ValueError('episode_limit must be positive')
            ends = ends[:int(episode_limit)]
        self.episode_ends = ends
        self.episode_starts = np.r_[0, ends[:-1]]
        n = int(ends[-1])
        self.state = np.asarray(data['observation_state'][:n], dtype=np.float32)
        if not np.isfinite(self.state).all():
            raise ValueError('Nonfinite observation_state')
        raw = np.asarray(data['action_raw'][:n], dtype=np.float32)
        valid = np.ones(n, dtype=bool)
        valid[ends - 1] = False
        self.arm_states, self.state_matrices, self.target_matrices, self.target_grippers = {}, {}, {}, {}
        for arm_index, arm in enumerate(self.arms):
            state = self.state[:, arm_index*7:arm_index*7+7]
            arm_raw = raw[:, arm_index*10:arm_index*10+10]
            measured = state7_to_matrix(state)
            target = measured.copy()
            target[valid] = measured[valid] @ pose10_to_matrix(arm_raw[valid])
            gripper = arm_raw[:, 9:10].copy()
            gripper[ends-1] = state[ends-1, 6:7]
            self.arm_states[arm], self.state_matrices[arm] = state, measured
            self.target_matrices[arm], self.target_grippers[arm] = target, gripper
        pca = BimanualTactilePCA.from_npz(tactile_pca_path)
        if pca.components_per_arm != 15:
            raise ValueError('Baseline requires the explicit 2x15 PCA artifact')
        self.tactile = np.empty((n, 15*len(self.arms)), np.float32)
        for offset, raw_cache in _raw_tactile_shards(tactile_cache_path, total_frames):
            for start in range(0, min(len(raw_cache), n-offset), 4096):
                stop = min(len(raw_cache), n-offset, start+4096)
                for arm_index, arm in enumerate(self.arms):
                    pca_index = 0 if arm == 'left' else 1
                    values = np.asarray(raw_cache[start:stop, pca_index*2:pca_index*2+2], dtype=np.float32).reshape(stop-start, 1024)
                    self.tactile[offset+start:offset+stop, arm_index*15:arm_index*15+15] = (
                        values-pca.means[pca_index].numpy()) @ pca.components[pca_index].numpy().T
        self.tactile_cache_path = str(tactile_cache_path)
        self.tactile_pca_path = str(tactile_pca_path)
        self.horizon = int(horizon)
        self.n_latency_steps = int(n_latency_steps)
        self.sequence_length = self.horizon + self.n_latency_steps
        self.n_obs_steps = self.horizon if n_obs_steps is None else int(n_obs_steps)
        self.obs_downsample_ratio = int(obs_temporal_downsample_ratio)
        if not (1 <= self.n_obs_steps <= self.sequence_length) or self.obs_downsample_ratio < 1:
            raise ValueError('Invalid observation history length/downsample ratio')
        self.obs_indices = np.arange(self.n_obs_steps)[::-self.obs_downsample_ratio][::-1]
        self.pad_before = min(max(0, int(pad_before)), self.sequence_length - 1)
        self.pad_after = min(max(0, int(pad_after)), self.sequence_length - 1)
        self.shape_meta = shape_meta
        cameras = ('camera1', 'camera2') if arms == 'both' else ('camera2',)
        self.rgb_keys = tuple(key for key in cameras if shape_meta is None or key in shape_meta.get('obs', {}))
        self.include_rgb = bool(self.rgb_keys)
        self.val_mask = np.zeros(len(ends), dtype=bool)
        rng = np.random.default_rng(seed)
        if val_ratio > 0:
            count = min(max(1, round(len(ends) * val_ratio)), len(ends)-1)
            self.val_mask[rng.choice(len(ends), size=count, replace=False)] = True
        self.train_mask = ~self.val_mask
        if max_train_episodes is not None and self.train_mask.sum() > max_train_episodes:
            selected = np.random.default_rng(seed).choice(np.flatnonzero(self.train_mask), size=int(max_train_episodes), replace=False)
            self.train_mask[:] = False
            self.train_mask[selected] = True
        self.indices = self._make_indices(self.train_mask)
        self._normalizer = None

    def _make_indices(self, mask):
        indices = []
        for ep in np.flatnonzero(mask):
            length = self.episode_ends[ep] - self.episode_starts[ep]
            for start in range(-self.pad_before, length-self.sequence_length+self.pad_after+1):
                indices.append((ep, start))
        return np.asarray(indices, dtype=np.int64).reshape(-1, 2)

    def get_validation_dataset(self):
        result = copy.copy(self)
        result.indices = self._make_indices(self.val_mask)
        return result

    def __len__(self):
        return len(self.indices)

    def _frame_indices(self, indices):
        ep, start = self.indices[np.asarray(indices, dtype=np.int64)].T
        frames = self.episode_starts[ep, None] + start[:, None] + np.arange(self.sequence_length)
        return np.clip(frames, self.episode_starts[ep, None], self.episode_ends[ep, None]-1)

    def get_lowdim_batch(self, indices):
        frames = self._frame_indices(indices)
        obs_frames = frames[:, self.obs_indices]
        obs, actions = {'tactile_embedding': self.tactile[obs_frames]}, []
        for arm in self.arms:
            base = self.state_matrices[arm][obs_frames[:, -1]]
            action_pose = matrix_to_pose9(relative_to_base(self.target_matrices[arm][frames], base))
            actions.append(np.concatenate((action_pose, self.target_grippers[arm][frames]), axis=-1))
            obs[arm+'_robot_tcp_pose'] = matrix_to_pose9(relative_to_base(self.state_matrices[arm][obs_frames], base))
            obs[arm+'_robot_gripper_width'] = self.arm_states[arm][obs_frames, 6:7]
        action = np.concatenate(actions, axis=-1)
        if self.n_latency_steps:
            action = action[:, self.n_latency_steps:]
        return {'obs': obs, 'extended_obs': {'tactile_embedding': self.tactile[frames]}, 'action': action}

    def _read_rgb(self, frames, key='camera2'):
        return np.stack([self.replay['data/'+key][int(frame)] for frame in frames])

    def __getitem__(self, idx):
        batch = self.get_lowdim_batch([idx])
        sample = {key: {name: torch.from_numpy(value[0].copy()) for name, value in group.items()}
                  for key, group in batch.items() if key != 'action'}
        sample['action'] = torch.from_numpy(batch['action'][0].copy())
        if self.include_rgb:
            frames = self._frame_indices([idx])[0, self.obs_indices]
            for key in self.rgb_keys:
                rgb = self._read_rgb(frames, key)
                sample['obs'][key] = torch.from_numpy(np.moveaxis(rgb, -1, 1).astype(np.float32) / 255)
        return sample

    def get_normalizer(self, batch_size=1024, **kwargs):
        if self._normalizer is not None:
            return self._normalizer
        stats = {key: _StreamingStats() for key in ('action', *(arm+'_robot_tcp_pose' for arm in self.arms))}
        for start in range(0, len(self), batch_size):
            batch = self.get_lowdim_batch(np.arange(start, min(start+batch_size, len(self))))
            stats['action'].update(batch['action'])
            for arm in self.arms:
                stats[arm+'_robot_tcp_pose'].update(batch['obs'][arm+'_robot_tcp_pose'])
        normalizer = LinearNormalizer()
        for key, stat in stats.items():
            normalizer[key] = _pose_normalizer(stat.finish())
        # Original non-relative scalar/tactile observations use unwindowed rows.
        # Keep validation episodes out of fitted statistics.
        scalar_values = [(arm+'_robot_gripper_width', self.arm_states[arm][:, 6:7]) for arm in self.arms]
        for key, values in [*scalar_values, ('tactile_embedding', self.tactile)]:
            stat = _StreamingStats()
            for ep in np.flatnonzero(self.train_mask):
                stat.update(values[self.episode_starts[ep]:self.episode_ends[ep]])
            normalizer[key] = get_range_normalizer_from_stat(stat.finish(), range_eps=1e-4)
        for key in self.rgb_keys:
            normalizer[key] = get_image_range_normalizer()
        self._normalizer = normalizer
        return normalizer

    def get_all_actions(self):
        raise NotImplementedError('Use get_lowdim_batch for chunk-relative actions; there is no single per-frame relative label')


class SingleRightChunkRelativeDataset(ChunkRelativeDataset):
    """Backwards-compatible entry point for the original Insert01 configuration."""
