from typing import Dict
import json
import torch
import numpy as np
import os
import zarr
from threadpoolctl import threadpool_limits
import copy
import tqdm
from reactive_diffusion_policy.common.pytorch_util import dict_apply
from reactive_diffusion_policy.dataset.base_dataset import BaseImageDataset
from reactive_diffusion_policy.model.common.normalizer import LinearNormalizer, SingleFieldLinearNormalizer
from reactive_diffusion_policy.common.replay_buffer import ReplayBuffer
from reactive_diffusion_policy.common.sampler import (
    SequenceSampler, downsample_mask)
from reactive_diffusion_policy.common.pick_tube_validation import (
    build_episode_split_manifest,
)
from reactive_diffusion_policy.common.artifact_manifest import stable_json_digest
from reactive_diffusion_policy.common.normalize_util import (
    get_image_range_normalizer,
    get_action_normalizer
)

class RealImageTactileDataset(BaseImageDataset):
    def __init__(self,
                 shape_meta: dict,
                 dataset_path: str,
                 horizon=1,
                 pad_before=0,
                 pad_after=0,
                 n_obs_steps=None,
                 obs_temporal_downsample_ratio=1, # for latent diffusion
                 n_latency_steps=0,
                 seed=42,
                 val_ratio=0.0,
                 max_train_episodes=None,
                 use_episode_repeats=True,
                 delta_action=False,
                 relative_action=False,
                 relative_tcp_obs_for_relative_action=True,
                 transform_params=None,
                 load_to_memory=True,
                 bimanual_contiguous_action=False,
                 allow_legacy_action_contract=False,
                 action_normalizer_version: str = "legacy_v1",
                 ):
        assert os.path.isdir(dataset_path)

        rgb_keys = list()
        lowdim_keys = list()
        obs_shape_meta = shape_meta['obs']
        for key, attr in obs_shape_meta.items():
            type = attr.get('type', 'low_dim')
            if type == 'rgb':
                rgb_keys.append(key)
            elif type == 'low_dim':
                lowdim_keys.append(key)

        extended_rgb_keys = list()
        extended_lowdim_keys = list()
        extended_obs_shape_meta = shape_meta.get('extended_obs', dict())
        for key, attr in extended_obs_shape_meta.items():
            type = attr.get('type', 'low_dim')
            if type == 'rgb':
                extended_rgb_keys.append(key)
            elif type == 'low_dim':
                extended_lowdim_keys.append(key)

        zarr_path = os.path.join(dataset_path, 'replay_buffer.zarr')
        zarr_root = zarr.open_group(zarr_path, mode="r")
        artifact_manifest_raw = zarr_root["meta"].attrs.get("v2_manifest_json")
        artifact_manifest = None
        if artifact_manifest_raw is not None:
            if isinstance(artifact_manifest_raw, str):
                try:
                    artifact_manifest = json.loads(artifact_manifest_raw)
                except json.JSONDecodeError as error:
                    raise ValueError("dataset v2 artifact manifest is invalid JSON") from error
            elif isinstance(artifact_manifest_raw, dict):
                artifact_manifest = dict(artifact_manifest_raw)
            else:
                raise ValueError("dataset v2 artifact manifest must be JSON metadata")
        contract_keys = {"action_valid", "idle_arm_mask"}
        present_contract_keys = contract_keys.intersection(zarr_root["data"].keys())
        if present_contract_keys and present_contract_keys != contract_keys:
            missing = sorted(contract_keys - present_contract_keys)
            raise ValueError(f"incomplete v2 action contract; missing arrays: {missing}")
        has_v2_action_contract = present_contract_keys == contract_keys
        if not has_v2_action_contract and not allow_legacy_action_contract:
            raise ValueError(
                "dataset is missing action_valid and idle_arm_mask; pass "
                "allow_legacy_action_contract=True only for an intentional legacy run"
            )
        zarr_load_keys = set(rgb_keys + lowdim_keys + extended_rgb_keys + extended_lowdim_keys + ['action'])
        if has_v2_action_contract:
            zarr_load_keys.update(contract_keys)
        zarr_load_keys = list(filter(lambda key: "wrt" not in key, zarr_load_keys))
        if load_to_memory:
            replay_buffer = ReplayBuffer.copy_from_path(
                zarr_path, keys=zarr_load_keys)
        else:
            replay_buffer = ReplayBuffer.create_from_path(
                zarr_path, mode='r', keys=zarr_load_keys)

        if delta_action:
            # replace action as relative to previous frame
            actions = replay_buffer['action'][:]
            # support positions only at this time
            assert actions.shape[1] <= 3
            actions_diff = np.zeros_like(actions)
            episode_ends = replay_buffer.episode_ends[:]
            for i in range(len(episode_ends)):
                start = 0
                if i > 0:
                    start = episode_ends[i-1]
                end = episode_ends[i]
                # delta action is the difference between previous desired position and the current
                # it should be scheduled at the previous timestep for the current timestep
                # to ensure consistency with positional mode
                actions_diff[start+1:end] = np.diff(actions[start:end], axis=0)
            replay_buffer['action'][:] = actions_diff
        
        self.relative_action = relative_action
        self.relative_tcp_obs_for_relative_action = relative_tcp_obs_for_relative_action
        self.bimanual_contiguous_action = bimanual_contiguous_action
        self.action_normalizer_version = action_normalizer_version
        self.artifact_manifest_raw = artifact_manifest_raw
        self.artifact_manifest = artifact_manifest
        self.has_v2_action_contract = has_v2_action_contract
        self.allow_legacy_action_contract = allow_legacy_action_contract
        self.transforms = None
        if relative_action or any('wrt' in key for key in lowdim_keys + extended_lowdim_keys):
            from reactive_diffusion_policy.real_world.real_world_transforms import RealWorldTransforms
            self.transforms = RealWorldTransforms(option=transform_params)

        key_first_k = dict()
        if n_obs_steps is not None:
            # only take first k obs from images
            for key in rgb_keys + lowdim_keys:
                if key not in extended_rgb_keys + extended_lowdim_keys:
                    key_first_k[key] = n_obs_steps * obs_temporal_downsample_ratio
        self.key_first_k = key_first_k

        self.seed = seed
        if "episode_dataset_ids" in zarr_root["meta"]:
            episode_sources = zarr_root["meta"]["episode_dataset_ids"][:]
        else:
            episode_sources = np.zeros(replay_buffer.n_episodes, dtype=np.int64)
        split_manifest = build_episode_split_manifest(
            episode_sources,
            val_ratio=val_ratio,
            seed=seed,
        )
        val_mask = np.zeros(replay_buffer.n_episodes, dtype=bool)
        val_mask[split_manifest["validation_episode_ids"]] = True
        train_mask = ~val_mask
        train_mask = downsample_mask(
            mask=train_mask, 
            max_n=max_train_episodes, 
            seed=seed)
        split_identity = {
            key: value
            for key, value in split_manifest.items()
            if key != "split_digest"
        }
        split_identity["train_episode_ids"] = np.flatnonzero(train_mask).tolist()
        split_identity["excluded_episode_ids"] = np.flatnonzero(
            ~(train_mask | val_mask)
        ).tolist()
        split_manifest = {
            **split_identity,
            "split_digest": stable_json_digest(split_identity),
        }

        episode_repeats = None
        if use_episode_repeats:
            if "episode_repeats" in zarr_root["meta"]:
                episode_repeats = zarr_root["meta"]["episode_repeats"][:]

        sampler = SequenceSampler(
            replay_buffer=replay_buffer, 
            sequence_length=horizon+n_latency_steps,
            pad_before=pad_before, 
            pad_after=pad_after,
            episode_mask=train_mask,
            episode_repeats=episode_repeats,
            key_first_k=key_first_k,
            canonical_action_padding=has_v2_action_contract)
        
        self.replay_buffer = replay_buffer
        self.sampler = sampler
        self.shape_meta = shape_meta
        self.rgb_keys = rgb_keys
        self.lowdim_keys = lowdim_keys
        self.extended_rgb_keys = extended_rgb_keys
        self.extended_lowdim_keys = extended_lowdim_keys
        self.n_obs_steps = n_obs_steps
        self.obs_downsample_ratio = obs_temporal_downsample_ratio
        self.val_mask = val_mask
        self.train_mask = train_mask
        self.split_manifest = split_manifest
        self.episode_repeats = episode_repeats
        self.horizon = horizon
        self.n_latency_steps = n_latency_steps
        self.pad_before = pad_before
        self.pad_after = pad_after

    def get_validation_dataset(self):
        val_set = copy.copy(self)
        val_set.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=self.horizon+self.n_latency_steps,
            pad_before=self.pad_before,
            pad_after=self.pad_after,
            episode_mask=self.val_mask,
            # Match the training sampler's I/O optimization. Validation does
            # not apply episode_repeats because oversampling is a training-only
            # weighting, but it should still avoid reading unused RGB frames.
            key_first_k=self.key_first_k,
            canonical_action_padding=getattr(self, "has_v2_action_contract", False),
            )
        val_set.val_mask = ~self.val_mask
        return val_set

    def _get_training_rows(self, key, width):
        values = self.replay_buffer[key]
        episode_ends = self.replay_buffer.episode_ends[:]
        chunks = []
        start = 0
        for include, end in zip(self.train_mask, episode_ends):
            if include:
                chunks.append(np.asarray(values[start:end, :width]))
            start = end
        return np.concatenate(chunks, axis=0)

    def _get_training_mask(self, key):
        values = self.replay_buffer[key]
        episode_ends = self.replay_buffer.episode_ends[:]
        chunks = []
        start = 0
        for include, end in zip(self.train_mask, episode_ends):
            if include:
                chunks.append(np.asarray(values[start:end], dtype=bool))
            start = end
        return np.concatenate(chunks, axis=0)

    def get_normalizer(self, **kwargs) -> LinearNormalizer:
        normalizer = LinearNormalizer()

        # calculate relative action / obs
        if "left_robot_wrt_right_robot_tcp_pose" in self.lowdim_keys or "right_robot_wrt_left_robot_tcp_pose" in self.lowdim_keys:
            inter_gripper_data_dict = {key: list() for key in self.lowdim_keys if 'robot_tcp_pose' in key and 'wrt' in key}
            for data in tqdm.tqdm(self, leave=False, desc='Calculating inter-gripper relative obs for normalizer'):
                for key in inter_gripper_data_dict.keys():
                    inter_gripper_data_dict[key].append(data['obs'][key])
            inter_gripper_data_dict = dict_apply(inter_gripper_data_dict, np.stack)

        if self.relative_action:
            relative_data_dict = {key: list() for key in (self.lowdim_keys + ['action']) if ('robot_tcp_pose' in key and 'wrt' not in key) or 'action' in key}
            for data in tqdm.tqdm(self, leave=False, desc='Calculating relative action/obs for normalizer'):
                for key in relative_data_dict.keys():
                    if key == 'action':
                        relative_data_dict[key].append(data[key])
                    else:
                        relative_data_dict[key].append(data['obs'][key])
            relative_data_dict = dict_apply(relative_data_dict, np.stack)

        # action
        if self.relative_action:
            action_all = relative_data_dict['action']
        else:
            action_all = self._get_training_rows('action', self.shape_meta['action']['shape'][0])
        if self.action_normalizer_version == "zero_centered_v2":
            action_all = action_all[self._get_training_mask("action_valid")]

        normalizer['action'] = get_action_normalizer(
            action_all,
            bimanual_contiguous=self.bimanual_contiguous_action,
            version=self.action_normalizer_version,
        )

        # obs
        for key in list(set(self.lowdim_keys)):
            if self.relative_action and key in relative_data_dict:
                normalizer[key] = get_action_normalizer(relative_data_dict[key])
            elif 'robot_tcp_pose' in key and 'wrt' in key:
                normalizer[key] = get_action_normalizer(inter_gripper_data_dict[key])
            elif 'robot_tcp_pose' in key and 'wrt' not in key:
                normalizer[key] = get_action_normalizer(self._get_training_rows(key, self.shape_meta['obs'][key]['shape'][0]))
            else:
                normalizer[key] = SingleFieldLinearNormalizer.create_fit(
                    self._get_training_rows(key, self.shape_meta['obs'][key]['shape'][0]))

        for key in list(set(self.extended_lowdim_keys)):
            if key in self.lowdim_keys:
                assert self.shape_meta['extended_obs'][key]['shape'][0] == self.shape_meta['obs'][key]['shape'][0], \
                    f"Extended obs {key} has different shape from obs {key}"
            else:
                if self.relative_action and key in relative_data_dict:
                    normalizer[key] = get_action_normalizer(relative_data_dict[key])
                elif 'robot_tcp_pose' in key and 'wrt' in key:
                    normalizer[key] = get_action_normalizer(inter_gripper_data_dict[key])
                elif 'robot_tcp_pose' in key and 'wrt' not in key: # not used now
                    normalizer[key] = get_action_normalizer(self._get_training_rows(key, self.shape_meta['extended_obs'][key]['shape'][0]))
                else:
                    normalizer[key] = SingleFieldLinearNormalizer.create_fit(
                        self._get_training_rows(key, self.shape_meta['extended_obs'][key]['shape'][0]))

        # image
        for key in list(set(self.rgb_keys + self.extended_rgb_keys)):
            normalizer[key] = get_image_range_normalizer()
        return normalizer

    def get_all_actions(self) -> torch.Tensor:
        return torch.from_numpy(self.replay_buffer['action'][:, :self.shape_meta['action']['shape'][0]])

    def __len__(self):
        return len(self.sampler)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        threadpool_limits(1)
        data = self.sampler.sample_sequence(idx)

        if self.has_v2_action_contract:
            valid_mask = data["action_valid"].astype(bool, copy=False)
            idle_arm_mask = data["idle_arm_mask"].astype(bool, copy=False)
        else:
            valid_mask = np.zeros(self.sampler.sequence_length, dtype=bool)
            _, _, sample_start, sample_end = self.sampler.indices[idx]
            valid_mask[sample_start:sample_end] = True
            idle_arm_mask = np.zeros((self.sampler.sequence_length, 2), dtype=bool)

        # to save RAM, only return first n_obs_steps of OBS
        # since the rest will be discarded anyway.
        # when self.n_obs_steps is None
        # this slice does nothing (takes all)
        T_slice = slice(self.n_obs_steps)
        obs_downsample_ratio = self.obs_downsample_ratio

        obs_dict = dict()
        for key in self.rgb_keys:
            # move channel last to channel first
            # T,H,W,C
            # convert uint8 image to float32
            obs_dict[key] = np.moveaxis(data[key][T_slice][::-obs_downsample_ratio][::-1],-1,1
                ).astype(np.float32) / 255.
            # T,C,H,W
            # save ram
            if key not in self.rgb_keys:
                del data[key]
        for key in self.lowdim_keys:
            if 'wrt' not in key:
                obs_dict[key] = data[key][:, :self.shape_meta['obs'][key]['shape'][0]][T_slice][::-obs_downsample_ratio][::-1].astype(np.float32)
                # save ram
                if key not in self.extended_lowdim_keys:
                    del data[key]

        # inter-gripper relative action
        if any('wrt' in key for key in self.lowdim_keys):
            from reactive_diffusion_policy.common.action_utils import get_inter_gripper_actions
            obs_dict.update(get_inter_gripper_actions(obs_dict, self.lowdim_keys, self.transforms))
        for key in ['left_robot_wrt_right_robot_tcp_pose', 'right_robot_wrt_left_robot_tcp_pose']:
            if key in obs_dict:
                obs_dict[key] = obs_dict[key][:, :self.shape_meta['obs'][key]['shape'][0]].astype(np.float32)
        
        extended_obs_dict = dict()
        for key in self.extended_rgb_keys:
            extended_obs_dict[key] = np.moveaxis(data[key],-1,1
                ).astype(np.float32) / 255.
            del data[key]
        for key in self.extended_lowdim_keys:
            if 'wrt' not in key:
                extended_obs_dict[key] = data[key][:, :self.shape_meta['extended_obs'][key]['shape'][0]].astype(np.float32)
                del data[key]

        action = data['action'][:, :self.shape_meta['action']['shape'][0]].astype(np.float32)
        # handle latency by dropping first n_latency_steps action
        # observations are already taken care of by T_slice
        if self.n_latency_steps > 0:
            action = action[self.n_latency_steps:]
            valid_mask = valid_mask[self.n_latency_steps:]
            idle_arm_mask = idle_arm_mask[self.n_latency_steps:]
        
        if self.relative_action:
            from reactive_diffusion_policy.common.action_utils import absolute_actions_to_relative_actions
            base_absolute_action = np.concatenate([
                obs_dict['left_robot_tcp_pose'][-1] if 'left_robot_tcp_pose' in obs_dict else np.array([]),
                obs_dict['right_robot_tcp_pose'][-1] if 'right_robot_tcp_pose' in obs_dict else np.array([])
            ], axis=-1)
            action = absolute_actions_to_relative_actions(action, base_absolute_action=base_absolute_action)

            if self.relative_tcp_obs_for_relative_action:
                for key in self.lowdim_keys:
                    if 'robot_tcp_pose' in key and 'wrt' not in key:
                        obs_dict[key]  = absolute_actions_to_relative_actions(obs_dict[key], base_absolute_action=obs_dict[key][-1])

        torch_data = {
            'obs': dict_apply(obs_dict, torch.from_numpy),
            'action': torch.from_numpy(action),
            'extended_obs': dict_apply(extended_obs_dict, torch.from_numpy),
            'valid_mask': torch.from_numpy(valid_mask),
            'idle_arm_mask': torch.from_numpy(idle_arm_mask),
        }
        return torch_data

def test():
    import hydra
    from hydra import initialize, compose
    from omegaconf import OmegaConf
    OmegaConf.register_new_resolver("eval", eval, replace=True)

    with initialize('../config'):
        cfg = hydra.compose('train_diffusion_unet_real_image_workspace',
                            overrides=['task=real_peel_image_gelsight_emb_absolute_12fps'])
        OmegaConf.resolve(cfg)
        dataset = hydra.utils.instantiate(cfg.task.dataset)

    from matplotlib import pyplot as plt
    normalizer = dataset.get_normalizer()

    for i in range(len(dataset)):
        data = dataset[i]

if __name__ == '__main__':
    test()
