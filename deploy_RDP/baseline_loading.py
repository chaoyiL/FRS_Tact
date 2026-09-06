"""Load original-RDP baseline weights without constructing training workspaces."""
from pathlib import Path
import copy
import sys

import hydra
import torch
from omegaconf import OmegaConf

CONTRACTS = {'single_right_chunk_relative10d_v1': ('right', 10, 15),
             'dual_arm_chunk_relative20d_v1': ('both', 20, 30)}


def is_baseline_config(cfg):
    return OmegaConf.select(cfg, 'action_contract') in CONTRACTS


def validate_baseline_metadata(cfg, at_cfg, pca_output_dim, profile, slow_update_interval):
    contract = OmegaConf.select(cfg, 'action_contract')
    if contract not in CONTRACTS or OmegaConf.select(at_cfg, 'action_contract') != contract:
        raise ValueError('Baseline AT/LDP action contracts do not match')
    arms, action_dim, tactile_dim = CONTRACTS[contract]
    if profile.action_dim != action_dim:
        raise ValueError('Baseline checkpoint and configured robot arm profile do not match')
    if pca_output_dim != 30:
        raise ValueError('Baseline requires the original two-arm PCA (2 x 15D)')
    arm_names = ('right',) if arms == 'right' else ('left', 'right')
    lowdim = {'tactile_embedding': [tactile_dim]}
    for arm in arm_names:
        lowdim.update({arm+'_robot_tcp_pose': [9], arm+'_robot_gripper_width': [1]})
    for role, item in [('AT', at_cfg), ('LDP', cfg)]:
        expected = dict(lowdim)
        if role == 'LDP':
            expected['camera2'] = [3, 224, 224]
            if arms == 'both':
                expected['camera1'] = [3, 224, 224]
        actual = OmegaConf.select(item, 'shape_meta.obs')
        if actual is None or set(actual) != set(expected):
            raise ValueError(f'Baseline {role} observation fields do not match {arms} arm schema')
        for key, shape in expected.items():
            if OmegaConf.select(item, f'shape_meta.obs.{key}.shape') != shape:
                raise ValueError(f'Baseline {role} observation {key} has incompatible shape')
        if (OmegaConf.select(item, 'shape_meta.action.shape') != [action_dim]
                or OmegaConf.select(item, 'shape_meta.extended_obs.tactile_embedding.shape') != [tactile_dim]
                or OmegaConf.select(item, 'task.arms') != arms):
            raise ValueError(f'Baseline {role} action/tactile/arm dimensions do not match')
    for field in ('horizon', 'n_obs_steps', 'dataset_obs_temporal_downsample_ratio'):
        value = OmegaConf.select(cfg, field)
        if type(value) is not int or value < 1 or value != OmegaConf.select(at_cfg, field):
            raise ValueError(f'Baseline AT/LDP {field} does not match')
    if OmegaConf.select(cfg, 'tactile_pca_path') != OmegaConf.select(at_cfg, 'tactile_pca_path'):
        raise ValueError('Baseline AT/LDP were configured with different PCA artifacts')
    usable = cfg.horizon - cfg.n_obs_steps * cfg.dataset_obs_temporal_downsample_ratio + 1
    if type(slow_update_interval) is not int or not 1 <= slow_update_interval <= usable:
        raise ValueError(f'Baseline slow interval must be between 1 and {usable}')
    if cfg.n_action_steps != usable:
        raise ValueError('Baseline action horizon does not match its observation history')


def validate_embedded_at(selected_state, at_payload):
    """LDP stores its frozen AT: compare network tensors, not file locations."""
    embedded = selected_state.get('_extra_state', {}).get('at')
    external = at_payload.get('state_dicts', {}).get('model')
    if embedded is None or external is None:
        raise ValueError('Baseline checkpoint is missing the embedded or selected AT weights')
    # LDP adds latent/image normalizers to AT; those keys need not occur in
    # the original AT checkpoint. Network weights must agree exactly.
    for component in ('encoder', 'decoder', 'quant', 'post_quant'):
        left, right = embedded.get(component), external.get(component)
        if left is None or right is None or set(left) != set(right):
            raise ValueError(f'Baseline AT {component} state does not match LDP embedded AT')
        for key in left:
            if left[key].shape != right[key].shape or not torch.equal(left[key].cpu(), right[key].cpu()):
                raise ValueError(f'Baseline selected AT differs from LDP embedded AT: {component}.{key}')


def load_baseline_policy(payload, at_payload, cfg, at_cfg, *, device,
                         num_inference_steps, pca_output_dim, profile, slow_update_interval):
    validate_baseline_metadata(cfg, at_cfg, pca_output_dim, profile, slow_update_interval)
    state_key = 'ema_model' if bool(cfg.training.use_ema) else 'model'
    selected = payload.get('state_dicts', {}).get(state_key)
    if selected is None:
        raise ValueError(f'Baseline LDP checkpoint is missing {state_key}')
    validate_embedded_at(selected, at_payload)
    training_root = Path(__file__).resolve().parents[1] / 'train_RDP'
    if not (training_root / 'rdp_baseline/policy.py').is_file():
        raise FileNotFoundError(f'Baseline inference package is missing: {training_root / "rdp_baseline"}')
    # Append: the already-loaded deployment reactive_diffusion_policy package
    # stays authoritative; only the distinct rdp_baseline package is added.
    if str(training_root) not in sys.path:
        sys.path.append(str(training_root))
    policy_cfg = copy.deepcopy(cfg.policy)
    if policy_cfg._target_ != 'rdp_baseline.policy.LatentDiffusionUnetImagePolicy':
        raise ValueError('Baseline checkpoint has an unexpected policy implementation')
    policy_cfg.at.load_dir = None
    policy_cfg.at.device = 'cpu'
    policy = hydra.utils.instantiate(policy_cfg)
    policy.load_state_dict(selected)
    policy.at.set_normalizer(policy.normalizer)
    policy.num_inference_steps = int(num_inference_steps)
    policy.eval().to(device)
    print(f'[rdp] Loaded baseline {cfg.action_contract}; tactile input={CONTRACTS[cfg.action_contract][2]}D; '
          f'weights={state_key}; embedded AT matches selected AT')
    return policy, cfg
