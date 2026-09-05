"""The new entry point must select original objectives and an isolated contract."""
import os
import subprocess
from pathlib import Path
import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
ROOT=Path(__file__).resolve().parents[1]

def config(name, overrides=None):
    OmegaConf.register_new_resolver('eval',eval,replace=True)
    with initialize_config_dir(version_base=None,config_dir=str(ROOT/'rdp_baseline/config')):
        return compose(config_name=name, overrides=overrides or [])

def test_baseline_configs_select_original_methods_without_experimental_overrides():
    at=config('train_at');ldp=config('train_ldp')
    assert at.policy._target_=='rdp_baseline.model.VAE'
    assert ldp.policy._target_=='rdp_baseline.policy.LatentDiffusionUnetImagePolicy'
    assert at.policy.kl_multiplier==1e-6
    assert at.at.policy.n_latent_dims==4 and at.at.policy.rnn_latent_dims==32
    assert ldp.training.use_ema is True
    assert ldp.policy.noise_scheduler.prediction_type=='epsilon'
    assert ldp.policy.num_inference_steps==100
    assert list(ldp.policy.obs_encoder.random_transforms)==[{'type':'RandomCrop','ratio':.9}]
    for cfg in [at,ldp]:
        assert cfg.action_contract=='single_right_chunk_relative10d_v1'
        assert cfg.task.dataset._target_=='rdp_baseline.dataset.ChunkRelativeDataset'
        assert cfg.shape_meta.extended_obs.tactile_embedding.shape==[15]
        assert cfg.shape_meta.action.shape==[10]
        assert cfg.checkpoint.topk.monitor_key=='train_loss'
        text=OmegaConf.to_yaml(cfg)
        for trick in ['physical_v2','zero_centered_v2','micro_motion_weight','idle_weight','release_validation','baseline_json','photometric_augmentation']:
            assert trick not in text

@pytest.mark.parametrize('task,arms,action_dim,touch_dim,cameras', [
    ('insert','right',10,15,['camera2']),
    ('press','right',10,15,['camera2']),
    ('two_tubes','both',20,30,['camera1','camera2']),
    ('bread','both',20,30,['camera1','camera2']),
    ('single_right','right',10,15,['camera2']),
    ('dual_arm','both',20,30,['camera1','camera2']),
])
def test_task_selection_resolves_at_and_ldp_contracts(task,arms,action_dim,touch_dim,cameras):
    for role in ['at','ldp']:
        cfg=config('train_'+role, ['task='+task])
        OmegaConf.resolve(cfg)
        assert cfg.task.dataset.arms==arms
        assert cfg.shape_meta.action.shape==[action_dim]
        assert cfg.shape_meta.extended_obs.tactile_embedding.shape==[touch_dim]
        assert cfg.policy.shape_meta==cfg.task.dataset.shape_meta
        assert sorted(k for k,v in cfg.shape_meta.obs.items() if v.type=='rgb') == (cameras if role=='ldp' else [])
        assert ('left_robot_tcp_pose' in cfg.shape_meta.obs)==(arms=='both')
        assert cfg.policy._target_.startswith('rdp_baseline.')

def test_launcher_runs_from_any_directory_and_uses_plain_latest(tmp_path):
    env=dict(os.environ,DRY_RUN='1',PYTHON_BIN='/usr/bin/python3',RUN_ID='entry_test',OUTPUT_ROOT=str(tmp_path/'results'))
    proc=subprocess.run(['bash',str(ROOT/'scripts/train_rdp_baseline.sh'),'all'],cwd=tmp_path,env=env,text=True,capture_output=True)
    assert proc.returncode==0,proc.stderr
    assert '--config-name=train_at' in proc.stdout
    assert '--config-name=train_ldp' in proc.stdout
    assert '/at/checkpoints/latest.ckpt' in proc.stdout
    assert 'deployable' not in proc.stdout
    assert 'BASELINE_JSON' not in proc.stdout
