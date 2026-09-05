"""Derived from original workspace/train_diffusion_unet_image_workspace.py.

Original diffusion objective, optimizer/LR scaling, EMA and ordinary metrics
remain. Engineering fixes cover streaming sampled-latent normalization, device
and distributed execution, accumulation, resume and final-epoch checkpoints.
FP32 is the default; Accelerate mixed precision is an explicit option.
"""
import os
import hydra
import torch
from omegaconf import OmegaConf
import pathlib
from torch.utils.data import DataLoader
import copy
import random
import wandb
import tqdm
import numpy as np
import shutil
import pickle
from contextlib import nullcontext
from reactive_diffusion_policy.workspace.base_workspace import BaseWorkspace
from reactive_diffusion_policy.policy.diffusion_unet_image_policy import DiffusionUnetImagePolicy
from reactive_diffusion_policy.dataset.base_dataset import BaseImageDataset
from reactive_diffusion_policy.common.checkpoint_util import TopKCheckpointManager
from reactive_diffusion_policy.common.json_logger import JsonLogger
from reactive_diffusion_policy.common.pytorch_util import dict_apply, optimizer_to
from reactive_diffusion_policy.model.diffusion.ema_model import EMAModel
from reactive_diffusion_policy.model.common.lr_scheduler import get_scheduler
from reactive_diffusion_policy.model.common.lr_decay import param_groups_lrd
from accelerate import Accelerator
from rdp_baseline.workspace_at import optimizer_steps_per_epoch
from reactive_diffusion_policy.model.common.normalizer import SingleFieldLinearNormalizer

OmegaConf.register_new_resolver("eval", eval, replace=True)

@torch.no_grad()
def fit_latent_normalizer(dataset, at, normalizer, batch_size=256,
                          use_latent_action_before_vq=False):
    """Fit upstream limits statistics on sampled posterior targets, without RGB.

    Batching and streaming moments avoid retaining the complete latent matrix.
    Each dataset sample is encoded once; no posterior-mode substitution or
    action-dependent weighting is applied.
    """
    at.set_normalizer(normalizer)
    at.eval()
    count = 0
    minimum = maximum = mean = m2 = None
    for start in range(0, len(dataset), batch_size):
        batch = dataset.get_lowdim_batch(range(start, min(start + batch_size, len(dataset))))
        actions = torch.as_tensor(batch['action'], dtype=torch.float32, device=at.device)
        encoded = at.encoder(at.preprocess(at.normalizer['action'].normalize(actions) / at.act_scale))
        if at.use_vq:
            latent = encoded if use_latent_action_before_vq else at.quant_state_with_vq(encoded)[0]
            dim = at.n_latent_dims
        else:
            latent, _ = at.quant_state_without_vq(encoded)
            dim = at.n_embed
        rows = latent.reshape(-1, dim).detach().cpu().double()
        n = len(rows)
        batch_mean = rows.mean(dim=0)
        batch_m2 = ((rows - batch_mean) ** 2).sum(dim=0)
        if count == 0:
            minimum, maximum = rows.amin(0), rows.amax(0)
            mean, m2 = batch_mean, batch_m2
        else:
            minimum = torch.minimum(minimum, rows.amin(0))
            maximum = torch.maximum(maximum, rows.amax(0))
            delta = batch_mean - mean
            m2 += batch_m2 + delta.square() * (count * n / (count + n))
            mean += delta * (n / (count + n))
        count += n
    if count < 2:
        raise ValueError('latent normalizer requires at least two latent samples')
    stats = {key: value.float() for key, value in
             {'min': minimum, 'max': maximum, 'mean': mean, 'std': (m2 / (count - 1)).sqrt()}.items()}
    input_range = stats['max'] - stats['min']
    constant = input_range < 1e-4
    input_range[constant] = 2.
    scale = 2. / input_range
    offset = -1. - scale * stats['min']
    offset[constant] = -stats['min'][constant]
    normalizer['latent_action'] = SingleFieldLinearNormalizer.create_manual(scale, offset, stats)
    return normalizer

class TrainDiffusionUnetImageWorkspace(BaseWorkspace):
    include_keys = ['global_step', 'epoch', 'optimizer_step']

    def __init__(self, cfg: OmegaConf, output_dir=None):
        super().__init__(cfg, output_dir=output_dir)

        # set seed
        seed = cfg.training.seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        # configure model
        policy_cfg = copy.deepcopy(cfg.policy)
        if cfg.training.resume and self.get_checkpoint_path().is_file():
            # The LDP checkpoint contains AT weights; a moved/deleted original
            # AT file must not prevent restoring a baseline training session.
            policy_cfg.at.load_dir = None
        self.model: DiffusionUnetImagePolicy = hydra.utils.instantiate(policy_cfg)

        self.ema_model: DiffusionUnetImagePolicy = None
        if cfg.training.use_ema:
            self.ema_model = copy.deepcopy(self.model)

        # configure training state

        if 'timm' in cfg.policy.obs_encoder._target_:
            if cfg.training.layer_decay < 1.0:
                assert not cfg.policy.obs_encoder.use_lora
                assert not cfg.policy.obs_encoder.share_rgb_model
                obs_encorder_param_groups = param_groups_lrd(self.model.obs_encoder,
                                                             shape_meta=cfg.shape_meta,
                                                             weight_decay=cfg.optimizer.encoder_weight_decay,
                                                             no_weight_decay_list=self.model.obs_encoder.no_weight_decay(),
                                                             layer_decay=cfg.training.layer_decay)
                count = 0
                for group in obs_encorder_param_groups:
                    count += len(group['params'])
                if cfg.policy.obs_encoder.feature_aggregation == 'map':
                    obs_encorder_param_groups.extend([{'params': self.model.obs_encoder.attn_pool.parameters()}])
                    for _ in self.model.obs_encoder.attn_pool.parameters():
                        count += 1
                print(f'obs_encorder params: {count}')
                param_groups = [{'params': self.model.model.parameters()}]
                param_groups.extend(obs_encorder_param_groups)
            else:
                obs_encorder_lr = cfg.optimizer.lr
                if cfg.policy.obs_encoder.pretrained and not cfg.policy.obs_encoder.use_lora:
                    obs_encorder_lr *= cfg.training.encoder_lr_coefficient
                    print('==> reduce pretrained obs_encorder\'s lr')
                obs_encorder_params = list()
                for param in self.model.obs_encoder.parameters():
                    if param.requires_grad:
                        obs_encorder_params.append(param)
                print(f'obs_encorder params: {len(obs_encorder_params)}')
                param_groups = [
                    {'params': self.model.model.parameters()},
                    {'params': obs_encorder_params, 'lr': obs_encorder_lr}
                ]
            optimizer_cfg = OmegaConf.to_container(cfg.optimizer, resolve=True)
            optimizer_cfg.pop('_target_')
            if 'encoder_weight_decay' in optimizer_cfg.keys():
                optimizer_cfg.pop('encoder_weight_decay')
            self.optimizer = torch.optim.AdamW(
                params=param_groups,
                **optimizer_cfg
            )
        else:
            optimizer_cfg = OmegaConf.to_container(cfg.optimizer, resolve=True)
            optimizer_cfg.pop('encoder_weight_decay', None)
            # hack: use larger learning rate for multiple gpus
            cuda_count = int(os.environ.get('WORLD_SIZE', '1'))
            print("###########################################")
            print(f"Number of available CUDA devices: {cuda_count}.")
            print(f"Original learning rate: {optimizer_cfg['lr']}")
            optimizer_cfg['lr'] = optimizer_cfg['lr'] * cuda_count
            print(f"Updated learning rate: {optimizer_cfg['lr']}")
            print("###########################################")
            self.optimizer = hydra.utils.instantiate(
                optimizer_cfg, params=self.model.parameters())

        # configure training state
        self.global_step = 0
        self.epoch = 0
        self.optimizer_step = 0

    def run(self):
        pathlib.Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        cfg = copy.deepcopy(self.cfg)

        if cfg.training.debug:
            cfg.training.num_epochs = 2
            cfg.training.max_train_steps = 3
            cfg.training.max_val_steps = 3
            cfg.training.rollout_every = 1
            cfg.training.checkpoint_every = 1
            cfg.training.val_every = 1
            cfg.training.sample_every = 1


        accelerator = Accelerator(log_with='wandb',
                                  mixed_precision=cfg.training.get('mixed_precision', 'no'),
                                  cpu=str(cfg.training.device) == 'cpu')
        wandb_cfg = OmegaConf.to_container(cfg.logging, resolve=True)
        wandb_cfg.pop('project')
        accelerator.init_trackers(
            project_name=cfg.logging.project,
            config=OmegaConf.to_container(cfg, resolve=True),
            init_kwargs={"wandb": wandb_cfg}
        )

        # resume training
        resumed = False
        if cfg.training.resume:
            lastest_ckpt_path = self.get_checkpoint_path()
            if lastest_ckpt_path.is_file():
                accelerator.print(f"Resuming from checkpoint {lastest_ckpt_path}")
                self.load_checkpoint(path=lastest_ckpt_path, map_location='cpu')
                self.advance_training_state_for_resume()
                resumed = True

        # configure dataset
        dataset: BaseImageDataset
        dataset = hydra.utils.instantiate(cfg.task.dataset)
        assert isinstance(dataset, BaseImageDataset)
        train_dataloader = DataLoader(dataset, **cfg.dataloader)

        # normalizer = dataset.get_normalizer()
        # compute normalizer on the main process and save to disk
        normalizer_path = os.path.join(self.output_dir, 'normalizer.pkl')
        if accelerator.is_main_process:
            if resumed:
                # Re-fitting random posterior samples would change the target
                # coordinates of a policy that has already been trained.
                normalizer = copy.deepcopy(self.model.normalizer).cpu()
            else:
                normalizer = dataset.get_normalizer()
                self.model.at.to(accelerator.device)
                normalizer = fit_latent_normalizer(
                    dataset, self.model.at, normalizer,
                    batch_size=cfg.training.get('latent_normalizer_batch_size', 256),
                    use_latent_action_before_vq=self.model.use_latent_action_before_vq)
            with open(normalizer_path, 'wb') as f:
                pickle.dump(normalizer, f)

        # load normalizer on all processes
        accelerator.wait_for_everyone()
        with open(normalizer_path, 'rb') as f:
            normalizer = pickle.load(f)

        # configure validation dataset
        val_dataset = dataset.get_validation_dataset()
        val_dataloader = DataLoader(val_dataset, **cfg.val_dataloader)

        self.model.set_normalizer(normalizer)
        if cfg.training.use_ema:
            self.ema_model.set_normalizer(normalizer)

        # Prepare each distributed component once. Validation remains complete
        # on rank zero and never forwards through DDP alone.
        self.model.to(accelerator.device)
        train_dataloader, self.model, self.optimizer = accelerator.prepare(
            train_dataloader, self.model, self.optimizer)

        # configure lr scheduler (one explicit step per optimizer update)
        lr_scheduler = get_scheduler(
            cfg.training.lr_scheduler,
            optimizer=self.optimizer,
            num_warmup_steps=cfg.training.lr_warmup_steps,
            num_training_steps=optimizer_steps_per_epoch(train_dataloader, cfg.training) * cfg.training.num_epochs,
            # pytorch assumes stepping LRScheduler every epoch
            # however huggingface diffusers steps it every batch
            last_epoch=self.optimizer_step-1
        )

        # configure ema
        ema: EMAModel = None
        if cfg.training.use_ema:
            ema = hydra.utils.instantiate(
                cfg.ema,
                model=self.ema_model)
            ema.optimization_step = self.optimizer_step

        # configure logging
        # wandb_run = wandb.init(
        #     dir=str(self.output_dir),
        #     config=OmegaConf.to_container(cfg, resolve=True),
        #     **cfg.logging
        # )
        # wandb.config.update(
        #     {
        #         "output_dir": self.output_dir,
        #     }
        # )

        # configure checkpoint
        topk_manager = TopKCheckpointManager(
            save_dir=os.path.join(self.output_dir, 'checkpoints'),
            **cfg.checkpoint.topk
        )

        # device transfer
        device = accelerator.device
        if self.ema_model is not None:
            self.ema_model.to(device)

        # save batch for sampling
        train_sampling_batch = None


        self.optimizer.zero_grad(set_to_none=True)
        # training loop
        log_path = os.path.join(self.output_dir, 'logs.json.txt')
        num_epochs_to_run = self.get_remaining_epochs(cfg.training.num_epochs)
        with (JsonLogger(log_path) if accelerator.is_main_process else nullcontext()) as json_logger:
            for local_epoch_idx in range(num_epochs_to_run):
                step_log = dict()
                # ========= train for this epoch ==========
                self.model.train()
                if cfg.training.freeze_encoder:
                    accelerator.unwrap_model(self.model).obs_encoder.eval()
                    accelerator.unwrap_model(self.model).obs_encoder.requires_grad_(False)

                effective_batches = min(len(train_dataloader), cfg.training.max_train_steps or len(train_dataloader))
                accumulation = int(cfg.training.gradient_accumulate_every)
                train_losses = list()
                with tqdm.tqdm(train_dataloader, desc=f"Training epoch {self.epoch}",
                        leave=False, mininterval=cfg.training.tqdm_interval_sec) as tepoch:
                    for batch_idx, batch in enumerate(tepoch):
                        # device transfer
                        batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
                        if train_sampling_batch is None:
                            train_sampling_batch = batch

                        # compute loss
                        raw_loss = self.model(batch)
                        group_start = (batch_idx // accumulation) * accumulation
                        group_size = min(accumulation, effective_batches - group_start)
                        loss = raw_loss / group_size
                        accelerator.backward(loss)

                        # step optimizer
                        if (batch_idx + 1) % accumulation == 0 or batch_idx + 1 == effective_batches:
                            self.optimizer.step()
                            self.optimizer.zero_grad()
                            lr_scheduler.step()
                            self.optimizer_step += 1
                            if cfg.training.use_ema:
                                ema.step(accelerator.unwrap_model(self.model))

                        # logging
                        raw_loss_cpu = raw_loss.item()
                        tepoch.set_postfix(loss=raw_loss_cpu, refresh=False)
                        train_losses.append(raw_loss_cpu)
                        step_log = {
                            'train_loss': raw_loss_cpu,
                            'global_step': self.global_step,
                            'epoch': self.epoch,
                            'lr': lr_scheduler.get_last_lr()[0]
                        }

                        is_last_batch = (batch_idx + 1 == effective_batches)
                        if not is_last_batch:
                            # log of last step is combined with validation and rollout
                            accelerator.log(step_log, step=self.global_step)
                            if json_logger is not None:
                                json_logger.log(step_log)
                            self.global_step += 1

                        if (cfg.training.max_train_steps is not None) \
                            and batch_idx >= (cfg.training.max_train_steps-1):
                            break

                # at the end of each epoch
                # replace train_loss with epoch average
                train_loss = np.mean(train_losses)
                step_log['train_loss'] = train_loss

                # ========= eval for this epoch ==========
                policy = accelerator.unwrap_model(self.model)
                policy.eval()
                if cfg.training.use_ema:
                    policy = self.ema_model
                policy.eval()

                # run validation
                if cfg.task.dataset.val_ratio > 0 and (self.epoch % cfg.training.val_every) == 0 and accelerator.is_main_process:
                    with torch.no_grad():
                        val_losses = list()
                        with tqdm.tqdm(val_dataloader, desc=f"Validation epoch {self.epoch}",
                                leave=False, mininterval=cfg.training.tqdm_interval_sec) as tepoch:
                            for batch_idx, batch in enumerate(tepoch):
                                batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
                                # Preserve original validation on live weights;
                                # EMA is used for the original sampling metric.
                                with accelerator.autocast():
                                    loss = accelerator.unwrap_model(self.model).compute_loss(batch)
                                val_losses.append(loss)
                                if (cfg.training.max_val_steps is not None) \
                                    and batch_idx >= (cfg.training.max_val_steps-1):
                                    break
                        if len(val_losses) > 0:
                            val_loss = torch.mean(torch.tensor(val_losses)).item()
                            # log epoch average validation loss
                            step_log['val_loss'] = val_loss

                # run diffusion sampling on a training batch
                if (self.epoch % cfg.training.sample_every) == 0:
                    with torch.no_grad():
                        # sample trajectory from training set, and evaluate difference
                        batch = dict_apply(train_sampling_batch, lambda x: x.to(device, non_blocking=True))
                        obs_dict = batch['obs']
                        extended_obs_dict = batch['extended_obs']
                        gt_action = batch['action']

                        if 'latent' in cfg.name:
                            dataset_obs_temporal_downsample_ratio = cfg.task.dataset.obs_temporal_downsample_ratio
                            result = policy.predict_action(obs_dict,
                                                           extended_obs_dict=extended_obs_dict,
                                                           dataset_obs_temporal_downsample_ratio=dataset_obs_temporal_downsample_ratio)
                        else:
                            result = policy.predict_action(obs_dict)
                        pred_action = result['action_pred']

                        all_preds, all_gt = accelerator.gather_for_metrics((pred_action, gt_action))

                        mse = torch.nn.functional.mse_loss(all_preds, all_gt)
                        step_log['train_action_mse_error'] = mse.item()
                        del batch
                        del obs_dict
                        del gt_action
                        del result
                        del pred_action
                        del mse
                accelerator.wait_for_everyone()

                # checkpoint
                if self.should_save_checkpoint(cfg.training.checkpoint_every, local_epoch_idx, num_epochs_to_run) and accelerator.is_main_process:
                    # unwrap the model to save ckpt
                    model_ddp = self.model
                    self.model = accelerator.unwrap_model(self.model)

                    # checkpointing
                    if cfg.checkpoint.save_last_ckpt:
                        self.save_checkpoint(use_thread=False)
                    if cfg.checkpoint.save_last_snapshot:
                        self.save_snapshot()

                    # sanitize metric names
                    metric_dict = dict()
                    for key, value in step_log.items():
                        new_key = key.replace('/', '_')
                        metric_dict[new_key] = value

                    # We can't copy the last checkpoint here
                    # since save_checkpoint uses threads.
                    # therefore at this point the file might have been empty!
                    topk_ckpt_path = topk_manager.get_ckpt_path(metric_dict)

                    if topk_ckpt_path is not None:
                        self.save_checkpoint(path=topk_ckpt_path, use_thread=False)

                    # recover the DDP model
                    self.model = model_ddp

                # ========= eval end for this epoch ==========
                self.model.train()

                # end of epoch
                # log of last step is combined with validation and rollout
                accelerator.log(step_log, step=self.global_step)
                if json_logger is not None:
                    json_logger.log(step_log)
                self.global_step += 1
                self.epoch += 1

        accelerator.end_training()
