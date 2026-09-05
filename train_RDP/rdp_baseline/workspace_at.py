"""Derived from original workspace/train_at_workspace.py.

Original objective, optimizer, validation metrics and latest/top-k selection
remain. Engineering fixes cover resume counters, partial gradient accumulation,
device transfer and final-epoch checkpoint persistence; no release gates apply.
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
import math

from reactive_diffusion_policy.common.pytorch_util import dict_apply, optimizer_to
from reactive_diffusion_policy.workspace.base_workspace import BaseWorkspace
from rdp_baseline.model import VAE
from reactive_diffusion_policy.dataset.base_dataset import BaseImageDataset
from reactive_diffusion_policy.common.checkpoint_util import TopKCheckpointManager
from reactive_diffusion_policy.common.json_logger import JsonLogger
from reactive_diffusion_policy.model.common.lr_scheduler import get_scheduler

OmegaConf.register_new_resolver("eval", eval, replace=True)

def optimizer_steps_per_epoch(dataloader, training):
    batches = min(len(dataloader), training.max_train_steps or len(dataloader))
    return math.ceil(batches / int(training.gradient_accumulate_every))

class TrainATWorkspace(BaseWorkspace):
    include_keys = ['global_step', 'epoch', 'optimizer_step']

    def __init__(self, cfg: OmegaConf, output_dir=None):
        super().__init__(cfg, output_dir=output_dir)

        # set seed
        seed = cfg.training.seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        # configure model
        self.model: VAE
        self.model = hydra.utils.instantiate(cfg.policy)

        # configure training state
        self.optimizer = hydra.utils.instantiate(
            cfg.optimizer, params=self.model.optim_params)

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
            cfg.training.checkpoint_every = 1
            cfg.training.val_every = 1


        # resume training
        if cfg.training.resume:
            lastest_ckpt_path = self.get_checkpoint_path()
            if lastest_ckpt_path.is_file():
                print(f"Resuming from checkpoint {lastest_ckpt_path}")
                self.load_checkpoint(path=lastest_ckpt_path, map_location='cpu')
                self.advance_training_state_for_resume()

        # configure dataset
        dataset: BaseImageDataset
        dataset = hydra.utils.instantiate(cfg.task.dataset)
        assert isinstance(dataset, BaseImageDataset)
        train_dataloader = DataLoader(dataset, **cfg.dataloader)
        normalizer = dataset.get_normalizer()

        # configure validation dataset
        val_dataset = dataset.get_validation_dataset()
        val_dataloader = DataLoader(val_dataset, **cfg.val_dataloader)

        self.model.set_normalizer(normalizer)

        # configure lr scheduler
        lr_scheduler = get_scheduler(
            cfg.training.lr_scheduler,
            optimizer=self.optimizer,
            num_warmup_steps=cfg.training.lr_warmup_steps,
            num_training_steps=optimizer_steps_per_epoch(train_dataloader, cfg.training) * cfg.training.num_epochs,
            # pytorch assumes stepping LRScheduler every epoch
            # however huggingface diffusers steps it every batch
            last_epoch=self.optimizer_step - 1
        )

        # configure logging
        wandb_run = wandb.init(
            dir=str(self.output_dir),
            config=OmegaConf.to_container(cfg, resolve=True),
            **cfg.logging
        )
        wandb.config.update(
            {
                "output_dir": self.output_dir,
            }
        )

        # configure checkpoint
        topk_manager = TopKCheckpointManager(
            save_dir=os.path.join(self.output_dir, 'checkpoints'),
            **cfg.checkpoint.topk
        )

        # device transfer
        device = torch.device(cfg.training.device)
        self.model.to(device)
        optimizer_to(self.optimizer, device)

        # save batch for sampling
        train_sampling_batch = None


        self.optimizer.zero_grad(set_to_none=True)
        # training loop
        log_path = os.path.join(self.output_dir, 'logs.json.txt')
        num_epochs_to_run = self.get_remaining_epochs(cfg.training.num_epochs)
        with JsonLogger(log_path) as json_logger:
            for local_epoch_idx in range(num_epochs_to_run):
                step_log = dict()
                # ========= train for this epoch ==========
                self.model.train()
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
                        loss_metric_dict = self.model.compute_loss_and_metric(batch)
                        raw_loss = loss_metric_dict["loss"]
                        group_start = (batch_idx // accumulation) * accumulation
                        group_size = min(accumulation, effective_batches - group_start)
                        loss = raw_loss / group_size
                        loss.backward()

                        # step optimizer
                        if (batch_idx + 1) % accumulation == 0 or batch_idx + 1 == effective_batches:
                            self.optimizer.step()
                            self.optimizer.zero_grad()
                            lr_scheduler.step()
                            self.optimizer_step += 1

                        # logging
                        raw_loss_cpu = raw_loss.item()
                        tepoch.set_postfix(loss=raw_loss_cpu, refresh=False)
                        train_losses.append(raw_loss_cpu)
                        # metric
                        encoder_loss = float(loss_metric_dict["encoder_loss"])
                        vae_recon_loss = loss_metric_dict["vae_recon_loss"]
                        step_log = {
                            'train_loss': raw_loss_cpu,
                            'global_step': self.global_step,
                            'epoch': self.epoch,
                            'lr': lr_scheduler.get_last_lr()[0],
                            # metric
                            'train_encoder_loss': encoder_loss,
                            'train_vae_recon_loss': vae_recon_loss
                        }
                        if "vq_code" in loss_metric_dict:
                            n_different_codes = len(torch.unique(loss_metric_dict["vq_code"]))
                            n_different_combinations = len(torch.unique(loss_metric_dict["vq_code"], dim=0))
                            step_log.update({
                                'train_n_different_codes': n_different_codes,
                                'train_n_different_combinations': n_different_combinations,
                            })
                        if "vq_loss_state" in loss_metric_dict:
                            vq_loss_state = float(loss_metric_dict["vq_loss_state"])
                            step_log.update({
                                'train_vq_loss_state': vq_loss_state,
                            })
                        if "kl_loss" in loss_metric_dict:
                            kl_loss = float(loss_metric_dict["kl_loss"])
                            step_log.update({
                                'train_kl_loss': kl_loss
                            })

                        is_last_batch = (batch_idx + 1 == effective_batches)
                        if not is_last_batch:
                            # log of last step is combined with validation and rollout
                            wandb_run.log(step_log, step=self.global_step)
                            json_logger.log(step_log)
                            self.global_step += 1

                        if (cfg.training.max_train_steps is not None) \
                                and batch_idx >= (cfg.training.max_train_steps - 1):
                            break

                # at the end of each epoch
                # replace train_loss with epoch average
                train_loss = np.mean(train_losses)
                step_log['train_loss'] = train_loss

                # ========= eval for this epoch ==========
                policy = self.model
                policy.eval()

                # run validation
                if (self.epoch % cfg.training.val_every) == 0:
                    with torch.no_grad():
                        val_losses = list()
                        # metric
                        val_n_different_codes = list()
                        val_n_different_combinations = list()
                        val_vq_loss_state = list()
                        val_kl_loss = list()
                        val_encoder_loss = list()
                        val_vae_recon_loss = list()
                        with tqdm.tqdm(val_dataloader, desc=f"Validation epoch {self.epoch}",
                                       leave=False, mininterval=cfg.training.tqdm_interval_sec) as tepoch:
                            for batch_idx, batch in enumerate(tepoch):
                                batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
                                loss_metric_dict = self.model.compute_loss_and_metric(batch)
                                loss = loss_metric_dict["loss"]
                                val_losses.append(loss)
                                # metric
                                val_encoder_loss.append(loss_metric_dict["encoder_loss"])
                                val_vae_recon_loss.append(loss_metric_dict["vae_recon_loss"])
                                if "vq_code" in loss_metric_dict:
                                    val_n_different_codes.append(len(torch.unique(loss_metric_dict["vq_code"])))
                                    val_n_different_combinations.append(
                                        len(torch.unique(loss_metric_dict["vq_code"], dim=0)))
                                if "vq_loss_state" in loss_metric_dict:
                                    val_vq_loss_state.append(loss_metric_dict["vq_loss_state"])
                                if "kl_loss" in loss_metric_dict:
                                    val_kl_loss.append(loss_metric_dict["kl_loss"])
                                if (cfg.training.max_val_steps is not None) \
                                        and batch_idx >= (cfg.training.max_val_steps - 1):
                                    break
                        if len(val_losses) > 0:
                            val_loss = torch.mean(torch.tensor(val_losses)).item()
                            # log epoch average validation loss
                            step_log['val_loss'] = val_loss
                            # metric
                            step_log['val_encoder_loss'] = np.mean(val_encoder_loss)
                            step_log['val_vae_recon_loss'] = np.mean(val_vae_recon_loss)
                            if len(val_n_different_codes) > 0:
                                step_log['val_n_different_codes'] = np.mean(val_n_different_codes)
                                step_log['val_n_different_combinations'] = np.mean(val_n_different_combinations)
                            if len(val_vq_loss_state) > 0:
                                step_log['val_vq_loss_state'] = np.mean(val_vq_loss_state)
                            if len(val_kl_loss) > 0:
                                step_log['val_kl_loss'] = np.mean(val_kl_loss)

                # checkpoint
                if self.should_save_checkpoint(cfg.training.checkpoint_every, local_epoch_idx, num_epochs_to_run):
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
                # ========= eval end for this epoch ==========
                self.model.train()

                # end of epoch
                # log of last step is combined with validation and rollout
                wandb_run.log(step_log, step=self.global_step)
                json_logger.log(step_log)
                self.global_step += 1
                self.epoch += 1

        wandb_run.finish()
