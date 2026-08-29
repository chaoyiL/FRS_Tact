if __name__ == "__main__":
    import sys
    import os
    import pathlib

    ROOT_DIR = str(pathlib.Path(__file__).parent.parent.parent)
    sys.path.append(ROOT_DIR)
    os.chdir(ROOT_DIR)

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
from reactive_diffusion_policy.workspace.base_workspace import BaseWorkspace
from reactive_diffusion_policy.policy.diffusion_unet_image_policy import DiffusionUnetImagePolicy
from reactive_diffusion_policy.dataset.base_dataset import BaseImageDataset
from reactive_diffusion_policy.common.checkpoint_util import (
    PeriodicCheckpointManager,
    TopKCheckpointManager,
)
from reactive_diffusion_policy.common.json_logger import JsonLogger
from reactive_diffusion_policy.common.pytorch_util import dict_apply, optimizer_to
from reactive_diffusion_policy.model.diffusion.ema_model import EMAModel
from reactive_diffusion_policy.model.common.lr_scheduler import get_scheduler
from reactive_diffusion_policy.model.common.lr_decay import param_groups_lrd
from reactive_diffusion_policy.common.artifact_manifest import (
    build_normalizer_cache_signature,
    load_normalizer_cache,
    save_normalizer_cache,
)
from reactive_diffusion_policy.common.pick_tube_validation import (
    compute_idle_rollout_metrics,
    evaluate_checkpoint_feasibility,
    load_active_metric_baselines,
    preserve_global_rng_state,
    validate_resume_action_contract,
)
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs
from reactive_diffusion_policy.workspace.train_at_workspace import (
    get_effective_num_batches,
    get_legacy_optimizer_step,
    get_num_training_steps,
    should_optimizer_step,
)

OmegaConf.register_new_resolver("eval", eval, replace=True)

class TrainDiffusionUnetImageWorkspace(BaseWorkspace):
    include_keys = ['global_step', 'optimizer_step', 'epoch']

    def __init__(self, cfg: OmegaConf, output_dir=None):
        super().__init__(cfg, output_dir=output_dir)

        # set seed
        seed = cfg.training.seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        # configure model
        self.model: DiffusionUnetImagePolicy = hydra.utils.instantiate(cfg.policy)

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
            optimizer_cfg.pop('encoder_weight_decay')
            # hack: use larger learning rate for multiple gpus
            accelerator = Accelerator()
            cuda_count = accelerator.num_processes
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
        self.optimizer_step = 0
        self.epoch = 0

    def run(self):
        cfg = copy.deepcopy(self.cfg)

        if cfg.training.debug:
            cfg.training.num_epochs = 2
            cfg.training.max_train_steps = 3
            cfg.training.max_val_steps = 3
            cfg.training.rollout_every = 1
            cfg.training.checkpoint_every = 1
            cfg.training.val_every = 1
            cfg.training.sample_every = 1

        active_baselines = load_active_metric_baselines(cfg)

        accelerator = Accelerator(
            log_with='wandb',
            kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=True)],
        )
        wandb_cfg = OmegaConf.to_container(cfg.logging, resolve=True)
        wandb_cfg.pop('project')
        accelerator.init_trackers(
            project_name=cfg.logging.project,
            config=OmegaConf.to_container(cfg, resolve=True),
            init_kwargs={"wandb": wandb_cfg}
        )

        # resume training
        resumed = False
        resumed_optimizer_step = False
        if cfg.training.resume:
            lastest_ckpt_path = self.get_checkpoint_path()
            if lastest_ckpt_path.is_file():
                accelerator.print(f"Resuming from checkpoint {lastest_ckpt_path}")
                payload = self.load_checkpoint(path=lastest_ckpt_path)
                validate_resume_action_contract(cfg, payload.get("cfg"))
                resumed_optimizer_step = "optimizer_step" in payload.get("pickles", {})
                self.advance_training_state_for_resume()
                resumed = True
                accelerator.print(
                    f"Continuing at epoch {self.epoch}, "
                    f"global step {self.global_step}"
                )

        # configure dataset
        dataset: BaseImageDataset
        dataset = hydra.utils.instantiate(cfg.task.dataset)
        assert isinstance(dataset, BaseImageDataset)
        OmegaConf.update(
            self.cfg,
            "validation_split",
            dataset.split_manifest,
            merge=False,
            force_add=True,
        )
        train_dataloader = DataLoader(dataset, **cfg.dataloader)
        if resumed and not resumed_optimizer_step:
            self.optimizer_step = get_legacy_optimizer_step(
                completed_epochs=self.epoch,
                num_batches=len(train_dataloader),
                max_train_steps=cfg.training.max_train_steps,
                accumulate_every=cfg.training.gradient_accumulate_every,
            )
        
        at_path_value = OmegaConf.select(cfg, "at_load_dir")
        if not at_path_value:
            raise ValueError("LDP training requires at_load_dir for artifact binding")
        at_path = pathlib.Path(str(at_path_value))
        normalizer_signature = build_normalizer_cache_signature(cfg, dataset, at_path)
        normalizer_path = pathlib.Path(self.output_dir) / "normalizer.pkl"
        if accelerator.is_main_process:
            normalizer = load_normalizer_cache(normalizer_path, normalizer_signature)
            if normalizer is not None:
                accelerator.print(f"Reusing normalizer from {normalizer_path}")
            else:
                normalizer = dataset.get_normalizer()
                save_normalizer_cache(normalizer_path, normalizer, normalizer_signature)

        # load normalizer on all processes
        accelerator.wait_for_everyone()
        normalizer = load_normalizer_cache(normalizer_path, normalizer_signature)
        if normalizer is None:
            raise RuntimeError("normalizer cache was not published by the main process")
        self.bind_checkpoint_artifacts(
            normalizer_signature,
            normalizer=normalizer,
            normalizer_path=normalizer_path,
            role="LDP",
        )

        # configure validation dataset
        val_dataset = dataset.get_validation_dataset()
        val_dataloader = DataLoader(val_dataset, **cfg.val_dataloader)

        self.model.set_normalizer(normalizer)
        if cfg.training.use_ema:
            self.ema_model.set_normalizer(normalizer)

        # configure lr scheduler
        lr_scheduler = get_scheduler(
            cfg.training.lr_scheduler,
            optimizer=self.optimizer,
            num_warmup_steps=cfg.training.lr_warmup_steps,
            num_training_steps=get_num_training_steps(
                num_batches=len(train_dataloader),
                max_train_steps=cfg.training.max_train_steps,
                accumulate_every=cfg.training.gradient_accumulate_every,
                num_epochs=cfg.training.num_epochs,
            ),
            # pytorch assumes stepping LRScheduler every epoch
            # however huggingface diffusers steps it every batch
            last_epoch=self.optimizer_step - 1
        )

        # configure ema
        ema: EMAModel = None
        if cfg.training.use_ema:
            ema = hydra.utils.instantiate(
                cfg.ema,
                model=self.ema_model)
            if resumed:
                # Continue EMA warmup from the restored training position.
                # Otherwise the first update overwrites the restored EMA.
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
        periodic_manager = PeriodicCheckpointManager(
            save_dir=os.path.join(self.output_dir, 'checkpoints', 'periodic'),
            **cfg.checkpoint.periodic,
        )

        # accelerator
        train_dataloader, val_dataloader, self.model, self.optimizer, lr_scheduler = accelerator.prepare(
            train_dataloader, val_dataloader, self.model, self.optimizer, lr_scheduler
        )

        # device transfer
        device = accelerator.device
        if self.ema_model is not None:
            self.ema_model.to(device)

        # save batch for sampling
        train_sampling_batch = None

        num_train_batches = get_effective_num_batches(
            len(train_dataloader),
            cfg.training.max_train_steps,
        )

        num_epochs_to_run = self.get_remaining_epochs(cfg.training.num_epochs)
        if resumed:
            accelerator.print(
                f"Remaining epochs: {num_epochs_to_run} "
                f"(target total: {cfg.training.num_epochs})"
            )

        # training loop
        log_name = 'logs.json.txt' if accelerator.is_main_process else f'logs.rank{accelerator.process_index}.json.txt'
        log_path = os.path.join(self.output_dir, log_name)
        with JsonLogger(log_path) as json_logger:
            for local_epoch_idx in range(num_epochs_to_run):
                step_log = dict()
                # ========= train for this epoch ==========
                if cfg.training.freeze_encoder:
                    self.model.obs_encoder.eval()
                    self.model.obs_encoder.requires_grad_(False)

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
                        group_start = (
                            batch_idx // cfg.training.gradient_accumulate_every
                        ) * cfg.training.gradient_accumulate_every
                        group_size = min(
                            cfg.training.gradient_accumulate_every,
                            num_train_batches - group_start,
                        )
                        loss = raw_loss / group_size
                        accelerator.backward(loss)

                        # step optimizer
                        if should_optimizer_step(
                            batch_idx,
                            num_train_batches,
                            cfg.training.gradient_accumulate_every,
                        ):
                            self.optimizer.step()
                            lr_scheduler.step()
                            if cfg.training.use_ema:
                                ema.step(accelerator.unwrap_model(self.model))
                            self.optimizer.zero_grad()
                            self.optimizer_step += 1

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

                        is_last_batch = batch_idx == num_train_batches - 1
                        if not is_last_batch:
                            # log of last step is combined with validation and rollout
                            accelerator.log(step_log, step=self.global_step)
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
                train_policy = accelerator.unwrap_model(self.model)
                policy = train_policy
                if cfg.training.use_ema:
                    policy = self.ema_model
                policy.eval()

                # run validation
                if cfg.task.dataset.val_ratio > 0 and (self.epoch % cfg.training.val_every) == 0 and accelerator.is_main_process:
                    with torch.no_grad():
                        val_losses = list()
                        val_targets = list()
                        val_predictions = list()
                        val_idle_masks = list()
                        val_valid_masks = list()
                        validation_seeds = [
                            int(seed) for seed in cfg.validation.get("seeds", [0])
                        ]
                        if not validation_seeds:
                            raise ValueError("validation.seeds must not be empty")
                        for validation_seed in validation_seeds:
                            with preserve_global_rng_state(validation_seed):
                                with tqdm.tqdm(
                                    val_dataloader,
                                    desc=(
                                        f"Validation epoch {self.epoch} "
                                        f"seed {validation_seed}"
                                    ),
                                    leave=False,
                                    mininterval=cfg.training.tqdm_interval_sec,
                                ) as tepoch:
                                    for batch_idx, batch in enumerate(tepoch):
                                        batch = dict_apply(
                                            batch,
                                            lambda x: x.to(
                                                device, non_blocking=True
                                            ),
                                        )
                                        # Validate the same policy used for
                                        # sampling and deployment, including EMA.
                                        loss = policy(batch)
                                        val_losses.append(loss.detach())
                                        if 'latent' in cfg.name:
                                            result = policy.predict_action(
                                                batch["obs"],
                                                extended_obs_dict=batch[
                                                    "extended_obs"
                                                ],
                                                dataset_obs_temporal_downsample_ratio=(
                                                    cfg.task.dataset.obs_temporal_downsample_ratio
                                                ),
                                            )
                                        else:
                                            result = policy.predict_action(
                                                batch["obs"]
                                            )
                                        val_targets.append(
                                            batch["action"].detach().cpu()
                                        )
                                        val_predictions.append(
                                            result["action_pred"].detach().cpu()
                                        )
                                        val_idle_masks.append(
                                            batch["idle_arm_mask"].detach().cpu()
                                        )
                                        val_valid_masks.append(
                                            batch["valid_mask"].detach().cpu()
                                        )
                                        if (
                                            cfg.training.max_val_steps is not None
                                            and batch_idx
                                            >= cfg.training.max_val_steps - 1
                                        ):
                                            break
                        if len(val_losses) > 0:
                            val_loss = torch.stack(val_losses).mean().item()
                            # log epoch average validation loss
                            step_log['val_loss'] = val_loss
                            step_log['val_validation_seed_count'] = len(
                                validation_seeds
                            )
                            physical_metrics = compute_idle_rollout_metrics(
                                torch.cat(val_targets),
                                torch.cat(val_predictions),
                                torch.cat(val_idle_masks),
                                horizon=cfg.n_action_steps,
                                valid_mask=torch.cat(val_valid_masks),
                            )
                            step_log.update(physical_metrics)
                            step_log.update(
                                evaluate_checkpoint_feasibility(
                                    idle_translation_29_mm=physical_metrics[
                                        "val_idle_translation_29_mm"
                                    ],
                                    idle_rotation_29_deg=physical_metrics[
                                        "val_idle_rotation_29_deg"
                                    ],
                                    idle_translation_p95_mm=physical_metrics[
                                        "val_idle_translation_p95_mm"
                                    ],
                                    idle_rotation_p95_deg=physical_metrics[
                                        "val_idle_rotation_p95_deg"
                                    ],
                                    active_translation_mm=physical_metrics[
                                        "val_active_left_translation_mae_mm"
                                    ],
                                    active_translation_baseline_mm=(
                                        active_baselines["translation_mm"]
                                        if active_baselines is not None
                                        else None
                                    ),
                                    active_rotation_deg=physical_metrics[
                                        "val_active_left_rotation_mae_deg"
                                    ],
                                    active_rotation_baseline_deg=(
                                        active_baselines["rotation_deg"]
                                        if active_baselines is not None
                                        else None
                                    ),
                                    micro_motion_recall=physical_metrics[
                                        "val_micro_motion_recall"
                                    ],
                                    max_active_degradation=cfg.validation.max_active_degradation,
                                    min_micro_motion_recall=cfg.validation.min_micro_motion_recall,
                                )
                            )

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
                if self.should_save_checkpoint(
                    cfg.training.checkpoint_every,
                    local_epoch_idx,
                    num_epochs_to_run,
                ) and accelerator.is_main_process:
                    # unwrap the model to save ckpt
                    model_ddp = self.model
                    self.model = accelerator.unwrap_model(self.model)

                    # checkpointing
                    if cfg.checkpoint.save_last_ckpt:
                        self.save_checkpoint()
                    if cfg.checkpoint.save_last_snapshot:
                        self.save_snapshot()
                    periodic_ckpt_path = periodic_manager.get_ckpt_path(self.epoch)
                    if periodic_ckpt_path is not None:
                        # Periodic files are the recovery history, so finish the
                        # write before pruning older checkpoints.
                        self.save_checkpoint(path=periodic_ckpt_path, use_thread=False)
                        periodic_manager.prune(periodic_ckpt_path)

                    # sanitize metric names
                    metric_dict = dict()
                    for key, value in step_log.items():
                        new_key = key.replace('/', '_')
                        metric_dict[new_key] = value
                    
                    # We can't copy the last checkpoint here
                    # since save_checkpoint uses threads.
                    # therefore at this point the file might have been empty!
                    topk_ckpt_path = None
                    if metric_dict.get("val_checkpoint_feasible", False):
                        topk_ckpt_path = topk_manager.get_ckpt_path(metric_dict)

                    if topk_ckpt_path is not None:
                        self.save_checkpoint(path=topk_ckpt_path)

                    # recover the DDP model
                    self.model = model_ddp
                    
                # ========= eval end for this epoch ==========
                train_policy.train()
                if self.ema_model is not None:
                    # EMA is an inference/evaluation copy and should never be
                    # left in train mode between validation passes.
                    self.ema_model.eval()

                # end of epoch
                # log of last step is combined with validation and rollout
                accelerator.log(step_log, step=self.global_step)
                json_logger.log(step_log)
                self.global_step += 1
                self.epoch += 1

        accelerator.end_training()

@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.parent.joinpath("config")), 
    config_name=pathlib.Path(__file__).stem)
def main(cfg):
    workspace = TrainDiffusionUnetImageWorkspace(cfg)
    workspace.run()

if __name__ == "__main__":
    main()
