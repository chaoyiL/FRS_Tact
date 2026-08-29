if __name__ == "__main__":
    import sys
    import os
    import pathlib

    ROOT_DIR = str(pathlib.Path(__file__).parent.parent.parent)
    sys.path.append(ROOT_DIR)
    os.chdir(ROOT_DIR)

import os
import math
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

from reactive_diffusion_policy.common.pytorch_util import dict_apply, optimizer_to
from reactive_diffusion_policy.workspace.base_workspace import BaseWorkspace
from reactive_diffusion_policy.model.vae.model import VAE
from reactive_diffusion_policy.dataset.base_dataset import BaseImageDataset
from reactive_diffusion_policy.common.checkpoint_util import (
    PeriodicCheckpointManager,
    TopKCheckpointManager,
)
from reactive_diffusion_policy.common.json_logger import JsonLogger
from reactive_diffusion_policy.model.common.lr_scheduler import get_scheduler
from reactive_diffusion_policy.common.artifact_manifest import (
    build_normalizer_cache_signature,
    load_normalizer_cache,
    save_normalizer_cache,
)
from reactive_diffusion_policy.common.pick_tube_validation import (
    compute_idle_rollout_metrics,
    evaluate_checkpoint_feasibility,
    load_active_metric_baselines,
    reconstruct_at_actions,
    validate_resume_action_contract,
)

OmegaConf.register_new_resolver("eval", eval, replace=True)


def should_optimizer_step(batch_idx, num_batches, accumulate_every):
    accumulate_every = int(accumulate_every)
    if accumulate_every < 1:
        raise ValueError("accumulate_every must be positive")
    return (
        (int(batch_idx) + 1) % accumulate_every == 0
        or int(batch_idx) + 1 == int(num_batches)
    )


def get_effective_num_batches(num_batches, max_train_steps):
    num_batches = int(num_batches)
    if max_train_steps is not None:
        num_batches = min(num_batches, int(max_train_steps))
    return num_batches


def get_num_training_steps(
    num_batches,
    max_train_steps,
    accumulate_every,
    num_epochs,
):
    accumulate_every = int(accumulate_every)
    if accumulate_every < 1:
        raise ValueError("accumulate_every must be positive")
    effective_batches = get_effective_num_batches(
        num_batches,
        max_train_steps,
    )
    return math.ceil(effective_batches / accumulate_every) * int(num_epochs)


def get_legacy_optimizer_step(
    completed_epochs,
    num_batches,
    max_train_steps,
    accumulate_every,
):
    return get_num_training_steps(
        num_batches=num_batches,
        max_train_steps=max_train_steps,
        accumulate_every=accumulate_every,
        num_epochs=completed_epochs,
    )


class TrainATWorkspace(BaseWorkspace):
    include_keys = ['global_step', 'optimizer_step', 'epoch']

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
        self.optimizer_step = 0
        self.epoch = 0

    def run(self):
        cfg = copy.deepcopy(self.cfg)

        if cfg.training.debug:
            cfg.training.num_epochs = 2
            cfg.training.max_train_steps = 3
            cfg.training.max_val_steps = 3
            cfg.training.checkpoint_every = 1
            cfg.training.val_every = 1

        active_baselines = load_active_metric_baselines(cfg)
        active_metric_arm = str(cfg.task.get("controlled_arms", ["left"])[0])

        # resume training
        resumed = False
        resumed_optimizer_step = False
        if cfg.training.resume:
            lastest_ckpt_path = self.get_checkpoint_path()
            if lastest_ckpt_path.is_file():
                print(f"Resuming from checkpoint {lastest_ckpt_path}")
                payload = self.load_checkpoint(path=lastest_ckpt_path)
                validate_resume_action_contract(cfg, payload.get("cfg"))
                resumed_optimizer_step = "optimizer_step" in payload.get("pickles", {})
                self.advance_training_state_for_resume()
                resumed = True
                print(
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
        normalizer_path = pathlib.Path(self.output_dir) / "normalizer.pkl"
        normalizer_signature = build_normalizer_cache_signature(cfg, dataset, None)
        normalizer = load_normalizer_cache(normalizer_path, normalizer_signature)
        if normalizer is None:
            normalizer = dataset.get_normalizer()
            save_normalizer_cache(normalizer_path, normalizer, normalizer_signature)
        else:
            print(f"Reusing normalizer from {normalizer_path}")
        self.bind_checkpoint_artifacts(
            normalizer_signature,
            normalizer=normalizer,
            normalizer_path=normalizer_path,
            role="AT",
        )
        if resumed and not resumed_optimizer_step:
            self.optimizer_step = get_legacy_optimizer_step(
                completed_epochs=self.epoch,
                num_batches=len(train_dataloader),
                max_train_steps=cfg.training.max_train_steps,
                accumulate_every=cfg.training.gradient_accumulate_every,
            )

        # configure validation dataset
        val_dataset = dataset.get_validation_dataset()
        val_dataloader = DataLoader(val_dataset, **cfg.val_dataloader)

        self.model.set_normalizer(normalizer)

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
        periodic_manager = PeriodicCheckpointManager(
            save_dir=os.path.join(self.output_dir, 'checkpoints', 'periodic'),
            **cfg.checkpoint.periodic,
        )

        # device transfer
        device = torch.device(cfg.training.device)
        self.model.to(device)
        optimizer_to(self.optimizer, device)
        use_bf16 = cfg.training.get('mixed_precision') == 'bf16' and device.type == 'cuda'

        # save batch for sampling
        train_sampling_batch = None

        num_train_batches = get_effective_num_batches(
            len(train_dataloader),
            cfg.training.max_train_steps,
        )

        num_epochs_to_run = self.get_remaining_epochs(cfg.training.num_epochs)
        if resumed:
            print(
                f"Remaining epochs: {num_epochs_to_run} "
                f"(target total: {cfg.training.num_epochs})"
            )

        # training loop
        log_path = os.path.join(self.output_dir, 'logs.json.txt')
        with JsonLogger(log_path) as json_logger:
            for local_epoch_idx in range(num_epochs_to_run):
                step_log = dict()
                # ========= train for this epoch ==========
                train_losses = list()
                with tqdm.tqdm(train_dataloader, desc=f"Training epoch {self.epoch}",
                               leave=False, mininterval=cfg.training.tqdm_interval_sec) as tepoch:
                    for batch_idx, batch in enumerate(tepoch):
                        # device transfer
                        batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
                        if train_sampling_batch is None:
                            train_sampling_batch = batch
                        # compute loss
                        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_bf16):
                            loss_metric_dict = self.model.compute_loss_and_metric(batch)
                        raw_loss = loss_metric_dict["loss"]
                        group_start = (
                            batch_idx // cfg.training.gradient_accumulate_every
                        ) * cfg.training.gradient_accumulate_every
                        group_size = min(
                            cfg.training.gradient_accumulate_every,
                            num_train_batches - group_start,
                        )
                        loss = raw_loss / group_size
                        loss.backward()

                        # step optimizer
                        if should_optimizer_step(
                            batch_idx,
                            num_train_batches,
                            cfg.training.gradient_accumulate_every,
                        ):
                            self.optimizer.step()
                            lr_scheduler.step()
                            self.optimizer.zero_grad()
                            self.optimizer_step += 1

                        # logging
                        raw_loss_cpu = raw_loss.item()
                        tepoch.set_postfix(loss=raw_loss_cpu, refresh=False)
                        train_losses.append(raw_loss_cpu)
                        # metric
                        encoder_loss = loss_metric_dict["encoder_loss"]
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
                            vq_loss_state = loss_metric_dict["vq_loss_state"]
                            step_log.update({
                                'train_vq_loss_state': vq_loss_state,
                            })
                        if "kl_loss" in loss_metric_dict:
                            kl_loss = loss_metric_dict["kl_loss"]
                            step_log.update({
                                'train_kl_loss': kl_loss
                            })

                        is_last_batch = batch_idx == num_train_batches - 1
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
                        val_posterior_mean = list()
                        val_posterior_std = list()
                        val_targets = list()
                        val_predictions = list()
                        val_idle_masks = list()
                        val_valid_masks = list()
                        with tqdm.tqdm(val_dataloader, desc=f"Validation epoch {self.epoch}",
                                       leave=False, mininterval=cfg.training.tqdm_interval_sec) as tepoch:
                            for batch_idx, batch in enumerate(tepoch):
                                batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
                                with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_bf16):
                                    loss_metric_dict = self.model.compute_loss_and_metric(
                                        batch,
                                        sample_posterior=False,
                                    )
                                    physical_prediction = reconstruct_at_actions(
                                        policy, batch
                                    )
                                loss = loss_metric_dict["loss"]
                                val_losses.append(loss)
                                val_targets.append(batch["action"].detach().cpu())
                                val_predictions.append(
                                    physical_prediction.detach().cpu()
                                )
                                val_idle_masks.append(
                                    batch["idle_arm_mask"].detach().cpu()
                                )
                                val_valid_masks.append(
                                    batch["valid_mask"].detach().cpu()
                                )
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
                                if "posterior_mean" in loss_metric_dict:
                                    val_posterior_mean.append(
                                        loss_metric_dict["posterior_mean"]
                                    )
                                    val_posterior_std.append(
                                        loss_metric_dict["posterior_std"]
                                    )
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
                            if len(val_posterior_mean) > 0:
                                step_log['val_posterior_mean'] = np.mean(
                                    val_posterior_mean
                                )
                                step_log['val_posterior_std'] = np.mean(
                                    val_posterior_std
                                )
                            physical_metrics = compute_idle_rollout_metrics(
                                torch.cat(val_targets),
                                torch.cat(val_predictions),
                                torch.cat(val_idle_masks),
                                horizon=cfg.n_action_steps,
                                valid_mask=torch.cat(val_valid_masks),
                                state_action_profile=cfg.task.get(
                                    "state_action_profile", None
                                ),
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
                                        f"val_active_{active_metric_arm}_translation_mae_mm"
                                    ],
                                    active_translation_baseline_mm=(
                                        active_baselines["translation_mm"]
                                        if active_baselines is not None
                                        else None
                                    ),
                                    active_rotation_deg=physical_metrics[
                                        f"val_active_{active_metric_arm}_rotation_mae_deg"
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

                # checkpoint
                if self.should_save_checkpoint(
                    cfg.training.checkpoint_every,
                    local_epoch_idx,
                    num_epochs_to_run,
                ):
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
                # ========= eval end for this epoch ==========
                policy.train()

                # end of epoch
                # log of last step is combined with validation and rollout
                wandb_run.log(step_log, step=self.global_step)
                json_logger.log(step_log)
                self.global_step += 1
                self.epoch += 1

@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.parent.joinpath("config")),
    config_name=pathlib.Path(__file__).stem)
def main(cfg):
    workspace = TrainATWorkspace(cfg)
    workspace.run()

if __name__ == "__main__":
    main()
