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
    build_canonical_noop_actions,
    compute_deployment_window_metrics,
    compute_idle_rollout_metrics,
    evaluate_checkpoint_feasibility,
    load_active_metric_baselines,
    reconstruct_at_actions,
    resolve_active_metric_baselines,
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


def get_deployment_phase_window(cfg) -> tuple[int, int]:
    """Return the fixed slow16 decoder window used by robot deployment."""
    phase_count = OmegaConf.select(cfg, "validation.deployment_slow_update_interval")
    n_obs_steps = OmegaConf.select(cfg, "n_obs_steps")
    ratio = OmegaConf.select(cfg, "dataset_obs_temporal_downsample_ratio")
    for name, value in (
        ("validation.deployment_slow_update_interval", phase_count),
        ("n_obs_steps", n_obs_steps),
        ("dataset_obs_temporal_downsample_ratio", ratio),
    ):
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"{name} must be a non-boolean integer")
    if phase_count != 16:
        raise ValueError(
            "validation.deployment_slow_update_interval must remain exactly 16"
        )
    phase_start = n_obs_steps * ratio - 1
    if phase_start != 3:
        raise ValueError("deployment phase_start must remain exactly 3")
    return phase_start, phase_count


def should_update_deployable_checkpoint(passed, score, best_score) -> bool:
    """Return whether a qualified checkpoint improves the release score."""
    try:
        candidate = float(score)
        best = float(best_score)
    except (TypeError, ValueError):
        return False
    return bool(passed) and math.isfinite(candidate) and candidate < best


def _release_scalar(value):
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        value = float(value)
        return value if math.isfinite(value) else None
    return None


def build_release_validation(
    *,
    passed,
    deployment_slow_update_interval,
    phase_start,
    score,
    epoch,
    metrics,
    active_baseline_source=None,
    active_baseline_epoch=None,
) -> dict[str, object]:
    """Create scalar checkpoint evidence that deployment can validate safely."""
    if (
        isinstance(deployment_slow_update_interval, bool)
        or not isinstance(deployment_slow_update_interval, int)
    ):
        raise ValueError("deployment_slow_update_interval must be a non-boolean integer")
    if deployment_slow_update_interval != 16:
        raise ValueError("deployment_slow_update_interval must remain exactly 16")
    if isinstance(phase_start, bool) or not isinstance(phase_start, int):
        raise ValueError("phase_start must be a non-boolean integer")
    if phase_start != 3:
        raise ValueError("phase_start must remain exactly 3")
    return {
        "passed": bool(passed),
        "deployment_slow_update_interval": deployment_slow_update_interval,
        "phase_start": phase_start,
        "score": _release_scalar(score),
        "epoch": int(epoch),
        "active_baseline_source": (
            active_baseline_source
            if active_baseline_source in {"auto", "external"}
            else None
        ),
        "active_baseline_epoch": (
            int(active_baseline_epoch)
            if active_baseline_source == "auto"
            and isinstance(active_baseline_epoch, (int, np.integer))
            and not isinstance(active_baseline_epoch, bool)
            else None
        ),
        "metrics": {
            str(name): _release_scalar(value) for name, value in metrics.items()
        },
    }


def merge_noop_idle_metrics(metrics, noop_metrics) -> dict:
    """Attach only canonical no-op idle metrics without corrupting real metrics."""
    merged = dict(metrics)
    merged.update(
        {
            key.replace("val_deploy_idle_", "val_deploy_noop_idle_", 1): value
            for key, value in noop_metrics.items()
            if key.startswith("val_deploy_idle_")
        }
    )
    return merged


def namespace_deployment_release_metrics(release_metrics) -> dict:
    """Keep deployment qualification output separate from historical metrics."""
    names = {
        "val_active_translation_degradation": "val_deploy_active_translation_degradation",
        "val_active_rotation_degradation": "val_deploy_active_rotation_degradation",
        "val_micro_motion_recall": "val_deploy_micro_motion_recall",
        "val_idle_score": "val_deploy_idle_score",
        "val_checkpoint_feasible": "val_deploy_checkpoint_feasible",
        "val_deployable": "val_deployable",
    }
    return {names[key]: value for key, value in release_metrics.items()}


class TrainATWorkspace(BaseWorkspace):
    include_keys = [
        'global_step', 'optimizer_step', 'epoch', 'best_deploy_idle_score',
        'active_translation_baseline_mm', 'active_rotation_baseline_deg',
        'active_baseline_source', 'active_baseline_epoch',
    ]

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
        self.best_deploy_idle_score = math.inf
        self.active_translation_baseline_mm = None
        self.active_rotation_baseline_deg = None
        self.active_baseline_source = None
        self.active_baseline_epoch = None

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
                        val_noop_targets = list()
                        val_noop_predictions = list()
                        val_noop_idle_masks = list()
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
                                    noop_batch = dict(batch)
                                    noop_batch["action"] = build_canonical_noop_actions(
                                        batch["action"]
                                    )
                                    noop_prediction = reconstruct_at_actions(
                                        policy, noop_batch
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
                                val_noop_targets.append(
                                    noop_batch["action"].detach().cpu()
                                )
                                val_noop_predictions.append(
                                    noop_prediction.detach().cpu()
                                )
                                val_noop_idle_masks.append(
                                    torch.ones_like(
                                        batch["idle_arm_mask"], dtype=torch.bool
                                    ).detach().cpu()
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
                            phase_start, phase_count = get_deployment_phase_window(cfg)
                            deployment_metrics = compute_deployment_window_metrics(
                                torch.cat(val_targets),
                                torch.cat(val_predictions),
                                torch.cat(val_idle_masks),
                                phase_start=phase_start,
                                phase_count=phase_count,
                                valid_mask=torch.cat(val_valid_masks),
                                state_action_profile=cfg.task.get(
                                    "state_action_profile", None
                                ),
                            )
                            noop_metrics = compute_deployment_window_metrics(
                                torch.cat(val_noop_targets),
                                torch.cat(val_noop_predictions),
                                torch.cat(val_noop_idle_masks),
                                phase_start=phase_start,
                                phase_count=phase_count,
                                valid_mask=torch.cat(val_valid_masks),
                                state_action_profile=cfg.task.get(
                                    "state_action_profile", None
                                ),
                            )
                            step_log.update(deployment_metrics)
                            step_log = merge_noop_idle_metrics(
                                step_log, noop_metrics
                            )
                            resolved_baselines = resolve_active_metric_baselines(
                                external_baselines=active_baselines,
                                auto_translation_baseline_mm=(
                                    self.active_translation_baseline_mm
                                ),
                                auto_rotation_baseline_deg=(
                                    self.active_rotation_baseline_deg
                                ),
                                auto_baseline_epoch=self.active_baseline_epoch,
                                active_translation_mm=deployment_metrics[
                                    f"val_deploy_active_{active_metric_arm}_translation_mae_mm"
                                ],
                                active_rotation_deg=deployment_metrics[
                                    f"val_deploy_active_{active_metric_arm}_rotation_mae_deg"
                                ],
                                epoch=self.epoch,
                            )
                            if resolved_baselines["calibrated"]:
                                self.active_translation_baseline_mm = (
                                    resolved_baselines["translation_mm"]
                                )
                                self.active_rotation_baseline_deg = (
                                    resolved_baselines["rotation_deg"]
                                )
                                self.active_baseline_source = "auto"
                                self.active_baseline_epoch = resolved_baselines["epoch"]
                            release_metrics = evaluate_checkpoint_feasibility(
                                idle_translation_29_mm=deployment_metrics[
                                    "val_deploy_idle_translation_window_mm"
                                ],
                                idle_rotation_29_deg=deployment_metrics[
                                    "val_deploy_idle_rotation_window_deg"
                                ],
                                idle_translation_p95_mm=noop_metrics[
                                    "val_deploy_idle_translation_step_p95_mm"
                                ],
                                idle_rotation_p95_deg=noop_metrics[
                                    "val_deploy_idle_rotation_step_p95_deg"
                                ],
                                active_translation_mm=deployment_metrics[
                                    f"val_deploy_active_{active_metric_arm}_translation_mae_mm"
                                ],
                                active_translation_baseline_mm=resolved_baselines[
                                    "translation_mm"
                                ],
                                active_rotation_deg=deployment_metrics[
                                    f"val_deploy_active_{active_metric_arm}_rotation_mae_deg"
                                ],
                                active_rotation_baseline_deg=resolved_baselines[
                                    "rotation_deg"
                                ],
                                micro_motion_recall=deployment_metrics[
                                    "val_deploy_micro_motion_recall"
                                ],
                                max_active_degradation=cfg.validation.max_active_degradation,
                                min_micro_motion_recall=cfg.validation.min_micro_motion_recall,
                            )
                            if resolved_baselines["calibrated"]:
                                release_metrics["val_deployable"] = False
                            step_log.update(
                                namespace_deployment_release_metrics(release_metrics)
                            )

                # checkpoint
                if self.should_save_checkpoint(
                    cfg.training.checkpoint_every,
                    local_epoch_idx,
                    num_epochs_to_run,
                ):
                    release_passed = bool(step_log.get("val_deployable", False))
                    release_score = step_log.get("val_deploy_idle_score")
                    release_phase_start, release_phase_count = get_deployment_phase_window(
                        cfg
                    )
                    OmegaConf.update(
                        self.cfg,
                        "release_validation",
                        build_release_validation(
                            passed=release_passed,
                            deployment_slow_update_interval=release_phase_count,
                            phase_start=release_phase_start,
                            score=release_score,
                            epoch=self.epoch,
                            metrics=step_log,
                            active_baseline_source=(
                                "external"
                                if active_baselines is not None
                                else self.active_baseline_source
                            ),
                            active_baseline_epoch=(
                                None
                                if active_baselines is not None
                                else self.active_baseline_epoch
                            ),
                        ),
                        merge=False,
                        force_add=True,
                    )
                    update_deployable = should_update_deployable_checkpoint(
                        release_passed,
                        release_score,
                        self.best_deploy_idle_score,
                    )
                    if update_deployable:
                        self.best_deploy_idle_score = float(release_score)
                    # latest.ckpt is unconditional recovery state, never a release.
                    self.save_checkpoint()
                    if update_deployable:
                        self.save_checkpoint(
                            path=self.get_checkpoint_path(tag="deployable"),
                            use_thread=False,
                        )
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
                    if metric_dict.get("val_deployable", False):
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
