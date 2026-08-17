# pi0.5 + FRS deployment

This client uses the same `vb3_robot_server` and `frs_steering_v1` wire protocol as
`FRS_Tact/deploy_smolvla`. Only the remote policy implementation changes from SmolVLA to
JAX pi0.5.

## Before running

Edit `configs/deploy_pi05_frs.yaml` and set these paths to the artifacts from one training run:

- `checkpoint`: the fine-tuned pi0.5 Orbax checkpoint root (the directory containing `params/`);
- `norm_stats.dir` and `norm_stats.asset_id`: the exact stats used when building the FRS cache;
- `frs.checkpoint`: the trained FRS checkpoint directory;
- `frs.tactile_encoder_checkpoint`: the frozen ResNet encoder checkpoint.

The example assumes the robot still sends `camera0`, `camera1`, four tactile images, and a
20-dimensional bimanual state. pi0.5 predicts `[50, 32]`; FRS also operates in 32 dimensions,
then the client unnormalizes and sends one 20-dimensional action to the robot per steering request.
The unavailable `base_0_rgb` slot is explicitly listed under `model.empty_cameras` and is converted
to a black, masked-out model input by the pi0.5 transform.

For a LoRA checkpoint, keep both model variants ending in `_lora`. A base/full checkpoint should use
`gemma_2b` and `gemma_300m` instead.

## Start

```bash
export VB_ROBOT_TOKEN='...'
bash deploy_pi05_frs/scripts/start_pi05_frs.sh --check
bash deploy_pi05_frs/scripts/start_pi05_frs.sh
```

The client first receives one observation, checks checkpoint/cache/FRS contracts, compiles a warmup
run, and waits for Enter before sending `START` unless `runtime.auto_start` is enabled.
