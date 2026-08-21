# Pi0.5 FRS 8000 Deployment Design

## Goal

Download the inference-only files from
`KaiyueChen/pi05_frs_pick_tube_8000_bimanual_last` and configure the Pi0.5 FRS
deployment to use the matching local Pi0.5, normalization, FRS decoder, and
tactile encoder checkpoints.

## Asset layout

- Pin the FRS repository at revision
  `d2bc8089b4a7e262594612099ef7e593eeb124af`.
- Store `checkpoint.json` and `params.npz` under
  `checkpoints/frs/pi05_frs_pick_tube_8000_bimanual_last/`.
- Store a provenance file containing the repository ID and pinned revision.
- Reuse the existing Pi0.5 checkpoint at
  `/home/typhon/ManiSkill-vitac/checkpoints/8000_pick_tube`.
- Reuse the existing tactile encoder at
  `/home/typhon/FRS_Tact/checkpoints/encoder/encoder_ckpt_0809`.

## Configuration

Update `deploy_pi05/configs/deploy_pi05_frs.yaml` so `checkpoint`,
`norm_stats.dir`, `frs.checkpoint`, and `frs.tactile_encoder_checkpoint` point
to those matching local assets. Leave camera mapping, task text, dimensions,
solver settings, networking, and runtime behavior unchanged.

## Validation

Verify file sizes and SHA-256 hashes against repository metadata, check the
checkpoint metadata dimensions and training-source settings against the YAML,
load the decoder and tactile encoder offline, and run the deployment launcher's
non-connecting `--check` mode. Do not start the robot client.
