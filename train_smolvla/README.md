# SmolVLA training

Pure-vision SmolVLA training uses the official PyTorch LeRobot implementation,
wrapped by FRS_Tact for multi-dataset training and dataset-contract handling.
The older JAX model modules remain package-internal because SmolVLA-FRS training
and deployment still consume converted JAX checkpoints.

## Package layout

- `torch_train.py`, `configs/train_pytorch*.yaml`, and `scripts/` are the active
  pure-vision PyTorch training path.
- `architecture.py`, `configuration.py`, `functional.py`, `modeling.py`,
  `preprocessing.py`, `rtc.py`, and `policy.py` are the JAX inference runtime
  retained for FRS, deployment, and likelihood evaluation.
- `checkpoint.py` and `validation.py` define the shared converted-checkpoint
  loading, publishing, and contract validation path.
- `data.py` contains the LeRobot/JAX input helpers still used while preparing
  FRS caches and running modality evaluation.

There is intentionally no direct JAX training entrypoint in this package. The
supported source-policy training path is PyTorch; JAX is retained only for the
downstream compatibility runtime described above.

## PyTorch vision training

Download/upgrade every training dataset through the shared data pipeline first:

```bash
bash scripts/download_data.sh --dataset pick_tube_05
```

List the resulting `/workspace/lerobot_v30/KaiyueChen/<dataset>` directories in
the training YAML's top-level `datasets` block. Training launchers do not
implement a second downloader.

For multiple datasets, repeat `--dataset` during preparation:

```bash
bash scripts/download_data.sh \
  --dataset pick_tube_05 \
  --dataset pick_tube_06
```

Then list the matching local sources:

```yaml
datasets:
  - repo_id: KaiyueChen/pick_tube_05
    root: /workspace/lerobot_v30/KaiyueChen/pick_tube_05
  - repo_id: KaiyueChen/pick_tube_06
    root: /workspace/lerobot_v30/KaiyueChen/pick_tube_06
```

Every source is split into train/eval episodes independently. Training then
concatenates the splits, aggregates normalization statistics, and globally
shuffles their frames. Sources must have identical FPS and feature schemas;
their natural sampling share is proportional to frame count.

Install FRS_Tact's isolated official LeRobot SmolVLA environment. Under RunPod
it is stored at `/workspace/venvs/smolvla_torch`; `env_path` records the Python
path automatically:

```bash
bash scripts/setup_env.sh --smolvla
bash train_smolvla/scripts/start_smolvla_train.sh
```

The setup pins TorchCodec 0.5 for the PyTorch 2.7 runtime. If an existing
RunPod environment was created before that pin, repair it once without
rebuilding the rest of the environment:

```bash
uv pip install --python /workspace/venvs/smolvla_torch/bin/python \
  'torchcodec==0.5.0'
```

W&B uses `mode: auto`: it runs online when `WANDB_API_KEY` or the credential in
`~/.netrc` exists, and otherwise records an offline run instead of aborting a
distributed job. To enable online logging, run:

```bash
/workspace/venvs/smolvla_torch/bin/wandb login
```

After the base checkpoint is loaded, the wrapper validates the final policy
contract before reading the first batch. The dual-arm launcher must report
`state=20D action=20D` and exactly `camera1`, `camera2`; the 6D/three-camera
values printed earlier are only defaults from the base checkpoint configuration.

Multi-GPU startup uses `distributed.timeout_seconds` (7200 seconds by default).
Official LeRobot initializes all local datasets on rank 0 before releasing the
other ranks, so a five-dataset job on network-backed RunPod storage can exceed
Accelerate's 600-second NCCL default even when both GPUs and NCCL are healthy.
With `training.existing_output: increment`, a failed or completed run directory
is preserved and the next fresh run receives a timestamped sibling directory.
Resume jobs still use `training.resume_from` and are never redirected.

The launchers place regenerable Hugging Face Arrow and temporary files under
`/tmp/frs_tact_smolvla` by default. This avoids RunPod network-volume user
quota failures while keeping source datasets, model caches, outputs, and
checkpoints in `/workspace`. Set `SMOLVLA_USE_LOCAL_ARROW_CACHE=0` only when the
workspace quota is known to have enough headroom, or override the local path
with `SMOLVLA_LOCAL_CACHE_ROOT`.

The default dual-arm YAML enables DECO's exact `balanced-light-v2` training
augmentation. It calls the same DECO implementation at batch level, so both
cameras in one sample share the branch and all sampled parameters. Official
LeRobot per-camera `dataset.image_transforms` must remain disabled while this
preset is enabled. Evaluation batches are not augmented. Set `preset` to
`low-light-v1` only when reproducing the older calibrated dark-light runs;
`train_deco/configs/low_light_reference.yaml` is provenance for that v1 preset,
not a separate v2 configuration.

Right-hand single-arm training:

```bash
bash train_smolvla/scripts/start_smolvla_right_train.sh
```

The right-hand dataset must expose a 7D `observation.state`, a 10D `action`,
and the camera contract declared by `rename_map`.

To inspect the generated official LeRobot command without training:

```bash
python -m train_smolvla.torch_train \
  --config train_smolvla/configs/train_smolvla_right.yaml --dry-run
```

## RTX 4090 end-to-end smoke test

The dedicated smoke path keeps the production dual-arm contract (20D state,
20D action, camera1/camera2 after rename, PEFT, and balanced-light-v2) while
using only `KaiyueChen/two_tubes_04`, one visible GPU, batch size 1, and five
training steps. Evaluation and W&B are disabled; step 5 writes a real
checkpoint and the launcher verifies that its model weights exist.

Run the complete environment -> data -> training -> checkpoint pipeline:

```bash
bash train_smolvla/scripts/run_smolvla_4090_smoke.sh
```

The setup and data stages are idempotent. To rerun only training after both
have already passed:

```bash
SMOLVLA_SMOKE_SKIP_SETUP=1 \
SMOLVLA_SMOKE_SKIP_DOWNLOAD=1 \
bash train_smolvla/scripts/run_smolvla_4090_smoke.sh
```

The smoke configuration is `configs/train_smolvla_4090_smoke.yaml`; it must not
replace the two-GPU production configuration in `configs/train_smolvla.yaml`.

## RunPod training issues already addressed

- An obsolete `/home/typhon/vb3` Python path was replaced by the managed
  `/workspace/venvs/smolvla_torch` environment and `env_path` lookup.
- Legacy dataset key `actions` is migrated to the LeRobot v3 `action` key.
- Extra tactile cameras are allowed in source datasets but pruned before the
  pure-vision reader decodes samples; only camera0/camera1 feed this policy.
- TorchCodec is pinned to 0.5.0 for PyTorch 2.7.1, removing the ABI mismatch.
- W&B automatically falls back to offline mode when no API key exists.
- Five-dataset initialization no longer hits Accelerate's 600-second barrier;
  the configured process-group timeout is 7200 seconds.
- Regenerable Arrow/temp files use `/tmp/frs_tact_smolvla`, preventing the
  RunPod workspace user quota from aborting Parquet conversion.
- The 6D/three-camera values printed before dataset creation are base-checkpoint
  defaults. A post-construction guard requires the final dual-arm policy to be
  20D state, 20D action, and exactly two renamed cameras before step 1.

## JAX FRS path

After PyTorch training, merge/export the selected PyTorch checkpoint with
`tools/merge_smolvla_peft_to_jax.py`. Generate FRS caches and train using
`train_smolvla_frs`. Do not deploy the converted JAX bundle as the normal
pure-vision policy; it is the source policy for the JAX FRS runtime.
