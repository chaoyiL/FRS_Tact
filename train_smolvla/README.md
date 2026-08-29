# SmolVLA training

Pure-vision SmolVLA training uses the official PyTorch LeRobot implementation,
following the command construction and dataset-contract handling used by VB3.
The older JAX model modules remain package-internal because SmolVLA-FRS training
and deployment still consume converted JAX checkpoints.

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

Use the Python environment that runs VB3 and contains official LeRobot SmolVLA:

```bash
export SMOLVLA_TORCH_PYTHON=/home/typhon/vb3/.venv/bin/python
bash train_smolvla/scripts/start_smolvla_train.sh
```

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
  --config train_smolvla/configs/train_pytorch_right.yaml --dry-run
```

## JAX FRS path

After PyTorch training, merge/export the selected PyTorch checkpoint with
`tools/merge_smolvla_peft_to_jax.py`. Generate FRS caches and train using
`train_smolvla_frs`. Do not deploy the converted JAX bundle as the normal
pure-vision policy; it is the source policy for the JAX FRS runtime.
