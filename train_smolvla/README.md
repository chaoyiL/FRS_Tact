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

Point the training YAML `dataset.root` at the resulting
`/workspace/lerobot_v30/KaiyueChen/<dataset>` directory. Training launchers do
not implement a second downloader.

Use the Python environment that runs VB3 and contains official LeRobot SmolVLA:

```bash
export SMOLVLA_TORCH_PYTHON=/home/typhon/vb3/.venv/bin/python
bash train_smolvla/scripts/start_smolvla_train.sh
```

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
