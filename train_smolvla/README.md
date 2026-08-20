# Visual SmolVLA Training

This package runs visual-only SmolVLA JAX fine-tuning from one YAML configuration.

## Start training

Prepare the environment, data, and checkpoint first. The launcher deliberately does
not install or download dependencies:

```bash
bash scripts/setup_env.sh --root
bash scripts/download_data.sh
bash scripts/download_encoder.sh
```

省略 `--root` 时，统一安装脚本会保持兼容行为，同时安装根环境和 Pi0.5 部署环境。

```bash
bash train_smolvla/scripts/train.sh
```

The default launcher starts a tmux session named `smolvla_train`. Attach to its
output with:

```bash
tmux attach -t smolvla_train
```

If that session already exists, the launcher stops instead of replacing it. Attach
to the existing session or choose a different `launcher.tmux_session` value in the
YAML.

Set `launcher.foreground: true` to run in the current terminal. To invoke the
training module directly, use:

```bash
uv run --no-sync python -m train_smolvla.train --config train_smolvla/configs/train.yaml
```

## Configure and resume

Edit `train_smolvla/configs/train.yaml` before starting a run. It holds the
dataset, checkpoint, output, resume, and training parameters, as well as launcher
settings. Pass a different YAML file as the first argument to the shell wrapper
when needed.

Set `resume` to an existing checkpoint directory to continue a run. With no
`resume` value, the launcher stops if the configured output already contains a
`checkpoint-*` directory, preventing an unsafe overwrite.

## Logs

Each foreground training process writes one timestamped log under
`train_smolvla/outputs/logs/`, for example:

`train_smolvla/outputs/logs/train_YYYYMMDD_HHMMSS.log`

The complete `train_smolvla/outputs/` tree is runtime output and is ignored by
Git.
