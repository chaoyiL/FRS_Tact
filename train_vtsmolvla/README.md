# VT-SmolVLA JAX Training

This package runs vision-tactile SmolVLA JAX fine-tuning from one YAML configuration.

## Prepare and start

Prepare the environment, data, and checkpoint before launching:

```bash
bash scripts/setup_env.sh
bash scripts/download_data.sh
bash scripts/download_ckpt.sh
bash train_vtsmolvla/scripts/train.sh
```

All datasets, checkpoints, tactile encoder/cache settings, training parameters,
output paths, and launcher behavior live in
`train_vtsmolvla/configs/train.yaml`. The Shell wrapper contains no training
parameters. Direct foreground training without launcher preflight is available as:

```bash
uv run --no-sync python -m train_vtsmolvla.train --config train_vtsmolvla/configs/train.yaml
```

## Cache, tmux, and logs

When `tactile_embedding_cache.enabled` is true, the launcher runs the shared
tactile embedding precompute command before training and stops if it fails. Set
`launcher.foreground: true` to stay in the current terminal. Otherwise the default
session is available with:

```bash
tmux attach -t vtsmolvla_train
```

Change `launcher.tmux_session` to select another session. Each run writes
`precompute_YYYYMMDD_HHMMSS.log` and `train_YYYYMMDD_HHMMSS.log` under the YAML
`launcher.logs_dir` (by default `train_vtsmolvla/outputs/logs`). The output tree is
ignored by Git.

## Resume safely

Set the YAML `resume` field to an existing checkpoint directory to continue a run.
When `resume` is empty, the launcher refuses to use an output directory that already
contains `checkpoint-*`, preventing accidental overwrite.

## Build a release wheel safely

Build an sdist into a clean temporary directory first, then build the wheel from
that archive. This prevents an ignored local `build/lib` tree from contributing
stale modules to a release artifact.

```bash
release_dir="$(mktemp -d)"
uv build --sdist --out-dir "$release_dir/sdist" .
uv build --wheel --out-dir "$release_dir/wheel" "$release_dir"/sdist/*.tar.gz
```
