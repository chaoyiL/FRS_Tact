# FRS Training

FRS training code, cache preparation, evaluation, configuration, and launcher live in this package.

## One-command pipeline

```bash
bash train_frs/scripts/start_frs_train.sh
```

Pass another YAML as the first argument when needed. The default is
`train_frs/configs/train_frs.yaml`.

The launcher keeps shared SmolVLA checkpoint merge and tactile embedding precomputation in
`tools/`, then runs the FRS-owned reverse-solver check, action-cache preparation, and trainer
through `python -m train_frs.<module>`.

## Direct training

```bash
uv run --no-sync python -m train_frs.train_frs \
  --config train_frs/configs/train_frs.yaml
```

Training outputs, datasets, action caches, tactile embedding caches, encoder checkpoints, and
merged SmolVLA checkpoints remain external resources configured by the YAML.
