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

## Multi-dataset evaluation

Evaluate the configured validation splits with the same combined action-cache digest,
precomputed tactile embeddings, Euler solver, and validation step count used by training:

```bash
uv run --no-sync python -m train_frs.evaluate \
  --config train_frs/configs/train_frs.yaml
```

The default checkpoint is `<frs_training.output>/best`; results are written to
`<frs_training.output>/evaluation`. `metrics.json` includes aggregate and per-dataset metrics,
while `per_sample.csv` records the source and source-local cache index.

`history.csv` records the complete objective as `train_loss_total` and its six weighted terms:
`train_loss_gt_fm`, `train_loss_vla_fm`, `train_loss_low_safety`, `train_loss_decode`,
`train_loss_rank`, and `train_loss_repair`. `train_flow_loss` remains as a
backward-compatible alias for the total.
