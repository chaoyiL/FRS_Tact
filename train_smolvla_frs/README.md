# FRS Training

FRS training code, cache preparation, evaluation, configuration, and launcher live in this package.

## One-command pipeline

```bash
bash train_smolvla_frs/scripts/start_frs_train.sh
```

Pass another YAML as the first argument when needed. The default is
`train_smolvla_frs/configs/train_frs.yaml`.

The launcher keeps shared SmolVLA checkpoint merge and tactile embedding precomputation in
`tools/`, then runs the FRS-owned reverse-solver check, action-cache preparation, and trainer
through `python -m train_smolvla_frs.<module>`.

## Direct training

```bash
uv run --no-sync python -m train_smolvla_frs.train_frs \
  --config train_smolvla_frs/configs/train_frs.yaml
```

Training outputs, datasets, action caches, tactile embedding caches, encoder checkpoints, and
merged SmolVLA checkpoints remain external resources configured by the YAML.

Set `model.freeze_tactile_encoder: false` to fine-tune every FRS conditioner end to end.
In this mode, training loads raw tactile windows, initializes the ResNet from
`model.tactile_encoder_path`, and stores the fine-tuned ResNet inside each FRS checkpoint.
The shared tactile GRU and optional state normalization/MLP are always trainable. Precomputed
tactile embeddings remain an immutable source for Gate labels, preventing the labels from
drifting as the ResNet changes. Use a substantially smaller batch size than cached-embedding
training; `frs_training.tactile_encode_microbatch_size` controls ResNet activation memory.
Raw-image training should set `frs_training.num_workers` greater than 1 so spawn workers
decode tactile video while the GPU trains the previous batch; `prefetch_batches` and
`pipeline_prefetch` keep those decoded windows queued ahead of the train step.

The current state-conditioned pipeline stores the frozen VLA preprocessor's normalized
current `observation.state` in action-cache v3. FRS maps it to one cross-attention token and
concatenates that token with the four tactile-history tokens. `model.state_dropout_rate`
masks the entire state token during training only; evaluation and deployment always use it.

The training-time `w` (gate) value is a supervision and reporting label only. Decoder input
v2 accepts the action base, tactile tokens, and optional state token; it never accepts raw
`w` or a gate value. Deployment therefore requires checkpoints declaring
`decoder_input_version: 2` and supplies no gate input. Legacy checkpoints with
`decoder_config.gate_conditioning: true` are incompatible with decoder input v2;
they are not migrated automatically and must be retrained.

For gated training, confident high-gate samples receive direct GT decode, GT-over-VLA rank,
and absolute repair constraints. Confident low-gate samples use only the weaker
nearest-endpoint safety hinge, so either a GT-like or VLA-like action remains acceptable.

## Multi-dataset evaluation

Evaluate the configured validation splits with the same combined action-cache digest,
precomputed tactile embeddings, the configured decode solver (`frs_training.aux_decode_solver`,
default Euler), and validation step count used by training:

```bash
uv run --no-sync python -m train_smolvla_frs.evaluate \
  --config train_smolvla_frs/configs/train_frs.yaml
```

The default checkpoint is `<frs_training.output>/best`; results are written to
`<frs_training.output>/evaluation`. `metrics.json` includes aggregate and per-dataset metrics,
while `per_sample.csv` records the source and source-local cache index.

`history.csv` records the complete objective as `train_loss_total` and its six weighted terms:
`train_loss_gt_fm`, `train_loss_vla_fm`, `train_loss_low_safety`, `train_loss_decode`,
`train_loss_rank`, and `train_loss_repair`. `train_flow_loss` remains as a
backward-compatible alias for the total.
