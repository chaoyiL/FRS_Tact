# H100 VT Training Consistency Design

## Goal

Make the two paper baselines `train_vtsmolvla_jax_tactile16.yaml` and
`train_vtsmolvla_jax_tactile32.yaml` safe to launch on one process spanning two
H100 GPUs, while changing neither tactile-token semantics nor model capacity.

## Fixed scientific contract

- K=8 and K=21 remain the only model difference, producing 32/209=15.31% and
  84/261=32.18% tactile prefix-token ratios.
- Both runs use the same four v3 datasets, episode split, train-only
  normalization statistics, seed, batch size, optimizer, schedule, augmentation,
  cache, and frozen tactile encoder.
- The correct tactile encoder identity is `liuchaoyi/encoder_ckpt_05`, installed
  at `/workspace/checkpoints/encoder_ckpt_05` on the training server.
- Model checkpoints retain FP32 master trainable weights for optimizer resume.
  Every training, validation, policy-inference, and modalities-eval compute path
  casts trainable floating leaves to BF16. Frozen leaves keep their existing
  storage dtype so inference matches the existing training path exactly.
- Old JAX training checkpoints that omit the compute-dtype field are interpreted
  as BF16, because the legacy trainer already computed with trainable BF16
  weights. Invalid or mismatched explicit compute dtype fails closed.
- `tactile_num_tokens=4`, the tactile cache shape `[F,4,512]`, tactile projection
  parameters, repeat ordering, masks, RoPE, and checkpoint parameter keys do not
  change.

## Launcher and encoder

`scripts/start_vtsmolvla_train.sh` accepts `--config PATH` and
`--config=PATH`, rejects unknown/duplicate arguments, resolves the selected file,
and forwards the exact config through the tmux relaunch. With no argument it
continues to select the K=1 YAML. Each K8/K21 YAML documents a launcher command
that names itself and uses a distinct tmux session.

The checkpoint downloader defaults to `liuchaoyi/encoder_ckpt_05` and
`/workspace/checkpoints/encoder_ckpt_05`. The three VT YAMLs use that same path.
The independent tactile-encoder-training and FRS pipelines remain on their
existing encoder identity.

## BF16 compute contract

Add a validated, serialized config field `trainable_compute_dtype` with the
supported value `bfloat16`. A single pure helper receives params and config and
casts only trainable floating leaves. The trainer uses it to form compute params;
`JaxSmolVLAPolicy` and `modalities_eval.SmolVLAEvalModel` apply it once after
loading. Save continues to write FP32 master weights and strict resume continues
to restore those masters.

The effective checkpoint config, publication manifest, validation contract, and
deployment contract carry the dtype field. A missing legacy field canonicalizes
to BF16; explicit unsupported values and cross-dtype mismatches are rejected.

## Train-only normalization protocol

The split is resolved before normalization. For every dataset, read only the
selected training episodes' `stats/*` columns from the LeRobot v3
`meta/episodes` parquet with the existing predicate-pushdown loader. Canonicalize
the action key and rename mapping, validate complete unique episode coverage,
finite 20-dimensional state/action mean/std/count tensors, and aggregate with
LeRobot's existing count-weighted `aggregate_stats`. Validation episodes never
contribute to the result. No image/video/frame decoding is required.

Both K runs name one shared `normalization.protocol_dir` that contains:

- the persisted `data_split.json`;
- normalization and unnormalization safetensors;
- `normalization_manifest.json` recording algorithm version, dataset order and
  revisions, action/rename contract, sorted training episode IDs, split digest,
  per-source selected-stat digests, final canonical-stat digest and dimensions.

Creation uses staging plus atomic rename. An existing identical artifact is
reused read-only; missing, corrupt, or different content fails closed rather than
being overwritten. Checkpoints copy the split and manifest. Resume uses the
checkpoint normalization assets as authoritative and verifies current selected
episode metadata against the recorded manifest before the first step.

## Failure boundaries

- A v2.x dataset, missing per-episode stats, action/schema mismatch, non-finite
  stats, duplicate/missing requested episode, cache fingerprint mismatch,
  encoder mismatch, compute-dtype mismatch, or protocol digest drift aborts
  before training.
- The implementation does not silently fall back to full-dataset stats or scan
  frames to synthesize missing episode stats.
- A real two-H100 smoke is still required after CPU tests: one step for K8 and
  K21 including forward, backward, all-reduce and save; then a two-step strict
  resume smoke. Production 40k runs must not start until those pass.

## Verification

Tests must prove launcher config forwarding, encoder_05 defaults, BF16
train/eval/save-load parity with FP32 masters retained, contract legacy/mismatch
behavior, exclusion of an extreme validation episode from normalization,
count-weighted multi-dataset aggregation, artifact reuse/drift rejection,
checkpoint resume provenance, YAML K8/K21 scientific parity, and existing tactile
cache/parameter schema compatibility.
