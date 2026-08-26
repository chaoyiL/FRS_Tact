# DECO Stage 2 Tactile-Image Training Design

Date: 2026-08-26

## Summary

Extend the existing `train_deco` Stage 1 visual policy into a Stage 2 image-tactile
policy. Stage 2 starts from
`checkpoints/deco/image_aug/deco_stage1_latest.pt`, keeps every Stage 1 parameter
frozen, encodes four tactile RGB streams with a frozen pretrained tactile
ResNet18, and trains only new tactile cross-attention and PI Adapter parameters.

The user supplies only the Stage 1 checkpoint path and the existing JAX tactile
encoder directory. Training automatically converts the JAX/Flax tactile
ResNet18 to a verified PyTorch `safetensors` artifact, caches the conversion by
source content hash, and loads it on every DDP rank.

## Goals

- Preserve the trained Stage 1 visual policy exactly at Stage 2 initialization.
- Consume the four tactile RGB fields already present in the Stage 1 datasets.
- Reuse and freeze the tactile encoder at
  `checkpoints/encoder/encoder_ckpt_0824`.
- Follow upstream DECO's decoupled tactile cross-attention and PI Adapter design.
- Make JAX-to-PyTorch conversion automatic, deterministic, cached, and verified.
- Retain the current masked flow-matching objective and episode-level split.
- Produce resumable Stage 2 checkpoints and a self-contained TorchScript model.

## Non-goals

- Training a new tactile encoder or unfreezing the existing encoder.
- Supporting upstream DECO's two 1062-dimensional raw pressure arrays.
- Adding tactile history or temporal tactile encoders.
- Treating tactile streams as four additional visual cameras.
- Extending the legacy preprocessed Stage 1 dataset format in the first version.
- Changing the robot-side Stage 2 deployment runtime in this implementation.
  The exporter and artifact contract are included; robot integration is a
  follow-up consumer change.

## Existing Contracts

The selected Stage 1 checkpoint has these fixed properties:

- action dimension: 20
- observation dimension: 20
- action chunk: 32
- hidden dimension: 512
- attention blocks: 6
- heads: 8
- visual streams, in order:
  `observation.images.camera0`, `observation.images.camera1`
- model type: `upstream-deco-stage1`
- objective: `masked-flow-mse-v1`

The tactile streams are RGB images shaped `[224, 224, 3]` at 30 Hz. Their fixed
order is:

1. `observation.images.tactile_left_0`
2. `observation.images.tactile_right_0`
3. `observation.images.tactile_left_1`
4. `observation.images.tactile_right_1`

Suffix `0` is the left wrist pair and suffix `1` is the right wrist pair. The
`left`/`right` component before the suffix identifies the sensor within a wrist.

The tactile encoder checkpoint is JAX/Flax. It contains a shared ResNet18 whose
per-image output dimension is 512. Its contrastive `future_projection`, optimizer
state, and memory bank are not part of the Stage 2 policy.

## Considered Approaches

### Selected: four global 512-dimensional tactile tokens

Apply the same frozen ResNet18 to each tactile image and retain one global token
per sensor. This reuses the pretrained representation, matches DECO's hidden
dimension directly, and makes tactile attention substantially cheaper than the
upstream 68-token raw-pressure representation.

### Deferred: sixteen spatial tactile tokens

Retain a 2-by-2 feature grid per tactile image, producing sixteen tokens. This
may preserve contact location better but changes the converted encoder output
contract and increases complexity. It is reserved for a later ablation if four
global tokens underperform.

### Rejected: six shared visual-camera streams

Passing tactile images through the Stage 1 ResNet34 as extra cameras conflates
modalities, discards the pretrained tactile encoder, expands visual positional
embeddings, and does not implement the DECO plugin design.

## Data Contract

The LeRobot v2.1 dataset backend loads the existing visual, state, and action
columns plus the four fixed tactile fields. One sample returns:

```text
images          float32 [2, 3, H, W] in [0, 1]
tactile_images  float32 [4, 3, H, W] in [0, 1]
observation     float32 [20]
action          float32 [32, 20]
is_pad          bool    [32]
task_index      int64   []
```

Every dataset root must expose all four tactile fields with the same RGB HWC
shape and frame rate as the policy anchor rows. Missing, duplicated, reordered,
non-RGB, or malformed tactile fields fail validation before training.

Visual images retain the current shared low-light augmentation and ImageNet
normalization. Tactile images receive no brightness, gamma, contrast,
saturation, blur, or visual low-light augmentation. They are only converted to
unit RGB and letterboxed to 224 by 224 with black padding, matching tactile
encoder pretraining.

The original episode split seed and root manifest semantics remain unchanged so
Stage 2 does not leak validation episodes. The formal Stage 1 manifest recorded
in the checkpoint is not present in the current checkout; formal training
requires restoring it or regenerating an equivalent multi-root manifest for
`pick_tube_01` through `pick_tube_06`.

The initial implementation supports the `lerobot-v21` backend. Selecting the
legacy `preprocessed` backend for Stage 2 fails with an actionable error rather
than silently omitting tactile inputs.

## Tactile Encoder Conversion and Cache

The training configuration accepts either:

- a JAX tactile checkpoint directory containing `checkpoint.json` and its
  referenced `params-*.npz`; or
- an already converted `encoder.safetensors` with matching conversion metadata.

The normal user configuration points directly at the existing JAX directory:

```yaml
tactile:
  encoder_checkpoint: /home/typhon/FRS_Tact/checkpoints/encoder/encoder_ckpt_0824
  freeze_encoder: true
```

For a JAX directory, conversion runs automatically before model construction:

1. Read `checkpoint.json` and resolve the referenced parameter archive.
2. Hash the conversion-relevant metadata and parameter archive contents.
3. Resolve the cache directory
   `train_deco/.cache/tactile_encoders/<source_sha256>/` unless explicitly
   overridden.
4. Reuse the cached artifact only when its metadata, source hash, converted hash,
   architecture contract, and validation thresholds all match.
5. Otherwise, global rank zero acquires a cache lock and performs conversion;
   other ranks wait at a distributed barrier.
6. Map only `tactile_resnet` parameters and BatchNorm statistics. Convolution
   kernels transpose from Flax HWIO to PyTorch OIHW. Dense kernels transpose from
   IO to OI. BatchNorm scale/bias/mean/variance map to
   weight/bias/running_mean/running_var.
7. Strict-load the converted state dict into the PyTorch `TactileResNet18` and
   reject missing, unexpected, duplicate, non-finite, or shape-mismatched leaves.
8. Force JAX to CPU and compare JAX and PyTorch eval outputs on deterministic
   fixed inputs shaped `[4, 224, 224, 3]`.
9. Require `allclose(rtol=2e-3, atol=2e-4)` and record maximum absolute and
   relative errors.
10. Write `encoder.safetensors` and `encoder.json` through temporary files and
    atomic rename. Incomplete artifacts are never visible to other ranks.
11. Broadcast success or failure. Every rank independently verifies final hashes
    before loading the encoder.

JAX/Flax are conversion-only dependencies installed by the `train_deco` setup
script. They are imported lazily, forced to CPU before import, and never used in
the training forward pass. `safetensors` is a runtime dependency for loading the
converted artifact.

The conversion metadata records the source files and hashes, output hash,
architecture, mapped key count, input contract, output dimension, test seed, and
numeric errors. The stale
`checkpoints/ablation/tactile_encoder/encoder.safetensors` is not reused because
its recorded source hash does not match `encoder_ckpt_0824`.

## Model Architecture

The Stage 2 forward input is:

```text
visual images   [B, 2, 3, H, W]
tactile images  [B, 4, 3, H, W]
observation     [B, 20]
action          [B, 32, 20] during training
```

Tactile encoding is:

```text
[B, 4, 3, 224, 224]
  -> reshape [4B, 3, 224, 224]
  -> shared frozen TactileResNet18
  -> [4B, 512]
  -> reshape [B, 4, 512]
  -> per-token RMS normalization
  -> add learned sensor identity [4, 512]
  -> tactile tokens [B, 4, 512]
```

Each DECO multimodal attention block adds:

- tactile key and value projections from 512 to 512;
- six rank-32 PI Adapter branches parallel to visual and action QKV, output
  projection, and MLP projections; and
- one scalar tactile cross-attention gate initialized to zero.

Visual and action queries jointly attend to the four tactile tokens. Fusion is:

```python
attn = base_self_attention + tanh(tactile_gate) * tactile_cross_attention
```

The zero gate makes the untrained Stage 2 policy exactly reduce to Stage 1. PI
Adapter up projections are also zero-initialized, preserving the Stage 1 output
at initialization. The tactile encoder remains in eval mode even while the
overall policy is training, preventing BatchNorm statistic drift.

## Stage 1 Initialization and Freezing

Stage 1 initialization is distinct from resume. A new CLI/config field supplies
the source checkpoint:

```text
--stage1-checkpoint checkpoints/deco/image_aug/deco_stage1_latest.pt
```

The loader:

1. Reads the Stage 1 checkpoint wrapper and validates its saved model, state,
   action, camera, normalization, objective, and architecture contracts.
2. Builds Stage 2 without separately loading ImageNet ResNet34 weights.
3. Loads every Stage 1 model tensor by exact name and shape.
4. Allows missing keys only from an explicit Stage 2 tactile/adapter allowlist.
5. Rejects all unexpected Stage 1 keys and every non-allowlisted missing key.
6. Freezes exactly the successfully loaded Stage 1 parameters.
7. Verifies the complete expected frozen and trainable parameter-name sets.
8. Starts a new optimizer, scheduler, scaler, epoch, global step, and RNG stream.

The loader freezes keys that were actually loaded, rather than every source key
whose name happens to exist. This avoids the upstream implementation's unsafe
behavior when a same-named tensor has an incompatible shape.

All Stage 1 parameters and the tactile ResNet18 are frozen. Trainable parameters
are limited to tactile sensor identity embeddings, tactile key/value
projections, tactile gates, and PI Adapters.

`--stage1-checkpoint` is required for a fresh Stage 2 run and is mutually
exclusive with Stage 2 `--resume-from`. Exact Stage 2 resume loads the full Stage
2 model and its optimizer/scheduler/scaler state; it does not repeat Stage 1
initialization.

## Training Configuration and Objective

A dedicated Stage 2 mode and configuration make the mode explicit. The core
configuration is:

```yaml
stage: 2
stage1_checkpoint: checkpoints/deco/image_aug/deco_stage1_latest.pt

tactile:
  enabled: true
  keys:
    - observation.images.tactile_left_0
    - observation.images.tactile_right_0
    - observation.images.tactile_left_1
    - observation.images.tactile_right_1
  encoder_checkpoint: checkpoints/encoder/encoder_ckpt_0824
  freeze_encoder: true
  token_count: 4
  token_dim: 512
  sensor_identity: true
  cross_attention_gate_init: 0.0

adapter:
  enabled: true
  rank: 32
```

The shell interface exposes a `server-stage2` mode and environment-variable
overrides. After environment setup, formal use only requires the Stage 1 path,
tactile encoder path, dataset manifest, output directory, and run ID.

Training retains the current masked flow-matching MSE:

```text
target = noise - normalized_action
loss = masked_mse(prediction, target, is_pad)
```

The initial defaults are a new 1e-4 adapter learning rate, warmup plus cosine
decay, 50 epochs, and existing episode-level validation and early stopping.
These values are saved in the checkpoint and remain CLI-overridable. No frozen
parameter is placed in an optimizer group.

## Validation and Safety Checks

Before the first optimizer step, a deterministic parity check uses identical
visual input, observation, action/noise, and RNG state to compare:

- the loaded Stage 1 policy; and
- Stage 2 with all zero-initialized tactile gates and PI Adapters.

Outputs must match within the configured floating-point tolerance. Failure
indicates an initialization or contract error and aborts training.

Validation reports:

- normal Stage 2 loss and MAE;
- tactile-disabled loss using a zero gate override; and
- shuffled-tactile loss using a deterministic within-batch permutation.

The latter two measurements show whether Stage 2 has learned a useful dependence
on correctly paired tactile inputs.

Runtime invariants include:

- every Stage 1 parameter has no gradient;
- every tactile encoder parameter has no gradient and its BatchNorm statistics
  remain unchanged;
- at least one tactile cross-attention parameter and one PI Adapter parameter
  receive a finite nonzero gradient;
- frozen tensors remain bit-identical across an optimizer step and checkpoint
  round trip; and
- all input tensors, predictions, losses, and trainable gradients are finite.

## Checkpoint and Export Contracts

Stage 2 outputs are named:

```text
deco_stage2_latest.pt
deco_stage2_best.pt
deco_stage2_epoch_<N>.pt
```

Each checkpoint stores the complete Stage 2 model plus normal training state. It
also records:

- source Stage 1 path and SHA256;
- tactile encoder input path and source SHA256;
- converted encoder cache path and SHA256;
- conversion numeric errors and metadata;
- frozen and trainable parameter names;
- tactile key order and token contract; and
- Stage 2 model, objective, dataset, and training-state schema versions.

The Stage 2 TorchScript wrapper accepts raw unit-space inputs:

```text
images          [B, 2, 3, H, W]
tactile_images  [B, 4, 3, H, W]
observation     [B, 20]
output          [B, 32, 20]
```

It embeds visual and tactile preprocessing, normalization statistics, frozen
tactile encoder, adapters, and action denormalization. Its artifact format is
`sudo-upstream-deco-stage2-torchscript-v1`. Metadata includes all six ordered
input keys and the two source checkpoint hashes.

Stage 2 training never emits a Stage 1 artifact name. If Stage 2 export is not
available for a requested save point, checkpoint saving succeeds but reports an
explicit export failure; it never copies or labels a Stage 1 TorchScript file as
Stage 2.

## Component Boundaries

- `train_deco/tactile_encoder_conversion.py`: source discovery, hashing, Flax
  mapping, numeric verification, atomic cache, and DDP-safe resolution.
- `train_deco/models/tactile_resnet.py`: standalone PyTorch tactile ResNet18.
- `train_deco/lerobot_vision_dataset.py`: strict four-stream tactile loading.
- `train_deco/input_adapter.py`: tactile unit-space letterbox preprocessing.
- `train_deco/models/deco/deco.py`: tactile tokens, gates, cross-attention, and
  PI Adapter activation.
- `train_deco/model_factory.py`: explicit Stage 1/Stage 2 model construction.
- `train_deco/stage2_checkpoint.py`: Stage 1 contract validation, superset load,
  freeze audit, and Stage 2 metadata.
- `train_deco/train.py`: Stage 2 lifecycle, optimizer, parity checks, metrics,
  resume, save, and export orchestration.
- `train_deco/export_torchscript.py`: six-stream Stage 2 artifact wrapper.
- `train_deco/scripts/train.sh`: `server-stage2` and local smoke modes.

The conversion module is independently testable and does not import the training
entrypoint. The dataset does not know about model internals. The model receives
already validated tensors and does not resolve filesystem paths.

## Test Plan

### Conversion

- Resolve the JAX parameter archive from metadata.
- Convert every expected ResNet18 parameter and BatchNorm statistic.
- Reject missing, extra, duplicate, non-finite, or shape-mismatched leaves.
- Verify Flax/PyTorch outputs meet the numeric tolerance.
- Verify source-hash cache hits and invalidation.
- Verify atomic failure leaves no valid-looking artifact.
- Verify one conversion under concurrent processes and DDP ranks.
- Reject the stale ablation artifact for the current source checkpoint.

### Dataset and preprocessing

- Enforce the exact tactile key order and image contract.
- Decode all four embedded images into `[4, 3, H, W]` unit tensors.
- Reject missing and malformed tactile fields before training.
- Verify tactile preprocessing uses black padding and no visual augmentation.
- Verify episode splits and state/action statistics remain unchanged.

### Model and freezing

- Encode four images into `[B, 4, 512]` tokens.
- Verify the shared frozen encoder is called over the flattened sensor batch.
- Verify sensor identity and tactile gate parameter shapes.
- Verify a zero-gate Stage 2 forward equals Stage 1 with identical noise.
- Verify only the explicit Stage 2 allowlist is missing during Stage 1 load.
- Verify optimizer groups contain every and only trainable parameter.
- Verify one optimizer step changes new parameters and no frozen tensor.

### Training, resume, and artifacts

- Run a synthetic Stage 2 training/validation step on CPU and GPU where
  available.
- Verify normal, disabled, and shuffled tactile validation paths.
- Verify exact Stage 2 resume restores all state.
- Reject Stage 1 checkpoints as Stage 2 resume checkpoints.
- Verify full checkpoint round trip and provenance metadata.
- Verify Stage 2 TorchScript eager/traced parity and metadata hashes.

## Success Criteria

- A user can launch Stage 2 by supplying the Stage 1 checkpoint, JAX tactile
  encoder directory, dataset manifest, and normal run settings.
- First launch converts and verifies the tactile encoder automatically; later
  launches reuse a content-addressed cache.
- Stage 2 before training reproduces Stage 1 under zero tactile gates.
- Only tactile fusion and PI Adapter parameters train.
- Four tactile images are consumed in the fixed semantic order.
- Stage 2 checkpoints resume exactly and preserve complete provenance.
- The exported artifact contains the full frozen visual and tactile encoders plus
  trainable Stage 2 modules.
