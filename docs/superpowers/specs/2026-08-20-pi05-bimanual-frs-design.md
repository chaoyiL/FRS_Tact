# Pi0.5 Bimanual FRS Migration Design

## Goal

Port the current `train_smolvla_frs` FRS method—`bimanual_gated`, objective version 2, and loss weighting version 7—into `train_pi05_frs` without breaking the existing scalar-gated pipeline, Pi0.5 cache/checkpoint contracts, or deployment behavior.

The migration includes training, checkpoint/resume validation, best-checkpoint selection, evaluation, history, bimanual diagnostics, tests, and the minimum `deploy_pi05` compatibility needed to accept the new checkpoint metadata. It does not add end-to-end tactile ResNet fine-tuning; Pi0.5 continues to train from frozen cached tactile embeddings.

## Compatibility Strategy

Pi0.5 operates in a configurable model action space that is currently 32-dimensional, while the physical bimanual action occupies the first 20 dimensions. The new objective therefore separates model width from steered physical width:

- left action slice: `[0, 10)`
- right action slice: `[10, 20)`
- steered action width: `20`
- padded tail: `[20, action_dim)`
- left tactile token indices: `[0, 1]`
- right tactile token indices: `[2, 3]`

The decoder remains 32-dimensional for the current Pi0.5 checkpoint and cache. The composite target uses independently gated GT/VLA endpoints in the first 20 dimensions and the frozen Pi0.5 endpoint in the padded tail. Bimanual FM and auxiliary losses are normalized over the active first 20 dimensions only; the padded tail is excluded from bimanual losses and metrics so it cannot dilute gradients. Native 20-dimensional Pi0.5 checkpoints remain supported by the same implementation.

Checkpoint metadata records objective version, action slices, steered width, padded-tail policy, and tactile token groups. Resume and deployment reject missing or mismatched bimanual metadata. Legacy `gt`, `predicted`, and scalar `gated` checkpoints keep their current validation behavior.

## Data and Gate Flow

The existing Pi0.5 action-cache and frozen tactile-embedding cache remain unchanged. For each sample, the conditioner compares the current four cached tactile tokens with the episode's first-frame baseline using cosine change. It averages tokens `[0, 1]` and `[2, 3]` separately, then applies the existing sigmoid Gate transform to produce `[B, 2]` left/right Gate weights.

The configured tactile keys must match the current fixed semantic order:

1. `tactile_left_0`
2. `tactile_right_0`
3. `tactile_left_1`
4. `tactile_right_1`

Gate labels remain supervision and reporting values only. They are not added to decoder inputs, so the decoder architecture and deployment inference inputs remain unchanged.

## Training Objective

For each wrist, raw Gate confidence is converted to the three-region effective weight

`w_eff = clip((w - low_threshold) / (high_threshold - low_threshold), 0, 1)`.

The first 20 dimensions of the endpoint are assembled per wrist as

`target = w_eff * GT + (1 - w_eff) * frozen_Pi0.5`.

Training makes one flow-matching call toward this composite endpoint. `gate_lambda` is invalid for `bimanual_gated`. The weighted objective also contains per-wrist decode, low-Gate nearest-endpoint safety, high-Gate GT-over-Pi0.5 ranking, and high-Gate absolute repair terms. Active terms are normalized within their active wrist/source groups so absent Gate regions and dataset size do not silently dilute gradients.

The independent bimanual configuration copies the current SmolVLA objective defaults: Gate `tau=0.4`, temperature `0.1`, thresholds `0.3/0.7`, FireFlow decode with 10 steps, decode weight `4.0`, low-safety weight `0.5`, rank weight `2.0`, and repair weight `0.0`. Existing Pi0.5 model, cache, batch, and schedule settings stay Pi0.5-specific.

## Training, Checkpoint, and Selection Integration

`train_pi05_frs` gains a new `bimanual_gated` mode while retaining all existing modes. The configuration validator accepts the new mode, rejects `gate_lambda` for it, validates tactile-key order and action width, and forwards the objective settings into training.

Training records total loss, composite FM, every weighted auxiliary term, and mean left/right Gate values. Bimanual checkpoint selection uses per-wrist and per-dataset safety, gain, and rank constraints, following the current SmolVLA worst/min aggregation rather than pooling wrists or datasets. Resume validates both existing Pi0.5 cache provenance and the complete bimanual objective contract.

A new `train_pi05_frs/configs/train_pi05_frs_bimanual_gated.yaml` enables the method and writes to a distinct output directory. The existing `train_pi05_frs.yaml` remains unchanged and reproducible.

## Evaluation and Diagnostics

Evaluation detects the checkpoint loss mode and computes the composite FM, left/right Gate distributions, per-wrist endpoint errors, per-dataset rollups, and the four confident Gate quadrants. CSV and JSON outputs retain legacy fields and add bimanual fields only when the objective is active.

With plotting enabled and at least one validation result, training/evaluation produce:

- `training_overview.png`
- `bimanual_behavior.png`
- `gate_diagnostics.png`
- `bimanual_action_examples.png`

Legacy `training_curves.png`, scalar history files, and scalar evaluation remain supported.

## Deployment Compatibility

The deployment decoder architecture is unchanged. `deploy_pi05` is updated only where it validates checkpoint training metadata: it accepts `bimanual_gated`, validates objective-v2 action/token mappings and padded-tail policy, and keeps the existing decoder/policy action-dimension equality check. Runtime steering and final truncation to the 20-dimensional robot action do not change.

## Testing

Implementation follows test-first development. Tests cover:

- per-wrist tactile change and Gate generation;
- fixed tactile-key order and objective metadata validation;
- 20D and 32D composite endpoints and active-dimension normalization;
- one composite FM call and per-wrist auxiliary behavior;
- legacy scalar objectives remaining unchanged;
- trainer/config integration, history fields, best selection, and resume rejection;
- per-wrist, per-source, and quadrant evaluation metrics;
- diagnostic plot generation and legacy-history compatibility;
- Pi0.5 transactional checkpoint round trips;
- deployment acceptance of valid bimanual metadata and rejection of mismatches.

Verification runs the focused red/green tests first, then the complete `train_pi05_frs` suite and affected `deploy_pi05` compatibility tests from their documented isolated environments.

## Out of Scope

- changing Pi0.5 action-cache generation or regenerating existing caches;
- converting the current 32D Pi0.5 checkpoint/deployment chain to 20D;
- training the tactile ResNet end to end from raw images;
- changing robot control, action unnormalization, or the final 20D robot-action truncation;
- altering the existing scalar-gated configuration or old checkpoint semantics.
