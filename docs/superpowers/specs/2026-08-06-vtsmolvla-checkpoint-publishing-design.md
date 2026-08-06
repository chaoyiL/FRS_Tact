# VT-SmolVLA Checkpoint Repair and Safe Publishing Design

## Objective

Repair `KaiyueChen/vtsmolvla_01_4w` so its inference metadata matches the trained
VB3 bimanual policy, update the local deployment configuration to consume the
repaired model, and prevent future uploads from mixing trained weights with
`lerobot/smolvla_base` sidecar files.

The trained contract is authoritative:

- state dimension: 20
- action dimension: 20
- action chunk size: 20
- action steps consumed per inference: 5
- RGB inputs: `observation.images.camera1` and
  `observation.images.camera2`
- tactile inputs: `observation.images.tactile_left_0`,
  `observation.images.tactile_right_0`,
  `observation.images.tactile_left_1`, and
  `observation.images.tactile_right_1`
- tactile encoder enabled with four 512-dimensional tactile tokens
- LoRA rank 16 on the configured VLM Q/V projections

## Confirmed Failure

The Hub repository contains the trained VT/JAX `model.safetensors`, but every
small inference sidecar is byte-identical to the corresponding file in
`lerobot/smolvla_base`. The uploaded config therefore incorrectly advertises a
6D state, 6D action, 50-step chunk, three ordinary cameras, no tactile encoder,
and base normalization statistics.

The training path itself uses the correct model overrides and writes effective
metadata. The vulnerable boundary is checkpoint assembly and publication:
`save_portable_params` first writes trained weights and copies base assets, then
higher-level code replaces those assets. Until the higher-level save completes,
the destination is a valid-looking but semantically incorrect directory. A
manual or external upload can publish that intermediate state.

## Scope

### Included

1. Build a corrected local inference bundle for the current 40k checkpoint.
2. Reconstruct its config and processor metadata from the effective training
   configuration.
3. Reconstruct normalization assets using the same dataset-stat aggregation
   path as training, then verify their feature names and dimensions.
4. Validate the existing Hub weight header against the expected VT modules and
   LoRA structure without rewriting the 1.15 GB weight file.
5. Publish only the corrected small sidecars after an explicit final review of
   the local validation report.
6. Update the deployment YAML to use the repaired Hub model and the VB3
   `vitac`, 20D, 20-step contract.
7. Make checkpoint creation atomic from the perspective of publishers.
8. Add a validation and publishing command that refuses incomplete or mixed
   checkpoints.
9. Add automated regression tests.

### Excluded

- Retraining or changing model weights.
- A 6D-to-20D control adapter.
- Changes to the robot's 20D action semantics or safety limits.
- Publishing training state, optimizer state, or dataset caches.
- Starting the physical robot during repair.

## Architecture

### 1. Checkpoint Contract Validator

A reusable validator in the SmolVLA JAX checkpoint layer will inspect a
checkpoint directory and return a structured report. It will not load all model
weights onto an accelerator.

Validation covers:

- required inference files are present;
- config state/action/chunk dimensions match the expected contract;
- RGB and tactile keys are disjoint and ordered as trained;
- `use_tactile_encoder`, token count, embedding dimension, LoRA targets, and
  module modes are internally consistent;
- preprocessor and postprocessor feature specifications agree with config;
- state and action normalization tensors exist under canonical keys and have
  20 elements;
- the Safetensors header contains tactile encoder, tactile projection, and the
  configured LoRA tensors;
- no required sidecar is byte-identical to the corresponding base sidecar when
  the effective contract differs from the base contract.

The validator will raise one aggregated error containing every discovered
contract violation. This provides one actionable report instead of a sequence
of failures during deployment.

### 2. Atomic Checkpoint Assembly

Training will save each checkpoint into a sibling staging directory whose name
ends in `.incomplete`. The sequence is:

1. write model parameters and copied source assets;
2. write effective config;
3. write training and resume metadata;
4. write dataset-derived normalization assets and processor configs;
5. copy the persisted train/validation split;
6. validate the complete checkpoint;
7. atomically rename the staging directory to `checkpoint-NNNNNNNN`.

The final checkpoint name will never expose the transient base-sidecar state.
If saving or validation fails, the `.incomplete` directory remains available
for diagnosis and is never considered publishable. Existing final checkpoint
directories will not be overwritten implicitly.

### 3. Inference Bundle Builder and Publisher

A command-line tool will accept a completed checkpoint and an output directory.
It will:

1. validate the source checkpoint;
2. copy only inference-required files into a new staging bundle;
3. generate a provenance manifest containing the source checkpoint, effective
   contract, file hashes, and weight hash;
4. validate the staged bundle again;
5. atomically finalize the local bundle;
6. optionally upload that exact bundle with `huggingface_hub`.

The inference allowlist is:

- `model.safetensors`
- `config.json`
- `policy_preprocessor.json`
- `policy_postprocessor.json`
- `policy_preprocessor_step_5_normalizer_processor.safetensors`
- `policy_postprocessor_step_0_unnormalizer_processor.safetensors`
- `conversion_manifest.json`
- an optional model card generated from the verified contract

Training state, optimizer state, `data_split.json`, and `trainable_keys.json`
remain local.

Upload is a separate explicit action. The tool will print the local validation
report and require the caller to pass a publish flag. Existing remote weights
will not be re-uploaded when only sidecars need repair; the repair path will
upload the corrected small files in one Hub commit.

### 4. Current Hub Repair

Because the original `/workspace/vtsmolvla_tactile_01/checkpoint-00040000`
directory is not visible on this machine, the repair path will rebuild sidecars
from:

- `configs/train_vtsmolvla_jax.yaml` for the effective model contract;
- the four pinned training dataset revisions for normalization statistics;
- the existing Hub weight header for module/LoRA validation.

The repair tool will use the same canonicalization, renaming, count handling,
and `aggregate_stats` functions used by `LeRobotJaxDataLoader`. It will record
the four dataset revisions and frame counts in the provenance manifest. The
bundle will not be publishable if any dataset revision differs from the
revision resolved for the original training configuration or if the regenerated
statistics do not have canonical 20D state/action tensors.

If exact training-time dataset revisions cannot be proven, publication stops.
The fallback is to retrieve the five correct sidecars from the original 40k
checkpoint rather than guessing statistics.

### 5. Deployment Configuration

After the repaired Hub commit exists, `configs/deploy_smolvla_jax.yaml` will use:

- checkpoint `KaiyueChen/vtsmolvla_01_4w`;
- the repaired immutable Hub revision;
- initial download enabled;
- `data_type: vitac`;
- bimanual state mode;
- `action_horizon: 20` and `steps_per_inference: 5`;
- RGB rename mapping from robot `camera0/1` to model `camera1/2`;
- tactile keys passed through unchanged.

The deployment client will run the checkpoint validator immediately after
resolution and before opening the robot WebSocket. Contract failures therefore
cannot reach the physical robot.

## Data Flow

Training data metadata determines canonical state/action statistics and input
keys. The effective YAML overrides the SmolVLA base config. Training produces
weights and normalization state using that effective contract. Atomic assembly
combines these artifacts, validates them, and exposes a final checkpoint only
after success. The publisher copies the validated inference subset and uploads
one coherent Hub revision. Deployment resolves that revision, validates it
again, then connects to `vb3_robot_server` and negotiates the matching 20D
contract.

## Error Handling and Safety

- Validation errors are fatal and list every mismatched field or file.
- A base-sidecar fingerprint in a non-base checkpoint is fatal.
- Unknown or unpinned dataset revisions prevent reconstructed-stat publication.
- Failed saves remain under `.incomplete` and never replace a valid checkpoint.
- Failed bundle builds remain staged and are not uploaded.
- Hub publication is one explicit external action after local review.
- Deployment validation occurs before WebSocket connection and before robot
  `START`.
- No real-robot command is part of the repair workflow.

## Testing

Unit and integration tests will cover:

1. a valid VT checkpoint contract passes;
2. the current Hub-style mixture of VT weights and base sidecars fails;
3. wrong state/action/chunk dimensions are reported together;
4. missing tactile keys or tensors fail;
5. wrong or absent normalization keys and dimensions fail;
6. processor/config disagreement fails;
7. atomic save exposes no final checkpoint before validation;
8. failed validation preserves only an `.incomplete` directory;
9. the inference bundle allowlist excludes training state;
10. the publisher refuses to run without a passing validation report;
11. deployment refuses an invalid checkpoint before constructing the bridge
    client;
12. the repaired sidecars load as 20D/20D/chunk-20 with four tactile tokens.

Relevant existing JAX and deployment tests will run in addition to the new
targeted tests. A local one-frame inference smoke test is required when a CUDA
device is available. Robot-server dry-run protocol exchange is required before
any physical-robot run.

## Acceptance Criteria

- The corrected local bundle reports the authoritative VB3 contract and passes
  all checkpoint validations.
- Reconstructed normalization tensors use canonical names and 20D shapes, with
  provenance tied to the four training dataset revisions.
- The repaired Hub revision contains coherent weights and sidecars and passes a
  clean download-and-validate test.
- The deployment YAML pins that repaired revision.
- A base-sidecar regression fails automatically in tests and in the publisher.
- A training checkpoint becomes visible under its final name only after all
  sidecars are written and validated.
- Existing test suites remain green.
- No physical robot is started as part of acceptance testing.
