# Direct Tactile Decoder Deployment Design

## Goal

Deploy the direct tactile action decoder in `checkpoints/ablation` through the
existing SmolVLA remote client. Its execution lifecycle must match FRS: create
one coarse action chunk, obtain a fresh observation before every action,
re-decode the complete chunk from that observation, and execute only the row
identified by the current `action_index`.

Keep the JAX SmolVLA checkpoint loader and FRS model implementation unchanged.
Reuse the existing `frs_steering_v1` transport and robot-side scheduling
protocol for direct-decoder execution without loading an FRS checkpoint or
running FRS reverse integration.

## Selected approach

Continue using the merged visual JAX checkpoint at
`checkpoints/model/pick_tube_01_jax` as the coarse-action producer. Adapt the
direct runtime to the same `begin_chunk` / `steer_action` / `end_chunk`
lifecycle used by FRS and route both backends through the server-directed
per-action protocol.

The following alternatives are rejected:

- Setting legacy `steps_per_inference: 1` would replan from row zero on every
  observation; it would not preserve FRS chunk or `action_index` semantics.
- Re-running SmolVLA for every steer request would move the prediction origin
  while retaining an index from the old chunk, making the row meaning
  inconsistent. SmolVLA therefore runs once at chunk start, exactly as in FRS.
- Replacing the JAX policy with upstream PyTorch SmolVLA adds dependency and
  package-name conflicts without changing the deployment behavior.

## Files and responsibilities

### `deploy_smolvla/direct_decoder.py`

This module owns the direct-decoder model and deployment lifecycle:

- the checkpoint-compatible two-layer `DirectTactileActionDecoder`;
- the converted tactile ResNet18 architecture and safetensors loader;
- current-frame tactile image preprocessing;
- strict validation of decoder, encoder, and fixed-noise assets;
- one immutable normalized coarse-action chunk per active chunk;
- strictly ordered steer requests and idempotent duplicate handling;
- full-chunk direct decoding from each request's current tactile observation;
- selection and one-time unnormalization of only
  `decoded[0, action_index]`.

The runtime accepts the existing `checkpoints/ablation` directory directly. It
does not depend on an FRS checkpoint, tactile history, reverse integration,
temporal ensembling, or an FRS model.

### `deploy_smolvla/remote_client.py`

The root-level `direct_tactile_decoder` backend:

- requires `observation.data_type: vitac` and a 20-step action horizon;
- loads visual-only `JaxSmolVLAPolicy`, not `VTJaxSmolVLAPolicy`;
- requires the four tactile keys in their fixed training order;
- passes the fixed training noise to SmolVLA;
- negotiates `frs_steering_v1` for server-directed per-action execution;
- runs SmolVLA once at chunk start and direct decoding once per steer request;
- reuses the protocol's chunk/request/index validation and acknowledgement
  ordering;
- disables RTC inputs and keeps the fixed inference seed.

The protocol runner becomes backend-neutral where needed. FRS and direct
decoder provide separate lifecycle implementations and trace builders so the
direct backend does not pretend to be an FRS model. Visual SmolVLA's ordinary
chunk path and the FRS model behavior remain unchanged.

### Configuration and launcher

The deployment continues to use:

- `deploy_smolvla/configs/deploy_direct_decoder.yaml`;
- `deploy_smolvla/scripts/start_direct_decoder.sh`.

The launcher delegates to `start_remote_client.sh` and sets
`XLA_PYTHON_CLIENT_PREALLOCATE=false` so JAX and PyTorch can share the GPU. The
existing visual/VT launchers, FRS launcher, and robot server are unchanged.

## Runtime data flow

1. The robot starts a chunk with two RGB images, four tactile RGB images, a 20D
   state, and the task prompt.
2. Visual JAX SmolVLA predicts normalized coarse actions shaped `[1, 20, 20]`
   once for that chunk using fixed noise shaped `[1, 20, 32]`. The runtime
   retains an immutable copy until chunk end.
3. Before action `i`, the robot captures a new observation and sends a steer
   request containing `action_index=i`.
4. The four current tactile frames are processed in this exact order:
   `tactile_left_0`, `tactile_right_0`, `tactile_left_1`,
   `tactile_right_1`.
5. Each tactile image is converted to RGB, aspect-ratio resized, center padded
   with black to `224x224`, converted to float NCHW, and divided by 255. No
   ImageNet normalization or frame differencing is applied.
6. The shared frozen tactile ResNet18 produces four 512D embeddings. The runtime
   RMS-normalizes them once, and the decoder preserves its own second RMS
   normalization from training.
7. The decoder returns normalized fine actions shaped `[1, 20, 20]`. Its output
   is not added to the coarse action.
8. The runtime selects `fine[0, i]`, unnormalizes that row exactly once, and
   sends only the resulting physical action vector.
9. The robot validates, converts, and schedules that one action, then returns a
   matching acknowledgement. The cycle repeats with a fresh observation for
   the next action index.
10. After the final index, the robot ends the chunk and the runtime clears all
    chunk-local state. The next chunk obtains a new coarse SmolVLA prediction.

## Fixed noise

The deployment runtime loads `checkpoints/ablation/fixed_noise.npy`, a float32
array shaped `[1, 20, 32]`. It contains CPU PyTorch seed-zero standard-normal
values in the first 20 channels and zero padding in the final 12 channels.

The runtime never regenerates noise per inference and does not use
`seed + iteration`. The same fixed noise is used for each chunk's single
SmolVLA coarse prediction.

## Configuration

The direct-decoder YAML uses:

- `backend: direct_tactile_decoder`;
- checkpoint `checkpoints/model/pick_tube_01_jax`;
- decoder bundle `checkpoints/ablation`;
- PyTorch device `cuda:0`;
- `observation.data_type: vitac`;
- `action_horizon: 20`;
- `steps_per_inference: 20`, satisfying the full-horizon requirement of
  `frs_steering_v1` rather than the checkpoint's training-only
  `execute_steps` metadata;
- no enabled FRS model section and no RTC action stitching.

Connection, prompt, observation logging, and warmup settings remain aligned
with the existing deployment. Acknowledgements use the strict per-action
chunk/request/index identity checks already used by FRS.

## Error handling

At startup, fail if required assets or checkpoint fields are missing, tensor
shapes do not match the recorded architecture, fixed noise is invalid, the
direct backend conflicts with an FRS model or RTC, or per-action protocol
configuration does not cover the full horizon.

During execution, reject missing tactile images, invalid image shapes,
non-finite output, an output shape other than `[1, 20, 20]`, a request outside
the active chunk, an invalid or non-increasing action index, or a conflicting
duplicate request. Exact duplicate requests return the cached result so a
transport retry cannot advance state twice.

No fallback to unrefined SmolVLA actions is added. A direct-decoder failure
terminates the client instead of silently changing policy behavior.

## Diagnostics

Chunk traces record the fixed coarse normalized and physical action chunks plus
prediction timing. Per-action traces record the newly decoded normalized chunk,
selected index, selected normalized and physical vectors, request identity, and
decode timing. Trace construction remains non-fatal and cannot alter action
scheduling.

## Verification scope

Automated verification covers:

- SmolVLA running once per chunk;
- every steer request reusing the fixed coarse chunk with its current tactile
  observation;
- selection and one-time unnormalization of only the requested row;
- request ordering, exact duplicates, conflicting duplicates, chunk boundaries,
  acknowledgement identity, and malformed actions;
- direct config advertising the steering protocol with full-horizon steps;
- regressions for unchanged FRS and ordinary visual action paths;
- Python compilation/import and YAML/config loading;
- a local released-assets forward check for finite `[1, 20, 20]` output when a
  suitable CUDA device is available.

No robot-server modification or real-robot safety evaluation is included.
