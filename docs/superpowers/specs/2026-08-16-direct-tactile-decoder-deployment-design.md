# Direct Tactile Decoder Deployment Design

## Goal

Deploy the direct tactile action decoder in `checkpoints/ablation` through the
existing SmolVLA remote client with the smallest practical code change. Keep the
existing robot bridge protocol, JAX SmolVLA checkpoint loader, acknowledgement
flow, and FRS implementation unchanged.

## Selected approach

Add one self-contained PyTorch runtime module and one backend branch in the
existing remote client. Continue using the already merged visual JAX checkpoint
at `checkpoints/model/pick_tube_01_jax` as the coarse-action producer.

The following alternatives are intentionally rejected:

- Splitting the decoder runtime into several small modules adds more files
  without changing the deployment boundary.
- Replacing the JAX policy with upstream PyTorch SmolVLA increases dependency
  and package-name conflicts and is unnecessary because the matching merged JAX
  checkpoint already exists.

## Files and responsibilities

### New `deploy_smolvla/direct_decoder.py`

This module contains only the direct-decoder deployment implementation:

- the checkpoint-compatible two-layer `DirectTactileActionDecoder`;
- the converted tactile ResNet18 architecture and safetensors loader;
- current-frame tactile image preprocessing;
- `DirectDecoderRuntime`, which loads the assets and refines normalized coarse
  actions;
- strict checks of the checkpoint fields needed to prevent loading a different
  decoder architecture by accident.

The runtime accepts the existing `checkpoints/ablation` directory directly. It
does not require copying the assets into another bundle layout.

### Modified `deploy_smolvla/remote_client.py`

Add a root-level backend named `direct_tactile_decoder`. For this backend only:

- require `observation.data_type: vitac` and a 20-step action horizon;
- load the visual-only `JaxSmolVLAPolicy`, not `VTJaxSmolVLAPolicy`;
- require the four tactile image keys in their fixed training order;
- pass the fixed training noise to SmolVLA;
- send normalized coarse actions through `DirectDecoderRuntime` before the
  existing action unnormalizer;
- disable RTC inputs and keep a fixed inference seed;
- use the existing ordinary `send_action` / `action_ack` protocol.

The existing visual SmolVLA and FRS paths retain their current behavior.

### New deployment configuration and launcher

Add:

- `deploy_smolvla/configs/deploy_direct_decoder.yaml`;
- `deploy_smolvla/scripts/start_direct_decoder.sh`.

The launcher delegates to `start_remote_client.sh` and sets
`XLA_PYTHON_CLIENT_PREALLOCATE=false` so JAX and PyTorch can share the GPU. The
existing `start_vtsmolvla.sh`, `start_frs.sh`, and robot server are unchanged.

## Runtime data flow

1. The robot supplies two RGB images, four tactile RGB images, a 20D state, and
   the task prompt.
2. Visual JAX SmolVLA predicts normalized coarse actions shaped `[1, 20, 20]`
   using the fixed noise shaped `[1, 20, 32]`.
3. The four current tactile frames are processed in this exact order:
   `tactile_left_0`, `tactile_right_0`, `tactile_left_1`,
   `tactile_right_1`.
4. Each tactile image is converted to RGB, aspect-ratio resized, center padded
   with black to `224x224`, converted to float NCHW, and divided by 255. No
   ImageNet normalization or frame differencing is applied.
5. The shared frozen tactile ResNet18 produces four 512D embeddings. The runtime
   RMS-normalizes them once, and the decoder preserves its own second RMS
   normalization from training.
6. The decoder directly returns normalized fine actions shaped `[1, 20, 20]`.
   Its output is not added to the coarse action.
7. The existing adapter action postprocessor unnormalizes the fine action
   exactly once, and the client sends the resulting physical action chunk.

## Fixed noise

The published Hugging Face payload does not contain `fixed_noise.npy`. Generate
it once with a CPU `torch.Generator` seeded with zero, sample `[1, 20, 20]`
float32 standard-normal values, append 12 zero channels, and save the resulting
`[1, 20, 32]` array as `checkpoints/ablation/fixed_noise.npy`.

The deployment runtime always loads that saved array. It does not regenerate
noise per inference, and direct-decoder inference does not use
`seed + iteration`.

## Configuration

The direct-decoder YAML uses:

- `backend: direct_tactile_decoder`;
- checkpoint `checkpoints/model/pick_tube_01_jax`;
- decoder bundle `checkpoints/ablation`;
- PyTorch device `cuda:0`;
- `observation.data_type: vitac`;
- `action_horizon: 20`;
- `steps_per_inference: 10`, matching the decoder checkpoint's
  `execute_steps` value;
- no enabled FRS section and no RTC action stitching.

Connection, prompt, observation logging, warmup, and acknowledgement settings
are copied from the current SmolVLA deployment configuration.

## Error handling

At startup, fail with a direct error if required assets or checkpoint fields are
missing, tensor shapes do not match the recorded architecture, fixed noise is
not finite or has the wrong zero padding, or the selected backend conflicts with
FRS/RTC. During inference, reject missing tactile images, invalid image shapes,
non-finite decoder output, or an action shape other than `[1, 20, 20]`.

No fallback to unrefined SmolVLA actions is added: a decoder failure terminates
the client instead of silently changing policy behavior.

## Verification scope

Verification is intentionally limited to what is needed for this integration:

- Python compilation/import checks for modified modules;
- YAML/config loading through the existing `--check` launcher path;
- one local runtime forward check using the released encoder and decoder assets,
  verifying output shape and finite values.

No broad test suite, unrelated code review, robot-server modification, or
real-robot safety evaluation is included.
