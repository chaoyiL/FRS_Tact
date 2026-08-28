# DECO GPU TorchScript Export Design

## Goal

Export DECO checkpoints as GPU-specific TorchScript artifacts whose traced
tensor factories and device conversions consistently target `cuda:0`, then
deploy the available `pick_two_tubes` epoch 18 best checkpoint on the GPU.

## Root Cause

The current exporter forces policy snapshots, deployment wrappers, trace
inputs, and post-save validation onto CPU. Torch tracing therefore specializes
input-derived device expressions into literal CPU operations. Loading such an
artifact with `map_location="cuda:0"` moves stored constants but does not rewrite
the CPU device literals, producing mixed CUDA/CPU execution.

## Export Architecture

The existing `device` argument becomes authoritative for checkpoint exports.
The checkpoint remains deserialized on CPU, after which the rebuilt policy is
moved to the requested target device. The deployment wrapper and every example
input use that same device for tracing. Post-save validation reloads the
artifact onto the same device and performs a complete inference using the trace
inputs.

Live training exports follow the live policy device so future artifacts saved
from GPU training are GPU traced. CPU policies and explicit CPU checkpoint
exports remain supported. Stage 1 and Stage 2 use the same device propagation;
Stage 2 retains its existing fixed-shape scripted guard.

The target device is normalized with `torch.device`. CUDA requests fail early
with a clear error when CUDA is unavailable rather than silently falling back
to CPU.

## Artifact Rollout

The existing epoch 20 `deco_stage1_latest.ts` is preserved. The available
`pick_two_tubes/deco_stage1_best.pt` (epoch 18) is exported to:

```text
checkpoints/model/deco_0828/pick_two_tubes/deco_stage1_best_gpu.ts
checkpoints/model/deco_0828/pick_two_tubes/deco_stage1_best_gpu.ts.json
```

Only after hash/metadata validation and a real CUDA forward succeed will
`deploy_deco/configs/deploy_deco.yaml` be changed to reference the new GPU
artifact.

## Tests and Verification

Tests first establish that checkpoint export honors its requested device and
that saved-artifact validation receives that device. Existing CPU export tests
must continue to pass. On the real RTX 5080, final verification loads the new
artifact with `map_location="cuda:0"`, runs inputs shaped `[1,2,3,224,224]` and
`[1,20]`, and checks CUDA output shape `[1,32,20]` with finite values.

The serialized graph is also audited to ensure the denoising `torch.full`
operations target `cuda:0`, preventing recurrence of the observed failure.

## Non-goals

- Recovering the unavailable epoch 20 checkpoint from the frozen artifact.
- Overwriting or deleting existing TorchScript artifacts.
- Re-exporting bread, insert, or Stage 2 before their source checkpoints finish
  downloading.
- Producing a single device-portable traced artifact. CPU and GPU artifacts are
  specialized for their trace device.
