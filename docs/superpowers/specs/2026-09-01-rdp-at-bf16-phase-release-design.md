# RDP AT BF16 and Deployment-Phase Repair Design

## Goal

Eliminate the BF16 small-angle rotation-loss blind spot, evaluate AT/LDP on the phases actually consumed by `slow_update_interval=16`, and prevent unqualified recovery checkpoints from being used for robot deployment.

## Scope

This repair changes the physical rotation loss, pick-tube validation metrics, checkpoint release flow, the single-right training launcher, and RDP deployment qualification. It does not redesign the GRU, preserve hidden state across control steps, skip decoder phases, or change the runtime control schedule.

## Training loss

The rotation branch of `compute_physical_action_loss` will run in an autocast-disabled precision island. Rotation 6D inputs are promoted to the common input dtype, with FP16/BF16 promoted to FP32 while FP64 remains FP64. The protected region covers Gram-Schmidt projection, relative rotation multiplication, trace/acos geodesic error, idle rotation loss, degeneracy loss, and the raw rotation-6D auxiliary loss. Position and gripper computation remain unchanged.

The duplicate training and deployment copies of `physical_action_loss.py` will remain synchronized. Regression tests will exercise 0.05, 0.5, and 1.2 degree errors under CPU BF16 autocast and require a nonzero FP32-equivalent loss plus finite, nonzero gradients. CUDA BF16 coverage will be conditional on device availability.

## Deployment-window validation

Existing 29-phase metrics remain available for historical diagnosis. A new deployment-window helper will evaluate exactly the decoder phases selected by the configured schedule:

```text
phase_start = n_obs_steps * dataset_obs_temporal_downsample_ratio - 1
phase_count = validation.deployment_slow_update_interval
```

For the single-right configuration this is decoder indices 3 through 18. Metrics use a separate `val_deploy_*` namespace so a 16-phase value cannot be confused with an existing 29-phase value.

AT validation will also reconstruct a deterministic canonical no-op action chunk: zero translation, identity 6D rotation, and the source gripper target. It will encode using posterior mode, decode with the recorded temporal tactile condition, and measure no-op pose drift over the deployment window. LDP validation will use the same deployment window on its held-out predictions, but will not synthesize a no-op intention from active observations.

The configured release thresholds remain focused on the already defined deployment contract:

- idle/no-op translation p95 below 0.05 mm;
- idle/no-op rotation p95 below 0.03 degrees;
- active translation and rotation no more than 5 percent worse than the supplied baseline;
- micro-motion recall at least 95 percent.

Missing active baselines keep release qualification false while still allowing recovery training.

## Checkpoint and deployment contract

`latest.ckpt` remains an unconditional recovery checkpoint and is never treated as a release. When all deployment-window and no-op gates pass and the deployment idle score improves, training writes `checkpoints/deployable.ckpt`. The best score and release evidence are stored in checkpoint state/config so resume preserves the comparison and deployment can verify qualification.

The single-right launcher requires `AT/deployable.ckpt` before starting LDP and reports an actionable error when it is absent. Deployment defaults to deployable checkpoint names. Before constructing either model, deployment verifies that AT and LDP both carry successful release evidence and that their validated slow-update interval equals the runtime interval. Missing evidence, a failed gate, or an interval mismatch is rejected. Existing artifact pairing and state/action dimension validation remain unchanged.

## Tests and acceptance

Implementation follows test-first red/green cycles for:

1. BF16 small-angle loss value and gradient preservation.
2. Deployment-window isolation using sequences whose early and late phases have opposing quality.
3. Runtime slow16 history/phase sequence `3..18, 3`.
4. Release checkpoint selection only when deployment qualification passes and improves.
5. Launcher rejection when deployable AT is absent.
6. Deployment rejection for missing/failed release evidence or a schedule mismatch.

Only the directly affected RDP unit/integration tests and a focused code review are required. No unrelated safety framework or broad repository refactor is included.
