# Remove FRS Gate Input Design

## Goal

Remove gate weight `w` from the FRS decoder and deployment runtime. Keep `w`
only as a training supervision signal for the existing gated loss, metrics, and
checkpoint selection.

## Model and Training

- Remove `gate_conditioning`, `gate_mlp`, and `gate_weights` from decoder and
  decode APIs.
- Keep computing `w` from tactile change during training and evaluation.
- Pass `w` only to gated loss and gate-stratified metrics.
- Keep the current gated objective and its training configuration unchanged.
- Save `decoder_input_version: 2`; do not save `gate_conditioning`.

## Deployment and Evaluation

- Remove `gate_tau` and `gate_temperature` from deployment configuration.
- Do not compute or pass `w` in hot-path decoding, warmup, or legacy steering.
- Remove `gate_weight` from deployment diagnostics and logs; keep
  `tactile_change`.
- Remove gate counterfactual inputs from modality evaluation while retaining
  gate-stratified reporting.

## Compatibility

- Require a newly trained checkpoint with `decoder_input_version: 2` for
  deployment.
- Reject old `gate_conditioning=true` checkpoints with a clear retraining error.
- Reject deprecated deployment gate fields instead of silently ignoring them.
- Do not provide automatic checkpoint conversion.

## Tests

- Verify decoder parameter trees and APIs contain no gate input.
- Verify `w` still changes gated losses and metrics, not decoder output.
- Verify new checkpoint round trips and old gated checkpoints fail clearly.
- Verify all deployment decode paths and diagnostics contain no gate value.
- Verify gate counterfactual interventions are removed.
- Run targeted tests, then the complete test suite.

## Existing Worktree Changes

Preserve the user's current checkpoint, FireFlow, and solver-contract edits while
making this change.
