# YAML-Controlled Execution Steps Design

## Goal

Make `configs/deploy_smolvla_jax.yaml` the single source of truth for the
number of actions executed after each inference. Changing only
`control.steps_per_inference` must be sufficient to select any value from 1
through the checkpoint action horizon of 20.

## Client behavior

- Keep `control.action_horizon` equal to the checkpoint `chunk_size` (20).
- Validate `control.steps_per_inference` is in the inclusive range
  `[1, action_horizon]`.
- Do not require `control.steps_per_inference` to equal checkpoint metadata
  `n_action_steps`. The remote client calls `predict_action_chunk()` directly,
  so the model still returns the complete 20-step chunk.
- Send the validated YAML value to the robot server as
  `steps_per_inference`.
- Do not modify checkpoint weights or checkpoint metadata.

## Robot server behavior

- Accept the client-provided `steps_per_inference` after validating it is in
  `[1, action_horizon]`.
- Remove the hidden default cap of 5 from both `steps_per_inference` and
  `max_executed_actions`; set the default maximum to the full 20-step action
  horizon.
- Execute exactly the validated client-requested number of actions. Do not
  silently reduce the value, because the client uses the same value to align
  the remaining action chunk for the next inference.

## User interface

The only required user change is:

```yaml
control:
  steps_per_inference: 10
```

No extra launcher option and no checkpoint rewrite are required.

## Validation

- Client tests cover checkpoint `n_action_steps=5` with deployment
  `steps_per_inference=10` and verify the server receives 10.
- Client and server reject values below 1 or above 20.
- Server tests verify requested values 5 and 10 are executed without a hidden
  five-step cap.
- Existing 20-step action-shape and bimanual contract tests remain passing.
