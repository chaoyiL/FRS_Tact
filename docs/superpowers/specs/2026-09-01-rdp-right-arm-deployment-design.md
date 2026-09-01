# RDP Single-Right Deployment Design

## Goal

Deploy the downloaded `rdp_0831` single-right RDP policy against the existing
bimanual robot bridge: the left arm stays connected and receives a relative
no-op while the right arm executes the policy output.

## Scope

- Preserve the existing bimanual RDP configuration and launcher.
- Add a dedicated right-arm YAML and launcher.
- Keep the server wire contract bimanual (`20D` state and `20D` action).
- Keep both RGB cameras and all four tactile streams because the checkpoint was
  trained with those inputs.
- Do not change the external robot server or add unrelated safety machinery.

## Artifacts

The right-arm configuration uses this matched bundle:

- `checkpoints/model/rdp/rdp_0831/ldp/latest.ckpt`
- `checkpoints/model/rdp/rdp_0831/at/latest.ckpt`
- `checkpoints/model/rdp/rdp_0831/pca/tactile_pca_insert_01_02_encoder0824_2x15.npz`
- `checkpoints/encoder/encoder_ckpt_0824`

The policy contract is `single-right-arm-7x10`: a `7D` right-arm state and a
`10D` relative right-arm action.

## Components

### Configuration and launcher

Create `deploy_RDP/configs/deploy_pick_tube_rdp_right.yaml` with the right-arm
artifact paths and `model.state_action_profile: single-right-arm-7x10`.
Create `deploy_RDP/scripts/start_pick_tube_rdp_right.sh`, which selects that
configuration and delegates to `start_pick_tube_rdp_client.sh`.

### Right-arm wire adapter

Add a small RDP-local adapter with two pure functions:

- Project the bridge's finite `20D` state to `state[7:14]` for policy input.
- Expand a finite `[H,10]` policy action to `[H,20]`. The left-arm half uses
  zero translation, identity 6D rotation, and the current left gripper width
  from `state[6]`; the right-arm half is copied unchanged.

### Runtime integration

Resolve the configured state/action profile at startup. The existing dual-arm
profile continues to consume and emit `20D`. The single-right profile projects
state before inference, accepts a `10D` policy output, then expands it before
sending to the bridge. Server negotiation remains `single_arm_mode: false` so
the existing bimanual observation and action wire format is unchanged.

Checkpoint metadata must agree with the selected profile's state and action
dimensions. Existing tactile-dimension and artifact-pairing checks remain in
place.

## Failure behavior

Reject unsupported profiles, invalid state/action shapes, or checkpoint/config
dimension mismatches before sending robot actions. Existing connection,
authentication, and artifact behavior is otherwise unchanged.

## Tests

- Unit-test `20D -> 7D` right-state projection.
- Unit-test `[H,10] -> [H,20]` expansion and left-arm no-op values.
- Test single-right runtime accepts `7D/10D` policy dimensions while the dual
  profile retains `20D/20D` behavior.
- Test the new YAML paths/profile and launcher delegation.
- Run the focused RDP deployment test suite and shell syntax checks.

## Success criteria

Launching `start_pick_tube_rdp_right.sh` loads the top-level `rdp_0831`
single-right bundle, keeps the bridge in bimanual mode, supplies `7D` state to
the policy, and sends `[1,20]` actions whose left half is a relative no-op and
whose right half is the policy's `[1,10]` output.
