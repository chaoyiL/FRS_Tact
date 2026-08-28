# DECO Right-Arm Deployment Design

## Goal

Deploy the CUDA-traced `insert/deco_stage1_best_gpu.ts` policy through the
existing DECO robot bridge without changing the robot server or the existing
bimanual DECO deployment.

The policy contract is fixed:

- two 224x224 RGB camera images;
- `single-right-arm-7x10` state/action profile;
- right-arm state shape `[1, 7]`;
- right-arm action shape `[1, 32, 10]`;
- 30 Hz sampling frequency.

## Chosen Approach

Keep the robot server in its existing bimanual mode. Add a small adapter in
`deploy_deco` that projects the server's bimanual state into the policy's
right-arm state and expands the policy's right-arm action back into the
server's bimanual action contract.

This is preferred over modifying the server because the server's current
`single_arm_mode` is left-arm-only. Enabling it for this policy would select
the wrong physical arm and remove one of the two required camera streams.

## Data Flow

1. The existing server publishes both RGB images and the 20D bimanual state.
2. The client copies state indices `[7:14]` into the policy observation. These
   fields are the right-arm relative-start pose6D and gripper width.
3. The insert policy produces a finite `(32, 10)` right-arm action chunk.
4. The client builds a left-arm hold action for every step:
   - translation delta `[0, 0, 0]`;
   - identity Rotation-6D columns `[1, 0, 0, 0, 1, 0]`;
   - absolute gripper width from bimanual state index `6` of the same
     observation.
5. The client concatenates `[left_hold, right_policy_action]` into `(32, 20)`
   and sends it with the original observation sequence number.
6. The unchanged bimanual server converts and executes the chunk normally.

## Components

### Artifact contract

`deploy_deco/artifact.py` will accept exactly two Stage 1 profiles:

- existing bimanual `20D -> 20D`;
- single-right-arm `7D -> 10D` with
  `state_action_profile=single-right-arm-7x10` and
  `controlled_arms=["right"]`.

Existing camera, dtype, action semantics, normalization, frequency, and hash
checks remain shared.

### Right-arm adapter

Add `deploy_deco/right_arm_adapter.py` with two focused functions:

- project a finite `(20,)` bimanual state to a copied `(7,)` right-arm state;
- expand a finite `(H, 10)` right-arm chunk and the matching 20D observation
  state into `(H, 20)`.

The adapter performs only shape and finite-value validation needed to prevent
misrouting. It does not add a new safety subsystem.

### Configuration and runtime

Add `deploy_deco/configs/deploy_deco_right.yaml` and
`deploy_deco/scripts/start_deco_right.sh`.

The right-arm YAML points to
`checkpoints/model/deco_0828/insert/deco_stage1_best_gpu.ts`, declares the
single-right-arm profile, uses 30 Hz, horizon 32, and
`steps_per_inference: 24` to match the current DECO server configuration.

The client uses the profile to enable the adapter. The wire configuration sent
to the server deliberately remains `single_arm_mode: false`, because the
physical runtime stays bimanual. The existing `deploy_deco.yaml` and
`start_deco.sh` retain their current behavior.

### Runtime loop

For both warmup and inference, the right profile passes a projected copy of
the observation to `DECOPolicy`. During inference only, the returned right-arm
chunk is expanded before `send_action`. The normal bimanual profile remains a
pass-through path.

## Errors

Fail before robot START when the YAML profile and artifact profile differ.
Reject only the contract violations required by the adapter: wrong state or
action shape, non-finite values, or a right profile that does not name the
right arm. No additional policy-output limits or hardware safety checks are
introduced.

## Tests

Use test-driven development for:

- accepting both exact artifact profiles and rejecting mixed dimensions;
- projecting state `[7:14]` without mutating the source observation;
- expanding right actions with identity left rotation, zero left translation,
  and the current left gripper width;
- sending `(32, 20)` from a simulated right-arm remote-client loop;
- loading the checked-in right YAML against the insert artifact;
- preserving the existing bimanual behavior.

Final verification includes the `deploy_deco` test suite, configuration
`--check`, and one CUDA forward pass using the actual insert artifact. No live
robot motion is part of automated verification. The first hardware run should
use `--max-iterations 1` with the existing manual START confirmation.

## Out of Scope

- modifying `/home/typhon/vb3_robot_server`;
- changing the existing server's left-only `single_arm_mode` behavior;
- native right-only controller or IK support;
- changing the existing bimanual DECO configuration;
- adding a new safety or protocol framework.
