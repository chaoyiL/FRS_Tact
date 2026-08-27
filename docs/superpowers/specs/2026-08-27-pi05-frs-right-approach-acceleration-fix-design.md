# PI0.5 FRS Task 1 Right-Approach Acceleration and Close-Gate Fix

## Goal

Speed up only the Task 1 right arm while it approaches the green tube before
the VLA requests closure, and restore closure after increasing the right
approach gripper opening to `0.045 m`.

## Selected Design

Add the FRS-only Task 1 YAML field
`right_approach_translation_gain: 1.5`. Keep the existing
`approach_translation_gain: 1.2` as the left-arm approach gain and keep
`translation_gain: 1.5` for post-latch translation. During Task 1 motion-gain
application, use the new gain only for the unlatched right arm. Task 0, plain
PI0.5 deployment, left-arm motion, rotations, and the pre-close speed remain
unchanged.

On the robot side, treat right-gripper feedback up to the configured
`task1_right_approach_min_open_m` as valid open feedback. This changes the
right Task 1 feedback ceiling from the hard-coded `0.04 m` to
`max(0.04, task1_right_approach_min_open_m)`, while leaving the left feedback
range unchanged. A raw right VLA width at or below the existing `0.09` close
threshold can then start pre-close from a measured `0.045 m` opening.

## Alternatives Considered

1. Raise the shared `approach_translation_gain` to `1.5`. Rejected because it
   also accelerates the left arm.
2. Multiply right-arm targets in the robot server. Rejected because the FRS
   steering layer already owns motion gains and server-side scaling would
   duplicate that responsibility.
3. Change only `right_preclose_speed_m_s`. Rejected because the requested
   acceleration is before closure, and the current failure prevents pre-close
   from starting at all.

## Validation

Use focused tests only:

- FRS configuration projects the new right-only gain and rejects invalid
  values.
- Task 1 motion gain applies `1.2` to unlatched left translation and `1.5` to
  unlatched right translation.
- Measured right-gripper feedback at `0.045 m` is accepted and a raw VLA close
  request starts pre-close.
- Existing Task 0 and right-gripper unsafe-feedback behavior remain unchanged.

The user will perform the physical robot validation after restarting the
server and client.
