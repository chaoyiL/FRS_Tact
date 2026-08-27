# PI0.5 FRS Right Pre-Close Limit Design

## Goal

Allow `task1.right_preclose_forward_m` values up to and including `0.05 m`
for PI0.5 FRS deployment. Keep the configured deployment value at `0.040 m`.

## Scope

The FRS deployment parser in `vb3_robot_server` will change its accepted range
from `[0.001, 0.02]` to `[0.001, 0.05]`. The robot environment's final Task 1
validation will accept the same range so a value accepted by the FRS parser is
not rejected during environment construction.

The plain-vision PI0.5 deployment parser will remain limited to `0.02 m`.
Therefore this change does not expand the values accepted by the ordinary
PI0.5 configuration interface. Motion direction, speed, tolerance, stable-step
count, and pre-close sequencing remain unchanged.

## Validation and tests

FRS parser tests will demonstrate that `0.050 m` is accepted and `0.0501 m` is
rejected. Robot environment validation tests will cover the same boundary and
retain the minimum-bound checks. Existing right pre-close scheduling tests will
verify that the behavioral sequence remains unchanged.

## Safety and operation

The upper bound remains finite and explicit. The compensation still moves the
right end effector along robot-base `+X` while holding Y, Z, and orientation.
Both the robot server and FRS client must be restarted after the change because
they load the shared YAML only at startup.
