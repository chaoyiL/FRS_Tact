# Pure-Vision Pi0.5 Gripper Hysteresis Switch

## Goal

Add one explicit Boolean switch to the pure-vision Pi0.5 deployment YAML so an
operator can disable the existing left/right gripper hysteresis latch without
changing thresholds or code.

## Scope

The shared deployment configuration gains this field:

```yaml
gripper:
  hysteresis_enabled: true
```

`true` preserves the current behavior. Each gripper latches closed when the
model-predicted width reaches its close threshold, remains at the configured
closed command through the deadband, and unlatches at its reopen threshold.

`false` disables hysteresis for both grippers. The robot server passes no
hysteresis parameters to `BimanualUmiEnv`, which already interprets three
`None` values per gripper as disabled. Each gripper then follows the existing
calibrated and clipped model-predicted width directly.

This is one switch for both grippers. Per-arm switches are out of scope.

## Configuration and Runtime Flow

The pure-vision client and robot server load the same YAML independently. The
client does not execute the gripper latch; the robot server does. Therefore:

1. `deploy_pi05/configs/deploy_pi05.yaml` declares
   `gripper.hysteresis_enabled: true` so the checked-in profile retains its
   current behavior by default.
2. `/home/typhon/vb3_robot_server/deploy_scripts/pi05_deployment_config.py`
   parses the field as a strict Boolean and exposes it in its deployment
   configuration object.
3. When enabled, the server continues to require and validate all six existing
   threshold and closed-command values exactly as today.
4. When disabled, the server passes `None` for all six hysteresis arguments to
   `BimanualUmiEnv`. Threshold values may remain in the YAML and are ignored at
   runtime, making A/B deployment changes a one-line edit.
5. `/home/typhon/vb3_robot_server/deploy_scripts/bimanual_pi05_online.py`
   remains responsible for wiring the parsed choice into the environment.

The environment's hysteresis algorithm is not changed.

## Validation and Errors

`gripper.hysteresis_enabled` is required and must be YAML `true` or `false`;
numeric and string substitutes are rejected. When it is `true`, the current
finite bounds and `close_threshold < reopen_threshold` checks remain unchanged.
When it is `false`, the six numeric fields remain present for easy re-enabling
but are not used to construct the environment.

## Safety and Compatibility

The unsafe action-chunk rejection and re-inference behavior remains always on,
including `control.max_consecutive_unsafe_chunks: 3`. Action shape and finite
checks, position/rotation delta limits, gripper bounds, stale-action filtering,
speed limits, the 50 ms first-action scheduling margin, and every other runtime
setting remain unchanged.

Existing configurations must add the required switch. The repository's default
configuration uses `true`, so its behavior is identical before and after this
change.

## Testing

Robot-server tests will cover:

- strict parsing of `true` and `false`;
- rejection of non-Boolean switch values;
- enabled mode preserving the six current environment arguments;
- disabled mode passing six `None` values to the environment;
- unchanged unsafe-chunk configuration and execution behavior.

The focused configuration and launcher tests will run first, followed by the
relevant pure-vision Pi0.5 test set. No hardware execution is required.
