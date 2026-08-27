# PI0.5 FRS Right Approach Opening Design

## Goal

Keep the Task 1 right gripper physically open to at least `0.035 m` while it
approaches the green tube and before the VLA requests closure.

## Configuration

Add the following required Task 1 field to the shared FRS deployment YAML:

```yaml
task1:
  right_approach_min_open_m: 0.035
```

The FRS server parser accepts finite values in `[0.01, 0.04] m`. Task 0 projects
the field as inactive. The ordinary plain-vision PI0.5 parser is unchanged.

## Runtime behavior

The robot server applies the floor after converting the raw VLA width to a
physical gripper command. For Task 1, while the right gripper latch is open and
the raw VLA width is above the close threshold, the command is:

```text
max(calibrated_vla_command, right_approach_min_open_m)
```

The floor is right-arm-only. It does not change the raw VLA close-intent test.
When the VLA requests closure, the existing right pre-close sequence starts,
preserves the measured open command during the configured forward move, and
then applies the existing closed command. Reopening, left-arm behavior, motion
targets, and FRS action tensors remain unchanged.

## Validation

Focused tests cover a small calibrated VLA approach command being raised to
`0.035 m`, a larger command passing through unchanged, and a close request
entering the existing pre-close path. Parser coverage checks the configured
value and rejects values outside `[0.01, 0.04] m`.

Both the robot server and FRS client must be restarted after the shared YAML
changes because they read it only during startup.
