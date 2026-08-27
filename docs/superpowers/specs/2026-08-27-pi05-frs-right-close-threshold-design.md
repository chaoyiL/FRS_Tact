# PI0.5 FRS Task 1 Right Close Threshold Design

## Goal

Allow the Task 1 right gripper to request closure when recent VLA predictions
bottom out near `0.10` instead of crossing the current `0.09` threshold.

## Design

Change only the shared PI0.5 FRS YAML hysteresis values:

```yaml
right_close_threshold: 0.105
right_reopen_threshold: 0.115
```

Both the FRS steering latch and robot-side Task 1 close detector already
consume these shared values. Keep `right_closed_command: 0.01`,
`right_approach_min_open_m: 0.045`, right approach gain `1.5`, and every
pre-close motion parameter unchanged.

The chosen close threshold captures the two latest observed minima
(`0.099625` and `0.104363`) late in their trajectories. A `0.11` threshold is
not used because all actions in the latest trace were already at or below
`0.11`, which could request closure at episode start. The `0.115` reopen value
preserves a `0.01` hysteresis gap and remains strictly above the close value.

## Validation

Focused tests must verify that the shared FRS configuration projects
`0.105/0.115`, a raw right width at `0.104` starts Task 1 pre-close from valid
`0.045` feedback, and a width above `0.105` does not request closure. Plain
PI0.5 configuration and left-hand thresholds remain unchanged. The user will
perform physical validation after restarting server and client.
