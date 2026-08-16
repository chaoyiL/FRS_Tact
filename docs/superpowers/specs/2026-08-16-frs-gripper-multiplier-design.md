# FRS Gripper Multiplier Design

## Goal

Replace FRS's additive robot-space gripper adjustment with a configurable
multiplicative adjustment. For the two gripper-width action dimensions, values
strictly below a configured threshold are multiplied by a configurable factor;
values equal to or above the threshold remain unchanged.

## Configuration

Keep the existing outer `frs.gripper_gain` section to minimize configuration
churn, but replace its additive `gain` field with an explicit `multiplier`:

```yaml
gripper_gain:
  threshold: 0.1
  multiplier: 1.5
```

`gripper_gain: null` or an omitted section continues to disable the
postprocessor. Both values must be real, finite, non-boolean numbers.
`multiplier` must be strictly positive.

The old `gain` field is not accepted as an alias. A stale additive
configuration fails with a missing-`multiplier` error instead of silently
changing meaning.

## Runtime behavior

The postprocessor remains after normalized-action selection and action
unnormalization, so the threshold and multiplication operate in robot-space
gripper-width units. It continues to require a 20-dimensional action and
modifies only zero-based indices 9 and 19.

For each gripper width `w`:

```text
w < threshold   -> w * multiplier
w >= threshold  -> w
```

The comparison remains strict: a width exactly equal to the threshold is not
modified. The normalized selected action, full decoded chunk, and all
non-gripper dimensions remain unchanged.

The default FRS deployment uses `threshold: 0.1` and `multiplier: 1.5`.
Existing user changes elsewhere in `deploy_frs.yaml`, including checkpoint
paths and unrelated runtime values, must be preserved.

## Error handling

Startup parsing and lightweight configuration validation share the same schema
checks. They reject:

- a non-mapping, non-null `gripper_gain`;
- a missing `threshold` or `multiplier`;
- boolean, non-numeric, NaN, or infinite values;
- a zero or negative `multiplier`.

Runtime continues to reject enabled gripper adjustment for actions not shaped
as a 20-dimensional vector. No clipping or fallback is introduced.

## Verification

Tests cover:

- disabled configuration remains `None`;
- valid `threshold` and `multiplier` are parsed and stored;
- stale `gain`, missing fields, invalid numeric values, and non-positive
  multipliers fail clearly;
- below-threshold widths are multiplied by 1.5 in robot space;
- widths equal to and above the threshold remain unchanged;
- only action indices 9 and 19 change;
- normalized outputs remain unchanged;
- the existing non-20D runtime guard remains active.
