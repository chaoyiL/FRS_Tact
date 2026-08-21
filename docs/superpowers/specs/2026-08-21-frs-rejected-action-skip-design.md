# FRS Rejected Action Skip Design

## Goal

When an FRS action fails the robot server's numeric action-safety validation,
do not schedule it on the controller and continue the remaining actions in the
same chunk. Preserve fail-fast behavior for protocol identity errors,
conflicting duplicates, malformed acknowledgements, transport failures, and
controller execution failures.

## Server behavior

The robot server remains the safety authority. In the FRS response loop it
validates each proposed action before conversion or controller submission. If
`validate_single_action` raises `UnsafeActionError`, the server publishes the
existing matching `rejected` acknowledgement and trace record, advances past
that action, and continues requesting the next action. The rejected action has
no absolute waypoint and no controller records.

Protocol-schema errors and identity/order errors continue through the existing
fatal rejection path. Controller failures also remain fatal because they do not
prove that an action was safely skipped before reaching hardware.

## Pi0.5 client behavior

After verifying acknowledgement type and identity, the Pi0.5 client treats a
matching `rejected` acknowledgement as a warning and resumes its receive loop.
It continues to raise on malformed or mismatched acknowledgements. `scheduled`
and `stale` behavior is unchanged.

## Compatibility and observability

No wire-schema change is required. Existing `frs_steer_ack.status=rejected`
and `chunk_trace.jsonl` fields remain authoritative. A warning identifies the
skipped `(chunk_id, request_id, action_index)`. The server's trace preserves
the precise safety error.

## Testing

Server tests prove that an unsafe action publishes `rejected`, never calls the
converter/controller, and proceeds to a later safe action. Existing fatal
tests cover protocol and controller rejection behavior. Pi0.5 client tests
prove a matching rejected acknowledgement is skipped and the next request and
chunk end are processed, while acknowledgement identity mismatches remain
fatal.
