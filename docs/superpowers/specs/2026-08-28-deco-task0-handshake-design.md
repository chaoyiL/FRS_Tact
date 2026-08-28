# DECO Task 0 Handshake Design

## Goal

Restore compatibility between the standalone DECO client and the current
`vb3_robot_server` handshake, which requires an integer `task` field.

## Design

DECO always uses the server's ordinary execution mode, `task = 0`. The value is
therefore part of the DECO deployment contract rather than a user-selectable
YAML option.

`deploy_deco.config.make_server_config()` will add `"task": 0` to every DECO
server configuration message. No `task` key will be added to
`deploy_deco/configs/deploy_deco.yaml`, and no task-selection validation or
runtime branching will be introduced.

This keeps the change isolated to DECO, preserves the existing YAML interface,
and prevents this deployment from accidentally enabling the server's task 1
motion behavior.

## Error Handling

The existing server remains responsible for validating the handshake. Because
the client always sends an integer zero, the new required-field validation is
satisfied without weakening the server contract or introducing a silent server
default.

## Tests

Update the DECO configuration test to assert that `make_server_config()` emits
`task == 0`. Existing artifact, frequency, horizon, and protocol assertions
remain unchanged.

The implementation will follow a red-green cycle: add the assertion and observe
it fail because `task` is absent, then add the single production mapping and
rerun the DECO tests.

## Non-goals

- Making task selectable from YAML.
- Supporting task 1 in DECO.
- Changing server-side validation or robot task behavior.
- Changing the checkpoint, action horizon, control frequency, or camera setup.
