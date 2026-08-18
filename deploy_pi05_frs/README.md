# pi0.5 and pi0.5 + FRS deployment

This directory deploys both plain pi0.5 and pi0.5 + FRS through the existing
`vb3_robot_server`. It does not modify the server or its robot-control code.
Start the server first, then start exactly one client mode.

## Shared configuration and modes

Both modes use the single `configs/deploy_pi05.yaml`. Set the shared
`checkpoint` and `norm_stats` fields to assets from the same pi0.5 training
run. The FRS-only assets remain under the `frs` section.

The two modes deliberately share the pi0.5 checkpoint, normalization stats,
model dimensions, camera mapping, and task prompt:

- Plain pi0.5 selects the `pi05` profile, requests `data_type: vision`, and
  ignores tactile images. It sends full legacy action chunks and waits for an
  acknowledgement from the server.
- pi0.5 + FRS selects the `frs` profile, requests `data_type: vitac`, and uses
  tactile observations only in the downstream FRS steering path. It uses the
  server's `frs_steering_v1` protocol.

`start_pi05.sh` and `start_pi05_frs.sh` are thin convenience wrappers: they
fix the mode and choose the shared default YAML. Both delegate argument
parsing, token resolution, Python selection, and Python module launch to
`start_remote_client.sh`. Set `PI05_DEPLOY_CONFIG` to replace the shared YAML
for either mode; the FRS wrapper also accepts the legacy
`PI05_FRS_DEPLOY_CONFIG` as a lower-priority compatibility override.

## Startup order and smoke checks

Make the robot server available before starting either client. First run its
dry run, then confirm both client launchers resolve the expected mode,
configuration, and entry point without loading a model or connecting to the
robot:

```bash
cd /home/typhon/vb3_robot_server
bash scripts/bimanual_smolvla.sh --dry-run

cd /home/typhon/FRS_Tact-pi05-frs-jax
export VB_ROBOT_TOKEN='...'
bash deploy_pi05_frs/scripts/start_pi05.sh --check
bash deploy_pi05_frs/scripts/start_pi05.sh --max-iterations 2
bash deploy_pi05_frs/scripts/start_pi05_frs.sh --check
bash deploy_pi05_frs/scripts/start_pi05_frs.sh --max-iterations 2
```

`VB_ROBOT_TOKEN` is preferred and must not be committed or printed. If it is
unset, the common launcher reads `VB3_TOKEN_FILE` (default:
`/home/typhon/vb3_robot_server/token_list.txt`). `--check` prints only token
provenance, never the token itself. `--max-iterations 2` is a bounded smoke
run: it limits plain mode by action chunk and FRS mode by FRS chunk.

## Hardware safety

Do not treat a successful `--check` as a robot validation. Before moving to
real hardware, verify the server dry run, use the bounded client run, confirm
the configured checkpoint and norm-stats paths, and inspect the server's
action trace. A trained operator must supervise every hardware run with a
working emergency stop immediately available. Stop and restart the client
after any disconnect or unexpected observation/action error; this deployment
does not automatically reconnect.
