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

## Plain pi0.5 legacy dry-run

`bimanual_smolvla.sh --dry-run` implements only the legacy
`obs`/`action`/`action_ack` exchange. It can validate the plain pi0.5 client,
but it does not implement `frs_steering_v1` and cannot validate the FRS client.
Use two terminals for the plain, hardware-free smoke test.

Terminal A (legacy-only dry-run server):

```bash
cd /home/typhon/vb3_robot_server
bash scripts/bimanual_smolvla.sh --dry-run
```

Terminal B (plain pi0.5 client):

```bash
cd /home/typhon/FRS_Tact-pi05-frs-jax
export VB_ROBOT_TOKEN='...'

# Plain pi0.5
bash deploy_pi05_frs/scripts/start_pi05.sh --check
bash deploy_pi05_frs/scripts/start_pi05.sh --max-iterations 2
```

`VB_ROBOT_TOKEN` is preferred and must not be committed or printed. If it is
unset, the common launcher reads `VB3_TOKEN_FILE` (default:
`/home/typhon/vb3_robot_server/token_list.txt`). `--check` prints only token
provenance, never the token itself. In this dry-run flow, `--max-iterations 2`
limits plain mode by action chunk.

## FRS real-server flow

The launcher above provides no hardware-free FRS server dry-run. Run the FRS
client only against a real `vb3_robot_server` flow already verified to support
`frs_steering_v1`; this repository does not prescribe or invent an additional
server flag. You may inspect the client configuration without connecting, then
start a bounded FRS run after that server is ready:

```bash
bash deploy_pi05_frs/scripts/start_pi05_frs.sh --check
bash deploy_pi05_frs/scripts/start_pi05_frs.sh --max-iterations 2
```

## Hardware safety

Do not treat a successful `--check` or the plain legacy dry-run as FRS robot
validation. Before START, confirm that the real server negotiated
`frs_steering_v1`, confirm the configured checkpoint and norm-stats paths, and
use a bounded client run. A trained operator must supervise every hardware run
with a working emergency stop immediately available. Stop and restart the
client after any disconnect or unexpected observation/action error; this
deployment does not automatically reconnect.
