# PI0.5 Right-Hand VLA-Only Shortcut Design

## Goal

Provide a dedicated right-hand deployment shortcut that evaluates the exact PI0.5 VLA used by the current FRS experiment while completely bypassing FRS and tactile inference.

## Selected approach

Reuse the existing plain PI0.5 client/server protocol. Add a dedicated deployment YAML and launcher instead of teaching the FRS steering protocol to support `frs.enabled: false`.

This keeps the ablation narrow and auditable:

- load `/home/typhon/FRS_Tact/checkpoints/model/pi05_task3_0830_1w`;
- use norm stats `insert_0102_train90` from the same checkpoint assets;
- expose only `observation.images.camera1` as `right_wrist_0_rgb`;
- use the same language prompt, seed, sample steps, 10 Hz control rate, 50-step horizon, and 50 executed steps per inference as the FRS run;
- project state and action through the existing `single-right-arm-7x10` adapter;
- use `observation.data_type: vision` so no tactile image is required or sent to the policy;
- omit the entire `frs` section so neither the tactile encoder nor FRS checkpoint can load;
- set `gripper.hysteresis_enabled: false` so the physical gripper follows the VLA output without the FRS run's threshold snap/latch;
- save observations under a distinct VLA-only output directory.

## Alternatives rejected

1. Add `frs.enabled: false` to the FRS client and server. This would require a new negotiation branch across both repositories and could accidentally retain FRS warmup, tactile history, or steering behavior.
2. Reuse `deploy_pi05_right.yaml` through environment overrides. That profile currently points at a different checkpoint, norm statistics, prompt, and task, so it is unsafe for this comparison.

## Files and interface

- Create `deploy_pi05/configs/deploy_pi05_vla_only_right.yaml` as the complete shared client/server experiment contract.
- Create `deploy_pi05/scripts/start_pi05_right_vla_only.sh` to select that YAML and invoke the existing plain PI0.5 launcher.
- Add focused tests that assert the shortcut uses mode `pi05`, the selected checkpoint and norm stats match the FRS profile, no FRS section exists, vision camera1 is preserved, and gripper hysteresis is disabled.
- Add a short README command pair for launching the server and client.

The server command will be:

```bash
cd /home/typhon/vb3_robot_server
bash scripts/bimanual_pi05.sh --mode vision \
  --config /home/typhon/FRS_Tact/deploy_pi05/configs/deploy_pi05_vla_only_right.yaml
```

The client command will be:

```bash
cd /home/typhon/FRS_Tact
bash deploy_pi05/scripts/start_pi05_right_vla_only.sh
```

## Runtime behavior

For each observation, the client performs one PI0.5 inference, unnormalizes the 50-step 10D right-arm action chunk, expands it to the bimanual wire contract with the left arm held, and sends the chunk through the existing plain-vision bridge. No FRS chunk start, tactile encoder, tactile history, reverse flow, FRS decode, or per-step steering request is involved.

## Safety and success criteria

- Existing server position, rotation, gripper-speed, and action-delta limits remain active.
- Manual confirmation remains enabled before START.
- The launcher check identifies `deploy_pi05.pi05_client`, not `deploy_pi05.remote_client`.
- Configuration validation succeeds in `pi05` mode and fails if accidentally treated as an FRS profile.
- Startup output names only the PI0.5 checkpoint and does not load an FRS/tactile checkpoint.
- Saved output uses a separate directory so the VLA-only run cannot be confused with an FRS run.
