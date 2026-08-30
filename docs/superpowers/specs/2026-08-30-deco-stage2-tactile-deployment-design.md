# DECO Stage 2 Tactile Deployment Design

## Goal

Deploy
`/home/typhon/FRS_Tact/checkpoints/model/deco_0830/insert_stage2/deco_stage2_latest.ts`
on the bimanual VB3 robot while preserving every existing DECO Stage 1 and
Bread deployment entrypoint.

Stage 2 receives two visual streams, four tactile streams, and a seven-value
right-arm state. It produces a 32-step, 10-value right-arm action chunk. The
robot bridge remains bimanual: the client projects the 20-value server state to
the right arm and expands the model output back to a safe 20-value wire action.

## Non-goals

- Do not replace or change the behavior of `start_deco_right.sh`,
  `start_deco.sh`, or the Bread phase entrypoint.
- Do not fork the complete client or robot runtime.
- Do not introduce long-running sensor audits, a new safety state machine, or
  multiple commissioning approval levels.
- Do not change the Stage 2 checkpoint, its normalization, or its TorchScript
  preprocessing.
- Do not switch the robot server to its legacy physical single-arm mode. Stage
  2 needs both wrist camera devices and the server's 20-value bimanual state.

## Chosen Architecture

Use separate Stage 2 configuration and launchers while extending shared
client/runtime components to understand both Stage 1 and Stage 2 contracts.
Artifact metadata, rather than a filename or a YAML boolean, selects the model
calling convention.

The existing Stage 1 paths remain unchanged. A wrong checkpoint, data type, or
observation profile must fail during configuration validation, before the
client can send `START`.

### Entrypoints

Add the following client files in `/home/typhon/FRS_Tact`:

- `deploy_deco/configs/deploy_deco_stage2_right.yaml`
- `deploy_deco/scripts/start_deco_stage2_right.sh`

Add the following server files in `/home/typhon/vb3_robot_server`:

- `configs/deco_stage2_server_config.py`
- `deploy_scripts/bimanual_deco_stage2_online.py`
- `scripts/bimanual_deco_stage2.sh`

The Stage 2 server config inherits the existing DECO hardware and action
defaults, then overrides only the policy identity and observation contract. It
uses `data_type=vitac`, `observation_profile=deco_vitac_224`, two cameras,
224-by-224 RGB observations, 30 Hz, a 32-step action horizon, and the legacy
bimanual chunk protocol. Its `policy_family` is the distinct value
`deco-stage2`, which selects Stage 2-only wire validation without changing
Stage 1 behavior. As in the existing DECO legacy-chunk path, the client owns
the deployment YAML and sends its validated runtime projection during the
bridge handshake. The shared server's `--config` option belongs to its separate
SmolVLA local-config startup path and is not used by either DECO Stage 1 or
Stage 2; the dedicated Stage 2 client launcher is therefore the sole selector
of `deploy_deco/configs/deploy_deco_stage2_right.yaml`.

The Stage 2 server entrypoint selects
`configs.deco_stage2_server_config` through `VB3_SERVER_CONFIG_MODULE` before
importing the shared runtime. The Stage 1 entrypoint continues selecting
`configs.deco_server_config`.

## Artifact and Configuration Contract

### Artifact formats

The client accepts both existing Stage 1 format
`sudo-upstream-deco-stage1-torchscript-v1` and Stage 2 format
`sudo-upstream-deco-stage2-torchscript-v1`.

Stage 2 metadata validation additionally requires:

- `input.images == [1, 2, 3, 224, 224]`
- `input.tactile_images == [1, 4, 3, 224, 224]`
- `input.observation == [1, 7]`
- `output.action == [1, 32, 10]`
- tactile dtype `float32` and range `[0.0, 1.0]`
- the exact tactile field order defined below
- `input.stream_order == camera_names + tactile_field_order`
- embedded TorchScript metadata matching its sidecar under the current hash
  comparison rules

The selected latest artifact has epoch 10 and SHA256
`ebd606ed8b4932e14fe0ec70718922a6829c1a9f4ee72ab42e694a0445fbc87d`.
The sidecar is present and must remain mandatory.

### Stage 2 client configuration

The new YAML uses:

```yaml
checkpoint: /home/typhon/FRS_Tact/checkpoints/model/deco_0830/insert_stage2/deco_stage2_latest.ts
device: cuda:0
seed: 0
model:
  state_action_profile: single-right-arm-7x10
observation:
  data_type: vitac
  observation_profile: deco_vitac_224
  language_prompt: Use the right hand to insert the object.
  single_arm_mode: true
  controlled_arm: right
  black_camera0: true
  no_state_obs_mode: false
control:
  control_frequency: 30.0
  controller_frequency: 80.0
  action_horizon: 32
  steps_per_inference: 32
runtime:
  auto_start: false
  warmup_runs: 5
  max_iterations: 0
```

Connection fields match the current right-arm config. The local
`single_arm_mode: true` describes the model. The config sent over the bridge
must continue using `single_arm_mode: false` so the server initializes both
arms and both wrist camera devices and publishes a 20-value state.

The client config validator enforces these pairings:

- Stage 1 artifact: `vision` plus `deco_vision_224`
- Stage 2 artifact: `vitac` plus `deco_vitac_224`

Cross-pairings are rejected before model loading or connection startup.

## Observation Data Flow

The server captures two 3840-by-800 wrist triptychs. Each triptych is split,
without rotation or flipping, into:

```text
left tactile panel | visual panel | right tactile panel
```

After the existing RGB conversion and scheme-A resize/crop, the bridge payload
contains six 224-by-224 HWC `uint8` RGB images in this semantic order:

1. `observation.images.camera0`
2. `observation.images.camera1`
3. `observation.images.tactile_left_0`
4. `observation.images.tactile_right_0`
5. `observation.images.tactile_left_1`
6. `observation.images.tactile_right_1`

Index `0` is the left-wrist triptych device and index `1` is the right-wrist
triptych device. All four tactile fields are required even though only the
right arm is actuated. The model has sensor-position embeddings, so the tactile
order cannot be changed.

The right-arm adapter copies the full observation, replaces only visual
`camera0` with zeros, and projects `observation.state[7:14]`. It must not zero,
drop, or reorder either tactile field from wrist 0. The training manifest proves
that visual camera0 was pure black for every frame in both insert training
sources.

The client converts each image from HWC `uint8` RGB to contiguous CHW
`float32` in `[0, 1]` and stacks:

- visual tensor: `[1, 2, 3, 224, 224]`
- tactile tensor: `[1, 4, 3, 224, 224]`
- right-arm state tensor: `[1, 7]`

The client performs no resize, crop, flip, ImageNet normalization, or tactile
normalization. Those operations are embedded in the TorchScript artifact.

## Policy and Action Data Flow

`DECOPolicy` selects one of three explicit calls:

- regular Stage 1: `model(images, state)`
- phase-conditioned Stage 1: `model(images, state, phase)`
- tactile Stage 2: `model(images, tactile_images, state)`

The Stage 2 result must be finite with shape `[32, 10]`. The existing right-arm
adapter expands it to `[32, 20]`:

- left translation is zero
- left Rotation-6D is identity
- left gripper holds its current server-observed width
- the Stage 2 output occupies the right 10 values

Before conversion or scheduling, the Stage 2 server verifies that the left-arm
wire slice has exactly this hold contract. It then applies the existing finite,
shape, gripper, per-step translation, and per-step rotation validation to the
full chunk.

## Shared Client Changes

Modify shared files without changing Stage 1 behavior:

- `deploy_deco/artifact.py`: validate both artifact formats and expose whether
  an artifact uses tactile input.
- `deploy_deco/policy.py`: collect and stack tactile inputs for Stage 2 and use
  the explicit three-way call dispatch.
- `deploy_deco/config.py`: validate artifact/data-type/profile pairings and
  forward the configured data type and observation profile to the server.
- `deploy_deco/remote_client.py`: log the artifact mode and tactile order and
  add the two bounded non-production modes described below.
- `deploy_deco/right_arm_adapter.py`: retain its implementation; add regression
  coverage proving tactile fields survive projection.
- `deploy_deco/bridge_client.py`: retain its implementation; its NumPy msgpack
  transport already supports the four additional arrays.

## Shared Server Changes

Modify only the shared seams needed by the separate Stage 2 entrypoint:

- `deploy_scripts/observation_profiles.py`: add `deco_vitac_224` with
  `data_type=vitac`, resolution `(224, 224)`, and two cameras.
- `deploy_scripts/bimanual_smolvla_online.py`: accept an optional
  `--max-executed-actions` hard cap, apply the right-only wire validation for
  the Stage 2 policy family, and otherwise retain existing action execution.
- Reuse `real_world/bimanual_umi_env.py` and
  `real_world/real_inference_util.py` unchanged for the base tactile path; they
  already split and publish all four tactile fields.

`--max-executed-actions` is independent of the negotiated 32-step model
horizon. If omitted, its effective value is `steps_per_inference`. If supplied,
the server validates the entire 32-step chunk and schedules at most the first
`min(steps_per_inference, max_executed_actions)` fresh actions. This lets the
first physical test execute one timestep without changing the production
model/config contract.

The active server worktree contains uncommitted user changes in the shared
runtime, DECO config, environment, and tests. Implementation must preserve and
integrate with those changes rather than overwrite or reset them.

## Bounded Deployment Modes

### Hardware-free server dry-run

Add client option `--server-dry-run`. It uses the normal Stage 2 config and
model with the server's synthetic `vitac` payload, sends the requested bounded
number of actions, then sends `STOP` immediately. It does not wait for the
post-action observation used by a real bounded run, because the server dry-run
stops publishing observations after its final exchange.

This fixes the current client/server final-exchange deadlock while leaving real
deployment behavior unchanged.

### Real observe-only

Add client option `--observe-only`. It connects to a normal Stage 2 server,
receives the real warmup observation, validates and prepares all six image
streams, runs the configured warmup inferences, and writes one timestamped
observation bundle under `deploy_deco/outputs/observe_only_*`. The bundle
contains six PNG files and a JSON summary of keys, shapes, dtypes, ranges, and
the predicted action shape/range. The PNG files represent the post-projection,
model-bound inputs, so visual camera0 is black while all four tactile streams
retain their received pixels.

Observe-only never sends `START` or an action. It sends `STOP` in normal cleanup
and exits after the one warmup observation.

## Concise Rollout

Only the following three gates are required.

### 1. Protocol dry-run

Start the Stage 2 server without hardware:

```bash
cd /home/typhon/vb3_robot_server
bash scripts/bimanual_deco_stage2.sh \
  --dry-run \
  --dry-run-iterations 1 \
  --action-timeout-s 30
```

In another terminal, start the bounded client:

```bash
cd /home/typhon/FRS_Tact
bash deploy_deco/scripts/start_deco_stage2_right.sh \
  --server-dry-run \
  --max-iterations 1
```

Success means one synthetic six-stream observation produces one valid 32-by-20
wire action and both processes exit after `STOP`, with no hardware initialized.

### 2. Real observe-only

Start the Stage 2 server normally, then run:

```bash
cd /home/typhon/FRS_Tact
bash deploy_deco/scripts/start_deco_stage2_right.sh --observe-only
```

Inspect the saved six PNG files once. The four tactile panels must be present in
metadata order and must not all be black. No extended sensor stability audit is
required.

### 3. One-action physical smoke, then normal deployment

For the first physical action, start the server with a one-action cap and
conservative command-line limits:

```bash
cd /home/typhon/vb3_robot_server
bash scripts/bimanual_deco_stage2.sh \
  --max-executed-actions 1 \
  --max-pos-speed 0.05 \
  --max-rot-speed 0.10 \
  --max_gripper_speed 0.02 \
  --max_action_pos_delta 0.01 \
  --max_action_rot_delta 0.17 \
  --action-timeout-s 5
```

Run one client iteration:

```bash
cd /home/typhon/FRS_Tact
bash deploy_deco/scripts/start_deco_stage2_right.sh --max-iterations 1
```

After confirming the single right-arm motion has the intended direction, run
the same server entrypoint without the temporary cap/commissioning limits and
run the client without `--max-iterations` for the normal 32-step loop. A
physical emergency stop remains available during the one-action smoke; no new
software approval state machine is added.

## Failure Handling

The deployment fails closed in these cases:

- missing or mismatched sidecar/hash/embedded metadata
- Stage 2 paired with `vision` or a non-Stage-2 observation profile
- missing, duplicated, reordered, non-RGB, non-finite, or wrong-shaped tactile
  input
- model output with wrong shape or non-finite values
- non-hold left-arm wire action
- action outside existing server translation, rotation, or gripper bounds
- connection loss, action timeout, or client `STOP`

Cleanup continues using the current bridge and controller shutdown paths. No
new force/contact limit or hardware emergency-stop API is introduced.

## Tests

### Client tests

Extend the existing `deploy_deco/tests` suite to cover:

- Stage 2 metadata success and malformed tactile metadata rejection
- legacy Stage 1 and Bread phase regression
- exact four-tactile stack order and missing/invalid stream rejection
- correct Stage 1, Bread phase, and Stage 2 model call signatures
- Stage 2 `vitac` configuration success and all cross-mode pairings rejected
- right-arm projection preserving all tactile keys while blacking only visual
  camera0
- Stage 2 remote loop expanding `[32, 10]` to `[32, 20]`
- observe-only never sending `START` or an action
- server-dry-run terminating without waiting for a final observation

Run:

```bash
cd /home/typhon/FRS_Tact
deploy_deco/.venv/bin/pytest -q deploy_deco/tests
bash deploy_deco/scripts/start_deco_stage2_right.sh --check
```

Add one opt-in CUDA smoke that loads the real latest artifact and proves a
finite `[1, 32, 10]` output for valid fixed-shape inputs.

### Server tests

Cover:

- `deco_vitac_224` profile resolution and strict pairing
- separate Stage 1 and Stage 2 launcher/config-module selection
- synthetic Stage 2 dry-run containing all four tactile streams
- exact tactile field mapping already covered by tactile-orientation tests
- rejection of any non-hold left-arm wire action
- `--max-executed-actions 1` validating 32 actions but scheduling only one
- default execution remaining equal to `steps_per_inference`

Run the focused DECO, dry-run, tactile, and action-scheduling test files before
the hardware-free paired dry-run. Existing unrelated dirty-worktree test
expectations must not be silently rewritten as part of this feature.

## Acceptance Criteria

The design is implemented when all of the following are true:

1. Existing Stage 1 and Bread launchers retain their original configs and model
   call behavior.
2. The new Stage 2 `--check` validates the exact latest checkpoint and rejects
   a Stage 1 artifact.
3. Hardware-free client/server dry-run completes one `vitac` exchange and exits
   without initializing hardware.
4. Observe-only processes one real six-stream observation and exits without
   sending `START` or an action.
5. The real checkpoint returns finite `[32, 10]` actions, which expand to a
   valid right-only `[32, 20]` wire chunk.
6. A server cap of one validates the whole chunk and schedules at most one
   action.
7. Focused client and server tests pass without overwriting pre-existing user
   worktree changes.
