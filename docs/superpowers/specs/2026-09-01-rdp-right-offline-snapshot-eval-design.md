# RDP Right-Hand Offline Snapshot Evaluation Design

## Goal

Run the `rdp_0831` single-right RDP checkpoint against the saved observations in
`/home/typhon/vb3_robot_server/eval_obs_data/eval_obs_20260901_143744` without
opening a robot connection, then produce compact artifacts that show whether the
predicted immediate motions are finite, smooth, and physically plausible.

## Scope

This is a deliberately minimal evaluator for one saved observation run:

- One new evaluator script and one focused test file.
- One configured single-right checkpoint bundle.
- Thirteen independent saved snapshots.
- No WebSocket, robot bridge, controller, or action transmission.
- No resume system, multi-episode abstraction, ground-truth accuracy claim, or
  changes to the existing untracked Pi0.5/SmolVLA evaluators.

## Input Contract

The evaluator sorts `step_######` directories numerically. Every directory must
contain two RGB JPEGs, four tactile JPEGs, bimanual end-effector position and
axis-angle JSON files, two gripper-width JSON files, and a timestamp JSON file.
JPEGs are decoded as RGB.

The first snapshot supplies the left and right episode-start poses. Each saved
pose is converted into the same finite 20D relative-start bridge state used by
the production RDP server:

1. left relative pose and left gripper;
2. right relative pose and right gripper;
3. left pose relative to the current right pose.

## Inference Semantics

The evaluator reuses `load_policy`, `PickTubeRDPRuntime`, and
`wire_action_for_profile` from the production deployment code. It resolves the
`single-right-arm-7x10` profile and loads the LDP, AT, tactile PCA, and tactile
encoder paths from `deploy_pick_tube_rdp_right.yaml`.

The saved frames are approximately 1.34 seconds apart, so they are not treated
as adjacent control ticks. Before every snapshot prediction, the runtime is
reset. Its existing first-frame padding supplies the required observation
history. A fixed seed is restored for each snapshot so differences reflect the
observation rather than diffusion RNG order.

Each prediction produces a 10D right-arm policy action and a 20D bridge action.
The bridge action contains a canonical left-arm relative no-op and holds the
recorded left gripper width. No action is executed.

## Outputs

The default output directory is
`outputs/rdp_right_offline/eval_obs_20260901_143744/`. The evaluator writes:

- `predictions.npz`: states, 10D policy actions, 20D wire actions, right poses,
  step IDs, timestamps, and inference latency;
- `trajectory.csv`: one row per snapshot with translation, rotation angle,
  gripper command, recorded right gripper, and latency;
- `summary.json`: finite/shape status plus min/max/mean motion and latency
  statistics;
- `action_overview.png`: translation components/norm, rotation angle, gripper,
  and latency over saved snapshots;
- `right_snapshot_responses.png`: recorded right-hand XYZ path with each
  predicted one-step translation drawn from its corresponding snapshot pose.

The report calls these counterfactual snapshot responses, not a closed-loop
rollout or task-success estimate. The directory contains no reference actions,
so accuracy metrics are out of scope.

## Errors and Resource Behavior

The command fails before inference on missing files, malformed/non-finite state,
wrong checkpoint profile, invalid action shape, or unavailable requested CUDA
device. Output files are written only beneath the selected output directory.
The 3.8 GB LDP checkpoint is loaded once and reused for all snapshots.

## Testing

Focused tests cover numeric step sorting, RGB/key mapping, exact 20D state
reconstruction, runtime reset per snapshot, output array contracts, and report
generation using a fake runtime. The real run then loads the actual checkpoint
on `cuda:0` and processes all 13 snapshots. Existing RDP deployment regression
tests remain unchanged and are rerun after implementation.
