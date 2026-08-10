# Tactile Orientation Alignment Design

## Goal

Use the deployed `observation.images.tactile_right_0` orientation as the canonical orientation for all four tactile streams. The user has confirmed that all training data uses this orientation; that confirmation is the compatibility requirement for the existing checkpoint and tactile embeddings.

## Current Behavior

`vb3_robot_server/real_world/bimanual_umi_env.py` splits each camera frame into `left_tactile`, `visual`, and `right_tactile`. It rotates only `left_tactile` by 180 degrees. Consequently, `tactile_left_0` and `tactile_left_1` are inverted relative to `tactile_right_0` and `tactile_right_1`.

The server-to-client key mapping and the SmolVLA observation saver preserve these images without changing their orientation.

## Selected Design

Remove the server-side 180-degree rotation of `left_tactile`. Both tactile panels will then use the raw orientation already used by `right_tactile`, for both camera indices.

The change will not:

- swap tactile keys;
- change the `0`/`1` wrist assignment;
- modify `rename_map`;
- add model-side image transforms;
- alter ordinary RGB camera orientation.

## Data Flow

1. Each raw three-panel camera frame is split into left tactile, RGB, and right tactile panels.
2. Both tactile panels retain their raw orientation.
3. Existing resize and BGR-to-RGB conversion run unchanged.
4. Existing observation keys carry the aligned images to recording and inference.

## Verification

Add a regression test in `vb3_robot_server` using asymmetric, coordinate-marked synthetic tactile panels. The test must fail with the existing left-only rotation and pass only when both panels retain the canonical `right_tactile` orientation. It must cover camera indices 0 and 1 and the four final observation keys, while asserting that key assignment, both `right_tactile` images, and ordinary RGB images remain unchanged.

Then run the focused regression test and the related camera tests. Capture a fresh observation without enabling robot motion and confirm all four tactile streams against a known asymmetric target or their raw three-panel source coordinates. Run no-motion VT-only and FRS-enabled inference smoke checks for finite, correctly shaped actions before enabling robot motion.

## Safety and Rollback

No robot control, state, or action code changes. The implementation changes observation pixels at the acquisition boundary, which can change model actions. Record the orientation convention with newly collected data and embedding caches so artifacts from different conventions are not mixed. Rollback consists of reverting the orientation change and using only checkpoints, data, and caches that match the restored convention. The no-motion verification gates above must pass before enabling robot motion.
