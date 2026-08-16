# FRS Inactive-Arm Protection Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Preserve VLA XYZ for either inactive arm when every robot-space VLA translation component is below a configurable threshold.

**Architecture:** Parse an optional meter-valued threshold into `FRSConfig`, validate that enabled protection is used only with the 20D bimanual contract, and apply an isolated selection helper after temporal ensembling but before unnormalization. The helper uses robot-space VLA only for activity detection and copies normalized VLA XYZ into the selected normalized action so public traces remain consistent.

**Tech Stack:** Python, NumPy, JAX, pytest, YAML.

## Global Constraints

- The configured threshold is `0.00025` meters by default in deployment YAML and `null` disables the feature.
- Activity uses `max(abs(x), abs(y), abs(z)) < threshold` independently for action slices `0:3` and `10:13`.
- Protection replaces XYZ only; rotation, gripper, the other arm, raw decoded output, and diagnostics remain unchanged.
- Existing user modifications in the dirty worktree must be preserved.

---

### Task 1: Configuration and selection behavior

**Files:**
- Modify: `tests/jax/test_frs_deployment.py`
- Modify: `deploy_smolvla/frs_runtime.py`
- Modify: `deploy_smolvla/configs/deploy_frs.yaml`

**Interfaces:**
- Consumes: cached `_action_vla_normalized` and `_action_vla` chunks on `FRSSteeringPolicy`.
- Produces: `FRSConfig.inactive_arm_xyz_threshold_m: float | None` and protected `FRSSteerResult.selected_normalized` / `selected_action`.

- [ ] **Step 1: Write failing configuration tests**

Add tests proving that absent/null config disables the feature, `0.00025` is stored, and bool/string/non-positive/non-finite values raise a `ValueError` mentioning `inactive_arm_xyz_threshold_m`.

- [ ] **Step 2: Run configuration tests and verify RED**

Run: `uv run --no-sync pytest tests/jax/test_frs_deployment.py -k 'inactive_arm_xyz_threshold' -q`

Expected: failures because the config field and validation do not exist.

- [ ] **Step 3: Implement minimal configuration parsing**

Add a nullable finite-positive parser, store it in `FRSConfig`, invoke it from both runtime parsing and static validation, and set this in YAML:

```yaml
inactive_arm_xyz_threshold_m: 0.00025
```

- [ ] **Step 4: Run configuration tests and verify GREEN**

Run: `uv run --no-sync pytest tests/jax/test_frs_deployment.py -k 'inactive_arm_xyz_threshold' -q`

Expected: all selected tests pass.

- [ ] **Step 5: Write failing steering tests**

Add 20D per-action tests using a non-identity unnormalizer. Prove left-only, right-only, and both-arm fallback; strict equality does not trigger; a negative XYZ inside the threshold triggers; rotations and grippers retain FRS values; and `decoded_normalized` remains unmodified.

- [ ] **Step 6: Run steering tests and verify RED**

Run: `uv run --no-sync pytest tests/jax/test_frs_deployment.py -k 'inactive_arm_xyz' -q`

Expected: selection assertions fail because FRS XYZ is still sent.

- [ ] **Step 7: Implement minimal post-ensemble selection protection**

Add a helper that receives selected normalized action plus the current VLA normalized and robot-space actions. For each `(0, 3)` and `(10, 13)` slice, copy normalized VLA XYZ only when the robot-space max-absolute component is strictly below the threshold. Call it after temporal ensemble and before `_immutable_public_array` / unnormalization. Reject enabled protection with non-20D action contracts during initialization.

- [ ] **Step 8: Run steering tests and verify GREEN**

Run: `uv run --no-sync pytest tests/jax/test_frs_deployment.py -k 'inactive_arm_xyz' -q`

Expected: all selected tests pass.

- [ ] **Step 9: Run focused regression verification**

Run: `uv run --no-sync pytest tests/jax/test_frs_deployment.py -q`

Expected: the complete FRS deployment test file passes.
