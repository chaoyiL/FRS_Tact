# RDP AT BF16 and Deployment-Phase Repair Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make small-angle AT rotation training correct under BF16, qualify checkpoints on the slow16 phases used by the robot, and deploy only explicitly qualified AT/LDP artifacts.

**Architecture:** Keep the current GRU and control schedule. Add an FP32/FP64 rotation-loss island, a deployment-window metric wrapper over existing physical metrics, stable release checkpoint metadata, and a deployment-time qualification check.

**Tech Stack:** Python 3.12, PyTorch, OmegaConf/Hydra, NumPy, pytest, Bash.

## Global Constraints

- `slow_update_interval` remains exactly `16`; runtime decoding is unchanged.
- The deployed window is indices `3..18`, derived from `n_obs_steps * ratio - 1` and count `16`.
- BF16/FP16 rotation loss computes in FP32; FP64 remains FP64.
- Existing 29-phase metrics remain; release metrics use `val_deploy_*`.
- Idle/no-op p95 limits are `0.05 mm` translation and `0.03 degrees` rotation.
- Active degradation is at most `5 percent`; micro-motion recall is at least `95 percent`.
- `latest.ckpt` is recovery-only; only a passing, improving model becomes `deployable.ckpt`.
- Missing baselines or release evidence fail closed for release/deployment but do not prevent recovery training.
- Do not change GRU hidden semantics, skip phases, change inference steps, or refactor unrelated packages.
- Preserve the user's `deploy_deco/configs/deploy_deco_right.yaml` change.

---

### Task 1: Preserve small-angle rotation loss under BF16

**Files:**
- Modify: `train_RDP/reactive_diffusion_policy/model/vae/physical_action_loss.py`
- Modify: `deploy_RDP/reactive_diffusion_policy/model/vae/physical_action_loss.py`
- Test: `train_RDP/tests/test_pick_tube_at_physical_loss.py`

**Interfaces:**
- Keep `compute_physical_action_loss(...) -> dict[str, torch.Tensor]` unchanged.
- Add private `_rotation_compute_dtype(*values) -> torch.dtype`.

- [ ] **Step 1: Write failing BF16 value and gradient tests**

Add a 10D Z-rotation helper. Parameterize `0.05`, `0.5`, and `1.2` degrees under `torch.autocast("cpu", dtype=torch.bfloat16)`; require nonzero FP32 output matching the autocast-disabled reference. Backpropagate `1.2` degrees and require finite, nonzero gradients in action dimensions `3:9`. Add a CUDA BF16 version guarded by `torch.cuda.is_available()` and `torch.cuda.is_bf16_supported()`.

- [ ] **Step 2: Verify RED**

```bash
PYTHONPATH=train_RDP .venv/bin/pytest -q train_RDP/tests/test_pick_tube_at_physical_loss.py -k bf16
```

Expected: small-angle values collapse to zero and/or gradients are non-finite.

- [ ] **Step 3: Implement the precision island**

Use this dtype rule before projecting rotation 6D:

```python
def _rotation_compute_dtype(*values):
    dtype = values[0].dtype
    for value in values[1:]:
        dtype = torch.promote_types(dtype, value.dtype)
    return torch.float32 if dtype in (torch.float16, torch.bfloat16) else dtype
```

Inside `torch.autocast(device_type=prediction.device.type, enabled=False)`, cast target/prediction rotation 6D to that dtype and compute Gram-Schmidt projection, relative matrix multiplication, trace/acos, rotation scaled Huber, idle rotation scaled Huber, degeneracy, and raw rot6 auxiliary loss. Construct the identity matrix in the same dtype. Do not cast the resulting scalars back to BF16.

- [ ] **Step 4: Verify GREEN and copy parity**

```bash
PYTHONPATH=train_RDP .venv/bin/pytest -q train_RDP/tests/test_pick_tube_at_physical_loss.py
cmp train_RDP/reactive_diffusion_policy/model/vae/physical_action_loss.py deploy_RDP/reactive_diffusion_policy/model/vae/physical_action_loss.py
```

- [ ] **Step 5: Commit**

```bash
git add train_RDP/reactive_diffusion_policy/model/vae/physical_action_loss.py deploy_RDP/reactive_diffusion_policy/model/vae/physical_action_loss.py train_RDP/tests/test_pick_tube_at_physical_loss.py
git commit -m "fix(rdp): compute AT rotation loss outside BF16"
```

### Task 2: Add deployment-window and canonical no-op metrics

**Files:**
- Modify: `train_RDP/reactive_diffusion_policy/common/pick_tube_validation.py`
- Modify: `deploy_RDP/reactive_diffusion_policy/common/pick_tube_validation.py`
- Modify: AT/LDP train configs under both `train_RDP/reactive_diffusion_policy/config/` and `deploy_RDP/reactive_diffusion_policy/config/`
- Test: `train_RDP/tests/test_pick_tube_validation_v2.py`

**Interfaces:**
- Add `compute_deployment_window_metrics(target, prediction, idle_mask, *, phase_start, phase_count, valid_mask=None, state_action_profile=None) -> dict[str, float]`.
- Add `build_canonical_noop_actions(actions: torch.Tensor) -> torch.Tensor`.
- Add `validation.deployment_slow_update_interval: 16`.

- [ ] **Step 1: Write failing metric tests**

For a 32-step prediction, put a one-degree error only in indices `3:19` and assert `val_deploy_idle_rotation_step_p95_deg == 1`. Then move a ten-degree error to `19:32` and assert the deployment metric is zero. Test invalid start/count/bounds. For no-op construction, assert translation is zero, rotation equals `[1,0,0,0,1,0]`, and gripper targets are unchanged for 10D and 20D actions.

- [ ] **Step 2: Verify RED**

```bash
PYTHONPATH=train_RDP .venv/bin/pytest -q train_RDP/tests/test_pick_tube_validation_v2.py -k 'deployment_window or canonical_noop'
```

Expected: both helpers are absent.

- [ ] **Step 3: Implement helpers**

Validate `phase_start >= 0`, `phase_count >= 1`, and `phase_start + phase_count <= T`. Slice all inputs to that exact window, call `compute_idle_rollout_metrics(..., horizon=phase_count)`, rename leading `val_` to `val_deploy_`, and rename `_29_` to `_window_`. The no-op builder accepts only contiguous 10D/20D action layouts and returns a clone with pose replaced.

- [ ] **Step 4: Add config and composition assertions**

Add `deployment_slow_update_interval: 16` to base AT/LDP validation blocks and require the composed single-right configs to resolve to 16.

- [ ] **Step 5: Verify and commit**

```bash
PYTHONPATH=train_RDP .venv/bin/pytest -q train_RDP/tests/test_pick_tube_validation_v2.py
cmp train_RDP/reactive_diffusion_policy/common/pick_tube_validation.py deploy_RDP/reactive_diffusion_policy/common/pick_tube_validation.py
git add train_RDP/reactive_diffusion_policy/common/pick_tube_validation.py deploy_RDP/reactive_diffusion_policy/common/pick_tube_validation.py train_RDP/reactive_diffusion_policy/config deploy_RDP/reactive_diffusion_policy/config train_RDP/tests/test_pick_tube_validation_v2.py
git commit -m "feat(rdp): validate the deployed AT phase window"
```

### Task 3: Produce stable deployable AT/LDP checkpoints

**Files:**
- Modify: AT and diffusion workspaces under both `train_RDP/reactive_diffusion_policy/workspace/` and `deploy_RDP/reactive_diffusion_policy/workspace/`
- Modify: `train_RDP/scripts/train_pick_tube_single_right_gpu.sh`
- Test: `train_RDP/tests/test_workspace_resume.py`
- Test: `train_RDP/tests/test_pick_tube_validation_v2.py`

**Interfaces:**
- Add `should_update_deployable_checkpoint(passed, score, best_score) -> bool`.
- Persist `best_deploy_idle_score`, initialized to positive infinity.
- Store `cfg.release_validation = {passed, deployment_slow_update_interval, score, epoch, metrics}`.
- Write `checkpoints/deployable.ckpt` only on a passing score improvement.

- [ ] **Step 1: Write failing release-selection tests**

Assert the helper accepts `(True, 1.0, inf)` and `(True, 0.9, 1.0)`, but rejects failed qualification, equal/worse scores, NaN, and infinity. Verify workspace checkpoint round-trip preserves `best_deploy_idle_score`. Verify the launcher resolves AT from `checkpoints/deployable.ckpt`, never latest.

- [ ] **Step 2: Verify RED**

```bash
PYTHONPATH=train_RDP .venv/bin/pytest -q train_RDP/tests/test_workspace_resume.py train_RDP/tests/test_pick_tube_validation_v2.py -k 'deployable or release_candidate'
```

- [ ] **Step 3: Wire AT/LDP deployment metrics**

Derive `phase_start = cfg.n_obs_steps * cfg.dataset_obs_temporal_downsample_ratio - 1` and `phase_count = cfg.validation.deployment_slow_update_interval`. Both workspaces evaluate held-out predictions through `compute_deployment_window_metrics`. AT additionally builds a canonical no-op chunk, encodes it with deterministic posterior mode, decodes it with recorded tactile temporal condition, and requires no-op p95 limits. Feed deployment-window values into existing active degradation, micro-motion, and idle threshold logic.

- [ ] **Step 4: Persist qualification and stable release**

Add `best_deploy_idle_score` to workspace include keys and initialize it to infinity. Before saving, write only scalar/OmegaConf-safe release evidence to `cfg.release_validation`. Save latest unconditionally as recovery. If `should_update_deployable_checkpoint` returns true, update the best score and synchronously save `checkpoints/deployable.ckpt`. Gate any named top-k file on `val_deployable` and monitor `val_deploy_idle_score`.

- [ ] **Step 5: Update launcher**

Require `${AT_DIR}/checkpoints/deployable.ckpt` before LDP starts; error text must state latest is recovery-only. After LDP training, require and report `${LDP_DIR}/checkpoints/deployable.ckpt`.

- [ ] **Step 6: Verify and commit**

```bash
PYTHONPATH=train_RDP .venv/bin/pytest -q train_RDP/tests/test_workspace_resume.py train_RDP/tests/test_pick_tube_validation_v2.py
cmp train_RDP/reactive_diffusion_policy/workspace/train_at_workspace.py deploy_RDP/reactive_diffusion_policy/workspace/train_at_workspace.py
cmp train_RDP/reactive_diffusion_policy/workspace/train_diffusion_unet_image_workspace.py deploy_RDP/reactive_diffusion_policy/workspace/train_diffusion_unet_image_workspace.py
git add train_RDP/reactive_diffusion_policy/workspace deploy_RDP/reactive_diffusion_policy/workspace train_RDP/scripts/train_pick_tube_single_right_gpu.sh train_RDP/tests/test_workspace_resume.py train_RDP/tests/test_pick_tube_validation_v2.py
git commit -m "feat(rdp): separate recovery and deployable checkpoints"
```

### Task 4: Enforce release qualification during deployment

**Files:**
- Modify: `deploy_RDP/deploy_pick_tube_rdp.py`
- Modify: `deploy_RDP/configs/deploy_pick_tube_rdp_right.yaml`
- Test: `deploy_RDP/tests/test_pick_tube_rdp_deploy.py`
- Test: `deploy_RDP/tests/test_pick_tube_rdp_right_launcher.py`

**Interfaces:**
- Add `validate_release_qualification(ldp_cfg, at_cfg, *, slow_update_interval, ldp_checkpoint, at_checkpoint) -> None`.
- Add required `slow_update_interval: int` to `load_policy(...)`.

- [ ] **Step 1: Write failing qualification and slow16 tests**

Use real OmegaConf configs with release evidence. Require both roles to have strict `passed is True`, finite score, and interval 16. Missing evidence, false passed, invalid score, or interval mismatch raises `ValueError`. Run the existing fake runtime for 17 steps at slow16 and assert history lengths are `list(range(4, 20)) + [4]`.

- [ ] **Step 2: Verify RED**

```bash
PYTHONPATH=deploy_RDP:/home/typhon/FRS_Tact/deploy_RDP/.venv/lib/python3.12/site-packages .venv/bin/pytest -q deploy_RDP/tests/test_pick_tube_rdp_deploy.py -k 'release_qualification or sixteen'
```

- [ ] **Step 3: Implement fail-closed validation**

Validate release evidence after checkpoint shape checks and before Hydra workspace construction. Pass `control.slow_update_interval` from configuration into `load_policy`. Do not add a bypass flag.

- [ ] **Step 4: Update deployment config**

Change AT/LDP filenames from `latest.ckpt` to `deployable.ckpt` and set `artifact_verification: strict`.

- [ ] **Step 5: Verify and commit**

```bash
PYTHONPATH=deploy_RDP:/home/typhon/FRS_Tact/deploy_RDP/.venv/lib/python3.12/site-packages .venv/bin/pytest -q deploy_RDP/tests/test_pick_tube_rdp_deploy.py deploy_RDP/tests/test_pick_tube_rdp_right_launcher.py deploy_RDP/tests/test_rdp_right_arm_adapter.py
git add deploy_RDP/deploy_pick_tube_rdp.py deploy_RDP/configs/deploy_pick_tube_rdp_right.yaml deploy_RDP/tests/test_pick_tube_rdp_deploy.py deploy_RDP/tests/test_pick_tube_rdp_right_launcher.py
git commit -m "fix(rdp): reject unqualified robot checkpoints"
```

### Task 5: Focused integration verification

**Files:**
- Verify only; modify production files only for a directly related regression.

**Interfaces:**
- Consume Tasks 1-4 and produce fresh test/review evidence.

- [ ] **Step 1: Run affected training tests**

```bash
PYTHONPATH=train_RDP:/home/typhon/FRS_Tact/deploy_RDP/.venv/lib/python3.12/site-packages .venv/bin/pytest -q train_RDP/tests/test_pick_tube_at_physical_loss.py train_RDP/tests/test_pick_tube_validation_v2.py train_RDP/tests/test_workspace_resume.py
```

- [ ] **Step 2: Run affected deployment tests**

```bash
PYTHONPATH=deploy_RDP:/home/typhon/FRS_Tact/deploy_RDP/.venv/lib/python3.12/site-packages .venv/bin/pytest -q deploy_RDP/tests/test_pick_tube_rdp_deploy.py deploy_RDP/tests/test_pick_tube_rdp_right_launcher.py deploy_RDP/tests/test_rdp_right_arm_adapter.py
```

- [ ] **Step 3: Verify parity and scope**

```bash
cmp train_RDP/reactive_diffusion_policy/model/vae/physical_action_loss.py deploy_RDP/reactive_diffusion_policy/model/vae/physical_action_loss.py
cmp train_RDP/reactive_diffusion_policy/common/pick_tube_validation.py deploy_RDP/reactive_diffusion_policy/common/pick_tube_validation.py
git diff --check
git status --short
```

The pre-existing `deploy_deco/configs/deploy_deco_right.yaml` modification must remain untouched.

- [ ] **Step 4: Request one focused final review**

Review only the approved design, TDD evidence, autocast/dtype correctness, exact phase-window alignment, checkpoint qualification, and fail-closed deployment. Fix Critical/Important findings and rerun their covering tests once.
