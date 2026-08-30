# DECO Stage 2 Tactile Deployment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a separate, real-robot deployment path for the tactile DECO Stage 2 checkpoint while keeping all existing Stage 1, right-arm, Press, and Bread entrypoints behaviorally unchanged.

**Architecture:** Extend the shared DECO client according to artifact metadata, but add dedicated Stage 2 YAML and shell entrypoints. Keep the robot server bimanual, publish two visual plus four tactile images, project the policy state to the right arm on the client, expand the 10D policy output to the existing 20D wire action, and select Stage 2 server behavior through a dedicated server config module and launcher. Keep the existing DECO client-owned config handshake; the generic server `--config` path is not part of the DECO startup flow.

**Tech Stack:** Python 3, PyTorch/TorchScript, NumPy, PyYAML, Pillow, msgpack/WebSocket robot bridge, Click, pytest, Bash.

## Global Constraints

- Work in two repositories:
  - client: `/home/typhon/FRS_Tact`
  - server: `/home/typhon/vb3_robot_server`
- Preserve the existing dirty worktrees. Never reset, checkout, stash, or overwrite user changes.
- Before each repository change, run `git status --short` and an explicit `git diff -- path/to/file` command for every file that will be edited.
- The client YAML files `deploy_deco/configs/deploy_deco.yaml`, `deploy_deco/configs/deploy_deco_bread_phase.yaml`, and `deploy_deco/configs/deploy_deco_right.yaml` already contain user changes. Do not edit or stage them.
- The server files `configs/deco_server_config.py`, `deploy_scripts/bimanual_smolvla_online.py`, `deploy_scripts/vbvla_safety.py`, and several tests already contain user changes. Reuse their current behavior and apply only narrow Stage 2 additions.
- Use `apply_patch` for source edits. Run formatting tools only on files changed by this plan.
- Do not initialize robot hardware during automated implementation or verification. The observe-only and physical smoke commands are operator-run rollout steps.
- Do not alter the Stage 2 TorchScript or sidecar. The required SHA256 remains `ebd606ed8b4932e14fe0ec70718922a6829c1a9f4ee72ab42e694a0445fbc87d`.
- Do not add resize, crop, flip, ImageNet normalization, or tactile normalization in the client. The TorchScript owns model preprocessing.
- Do not change server resize scheme A, gripper hysteresis, action semantics, or the legacy bimanual chunk protocol.
- Do not commit an entire pre-dirty shared server file. Commit clean/new files separately; leave additions to overlapping dirty files unstaged unless their Stage 2-only hunks can be staged without including prior user changes.
- Every planned commit uses `git commit --only -- path...` so unrelated files that the user may already have staged are excluded. Inspect `git diff --cached --name-only` before and after each commit and never unstage those unrelated paths.

---

## Task 1: Accept and strictly validate the Stage 2 artifact contract

**Files:**

- Modify: `/home/typhon/FRS_Tact/deploy_deco/artifact.py`
- Modify: `/home/typhon/FRS_Tact/deploy_deco/tests/test_artifact.py`

- [ ] **Step 1: Record the client baseline for the two files**

Run:

```bash
cd /home/typhon/FRS_Tact
git status --short
git diff -- deploy_deco/artifact.py deploy_deco/tests/test_artifact.py
```

Expected: neither target file has a pre-existing diff. If either is dirty, preserve that diff and do not stage it wholesale later.

- [ ] **Step 2: Add failing Stage 2 metadata tests**

Add a `stage2_metadata()` fixture helper derived from `right_metadata()` with these exact changes:

```python
contract["format"] = "sudo-upstream-deco-stage2-torchscript-v1"
contract["input"]["tactile_images"] = [1, 4, 3, 224, 224]
contract["input"]["tactile_images_dtype"] = "float32"
contract["input"]["tactile_images_range"] = [0.0, 1.0]
contract["tactile_field_order"] = [
    "observation.images.tactile_left_0",
    "observation.images.tactile_right_0",
    "observation.images.tactile_left_1",
    "observation.images.tactile_right_1",
]
contract["input"]["stream_order"] = (
    contract["camera_names"] + contract["tactile_field_order"]
)
```

Add tests that prove:

```python
def test_stage2_contract_is_accepted_and_detected_as_tactile():
    validated = validate_metadata(stage2_metadata(b""))
    assert artifact_uses_tactile(validated) is True


def test_stage1_contract_is_not_detected_as_tactile():
    assert artifact_uses_tactile(right_metadata(b"")) is False


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda value: value["input"].update(tactile_images=[1, 3, 3, 224, 224]), "tactile_images"),
        (lambda value: value.update(tactile_field_order=list(reversed(value["tactile_field_order"]))), "tactile field order"),
        (lambda value: value["input"].update(stream_order=value["camera_names"]), "stream_order"),
        (lambda value: value["input"].update(tactile_images_dtype="uint8"), "tactile_images_dtype"),
        (lambda value: value["input"].update(tactile_images_range=[-1.0, 1.0]), "tactile_images_range"),
    ],
)
def test_stage2_rejects_wrong_tactile_contract(mutation, message):
    contract = stage2_metadata(b"")
    mutation(contract)
    with pytest.raises(ValueError, match=message):
        validate_metadata(contract)
```

- [ ] **Step 3: Run the tests to verify the red state**

Run:

```bash
cd /home/typhon/FRS_Tact
uv run --project deploy_deco pytest -q deploy_deco/tests/test_artifact.py
```

Expected: collection fails because `artifact_uses_tactile` does not exist, or Stage 2 is rejected as an unsupported format.

- [ ] **Step 4: Implement format-aware validation**

In `artifact.py`, replace the single-format constant with:

```python
STAGE1_EXPORT_FORMAT = "sudo-upstream-deco-stage1-torchscript-v1"
STAGE2_EXPORT_FORMAT = "sudo-upstream-deco-stage2-torchscript-v1"
EXPORT_FORMAT = STAGE1_EXPORT_FORMAT  # compatibility for existing imports
SUPPORTED_EXPORT_FORMATS = frozenset({STAGE1_EXPORT_FORMAT, STAGE2_EXPORT_FORMAT})
TACTILE_FIELD_ORDER = (
    "observation.images.tactile_left_0",
    "observation.images.tactile_right_0",
    "observation.images.tactile_left_1",
    "observation.images.tactile_right_1",
)


def artifact_uses_tactile(metadata: Mapping[str, Any]) -> bool:
    return metadata.get("format") == STAGE2_EXPORT_FORMAT
```

Change the format check to membership in `SUPPORTED_EXPORT_FORMATS`. Keep all existing Stage 1 profile, shape, action, normalization, stochastic, and sidecar/hash checks intact. For Stage 2 only, require:

```python
if artifact_uses_tactile(metadata):
    tactile = _shape(metadata, "input", "tactile_images")
    if tactile != (1, 4, 3, 224, 224):
        raise ValueError("DECO Stage 2 tactile_images must have shape [1,4,3,224,224]")
    if input_contract.get("tactile_images_dtype") != "float32":
        raise ValueError("DECO Stage 2 tactile_images_dtype must be float32")
    if input_contract.get("tactile_images_range") != [0.0, 1.0]:
        raise ValueError("DECO Stage 2 tactile_images_range must be [0.0, 1.0]")
    if metadata.get("tactile_field_order") != list(TACTILE_FIELD_ORDER):
        raise ValueError(f"DECO Stage 2 tactile field order must be {list(TACTILE_FIELD_ORDER)}")
    expected_stream_order = expected_cameras + list(TACTILE_FIELD_ORDER)
    if input_contract.get("stream_order") != expected_stream_order:
        raise ValueError(f"DECO Stage 2 stream_order must be {expected_stream_order}")
```

Also require the Stage 2 visual shape `(1, 2, 3, 224, 224)`, profile `single-right-arm-7x10`, state `(1, 7)`, and output `(1, 32, 10)`. Do not tighten the legacy Stage 1 spatial-size rule.

- [ ] **Step 5: Verify artifact tests and the real sidecar**

Run:

```bash
cd /home/typhon/FRS_Tact
uv run --project deploy_deco pytest -q deploy_deco/tests/test_artifact.py
uv run --project deploy_deco python -c "from deploy_deco.artifact import load_sidecar; m=load_sidecar('checkpoints/model/deco_0830/insert_stage2/deco_stage2_latest.ts'); assert m['format']=='sudo-upstream-deco-stage2-torchscript-v1'; print(m['torchscript_sha256'])"
```

Expected: all tests pass and the printed hash is `ebd606ed8b4932e14fe0ec70718922a6829c1a9f4ee72ab42e694a0445fbc87d`.

- [ ] **Step 6: Commit only the clean client artifact files**

```bash
cd /home/typhon/FRS_Tact
git add deploy_deco/artifact.py deploy_deco/tests/test_artifact.py
git diff --cached --check -- deploy_deco/artifact.py deploy_deco/tests/test_artifact.py
git commit --only -m "feat(deco): validate stage2 tactile artifacts" -- \
  deploy_deco/artifact.py deploy_deco/tests/test_artifact.py
```

---

## Task 2: Build the four tactile inputs and dispatch the Stage 2 TorchScript call

**Files:**

- Modify: `/home/typhon/FRS_Tact/deploy_deco/policy.py`
- Modify: `/home/typhon/FRS_Tact/deploy_deco/tests/test_policy.py`

- [ ] **Step 1: Add failing policy tests for exact tensor order and call signature**

Extend the test policy factory so it sets `uses_tactile`, `tactile_keys`, `visual_hw`, and `tactile_hw`. Add a Stage 2 observation containing 224-by-224 visual images and four 224-by-224 constant tactile images whose values are `10`, `20`, `30`, and `40`. The test model must capture its inputs and return `(1, 2, 3)`.

Assert:

```python
action = policy.predict(stage2_observation, seed=1)
assert action.shape == (2, 3)
assert calls == [3]
assert captured[0].shape == (1, 2, 3, 224, 224)
assert captured[1].shape == (1, 4, 3, 224, 224)
np.testing.assert_allclose(
    captured[1][0, :, 0, 0, 0].cpu().numpy(),
    np.array([10, 20, 30, 40], dtype=np.float32) / 255.0,
)
```

Also add tests that a missing tactile key is rejected, that 223-by-224 or 256-by-256 Stage 2 visual/tactile inputs are rejected before the model call, and that a Stage 2 policy cannot also declare `phase_count`. Keep one regular Stage 1 test with 4-by-5 images to prove its existing flexible shape behavior remains unchanged.

- [ ] **Step 2: Verify the tests fail for the current two-input implementation**

```bash
cd /home/typhon/FRS_Tact
uv run --project deploy_deco pytest -q deploy_deco/tests/test_policy.py
```

Expected: Stage 2 tests fail because `prepare_inputs()` returns only visual images and state.

- [ ] **Step 3: Implement tactile-aware input preparation**

Import `TACTILE_FIELD_ORDER` and `artifact_uses_tactile`. In `DECOPolicy.__init__`, set:

```python
self.uses_tactile = artifact_uses_tactile(self.metadata)
self.tactile_keys = TACTILE_FIELD_ORDER if self.uses_tactile else ()
self.visual_hw = tuple(self.metadata["input"]["images"][3:5])
self.tactile_hw = (
    tuple(self.metadata["input"]["tactile_images"][3:5])
    if self.uses_tactile
    else None
)
if self.uses_tactile and self.phase_count is not None:
    raise ValueError("DECO Stage 2 tactile artifacts cannot also be phase-conditioned")
```

Refactor stacking into a private helper that validates equal HWC shapes and returns contiguous `[1,N,3,H,W]` float32 data. For Stage 2, compare every visual HWC shape with `(*self.visual_hw, 3)` and every tactile HWC shape with `(*self.tactile_hw, 3)` before tensor conversion; raise `ValueError` on any mismatch. Keep Stage 1's current rule that its two visual shapes only need to match each other. `prepare_inputs()` must return:

```python
if self.uses_tactile:
    return visual_tensor, tactile_tensor, state_tensor
return visual_tensor, state_tensor
```

In `predict()`, dispatch explicitly:

```python
inputs = self.prepare_inputs(observation)
with torch.inference_mode():
    if self.uses_tactile:
        images, tactile_images, state = inputs
        output = self.model(images, tactile_images, state)
    elif self.phase_count is not None:
        images, state = inputs
        phase = torch.tensor([phase_id], dtype=torch.long, device=self.device)
        output = self.model(images, state, phase)
    else:
        images, state = inputs
        output = self.model(images, state)
```

Keep seed handling and output validation unchanged.

- [ ] **Step 4: Run policy and Stage 1 Bread regressions**

```bash
cd /home/typhon/FRS_Tact
uv run --project deploy_deco pytest -q \
  deploy_deco/tests/test_policy.py \
  deploy_deco/tests/test_bread_phase_client.py \
  deploy_deco/tests/test_bread_phase_controller.py
```

Expected: all tests pass; regular Stage 1 still records two inputs and Bread still records three inputs with phase last.

- [ ] **Step 5: Commit the policy boundary**

```bash
cd /home/typhon/FRS_Tact
git add deploy_deco/policy.py deploy_deco/tests/test_policy.py
git diff --cached --check -- deploy_deco/policy.py deploy_deco/tests/test_policy.py
git commit --only -m "feat(deco): feed tactile streams to stage2 policy" -- \
  deploy_deco/policy.py deploy_deco/tests/test_policy.py
```

---

## Task 3: Add the separate Stage 2 client configuration and launcher

**Files:**

- Modify: `/home/typhon/FRS_Tact/deploy_deco/config.py`
- Modify: `/home/typhon/FRS_Tact/deploy_deco/tests/test_config.py`
- Create: `/home/typhon/FRS_Tact/deploy_deco/configs/deploy_deco_stage2_right.yaml`
- Create: `/home/typhon/FRS_Tact/deploy_deco/scripts/start_deco_stage2_right.sh`

- [ ] **Step 1: Add failing config and launcher tests**

Define `STAGE2_CONFIG` in `test_config.py`. Add tests that load the checked-in Stage 2 YAML and assert:

```python
metadata = load_sidecar(config["checkpoint"])
validate_artifact_contract(config, metadata)
server = make_server_config(config)
assert server["data_type"] == "vitac"
assert server["observation_profile"] == "deco_vitac_224"
assert server["single_arm_mode"] is False
assert server["action_horizon"] == 32
assert server["steps_per_inference"] == 32
```

Add two cross-pairing tests: Stage 2 metadata with `vision/deco_vision_224` must fail, and Stage 1 metadata with `vitac/deco_vitac_224` must fail. Add a launcher delegation test identical in structure to the right-arm launcher test but expecting `deploy_deco_stage2_right.yaml`.

- [ ] **Step 2: Run the config tests and confirm they fail**

```bash
cd /home/typhon/FRS_Tact
uv run --project deploy_deco pytest -q deploy_deco/tests/test_config.py
```

Expected: failure because the Stage 2 files do not exist and `vitac` is currently rejected.

- [ ] **Step 3: Generalize the YAML observation pairing without weakening it**

In `config.py`, define:

```python
DECO_VISION_PROFILE = "deco_vision_224"
DECO_VITAC_PROFILE = "deco_vitac_224"
DECO_OBSERVATION_PROFILE = DECO_VISION_PROFILE  # compatibility
_OBSERVATION_CONTRACTS = {
    "vision": DECO_VISION_PROFILE,
    "vitac": DECO_VITAC_PROFILE,
}
```

`validate_config()` must accept only these two exact `data_type/profile` pairs. `validate_artifact_contract()` must compare `artifact_uses_tactile(metadata)` against `data_type == "vitac"` and reject cross-pairings before deployment. `make_server_config()` must use the YAML values:

```python
"data_type": str(observation["data_type"]),
"observation_profile": str(observation["observation_profile"]),
```

Keep bridge `single_arm_mode=False` and `no_state_obs_mode=False` hard-coded.

- [ ] **Step 4: Create the dedicated Stage 2 YAML**

Create `deploy_deco_stage2_right.yaml` with the connection values from the current right-arm YAML and these model-specific values:

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

The `connection` section must contain address `127.0.0.1`, port `26421`, `add_port: true`, the current timeouts, `token_env: VB_ROBOT_TOKEN`, and `require_token: true`.

- [ ] **Step 5: Create the separate launcher**

Create executable `start_deco_stage2_right.sh`:

```bash
#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PACKAGE_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
export CONFIG="${CONFIG:-${PACKAGE_ROOT}/configs/deploy_deco_stage2_right.yaml}"
exec bash "${SCRIPT_DIR}/start_deco.sh" "$@"
```

Then run `chmod +x deploy_deco/scripts/start_deco_stage2_right.sh`.

- [ ] **Step 6: Verify the checked-in config and unchanged launchers**

```bash
cd /home/typhon/FRS_Tact
uv run --project deploy_deco pytest -q deploy_deco/tests/test_config.py
bash deploy_deco/scripts/start_deco_stage2_right.sh --check
bash deploy_deco/scripts/start_deco_right.sh --check
```

Expected: all tests pass; both checks validate their own distinct artifact and neither connects to the server.

- [ ] **Step 7: Commit only the new YAML, launcher, and clean shared config files**

```bash
cd /home/typhon/FRS_Tact
git add deploy_deco/config.py deploy_deco/tests/test_config.py \
  deploy_deco/configs/deploy_deco_stage2_right.yaml \
  deploy_deco/scripts/start_deco_stage2_right.sh
git diff --cached --check -- deploy_deco/config.py deploy_deco/tests/test_config.py \
  deploy_deco/configs/deploy_deco_stage2_right.yaml \
  deploy_deco/scripts/start_deco_stage2_right.sh
git commit --only -m "feat(deco): add stage2 tactile deployment entrypoint" -- \
  deploy_deco/config.py deploy_deco/tests/test_config.py \
  deploy_deco/configs/deploy_deco_stage2_right.yaml \
  deploy_deco/scripts/start_deco_stage2_right.sh
```

Confirm the three pre-existing YAML modifications remain unstaged with `git status --short`.

---

## Task 4: Add bounded server-dry-run and real observe-only client modes

**Files:**

- Modify: `/home/typhon/FRS_Tact/deploy_deco/remote_client.py`
- Modify: `/home/typhon/FRS_Tact/deploy_deco/tests/test_remote_client.py`
- Modify: `/home/typhon/FRS_Tact/deploy_deco/tests/test_right_arm_adapter.py`
- Modify: `/home/typhon/FRS_Tact/deploy_deco/pyproject.toml`
- Modify: `/home/typhon/FRS_Tact/deploy_deco/uv.lock`

- [ ] **Step 1: Add failing protocol-mode tests**

Add a `server_dry_run=True` test using two observations: one warmup and one action input. Its exact terminal events must be:

```python
[
    ("observation", 0),
    ("state", "start"),
    ("observation", 1),
    ("action", 1),
    ("state", "stop"),
    "close",
]
```

Add a second test asserting `server_dry_run=True` with effective `max_iterations == 0` raises `ValueError` mentioning `max_iterations`; server dry-run is always bounded.

Keep `test_bounded_loop_waits_for_post_action_observation_before_stop` unchanged so the real bounded mode still consumes the post-action observation.

Add an observe-only test that asserts:

- one warmup observation is received;
- `START` is never sent;
- no action is sent;
- `STOP` and close still occur;
- the saver receives exactly the two visual keys and four tactile keys after right-arm projection;
- saved visual camera0 is black and tactile-left-0 is unchanged.

Add an adapter regression test:

```python
projected = project_right_observation(observation, black_camera0=True)
for key in TACTILE_FIELD_ORDER:
    assert projected[key] is observation[key]
```

- [ ] **Step 2: Confirm the new tests fail**

```bash
cd /home/typhon/FRS_Tact
uv run --project deploy_deco pytest -q \
  deploy_deco/tests/test_remote_client.py \
  deploy_deco/tests/test_right_arm_adapter.py
```

Expected: failure because `run()` and the parser do not expose the two modes and the bounded loop always receives a final observation.

- [ ] **Step 3: Add a deterministic observe-only bundle writer**

Add `"Pillow>=10,<13"` to the deployment project's runtime dependencies and refresh the lockfile:

```bash
cd /home/typhon/FRS_Tact
uv lock --project deploy_deco
```

In `remote_client.py`, add `save_observe_only_bundle(output_root, observation, policy, action) -> Path`. It must:

- create `observe_only_YYYYmmdd_HHMMSS` under the supplied root;
- write exactly six RGB PNGs named from `policy.image_keys + policy.tactile_keys`, replacing dots with underscores;
- preserve RGB by writing each array with `PIL.Image.fromarray(rgb_uint8, mode="RGB").save(path)`;
- accept source `uint8` or float `[0,1]` and write PNG as `uint8`;
- write `summary.json` containing key, shape, dtype, min, max for each image plus state shape/range and action shape/range;
- return the created directory.

Use `Path(__file__).resolve().parent / "outputs"` as the default output root. Do not save the original unprojected camera0.

- [ ] **Step 4: Extend `run()` and CLI mode selection**

Use this signature:

```python
def run(
    config_path: Path,
    max_iterations_override: int | None = None,
    *,
    server_dry_run: bool = False,
    observe_only: bool = False,
) -> None:
```

Reject both modes being true. Add mutually exclusive parser flags `--server-dry-run` and `--observe-only`.

After resolving `max_iterations`, enforce:

```python
if server_dry_run and max_iterations <= 0:
    raise ValueError("--server-dry-run requires a positive --max-iterations")
```

After receiving the warmup observation, project it once. Run `max(1, warmup_runs)` inference calls in observe-only so the summary always contains an action shape/range. Save the bundle, print its path, and return before the manual prompt and before `bridge.send_state("start")`. Cleanup must still send `STOP`.

At the end of a bounded action loop, keep the existing final `receive_observation()` only when `server_dry_run` is false:

```python
if max_iterations > 0 and not server_dry_run:
    bridge.receive_observation(timeout=observation_timeout)
```

Log the artifact mode and, for Stage 2, the exact `policy.tactile_keys` order after loading.

- [ ] **Step 5: Run client protocol tests**

```bash
cd /home/typhon/FRS_Tact
uv run --project deploy_deco pytest -q \
  deploy_deco/tests/test_remote_client.py \
  deploy_deco/tests/test_right_arm_adapter.py \
  deploy_deco/tests/test_bridge_client.py
```

Expected: all pass, including both the real bounded post-observation behavior and the dry-run immediate STOP behavior.

- [ ] **Step 6: Commit the client runtime modes**

```bash
cd /home/typhon/FRS_Tact
git add deploy_deco/remote_client.py \
  deploy_deco/tests/test_remote_client.py \
  deploy_deco/tests/test_right_arm_adapter.py \
  deploy_deco/pyproject.toml deploy_deco/uv.lock
git diff --cached --check -- deploy_deco/remote_client.py \
  deploy_deco/tests/test_remote_client.py \
  deploy_deco/tests/test_right_arm_adapter.py \
  deploy_deco/pyproject.toml deploy_deco/uv.lock
git commit --only -m "feat(deco): add bounded stage2 rollout modes" -- \
  deploy_deco/remote_client.py deploy_deco/tests/test_remote_client.py \
  deploy_deco/tests/test_right_arm_adapter.py \
  deploy_deco/pyproject.toml deploy_deco/uv.lock
```

---

## Task 5: Verify the real Stage 2 TorchScript offline and document the client commands

**Files:**

- Modify: `/home/typhon/FRS_Tact/deploy_deco/README.md`

- [ ] **Step 1: Run the complete DECO client test suite**

```bash
cd /home/typhon/FRS_Tact
uv run --project deploy_deco pytest -q deploy_deco/tests
```

Expected: all DECO tests pass. If an unrelated pre-existing config edit causes a failure, record it without changing that YAML and re-run the focused tests from Tasks 1–4.

- [ ] **Step 2: Load and infer the real Stage 2 artifact on CUDA**

Run a one-shot inference using exact shaped synthetic inputs:

```bash
cd /home/typhon/FRS_Tact
uv run --project deploy_deco python -c "import numpy as np; from deploy_deco.policy import DECOPolicy; p=DECOPolicy('checkpoints/model/deco_0830/insert_stage2/deco_stage2_latest.ts', device='cuda:0'); o={'observation.images.camera0':np.zeros((224,224,3),np.uint8),'observation.images.camera1':np.zeros((224,224,3),np.uint8),'observation.images.tactile_left_0':np.zeros((224,224,3),np.uint8),'observation.images.tactile_right_0':np.zeros((224,224,3),np.uint8),'observation.images.tactile_left_1':np.zeros((224,224,3),np.uint8),'observation.images.tactile_right_1':np.zeros((224,224,3),np.uint8),'observation.state':np.zeros(7,np.float32)}; a=p.predict(o,seed=0); assert a.shape==(32,10) and np.isfinite(a).all(); print(a.shape, float(a.min()), float(a.max()))"
```

Expected: the model loads on `cuda:0` and prints shape `(32, 10)` with finite bounds.

- [ ] **Step 3: Add a concise Stage 2 section to the client README**

Document only these commands and their meaning:

```bash
bash deploy_deco/scripts/start_deco_stage2_right.sh --check
bash deploy_deco/scripts/start_deco_stage2_right.sh --server-dry-run --max-iterations 1
bash deploy_deco/scripts/start_deco_stage2_right.sh --observe-only
bash deploy_deco/scripts/start_deco_stage2_right.sh --max-iterations 1
bash deploy_deco/scripts/start_deco_stage2_right.sh
```

State that Stage 2 uses four tactile fields in metadata order, camera0 is blacked only on the client, and the server remains bimanual. Leave existing Stage 1 and right-arm instructions intact.

- [ ] **Step 4: Commit the README**

```bash
cd /home/typhon/FRS_Tact
git add deploy_deco/README.md
git diff --cached --check -- deploy_deco/README.md
git commit --only -m "docs(deco): add stage2 tactile rollout commands" -- \
  deploy_deco/README.md
```

---

## Task 6: Add the separate Stage 2 server profile, config, entrypoint, and launcher

**Files:**

- Modify: `/home/typhon/vb3_robot_server/deploy_scripts/observation_profiles.py`
- Create: `/home/typhon/vb3_robot_server/configs/deco_stage2_server_config.py`
- Create: `/home/typhon/vb3_robot_server/deploy_scripts/bimanual_deco_stage2_online.py`
- Create: `/home/typhon/vb3_robot_server/scripts/bimanual_deco_stage2.sh`
- Create: `/home/typhon/vb3_robot_server/tests/test_deco_stage2_server_config.py`
- Create: `/home/typhon/vb3_robot_server/tests/test_bimanual_deco_stage2_launcher.py`

- [ ] **Step 1: Record the server baseline and request write access if required**

```bash
cd /home/typhon/vb3_robot_server
git status --short
git diff -- deploy_scripts/observation_profiles.py
```

Expected: `observation_profiles.py` is clean; the known dirty files remain untouched. Because this repository is outside `/home/typhon/FRS_Tact`, obtain the required filesystem approval before editing it.

- [ ] **Step 2: Add failing Stage 2 server contract tests**

In `test_deco_stage2_server_config.py`, assert:

```python
assert DECO_STAGE2_SERVER_CONFIG.expected_data_type == "vitac"
assert DECO_STAGE2_SERVER_CONFIG.expected_observation_profile == "deco_vitac_224"
assert DECO_STAGE2_SERVER_CONFIG.expected_control_frequency == 30.0
assert DECO_STAGE2_SERVER_CONFIG.expected_action_horizon == 32
assert DECO_STAGE2_SERVER_CONFIG.expected_steps_per_inference == 32
assert DECO_STAGE2_SERVER_CONFIG.policy_family == "deco-stage2"
assert DECO_STAGE2_SERVER_CONFIG.image_resize_scheme == "A"
profile = resolve_observation_profile("deco_vitac_224", data_type="vitac")
assert profile.resolution == (224, 224)
assert profile.camera_count == 2
```

Monkeypatch the shared runtime `SERVER_CONFIG` to the Stage 2 config and prove that `validate_smolvla_config()` accepts `vitac/deco_vitac_224` and rejects `vision/deco_vision_224`.

In the launcher test, copy the structure of `test_bimanual_deco_launcher.py`, but assert the delegated Python path is `deploy_scripts/bimanual_deco_stage2_online.py` and all CLI arguments are preserved.

- [ ] **Step 3: Confirm the tests fail before creating the files**

```bash
cd /home/typhon/vb3_robot_server
.venv/bin/python -m pytest -q \
  tests/test_deco_stage2_server_config.py \
  tests/test_bimanual_deco_stage2_launcher.py
```

Expected: import/file-not-found failures for the new Stage 2 modules and launcher.

- [ ] **Step 4: Add the `deco_vitac_224` observation profile**

Add exactly:

```python
"deco_vitac_224": ObservationProfile(
    name="deco_vitac_224",
    data_type="vitac",
    resolution=(224, 224),
    camera_count=2,
),
```

Do not change existing profiles.

- [ ] **Step 5: Create the Stage 2 server config by inheritance**

Create `configs/deco_stage2_server_config.py`:

```python
"""Server-owned hardware and runtime defaults for tactile DECO Stage 2."""

from dataclasses import dataclass
from configs.deco_server_config import DecoServerConfig


@dataclass(frozen=True)
class DecoStage2ServerConfig(DecoServerConfig):
    expected_data_type: str = "vitac"
    expected_observation_profile: str = "deco_vitac_224"
    expected_steps_per_inference: int = 32
    policy_family: str = "deco-stage2"


DECO_STAGE2_SERVER_CONFIG = DecoStage2ServerConfig()
SERVER_CONFIG = DECO_STAGE2_SERVER_CONFIG
```

Inheritance intentionally retains scheme A, RGB, two cameras, action limits, gripper behavior, 30 Hz, and horizon 32 from the current server config.

Do not add an unused server-local deployment path. The DECO Stage 2 client sends the validated YAML projection over the existing bridge handshake, exactly as Stage 1 does; the generic SmolVLA `--config` option remains outside this DECO path.

- [ ] **Step 6: Create the Python and shell entrypoints**

The Python entrypoint must mirror Stage 1 but set:

```python
os.environ["VB3_SERVER_CONFIG_MODULE"] = "configs.deco_stage2_server_config"
```

before importing `main` from `deploy_scripts.bimanual_smolvla_online`.

Copy `scripts/bimanual_deco.sh` to `scripts/bimanual_deco_stage2.sh`, change only the displayed entrypoint and final Python path to `deploy_scripts/bimanual_deco_stage2_online.py`, and keep token handling unchanged. Make it executable.

- [ ] **Step 7: Run the dedicated server tests**

```bash
cd /home/typhon/vb3_robot_server
.venv/bin/python -m pytest -q \
  tests/test_deco_stage2_server_config.py \
  tests/test_bimanual_deco_stage2_launcher.py \
  tests/test_tactile_orientation.py \
  tests/test_vbvla_dry_run.py::test_vitac_dry_run_observation_contains_four_tactile_images
```

Expected: all selected tests pass. Do not change stale Stage 1 assertions about scheme B or eight execution steps; the active worktree deliberately uses scheme A and 32 steps.

- [ ] **Step 8: Commit only independent server files**

```bash
cd /home/typhon/vb3_robot_server
git add deploy_scripts/observation_profiles.py \
  configs/deco_stage2_server_config.py \
  deploy_scripts/bimanual_deco_stage2_online.py \
  scripts/bimanual_deco_stage2.sh \
  tests/test_deco_stage2_server_config.py \
  tests/test_bimanual_deco_stage2_launcher.py
git diff --cached --check -- deploy_scripts/observation_profiles.py \
  configs/deco_stage2_server_config.py \
  deploy_scripts/bimanual_deco_stage2_online.py \
  scripts/bimanual_deco_stage2.sh \
  tests/test_deco_stage2_server_config.py \
  tests/test_bimanual_deco_stage2_launcher.py
git commit --only -m "feat(deco): add tactile stage2 server entrypoint" -- \
  deploy_scripts/observation_profiles.py configs/deco_stage2_server_config.py \
  deploy_scripts/bimanual_deco_stage2_online.py scripts/bimanual_deco_stage2.sh \
  tests/test_deco_stage2_server_config.py \
  tests/test_bimanual_deco_stage2_launcher.py
```

---

## Task 7: Add the one-action execution cap and Stage 2 left-arm hold guard

**Files:**

- Modify with pre-existing user changes: `/home/typhon/vb3_robot_server/deploy_scripts/bimanual_smolvla_online.py`
- Modify: `/home/typhon/vb3_robot_server/deploy_scripts/bimanual_smolvla_online_test.py`
- Create: `/home/typhon/vb3_robot_server/tests/test_deco_stage2_runtime.py`

- [ ] **Step 1: Capture the overlapping runtime diff before editing**

```bash
cd /home/typhon/vb3_robot_server
git diff -- deploy_scripts/bimanual_smolvla_online.py > /tmp/deco_stage2_bimanual_smolvla_before.patch
git diff --stat -- deploy_scripts/bimanual_smolvla_online.py
```

This patch is a comparison aid only. Do not reverse-apply it and do not stage the whole shared runtime.

- [ ] **Step 2: Add failing unit tests for limit resolution and the Stage 2 guard**

In `test_deco_stage2_runtime.py`, test a new helper:

```python
@pytest.mark.parametrize(
    ("steps", "override", "expected"),
    [(32, None, 32), (32, 1, 1), (8, 20, 8)],
)
def test_effective_execution_limit(steps, override, expected):
    assert resolve_execution_limit(steps, override) == expected


@pytest.mark.parametrize("override", [0, -1, True, 1.5])
def test_effective_execution_limit_rejects_invalid_override(override):
    with pytest.raises(ValueError, match="max_executed_actions"):
        resolve_execution_limit(32, override)
```

Test `validate_policy_wire_action()` with `SERVER_CONFIG.policy_family="deco-stage2"`: an identity/hold left 10D block with the observed left gripper passes; changing left x or left gripper raises `UnsafeActionError`. With policy family `deco`, the helper returns a generic bimanual action without applying the right-only guard.

Add a Click/runtime test based on the existing `test_main_schedules_exactly_negotiated_steps_per_inference` fixture. Invoke `main` with `--max-executed-actions 1`, provide a valid `(32,20)` raw chunk, capture the length passed to `env.exec_actions()`, and assert:

```python
assert received_raw_shapes == [(32, 20)]
assert scheduled_counts == [1]
```

Replace the existing `test_cli_does_not_expose_server_execution_ceiling` assertion with a compatibility test that locates the `max_executed_actions` Click parameter and asserts its default is `None` and its type is `click.IntRange(min=1)`. Add `"max_executed_actions": None` to the `call_main()` default kwargs so all existing direct callback tests keep their prior behavior.

- [ ] **Step 3: Confirm red tests**

```bash
cd /home/typhon/vb3_robot_server
.venv/bin/python -m pytest -q tests/test_deco_stage2_runtime.py
```

Expected: import failures for the two new helpers.

- [ ] **Step 4: Implement execution-limit resolution**

Add:

```python
def resolve_execution_limit(
    steps_per_inference: int,
    max_executed_actions: int | None,
) -> int:
    negotiated = _validate_max_executed_actions(steps_per_inference)
    if max_executed_actions is None:
        return negotiated
    return min(negotiated, _validate_max_executed_actions(max_executed_actions))
```

Add a Click option:

```python
@click.option(
    "--max-executed-actions",
    type=click.IntRange(min=1),
    default=None,
    help="hard cap on fresh actions scheduled from each validated chunk",
)
```

Pass it into `main` as `max_executed_actions=None`, compute the effective limit after negotiated `steps_per_inference` is available, print it once, and replace only:

```python
max_executed_actions=steps_per_inference
```

with:

```python
max_executed_actions=execution_limit
```

The full 32-step chunk must still be shape/finite/delta validated before the existing fresh-action selection applies the cap.

- [ ] **Step 5: Implement the policy-family-specific wire guard**

Import the existing `validate_single_right_wire_action` from `vbvla_safety` and add:

```python
def validate_policy_wire_action(raw_action, *, obs):
    if getattr(SERVER_CONFIG, "policy_family", None) != "deco-stage2":
        return raw_action
    return validate_single_right_wire_action(
        raw_action,
        expected_left_gripper=np.asarray(obs["robot0_gripper_width"][-1]).reshape(-1),
    )
```

Immediately after receiving an action and before defining/dispatching `dispatch_action_chunk`, assign:

```python
raw_action = validate_policy_wire_action(raw_action, obs=obs)
```

This must not run for Stage 1 or other policy families.

- [ ] **Step 6: Run runtime unit and scheduling regressions**

```bash
cd /home/typhon/vb3_robot_server
.venv/bin/python -m pytest -q \
  tests/test_deco_stage2_runtime.py \
  tests/test_bimanual_action_scheduling.py \
  deploy_scripts/bimanual_smolvla_online_test.py
```

Expected: selected tests pass, including the Click/runtime assertion that `--max-executed-actions 1` reaches `env.exec_actions()` as one scheduled action while the received raw action shape remains `(32,20)`.

- [ ] **Step 7: Audit that prior shared-runtime changes survived**

```bash
cd /home/typhon/vb3_robot_server
git diff --check -- deploy_scripts/bimanual_smolvla_online.py
git diff -- deploy_scripts/bimanual_smolvla_online.py
```

Compare against `/tmp/deco_stage2_bimanual_smolvla_before.patch`. The resulting diff must contain every pre-existing user hunk plus only the new import, two helpers, Click option/argument, guard call, and execution-limit substitution.

- [ ] **Step 8: Do not stage the overlapping shared runtime wholesale**

Commit the new test only if the runtime Stage 2 hunks can also be selectively staged without any user-owned hunks. Otherwise leave both files uncommitted and report them together at handoff. Never run `git add deploy_scripts/bimanual_smolvla_online.py` against the pre-dirty file.

---

## Task 8: Run the paired hardware-free protocol test

**Files:**

- No source changes expected.

- [ ] **Step 1: Run the Stage 2 server dry-run in terminal 1**

```bash
cd /home/typhon/vb3_robot_server
export VB_ROBOT_TOKEN="$(head -n 1 token_list.txt)"
bash scripts/bimanual_deco_stage2.sh \
  --dry-run \
  --dry-run-iterations 1 \
  --action-timeout-s 30
```

Expected before the client starts: server waits for the policy connection and does not initialize cameras or robot controllers.

- [ ] **Step 2: Run the Stage 2 client dry-run in terminal 2**

```bash
cd /home/typhon/FRS_Tact
export VB_ROBOT_TOKEN="$(head -n 1 /home/typhon/vb3_robot_server/token_list.txt)"
bash deploy_deco/scripts/start_deco_stage2_right.sh \
  --server-dry-run \
  --max-iterations 1
```

Expected:

- the client logs visual shape `[1,2,3,224,224]`, tactile shape `[1,4,3,224,224]`, and the exact four-key tactile order;
- the client sends one `(32,20)` bimanual wire action and then `STOP` without waiting for another observation;
- the server prints `completed 1/1 dry-run exchanges; no hardware actions executed`;
- both processes exit normally without camera/controller initialization.

- [ ] **Step 3: Run final static and focused regressions in both repositories**

Client:

```bash
cd /home/typhon/FRS_Tact
git diff --check
uv run --project deploy_deco pytest -q deploy_deco/tests
bash deploy_deco/scripts/start_deco_stage2_right.sh --check
```

Server:

```bash
cd /home/typhon/vb3_robot_server
git diff --check
.venv/bin/python -m pytest -q \
  tests/test_deco_stage2_server_config.py \
  tests/test_bimanual_deco_stage2_launcher.py \
  tests/test_deco_stage2_runtime.py \
  tests/test_tactile_orientation.py \
  tests/test_vbvla_dry_run.py::test_vitac_dry_run_observation_contains_four_tactile_images
```

Expected: all focused Stage 2 tests pass. Report unrelated pre-existing server test failures without changing active scheme A or 32-step configuration merely to satisfy stale assertions.

---

## Task 9: Operator-run real-robot rollout after implementation

**Files:**

- No automated source changes.

- [ ] **Step 1: Observe real inputs without START or actions**

Start the Stage 2 server normally, then run:

```bash
cd /home/typhon/FRS_Tact
export VB_ROBOT_TOKEN="$(head -n 1 /home/typhon/vb3_robot_server/token_list.txt)"
bash deploy_deco/scripts/start_deco_stage2_right.sh --observe-only
```

Expected: the client writes `deploy_deco/outputs/observe_only_*` with six PNGs and `summary.json`, sends no START/action, then sends STOP during cleanup. Inspect once that visual camera0 is black and the four tactile PNGs exist in metadata order and are not all black.

- [ ] **Step 2: Execute the one-action physical smoke**

Server:

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

Client:

```bash
cd /home/typhon/FRS_Tact
export VB_ROBOT_TOKEN="$(head -n 1 /home/typhon/vb3_robot_server/token_list.txt)"
bash deploy_deco/scripts/start_deco_stage2_right.sh --max-iterations 1
```

Expected: the server validates the complete `(32,20)` chunk, verifies the inactive left-arm hold, and schedules at most one fresh action.

- [ ] **Step 3: Start normal Stage 2 deployment**

After the one-action result is accepted, start the server without `--max-executed-actions` and the client without `--max-iterations`:

```bash
cd /home/typhon/vb3_robot_server
bash scripts/bimanual_deco_stage2.sh
```

```bash
cd /home/typhon/FRS_Tact
export VB_ROBOT_TOKEN="$(head -n 1 /home/typhon/vb3_robot_server/token_list.txt)"
bash deploy_deco/scripts/start_deco_stage2_right.sh
```

Expected: normal 32-step Stage 2 tactile deployment; all original Stage 1 launchers remain available and unchanged.

---

## Final Handoff Checklist

- [ ] `git status --short` from both repositories is included in the handoff.
- [ ] The three pre-existing client YAML modifications remain preserved and unstaged unless they were already staged by the user.
- [ ] All pre-existing server shared-runtime changes remain present.
- [ ] The Stage 2 artifact check, CUDA inference, focused client tests, focused server tests, and paired hardware-free protocol test have concrete pass/fail output recorded.
- [ ] No real hardware command was run automatically.
- [ ] The handoff lists the observe-only and one-action commands as the only remaining operator steps.
