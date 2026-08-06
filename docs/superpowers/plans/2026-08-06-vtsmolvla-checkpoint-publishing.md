# VT-SmolVLA Checkpoint Repair and Safe Publishing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Repair the current VT-SmolVLA inference sidecars, make checkpoint assembly atomic, reject mixed base/trained artifacts, and deploy only a validated immutable Hub revision.

**Architecture:** A lightweight validation module owns all inference-contract checks. Training assembles checkpoints under `.incomplete` and exposes the final path only after validation; a separate bundle/publish tool copies an allowlisted inference subset and validates it before upload. Deployment validates the resolved checkpoint before opening a robot connection.

**Tech Stack:** Python 3.12, JAX/Flax Safetensors, Hugging Face Hub, PyYAML, pytest.

## Global Constraints

- Authoritative contract: state dimension 20, action dimension 20, chunk size 20, action steps 5.
- RGB keys: `observation.images.camera1`, `observation.images.camera2`.
- Tactile keys: `observation.images.tactile_left_0`, `observation.images.tactile_right_0`, `observation.images.tactile_left_1`, `observation.images.tactile_right_1`.
- Tactile encoder must be enabled with four 512-dimensional tokens and LoRA rank 16 on VLM Q/V projections.
- Never rewrite or retrain the 1.15 GB Hub weight for a sidecar-only repair.
- Never upload training state, optimizer state, dataset cache, or an `.incomplete` directory.
- Never connect to or start the physical robot while a checkpoint validation error exists.
- Every behavior change follows test-first RED/GREEN evidence.

---

### Task 1: Checkpoint Contract Validator

**Files:**
- Create: `src/lerobot/policies/smolvla_jax/validation.py`
- Modify: `src/lerobot/policies/smolvla_jax/__init__.py`
- Create: `tests/jax/test_checkpoint_validation.py`

**Interfaces:**
- Produces: `CheckpointContract`, `CheckpointValidationReport`, `validate_checkpoint(path, *, expected=None, base_sidecars=None, require_weight=True)`.
- Consumers: Tasks 2–4 use `validate_checkpoint` as their only validation entry point.

- [ ] **Step 1: Write failing validation tests**

Create fixtures with tiny Safetensors headers and sidecars. Cover a valid 20D VT bundle, base 6D/50 sidecars, processor/config disagreement, missing or wrong-dimensional stats, missing tactile tensors, and aggregated diagnostics.

```python
def test_mixed_base_sidecars_are_rejected(vt_bundle, base_sidecars):
    report = validate_checkpoint(vt_bundle, expected=VT_CONTRACT, base_sidecars=base_sidecars)
    assert not report.ok
    assert any("byte-identical to base" in issue for issue in report.issues)

def test_valid_vt_bundle_passes(vt_bundle):
    report = validate_checkpoint(vt_bundle, expected=VT_CONTRACT)
    assert report.ok, report.format_errors()
```

- [ ] **Step 2: Run tests and verify RED**

Run: `python -m pytest -q tests/jax/test_checkpoint_validation.py`

Expected: collection fails because `lerobot.policies.smolvla_jax.validation` does not exist.

- [ ] **Step 3: Implement the validator**

Use dataclasses for the expected contract and report. Parse JSON directly, inspect normalization Safetensors with `safe_open`, inspect only the model header/keys, compare configured keys and dimensions, and hash sidecars only when a base directory is supplied.

```python
@dataclass(frozen=True)
class CheckpointContract:
    state_dim: int
    action_dim: int
    chunk_size: int
    image_keys: tuple[str, ...]
    tactile_keys: tuple[str, ...] = ()
    tactile_embedding_dim: int = 512
    tactile_num_tokens: int = 0
    lora_rank: int = 0
    vlm_lora_target_modules: tuple[str, ...] = ()

@dataclass(frozen=True)
class CheckpointValidationReport:
    path: Path
    issues: tuple[str, ...]

    @property
    def ok(self) -> bool:
        return not self.issues

    def require_valid(self) -> None:
        if self.issues:
            raise ValueError(self.format_errors())
```

- [ ] **Step 4: Run focused tests and verify GREEN**

Run: `python -m pytest -q tests/jax/test_checkpoint_validation.py`

Expected: all validation tests pass with no warnings.

- [ ] **Step 5: Run neighboring tests and commit**

Run: `python -m pytest -q tests/jax/test_checkpoint.py tests/jax/test_checkpoint_validation.py`

Commit: `feat: validate SmolVLA checkpoint contracts`

---

### Task 2: Atomic Training Checkpoint Assembly

**Files:**
- Create: `src/lerobot/policies/smolvla_jax/atomic_checkpoint.py`
- Modify: `tools/train_smolvla_jax.py:659-665`
- Create: `tests/jax/test_atomic_checkpoint.py`
- Modify: `tests/jax/test_train_script.py`

**Interfaces:**
- Consumes: `validate_checkpoint` from Task 1.
- Produces: `assemble_checkpoint_atomically(final_path, writer, validator)` and a training-loop call that writes all assets through one callback.

- [ ] **Step 1: Write failing atomicity tests**

```python
def test_final_path_appears_only_after_validation(tmp_path):
    final = tmp_path / "checkpoint-00000020"
    observed = []
    def writer(staging):
        observed.append(final.exists())
        (staging / "marker").write_text("complete")
    assemble_checkpoint_atomically(final, writer, lambda path: None)
    assert observed == [False]
    assert (final / "marker").read_text() == "complete"

def test_failed_validation_preserves_incomplete_directory(tmp_path):
    final = tmp_path / "checkpoint-00000020"
    with pytest.raises(ValueError, match="invalid"):
        assemble_checkpoint_atomically(final, lambda p: (p / "x").write_text("x"), _fail)
    assert not final.exists()
    assert final.with_name(final.name + ".incomplete").exists()
```

- [ ] **Step 2: Run tests and verify RED**

Run: `python -m pytest -q tests/jax/test_atomic_checkpoint.py`

Expected: import failure for the missing atomic checkpoint module.

- [ ] **Step 3: Implement atomic assembly and integrate training**

The helper rejects an existing final or staging directory, creates the staging directory, invokes the writer, invokes validation, then uses `Path.replace` for the same-filesystem atomic rename. It never deletes a failed staging directory.

The training writer callback must perform, in order: `trainer.save`, `save_normalization_assets`, and `data_split.json` copy. Validation runs after all three.

- [ ] **Step 4: Run focused tests and verify GREEN**

Run: `python -m pytest -q tests/jax/test_atomic_checkpoint.py tests/jax/test_train_script.py`

Expected: all tests pass with no warnings.

- [ ] **Step 5: Commit**

Commit: `fix: publish only complete training checkpoints`

---

### Task 3: Validated Inference Bundle and Sidecar Repair Tool

**Files:**
- Create: `tools/publish_smolvla_checkpoint.py`
- Create: `tests/jax/test_publish_checkpoint.py`
- Modify: `pyproject.toml` only if an already-declared dependency is insufficient.

**Interfaces:**
- Consumes: `validate_checkpoint`, `write_effective_config`, `JaxSmolVLAPreprocessor.save_normalization_assets`, dataset-stat canonicalization helpers.
- Produces: `build_inference_bundle(source, destination, *, expected, include_model=True)`, `repair_sidecars(...)`, and CLI modes `validate`, `bundle`, `repair-sidecars`, `publish`.

- [ ] **Step 1: Write failing bundle and publish-guard tests**

Cover the exact inference allowlist, exclusion of `training_state.msgpack`, refusal of `.incomplete`, refusal to publish a failing report, preservation of the existing weight hash during sidecar-only repair, and provenance manifest content.

```python
def test_bundle_excludes_training_state(valid_checkpoint, tmp_path):
    bundle = build_inference_bundle(valid_checkpoint, tmp_path / "bundle", expected=VT_CONTRACT)
    assert not (bundle / "training_state.msgpack").exists()
    assert (bundle / "conversion_manifest.json").is_file()

def test_publish_refuses_invalid_bundle(invalid_bundle):
    with pytest.raises(ValueError, match="checkpoint validation failed"):
        publish_bundle(invalid_bundle, repo_id="owner/model", api=RecordingApi())
```

- [ ] **Step 2: Run tests and verify RED**

Run: `python -m pytest -q tests/jax/test_publish_checkpoint.py`

Expected: import failure because `tools.publish_smolvla_checkpoint` is absent.

- [ ] **Step 3: Implement bundle, repair, and publishing commands**

The repair command reads the effective `model` block from the training YAML, resolves every dataset revision through `HfApi.dataset_info`, and verifies from repository history that each resolved SHA existed no later than the model-weight upload. If that training-time revision cannot be proven, repair stops before writing or publishing. It then loads metadata-only dataset snapshots, aggregates state/action statistics through the same helpers used by training, writes canonical processor assets, records dataset SHAs and file hashes, and validates the result.

Publishing uses `HfApi.create_commit` with `CommitOperationAdd` for corrected small sidecars. Sidecar-only mode verifies the remote model LFS SHA-256 before and after and never sends `model.safetensors`.

The CLI prints JSON validation output and exits nonzero on any issue.

- [ ] **Step 4: Run focused tests and verify GREEN**

Run: `python -m pytest -q tests/jax/test_publish_checkpoint.py tests/jax/test_checkpoint_validation.py`

Expected: all tests pass with no warnings.

- [ ] **Step 5: Run CLI help and commit**

Run: `python tools/publish_smolvla_checkpoint.py --help`

Expected: exits 0 and lists `validate`, `bundle`, `repair-sidecars`, `publish`.

Commit: `feat: build and publish validated SmolVLA bundles`

---

### Task 4: Deployment Validation and Correct VT Configuration

**Files:**
- Modify: `deploy_smolvla/remote_client.py:388-417`
- Modify: `configs/deploy_smolvla_jax.yaml:7-21`
- Modify: `tests/jax/test_tactile_integration.py`

**Interfaces:**
- Consumes: `validate_checkpoint` from Task 1.
- Produces: deployment aborts before `RobotBridgeClient` construction when the checkpoint contract is invalid.

- [ ] **Step 1: Write failing deployment-order test**

Extract or inject a policy-loading boundary so the test can prove validation occurs before bridge construction without using a real WebSocket.

```python
def test_invalid_checkpoint_fails_before_robot_connection(tmp_path, monkeypatch):
    connected = False
    monkeypatch.setattr(remote_client, "RobotBridgeClient", lambda *a, **k: _mark_connected())
    with pytest.raises(ValueError, match="checkpoint validation failed"):
        remote_client.run(invalid_config_path)
    assert connected is False
```

- [ ] **Step 2: Run test and verify RED**

Run: `python -m pytest -q tests/jax/test_tactile_integration.py -k invalid_checkpoint`

Expected: test fails because deployment does not call the shared validator.

- [ ] **Step 3: Integrate validation and update YAML**

Validate the resolved local snapshot before policy construction. Update YAML to:

```yaml
checkpoint: KaiyueChen/vtsmolvla_01_4w
revision: null  # replaced with repaired immutable SHA after publish
allow_download: true
```

Keep `data_type: vitac`, bimanual state mode, `action_horizon: 20`, `steps_per_inference: 5`, and only the two RGB rename mappings.

- [ ] **Step 4: Run focused tests and verify GREEN**

Run: `python -m pytest -q tests/jax/test_tactile_integration.py tests/jax/test_checkpoint_validation.py`

Expected: all tests pass with no warnings.

- [ ] **Step 5: Commit**

Commit: `fix: validate VT checkpoint before robot connection`

---

### Task 5: Repair Current Hub Revision and End-to-End Verification

**Files:**
- Create locally generated bundle under `outputs/vtsmolvla_01_4w_repair/` (ignored artifact, not committed).
- Modify: `configs/deploy_smolvla_jax.yaml` to pin the repaired Hub SHA.
- Test: all targeted JAX/deployment test files and clean-download validation.

**Interfaces:**
- Consumes: publisher and validator from Tasks 1–4.
- Produces: repaired Hub sidecars and immutable deployment revision.

- [ ] **Step 1: Generate and validate the local repair bundle**

Run:

```bash
python tools/publish_smolvla_checkpoint.py repair-sidecars \
  --repo-id KaiyueChen/vtsmolvla_01_4w \
  --training-config configs/train_vtsmolvla_jax.yaml \
  --output outputs/vtsmolvla_01_4w_repair
```

Expected: JSON report says valid, weight SHA is `9f52272d5202289e4a98ec45f4ba3bd7c567e59c0a1f8967faeb1bed106e89b4`, and state/action/chunk are 20/20/20.

- [ ] **Step 2: Publish corrected small sidecars**

Run only after Step 1 succeeds:

```bash
python tools/publish_smolvla_checkpoint.py publish \
  --bundle outputs/vtsmolvla_01_4w_repair \
  --repo-id KaiyueChen/vtsmolvla_01_4w \
  --sidecars-only
```

Expected: one Hub commit URL; no weight upload operation.

- [ ] **Step 3: Clean-download and validate the repaired revision**

Resolve the returned immutable SHA into a fresh temporary cache and run the validator. Expected: valid 20D/20D/chunk-20 VT contract and unchanged model SHA.

- [ ] **Step 4: Pin revision and run full relevant tests**

Update `revision:` to the returned Hub SHA.

Run:

```bash
python -m pytest -q \
  tests/jax/test_checkpoint.py \
  tests/jax/test_checkpoint_validation.py \
  tests/jax/test_atomic_checkpoint.py \
  tests/jax/test_publish_checkpoint.py \
  tests/jax/test_tactile_integration.py \
  tests/jax/test_train_script.py
```

Expected: all tests pass with no warnings.

- [ ] **Step 5: Run static checks and commit**

Run: `python -m compileall -q src/lerobot/policies/smolvla_jax deploy_smolvla tools`

Run: `git diff --check`

Expected: both commands exit 0.

Commit: `chore: pin repaired VT-SmolVLA revision`
