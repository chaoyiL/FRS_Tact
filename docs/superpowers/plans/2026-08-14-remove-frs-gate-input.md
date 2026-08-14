# Remove FRS Gate Input Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove `w` from every FRS decoder and deployment input while retaining it as a training-only loss and evaluation label.

**Architecture:** The decoder consumes action flow state, time, tactile history, and optional robot state only. Training still computes `w` for the gated objective and stratified metrics. Deployment accepts only newly trained input-v2 checkpoints and contains no gate configuration or diagnostic value.

**Tech Stack:** Python, JAX, Flax NNX, NumPy, pytest, YAML.

## Global Constraints

- Preserve the user's current checkpoint, FireFlow, and solver-contract worktree changes.
- Do not migrate or silently load `gate_conditioning=true` checkpoints.
- Keep the gated loss formula and training gate configuration unchanged.
- Use TDD: run each new/changed test red before implementation and green afterward.

---

### Task 1: Remove the Decoder Gate Input

**Files:**
- Modify: `tests/train_frs/test_model.py`
- Modify: `train_frs/utils/model.py`
- Modify: `train_frs/utils/checkpoint.py`
- Modify: `train_frs/utils/visualize.py`

**Interfaces:**
- Produces: `TactileConditionedFlowDecoder.__call__(x_t, t, tactile_seq, *, state=None, state_keep_mask=None)`.
- Produces: `decode_actions(model, x_base, tactile_seq, *, num_steps, solver="euler", state=None, state_keep_mask=None)`.
- Keeps: gated loss functions accept `gate_weights` only as supervision.

- [ ] **Step 1: Write failing model-boundary tests**

Replace gate-conditioned decoder tests with assertions equivalent to:

```python
def test_decoder_has_no_gate_input(decoder):
    assert "gate_conditioning" not in dataclasses.asdict(decoder.config)
    assert not hasattr(decoder, "gate_mlp")
    assert "gate_weights" not in inspect.signature(decoder.__call__).parameters
    assert "gate_weights" not in inspect.signature(decode_actions).parameters


def test_legacy_gate_conditioned_checkpoint_is_rejected(tmp_path, decoder):
    save_checkpoint(tmp_path, decoder, epoch=1, metrics={"val_mse": 0.5})
    metadata_path = tmp_path / "checkpoint.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["decoder_config"]["gate_conditioning"] = True
    metadata_path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="gate-conditioned.*retrain"):
        load_checkpoint(tmp_path)
```

Update decode reference tests so calls contain no positional gate argument.

- [ ] **Step 2: Run tests and verify RED**

Run:

```bash
JAX_PLATFORMS=cpu uv run --no-sync pytest tests/train_frs/test_model.py -q
```

Expected: failures show `gate_conditioning`, `gate_mlp`, or `gate_weights` still exists.

- [ ] **Step 3: Remove gate parameters from the model**

Implement these gate-free boundaries and remove all internal forwarding of `gate_weights`:

```python
class TactileConditionedFlowDecoder(nnx.Module):
    def velocity_from_condition(self, x_t, t, tactile_condition):
        return self._decode_velocity(x_t, t, tactile_condition)

    def __call__(
        self,
        x_t,
        t,
        tactile_seq,
        *,
        state=None,
        state_keep_mask=None,
    ):
        condition = self.encode_condition(tactile_seq, state, state_keep_mask)
        return self.velocity_from_condition(x_t, t, condition)


def decode_actions(
    model,
    x_base,
    tactile_seq,
    *,
    num_steps,
    solver="euler",
    state=None,
    state_keep_mask=None,
):
    if num_steps <= 0:
        raise ValueError(f"num_steps must be positive, got {num_steps}.")
    if solver not in ("euler", "fireflow"):
        raise ValueError(f"solver must be 'euler' or 'fireflow', got {solver!r}.")
    return _decode_actions_jitted(
        model,
        x_base,
        tactile_seq,
        state,
        state_keep_mask,
        num_steps=num_steps,
        solver=solver,
    )
```

Remove `gate_weights` from `flow_matching_loss_per_sample`, `decode_mse_per_sample`, `gt_supervised_loss_per_sample`, Euler/FireFlow helpers, and visualization decode calls. In `gated_loss_components_per_sample`, retain the argument for weighting but call model/loss/decode helpers without it.

In `load_checkpoint`, reject old gated metadata before constructing the config:

```python
raw_config = dict(metadata["decoder_config"])
if bool(raw_config.pop("gate_conditioning", False)):
    raise ValueError(
        "Gate-conditioned FRS checkpoints are incompatible with decoder input v2; retrain the model."
    )
config = DecoderConfig(**raw_config)
```

- [ ] **Step 4: Run model tests and verify GREEN**

Run the Step 2 command. Expected: PASS.

- [ ] **Step 5: Commit clean model files**

```bash
git add train_frs/utils/model.py train_frs/utils/checkpoint.py train_frs/utils/visualize.py tests/train_frs/test_model.py
git commit -m "refactor: remove gate input from FRS decoder"
```

---

### Task 2: Keep Gate as Training-Only Supervision

**Files:**
- Modify: `train_frs/train.py`
- Modify: `train_frs/evaluate.py`
- Modify: `train_frs/utils/metrics.py`
- Modify: `train_frs/train_frs.py`
- Modify: `tests/test_frs_run_config.py`
- Modify: `tests/train_frs/test_model.py`

**Interfaces:**
- Consumes: gate-free decoder APIs from Task 1.
- Produces: checkpoint metadata `decoder_input_version: 2`.
- Keeps: `gate_tau`, `gate_temperature`, thresholds, gated losses, history, and checkpoint selection.

- [ ] **Step 1: Write failing training metadata and evaluation tests**

Add assertions that a gated training checkpoint contains:

```python
assert extra_metadata["decoder_input_version"] == 2
assert "gate_conditioning" not in extra_metadata
```

Update evaluation tests to verify gate labels are still reported without being passed to model APIs.

- [ ] **Step 2: Run tests and verify RED**

```bash
JAX_PLATFORMS=cpu uv run --no-sync pytest tests/train_frs/test_model.py tests/test_frs_run_config.py -q
```

Expected: metadata still contains `gate_conditioning`, or old model calls still pass `w`.

- [ ] **Step 3: Decouple training supervision from decoder input**

Construct `DecoderConfig` without `gate_conditioning`, remove resume checks against that field, remove the `"gate_conditioning"` entry from the checkpoint metadata literal, and add:

```python
"decoder_input_version": 2,
"loss_weighting_version": 7,
```

In `train_step`, `evaluate_split`, and training evaluation, continue computing and aggregating `gate_w`, but call these APIs without it:

```python
flow_gt = flow_matching_loss_per_sample(model, x_base, gt_action, t, tactile_seq, state=state)
prediction = decode_actions(
    model,
    x_base,
    tactile_seq,
    num_steps=num_steps,
    solver=solver,
    state=state,
)
```

Use `loss_mode == "gated"` alone to enable gate metrics in `train_frs/evaluate.py`.

- [ ] **Step 4: Run training tests and verify GREEN**

Run the Step 2 command. Expected: PASS.

- [ ] **Step 5: Commit training files**

```bash
git add train_frs/train.py train_frs/evaluate.py train_frs/utils/metrics.py train_frs/train_frs.py tests/test_frs_run_config.py tests/train_frs/test_model.py
git commit -m "refactor: make FRS gate training-only"
```

---

### Task 3: Remove Gate from Deployment

**Files:**
- Modify: `deploy_smolvla/frs_runtime.py`
- Modify: `deploy_smolvla/remote_client.py`
- Modify: `deploy_smolvla/configs/deploy_frs.yaml`
- Modify: `tests/jax/test_frs_deployment.py`
- Modify: `tests/jax/test_tactile_integration.py`

**Interfaces:**
- Consumes: `decoder_input_version: 2` checkpoints and gate-free `decode_actions`.
- Produces: deployment diagnostics without `gate_weight`.

- [ ] **Step 1: Write failing deployment tests**

Update fixtures to omit gate config and assert:

```python
assert not hasattr(runtime.config, "gate_tau")
assert not hasattr(runtime.config, "gate_temperature")
assert not hasattr(result.diagnostics, "gate_weight")
assert "gate_weight" not in trace["frs_diagnostics"]
```

Add a stale-config test:

```python
with pytest.raises(ValueError, match="deprecated"):
    validate_frs_config_section({**config, "frs": {**config["frs"], "gate_tau": 0.4}})
```

Change the contract fixture to `decoder_input_version=2` and add rejection of any other version.

- [ ] **Step 2: Run tests and verify RED**

```bash
JAX_PLATFORMS=cpu uv run --no-sync pytest tests/jax/test_frs_deployment.py tests/jax/test_tactile_integration.py -q
```

Expected: deployment still requires, computes, logs, or serializes gate values.

- [ ] **Step 3: Remove deployment gate behavior**

Remove `gate_weights_from_change` import, `FRSConfig.gate_tau`, `FRSConfig.gate_temperature`, and `FRSDiagnostics.gate_weight`. Reject stale gate keys:

```python
deprecated = {"gate_tau", "gate_temperature"}.intersection(raw)
if deprecated:
    raise ValueError(f"Deprecated FRS gate config values: {sorted(deprecated)}")
```

Require the new checkpoint contract:

```python
_require_equal(extra.get("loss_mode"), "gated", "loss_mode")
_require_equal(int(extra.get("decoder_input_version", 0)), 2, "decoder_input_version")
```

In `steer_action`, warmup, and legacy `steer`, remove gate calculation and call `decode_actions` with tactile and optional state only. Keep tactile change diagnostics in real steering paths. Remove `gate_weight` from trace dictionaries and console logs.

- [ ] **Step 4: Run deployment tests and verify GREEN**

Run the Step 2 command. Expected: PASS.

- [ ] **Step 5: Preserve overlapping user edits without committing them**

```bash
git diff -- deploy_smolvla/configs/deploy_frs.yaml deploy_smolvla/frs_runtime.py tests/jax/test_frs_deployment.py
```

Expected: the pre-existing checkpoint, FireFlow, and solver-contract changes remain alongside the gate removal. Leave these overlapping files uncommitted for user review.

---

### Task 4: Remove Gate Counterfactual Inputs

**Files:**
- Modify: `modalities_eval/frs/interventions.py`
- Modify: `modalities_eval/frs/evaluate.py`
- Modify: `modalities_eval/frs/test/test_interventions.py`
- Modify: `modalities_eval/frs/test/test_evaluate.py`

**Interfaces:**
- Consumes: gate-free `decode_actions`.
- Keeps: gate labels in report rows for stratified analysis.
- Removes: `gate_0.0`, `gate_0.5`, and `gate_1.0` interventions.

- [ ] **Step 1: Write failing modality tests**

Assert the default intervention set is exactly tactile-only and decode callbacks keep robot state while dropping gate input:

```python
assert {item.name for item in DEFAULT_INTERVENTIONS} == {
    "baseline_fixed",
    "baseline_recomputed",
    "current_only",
    "drop_sensor_0",
    "drop_sensor_1",
    "drop_sensor_2",
    "drop_sensor_3",
}

def decode(x_base, tactile, state):
    assert state.shape[0] == x_base.shape[0]
    return x_base
```

Assert the removed gate intervention fails explicitly:

```python
with pytest.raises(ValueError, match="unsupported intervention"):
    apply_intervention(
        "gate_0.5",
        tactile,
        baseline,
        gate,
        tau=0.4,
        temperature=0.1,
    )
```

- [ ] **Step 2: Run tests and verify RED**

```bash
JAX_PLATFORMS=cpu uv run --no-sync pytest modalities_eval/frs/test -q
```

Expected: gate interventions remain or decode callbacks still require a gate input.

- [ ] **Step 3: Make interventions tactile-only**

Remove gate entries and the `gate_` branch. Include state in `EvaluationContext.batches()` and change decode boundaries to:

```python
def _decode_checked(decode_fn, x_base, tactile, state):
    prediction = np.asarray(
        decode_fn(x_base.copy(), tactile, state),
        dtype=np.float32,
    )
    if prediction.shape != x_base.shape:
        raise ValueError(
            f"decode output shape {prediction.shape} does not match action shape {x_base.shape}"
        )
    return prediction

full = _decode_checked(decode_fn, fixed_x_base, tactile, state)
predictions[name] = _decode_checked(
    decode_fn,
    fixed_x_base,
    changed.tactile,
    state,
)
```

Continue calculating original and recomputed gate labels only for `sample_error_rows`; never pass them to `decode_actions`. Forward `state` to `decode_actions`. Require `loss_mode=gated` and `decoder_input_version=2` when loading the evaluation context.

- [ ] **Step 4: Run modality tests and verify GREEN**

Run the Step 2 command. Expected: PASS.

- [ ] **Step 5: Commit modality files**

```bash
git add modalities_eval/frs/interventions.py modalities_eval/frs/evaluate.py modalities_eval/frs/test/test_interventions.py modalities_eval/frs/test/test_evaluate.py
git commit -m "refactor: remove FRS gate counterfactual inputs"
```

---

### Task 5: Final Verification

**Files:**
- Modify if needed: `train_frs/README.md`
- Modify if needed: `modalities_eval/frs/README.md`

**Interfaces:**
- Verifies: no model or deployment gate input remains.

- [ ] **Step 1: Update documentation language**

State that `w` is a training-only supervision signal and that deployment checkpoints must use decoder input v2. Remove statements claiming raw `w` is passed to the decoder.

- [ ] **Step 2: Scan forbidden runtime/model references**

```bash
rg -n 'gate_conditioning|gate_mlp|gate_weight|gate_tau|gate_temperature' train_frs deploy_smolvla modalities_eval/frs --glob '*.py' --glob '*.yaml' --glob '*.md'
```

Expected: gate training configuration and metric labels may remain under `train_frs`; no decoder, deployment config, deployment diagnostic, or gate-counterfactual reference remains.

- [ ] **Step 3: Run targeted suites**

```bash
JAX_PLATFORMS=cpu uv run --no-sync pytest tests/train_frs tests/flow_decoder tests/jax/test_frs_deployment.py tests/jax/test_tactile_integration.py modalities_eval/frs/test tests/test_frs_run_config.py -q
```

Expected: PASS.

- [ ] **Step 4: Run the complete suite**

```bash
JAX_PLATFORMS=cpu uv run --no-sync pytest -q
```

Expected: PASS, or report unrelated environment-only failures with their full command and output.

- [ ] **Step 5: Review final diff**

```bash
git diff --check
git status --short
git diff --stat
```

Expected: no whitespace errors; the user's original deployment edits are preserved.
