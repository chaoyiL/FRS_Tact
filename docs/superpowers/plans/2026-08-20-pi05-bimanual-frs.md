# Pi0.5 Bimanual FRS Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add the SmolVLA `bimanual_gated` objective-v2/weighting-v7 training method to Pi0.5, including evaluation, diagnostics, checkpoint/resume validation, and deployment acceptance, while preserving the existing scalar pipeline.

**Architecture:** Keep Pi0.5's decoder/cache width unchanged and introduce a fixed physical-action contract for the first 20 dimensions. Build independent left/right Gate labels from the frozen embedding cache, train one masked composite-endpoint FM objective plus per-wrist auxiliaries, and extend metrics/checkpoints/deployment through metadata rather than changing decoder inputs.

**Tech Stack:** Python 3.12, JAX 0.5.3, Flax NNX 0.10.2, Optax, NumPy, Matplotlib, PyYAML, pytest/unittest.

## Global Constraints

- Existing `gt`, `predicted`, and scalar `gated` behavior and checkpoints remain compatible.
- The current 32D Pi0.5 model/cache stays 32D; physical steering uses `[0, 20)` and deployment still truncates to the 20D robot action.
- Left/right action slices are `[0, 10)` and `[10, 20)`; token groups are `[0, 1]` and `[2, 3]`.
- The padded tail `[20, action_dim)` is assigned the frozen Pi0.5 endpoint and excluded from bimanual loss/metrics normalization.
- Gate labels come from the immutable tactile embedding cache and are never decoder inputs.
- Do not add trainable tactile ResNet/raw-image training.
- Preserve Pi0.5's transactional generation checkpoint format and cache provenance checks.
- Add `train_pi05_frs/configs/train_pi05_frs_bimanual_gated.yaml`; do not change the semantics of `train_pi05_frs/configs/train_pi05_frs.yaml`.
- Do not overwrite the user's unrelated dirty files.

## File Structure

- Create `train_pi05_frs/utils/bimanual_schema.py`: fixed physical-action/token contract and metadata validators.
- Create `train_pi05_frs/utils/bimanual_metrics.py`: pure NumPy quadrant and per-wrist rollup helpers.
- Create `train_pi05_frs/utils/bimanual_visualize.py`: four stable bimanual training/evaluation plots.
- Create `train_pi05_frs/configs/train_pi05_frs_bimanual_gated.yaml`: independent production configuration.
- Modify `train_pi05_frs/utils/data.py`: per-wrist tactile changes and Gate accessors.
- Modify `train_pi05_frs/utils/model.py`: masked composite endpoint, per-wrist auxiliaries, and train-step branch.
- Modify `train_pi05_frs/train.py`: bimanual loop, metadata, history, selection, and resume validation.
- Modify `train_pi05_frs/tools/train_frs.py`: strict YAML schema/validation and argument forwarding.
- Modify `train_pi05_frs/utils/metrics.py`: bimanual validation result fields and aggregations.
- Modify `train_pi05_frs/evaluate.py`: objective-aware metrics/CSV/JSON/plots.
- Modify `train_pi05_frs/utils/history_plot.py` and `train_pi05_frs/plot_history.py`: legacy-safe bimanual history plotting.
- Modify `deploy_pi05/frs_runtime.py`: accept and validate objective-v2 metadata without changing inference.
- Modify `train_pi05_frs/README.md`: new configuration, outputs, and deployment handoff.
- Extend `train_pi05_frs/tests/` with focused schema/data/model/pipeline/metrics/visualization/deployment tests.

---

### Task 1: Fixed bimanual schema and independent configuration

**Files:**
- Create: `train_pi05_frs/utils/bimanual_schema.py`
- Create: `train_pi05_frs/configs/train_pi05_frs_bimanual_gated.yaml`
- Modify: `train_pi05_frs/tests/test_pipeline.py`
- Modify: `train_pi05_frs/tools/train_frs.py:42-137,825-1087`

**Interfaces:**
- Produces: `BIMANUAL_LOSS_MODE`, `BIMANUAL_OBJECTIVE_VERSION`, `STEERED_ACTION_DIM`, action slices/token groups, `bimanual_objective_metadata()`, `validate_bimanual_tactile_keys()`, and `validate_bimanual_objective_metadata()`.
- Consumes: YAML `model.action_dim`, `model.tactile_keys`, and `frs_training.loss_mode`.

- [ ] **Step 1: Write failing schema/config tests**

Add these tests to `train_pi05_frs/tests/test_pipeline.py`:

```python
from train_pi05_frs.utils.bimanual_schema import (
    BIMANUAL_LOSS_MODE,
    bimanual_objective_metadata,
    validate_bimanual_objective_metadata,
)


def test_bimanual_config_is_independent_and_validated():
    path = TRAIN_ROOT / "configs" / "train_pi05_frs_bimanual_gated.yaml"
    config = load_config(path)
    assert config["frs_training"]["loss_mode"] == BIMANUAL_LOSS_MODE
    assert "gate_lambda" not in config["frs_training"]
    validate_config(config, check_paths=False)


def test_bimanual_metadata_supports_native_and_padded_action_widths():
    for action_dim in (20, 32):
        metadata = bimanual_objective_metadata(action_dim=action_dim)
        validate_bimanual_objective_metadata(metadata, action_dim=action_dim)
        assert metadata["steered_action_dim"] == 20
        assert metadata["padded_tail_policy"] == "vla_endpoint_masked"


def test_bimanual_metadata_rejects_wrong_tail_policy():
    metadata = bimanual_objective_metadata(action_dim=32)
    metadata["padded_tail_policy"] = "train_gt"
    with pytest.raises(ValueError, match="padded_tail_policy"):
        validate_bimanual_objective_metadata(metadata, action_dim=32)
```

- [ ] **Step 2: Run the focused tests and verify RED**

Run:

```bash
cd /home/typhon/FRS_Tact/train_pi05_frs
PYTHONPATH="$PWD/src:$(dirname "$PWD")" PYTHONSAFEPATH=1 .venv/bin/python -m pytest tests/test_pipeline.py -k bimanual -q
```

Expected: collection/import failure because `train_pi05_frs.utils.bimanual_schema` and the independent YAML do not exist.

- [ ] **Step 3: Implement the schema**

Create `train_pi05_frs/utils/bimanual_schema.py` with this public contract:

```python
"""Fixed schema for Pi0.5's physical bimanual FRS objective."""

from collections.abc import Mapping, Sequence
from typing import Any

BIMANUAL_LOSS_MODE = "bimanual_gated"
BIMANUAL_OBJECTIVE_VERSION = 2
LOSS_WEIGHTING_VERSION = 7
STEERED_ACTION_DIM = 20
LEFT_ACTION_SLICE = slice(0, 10)
RIGHT_ACTION_SLICE = slice(10, 20)
LEFT_WRIST_TOKEN_INDICES = (0, 1)
RIGHT_WRIST_TOKEN_INDICES = (2, 3)
PADDED_TAIL_POLICY = "vla_endpoint_masked"
BIMANUAL_TACTILE_KEY_BASENAMES = (
    "tactile_left_0", "tactile_right_0",
    "tactile_left_1", "tactile_right_1",
)


def validate_bimanual_tactile_keys(
    tactile_keys: Sequence[object], *, field_name: str = "model.tactile_keys"
) -> tuple[str, ...]:
    if isinstance(tactile_keys, (str, bytes)):
        raise ValueError(f"{field_name} must contain the fixed bimanual tactile key order")
    actual = tuple(str(key) for key in tactile_keys)
    basenames = tuple(key.rsplit(".", 1)[-1] for key in actual)
    if basenames != BIMANUAL_TACTILE_KEY_BASENAMES:
        raise ValueError(
            f"{field_name} must contain {BIMANUAL_TACTILE_KEY_BASENAMES!r}, got {basenames!r}"
        )
    return actual


def bimanual_objective_metadata(*, action_dim: int) -> dict[str, object]:
    if action_dim < STEERED_ACTION_DIM:
        raise ValueError(f"bimanual objective requires action_dim >= {STEERED_ACTION_DIM}")
    return {
        "loss_mode": BIMANUAL_LOSS_MODE,
        "loss_objective_version": BIMANUAL_OBJECTIVE_VERSION,
        "loss_weighting_version": LOSS_WEIGHTING_VERSION,
        "action_dim": int(action_dim),
        "steered_action_dim": STEERED_ACTION_DIM,
        "action_slices": {"left": [0, 10], "right": [10, 20]},
        "wrist_token_indices": {"left": [0, 1], "right": [2, 3]},
        "padded_tail_policy": PADDED_TAIL_POLICY,
    }


def validate_bimanual_objective_metadata(
    metadata: Mapping[str, Any], *, action_dim: int
) -> None:
    expected = bimanual_objective_metadata(action_dim=action_dim)
    for field in (
        "loss_mode", "loss_objective_version", "loss_weighting_version",
        "action_dim", "steered_action_dim", "action_slices",
        "wrist_token_indices", "padded_tail_policy",
    ):
        if metadata.get(field) != expected[field]:
            raise ValueError(f"bimanual objective metadata has invalid {field}")
```

Extend `TRAINING_KEYS`, the `loss_mode` validator, and tactile-key/action-width validation in `tools/train_frs.py`. Reject `gate_lambda` only when mode is `bimanual_gated`.

Create `configs/train_pi05_frs_bimanual_gated.yaml` by retaining the Pi0.5 checkpoint/cache/model settings and setting:

```yaml
frs_training:
  output: /workspace/frs_pick_tube_pi05/run_bimanual_gated_v1
  loss_mode: bimanual_gated
  gate_tau: 0.4
  gate_temperature: 0.1
  aux_decode_weight: 4.0
  aux_decode_steps: 10
  aux_decode_solver: fireflow
  low_gate_safety_weight: 0.5
  low_gate_safety_margin: 0.03
  rank_low_gate_threshold: 0.3
  rank_high_gate_threshold: 0.7
  rank_weight: 2.0
  rank_margin: 0.0
  repair_weight: 0.0
  repair_margin: 0.0
```

Copy the remaining Pi0.5-specific schedule/batch fields exactly from `train_pi05_frs.yaml`; do not copy SmolVLA's horizon, action width, raw-image, or ResNet settings.

- [ ] **Step 4: Run schema/config tests and verify GREEN**

Run the Step 2 command. Expected: all selected tests pass.

- [ ] **Step 5: Commit**

```bash
git add train_pi05_frs/utils/bimanual_schema.py train_pi05_frs/configs/train_pi05_frs_bimanual_gated.yaml train_pi05_frs/tools/train_frs.py train_pi05_frs/tests/test_pipeline.py
git commit -m "feat: define pi05 bimanual FRS contract"
```

---

### Task 2: Per-wrist Gate labels from frozen embeddings

**Files:**
- Modify: `train_pi05_frs/utils/data.py:97-129,343-354,598-625`
- Modify: `train_pi05_frs/tests/test_data.py`

**Interfaces:**
- Consumes: current/baseline arrays shaped `[B, 4, D]`.
- Produces: `tactile_change_per_wrist_from_tokens(...) -> np.ndarray[B,2]` and conditioner method `tactile_change_per_wrist_for_cache_indices(...)`.

- [ ] **Step 1: Write failing per-wrist tests**

```python
from train_pi05_frs.utils.data import tactile_change_per_wrist_from_tokens


def test_tactile_change_is_aggregated_per_fixed_wrist_group():
    baseline = np.zeros((1, 4, 2), dtype=np.float32)
    baseline[..., 0] = 1.0
    current = baseline.copy()
    current[:, 2:, :] = np.asarray([0.0, 1.0], dtype=np.float32)
    change = tactile_change_per_wrist_from_tokens(current, baseline)
    np.testing.assert_allclose(change, [[0.0, 1.0]], atol=1e-6)


def test_per_wrist_gate_preserves_batch_and_wrist_axes():
    change = np.asarray([[0.0, 1.0], [0.5, 0.5]], dtype=np.float32)
    gate = gate_weights_from_change(change, tau=0.5, temperature=0.1)
    assert gate.shape == (2, 2)
    assert gate[0, 0] < 0.05 and gate[0, 1] > 0.95
```

- [ ] **Step 2: Verify RED**

Run:

```bash
cd /home/typhon/FRS_Tact/train_pi05_frs
PYTHONPATH="$PWD/src:$(dirname "$PWD")" PYTHONSAFEPATH=1 .venv/bin/python -m pytest tests/test_data.py -k "per_wrist or aggregated" -q
```

Expected: import failure for `tactile_change_per_wrist_from_tokens`.

- [ ] **Step 3: Implement per-wrist changes and conditioner accessors**

Add:

```python
from train_pi05_frs.utils.bimanual_schema import (
    LEFT_WRIST_TOKEN_INDICES, RIGHT_WRIST_TOKEN_INDICES,
)


def tactile_change_per_wrist_from_tokens(
    current_tokens: np.ndarray,
    baseline_tokens: np.ndarray,
) -> np.ndarray:
    current = np.asarray(current_tokens, dtype=np.float32)
    baseline = np.asarray(baseline_tokens, dtype=np.float32)
    if current.shape != baseline.shape or current.ndim != 3 or current.shape[1] != 4:
        raise ValueError("Expected matching [B, four tactile tokens, D]")
    per_token = 1.0 - np.sum(_l2_normalize(current) * _l2_normalize(baseline), axis=-1)
    return np.stack(
        [
            np.mean(per_token[:, LEFT_WRIST_TOKEN_INDICES], axis=1),
            np.mean(per_token[:, RIGHT_WRIST_TOKEN_INDICES], axis=1),
        ],
        axis=1,
    ).astype(np.float32)
```

For both conditioner classes, build the same source-aware baseline batch already used by `tactile_change_for_cache_indices`, then call this new function. Keep scalar accessors unchanged.

- [ ] **Step 4: Verify GREEN and legacy data behavior**

Run:

```bash
PYTHONPATH="$PWD/src:$(dirname "$PWD")" PYTHONSAFEPATH=1 .venv/bin/python -m pytest tests/test_data.py -q
```

Expected: all data tests pass.

- [ ] **Step 5: Commit**

```bash
git add train_pi05_frs/utils/data.py train_pi05_frs/tests/test_data.py
git commit -m "feat: compute per-wrist pi05 Gate labels"
```

---

### Task 3: 20D/32D masked composite flow objective

**Files:**
- Modify: `train_pi05_frs/utils/model.py:22,275-295,362-608,611-742`
- Modify: `train_pi05_frs/tests/test_model.py`

**Interfaces:**
- Produces: `bimanual_composite_endpoint`, `bimanual_mse_per_sample`, `masked_flow_matching_loss_per_sample`, `bimanual_loss_components_per_sample`.
- Extends: `train_step(..., loss_mode="bimanual_gated", gate_weights=[B,2], source_indices=[B])`.

- [ ] **Step 1: Write failing model tests**

```python
from train_pi05_frs.utils.model import (
    bimanual_composite_endpoint,
    bimanual_mse_per_sample,
)


def test_32d_composite_steers_first_20_and_preserves_vla_tail():
    gt = jnp.ones((2, 3, 32))
    vla = jnp.full((2, 3, 32), 2.0)
    gates = jnp.asarray([[1.0, 0.0], [0.0, 1.0]])
    target, effective = bimanual_composite_endpoint(gt, vla, gates)
    np.testing.assert_allclose(target[0, :, :10], 1.0)
    np.testing.assert_allclose(target[0, :, 10:20], 2.0)
    np.testing.assert_allclose(target[..., 20:], 2.0)
    assert effective.shape == (2, 2)


def test_bimanual_mse_ignores_32d_padding_tail():
    left = jnp.zeros((1, 2, 32))
    right = left.at[..., 20:].set(100.0)
    np.testing.assert_allclose(bimanual_mse_per_sample(left, right), [[0.0, 0.0]])


def test_bimanual_composite_rejects_width_below_physical_action():
    with pytest.raises(ValueError, match="at least 20"):
        bimanual_composite_endpoint(
            jnp.zeros((1, 2, 19)), jnp.zeros((1, 2, 19)), jnp.ones((1, 2))
        )
```

- [ ] **Step 2: Verify RED**

Run:

```bash
cd /home/typhon/FRS_Tact/train_pi05_frs
PYTHONPATH="$PWD/src:$(dirname "$PWD")" PYTHONSAFEPATH=1 .venv/bin/python -m pytest tests/test_model.py -k bimanual -q
```

Expected: import failures for the bimanual model helpers.

- [ ] **Step 3: Implement the composite endpoint and physical-dimension metrics**

```python
def bimanual_composite_endpoint(gt_action, predicted_action, gate_weights, *,
                                low_gate_threshold=0.3, high_gate_threshold=0.7):
    if gt_action.ndim != 3 or gt_action.shape != predicted_action.shape:
        raise ValueError("bimanual composite endpoint requires matching actions")
    if gt_action.shape[-1] < STEERED_ACTION_DIM:
        raise ValueError("bimanual composite endpoint requires at least 20 action dimensions")
    if gate_weights.shape != (gt_action.shape[0], 2):
        raise ValueError("bimanual gate_weights must have shape [B, 2]")
    effective = three_region_effective_gate_weights(
        gate_weights,
        low_gate_threshold=low_gate_threshold,
        high_gate_threshold=high_gate_threshold,
    )
    physical_weights = jnp.concatenate(
        [jnp.repeat(effective[:, :1], 10, axis=1),
         jnp.repeat(effective[:, 1:], 10, axis=1)], axis=1
    )[:, None, :]
    physical = physical_weights * gt_action[..., :20] + (
        1.0 - physical_weights
    ) * predicted_action[..., :20]
    target = jnp.concatenate([physical, predicted_action[..., 20:]], axis=-1)
    return target, effective


def bimanual_mse_per_sample(left, right):
    squared = jnp.square(left[..., :STEERED_ACTION_DIM] - right[..., :STEERED_ACTION_DIM])
    return jnp.stack(
        [jnp.mean(squared[..., :10], axis=(1, 2)),
         jnp.mean(squared[..., 10:20], axis=(1, 2))], axis=1
    )
```

Implement masked FM by computing the normal velocity residual and averaging only `[..., :20]`. Implement the current SmolVLA per-wrist decode/safety/rank/repair equations with Pi0.5 namespaces and source indices. Add `composite_fm` to training component names and a `bimanual_gated` branch that performs one composite FM call. Preserve every scalar branch byte-for-byte except necessary shared signatures.

- [ ] **Step 4: Add train-step and active-group tests, then verify GREEN**

Add tests showing low/high wrists receive independent gradients, an inactive wrist does not dilute an active wrist, Gate non-finiteness is rejected before JIT, and scalar `gated` outputs remain unchanged. Then run:

```bash
PYTHONPATH="$PWD/src:$(dirname "$PWD")" PYTHONSAFEPATH=1 .venv/bin/python -m pytest tests/test_model.py -q
```

Expected: all model tests pass.

- [ ] **Step 5: Commit**

```bash
git add train_pi05_frs/utils/model.py train_pi05_frs/tests/test_model.py
git commit -m "feat: add masked bimanual pi05 FRS objective"
```

---

### Task 4: Trainer, history, checkpoint selection, and resume contract

**Files:**
- Modify: `train_pi05_frs/train.py:16-59,123-238,277-435,483-793`
- Modify: `train_pi05_frs/tools/train_frs.py:1209-1289`
- Modify: `train_pi05_frs/tests/test_model.py`
- Modify: `train_pi05_frs/tests/test_pipeline.py`

**Interfaces:**
- Consumes: schema/data/model interfaces from Tasks 1–3.
- Produces: objective-v2 metadata, bimanual history rows, bimanual best key, strict resume validation.

- [ ] **Step 1: Write failing trainer integration tests**

Add tests that assert:

```python
def test_bimanual_resume_requires_exact_objective_metadata():
    valid = {"extra_metadata": bimanual_objective_metadata(action_dim=32)}
    _validate_resume_loss_objective(valid, loss_mode="bimanual_gated", action_dim=32)
    invalid = copy.deepcopy(valid)
    invalid["extra_metadata"]["steered_action_dim"] = 19
    with pytest.raises(ValueError, match="steered_action_dim"):
        _validate_resume_loss_objective(
            invalid, loss_mode="bimanual_gated", action_dim=32
        )


def test_pipeline_forwards_bimanual_mode_without_gate_lambda(monkeypatch, valid_config):
    config = copy.deepcopy(valid_config)
    config["frs_training"].pop("gate_lambda", None)
    config["frs_training"]["loss_mode"] = "bimanual_gated"
    captured = {}
    monkeypatch.setattr(train_tool, "train_decoder", lambda **kwargs: captured.update(kwargs))
    train_tool.train_from_config(config)
    assert captured["loss_mode"] == "bimanual_gated"
```

- [ ] **Step 2: Verify RED**

Run:

```bash
cd /home/typhon/FRS_Tact/train_pi05_frs
PYTHONPATH="$PWD/src:$(dirname "$PWD")" PYTHONSAFEPATH=1 .venv/bin/python -m pytest tests/test_model.py tests/test_pipeline.py -k bimanual -q
```

Expected: failures because the trainer does not recognize the mode or objective metadata.

- [ ] **Step 3: Wire bimanual training and metadata**

Implement:

```python
def _validate_resume_loss_objective(checkpoint_metadata, *, loss_mode, action_dim):
    if loss_mode != BIMANUAL_LOSS_MODE:
        return
    extra = checkpoint_metadata.get("extra_metadata")
    if not isinstance(extra, Mapping):
        raise ValueError("resume checkpoint is missing bimanual objective metadata")
    validate_bimanual_objective_metadata(extra, action_dim=action_dim)
```

In each training batch:

```python
if loss_mode == BIMANUAL_LOSS_MODE:
    current = train_batches.gate_current_tokens(indices, tactile_seq)
    change = train_batches.tactile_change_per_wrist_for_cache_indices(indices, current)
    gate_w = gate_weights_from_change(change, tau=gate_tau, temperature=gate_temperature)
elif loss_mode == "gated":
    gate_w = train_batches.gate_weights_for_cache_indices(
        indices, tau=gate_tau, temperature=gate_temperature
    )
else:
    gate_w = np.ones((batch_n,), dtype=np.float32)
```

Pass source indices to `train_step`; add total/component/left/right Gate history fields; append objective metadata from `bimanual_objective_metadata(action_dim=model.config.action_dim)`; validate it before optimizer restore or output creation. Implement the source/wrist worst/min five-element selection key used by the source project. Preserve Pi0.5 transactional checkpoint calls.

- [ ] **Step 4: Verify GREEN and full trainer/pipeline files**

Run:

```bash
PYTHONPATH="$PWD/src:$(dirname "$PWD")" PYTHONSAFEPATH=1 .venv/bin/python -m pytest tests/test_model.py tests/test_pipeline.py -q
```

Expected: all selected tests pass.

- [ ] **Step 5: Commit**

```bash
git add train_pi05_frs/train.py train_pi05_frs/tools/train_frs.py train_pi05_frs/tests/test_model.py train_pi05_frs/tests/test_pipeline.py
git commit -m "feat: wire bimanual pi05 FRS training"
```

---

### Task 5: Per-wrist validation, source rollups, and evaluation outputs

**Files:**
- Create: `train_pi05_frs/utils/bimanual_metrics.py`
- Create: `train_pi05_frs/tests/test_bimanual_metrics.py`
- Modify: `train_pi05_frs/utils/metrics.py:19-280`
- Modify: `train_pi05_frs/evaluate.py:20-210`
- Modify: `train_pi05_frs/tests/test_model.py`

**Interfaces:**
- Produces: quadrant metrics/counts, per-wrist sample arrays, composite FM, and source-aware rollups.
- Consumes: retained decoded/GT/VLA actions and `[N,2]` Gates.

- [ ] **Step 1: Write failing pure-metric tests**

Create `tests/test_bimanual_metrics.py` with:

```python
import numpy as np
import pytest
from train_pi05_frs.utils.bimanual_metrics import (
    bimanual_gate_region_counts, bimanual_quadrant_metrics,
)


def test_quadrants_keep_left_and_right_independent():
    result = bimanual_quadrant_metrics(
        mse_gt=np.asarray([[0.25, 4.0]]),
        mse_vla=np.asarray([[1.0, 0.04]]),
        mse_vla_gt=np.asarray([[1.0, 4.0]]),
        gate_weights=np.asarray([[0.8, 0.2]]),
        low_threshold=0.3,
        high_threshold=0.7,
    )
    assert result["high_low"]["n"] == 1
    assert result["high_low"]["left"]["relative_gt_error"] == pytest.approx(0.25)
    assert result["high_low"]["right"]["vla_preserve_ratio"] == pytest.approx(0.01)


def test_region_counts_include_mid_without_forcing_quadrant():
    counts = bimanual_gate_region_counts(
        np.asarray([[0.0, 0.0], [0.8, 0.2], [0.5, 0.9]]),
        low_threshold=0.3,
        high_threshold=0.7,
    )
    np.testing.assert_array_equal(counts, [[1, 0, 0], [0, 0, 1], [1, 0, 0]])
```

- [ ] **Step 2: Verify RED**

Run:

```bash
cd /home/typhon/FRS_Tact/train_pi05_frs
PYTHONPATH="$PWD/src:$(dirname "$PWD")" PYTHONSAFEPATH=1 .venv/bin/python -m pytest tests/test_bimanual_metrics.py -q
```

Expected: import failure because the metrics module does not exist.

- [ ] **Step 3: Implement metrics and evaluation propagation**

Transfer the complete pure-NumPy implementation from
`train_smolvla_frs/utils/bimanual_metrics.py` into the Pi0.5 namespace. Its public
functions are `bimanual_quadrant_metrics`, `flatten_bimanual_quadrant_metrics`,
and `bimanual_gate_region_counts`. Keep fixed quadrant names `low_low`,
`high_low`, `low_high`, `high_high`, inclusive low/high boundaries, and NaN
metrics for empty groups. In `evaluate_split`, compute per-wrist MSE with
`bimanual_mse_per_sample`, preserve source/sample indices, return composite FM
and retained actions, and aggregate each dataset before computing worst/min
rollups. In `evaluate.py`, emit bimanual JSON fields and per-sample columns
without removing legacy columns.

- [ ] **Step 4: Verify GREEN and evaluation regression**

Run:

```bash
PYTHONPATH="$PWD/src:$(dirname "$PWD")" PYTHONSAFEPATH=1 .venv/bin/python -m pytest tests/test_bimanual_metrics.py tests/test_model.py -q
```

Expected: all selected tests pass.

- [ ] **Step 5: Commit**

```bash
git add train_pi05_frs/utils/bimanual_metrics.py train_pi05_frs/utils/metrics.py train_pi05_frs/evaluate.py train_pi05_frs/tests/test_bimanual_metrics.py train_pi05_frs/tests/test_model.py
git commit -m "feat: evaluate pi05 FRS per wrist"
```

---

### Task 6: Bimanual history and diagnostics

**Files:**
- Create: `train_pi05_frs/utils/bimanual_visualize.py`
- Create: `train_pi05_frs/tests/test_bimanual_visualize.py`
- Modify: `train_pi05_frs/utils/history_plot.py:1-270`
- Modify: `train_pi05_frs/plot_history.py`
- Modify: `train_pi05_frs/train.py`
- Modify: `train_pi05_frs/evaluate.py`

**Interfaces:**
- Produces: `training_overview.png`, `bimanual_behavior.png`, `gate_diagnostics.png`, `bimanual_action_examples.png`.
- Consumes: new history columns and retained validation predictions from Task 5.

- [ ] **Step 1: Write failing diagnostic smoke tests**

```python
def test_bimanual_plot_bundle_writes_stable_filenames(tmp_path):
    history = write_bimanual_history_fixture(tmp_path / "history.csv")
    result = make_bimanual_evaluation_result(action_dim=32)
    paths = plot_bimanual_diagnostics(history, result, output_dir=tmp_path)
    assert {path.name for path in paths} == {
        "training_overview.png", "bimanual_behavior.png",
        "gate_diagnostics.png", "bimanual_action_examples.png",
    }
    assert all(path.stat().st_size > 0 for path in paths)


def test_legacy_history_still_writes_training_curves(tmp_path):
    history = write_legacy_history_fixture(tmp_path / "history.csv")
    output = plot_training_history(history)
    assert output.name == "training_curves.png"
    assert output.stat().st_size > 0
```

- [ ] **Step 2: Verify RED**

Run:

```bash
cd /home/typhon/FRS_Tact/train_pi05_frs
PYTHONPATH="$PWD/src:$(dirname "$PWD")" PYTHONSAFEPATH=1 .venv/bin/python -m pytest tests/test_bimanual_visualize.py -q
```

Expected: import/fixture failures because bimanual visualization support is absent.

- [ ] **Step 3: Implement diagnostics**

Transfer the complete plotting helpers from
`train_smolvla_frs/utils/bimanual_visualize.py`, changing imports to the Pi0.5
namespace and keeping these public entry points:
`plot_bimanual_training_overview`, `plot_bimanual_behavior`,
`plot_bimanual_gate_diagnostics`, and `plot_bimanual_action_examples`. Add a
`plot_bimanual_diagnostics` coordinator that calls them with the four exact
filenames asserted in Step 1. Preserve the source project's empty-quadrant
annotations and median/worst mixed-quadrant sample selection. Use only the
first 20 physical action dimensions and label grippers as dimensions 9 and 19.
Keep legacy history parsing tolerant of absent bimanual columns.
Training/evaluation should catch plotting errors, print warnings, and never
invalidate a valid checkpoint.

- [ ] **Step 4: Verify GREEN**

Run:

```bash
MPLBACKEND=Agg PYTHONPATH="$PWD/src:$(dirname "$PWD")" PYTHONSAFEPATH=1 .venv/bin/python -m pytest tests/test_bimanual_visualize.py -q
```

Expected: all visualization tests pass and generated PNGs are non-empty.

- [ ] **Step 5: Commit**

```bash
git add train_pi05_frs/utils/bimanual_visualize.py train_pi05_frs/utils/history_plot.py train_pi05_frs/plot_history.py train_pi05_frs/train.py train_pi05_frs/evaluate.py train_pi05_frs/tests/test_bimanual_visualize.py
git commit -m "feat: add pi05 bimanual FRS diagnostics"
```

---

### Task 7: Deployment objective-v2 acceptance

**Files:**
- Modify: `deploy_pi05/frs_runtime.py:310-350`
- Modify: `tests/jax/test_frs_deployment.py:2287-2388`
- Test: `train_pi05_frs/tests/test_deployment_checkpoint_compatibility.py`

**Interfaces:**
- Consumes: training checkpoint `extra_metadata` and decoder action width.
- Produces: strict acceptance for valid bimanual metadata; no inference API changes.

- [ ] **Step 1: Write failing deployment contract tests**

Extend `tests/jax/test_frs_deployment.py` so its deployment contract fixture uses
a 32D decoder/policy and valid objective-v2 metadata. Mutate each of
`action_slices`, `wrist_token_indices`, `steered_action_dim`, and
`padded_tail_policy` in subtests and assert the runtime rejects it before
inference.

The core assertions are:

```python
runtime._validate_contract(source_sample_steps=10)
for field, invalid in (
    ("steered_action_dim", 19),
    ("padded_tail_policy", "train_gt"),
):
    extra = bimanual_objective_metadata(action_dim=32)
    extra[field] = invalid
    runtime.metadata["extra_metadata"] = extra
    with pytest.raises(ValueError, match=field):
        runtime._validate_contract(source_sample_steps=10)
```

- [ ] **Step 2: Verify RED**

Run:

```bash
cd /home/typhon/FRS_Tact
.venv/bin/python -m pytest tests/jax/test_frs_deployment.py -k "contract and bimanual" -q
```

Expected: valid bimanual metadata is rejected because runtime requires scalar
`gated` or does not validate the complete 32D metadata contract.

- [ ] **Step 3: Implement metadata-aware runtime validation**

Add a dependency-light validator in `deploy_pi05/frs_runtime.py`:

```python
def _validate_loss_contract(extra: Mapping[str, Any], *, action_dim: int) -> None:
    loss_mode = extra.get("loss_mode")
    if loss_mode == "gated":
        return
    if loss_mode != "bimanual_gated":
        raise ValueError(f"unsupported FRS loss_mode: {loss_mode!r}")
    expected = {
        "loss_objective_version": 2,
        "loss_weighting_version": 7,
        "action_dim": action_dim,
        "steered_action_dim": 20,
        "action_slices": {"left": [0, 10], "right": [10, 20]},
        "wrist_token_indices": {"left": [0, 1], "right": [2, 3]},
        "padded_tail_policy": "vla_endpoint_masked",
    }
    for field, value in expected.items():
        _require_equal(extra.get(field), value, field)
```

Replace the scalar-only equality check with this validator. Do not import training modules into deployment, change action truncation, or alter decoder construction.

- [ ] **Step 4: Verify GREEN and checkpoint cross-runtime loading**

Run:

```bash
cd /home/typhon/FRS_Tact/train_pi05_frs
DEPLOY_PI05_PYTHON=/home/typhon/FRS_Tact/deploy_pi05/.venv/bin/python \
PYTHONPATH="$PWD/src:$(dirname "$PWD")" PYTHONSAFEPATH=1 \
.venv/bin/python -m pytest tests/test_deployment_checkpoint_compatibility.py -q
```

Then run the affected root deployment test file. Expected: all selected tests pass.

- [ ] **Step 5: Commit**

```bash
git add deploy_pi05/frs_runtime.py tests/jax/test_frs_deployment.py
git commit -m "feat: accept bimanual pi05 FRS checkpoints"
```

---

### Task 8: Documentation and full regression verification

**Files:**
- Modify: `train_pi05_frs/README.md`
- Modify: `train_pi05_frs/source_manifest.sha256` only if its documented boundary requires hashes for changed vendored/source files.
- Test: complete `train_pi05_frs/tests` and affected repository tests.

**Interfaces:**
- Documents: launcher/config usage, 32D/20D semantics, output artifacts, resume rules, and deployment handoff.

- [ ] **Step 1: Update README with exact commands and semantics**

Document:

```bash
bash train_pi05_frs/scripts/start_frs_pi05_train.sh \
  train_pi05_frs/configs/train_pi05_frs_bimanual_gated.yaml
```

State explicitly that the first 20 dimensions are optimized per wrist, the 12D padded tail is excluded from bimanual loss/metrics, Gate is not a decoder input, old scalar config remains available, old checkpoints cannot resume as bimanual, and the four diagnostic filenames are stable.

- [ ] **Step 2: Run formatting/static checks**

```bash
cd /home/typhon/FRS_Tact
git diff --check
bash -n train_pi05_frs/scripts/setup_env.sh train_pi05_frs/scripts/start_frs_pi05_train.sh
train_pi05_frs/.venv/bin/python -m compileall -q train_pi05_frs deploy_pi05/frs_runtime.py
```

Expected: every command exits 0.

- [ ] **Step 3: Run the isolated Pi0.5 training suite**

```bash
cd /home/typhon/FRS_Tact/train_pi05_frs
MPLBACKEND=Agg PYTHONPATH="$PWD/src:$(dirname "$PWD")" PYTHONSAFEPATH=1 \
.venv/bin/python -m pytest tests -q
```

Expected: 0 failures and 0 collection errors.

- [ ] **Step 4: Run affected repository/deployment tests**

```bash
cd /home/typhon/FRS_Tact
.venv/bin/python -m pytest \
  tests/jax/test_frs_deployment.py \
  tests/test_train_pi05_frs_project_boundary.py \
  tests/test_deploy_pi05_deployment_only.py -q
```

Expected: 0 failures.

- [ ] **Step 5: Review requirements and working tree**

Verify each global constraint against the diff, inspect `git status --short`, and confirm unrelated pre-existing modifications were not staged or changed.

- [ ] **Step 6: Commit documentation/final adjustments**

```bash
git add train_pi05_frs/README.md train_pi05_frs/source_manifest.sha256
git commit -m "docs: document pi05 bimanual FRS training"
```

Omit `source_manifest.sha256` from `git add` when its project-boundary tests show no hash update is required.
