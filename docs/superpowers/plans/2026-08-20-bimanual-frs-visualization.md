# Bimanual FRS Visualization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add intuitive convergence, per-wrist four-quadrant behavior, Gate diagnostics, and action-example plots for `bimanual_gated` without changing its loss or legacy outputs.

**Architecture:** Put pure NumPy four-quadrant aggregation in a focused `bimanual_metrics.py`, keep model evaluation in `metrics.py`, and put all new Matplotlib rendering in `bimanual_visualize.py`. `evaluate_split` returns the already-decoded actions and references when plotting is requested; the training loop writes JSON-friendly quadrant scalars to `history.csv` and refreshes stable plot files after each validation epoch.

**Tech Stack:** Python 3.12, NumPy, JAX, Matplotlib Agg, CSV/JSON, pytest.

## Global Constraints

- Preserve `training_curves.png`, standalone evaluation JSON/CSV, and all legacy plot names and meanings.
- Generate new plots only for `loss_mode: bimanual_gated`.
- Keep action slices fixed at left `[0, 10)` and right `[10, 20)`; grippers are dimensions 9 and 19.
- Use configured low/high Gate thresholds, defaulting to 0.3 and 0.7.
- Do not modify loss, Gate computation, optimizer, checkpoint selection, or deployment inference.
- Do not run a second ODE decode for plotting.
- Treat fewer than 20 samples as a visualization warning only; it must not affect checkpoint selection.
- Preserve the user's uncommitted `train_smolvla_frs/configs/train_frs_bimanual_gated.yaml` changes.

---

## File Structure

- Create `train_smolvla_frs/utils/bimanual_metrics.py`: pure quadrant masks, per-wrist aggregates, flattening, and 3×3 Gate-region counts.
- Create `train_smolvla_frs/utils/bimanual_visualize.py`: the four new PNG writers; no model execution.
- Modify `train_smolvla_frs/utils/metrics.py`: attach quadrant results and optionally retained GT/VLA/decoded arrays to `EvaluationResult`.
- Modify `train_smolvla_frs/train_frs.py`: add stable history fields and call the new plot writers after validation.
- Modify `train_smolvla_frs/evaluate.py`: expose quadrant metrics and per-wrist columns in standalone JSON/CSV and call new plots.
- Modify `train_smolvla_frs/utils/history_plot.py`: make new history fields parseable while leaving the legacy plot unchanged.
- Create `tests/train_frs/test_bimanual_metrics.py`: exact aggregation tests.
- Create `tests/train_frs/test_bimanual_visualize.py`: panel, label, empty-group, and no-second-decode tests.
- Modify `tests/train_frs/test_evaluate.py`, `tests/train_frs/test_history_plot.py`, and `tests/train_frs/test_model.py`: evaluation/history/training integration coverage.
- Modify `train_smolvla_frs/README.md`: document generated files and interpretation.

---

### Task 1: Pure four-quadrant and joint-Gate metrics

**Files:**
- Create: `train_smolvla_frs/utils/bimanual_metrics.py`
- Create: `tests/train_frs/test_bimanual_metrics.py`

**Interfaces:**
- Consumes: one-dimensional per-sample left/right Gate and per-wrist MSE arrays.
- Produces: `bimanual_quadrant_metrics(*, mse_gt, mse_vla, mse_vla_gt, gate_weights, low_threshold, high_threshold, ranking_margin=0.0) -> dict[str, dict[str, object]]`, `flatten_bimanual_quadrant_metrics(metrics, *, prefix="val_quadrant") -> dict[str, float | int]`, and `bimanual_gate_region_counts(gate_weights, *, low_threshold, high_threshold) -> np.ndarray`.

- [ ] **Step 1: Write failing exact-formula and boundary tests**

```python
def test_quadrants_keep_high_left_and_low_right_independent():
    result = bimanual_quadrant_metrics(
        mse_gt=np.asarray([[0.25, 4.0], [1.0, 1.0]]),
        mse_vla=np.asarray([[1.0, 0.04], [0.5, 0.5]]),
        mse_vla_gt=np.asarray([[1.0, 4.0], [1.0, 1.0]]),
        gate_weights=np.asarray([[0.7, 0.3], [0.5, 0.5]]),
        low_threshold=0.3,
        high_threshold=0.7,
    )
    assert result["high_low"]["n"] == 1
    assert result["high_low"]["left"]["relative_gt_error"] == pytest.approx(0.25)
    assert result["high_low"]["right"]["vla_preserve_ratio"] == pytest.approx(0.01)
    assert result["low_high"]["n"] == 0


def test_joint_region_counts_include_mid_without_forcing_a_quadrant():
    counts = bimanual_gate_region_counts(
        np.asarray([[0.0, 0.0], [0.8, 0.2], [0.5, 0.9]]),
        low_threshold=0.3,
        high_threshold=0.7,
    )
    np.testing.assert_array_equal(counts, [[1, 0, 0], [0, 0, 1], [1, 0, 0]])
```

- [ ] **Step 2: Run tests and confirm RED**

Run: `uv run --no-sync pytest -q tests/train_frs/test_bimanual_metrics.py`

Expected: collection fails because `train_smolvla_frs.utils.bimanual_metrics` does not exist.

- [ ] **Step 3: Implement validated JSON-friendly aggregation**

```python
BIMANUAL_QUADRANTS = ("low_low", "high_low", "low_high", "high_high")
BIMANUAL_WRISTS = ("left", "right")


def bimanual_quadrant_metrics(*, mse_gt, mse_vla, mse_vla_gt, gate_weights,
                              low_threshold, high_threshold, ranking_margin=0.0):
    arrays = [np.asarray(value, dtype=np.float64) for value in
              (mse_gt, mse_vla, mse_vla_gt, gate_weights)]
    if any(value.ndim != 2 or value.shape[1] != 2 for value in arrays):
        raise ValueError("bimanual quadrant inputs must all have shape [N, 2]")
    if len({value.shape for value in arrays}) != 1 or any(not np.all(np.isfinite(value)) for value in arrays):
        raise ValueError("bimanual quadrant inputs must have matching finite values")
    gt, vla, baseline, gates = arrays
    low = gates <= float(low_threshold)
    high = gates >= float(high_threshold)
    masks = {
        "low_low": low[:, 0] & low[:, 1],
        "high_low": high[:, 0] & low[:, 1],
        "low_high": low[:, 0] & high[:, 1],
        "high_high": high[:, 0] & high[:, 1],
    }
    output = {}
    for name, mask in masks.items():
        group = {"n": int(mask.sum())}
        for wrist_index, wrist in enumerate(BIMANUAL_WRISTS):
            if not np.any(mask):
                group[wrist] = {key: float("nan") for key in (
                    "mse_gt", "mse_vla", "mse_vla_gt", "gt_gain",
                    "relative_gt_error", "vla_preserve_ratio", "rank_satisfied_frac")}
                continue
            mean_gt = float(np.mean(gt[mask, wrist_index]))
            mean_vla = float(np.mean(vla[mask, wrist_index]))
            mean_baseline = float(np.mean(baseline[mask, wrist_index]))
            group[wrist] = {
                "mse_gt": mean_gt,
                "mse_vla": mean_vla,
                "mse_vla_gt": mean_baseline,
                "gt_gain": mean_baseline - mean_gt,
                "relative_gt_error": mean_gt / max(mean_baseline, 1e-8),
                "vla_preserve_ratio": mean_vla / max(mean_baseline, 1e-8),
                "rank_satisfied_frac": float(np.mean(
                    gt[mask, wrist_index] + float(ranking_margin) <= vla[mask, wrist_index]
                )),
            }
        output[name] = group
    return output
```

Implement `flatten_bimanual_quadrant_metrics` with keys such as `val_quadrant_high_low_n` and `val_quadrant_high_low_vla_preserve_ratio_right`. Implement 3×3 counts with region order `(low, mid, high)` for each axis.

- [ ] **Step 4: Run focused tests and confirm GREEN**

Run: `uv run --no-sync pytest -q tests/train_frs/test_bimanual_metrics.py`

Expected: all tests pass, including non-finite, shape mismatch, threshold inclusivity, empty quadrants, flattening, and joint counts.

- [ ] **Step 5: Commit Task 1**

```bash
git add train_smolvla_frs/utils/bimanual_metrics.py tests/train_frs/test_bimanual_metrics.py
git commit -m "feat: add bimanual quadrant metrics"
```

---

### Task 2: Thread quadrant data through evaluation and standalone artifacts

**Files:**
- Modify: `train_smolvla_frs/utils/metrics.py`
- Modify: `train_smolvla_frs/evaluate.py`
- Modify: `tests/train_frs/test_evaluate.py`

**Interfaces:**
- Consumes: Task 1 aggregation functions and arrays already produced by `evaluate_split`.
- Produces: `EvaluationResult.bimanual_quadrants`, `.bimanual_gate_region_counts`, `.gt_actions`, and `.vla_actions`; JSON `bimanual_quadrants`; explicit per-sample wrist baseline and Gate-region columns.

- [ ] **Step 1: Extend the existing evaluation test with failing assertions**

```python
assert result.bimanual_quadrants["high_low"]["n"] == 1
assert result.bimanual_gate_region_counts.shape == (3, 3)

result_with_actions = evaluate_split(
    object(),
    FakeConditioner(),
    split="val",
    batch_size=2,
    num_steps=1,
    keep_predictions=True,
    loss_mode="bimanual_gated",
    gate_tau=0.5,
    gate_temperature=0.1,
)
np.testing.assert_allclose(result_with_actions.gt_actions, gt_action)
np.testing.assert_allclose(result_with_actions.vla_actions, predicted_action)
np.testing.assert_allclose(result_with_actions.predictions, prediction)
```

Extend `test_checkpoint_evaluation_tracks_gate_only_for_gated_loss_mode` to require:

```python
assert "bimanual_quadrants" in written_metrics
assert rows[0]["mse_vla_gt_left"] != ""
assert rows[0]["gate_region_left"] in {"low", "mid", "high"}
assert rows[0]["gate_region_right"] in {"low", "mid", "high"}
assert rows[0]["bimanual_quadrant"] in {"", "low_low", "high_low", "low_high", "high_high"}
```

- [ ] **Step 2: Run focused tests and confirm RED**

Run: `uv run --no-sync pytest -q tests/train_frs/test_evaluate.py -k bimanual`

Expected: failures report missing `EvaluationResult` fields and standalone columns.

- [ ] **Step 3: Add optional result fields and compute them once**

Add to `EvaluationResult`:

```python
bimanual_quadrants: dict[str, dict[str, object]] | None = None
bimanual_gate_region_counts: np.ndarray | None = None
gt_actions: np.ndarray | None = None
vla_actions: np.ndarray | None = None
```

Accumulate GT and VLA arrays only when `keep_predictions` is true. After concatenating per-wrist arrays, call Task 1 exactly once and return the nested metrics and joint counts. Do not alter legacy scalar-Gate fields.

- [ ] **Step 4: Extend standalone JSON/CSV without renaming old fields**

In `evaluate.py`, write:

```python
if result.bimanual_quadrants is not None:
    metrics["bimanual_quadrants"] = result.bimanual_quadrants
    metrics["bimanual_gate_region_counts"] = result.bimanual_gate_region_counts.tolist()
```

Add per-sample columns `mse_vla_gt_left`, `mse_vla_gt_right`, `gate_region_left`, `gate_region_right`, and `bimanual_quadrant`. Leave the legacy `gate_region` and `gate_bin` empty for bimanual samples.

- [ ] **Step 5: Run evaluation tests and confirm GREEN**

Run: `uv run --no-sync pytest -q tests/train_frs/test_evaluate.py`

Expected: all evaluation tests pass; legacy scalar-Gate JSON/CSV assertions remain unchanged.

- [ ] **Step 6: Commit Task 2**

```bash
git add train_smolvla_frs/utils/metrics.py train_smolvla_frs/evaluate.py tests/train_frs/test_evaluate.py
git commit -m "feat: expose bimanual evaluation quadrants"
```

---

### Task 3: Persist bimanual quadrant history during training

**Files:**
- Modify: `train_smolvla_frs/train_frs.py`
- Modify: `train_smolvla_frs/utils/history_plot.py`
- Modify: `tests/train_frs/test_model.py`
- Modify: `tests/train_frs/test_history_plot.py`

**Interfaces:**
- Consumes: `flatten_bimanual_quadrant_metrics` and Task 2 result fields.
- Produces: stable `history.csv` columns for every quadrant/wrist metric and retained validation arrays only when plots are enabled.

- [ ] **Step 1: Add failing history and training-entry assertions**

Add a bimanual mocked-validation test that asserts:

```python
assert row["val_quadrant_high_low_n"] == "2"
assert float(row["val_quadrant_high_low_relative_gt_error_left"]) == pytest.approx(0.5)
assert float(row["val_quadrant_high_low_vla_preserve_ratio_right"]) == pytest.approx(0.1)
```

Patch `evaluate_split` and assert its call receives:

```python
assert kwargs["keep_predictions"] is True  # write_plots + bimanual only
```

Also assert legacy `gated` training still passes `keep_predictions=False`.

- [ ] **Step 2: Run tests and confirm RED**

Run: `uv run --no-sync pytest -q tests/train_frs/test_model.py tests/train_frs/test_history_plot.py -k 'bimanual or legacy_history'`

Expected: missing quadrant history fields and wrong `keep_predictions` value.

- [ ] **Step 3: Define stable quadrant history fields**

Generate fields from shared constants rather than hand-writing names:

```python
QUADRANT_HISTORY_METRICS = (
    "mse_gt", "mse_vla", "mse_vla_gt", "gt_gain",
    "relative_gt_error", "vla_preserve_ratio", "rank_satisfied_frac",
)
history_fields.extend(f"val_quadrant_{quadrant}_n" for quadrant in BIMANUAL_QUADRANTS)
history_fields.extend(
    f"val_quadrant_{quadrant}_{metric}_{wrist}"
    for quadrant in BIMANUAL_QUADRANTS
    for wrist in BIMANUAL_WRISTS
    for metric in QUADRANT_HISTORY_METRICS
)
```

Add the same fields to `history_plot.HISTORY_FIELDS` so old and new CSV parsing remains tolerant.

- [ ] **Step 4: Flatten evaluation metrics and retain actions only for plots**

Use:

```python
keep_validation_actions = bool(write_plots and loss_mode == BIMANUAL_LOSS_MODE)
if validation.bimanual_quadrants is not None:
    metrics.update(flatten_bimanual_quadrant_metrics(validation.bimanual_quadrants))
```

In the existing `evaluate_split` call, replace the literal `keep_predictions=False` argument with `keep_predictions=keep_validation_actions`; keep every other existing argument unchanged.

Do not add quadrant values to `checkpoint_selection_key`.

- [ ] **Step 5: Run focused tests and confirm GREEN**

Run: `uv run --no-sync pytest -q tests/train_frs/test_model.py tests/train_frs/test_history_plot.py -k 'bimanual or legacy_history'`

Expected: new fields persist, old CSV still plots, and legacy training does not retain action tensors.

- [ ] **Step 6: Commit Task 3**

```bash
git add train_smolvla_frs/train_frs.py train_smolvla_frs/utils/history_plot.py tests/train_frs/test_model.py tests/train_frs/test_history_plot.py
git commit -m "feat: record bimanual quadrant history"
```

---

### Task 4: Render convergence overview and four-quadrant behavior

**Files:**
- Create: `train_smolvla_frs/utils/bimanual_visualize.py`
- Create: `tests/train_frs/test_bimanual_visualize.py`
- Modify: `train_smolvla_frs/train_frs.py`

**Interfaces:**
- Consumes: `history.csv`, Gate thresholds, and configured checkpoint feasibility thresholds.
- Produces: `plot_bimanual_training_overview(history_path, *, output_path, min_rank_satisfied=0.8, min_low_safe=0.9) -> Path` and `plot_bimanual_behavior(history_path, *, output_path, min_reliable_samples=20) -> Path`.

- [ ] **Step 1: Write failing panel/label/low-sample tests**

Create a two-epoch history fixture with one `high_low` group having `n=8`. Patch `plt.subplots` and assert:

```python
overview = plot_bimanual_training_overview(history, output_path=root / "training_overview.png")
behavior = plot_bimanual_behavior(history, output_path=root / "bimanual_behavior.png")
assert overview.is_file() and overview.stat().st_size > 0
assert behavior.is_file() and behavior.stat().st_size > 0
assert overview_subplots.call_args.args[:2] == (6, 1)
assert behavior_subplots.call_args.args[:2] == (4, 2)
assert any(
    "样本不足" in text.get_text()
    for axis in behavior_figure.axes
    for text in axis.texts
)
```

Also test a legacy CSV: both functions must raise a specific `ValueError("bimanual history fields are absent")`; the training wrapper catches it and leaves `training_curves.png` unaffected.

- [ ] **Step 2: Run plot tests and confirm RED**

Run: `uv run --no-sync pytest -q tests/train_frs/test_bimanual_visualize.py -k 'overview or behavior'`

Expected: import fails because `bimanual_visualize.py` does not exist.

- [ ] **Step 3: Implement history-only plotting functions**

Use Matplotlib Agg and stable temporary-file replacement. Implement the exact signatures `plot_bimanual_training_overview(history_path: Path, *, output_path: Path, min_rank_satisfied: float = 0.8, min_low_safe: float = 0.9) -> Path` and `plot_bimanual_behavior(history_path: Path, *, output_path: Path, min_reliable_samples: int = 20) -> Path`.

Behavior uses four rows `(low_low, high_low, low_high, high_high)` and two columns `(left, right)`. Plot `relative_gt_error` and `vla_preserve_ratio`; make the expected curve opaque and the reference curve light, add the `y=1` GT baseline, and annotate latest `n` plus raw MSE/gain/rank values.

- [ ] **Step 4: Wire stable refresh after history flush**

Extend `_refresh_training_plot` so bimanual mode writes:

```python
plot_bimanual_training_overview(
    history_path,
    output_path=output_dir / "training_overview.png",
    min_rank_satisfied=best_min_high_gate_rank_satisfied,
    min_low_safe=1.0 - best_max_low_gate_unsafe_frac,
)
plot_bimanual_behavior(history_path, output_path=output_dir / "bimanual_behavior.png")
```

Each new plot call gets its own exception boundary so one broken figure cannot block history or checkpoint saves.

- [ ] **Step 5: Run plot and training integration tests**

Run: `uv run --no-sync pytest -q tests/train_frs/test_bimanual_visualize.py tests/train_frs/test_history_plot.py tests/train_frs/test_model.py -k 'plot or bimanual'`

Expected: new files and panel labels pass; the legacy 5/6-panel `training_curves.png` tests remain green.

- [ ] **Step 6: Commit Task 4**

```bash
git add train_smolvla_frs/utils/bimanual_visualize.py train_smolvla_frs/train_frs.py tests/train_frs/test_bimanual_visualize.py
git commit -m "feat: plot bimanual training behavior"
```

---

### Task 5: Render Gate diagnostics and decoded action examples

**Files:**
- Modify: `train_smolvla_frs/utils/bimanual_visualize.py`
- Modify: `train_smolvla_frs/train_frs.py`
- Modify: `train_smolvla_frs/evaluate.py`
- Modify: `tests/train_frs/test_bimanual_visualize.py`
- Modify: `tests/train_frs/test_evaluate.py`

**Interfaces:**
- Consumes: Task 2 `EvaluationResult` with retained actions and existing pairs metadata.
- Produces: `plot_gate_diagnostics(history_path, *, result, output_path) -> Path` and `plot_bimanual_action_examples(result, pairs, *, output_path) -> Path` without accepting a model or decoder.

- [ ] **Step 1: Write failing Gate heatmap and action-schema tests**

Build a synthetic result containing `high_low` and `low_high`. Assert:

```python
gate_plot = plot_gate_diagnostics(history, result=result, output_path=root / "gate_diagnostics.png")
action_plot = plot_bimanual_action_examples(result, pairs, output_path=root / "bimanual_action_examples.png")
assert gate_plot.is_file() and action_plot.is_file()
assert heatmap.get_array().shape == (3, 3)
assert "gripper 9" in labels
assert "gripper 19" in labels
```

Pass a sentinel object as `model`; the plot API must not accept or call it. Verify selected examples include one median and one maximum low-Gate-wrist `mse_vla` sample for both mixed quadrants.

- [ ] **Step 2: Run focused tests and confirm RED**

Run: `uv run --no-sync pytest -q tests/train_frs/test_bimanual_visualize.py -k 'gate or action'`

Expected: missing plot functions.

- [ ] **Step 3: Implement Gate diagnostics**

Implement the exact signature `plot_gate_diagnostics(history_path: Path, *, result: EvaluationResult, output_path: Path) -> Path`.

Render separate Gate and tactile-change percentile panels, per-wrist low/mid/high counts, and a latest-result 3×3 heatmap with count and percentage text. Do not use a twin axis.

- [ ] **Step 4: Implement schema-aware action examples**

Implement the exact signature `plot_bimanual_action_examples(result: EvaluationResult, pairs: CachedPairs | MultiCachedPairs, *, output_path: Path) -> Path`.

Require retained `predictions`, `gt_actions`, and `vla_actions`; select median and worst-preservation examples in `high_low` and `low_high`; plot left/right per-step distances, a `FRS−VLA` horizon×20 heatmap, and both gripper trajectories. Empty mixed quadrants render a labeled placeholder.

- [ ] **Step 5: Wire plots into training and standalone evaluation**

After a bimanual validation row is flushed, call both functions with the same `EvaluationResult`; never call `decode_actions` from `bimanual_visualize.py`. In standalone evaluation, set effective action retention to:

```python
keep_actions = bool(save_predictions or (write_plots and loss_mode == BIMANUAL_LOSS_MODE))
```

Continue writing `predictions.npz` only when the user requested `save_predictions`; internal plot retention must not silently create it.

- [ ] **Step 6: Run focused no-second-decode and artifact tests**

Run: `uv run --no-sync pytest -q tests/train_frs/test_bimanual_visualize.py tests/train_frs/test_evaluate.py -k 'bimanual or plot'`

Expected: four new files are non-empty, actions cover both slices and grippers, and mocked `decode_actions` call count is unchanged by plotting.

- [ ] **Step 7: Commit Task 5**

```bash
git add train_smolvla_frs/utils/bimanual_visualize.py train_smolvla_frs/train_frs.py train_smolvla_frs/evaluate.py tests/train_frs/test_bimanual_visualize.py tests/train_frs/test_evaluate.py
git commit -m "feat: add bimanual gate and action diagnostics"
```

---

### Task 6: Documentation, compatibility, and full verification

**Files:**
- Modify: `train_smolvla_frs/README.md`
- Modify only if required by test discovery: `tests/train_frs/test_package_layout.py`

**Interfaces:**
- Consumes: all previous tasks.
- Produces: user-facing plot interpretation and a verified compatible branch.

- [ ] **Step 1: Add README output and interpretation table**

Document exact filenames and the two primary ratios:

```markdown
| Plot | Question |
|---|---|
| `training_overview.png` | Is optimization and validation converging? |
| `bimanual_behavior.png` | Does the high-Gate wrist approach GT while the low-Gate wrist preserves VLA? |
| `gate_diagnostics.png` | Are all left/right Gate combinations represented? |
| `bimanual_action_examples.png` | Which joints or grippers drift in mixed-Gate examples? |

`relative_gt_error < 1` means FRS is closer to GT than frozen VLA.
`vla_preserve_ratio -> 0` means the low-Gate wrist preserves the frozen VLA action.
```

- [ ] **Step 2: Run all focused visualization and training tests**

Run: `uv run --no-sync pytest -q tests/train_frs/test_bimanual_metrics.py tests/train_frs/test_bimanual_visualize.py tests/train_frs/test_evaluate.py tests/train_frs/test_history_plot.py tests/train_frs/test_model.py`

Expected: all tests pass.

- [ ] **Step 3: Run the complete FRS regression suite**

Run: `uv run --no-sync pytest -q tests/train_frs tests/flow_decoder`

Expected: all tests pass. If the spawn-only `flow_decoder` test cannot import from a package-layout environment, rerun it with `PYTHONPATH=.:tests` and record both results rather than changing production imports.

- [ ] **Step 4: Run lint and diff integrity checks**

Run: `uv run --no-sync ruff check train_smolvla_frs tests/train_frs`

Expected: no new findings in changed files.

Run: `git diff --check`

Expected: no whitespace errors.

Run: `git status --short`

Expected: the user's pre-existing YAML modification remains unstaged unless the user explicitly requests it; only intended visualization files are part of task commits.

- [ ] **Step 5: Commit documentation**

```bash
git add train_smolvla_frs/README.md tests/train_frs/test_package_layout.py
git commit -m "docs: explain bimanual FRS plots"
```

- [ ] **Step 6: Perform final code review**

Review the full diff against `docs/superpowers/specs/2026-08-20-bimanual-frs-visualization-design.md`. Confirm all four plot files, legacy compatibility, no loss changes, no second decode, and no accidental YAML staging before reporting completion.
