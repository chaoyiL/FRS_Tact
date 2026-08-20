# Task 6 — Bimanual history and diagnostics

## RED

Added `train_pi05_frs/tests/test_bimanual_visualize.py` with the required
stable-filename bundle smoke test and legacy-history compatibility test.

Command:

```bash
MPLBACKEND=Agg PYTHONPATH="$PWD/src:$(dirname "$PWD")" PYTHONSAFEPATH=1 \
  .venv/bin/python -m pytest tests/test_bimanual_visualize.py -q
```

Observed expected failure: `ModuleNotFoundError` for
`train_pi05_frs.utils.bimanual_visualize`.

## GREEN

Implemented Pi0.5 bimanual dashboard helpers and a stable four-file
coordinator. Action examples preserve retained model-width arrays but render
only physical dimensions `0:20`, with grippers at 9 and 19. Empty quadrants
render annotations. Legacy training histories still render
`training_curves.png`.

Training and evaluation plotting is best-effort: plot failures print warnings
and do not prevent checkpoint persistence.

Verification:

```text
tests/test_bimanual_visualize.py tests/test_bimanual_metrics.py: 4 passed
tests/test_model.py -k bimanual: 14 passed, 26 deselected
train_pi05_frs full pytest suite: executed once with MPLBACKEND=Agg
compileall and git diff --check: passed
```

## Full-suite verification (post-commit)

Command:

```bash
cd /home/typhon/FRS_Tact/.worktrees/pi05-bimanual-frs/train_pi05_frs
MPLBACKEND=Agg PYTHONPATH="$PWD/src:$(dirname "$PWD")" PYTHONSAFEPATH=1 \
  .venv/bin/python -m pytest -q
```

Exit code: `0`

Final summary:

```text
347 passed, 18 subtests passed in 71.56s (0:01:11)
```

## Review follow-up — RED

Added integration coverage for trainer-produced bimanual histories, retained
action policy, and independent dashboard attempts.

```text
test_diagnostic_bundle_attempts_other_plots_after_one_plot_fails:
  FAILED because plot_bimanual_diagnostics stopped at behavior failure.

test_bimanual_trainer_validation_writes_finite_source_wrist_selection:
  FAILED with KeyError: checkpoint_selection_feasible.
```

## Review follow-up — GREEN

The trainer now records the full stable `val_quadrant_*` schema and
`checkpoint_selection_feasible` before writing validation history. It retains
actions for bimanual diagnostics when `write_plots` is enabled. Bimanual
standalone evaluation retains actions for plots without writing
`predictions.npz` unless `save_predictions` is requested. The coordinator
attempts every sibling plot independently.

Focused command:

```bash
cd /home/typhon/FRS_Tact/.worktrees/pi05-bimanual-frs/train_pi05_frs
MPLBACKEND=Agg PYTHONPATH="$PWD/src:$(dirname "$PWD")" PYTHONSAFEPATH=1 \
  .venv/bin/python -m pytest tests/test_bimanual_visualize.py tests/test_model.py -k bimanual -q
```

```text
19 passed, 26 deselected in 17.38s
```

Full-suite command:

```bash
cd /home/typhon/FRS_Tact/.worktrees/pi05-bimanual-frs/train_pi05_frs
MPLBACKEND=Agg PYTHONPATH="$PWD/src:$(dirname "$PWD")" PYTHONSAFEPATH=1 \
  .venv/bin/python -m pytest -q
```

Exit code: `0`

```text
350 passed, 18 subtests passed in 72.58s (0:01:12)
```
