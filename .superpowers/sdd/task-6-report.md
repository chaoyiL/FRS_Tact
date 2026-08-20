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
