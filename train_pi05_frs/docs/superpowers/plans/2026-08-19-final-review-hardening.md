# Final Review Hardening Implementation Plan

> **For Codex:** Execute this plan task-by-task with the `executing-plans` skill. Every task starts with the named failing tests, then receives only the smallest production change needed to make them pass.

**Goal:** Close the four remaining Important findings without importing JAX/models during `--check`, weakening path boundaries, or changing unrelated training/deployment behavior.

**Architecture:** Keep lightweight validation in `tools/train_frs.py` and centralize shared filesystem/cache invariants in `utils/path_safety.py` and `pi05_cache/cache.py`. Production dependency requirements come from `pyproject.toml`; the lock-matched CUDA plugin and PJRT distributions are explicit platform-marked project dependencies, so the same parser checks them without a second hardcoded list or module imports. Cache identity is derived from immutable file contents and the exact `SampleRecord` serialization already used by `records_digest`.

**Tech Stack:** Python 3.12, `tomllib`, `packaging`, `importlib.metadata`, NumPy memmaps, pytest.

---

## Task 1: Metadata-only environment contract

**Files:**

- Modify: `train_pi05_frs/tools/train_frs.py`
- Modify: `train_pi05_frs/tests/test_pipeline.py`

1. Add parametrized RED tests proving that preflight rejects Python 3.11/3.13, discovers the applicable production requirements from `pyproject.toml`, rejects every applicable missing distribution one at a time, rejects every unsatisfied specifier, and includes the JAX CUDA plugin and PJRT requirements. Add a guard that `jax` is absent from `sys.modules` after the check.
2. Run the focused tests and record the expected failures against `REQUIRED_DISTRIBUTIONS` and the absent Python check.
3. Replace the hardcoded dependency map with a parser that loads `[project].dependencies`, evaluates PEP 508 markers for the current platform, and validates installed versions with `Requirement.specifier.contains`. Declare the lock-matched CUDA plugin and PJRT as explicit Linux requirements in that same dependency source. Inject version, marker environment, and Python version for CPU-only tests. Require exactly Python 3.12 before dependency/GPU probes.
4. Run the focused tests GREEN and verify a subprocess `--check` light-import guard still never imports JAX or a model.

## Task 2: Read/write filesystem isolation and no implicit overwrite

**Files:**

- Modify: `train_pi05_frs/utils/path_safety.py`
- Modify: `train_pi05_frs/tools/train_frs.py`
- Modify: `train_pi05_frs/train.py`
- Modify: `train_pi05_frs/pi05_cache/prepare.py`
- Modify: `train_pi05_frs/tests/test_pipeline.py`
- Modify: `train_pi05_frs/tests/test_model.py`
- Modify: `train_pi05_frs/tests/test_pi05_cache.py`

1. Add RED matrices for every writable root versus Pi checkpoint, encoder root, each dataset root, norm asset root, and resume checkpoint root, covering equality, writable ancestor, input ancestor, and symlink aliases. Add direct `train_decoder` rejection before heavy imports. Add fresh non-resume non-empty output and non-empty cache-without-manifest rejection tests.
2. Run those focused tests and record the accepted unsafe cases.
3. Extend the shared validator to canonicalize labeled read-only asset roots and reject bidirectional overlap with all writable roots. Wire the same root sets through `validate_config(check_paths=...)` and direct `train_decoder`; local inputs use their containing asset directory, not a params leaf. Before training, reject a non-empty output unless resuming. Before cache creation, reject a non-empty directory lacking `manifest.json`; retain existing manifest resume/skip and empty-directory creation.
4. Run the focused tests GREEN, including symlink cases and a light-import assertion for direct decoder rejection.

## Task 3: Strict normalization vectors

**Files:**

- Modify: `train_pi05_frs/tools/train_frs.py`
- Modify: `train_pi05_frs/tests/test_pipeline.py`

1. Add RED tests for rank-2 stats, inconsistent per-field shapes, negative standard deviation, equal quantiles, reversed quantiles, and invalid values at individual vector positions.
2. Run the focused tests and confirm current last-axis-only width validation accepts them.
3. Replace `_stat_width` with strict finite numeric 1-D vector conversion. Require every required field for a feature to have the identical `(expected_width,)` shape, `std >= 0` elementwise, and `q99 > q01` elementwise when quantile normalization is enabled.
4. Run the focused norm tests GREEN.

## Task 4: Cache records and local checkpoint content identity

**Files:**

- Modify: `train_pi05_frs/pi05_cache/cache.py`
- Modify: `train_pi05_frs/pi05_cache/prepare.py`
- Modify: `train_pi05_frs/tests/test_pi05_cache.py`

1. Add RED tests for complete skip, partial resume, and `CachedPairs` training load when dataset indices, episode indices, or split bytes are altered without updating the manifest. Cover split values outside `{0,1}`, train/val count mismatches, and checkpoint content changes that preserve path, size, and mtime.
2. Run the focused tests and record that manifest-only provenance permits the corruptions.
3. Add one cache helper that validates the three static record arrays are one-dimensional at `sample_count`, requires split values in `{0,1}`, rebuilds `SampleRecord` in row order via the existing `records_from_arrays` semantics, and compares `records_digest`, `train_sample_count`, and `val_sample_count`. Call it before prepare skip/resume and from `CachedPairs` immediately after arrays open. Make `records_from_arrays` fail closed instead of mapping every nonzero split to validation.
4. Change the local checkpoint fingerprint to stream file bytes into SHA-256 together with stable relative paths (or consume a verified trusted content manifest if one exists); never use mtime or size as identity.
5. Run the focused tests GREEN, including same-size/same-mtime content mutation.

## Task 5: Documentation, complete verification, and review

**Files:**

- Modify: `.superpowers/sdd/final-review-fixes-report.md`
- Modify: `.superpowers/sdd/progress.md` if status text requires it
- Verify: all modified sources/tests and protected-source hash inventory

1. Run all focused groups, then the repository's complete standalone matrix under the actual Python-safe launcher environment.
2. Run `git diff --check`, all shell `bash -n` checks, private-package/boundary tests, protected-source hash verification, and confirm `--check` performs no JAX/model import.
3. Append the RED/GREEN evidence and final counts to the final-review report, self-review the full diff against all four findings, and commit only the intended files.
4. Start the same read-only reviewer against the committed HEAD. If it reports any Critical/Important issue, add a new RED test and repeat the minimum GREEN fix, full matrix, report, commit, and review until `Critical=0, Important=0`.
