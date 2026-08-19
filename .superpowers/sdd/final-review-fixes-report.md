# Final branch-review fixes report

## Scope and result

This follow-up addresses all six Important and two Minor findings from the complete
`e0209e6..ac2d164` branch review without modifying the protected root projects or either existing
environment. Work remained inside the standalone `train_pi05_frs` project, its boundary test, and
the SDD records. Every behavioral fix began with a focused failing test and was then implemented
minimally.

## Important findings

1. **Standalone package discovery.** Replaced overlapping `where = ["src", ".."]` discovery with
   an explicit, non-overlapping package list and `package-dir` mappings. The test expands the real
   setuptools configuration and verifies every `lerobot*` package resolves below
   `train_pi05_frs/src/lerobot`, every `train_pi05_frs*` package resolves below the standalone
   project, and no root-only `lerobot.processor` is included.

2. **Complete action-cache provenance.** Cache configuration now records `action_dim`,
   `action_horizon`, `paligemma_variant`, and `action_expert_variant`. Before either complete-cache
   skip or partial resume it compares the full configuration key union, record digest, and top-level
   action/state dimensions, then opens and validates all eight cache-array shapes. Four
   configuration fields plus all three top-level dimensions are exercised for both complete and
   incomplete manifests (14 provenance mismatch cases); every real array is also corrupted in
   complete and incomplete caches (16 shape mismatch cases).

3. **Decoder resume identity.** Resume pins an immutable checkpoint snapshot, constructs the
   current cache pairs, and compares `cache_records_sha256` plus `cache_configuration` before
   optimizer restoration, history/output writes, or the already-finished return. Same-shape caches
   with different record selection/configuration and checkpoints missing provenance are rejected
   with an explicit instruction to use a separate fine-tune workflow/output instead of resume.

4. **Dependency-light `--check`.** Validation imports neither JAX nor model code. It reads critical
   installed versions through `importlib.metadata`, performs an injectable `nvidia-smi -L` probe,
   opens encoder archives using `numpy.load(..., allow_pickle=False)` and validates the expected
   arrays, checks complete LeRobot v3 info/tasks/stats/episodes metadata, validates episode data and
   video locator fields including per-video timestamp bounds, opens every referenced data parquet,
   requires the five LeRobot default feature declarations, verifies every non-video feature column
   (including runtime `index`/`task_index` and image-backed columns), and verifies every referenced
   video asset. The positive fixture is loaded through the real schema-derived LeRobot reader.
   Dataset/internal and external normalization-stat widths are checked against state/action feature
   widths. The checked-in default command now truthfully fails on this host because no NVIDIA GPU
   is reported.

5. **Output/cache containment.** A single canonical path validator rejects filesystem/repository
   roots, protected root files and projects, source/config/test descendants, symlink aliases
   (including a designated generated root whose own directory entry is replaced by a symlink), and
   equal or ancestor/descendant collisions between action cache, tactile cache, and training
   output. Both configuration validation and the direct `train_decoder` API run the check before
   JAX import or filesystem writes.

6. **Generation-transactional checkpoints.** A save now writes canonical checkpoint files into an
   immutable `.checkpoint-generations/<uuid>` snapshot, fsyncs every file, stores generation-bound
   size/SHA256 records in v3 metadata, reloads and validates the complete snapshot, fsyncs its
   directory, publishes the generation, and finally atomically replaces the relative `last`/`best`
   symlink. Existing real-directory legacy aliases are upgraded with Linux
   `renameat2(RENAME_EXCHANGE)` and archived rather than deleted; unsupported platforms fail closed
   only for that one-time legacy upgrade. Loaders pin aliases once, strictly validate v3 checksums,
   never silently discard corrupt v3 optimizer state, and retain v1/v2 directory compatibility.
   Training/resume loaders resolve the alias once before reading any file. A v3 snapshot's
   generation is an internal manifest identity rather than a directory-name constraint, so a
   fully dereferenced snapshot can be copied into a named handoff directory while retaining strict
   generation/file/checksum validation; snapshots inside the canonical generation root must still
   match their directory name. The protected deployment loader opens files separately and
   therefore must receive a pinned immutable generation path (or dereferenced copy), never a live
   `last`/`best` alias while training is publishing. The README and executable example enforce that
   handoff contract, supplies a shell-valid `readlink -f` pin example, and the unchanged deployment
   runtime is tested after its mutable alias advances.

   Fault injection after each of the three binary writes, metadata write, validation, directory
   fsync, generation publish, pointer preparation, pointer publish, and parent fsync proves that a
   reader sees either all of generation A or all of generation B, never a mixed checkpoint.

## Minor findings

- `.superpowers/sdd/progress.md` now records completed Task 5 and this follow-up.
- The two vendored `__init__.py` descriptions no longer refer to a nonexistent README or claim that
  the trimmed training namespace mirrors upstream modules which are not present. Because they are
  now deliberate adaptations, both moved from the unchanged-source hash manifest into the explicit
  approved-adaptation boundary set.
- The private cache helper is also now an explicit approved adaptation: its manifest loader and the
  prepare path both reject invalid status/count progress before any cache can be skipped, resumed,
  or consumed.

The operator README also explains the immutable generation layout, requires a pinned immutable
deployment path, and shows a shell-valid `readlink -f` handoff in the executable example.

## TDD evidence

- Package discovery: RED `2 failed`; GREEN `2 passed`.
- Cache provenance mismatch matrix: RED `14 failed`; GREEN `15 passed`. Actual cache-array matrix:
  RED `8 failed, 8 passed`; GREEN `16 passed` across all eight arrays and both cache statuses.
- Resume provenance/order: RED import/ordering failure; GREEN `3 passed` with two mismatch subtests.
- Lightweight preflight/assets: RED `11 failed`; GREEN `12 passed` focused. Full asset follow-up:
  RED for both missing data/video assets and all four required episode locator fields; GREEN in the
  complete `116 passed` pipeline suite.
- Path containment: RED `15 failed`, followed by two protected-root-file failures and three missed
  repository/environment descendants; GREEN `17 passed` initial focus and `116 passed` final
  pipeline suite.
- Checkpoint transaction: RED `13 failures` across transaction/checksum cases; GREEN `7 passed,
  12 subtests passed` for checkpoint/resume focus.
- Vendored descriptions and handoff documentation each had a focused one-test RED before GREEN.
- Dereferenced v3 handoff: RED generation mismatch; GREEN in the model suite.
- Deployment pinned-generation compatibility after alias advancement: GREEN with the unchanged
  protected deployment loader; full cross-environment focus `76 passed, 14 subtests passed`.
- Deployment README example contract: RED `1 failed`; GREEN `1 passed`.
- Third review follow-ups (symlinked generated root, video timestamp fields, `task_index`,
  image-backed columns, shell-valid handoff, and canonical generation name): RED `7 failed,
  4 passed`; GREEN `11 passed`. Final focused suites: pipeline `121 passed`; model/cache/deployment
  `77 passed, 14 subtests passed`.
- Fourth review follow-ups: default-feature/index/real-reader RED `7 failed, 1 passed`, exact default
  specs RED `3 failed`, and cache progress RED `8 failed` at each of prepare and loader boundaries;
  GREEN combined focus `27 passed`, boundary focus `18 passed`.
- Fifth review follow-up: boolean/float shapes and missing `names` RED `3 failed, 3 passed`; strict
  field presence/type GREEN `7 passed` including the real-reader positive control.
- Sixth review follow-up: real launcher cwd reproduced top-level `utils` shadowing with RED
  `2 failed`; exporting `PYTHONSAFEPATH=1` in launcher/manual environments gives GREEN `3 passed`
  and proves the heavy training import resolves the protected repository `utils` package.

## Fresh verification matrix

- `bash -n train_pi05_frs/scripts/setup_env.sh`: passed.
- `bash -n train_pi05_frs/scripts/start_frs_pi05_train.sh`: passed.
- `uv lock --check --offline --project train_pi05_frs`: passed, 154 packages resolved.
- `uv sync --frozen --python 3.12 --project train_pi05_frs --dry-run`: passed, 150 packages
  checked and no changes required.
- Full standalone package plus boundary suite with bytecode/cacheprovider disabled:
  `259 passed, 14 subtests passed in 38.62s`.
- `bash train_pi05_frs/scripts/setup_env.sh --check`: passed and selected only
  `train_pi05_frs/.venv`.
- `sha256sum -c train_pi05_frs/source_manifest.sha256`: all 47 unchanged source mappings passed;
  the adapted cache loader has an explicit boundary mapping and behavior test.
- `git diff --check`: passed.

The real default launcher check exited before JAX/model import with
`RuntimeError: GPU preflight failed: no NVIDIA GPU was reported`. The `/workspace` example assets
and a real GPU remain unavailable, so no production cache/training run is claimed.

## Post-commit review gate

The read-only reviews of `5ddcdd4`, `5bb1fc1`, `9475411`, `3212bb7`, and `e2c6c2c` respectively reported
`Critical=0, Important=5, Minor=2`, `Critical=0, Important=3, Minor=1`, and
`Critical=0, Important=2, Minor=0`, `Critical=0, Important=1, Minor=0`, and
`Critical=0, Important=1, Minor=0`. Commits `5bb1fc1`, `9475411`, `3212bb7`, `e2c6c2c`, and
`2a90d99` repair each successive finding with the RED/GREEN cycles documented above. The final
read-only review of `2a90d99` reported `Critical=0, Important=0, Minor=0`. It independently reran
the actual launcher heavy import, foreground/tmux/manual environments, the full `259 passed,
14 subtests passed` matrix, all 47 unchanged-source hashes, shell syntax, and incremental/total
diff checks; no regression remained.

## 2026-08-19 final hardening round

The subsequent complete-branch audit found four additional Important fail-open boundaries. This
round closes them without modifying protected deployment/source projects:

1. `--check` now requires exactly Python 3.12 and parses the applicable PEP 508 requirements,
   markers, and version specifiers directly from `train_pi05_frs/pyproject.toml`. The Linux CUDA
   contract explicitly includes the lock-matched `jax-cuda12-plugin` and `jax-cuda12-pjrt` 0.5.3
   distributions. Every applicable dependency is queried through `importlib.metadata`; no JAX or
   model module is imported by preflight.
2. The shared canonical path validator now rejects equality and both ancestor directions between
   every writable action-cache/tactile-cache/output root and every local Pi checkpoint, encoder,
   dataset, norm asset, and resume-checkpoint root, including symlink aliases. Direct decoder calls
   apply the same read-only boundary before heavy imports. Fresh non-resume training rejects a
   non-empty output, and action-cache preparation rejects a non-empty directory without a manifest
   before loading stats or a model.
3. External normalization statistics are finite numeric vectors with exact one-dimensional dataset
   width. Every validated field shares that shape, standard deviations are elementwise
   non-negative, and every present/required quantile pair satisfies elementwise `q99 > q01`.
4. Complete skip, partial resume, and `CachedPairs` training load rebuild `SampleRecord` objects in
   exact row order from `dataset_indices.npy`, `episode_indices.npy`, and `split.npy`. Split values
   must be exactly 0/1, and the rebuilt digest plus train/validation counts must match the manifest.
   Local Pi checkpoint identity now hashes every regular file below the checkpoint root in stable
   relative-path order, streaming each file into its own SHA-256 and folding those content digests
   into the root digest; path, size, and mtime are not treated as content identity.

### Hardening TDD evidence

- Environment RED: six failures for the absent pyproject parser/Python contract; GREEN: `6 passed`,
  followed by marker-platform coverage and a per-applicable-distribution missing loop.
- Filesystem RED: 17 accepted writable/input overlap, symlink, and stale-output cases plus one
  action-cache overwrite case; GREEN expanded to all three writable roots against all five input
  roots and direct decoder coverage (`61 passed` combined hardening focus).
- Norm RED: four accepted rank/negative-std/equal-or-reversed-quantile cases; GREEN: `7 passed`
  including the existing width matrix.
- Cache identity RED: 14 failures covering complete/incomplete prepare, training load, invalid split
  values, split counts, and same-size/same-mtime content mutation; GREEN: `14 passed` initially and
  `15 passed` after the early no-manifest guard.

### Hardening verification before review

- Full standalone, project-boundary, and protected-deployment compatibility matrix under the actual
  `PYTHONSAFEPATH=1` environment: `341 passed, 18 subtests passed in 40.81s`.
- `uv lock --check --offline --project train_pi05_frs`: passed, 154 packages resolved.
- Frozen Python 3.12 sync dry-run: passed, 150 packages checked and no changes required.
- Both shell entrypoints pass `bash -n`; `setup_env.sh --check` selects only the standalone Python
  3.12 environment.
- All 47 protected-source hashes pass; `git diff --check` passes.
- The real default launcher truthfully exits with `GPU preflight failed: no NVIDIA GPU was reported`
  before JAX/model import on this host.
