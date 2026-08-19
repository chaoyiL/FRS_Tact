# Final Review Hardening Design

## Goal

Close the four remaining Important findings against `2a90d99` without importing JAX or model code
during preflight, weakening the standalone/private package boundary, or modifying protected source
projects.

## Runtime preflight contract

`--check` will reject any interpreter whose version is not Python 3.12. It will read
`train_pi05_frs/pyproject.toml`, parse every production dependency with `packaging.requirements`,
evaluate each environment marker for the current platform, and use `importlib.metadata.version` to
verify that every applicable installed distribution satisfies its declared specifier. This makes
the project metadata the only dependency contract: the checker will not maintain a second package
or version table. Platform-applicable JAX CUDA plugin and PJRT distributions present in the locked
production set are checked by metadata only. Tests inject metadata/version and marker environments,
including every production dependency missing in turn, and assert that neither JAX nor model code
is imported.

## Canonical read/write isolation

A shared path-safety API will canonicalize all writable roots and the parent asset roots of every
read-only input: Pi checkpoint, tactile encoder checkpoint, each dataset, normalization asset,
and foreign resume checkpoint. It rejects equality and either ancestor direction across every
read/write pair, including symlink aliases. Existing protected-source and writable/writable overlap
checks remain in force. The sole controlled exception is implicit `<output>/last`: a legacy real
directory must remain exactly there, and a symlink must resolve to a direct child of the same
output's `.checkpoint-generations/` directory. `validate_config` and direct `train_decoder` call
the same API before JAX work or filesystem writes.

A non-resume run rejects an already-existing, non-empty training output. An action/tactile cache
directory that is non-empty but has no recognized manifest is rejected before any memmap can be
opened with a writable mode. A foreign resume requires an explicit valid non-overlapping
`resume_from`; transactional implicit resume pins only the controlled same-output `last` target
described above. No implicit overwrite behavior is introduced.

## Normalization statistics

All state/action normalization fields are validated through one strict vector routine. Each value
must be a one-dimensional numeric array, all fields for a feature must have identical shape, and the
width must equal the dataset feature width. Standard deviations must be elementwise non-negative.
When quantile normalization is enabled, every `q99` element must be strictly greater than the
corresponding `q01` element. Rank-two, inconsistent-shape, negative-standard-deviation, and
degenerate/reversed-quantile fixtures fail preflight.

## Cache and checkpoint provenance

Skip, resume, and training cache loads will rebuild `SampleRecord` values from the complete static
record arrays `dataset_index`, `episode_index`, and `split`, using their stored row order and the
same `train`/`val` mapping used by manifest creation. Split values must be exactly 0 or 1. The
recomputed record digest and train/validation counts must match the manifest before the cache is
accepted. The fourth static array, `inversion_mse`, remains shape-validated but is not a
`SampleRecord` field and therefore does not enter the record digest.

A local Pi checkpoint fingerprint will hash sorted relative file paths plus streamed file contents.
It will never use path identity, file size, or modification timestamps as a substitute for content.
Remote checkpoint identifiers retain their existing non-local handling.

## Testing and completion

Each behavior begins with a focused failing test and receives only the implementation needed to
turn that test green. After all focused suites pass, the full standalone/boundary matrix, shell
syntax, offline lock, frozen sync dry-run, source hashes, default no-GPU preflight, and diff checks
must be rerun. The SDD report and progress record are appended, changes are committed, and the same
read-only reviewer must report zero Critical and Important findings.
