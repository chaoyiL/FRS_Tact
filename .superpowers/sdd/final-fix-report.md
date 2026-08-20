# Selectable setup environments: final-review fixes

## Scope

- Reject every combination of a setup selector and `-h`/`--help`, in either order, before setup
  side effects run.
- Correct the Pi0.5 setup documentation so the displayed command is described as Pi0.5-only and
  the no-argument behavior as dual-environment setup.
- Make the plan's regression-verification command root-environment compatible without changing any
  existing boundary test.

## TDD evidence

### RED

Command:

```bash
uv run --no-sync pytest tests/test_setup_env_dual_environment.py::test_help_cannot_be_combined_with_a_selector_in_either_order -v
```

Result: exit 1; 4 failed and 4 passed. The failures were all selector-then-help cases
(`--root`/`--pi05_deploy` with `-h`/`--help`), each returning 0 when the regression required a
nonzero exit. Help-then-selector cases already failed correctly.

### GREEN

The same command after the one-condition parser fix returned exit 0: 8 passed in 0.06s. Each case
also asserted that the stubbed setup event log remained empty.

## Final verification

```bash
JAX_PLATFORMS=cpu uv run --no-sync pytest \
  tests/test_setup_env_dual_environment.py \
  tests/train_smolvla/test_package_boundary.py \
  tests/train_vtsmolvla/test_package_boundary.py -v
```

Result: exit 0; 35 passed in 20.89s.

```bash
bash scripts/setup_env.sh --help
bash scripts/setup_env.sh -h
bash -n scripts/setup_env.sh
git diff --check
```

Result: every command exited 0. The two help aliases produced identical usage output.

## Concern

`uv run --no-sync ruff check tests/test_setup_env_dual_environment.py` could not run because Ruff
is not installed in the root environment. It is not part of the requested verification matrix;
the focused pytest suite, Bash syntax check, and diff check passed.
