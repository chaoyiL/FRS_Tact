# Task 7 report: deployment objective-v2 acceptance

## Scope

- Added a deployment-local loss-contract validator in `deploy_pi05/frs_runtime.py`.
- Preserved legacy `loss_mode="gated"` acceptance.
- Accepted `loss_mode="bimanual_gated"` only when the complete objective-v2,
  weighting-v7, 32D metadata contract matches the decoder action width.
- Upgraded the deployment contract acceptance tests to use a 32D Pi0.5 decoder
  and policy plus metadata produced by the authoritative training schema.
- Did not change decoder construction, steering, action unnormalization or
  truncation, deployment configuration, or user files.
- Production deployment code does not import training modules.

## RED

Command:

```bash
cd /home/typhon/FRS_Tact/.worktrees/pi05-bimanual-frs
/home/typhon/FRS_Tact/.venv/bin/python -m pytest \
  tests/jax/test_frs_deployment.py -k 'contract and bimanual' -q
```

Result before implementation: `8 failed, 140 deselected`.

- The valid 32D bimanual checkpoint failed with
  `FRS checkpoint mismatch for loss_mode: 'bimanual_gated' != 'gated'`.
- Each invalid-field case was rejected at the old scalar `loss_mode` check
  instead of identifying `loss_objective_version`, `loss_weighting_version`,
  `action_dim`, `steered_action_dim`, `action_slices`,
  `wrist_token_indices`, or `padded_tail_policy`.

## GREEN

Focused deployment contract:

```text
........                                                                 [100%]
8 passed, 140 deselected in 3.20s
```

Training checkpoint cross-runtime loading:

```bash
cd /home/typhon/FRS_Tact/.worktrees/pi05-bimanual-frs/train_pi05_frs
DEPLOY_PI05_PYTHON=/home/typhon/FRS_Tact/deploy_pi05/.venv/bin/python \
PYTHONPATH="$PWD/src:$(dirname "$PWD")" PYTHONSAFEPATH=1 \
.venv/bin/python -m pytest tests/test_deployment_checkpoint_compatibility.py -q
```

```text
..                                                                     [100%]
2 passed, 2 subtests passed in 10.67s
```

Pi0.5 deployment-only suite in its dependency environment:

```bash
cd /home/typhon/FRS_Tact/.worktrees/pi05-bimanual-frs
PYTHONPATH="$PWD/deploy_pi05/src:$PWD/deploy_pi05:$PWD" PYTHONSAFEPATH=1 \
train_pi05_frs/.venv/bin/python -m pytest \
  tests/test_deploy_pi05_deployment_only.py -q
```

```text
...................                                                      [100%]
19 passed in 6.44s
```

## Full affected root deployment file

Command:

```bash
cd /home/typhon/FRS_Tact/.worktrees/pi05-bimanual-frs
/home/typhon/FRS_Tact/.venv/bin/python -m pytest \
  tests/jax/test_frs_deployment.py -q
```

Result: `146 passed, 2 failed in 3.69s`.

The two failures are outside this diff and are unchanged SmolVLA deployment
configuration expectations:

1. `tests/jax/test_frs_deployment.py::test_deploy_frs_config_uses_project_local_downloads`
   - Actual checkpoint path:
     `/home/typhon/FRS_Tact/checkpoints/model/pick_tube_01_jax`
   - Expected checkpoint path under this worktree:
     `/home/typhon/FRS_Tact/.worktrees/pi05-bimanual-frs/checkpoints/model/pick_tube_01_jax`
   - The checked-in SmolVLA config resolves to the main repository path; Task 7
     does not modify that config or its loader.
2. `tests/jax/test_frs_deployment.py::test_deploy_frs_config_preserves_training_time_scale`
   - Actual `config["control"]["control_frequency"]`: `20.0`
   - Expected: `10.0`
   - Task 7 does not modify the SmolVLA config or control frequency.

No configuration was changed to mask either environment/pre-existing failure.
