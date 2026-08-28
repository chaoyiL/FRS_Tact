# DECO Balanced-Light V2 Design

## Goal

Train a new Bread Stage 1 DECO model from the normal ResNet34 initialization while
matching the current deployment illumination more closely. The new augmentation
must cover 90% to 120% of the collected-image brightness, preserve mild color and
blur robustness, and keep the existing `low-light-v1` behavior reproducible for old
checkpoints.

## Selected approach

Add a named, versioned augmentation preset and reuse the existing augmentation
engine. Do not duplicate the random transform implementation and do not represent
the new policy as an unversioned collection of environment overrides.

The two supported presets are:

| Setting | `low-light-v1` | `balanced-light-v2` |
|---|---:|---:|
| Identity probability | 0.25 | 0.25 |
| Low-light probability | 0.55 | 0.00 |
| Mild/balanced probability | 0.20 | 0.75 |
| Exposure range | 0.58-0.90 | unused |
| Gamma range | 1.10-1.50 | unused |
| Mild brightness range | 0.90-1.10 | 0.90-1.20 |
| Contrast range | 0.85-1.10 | 0.85-1.10 |
| Saturation range | 0.90-1.10 | 0.90-1.10 |
| Blur probability within non-identity branch | 0.20 | 0.20 |
| Blur kernel sizes | 3, 5 | 3, 5 |
| Blur sigma | 0.1-1.0 | 0.1-1.0 |
| Shared across cameras | yes | yes |

For `balanced-light-v2`, 25% of samples remain bitwise unchanged. The other 75%
receive brightness, contrast, and saturation transforms in that order. Twenty
percent of the transformed branch receives Gaussian blur, so blur affects about 15%
of all training samples. Both cameras in one sample share every sampled parameter.

## Configuration and compatibility

A centralized preset resolver returns a complete canonical augmentation config.
Unknown preset names fail immediately. The canonical resolved dictionary, including
the correct `version`, is stored in every checkpoint.

The low-level Python training entry point keeps a legacy-compatible default: no
explicit preset means the existing `low-light-v1` argument path. The shell launcher
selects `balanced-light-v2` explicitly for new training. An operator can explicitly
select `low-light-v1` when reproducing or resuming an old run.

Named presets are atomic. Fine-grained legacy augmentation arguments must not
silently overwrite a named preset and create a configuration whose values disagree
with its version. Conflicting input is rejected with a clear error.

Exact resume continues comparing the complete canonical augmentation dictionary:

- v1 checkpoint plus v1 configuration: accepted;
- v2 checkpoint plus v2 configuration: accepted;
- v1/v2 mismatch or any parameter mismatch: rejected;
- old Stage 1 and Stage 2 checkpoints without a preset field retain the legacy v1
  path rather than silently switching to v2.

## Training and inference data flow

The existing order remains unchanged:

1. decode JPEG to RGB floating-point values in `[0, 1]`;
2. apply the selected random augmentation during training only;
3. resize/letterbox to the configured image size;
4. apply ImageNet normalization;
5. run the DECO visual encoder and policy.

Validation, TorchScript export, offline evaluation, and real-robot inference do not
apply random augmentation.

## New Bread training run

The v2 Bread model starts from the normal ResNet34 backbone initialization, not from
the existing Bread `best.pt`. It uses the same labeled Bread dataset, split seed,
model architecture, optimizer, scheduler, and training duration as the original run,
so augmentation policy is the intended controlled variable. It uses a new `RUN_ID`
and no `RESUME_FROM`.

The implementation will make the launcher command auditable and verify it with a
dry run. This workspace does not contain the original Bread dataset manifest saved
in the checkpoint (`/home/ljl/.../bread_01_03.json`), so starting the multi-hour
training job requires the valid manifest on the target training machine. The code
change must not invent or silently substitute another dataset.

## Validation and model selection

Training validation remains unaugmented. The new run must retain the same unseen
episode validation protocol as the original run. Comparison against the old Bread
model should include:

- unseen validation loss and velocity MAE;
- offline inference on episodes 34 and 35;
- the saved real-observation sets `220846` and `221116` as descriptive visual-domain
  diagnostics;
- real-robot task success, because visual feature distance alone is not a control
  performance metric.

## Tests

Automated tests must cover:

1. complete canonical snapshots for both presets;
2. v2 probabilities and the `0.90-1.20` brightness range;
3. v2 never enters exposure or gamma branches;
4. contrast, saturation, and blur remain active in v2;
5. identity samples remain bitwise unchanged and are never blurred;
6. both cameras share parameters and a fixed seed is reproducible;
7. unknown presets and preset/override conflicts fail clearly;
8. the launcher dry run selects v2 for a new run and can explicitly select v1;
9. checkpoints store the correct canonical version and parameters;
10. same-version exact resume succeeds, cross-version resume fails, and legacy v1
    Stage 1/Stage 2 restore behavior remains intact.

## Non-goals

- No random augmentation during deployment.
- No change to state/action representation, model architecture, control frequency,
  inference horizon, or camera ordering.
- No attempt to solve camera extrinsic, framing, wall-hotspot, or object-placement
  domain gaps with global brightness jitter alone.
- No modification of existing Bread checkpoint files.
