# Bread Phase-Conditioned DECO Design

## Goal

Retrain the Bread policy as one phase-conditioned DECO checkpoint that is less
sensitive to small visual and robot-state changes and cannot execute the left-hand
ketchup stage before the right hand has released the bread. The solution must use
only the existing Bread data, produce exactly one PT/TS weight pair, and keep Bread
training isolated from the generic DECO training launchers.

## Evidence and constraints

The current checkpoint was trained on 1,362 episodes and about 717,000 frames, not
the three local Parquet files used for offline diagnostics. Its epoch-30 unseen
validation loss is 0.205433 versus 0.076674 on a held-out subset of training
episodes, a 2.68x generalization gap. Repeating the same training configuration is
therefore not sufficient.

The current `balanced-light-v2` policy covers global brightness, contrast,
saturation, and blur, but not camera framing, per-camera exposure differences,
object placement, initial state variation, or task phase ambiguity. Deployment
traces also show that small input changes can select incompatible modes: one run
remains in the right-hand stage while another skips directly to the left-hand
stage. The current policy has no phase/history input, and executing all 32 predicted
steps amplifies a wrong mode for about 1.07 seconds before replanning.

No new demonstrations will be collected. The existing 1,362 episodes are the only
training source.

## Selected architecture

Train one model and export one artifact pair:

```text
bread_phase_v3.pt
bread_phase_v3.ts
```

The deployment boundary is:

```text
bread_phase_v3(images, state, phase_id) -> [32, 20]
```

The model shares one visual encoder, one DECO transformer, and one action head.
A learned two-value phase embedding conditions the policy:

- `phase_id = 0`: right hand picks, transports, places, and releases the bread;
- `phase_id = 1`: left hand picks and uses the ketchup.

This is one checkpoint, not two policies packaged together. The output remains the
existing 20-dimensional bimanual action so the robot protocol and action layout do
not change.

## Bread-only implementation boundary

Bread phase training is isolated in a dedicated package and launcher:

```text
train_deco/
  bread_phase/
    augmentation.py
    dataset.py
    export.py
    model.py
    train.py
  scripts/
    train_bread_phase.sh
```

The package may reuse stable low-level DECO utilities, but it owns phase labeling,
phase-balanced sampling, Bread augmentation, the phase-conditioned wrapper,
training orchestration, and export. Existing generic launchers, presets, and other
task behavior remain unchanged.

`train_bread_phase.sh` defaults to `bread_01_03.json`, starts from the normal
ResNet34 initialization, and does not resume the existing Bread policy. It supports
`--dry-run` plus environment overrides for GPU selection, output directory, batch
size, workers, and run ID. Its only model output is the single phase-conditioned
PT/TS artifact pair.

## Automatic phase labeling

Each episode is labeled from recorded action/state events:

1. detect the right-gripper close event;
2. detect the subsequent right-gripper reopen event;
3. detect the first sustained left-arm motion after release;
4. assign phase 0 from episode start through the right-hand release and clearance
   motion preceding sustained left-arm movement;
5. assign phase 1 from sustained left-arm movement through episode end.

The phase-1 state is rebased at its segment boundary so the left-stage policy does
not infer task phase from elapsed global trajectory displacement. The bimanual
relative state remains geometrically consistent after rebasing.

Episodes missing the required close, reopen, or ordered left-motion events are
excluded from phase-conditioned training and listed in a machine-readable report.
The pipeline fails if no valid examples remain for either phase; it never silently
assigns ambiguous samples.

## Sampling and objective

Training batches contain phase 0 and phase 1 with equal probability. Frames near
right close, right reopen, and left-motion onset are oversampled 3x relative to
ordinary frames. This prevents long steady segments from dominating the objective.

The model continues to predict the complete bimanual action. The active arm for the
selected phase receives the primary action loss; the inactive arm retains ordinary
hold-action supervision at a lower weight. Deployment masking, rather than an
unconstrained inactive output, provides the hard cross-arm guarantee.

## Bread-specific augmentation

The Bread augmentation contract uses:

- identity probability: 0.25;
- augmented probability: 0.75;
- brightness range: **0.80-1.20**;
- contrast range: 0.85-1.30;
- saturation range: 0.80-1.15;
- the existing mild Gaussian blur;
- small image translation up to 8 pixels;
- scale range 0.95-1.05;
- in-plane rotation up to 2 degrees;
- low-probability independent exposure and color-temperature changes per camera;
- small state measurement perturbations of approximately 2 mm translation,
  0.5 degrees rotation, and 2 mm gripper width.

State perturbations are generated through consistent pose transforms before the
20D state is rebuilt; correlated state fields are not independently jittered.
Validation, export, and deployment never apply random augmentation.

## Validation and model selection

Validation is grouped by data source and collection order rather than using only a
random within-source episode split. Model selection uses unseen grouped validation
loss plus phase-specific metrics:

- phase-0 left-arm path and gripper-command leakage;
- phase-1 right-gripper reclose leakage;
- active-arm trajectory error for each phase;
- transition-window error around close, reopen, and left-motion onset;
- prediction contrast when the same observation is evaluated with phase 0 and
  phase 1;
- sensitivity across fixed inference seeds and recorded state perturbations.

The final all-data training duration is selected from the grouped validation run;
training is not continued to 100 epochs merely because the launcher requested it.

## Deployment

Bread deployment uses a dedicated client and YAML configuration so other DECO
tasks retain their existing contracts. One TorchScript weight is loaded once, and
the client supplies the current `phase_id` on every inference.

### Phase 0

- Execute only the predicted right-arm action.
- Replace the left-arm action with identity pose delta and an open/current gripper
  hold command.
- Record that the right gripper has physically closed.
- Permit transition only after the right arm enters the release/clearance envelope
  learned from valid training boundaries and the measured right gripper is open for
  two consecutive observations.
- If the transition does not occur before the training phase-0 duration P99 plus
  two seconds, stop the run. Never enter phase 1 on timeout.

### Phase 1

- Rebase the phase-local state at transition.
- Execute only the predicted left-arm action.
- Replace the right-arm action with an identity pose delta and an explicitly open
  gripper command.
- Record that the left gripper has physically closed, then declare completion only
  when the left TCP remains inside the terminal squeeze envelope derived from the
  final 32 valid training frames for two consecutive observations while its motion
  remains below the training terminal-motion P95 threshold.
- Stop without declaring success if the phase-1 duration P99 plus two seconds is
  reached before that completion condition.

The model horizon remains 32 because that is the training/export contract, while
deployment executes only the first 16 steps before obtaining a new observation.
At 30 Hz this limits an incorrect open-loop prefix to about 0.53 seconds.

## Error handling and observability

The Bread launcher rejects checkpoints without the phase-conditioned metadata,
phase count, phase-labeling version, transition statistics, or the expected
`[32, 20]` action contract. Phase IDs outside `{0, 1}` are errors.

Every run logs the current phase, close/reopen observations, transition evidence,
timeout deadline, active-arm mask, original model action, and executed masked
action. Labeling reports include accepted/rejected episode counts and per-event
frame distributions.

## Verification

Automated verification covers:

1. close/reopen/left-motion labeling and rejection of malformed event order;
2. phase-local state rebasing and geometric consistency;
3. 50/50 phase sampling and 3x transition sampling;
4. Bread augmentation bounds, including brightness 0.80-1.20;
5. fixed-seed reproducibility and paired-camera versus independent-camera rules;
6. phase-conditioned eager/TorchScript parity;
7. single PT/TS artifact production and metadata validation;
8. launcher `--dry-run` output and isolation from generic `train.sh` behavior;
9. inactive-arm masking and phase transition/timeout state-machine behavior;
10. offline replay of current failure observations, including `215100`, where
    phase 0 must execute zero left-arm displacement and must not transition without
    confirmed right release;
11. multiple inference seeds and small state perturbations without cross-phase
    execution leakage.

## Non-goals

- Training or packaging two independent weights.
- Collecting new demonstrations.
- Modifying augmentation or training defaults for non-Bread tasks.
- Changing the 30 Hz training/control contract or 32-step model horizon.
- Treating lower validation loss alone as proof of real-robot success.
- Allowing a timeout to skip the right-hand stage and start the left hand.
