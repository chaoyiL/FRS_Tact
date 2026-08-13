# FRS modality intervention evaluation

Run the modality evaluation against a trained gated FRS checkpoint:

```bash
PYTHONPATH=.:src:tests uv run --no-sync python -m modalities_eval.frs.evaluate \
  --config train_frs/configs/train_frs.yaml \
  --checkpoint-dir /path/to/frs/best \
  --output-dir eval_outputs/frs_modalities \
  --allow-unverified-provenance
```

Existing checkpoints require the explicit `--allow-unverified-provenance`
override.  Their metadata verifies cache record identity and configuration,
plus compatible model/cache shapes and tactile settings, but it does not carry
strong content hashes for the action arrays, tactile embedding arrays, or
tactile encoder contents.  Without the flag the evaluator fails closed.  With
the flag, `summary.json` records `provenance.status` as
`configuration_only`, `strong_content_hashes_verified` as `false`,
`override_used` as `true`, and includes the remaining warning.  The override
does not make those contents verified.

The command evaluates the configured validation split by default.  It writes
`per_sample.csv`, `per_episode.csv`, `summary.json`, and `contribution.png` to
the output directory.  The per-episode file averages samples within each
condition/source/episode group.  The plot reports 95% confidence intervals
from an episode-cluster bootstrap, so repeated samples from an episode remain
together when resampled.  The bootstrap preserves row-weighted means: it
resamples episode clusters and divides selected per-cluster sums by selected
row counts.  Within each condition and gate stratum, every metric uses the
same resampled episode draws.

Gate strata use the checkpoint's ranking thresholds and always classify the
original gate, never a counterfactual gate: `low` is
`original_gate <= rank_low_gate_threshold`, `transition` lies strictly between
the thresholds, and `high` is
`original_gate >= rank_high_gate_threshold`.  Each condition in `summary.json`
records both thresholds under `gate_thresholds`.

`contribution` is `MSE(condition) - MSE(full)`.  A positive contribution means
the intervention made the decoded action less accurate than the full tactile
input, which is evidence that the removed or altered tactile information was
helpful for that sample.  A negative value means the intervention was more
accurate under this metric; it is not by itself evidence of causality.

The two baseline interventions separate tactile content from gate behavior:

- `baseline_fixed` replaces the tactile window with the episode baseline and
  keeps the gate computed from the original tactile observation.
- `baseline_recomputed` makes the same tactile replacement but recomputes the
  gate from the altered window.  Its result includes both the missing tactile
  content and the gate response to that change.

The `val` split is exploratory only: use it to inspect interventions and choose
follow-up hypotheses, not as a final confirmatory result.  Confirm claims on a
separate held-out evaluation split that was not used to tune the checkpoint,
interventions, or analysis choices.
