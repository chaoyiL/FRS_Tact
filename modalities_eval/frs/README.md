# FRS modality intervention evaluation

Run the modality evaluation against a trained FRS checkpoint with gated-training provenance:

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
row counts.  Within each condition and training-label stratum, every metric
uses the same resampled episode draws.

`w` is a training-only supervision and reporting label.  Every evaluation
decode uses decoder input v2—the action base, tactile tokens, and optional
state token—and never passes `w` or a gate value.  Checkpoints must therefore
declare `decoder_input_version: 2`.  Legacy checkpoints with
`decoder_config.gate_conditioning: true` are incompatible, are not converted
automatically, and must be retrained before evaluation.

`contribution` is `MSE(condition) - MSE(full)`.  A positive contribution means
the intervention made the decoded action less accurate than the full tactile
input, which is evidence that the removed or altered tactile information was
helpful for that sample.  A negative value means the intervention was more
accurate under this metric; it is not by itself evidence of causality.

Baseline interventions replace the tactile window with the episode baseline.
They affect the decoder only through tactile content; `w` is a reporting label
and never a decoder input.

The `val` split is exploratory only: use it to inspect interventions and choose
follow-up hypotheses, not as a final confirmatory result.  Confirm claims on a
separate held-out evaluation split that was not used to tune the checkpoint,
interventions, or analysis choices.
