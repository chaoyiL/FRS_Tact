# Pi0.5 FRS training

This directory is the standalone training project for the three-stage Pi0.5 flow-steering
pipeline: tactile embedding precomputation, Pi0.5 action-cache preparation, and FRS decoder
training. It owns a third environment, `train_pi05_frs/.venv`, separate from the repository
training environment and `deploy_pi05/.venv`.

## Set up and run

Run commands from the repository root. The launcher selects only the package-local interpreter,
puts `train_pi05_frs/src` before the repository on `PYTHONPATH`, and then changes into this project.

```bash
cd /home/typhon/FRS_Tact
bash train_pi05_frs/scripts/setup_env.sh
bash train_pi05_frs/scripts/setup_env.sh --check

# Edit the /workspace examples first, then run a dependency-light preflight.
bash train_pi05_frs/scripts/start_frs_pi05_train.sh --check \
  train_pi05_frs/configs/train_pi05_frs.yaml

# tmux is used when available.
bash train_pi05_frs/scripts/start_frs_pi05_train.sh \
  train_pi05_frs/configs/train_pi05_frs.yaml

# Force the same ordered pipeline in the current terminal.
FRS_FOREGROUND=1 bash train_pi05_frs/scripts/start_frs_pi05_train.sh \
  train_pi05_frs/configs/train_pi05_frs.yaml
```

The normal pipeline validates the complete schema and all input paths, performs a Pi0.5
checkpoint/GPU shape smoke, and runs the three stages in order. A failed stage stops the pipeline.
`--check` stops before output/cache directory creation, JAX import, GPU initialization, model
loading, or tmux. Each foreground run creates one timestamped
`frs_training.output/pipeline_YYYYmmdd_HHMMSS.log`. Set `FRS_TMUX_SESSION` to choose the tmux
session name; attach with `tmux attach -t <name>`.

## Configuration paths

The default file is `configs/train_pi05_frs.yaml`. Absolute paths are recommended; relative local
paths are resolved from the repository root. URL values such as `gs://...` remain strings and are
never converted through `Path`.

- `checkpoint`: Pi0.5 checkpoint URL or a local directory containing `params/`.
- `datasets[].root`: a complete LeRobot v3 dataset containing info/tasks/stats/episodes metadata,
  every referenced data parquet, and all referenced video assets; `repo_id`, `action_key`, and
  `rename_map` describe its identity and source columns.
- `action_cache.root`: parent of one sanitized `repo_id` subdirectory per dataset.
- `tactile_embedding_cache.root`: parent of per-dataset, per-frame frozen ResNet embeddings.
- `model.tactile_encoder_path`: checkpoint produced by the repository's `train_encoder` project.
- `model.camera_map`: Pi0.5 image slots mapped to post-rename dataset observation keys.
- `norm_stats.dir` and `norm_stats.asset_id`: the exact statistics used by cache generation and
  deployment.
- `frs_training.output`: history, plots, `last`, `best`, and pipeline logs.
- `frs_training.resume` / `resume_from`: resume transactionally from `<output>/last`, or from an
  explicit checkpoint asset root that does not overlap any writable output/cache root.

Boolean YAML fields are strict: use unquoted `true` and `false`, not strings or `0`/`1`.

## Cache provenance and resume

Embedding and action caches record dataset, revision, checkpoint/config, selection, and shape
provenance. A complete matching cache is skipped; an incomplete matching cache resumes. A mismatch
is rejected instead of overwriting existing data. Use a new cache root for changed inputs, or use
the precompute tool's explicit `--overwrite` only when replacement is intentional.

To resume decoder training from the same transactional output, set `frs_training.resume: true` so
the loader pins the immutable generation currently referenced by `<output>/last` before any write.
For a checkpoint from another run, copy or pin it outside every writable output/cache root and set
`frs_training.resume_from` to that directory. Completed cache stages remain skip/resume safe and
the decoder restores compatible parameters and optimizer state.
Each save first writes and verifies an immutable generation under
`<output>/.checkpoint-generations/`, then atomically switches the `last` or `best` symlink. Tools
may consume those aliases directly. When copying a checkpoint outside its output directory,
dereference the symlink so the canonical metadata, parameters, and optimizer files travel together.
Because the protected deployment loader opens metadata and parameters separately, never point
deployment at the mutable `last` or `best` alias while training can update it. Resolve the alias to
its immutable `.checkpoint-generations/<generation>` directory (or use a dereferenced copy) and put
that pinned path in the deployment configuration.

## Manual stages and evaluation

The launcher is preferred, but each stage can be invoked with the same environment:

```bash
cd /home/typhon/FRS_Tact/train_pi05_frs
export PYTHONPATH="$PWD/src:$(dirname "$PWD")${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONSAFEPATH=1

.venv/bin/python -m train_pi05_frs.tools.precompute_tactile_embeddings \
  --config configs/train_pi05_frs.yaml
.venv/bin/python -m train_pi05_frs.tools.prepare_frs_pi05_cache \
  --config configs/train_pi05_frs.yaml
.venv/bin/python -m train_pi05_frs.tools.train_frs \
  --config configs/train_pi05_frs.yaml
```

Evaluate one configured dataset/cache against a saved decoder checkpoint:

Evaluation resolves images from the dataset repository ID recorded in the action-cache manifest
(or the explicit `--dataset-repo-id` override) through the inherited dataset loader. The current
loader does not accept a local dataset-root override.

```bash
PINNED_DECODER_CHECKPOINT="$(
  readlink -f /workspace/frs_pick_tube_pi05/run_gated_v7_state_01/best
)"
.venv/bin/python -m train_pi05_frs.evaluate \
  --cache-dir /workspace/frs_pick_tube_pi05/action_cache_slerpflow_k50_state_v3/KaiyueChen/pick_tube_05 \
  --dataset-repo-id KaiyueChen/pick_tube_05 \
  --tactile-encoder-dir /workspace/checkpoints/encoder_ckpt_0809 \
  --checkpoint-dir "${PINNED_DECODER_CHECKPOINT}" \
  --output-dir /workspace/frs_pick_tube_pi05/run_gated_v7_state_01/evaluation
```

## Deployment handoff and project boundary

After evaluation, point `deploy_pi05/configs/deploy_pi05_frs.yaml` at the selected FRS decoder,
the same tactile encoder, Pi0.5 checkpoint, norm stats, dimensions, camera map, and decoder solver.
Then follow `deploy_pi05/README.md`; training does not start or modify a robot client.

Encoder training remains in `train_encoder`, and modality analysis remains outside this project.
This package consumes their stable outputs only. It does not contain deployment clients, encoder
training, SmolVLA training, or general modality-analysis workflows.

## Verification status

### Automated mock/CPU verification

The migration verification covers shell syntax, the isolated offline lock and frozen environment,
source hashes and project boundaries, dependency-light preflight behavior, and the complete Python
test suite. The tests use CPU-sized decoder models and synthetic/mock data to exercise forward,
loss, one optimizer step, checkpoint save/restore, deployment-runtime loading, real spawned tactile
workers, exact three-stage ordering, failure-stop behavior, and cache resume/provenance contracts.
These checks validate integration contracts; they are not evidence of a production dataset run or
a completed GPU training job.

### Real GPU/data/checkpoint verification

The real three-stage GPU pipeline has not been run as part of this migration handoff. The checked-in
configuration intentionally retains `/workspace` example paths, and those paths must be replaced
with the deployment machine's real Pi0.5 checkpoint, LeRobot v3 datasets, tactile encoder, norm
statistics, cache roots, and output directory. On that machine, first run the launcher with
`--check`, confirm a GPU is visible and every asset passes preflight, then run a bounded cache/training
smoke before starting the full job. Preserve the resulting pipeline log and checkpoint/evaluation
metadata as the evidence for that real run.
