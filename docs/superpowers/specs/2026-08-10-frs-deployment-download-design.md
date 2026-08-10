# FRS Deployment Download Script Design

## Goal

Add `deploy_smolvla/scripts/download.sh` as the single entry point for preparing every model asset required by FRS deployment under `/home/typhon/FRS_Tact/checkpoints`.

The script must be safe to rerun. It skips complete assets, repairs missing or incomplete assets by downloading them again, and never treats a merely existing directory as a complete checkpoint.

## Assets and Layout

The script prepares this layout relative to the repository root:

```text
checkpoints/
├── model/
│   └── pick_tube_02_3w_jax/
├── frs/
│   └── frs_0809_02/
└── encoder/
    └── encoder_ckpt_0809/
```

The three assets are:

1. A complete JAX SmolVLA checkpoint produced by merging `lerobot/smolvla_base` with `KaiyueChen/pick_tube_02_3w`.
2. The FRS decoder from `KaiyueChen/frs_0809_02`.
3. The tactile encoder from `KaiyueChen/encoder_ckpt_0809`.

## User Configuration

All values intended for routine editing live together near the top of `download.sh`:

- base repository and revision;
- adapter repository and revision;
- FRS repository and revision;
- tactile encoder repository and revision;
- output names below the project-local `checkpoints` root.

The script derives the repository root from its own location, so it works regardless of the caller's current working directory. Output remains project-local unless the user explicitly edits the checkpoint-root variable.

## Download and Skip Behavior

The script processes assets independently in this order:

1. Base SmolVLA JAX checkpoint.
2. FRS decoder.
3. Tactile encoder.

Each asset has a completeness check:

- The JAX checkpoint must contain non-empty `config.json`, `model.safetensors`, the preprocessor and postprocessor JSON files, their normalization safetensors, and `conversion_manifest.json`. The manifest's base, adapter, and pinned revisions must match the variables configured in `download.sh`.
- The FRS directory must contain readable `checkpoint.json`; the `params_file` named by that metadata must exist, be non-empty, and be a valid NumPy archive.
- The tactile encoder uses the existing `deploy_smolvla/src/download_ckpt.py` verification, including metadata, parameter structure, and archive checks.

If an asset is complete, the script prints a clear `skip` message and continues. If it is missing or incomplete, the script invokes the appropriate existing downloader or merger. Hugging Face's resumable/cache behavior supplies already-downloaded blobs where possible.

The script downloads inference files only. It excludes optimizer state, memory bank files, and pickle files.

## Base Model Merge

When the complete JAX checkpoint is absent, the script calls `tools/merge_smolvla_peft_to_jax.py` with the configured base, adapter, revisions, and project-local output directory.

An incomplete or source-mismatched output directory may be left by an interrupted or older merge. Because the destination is an exact project-local path derived by the script rather than user input, the script reruns the merger with `--overwrite` for that destination. It never deletes the directory and never applies overwrite to a path outside the configured project checkpoint root. Tests cover that the command is skipped only for a complete, source-matched output and invoked with `--overwrite` for missing or incomplete output.

## Failure Handling

The script uses strict Bash mode and stops on the first failed download, validation, or merge. Its error output identifies the failing asset and destination.

It must locate `uv` and the project Python environment consistently with existing launchers. No secrets are embedded in the file; Hugging Face authentication continues to use `HF_TOKEN` or locally configured credentials.

The final success message prints the three absolute checkpoint paths and the corresponding deployment YAML keys.

## Deployment Configuration

The implementation will update the existing deployment YAML paths to:

```yaml
checkpoint: /home/typhon/FRS_Tact/checkpoints/model/pick_tube_02_3w_jax

frs:
  checkpoint: /home/typhon/FRS_Tact/checkpoints/frs/frs_0809_02
  tactile_encoder_checkpoint: /home/typhon/FRS_Tact/checkpoints/encoder/encoder_ckpt_0809
```

The user's existing uncommitted FRS parameter changes, including `gate_tau: 0.4` and `verify_source_checkpoint_fingerprint: false`, must be preserved.

## Testing

Shell-level tests will execute a copied script against a temporary project tree with fake `uv`/Python/Hugging Face commands. They will verify:

- all three missing assets trigger their corresponding commands;
- one complete asset is skipped while the other missing assets continue;
- all complete assets produce no download or merge calls;
- an incomplete FRS checkpoint is not skipped;
- configured paths are project-local and safely handle spaces;
- the script prints the final deployment paths;
- existing encoder download tests and FRS deployment configuration tests remain green.

Network access and real model downloads are not part of automated tests.
