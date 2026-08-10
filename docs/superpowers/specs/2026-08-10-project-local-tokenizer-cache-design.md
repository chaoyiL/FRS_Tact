# Project-Local Tokenizer Cache Design

## Goal

Make the FRS deployment fully offline and self-contained below
`/home/typhon/FRS_Tact/checkpoints` by downloading the tokenizer required by
the merged SmolVLA checkpoint. A default `start_frs.sh` invocation must no
longer depend on a pre-existing user-level Hugging Face cache.

## Root cause

The merged checkpoint's `policy_preprocessor.json` names
`HuggingFaceTB/SmolVLM2-500M-Video-Instruct`. Deployment sets
`local_files_only=True`, while the launcher defaults `HF_HUB_CACHE` to
`<project>/checkpoints/model`. The downloader currently installs the merged
model, FRS decoder, and tactile encoder there, but not the tokenizer's standard
Hugging Face cache entry. Consequently `AutoTokenizer.from_pretrained` cannot
resolve the repository offline.

The same tokenizer loads successfully from the existing user cache at commit
`7b375e1b73b11138ff12fe22c8f2822d8fe03467`. A test cache containing only the
eight files below loads successfully and occupies about 4.7 MB, so the VLM
weights are not required for tokenization.

## Selected design

Extend `deploy_smolvla/scripts/download.sh` with a fourth pinned deployment
asset:

- Repository: `HuggingFaceTB/SmolVLM2-500M-Video-Instruct`
- Revision: `7b375e1b73b11138ff12fe22c8f2822d8fe03467`
- Cache root: `${CHECKPOINT_ROOT}/model`
- Required files:
  - `config.json`
  - `tokenizer_config.json`
  - `tokenizer.json`
  - `special_tokens_map.json`
  - `added_tokens.json`
  - `chat_template.json`
  - `merges.txt`
  - `vocab.json`

The repository and revision remain editable beside the other pinned inputs at
the top of the script. The download must use an exact allowlist and must not
fetch `model.safetensors`, processor assets, optimizer state, or arbitrary
repository files.

## Cache and provenance layout

The files use the standard Hugging Face cache layout under
`${CHECKPOINT_ROOT}/model`:

```text
models--HuggingFaceTB--SmolVLM2-500M-Video-Instruct/
  refs/main
  snapshots/7b375e1b73b11138ff12fe22c8f2822d8fe03467/
```

After a successful pinned download, the downloader atomically writes
`refs/main` to the pinned revision. This is necessary because the checkpoint's
preprocessor requests the repository by name without passing a revision;
offline Hugging Face resolution therefore follows `refs/main`.

The downloader owns this project-local ref. It never changes the user's
`~/.cache/huggingface` tree.

## Completeness and rerun behavior

The tokenizer is complete only when all of these conditions hold:

1. `refs/main` contains the exact configured revision.
2. All eight allowlisted snapshot files exist and are non-empty.
3. With `HF_HUB_CACHE=${CHECKPOINT_ROOT}/model`, `HF_HUB_OFFLINE=1`, and
   `TRANSFORMERS_OFFLINE=1`, the project Python can successfully call
   `AutoTokenizer.from_pretrained(repo_id, local_files_only=True)`.

When complete, the script prints an explicit `skip:` message. A missing file,
wrong ref, malformed tokenizer, or incompatible configuration triggers a
cache-backed refresh of only the tokenizer asset. Existing model, FRS, and
encoder assets remain independently skippable.

## Failure handling and output

- A delegated tokenizer download failure exits nonzero and names both the
  repository and `${CHECKPOINT_ROOT}/model` destination.
- Failure to validate after download exits nonzero and reports the tokenizer
  cache destination.
- The final deployment summary includes
  `HF_HUB_CACHE: ${CHECKPOINT_ROOT}/model` in addition to the three checkpoint
  paths.
- `start_remote_client.sh` keeps its current default cache behavior; after this
  asset is installed, `bash deploy_smolvla/scripts/start_frs.sh` works without
  a manual cache override.

## Tests

Extend the existing black-box downloader suite to cover:

1. Exact tokenizer repository, revision, cache directory, and eight-file
   allowlist in the delegated command.
2. First run downloads four independent assets in the intended order and
   produces a usable project-local tokenizer cache.
3. A complete tokenizer is skipped, including on an immediate second run.
4. A wrong `refs/main`, missing file, or offline-load failure refreshes only
   the tokenizer.
5. A tokenizer download failure names the asset and cache root and prevents a
   false success summary.
6. The final summary prints the exact `HF_HUB_CACHE` consumed by the launcher.
7. The real project cache passes an offline `AutoTokenizer` smoke test after
   executing the downloader.

Test fixtures may build a tiny valid tokenizer/cache or provide an isolated
test-only `transformers` implementation, but the production smoke test must use
the installed Transformers package.

## Non-goals

- Downloading the SmolVLM model weights.
- Enabling online model loading at deployment time.
- Changing `allow_download` in the deployment YAML.
- Rewriting `policy_preprocessor.json` to an absolute local tokenizer path.
- Modifying the user-level Hugging Face cache.
