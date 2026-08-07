# Local Checkpoints Layout Design

## Goal

Store future VLA and tactile encoder downloads inside the project while leaving
existing Hugging Face cache files untouched.

## Directory layout

```text
checkpoints/
├── model/      # Hugging Face Hub cache for VLA checkpoints and model assets
└── encoder/    # Explicit tactile encoder checkpoint downloads
```

The complete `checkpoints/` tree is local runtime data and must be ignored by
Git. Empty directories are created by the relevant launch/download scripts,
not tracked with placeholder files.

## VLA download behavior

The remote deployment launch path defaults `HF_HUB_CACHE` to
`<project>/checkpoints/model` and creates the directory before Python starts.
An existing caller-provided `HF_HUB_CACHE` remains authoritative, so deployments
can still use another disk without modifying repository files.

Both supported launch paths must behave consistently:

- `deploy_smolvla/start_vtsmolvla.sh` followed by `run_client.sh`;
- direct execution of `deploy_smolvla/run_client.sh`.

The check-only output reports the resolved model cache location. Existing files
under the user's current Hugging Face cache are neither copied nor deleted.

## Encoder download behavior

`download_ckpt.py` defaults encoder output to
`<project>/checkpoints/encoder/encoder_ckpt_06`. Callers may continue to override
it with `--output-dir`. The shell wrapper reports the same repository and output
directory as the Python defaults, removing its current `encoder_ckpt_05` versus
`encoder_ckpt_06` mismatch.

## Error handling

Directory creation uses the existing fail-fast shell behavior. An unwritable
checkpoint directory stops before model resolution or robot connection. Explicit
environment and CLI overrides retain their existing precedence.

## Verification

Automated tests cover:

1. `checkpoints/` is ignored by Git;
2. the deployment launcher selects `<project>/checkpoints/model` by default;
3. an explicit `HF_HUB_CACHE` is preserved;
4. check-only output reports the selected cache;
5. the encoder downloader defaults to
   `<project>/checkpoints/encoder/encoder_ckpt_06`;
6. no existing cache or checkpoint is moved or removed.

