# Offline Vision Cache and Four-GPU VT-SmolVLA Training Design

Date: 2026-08-09

## Goal

Remove the sustained CPU data bottleneck from VT-SmolVLA training by preparing every expensive deterministic input offline. The target server has four `NVIDIA RTX PRO 6000 Blackwell Server Edition` GPUs with about 96 GiB each. K8 and K21 must train concurrently, using two GPUs per experiment.

This remains a laboratory workflow. It does not add remote commit pinning, large-file hashes, publishing provenance, or deployment changes.

## Filesystem layout

Program files and large experiment assets are deliberately separated:

```text
repository:    /home/ljl/FRS_Tact
virtualenv:    /home/ljl/.venvs/frs_tact
storage root:  /DATA/ljl/substage
```

All large or frequently written assets live below the storage root:

```text
/DATA/ljl/substage/lerobot_v30
/DATA/ljl/substage/checkpoints
/DATA/ljl/substage/tactile_embeddings
/DATA/ljl/substage/smolvla_training_cache
/DATA/ljl/substage/normalization_protocols
/DATA/ljl/substage/outputs
/DATA/ljl/substage/logs
/DATA/ljl/substage/huggingface
/DATA/ljl/substage/tmp
```

`scripts/setup_env.sh` writes these resolved values to `/home/ljl/FRS_Tact/.env.frs`. The download, precompute, and training launchers source that file and must not reconstruct paths from the repository location or fall back to `/workspace`.

## Current bottleneck

The tactile ResNet path is already offline through the existing `[N,4,512]` tactile embedding cache. The remaining training loader still performs expensive work for every sampled frame:

- decode two compressed RGB observations from the LeRobot v3 source;
- apply torchvision RGB augmentation;
- convert and resize RGB to the model input size;
- query and stack a 20-step action chunk;
- tokenize the task text;
- run the frozen vision encoder and frozen connector;
- synchronously move the resulting batch to the two JAX devices.

Increasing `num_workers` alone does not remove this work and creates more process, thread, shared-memory, and CPU contention when K8 and K21 run together.

## Chosen approach

The user accepts disabling online RGB augmentation for maximum throughput. The offline stage therefore caches the output of the frozen vision encoder and connector, rather than decoded RGB pixels.

The rejected alternatives are:

- decoded-RGB cache: simpler, but retains resize and frozen vision forward compute;
- fixed-batch cache or WebDataset stream: harms exact shuffle, weighted sampling, validation subsets, and resume position;
- worker/prefetch tuning only: does not remove the root CPU work.

## Offline cache schema

Each dataset has an independent directory under `/DATA/ljl/substage/smolvla_training_cache/<namespace>/<dataset>/`:

```text
metadata.json
vision_tokens.uint16.npy
state.npy
actions.npy
action_is_pad.npy
language_tokens.npy
language_masks.npy
episode_index.npy
frame_index.npy
progress.json
```

Logical array shapes are:

```text
vision_tokens   [N, 2, 64, 960] bfloat16
state           [N, 20]         float32
actions         [N, 20, 20]     float32
action_is_pad   [N, 20]         bool
language_tokens [N, 48]         int32
language_masks  [N, 48]         bool
episode_index   [N]             integer
frame_index     [N]             integer
```

NumPy does not preserve `ml_dtypes.bfloat16` as a portable native NPY dtype. Vision tokens are therefore stored as the exact BF16 bit pattern in `uint16`, and viewed as BF16 after loading. No FP16 conversion is permitted.

The existing tactile cache remains separate and is not duplicated. State and action remain raw FP32 values; the existing train-only normalizer continues to run online because its cost is negligible and keeping it online avoids coupling the cache to a particular split artifact.

At roughly 341,000 total frames, the new vision cache is expected to use about 78 GiB, with the other arrays adding less than 1 GiB. The server has several terabytes available.

## Cache generation

A new entry point, `tools/precompute_smolvla_training_cache.py`, will:

1. read the same K8/K21 model and dataset configuration used by training;
2. reject enabled image augmentation;
3. reject any non-frozen `vision` or `connector` module mode;
4. load the same effective checkpoint parameters as training;
5. decode the two canonical RGB cameras without augmentation;
6. apply the existing resize and normalization path;
7. call the existing frozen `embed_image` path;
8. write raw connector outputs as BF16 bits;
9. write raw state, action chunk, padding, language tokens, and frame identity;
10. validate shapes, dtypes, frame coverage, and finite numeric values before marking the dataset complete.

Each dataset is written independently and supports interruption recovery through `progress.json`. A cache is usable only when every expected row is present and the final status is `complete`. Generation never replaces a complete cache in place.

The four datasets may be prepared concurrently, one GPU per dataset. The final launcher waits for and validates all four jobs before starting either experiment.

Cache metadata records only the compatibility information needed by this experiment: schema version, repository identity, total frames, camera order, image/vision dimensions, chunk size, action/state shape, tokenizer length, checkpoint source, module modes, and logical/storage dtypes. It does not add remote SHA or publishing provenance.

## Training loader integration

A map-style `OfflineTrainingCacheDataset` will expose the same logical sample ordering as the current episode-filtered LeRobot dataset. It will map selected episodes to absolute cache rows while preserving:

- the persisted train/validation split;
- deterministic epoch shuffle;
- source weights;
- fixed validation subset indices;
- `start_batch` resume position;
- action padding semantics.

The loader reads vision tokens and the small numeric arrays from memmap, then reads the existing tactile embedding row. Online work is limited to state/action normalization, deterministic sampling, flow noise, modality dropout, host-to-device transfer, and optimizer computation.

Each training process uses two lightweight cache workers and a two-batch host prefetch queue. Training logs include data-wait time so cache effectiveness can be measured independently of model compute.

## Model integration

`JaxSmolVLA` gains an optional `vision_embeddings` input with shape `[B,Ncam,64,960]`.

- Live inference and uncached training continue to use `images` and `embed_image`.
- Cached training uses `vision_embeddings` and skips `embed_image`.
- Supplying both or neither input is an error.
- Cached tokens receive the same hidden-size scaling, camera mask, concatenation order, attention segmentation, and RoPE position handling as online image embeddings.
- Validation rollout accepts the same cached tokens.

No trainable parameter is added, removed, or renamed. Portable checkpoint parameter schema and live policy behavior remain unchanged.

## Configuration

Both K8 and K21 configurations use the same block:

```yaml
offline_training_cache:
  enabled: true
  root: /DATA/ljl/substage/smolvla_training_cache
  dtype: bfloat16
  precompute_batch_size: 64
  precompute_num_workers: 8
  loader_num_workers: 2
  host_prefetch_batches: 2
```

Both configurations explicitly set `image_transforms.enable: false`. Offline mode fails before training when vision or connector is not frozen, augmentation is enabled, cache metadata is incompatible, or any dataset is incomplete.

The related active paths are:

```text
datasets:       /DATA/ljl/substage/lerobot_v30/KaiyueChen/pick_tube_01..04
encoder:        /DATA/ljl/substage/checkpoints/encoder_ckpt_05
tactile cache:  /DATA/ljl/substage/tactile_embeddings
normalization:  /DATA/ljl/substage/normalization_protocols/pick_tube_vt_k8_k21
K8 output:      /DATA/ljl/substage/outputs/vtsmolvla_tactile_repeat16
K21 output:     /DATA/ljl/substage/outputs/vtsmolvla_tactile_repeat32
logs:           /DATA/ljl/substage/logs
```

K8 and K21 continue to use BF16 compute and share the tactile cache, offline training cache, source datasets, split, and normalization protocol. Their repeat factor and output identities remain distinct.

## Four-GPU execution

The supported hardware profile is:

```text
4 x NVIDIA RTX PRO 6000 Blackwell Server Edition
driver 595.84 or compatible newer driver
```

`scripts/setup_env.sh` validates all four GPUs and verifies that both PyTorch and JAX see four CUDA devices. It retains the existing CUDA, cuDNN, NCCL, and libdevice checks, updated for a four-device collective smoke.

The launcher runs these stages:

```text
prepare/validate tactile cache once
prepare/validate all four offline training caches once
start K8  with CUDA_VISIBLE_DEVICES=0,1
start K21 with CUDA_VISIBLE_DEVICES=2,3
```

K8 and K21 are separate single-process JAX data-parallel jobs. `torchrun`, MPI, and one-process-per-GPU launchers are not used.

The default tmux sessions are:

```text
vtsmolvla_k8
vtsmolvla_k21
```

Each experiment has independent logs and output directories. If one experiment fails, the other continues; the launcher reports both final statuses. Shared cache and normalization artifacts are read-only after publication. Existing atomic normalization publication handles identical concurrent initialization.

## Failure handling

The workflow fails closed when:

- fewer or more than four target GPUs are visible during the default four-GPU launch;
- either child process sees anything other than its assigned two GPUs;
- the GPU model is not the approved RTX PRO 6000 Blackwell Server Edition profile;
- vision or connector is trainable;
- RGB augmentation is enabled;
- a cache is incomplete or incompatible;
- BF16 storage cannot be restored bit-exactly;
- train and cache camera order, chunk size, shapes, or frame coverage differ.

An interrupted precompute leaves resumable partial state, not a complete marker. The launcher never starts training from a partial cache and never deletes an existing cache or checkpoint automatically.

## Verification

Automated tests cover:

- BF16-to-`uint16`-to-BF16 bit-exact round trip;
- online image embedding versus cached-token prefix parity with augmentation disabled;
- cache shape, dtype, frame coverage, completion, interruption, and resume behavior;
- mismatched camera order, model mode, chunk size, tokenizer length, and incomplete state rejection;
- split, weighted sampling, validation subset, padding, and resume-order parity;
- K8/K21 cache sharing;
- four-GPU profile parsing and process isolation;
- concurrent K8/K21 launch and independent failure handling;
- shell syntax and existing uncached/live-policy regressions.

Server acceptance requires:

1. one dataset precompute smoke and resume smoke;
2. complete cache generation for all four datasets;
3. concurrent one-step K8 and K21 runs on GPU pairs `0,1` and `2,3`;
4. finite loss and gradient metrics;
5. no cache, split, normalization, or checkpoint error;
6. data-wait time and GPU utilization measurements demonstrating that the loader no longer stalls both training jobs.

These smokes prove runtime compatibility and throughput behavior, not paper-level model quality.
