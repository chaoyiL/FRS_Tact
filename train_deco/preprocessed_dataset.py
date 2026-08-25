import hashlib
import json
import os
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


DATASET_FORMAT = "sudo-deco-stage1-preprocessed-v1"


def _skip_sha256() -> bool:
    # Reuse the same env the preprocessing pipeline uses to skip shard/manifest
    # sha256 generation. When set, dataset-load verification (verify_preprocessed_dataset
    # and PreprocessedDECODataset.__init__) skips the expensive per-file sha256
    # recomputation and only checks file existence + byte size. On the full 4431-episode
    # cloth-opt snapshot this turns a ~tens-of-minutes full-checksum scan of 146k npy
    # files into a few seconds of stat() calls, at the cost of not detecting bit-rot
    # (size/shape still guarded; content corruption would slip through).
    return os.environ.get("PREPROCESS_SKIP_SHA256", "0") == "1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_preprocessed_dataset(dataset_dir: str) -> None:
    root = Path(dataset_dir)
    ready_path = root / "READY"
    manifest_path = root / "manifest.json"
    if not ready_path.is_file() or not manifest_path.is_file():
        raise ValueError(f"Preprocessed dataset is incomplete: {root}")
    ready = json.loads(ready_path.read_text())
    if not _skip_sha256() and sha256_file(manifest_path) != ready.get("manifest_sha256"):
        raise ValueError(f"Preprocessed dataset manifest checksum mismatch: {root}")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("format") != DATASET_FORMAT:
        raise ValueError(
            f"Unsupported DECO Stage 1 dataset format: {manifest.get('format')!r}"
        )
    statistics = manifest.get("statistics")
    if statistics:
        statistics_path = root / statistics["artifact"]
        if (
            not statistics_path.is_file()
            or (
                not _skip_sha256()
                and sha256_file(statistics_path) != statistics.get("artifact_sha256")
            )
        ):
            raise ValueError(
                f"Preprocessed dataset statistics checksum mismatch: {statistics_path}"
            )
    for split in ("train", "val"):
        for shard in manifest["splits"][split]["shards"]:
            shard_dir = root / shard["path"]
            for name, expected in shard["files"].items():
                path = shard_dir / f"{name}.npy"
                if not path.is_file() or path.stat().st_size != int(expected["bytes"]):
                    raise ValueError(f"Preprocessed shard size mismatch: {path}")
                if not _skip_sha256() and sha256_file(path) != expected["sha256"]:
                    raise ValueError(f"Preprocessed shard checksum mismatch: {path}")


class PreprocessedDECODataset(Dataset):
    """Read validated DECO Stage 1 samples from immutable mmap shards."""

    def __init__(
        self,
        dataset_dir: str,
        split: str,
        limit: int | None = None,
        action_chunk_size: int | None = None,
    ):
        self.root = Path(dataset_dir)
        ready_path = self.root / "READY"
        if not ready_path.is_file():
            raise ValueError(f"Preprocessed dataset is incomplete (READY missing): {self.root}")
        manifest_path = self.root / "manifest.json"
        ready = json.loads(ready_path.read_text())
        if not _skip_sha256() and sha256_file(manifest_path) != ready.get("manifest_sha256"):
            raise ValueError(f"Preprocessed dataset manifest checksum mismatch: {self.root}")
        self.manifest = json.loads(manifest_path.read_text())
        if self.manifest.get("format") != DATASET_FORMAT:
            raise ValueError(
                f"Unsupported preprocessed dataset format: {self.manifest.get('format')!r}"
            )
        if split not in ("train", "val"):
            raise ValueError(f"split must be train or val, got {split!r}")
        self.split = split
        source_contract = self.manifest["contract"]
        self.source_chunk_size = int(source_contract["chunk_size"])
        self.action_chunk_size = (
            self.source_chunk_size
            if action_chunk_size is None
            else int(action_chunk_size)
        )
        if not 1 <= self.action_chunk_size <= self.source_chunk_size:
            raise ValueError(
                "action_chunk_size must be in [1, source chunk_size]: "
                f"got {self.action_chunk_size}, source={self.source_chunk_size}"
            )
        # Present an effective training contract without mutating the immutable
        # source manifest. A chunk-32 artifact can therefore provide identical
        # image/observation anchors and the first N future actions to smaller
        # chunk policies without duplicating the image shards.
        self.metadata = dict(source_contract)
        self.metadata["chunk_size"] = self.action_chunk_size
        self.stats = {
            key: np.asarray(value, dtype=np.float32)
            for key, value in self.manifest["stats"].items()
        }
        self.normalized = bool(self.manifest.get("normalized", True))
        self.shards = self.manifest["splits"][split]["shards"]
        task_ids = sorted({
            str(shard.get("task_id", "default")) for shard in self.shards
        })
        self.task_to_index = {task_id: index for index, task_id in enumerate(task_ids)}
        self.task_ids = task_ids
        self.index = []
        for shard_index, shard in enumerate(self.shards):
            self.index.extend((shard_index, row) for row in range(int(shard["samples"])))
        if limit is not None:
            self.index = self.index[:limit]
        if not self.index:
            raise ValueError(f"No preprocessed samples in split={split}: {self.root}")
        self._arrays = {}
        # LRU cap for mmap'd shards. Each shard opens 4 npy files (observation,
        # action, images, is_pad) via mmap_mode="r"; without a cap, ALL ~33k
        # train shards get mmap'd once and stay open forever (~131k file
        # descriptors per worker), which exhausts the node fs.file-max and
        # crashes DDP with EMFILE "Too many open files in system". Capping the
        # cache to a recent working set closes evicted shards' mmaps (array
        # .close() releases the fd) so the fd count stays bounded (~cap*4).
        self._shard_cache_size = int(os.environ.get("SHARD_CACHE_SIZE", "32"))
        self._shard_lru = OrderedDict()

    def _open_shard(self, shard_index: int) -> dict:
        if shard_index in self._shard_lru:
            self._shard_lru.move_to_end(shard_index)
            return self._shard_lru[shard_index]
        shard = self.shards[shard_index]
        shard_dir = self.root / shard["path"]
        arrays = {
            name: np.load(shard_dir / f"{name}.npy", mmap_mode="r", allow_pickle=False)
            for name in ("observation", "action", "images", "is_pad")
        }
        for name, array in arrays.items():
            expected = shard["files"][name]
            if list(array.shape) != expected["shape"] or str(array.dtype) != expected["dtype"]:
                raise ValueError(
                    f"Preprocessed shard contract mismatch: shard={shard['path']}, array={name}"
                )
        self._shard_lru[shard_index] = arrays
        # Evict the least-recently-used shard and close its mmaps to release
        # file descriptors. np.lib.format.open_memmap returns a memmap object
        # whose .close() unmaps the file and frees the fd.
        while len(self._shard_lru) > self._shard_cache_size:
            _evicted_index, evicted = self._shard_lru.popitem(last=False)
            for array in evicted.values():
                try:
                    array.close()
                except Exception:
                    pass
        return arrays

    def __len__(self):
        return len(self.index)

    def __getitem__(self, index: int):
        shard_index, row = self.index[index]
        arrays = self._open_shard(shard_index)
        observation = np.array(arrays["observation"][row], dtype=np.float32, copy=True)
        action = np.array(
            arrays["action"][row, : self.action_chunk_size],
            dtype=np.float32,
            copy=True,
        )
        if not self.normalized:
            observation = (observation - self.stats["observation_mean"]) / self.stats["observation_std"]
            action = (action - self.stats["action_mean"]) / self.stats["action_std"]
        return {
            "observation": torch.from_numpy(observation),
            "action": torch.from_numpy(action),
            "images": torch.from_numpy(np.array(arrays["images"][row], copy=True)).float().div_(255.0),
            "is_pad": torch.from_numpy(
                np.array(
                    arrays["is_pad"][row, : self.action_chunk_size],
                    copy=True,
                )
            ),
            "task_index": torch.tensor(
                self.task_to_index[str(self.shards[shard_index].get("task_id", "default"))],
                dtype=torch.long,
            ),
        }
