import os
import random
import re
from pathlib import Path

import numpy as np
import torch

_EPOCH_CHECKPOINT_RE = re.compile(r"deco_stage1_epoch_(\d+)\.pt$")
_EPOCH_CHECKPOINT_SUFFIXES = (".pt", ".ts", ".ts.json")


def capture_rng_state() -> dict:
    state = {
        "python": random.getstate(),
        "numpy": {
            "name": np.random.get_state()[0],
            "keys": np.random.get_state()[1].tolist(),
            "pos": np.random.get_state()[2],
            "has_gauss": np.random.get_state()[3],
            "cached_gaussian": np.random.get_state()[4],
        },
        "torch_cpu": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: dict) -> None:
    random.setstate(state["python"])
    numpy_state = state["numpy"]
    np.random.set_state((
        numpy_state["name"],
        np.asarray(numpy_state["keys"], dtype=np.uint32),
        numpy_state["pos"],
        numpy_state["has_gauss"],
        numpy_state["cached_gaussian"],
    ))
    # ``load_checkpoint(..., map_location=device)`` maps every tensor in the
    # payload, including serialized RNG states, onto the local CUDA device.
    # PyTorch's RNG restoration APIs require CPU ByteTensors even when the
    # restored generator belongs to CUDA.
    torch.set_rng_state(state["torch_cpu"].detach().cpu())
    if torch.cuda.is_available() and "torch_cuda" in state:
        # A topology-migrated checkpoint can contain generator states for more
        # GPUs than are visible in the resumed process (for example 6 -> 4).
        # ``set_rng_state_all`` indexes the current generators for every saved
        # state and raises IndexError when the saved list is longer.  Restore
        # the generators that exist in the current topology; when scaling up,
        # any additional generators retain the deterministic seed established
        # during process initialization.
        device_count = torch.cuda.device_count()
        cuda_states = state["torch_cuda"][:device_count]
        torch.cuda.set_rng_state_all(
            [rng_state.detach().cpu() for rng_state in cuda_states]
        )


def atomic_torch_save(payload: dict, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("wb") as handle:
            torch.save(payload, handle)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)


def load_checkpoint(path: str | Path, map_location) -> dict:
    return torch.load(path, map_location=map_location, weights_only=True)


def prune_old_checkpoints(output_dir: str | Path, keep_last: int) -> list[str]:
    """Keep only the newest ``keep_last`` per-epoch checkpoints.

    Removes older ``deco_stage1_epoch_<N>.{pt,ts,ts.json}`` triplets, keeping the
    ``keep_last`` highest epoch numbers. The named ``latest``/``best`` artifacts
    are never touched. ``keep_last <= 0`` disables pruning.
    """
    if keep_last <= 0:
        return []
    output_dir = Path(output_dir)
    epochs: list[tuple[int, Path]] = []
    for path in output_dir.glob("deco_stage1_epoch_*.pt"):
        match = _EPOCH_CHECKPOINT_RE.search(path.name)
        if match:
            epochs.append((int(match.group(1)), output_dir / f"deco_stage1_epoch_{int(match.group(1))}"))
    epochs.sort(key=lambda item: item[0])
    removed: list[str] = []
    for _, stem in epochs[:-keep_last]:
        for suffix in _EPOCH_CHECKPOINT_SUFFIXES:
            target = stem.with_name(stem.name + suffix)
            if target.is_file():
                target.unlink()
                removed.append(target.name)
    return removed
