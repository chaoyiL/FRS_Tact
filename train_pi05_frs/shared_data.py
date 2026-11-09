"""Read task-specific frozen VLA caches through one masked 20D FRS interface."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from train_pi05_frs.pi05_cache.cache import CachedPairs

CANONICAL_TACTILE_KEYS = (
    "tactile_left_0",
    "tactile_right_0",
    "tactile_left_1",
    "tactile_right_1",
)


def encoder_fingerprint(directory: str | Path) -> str:
    """Same checkpoint/params byte identity as the native tactile cache contract."""
    directory = Path(directory).resolve()
    metadata = json.loads((directory / "checkpoint.json").read_text())
    params_name = metadata.get("params_file", "params.npz")
    params = (directory / params_name).resolve()
    if Path(params_name).is_absolute() or not params.is_relative_to(directory):
        raise ValueError("Encoder params_file must remain within its checkpoint")
    digest = hashlib.sha256()
    for name, path in (("checkpoint.json", directory / "checkpoint.json"), (params_name, params)):
        digest.update(name.encode())
        with path.open("rb") as stream:
            while chunk := stream.read(8 * 1024 * 1024):
                digest.update(chunk)
    return digest.hexdigest()


def canonical_action(action: np.ndarray, active_wrists: list[str]) -> np.ndarray:
    """Map native right-arm 10D or dual-arm 20/32D actions, then mask idle wrists."""
    value = np.asarray(action, dtype=np.float32)
    if value.shape[-1] not in (10, 20, 32):
        raise ValueError(f"Unsupported native action width: {value.shape[-1]}")
    if value.shape[-1] == 10 and active_wrists != ["right"]:
        raise ValueError("Native 10D caches can only supply the physical right wrist")
    result = np.zeros((*value.shape[:-1], 20), dtype=np.float32)
    if value.shape[-1] == 10:
        result[..., 10:20] = value
    else:
        result[...] = value[..., :20]
    mask = np.repeat(["left" in active_wrists, "right" in active_wrists], 10)
    return np.where(mask, result, 0).astype(np.float32)


def canonical_tactile(tokens: np.ndarray, tactile_keys: list[str], active_wrists: list[str]) -> np.ndarray:
    keys = [key.rsplit(".", 1)[-1] for key in tactile_keys]
    if len(set(keys)) != len(keys) or len(keys) != tokens.shape[-2]:
        raise ValueError("Invalid tactile token/key mapping")
    result = np.zeros((*tokens.shape[:-2], 4, tokens.shape[-1]), dtype=np.float32)
    for i, key in enumerate(CANONICAL_TACTILE_KEYS):
        if ("left" if i < 2 else "right") in active_wrists:
            if key not in keys:
                raise ValueError(f"Required tactile sensor missing: {key}")
            result[..., i, :] = tokens[..., keys.index(key), :]
    return result


def episode_starts(root: Path) -> np.ndarray:
    import pyarrow.parquet as pq

    paths = sorted((root / "meta/episodes").rglob("*.parquet"))
    if not paths:
        raise FileNotFoundError(f"No LeRobot v3 episode metadata under {root}")
    rows = [
        row
        for path in paths
        for row in pq.read_table(path, columns=["episode_index", "dataset_from_index"]).to_pylist()
    ]
    rows.sort(key=lambda row: row["episode_index"])
    if [row["episode_index"] for row in rows] != list(range(len(rows))):
        raise ValueError("Episode IDs must be contiguous in the prepared v3 dataset")
    return np.asarray([row["dataset_from_index"] for row in rows], dtype=np.int64)


class SharedTaskData:
    """Keep source normalization and dataset indices local; share only canonical tensors."""

    def __init__(
        self,
        sources: list[dict],
        *,
        encoder_sha256: str,
        tactile_window: int,
        history_stride: int,
        num_train_episodes: int,
        num_val_episodes: int,
        seed: int,
    ):
        self.sources = sources
        self.window = tactile_window
        self.stride = history_stride
        self.caches, self.embeddings, self.tactile_meta, self.starts = [], [], [], []
        self.rows = {"train": [], "val": []}
        provenance = []
        names = [source["name"] for source in sources]
        if len(set(names)) != len(names):
            raise ValueError("Shared source names must be unique")
        self.horizon = None
        self.embedding_dim = None
        self.selection = {}
        for source_i, source in enumerate(sources):
            wrists = source["active_wrists"]
            if not wrists or len(set(wrists)) != len(wrists) or set(wrists) - {"left", "right"}:
                raise ValueError(f"Invalid active_wrists for {source['name']}")
            cache = CachedPairs(Path(source["action_cache"]))
            if cache.manifest.get("configuration", {}).get("base_model") != "pi0.5":
                raise ValueError("Shared FRS requires frozen Pi0.5 action caches")
            canonical_action(np.zeros((1, int(cache.manifest["action_dim"]))), wrists)
            horizon = int(cache.manifest["action_horizon"])
            expected_actions = (
                int(cache.manifest["sample_count"]),
                horizon,
                int(cache.manifest["action_dim"]),
            )
            if any(cache.arrays[key].shape != expected_actions for key in ("x_base", "target", "gt_action")):
                raise ValueError(f"Malformed native action cache shape for {source['name']}")
            if self.horizon not in (None, horizon):
                raise ValueError("All tasks must use the same action horizon")
            self.horizon = horizon
            directory = Path(source["tactile_cache"])
            meta = json.loads((directory / "metadata.json").read_text())
            if (
                meta.get("status") != "complete"
                or meta.get("encoder_sha256") != encoder_sha256
                or meta.get("repo_id") != source["repo_id"]
            ):
                raise ValueError(f"Tactile cache provenance mismatch for {source['name']}")
            embeddings = np.load(directory / "embeddings.npy", mmap_mode="r", allow_pickle=False)
            expected = (
                int(meta["total_frames"]),
                int(meta["num_tactile_tokens"]),
                int(meta["embedding_dim"]),
            )
            if embeddings.shape != expected or int(meta.get("completed_frames", -1)) != expected[0]:
                raise ValueError(f"Incomplete/malformed tactile cache for {source['name']}")
            if self.embedding_dim not in (None, expected[-1]):
                raise ValueError("All tasks must use the same tactile encoder width")
            self.embedding_dim = expected[-1]
            canonical_tactile(np.zeros((1, expected[1], expected[2])), meta["tactile_keys"], wrists)
            starts = episode_starts(Path(source["dataset_root"]))
            configuration = cache.manifest["configuration"]
            if configuration.get("dataset_repo_id") != source["repo_id"]:
                raise ValueError(f"Action cache dataset mismatch for {source['name']}")
            if source.get("checkpoint") and configuration.get("checkpoint_dir") != source["checkpoint"]:
                raise ValueError(f"Action cache Pi0.5 checkpoint mismatch for {source['name']}")
            if (
                source.get("norm_stats_asset_id")
                and configuration.get("norm_stats_asset_id") != source["norm_stats_asset_id"]
            ):
                raise ValueError(f"Action cache normalization mismatch for {source['name']}")
            if meta.get("revision") != configuration.get("dataset_revision"):
                raise ValueError(f"Action/tactile cache revision mismatch for {source['name']}")
            if Path(meta["dataset_root"]).resolve() != Path(source["dataset_root"]).resolve():
                raise ValueError(f"Tactile dataset root mismatch for {source['name']}")
            if Path(configuration["dataset_root"]).resolve() != Path(source["dataset_root"]).resolve():
                raise ValueError(f"Action dataset root mismatch for {source['name']}")
            self.caches.append(cache)
            self.embeddings.append(embeddings)
            self.tactile_meta.append(meta)
            self.starts.append(starts)
            self.selection[source["name"]] = {}
            seen = set()
            for split, budget in (("train", num_train_episodes), ("val", num_val_episodes)):
                indices = cache.indices(split)
                episodes = np.asarray(cache.arrays["episode_index"][indices], dtype=np.int64)
                unique = np.unique(episodes)
                if len(unique) < budget:
                    raise ValueError(
                        f"{source['name']}: need {budget} {split} episodes, cache has {len(unique)}"
                    )
                chosen = np.sort(
                    np.random.default_rng(
                        np.random.SeedSequence([seed, source_i, int(split == "val")])
                    ).permutation(unique)[:budget]
                )
                if seen & set(chosen.tolist()):
                    raise ValueError(f"Train/validation episode leakage in {source['name']}")
                seen.update(chosen.tolist())
                self.selection[source["name"]][split] = chosen.tolist()
                self.rows[split].extend((source_i, int(i)) for i in indices[np.isin(episodes, chosen)])
            provenance.append(
                {
                    "source": source,
                    "manifest": cache.manifest,
                    "tactile": meta,
                    "selection": self.selection[source["name"]],
                }
            )
        self.fingerprint = hashlib.sha256(json.dumps(provenance, sort_keys=True).encode()).hexdigest()

    def batch(self, rows: list[tuple[int, int]], *, tau: float, temperature: float) -> dict[str, np.ndarray]:
        if temperature <= 0:
            raise ValueError("gate temperature must be positive")
        output = {
            key: []
            for key in (
                "x_base",
                "predicted",
                "gt",
                "tactile",
                "action_mask",
                "token_mask",
                "gates",
                "source",
            )
        }
        offsets = np.arange(self.window - 1, -1, -1) * self.stride
        for source_i, index in rows:
            source, cache = self.sources[source_i], self.caches[source_i]
            wrists = source["active_wrists"]
            arrays = cache.arrays
            frame, episode = int(arrays["dataset_index"][index]), int(arrays["episode_index"][index])
            if episode < 0 or episode >= len(self.starts[source_i]):
                raise ValueError("Action cache episode out of range")
            start = int(self.starts[source_i][episode])
            end = (
                int(self.starts[source_i][episode + 1])
                if episode + 1 < len(self.starts[source_i])
                else len(self.embeddings[source_i])
            )
            if not start <= frame < end:
                raise ValueError("Action cache frame outside its dataset")
            indices = np.maximum(frame - offsets, start)
            native = np.asarray(self.embeddings[source_i][indices], dtype=np.float32)
            keys = self.tactile_meta[source_i]["tactile_keys"]
            tactile = canonical_tactile(native, keys, wrists)
            baseline = canonical_tactile(np.asarray(self.embeddings[source_i][start]), keys, wrists)
            current = tactile[-1]

            def normalize(x):
                return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-8)

            change = (
                (1 - np.sum(normalize(current) * normalize(baseline), axis=-1)).reshape(2, 2).mean(axis=-1)
            )
            active = np.asarray(["left" in wrists, "right" in wrists])
            gates = 1 / (1 + np.exp(-np.clip((change - tau) / temperature, -60, 60)))
            for key, native_key in (("x_base", "x_base"), ("predicted", "target"), ("gt", "gt_action")):
                output[key].append(canonical_action(arrays[native_key][index], wrists))
            output["tactile"].append(tactile)
            output["action_mask"].append(np.repeat(active, 10))
            output["token_mask"].append(np.repeat(active, 2))
            output["gates"].append(gates * active)
            output["source"].append(source_i)
        result = {
            key: np.asarray(values, dtype=np.int32 if key == "source" else np.float32)
            for key, values in output.items()
        }
        if any(not np.isfinite(value).all() for value in result.values()):
            raise ValueError("Shared FRS batch contains non-finite active data")
        return result

    def batches(self, split: str, *, batch_size: int, seed: int, tau: float, temperature: float):
        rows = self.rows[split]
        order = np.arange(len(rows))
        if split == "train":
            order = np.random.default_rng(seed).permutation(order)
        for start in range(0, len(order), batch_size):
            yield self.batch(
                [rows[i] for i in order[start : start + batch_size]], tau=tau, temperature=temperature
            )
