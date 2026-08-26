"""Decoder-only PyTorch training over frozen Pi0.5 and tactile caches."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from train_baseline_pi05.action_cache import ActionCache
from train_baseline_pi05.checkpoint import load_decoder_checkpoint, save_best_checkpoint, save_last_checkpoint
from train_baseline_pi05.data import BaselineCacheDataset, make_loader
from train_baseline_pi05.evaluate import evaluate_decoder
from train_baseline_pi05.model import DirectDecoderConfig, DirectTactileActionDecoder, masked_smooth_l1
from train_baseline_pi05.tactile_cache import TactileEmbeddingCache


def _field(config: Any, name: str, default: Any = None) -> Any:
    value = config
    for component in name.split("."):
        value = getattr(value, component, default)
        if value is default:
            return default
    return value


def _seed(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _rng_state() -> dict[str, object]:
    numpy_state = np.random.get_state()
    state: dict[str, object] = {
        "python": random.getstate(),
        "numpy": {"name": numpy_state[0], "keys": torch.from_numpy(numpy_state[1]), "position": numpy_state[2], "has_gauss": numpy_state[3], "cached_gaussian": numpy_state[4]},
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng(state: Mapping[str, object]) -> None:
    random.setstate(state["python"])
    numpy_state = state["numpy"]
    if not isinstance(numpy_state, Mapping):
        raise ValueError("resume numpy RNG state is invalid")
    np.random.set_state((str(numpy_state["name"]), np.asarray(numpy_state["keys"], dtype=np.uint32), int(numpy_state["position"]), int(numpy_state["has_gauss"]), float(numpy_state["cached_gaussian"])))
    torch.set_rng_state(state["torch"])
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def _decoder_config(config: Any) -> DirectDecoderConfig:
    decoder = _field(config, "decoder")
    return DirectDecoderConfig(
        action_horizon=int(decoder.action_horizon), action_dim=int(decoder.action_dim), tactile_dim=int(decoder.tactile_dim),
        d_model=int(decoder.d_model), nhead=int(decoder.nhead), num_layers=int(decoder.num_layers),
        dim_feedforward=int(decoder.dim_feedforward), dropout=float(decoder.dropout), tactile_keys=tuple(decoder.tactile_keys),
    )


def _load_norm_stats(config: Any) -> Mapping[str, Any]:
    injected = getattr(config, "norm_stats", None)
    if injected is not None:
        return injected
    root = Path(_field(config, "source.norm_stats_dir")) / str(_field(config, "source.norm_stats_asset_id")) / "norm_stats.json"
    raw = json.loads(root.read_text(encoding="utf-8"))
    return raw.get("norm_stats", raw)


def _small_file_fingerprint(path: Path) -> dict[str, object] | None:
    if not path.is_file():
        return None
    stat = path.stat()
    result: dict[str, object] = {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns}
    if stat.st_size <= 16 * 1024 * 1024:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        result["sha256"] = digest.hexdigest()
    return result


def _source_contract(config: Any, action: ActionCache, tactile: Any) -> dict[str, Any]:
    source = _field(config, "source")
    action_files = {name: _small_file_fingerprint(action.cache_dir / name) for name in ("manifest.json", "coarse_actions.npy", "expert_actions.npy", "valid_masks.npy")}
    tactile_files = {name: _small_file_fingerprint(Path(tactile.cache_dir) / name) for name in ("manifest.json", "embeddings.npy")}
    norm_path = Path(source.norm_stats_dir) / str(source.norm_stats_asset_id) / "norm_stats.json"
    checkpoint = Path(source.checkpoint)
    return {
        "action_cache": {"path": str(action.cache_dir.resolve()), "records_sha256": action.manifest.get("records_sha256"), "action_space": action.manifest.get("action_space"), "files": action_files},
        "tactile_cache": {"path": str(Path(tactile.cache_dir).resolve()), "encoder_identity": tactile.metadata.get("encoder_identity"), "tactile_keys": tactile.metadata.get("tactile_keys"), "files": tactile_files},
        "pi": {"checkpoint": str(checkpoint.resolve()), "norm_stats_dir": str(Path(source.norm_stats_dir).resolve()), "norm_stats_asset_id": source.norm_stats_asset_id, "norm_stats": _small_file_fingerprint(norm_path), "metadata": {name: _small_file_fingerprint(checkpoint / name) for name in ("_CHECKPOINT_METADATA", "metadata", "params/metadata")}, "variant": {"paligemma": source.paligemma_variant, "action_expert": source.action_expert_variant}, "model_action_width": getattr(source, "model_action_dim", action.manifest.get("source_model_action_width")), "sample_steps": source.sample_steps},
        "encoder": {"checkpoint": str(Path(_field(config, "tactile.encoder_checkpoint")).resolve()), "key_order": list(_field(config, "decoder.tactile_keys"))},
    }


def _open_caches(config: Any) -> tuple[ActionCache, Any]:
    action = getattr(config, "action_cache", None) or ActionCache.open(_field(config, "cache.action_root"))
    tactile = getattr(config, "tactile_cache", None) or TactileEmbeddingCache.open(
        _field(config, "cache.tactile_root"), encoder_path=_field(config, "tactile.encoder_checkpoint")

    )
    return action, tactile

def train_decoder(config: Any, *, max_steps: int | None = None) -> Path:
    """Train only ``DirectTactileActionDecoder`` with AdamW and exact resumable state."""
    decoder_settings = _field(config, "decoder")
    output = Path(decoder_settings.output); device = torch.device(getattr(decoder_settings, "device", "cpu"))
    action, tactile = _open_caches(config)
    train_set, valid_set = BaselineCacheDataset(action, tactile, "train"), BaselineCacheDataset(action, tactile, "validation")
    if len(train_set) == 0 or len(valid_set) == 0:
        raise ValueError("training and validation splits must both contain samples")
    seed = int(decoder_settings.seed); _seed(seed)
    spec = _decoder_config(config); model = DirectTactileActionDecoder(spec).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(decoder_settings.learning_rate), weight_decay=float(decoder_settings.weight_decay))
    contract = _source_contract(config, action, tactile); norm_stats = _load_norm_stats(config)
    epoch = batch_offset = global_step = 0; best_metric = float("inf"); best_epoch = -1
    resume = getattr(decoder_settings, "resume", False)
    last_path = output / "last.pt"
    if resume:
        if not last_path.exists():
            raise FileNotFoundError(f"resume checkpoint is missing: {last_path}")
        restored, payload = load_decoder_checkpoint(last_path, map_location=device)
        if payload["source_contract"] != contract:
            raise ValueError("resume source contract does not match frozen inputs")
        model.load_state_dict(restored.state_dict()); optimizer.load_state_dict(payload["optimizer_state"])
        state = payload.get("best_state", {}); epoch = int(payload["epoch"]); global_step = int(payload["global_step"])
        batch_offset = int(state.get("batch_offset", 0)); best_metric = float(state.get("best_metric", float("inf"))); best_epoch = int(state.get("best_epoch", -1))
        _restore_rng(payload["rng_state"])
        if max_steps is not None and global_step >= max_steps:
            return save_last_checkpoint(output, model, spec, epoch=epoch, global_step=global_step, metrics=payload["metrics"], source_contract=contract, optimizer=optimizer, rng_state=_rng_state(), best_state={"best_metric": best_metric, "best_epoch": best_epoch, "batch_offset": batch_offset})
    workers = int(getattr(decoder_settings, "workers", 0)); pin_memory = bool(getattr(decoder_settings, "pin_memory", False))
    epochs = int(decoder_settings.epochs); stopped = False
    for current_epoch in range(epoch, epochs):
        loader = make_loader(train_set, batch_size=int(decoder_settings.batch_size), shuffle=True, seed=seed + current_epoch, workers=workers, pin_memory=pin_memory)
        model.train()
        for batch_index, batch in enumerate(loader):
            if current_epoch == epoch and batch_index < batch_offset:
                continue
            coarse, target, tactile_tokens, valid = (batch["coarse"].to(device), batch["target"].to(device), batch["tactile"].to(device), batch["valid"].to(device))
            optimizer.zero_grad(set_to_none=True)
            loss = masked_smooth_l1(model(coarse, tactile_tokens), target, valid)
            loss.backward(); optimizer.step(); global_step += 1
            next_epoch, next_offset = current_epoch, batch_index + 1
            if next_offset == len(loader):
                next_epoch, next_offset = current_epoch + 1, 0
            if max_steps is not None and global_step >= max_steps:
                epoch, batch_offset, stopped = next_epoch, next_offset, True
                break
        if stopped:
            break
        epoch, batch_offset = current_epoch + 1, 0
        validation_loader = make_loader(valid_set, batch_size=int(decoder_settings.batch_size), shuffle=False, seed=seed, workers=workers, pin_memory=pin_memory)
        metrics = evaluate_decoder(model, validation_loader, norm_stats)
        validation = metrics["decoder_smooth_l1"]
        if validation < best_metric:
            best_metric, best_epoch = validation, current_epoch
            save_best_checkpoint(output, model, spec, epoch=epoch, global_step=global_step, metrics=metrics, source_contract=contract)
        save_last_checkpoint(output, model, spec, epoch=epoch, global_step=global_step, metrics=metrics, source_contract=contract, optimizer=optimizer, rng_state=_rng_state(), best_state={"best_metric": best_metric, "best_epoch": best_epoch, "batch_offset": 0})
    validation_loader = make_loader(valid_set, batch_size=int(decoder_settings.batch_size), shuffle=False, seed=seed, workers=workers, pin_memory=pin_memory)
    metrics = evaluate_decoder(model, validation_loader, norm_stats)
    validation = metrics["decoder_smooth_l1"]
    if validation < best_metric:
        best_metric, best_epoch = validation, max(epoch - 1, 0)
        save_best_checkpoint(output, model, spec, epoch=epoch, global_step=global_step, metrics=metrics, source_contract=contract)
    return save_last_checkpoint(output, model, spec, epoch=epoch, global_step=global_step, metrics=metrics, source_contract=contract, optimizer=optimizer, rng_state=_rng_state(), best_state={"best_metric": best_metric, "best_epoch": best_epoch, "batch_offset": batch_offset})


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--config", required=True, type=Path); parser.add_argument("--max-steps", type=int)
    args = parser.parse_args()
    from train_baseline_pi05.config import load_config
    print(train_decoder(load_config(args.config), max_steps=args.max_steps))


if __name__ == "__main__":
    main()
