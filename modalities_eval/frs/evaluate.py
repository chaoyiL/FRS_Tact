"""Run fixed-``x_base`` FRS modality interventions from a training YAML."""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np

from modalities_eval.frs.interventions import (
    DEFAULT_INTERVENTIONS,
    Intervention,
    apply_intervention,
    gate_weights_from_change,
    tactile_change_from_tokens,
)
from modalities_eval.frs.statistics import sample_error_rows


DEFAULT_OUTPUT_DIR = Path("eval_outputs/frs_modalities")
_UNVERIFIED_PROVENANCE_WARNING = (
    "Checkpoint/cache identity and configuration were checked, but strong content hashes "
    "for action arrays, tactile embedding arrays, and tactile encoder contents are unavailable."
)


def _intervention_name(intervention: str | Intervention) -> str:
    return intervention.name if isinstance(intervention, Intervention) else str(intervention)


def _decode_checked(
    decode_fn: Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray],
    x_base: np.ndarray,
    tactile: np.ndarray,
    state: np.ndarray,
) -> np.ndarray:
    prediction = np.asarray(decode_fn(x_base.copy(), tactile, state), dtype=np.float32)
    if prediction.shape != x_base.shape:
        raise ValueError(f"decode output shape {prediction.shape} does not match action shape {x_base.shape}")
    return prediction


def evaluate_batches(
    *,
    batches: Iterable[
        tuple[
            np.ndarray,
            np.ndarray,
            np.ndarray,
            np.ndarray,
            np.ndarray,
            np.ndarray,
            Sequence[Mapping[str, object]],
        ]
    ],
    baseline_fn: Callable[[np.ndarray], np.ndarray],
    decode_fn: Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray],
    tau: float,
    temperature: float,
    interventions: Iterable[str | Intervention],
) -> list[dict[str, object]]:
    """Evaluate full and counterfactual decodes while keeping every ``x_base`` fixed."""

    rows: list[dict[str, object]] = []
    intervention_names = tuple(_intervention_name(item) for item in interventions)
    if len(set(intervention_names)) != len(intervention_names):
        raise ValueError("interventions must have unique names")
    if "full" in intervention_names:
        raise ValueError("'full' is reserved for the unmodified tactile condition")

    for indices, x_base, vla, gt, state, tactile, metadata in batches:
        indices = np.asarray(indices, dtype=np.int64)
        x_base = np.asarray(x_base, dtype=np.float32)
        vla = np.asarray(vla, dtype=np.float32)
        gt = np.asarray(gt, dtype=np.float32)
        state = np.asarray(state, dtype=np.float32)
        tactile = np.asarray(tactile, dtype=np.float32)
        baseline = np.asarray(baseline_fn(indices), dtype=np.float32)
        if len(metadata) != len(indices):
            raise ValueError("metadata must contain one mapping for every batch sample")
        if x_base.shape != vla.shape or x_base.shape != gt.shape:
            raise ValueError("x_base, vla, and gt must have identical action shapes")
        if x_base.shape[0] != len(indices):
            raise ValueError("action batch size must match indices")

        original_change = tactile_change_from_tokens(tactile[:, -1], baseline)
        original_gate = gate_weights_from_change(original_change, tau=tau, temperature=temperature)
        fixed_x_base = x_base.copy()
        full = _decode_checked(decode_fn, fixed_x_base, tactile, state)
        predictions: dict[str, np.ndarray] = {"full": full}
        gates: dict[str, np.ndarray] = {"full": original_gate}
        for name in intervention_names:
            changed = apply_intervention(
                name,
                tactile,
                baseline,
                original_gate,
                tau=tau,
                temperature=temperature,
            )
            predictions[name] = _decode_checked(decode_fn, fixed_x_base, changed.tactile, state)
            gates[name] = changed.gate
        rows.extend(
            sample_error_rows(
                full=full,
                conditions=predictions,
                vla=vla,
                gt=gt,
                metadata=metadata,
                original_gate=original_gate,
                counterfactual_gates=gates,
            )
        )
    return rows


@dataclass
class EvaluationContext:
    """Loaded train-FRS resources needed by the NumPy-facing evaluation runner."""

    pairs: Any
    conditioner: Any
    model: Any
    gate_tau: float
    gate_temperature: float
    rank_low_gate_threshold: float
    rank_high_gate_threshold: float
    default_num_steps: int
    checkpoint_metadata: Mapping[str, Any]
    provenance: Mapping[str, Any]

    def batches(self, *, split: str, batch_size: int):
        for indices, x_base, vla, gt, state, tactile in self.conditioner.batches(
            split, batch_size=batch_size, shuffle=False, seed=0
        ):
            source_indices, local_indices = self.pairs.source_and_local_indices(indices)
            dataset_indices = self.pairs.metadata_values(indices, "dataset_index")
            episode_indices = self.pairs.metadata_values(indices, "episode_index")
            metadata = [
                {
                    "cache_index": int(cache_index),
                    "source": self.pairs.source_names[int(source_index)],
                    "source_index": int(source_index),
                    "source_cache_index": int(local_index),
                    "dataset_index": int(dataset_index),
                    "episode_index": int(episode_index),
                }
                for cache_index, source_index, local_index, dataset_index, episode_index in zip(
                    indices,
                    source_indices,
                    local_indices,
                    dataset_indices,
                    episode_indices,
                    strict=True,
                )
            ]
            yield (
                indices,
                x_base,
                vla,
                gt,
                np.asarray(state, dtype=np.float32),
                np.asarray(tactile, dtype=np.float32),
                metadata,
            )

    def baselines(self, indices: np.ndarray) -> np.ndarray:
        source_indices, _ = self.pairs.source_and_local_indices(indices)
        episode_indices = self.pairs.metadata_values(indices, "episode_index")
        try:
            return np.stack(
                [
                    self.conditioner.episode_baselines[(int(source_index), int(episode_index))]
                    for source_index, episode_index in zip(source_indices, episode_indices, strict=True)
                ],
                axis=0,
            ).astype(np.float32, copy=False)
        except KeyError as exc:
            raise ValueError("missing cached first-frame tactile baseline for evaluation sample") from exc

    def decode(
        self,
        x_base: np.ndarray,
        tactile: np.ndarray,
        state: np.ndarray,
        *,
        num_steps: int,
        solver: str,
    ) -> np.ndarray:
        from train_smolvla_frs.utils.model import decode_actions

        return np.asarray(
            decode_actions(
                self.model,
                x_base,
                tactile,
                num_steps=num_steps,
                solver=solver,
                state=state,
            ),
            dtype=np.float32,
        )

    def close(self) -> None:
        self.conditioner.close()


def _required_mapping(config: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = config.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"config.{key} must be a mapping")
    return value


def _positive_int(config: Mapping[str, Any], key: str, default: int) -> int:
    value = int(config.get(key, default))
    if value <= 0:
        raise ValueError(f"{key} must be positive, got {value}")
    return value


def _required_checkpoint_metadata(extra: Mapping[str, Any], key: str) -> Any:
    if key not in extra:
        raise ValueError(f"checkpoint metadata is missing {key}")
    return extra[key]


def _load_checkpoint_metadata_only(checkpoint_dir: Path) -> Mapping[str, Any]:
    """Read checkpoint JSON without importing or instantiating the decoder."""

    metadata_path = checkpoint_dir / "checkpoint.json"
    with metadata_path.open(encoding="utf-8") as file:
        metadata = json.load(file)
    if not isinstance(metadata, Mapping):
        raise ValueError(f"checkpoint metadata must be a JSON object: {metadata_path}")
    return metadata


def _has_verified_strong_content_hashes(checkpoint_metadata: Mapping[str, Any]) -> bool:
    """Return whether the understood checkpoint schema proves content identity.

    Version 2 checkpoints contain record/configuration identity only. A future
    schema can add verified action-array, tactile-array, and encoder hashes here.
    """

    del checkpoint_metadata
    return False


def load_evaluation_context(
    *,
    config_path: Path,
    checkpoint_dir: Path | None = None,
    allow_unverified_provenance: bool = False,
) -> EvaluationContext:
    """Load and validate a gated FRS checkpoint plus its multi-source caches."""

    from train_smolvla_frs.train_frs import load_config, source_cache_dir

    config = load_config(Path(config_path))
    datasets = config.get("datasets")
    if not isinstance(datasets, list) or not datasets or not all(isinstance(item, Mapping) for item in datasets):
        raise ValueError("config.datasets must be a non-empty list of mappings")
    action_cache = _required_mapping(config, "action_cache")
    tactile_cache = _required_mapping(config, "tactile_embedding_cache")
    model_config = _required_mapping(config, "model")
    training = _required_mapping(config, "frs_training")
    if not action_cache.get("root") or not tactile_cache.get("root"):
        raise ValueError("config action_cache.root and tactile_embedding_cache.root are required")
    tactile_keys = tuple(str(key) for key in model_config.get("tactile_keys", ()))
    tactile_num_tokens = _positive_int(model_config, "tactile_num_tokens", len(tactile_keys))
    if tactile_num_tokens != len(tactile_keys) or tactile_num_tokens != 4:
        raise ValueError("model.tactile_num_tokens and model.tactile_keys must describe four tactile streams")
    if not model_config.get("tactile_encoder_path"):
        raise ValueError("config.model.tactile_encoder_path is required")
    if not training.get("output") and checkpoint_dir is None:
        raise ValueError("config.frs_training.output is required when --checkpoint-dir is omitted")

    resolved_checkpoint = (
        Path(checkpoint_dir).expanduser()
        if checkpoint_dir is not None
        else Path(str(training["output"])).expanduser() / "best"
    )
    preflight_metadata = _load_checkpoint_metadata_only(resolved_checkpoint)
    preflight_extra = preflight_metadata.get("extra_metadata")
    if not isinstance(preflight_extra, Mapping):
        raise ValueError("checkpoint is missing extra_metadata required for modality evaluation")
    strong_content_hashes_verified = _has_verified_strong_content_hashes(preflight_metadata)
    if not strong_content_hashes_verified and not allow_unverified_provenance:
        raise ValueError(
            f"{_UNVERIFIED_PROVENANCE_WARNING} Existing checkpoints can be evaluated only "
            "with the explicit --allow-unverified-provenance flag."
        )

    from train_smolvla_frs.utils.checkpoint import load_checkpoint
    from train_smolvla_frs.utils.data import CachedTactileEmbeddingBatches, resolve_tactile_window
    from utils.cache import MultiCachedPairs

    source_names = []
    cache_dirs = []
    for source in datasets:
        repo_id = source.get("repo_id")
        if not repo_id:
            raise ValueError("every config.datasets entry requires repo_id")
        source_names.append(str(repo_id))
        cache_dirs.append(source_cache_dir(action_cache["root"], str(repo_id)))
    pairs = MultiCachedPairs(cache_dirs, source_names=source_names)
    model, checkpoint_metadata = load_checkpoint(resolved_checkpoint)
    extra = checkpoint_metadata.get("extra_metadata")
    if not isinstance(extra, Mapping):
        raise ValueError("checkpoint is missing extra_metadata required for modality evaluation")
    checkpoint_digest = extra.get("cache_records_sha256")
    if not checkpoint_digest or checkpoint_digest != pairs.manifest["records_sha256"]:
        raise ValueError("Checkpoint was trained from a different cache sample set.")
    checkpoint_cache_configuration = _required_checkpoint_metadata(extra, "cache_configuration")
    if checkpoint_cache_configuration != pairs.manifest["configuration"]:
        raise ValueError("checkpoint cache_configuration does not match cache configuration")
    expected_action_shape = (int(pairs.manifest["action_horizon"]), int(pairs.manifest["action_dim"]))
    actual_action_shape = (int(model.config.action_horizon), int(model.config.action_dim))
    if actual_action_shape != expected_action_shape:
        raise ValueError(f"Checkpoint/cache action shape mismatch: {actual_action_shape} != {expected_action_shape}.")
    decoder_input_version = extra.get("decoder_input_version")
    if (
        str(extra.get("loss_mode")) != "gated"
        or not isinstance(decoder_input_version, int)
        or isinstance(decoder_input_version, bool)
        or decoder_input_version != 2
    ):
        raise ValueError(
            "modality evaluation requires loss_mode=gated and decoder_input_version=2"
        )
    try:
        gate_tau = float(extra["gate_tau"])
        gate_temperature = float(extra["gate_temperature"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("checkpoint is missing numeric gate_tau/gate_temperature metadata") from exc
    if not math.isfinite(gate_tau) or not math.isfinite(gate_temperature) or gate_temperature <= 0:
        raise ValueError("checkpoint gate metadata must contain finite tau and positive temperature")
    try:
        rank_low_gate_threshold = float(extra["rank_low_gate_threshold"])
        rank_high_gate_threshold = float(extra["rank_high_gate_threshold"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("checkpoint is missing numeric rank gate thresholds") from exc
    if not 0.0 <= rank_low_gate_threshold < rank_high_gate_threshold <= 1.0:
        raise ValueError("checkpoint rank gate thresholds must satisfy 0 <= low < high <= 1")
    tactile_encoder_dir = Path(str(model_config["tactile_encoder_path"])).expanduser().resolve()
    checkpoint_encoder_dir = Path(
        str(_required_checkpoint_metadata(extra, "tactile_encoder_dir"))
    ).expanduser().resolve()
    if checkpoint_encoder_dir != tactile_encoder_dir:
        raise ValueError("checkpoint tactile_encoder_dir does not match config tactile_encoder_path")
    history_stride = _positive_int(training, "history_stride", 3)
    try:
        checkpoint_history_stride = int(_required_checkpoint_metadata(extra, "history_stride"))
    except (TypeError, ValueError) as exc:
        raise ValueError("checkpoint history_stride must be an integer") from exc
    if checkpoint_history_stride != history_stride:
        raise ValueError("checkpoint history_stride does not match config")
    tactile_window_divisor = _positive_int(training, "tactile_window_divisor", 1)
    try:
        checkpoint_window_divisor = int(_required_checkpoint_metadata(extra, "tactile_window_divisor"))
    except (TypeError, ValueError) as exc:
        raise ValueError("checkpoint tactile_window_divisor must be an integer") from exc
    if checkpoint_window_divisor != tactile_window_divisor:
        raise ValueError("checkpoint tactile_window_divisor does not match config")
    if int(model.config.num_tactile_tokens) != tactile_num_tokens:
        raise ValueError(
            f"checkpoint tactile token count={model.config.num_tactile_tokens} does not match config={tactile_num_tokens}"
        )
    tactile_window = resolve_tactile_window(
        action_horizon=expected_action_shape[0],
        window_divisor=tactile_window_divisor,
    )
    if tactile_window != int(model.config.tactile_window):
        raise ValueError(
            f"Resolved tactile_window={tactile_window} does not match checkpoint tactile_window={model.config.tactile_window}."
        )

    provenance = {
        "status": "verified" if strong_content_hashes_verified else "configuration_only",
        "strong_content_hashes_verified": strong_content_hashes_verified,
        "override_used": bool(allow_unverified_provenance),
        "warning": None if strong_content_hashes_verified else _UNVERIFIED_PROVENANCE_WARNING,
    }

    conditioner = CachedTactileEmbeddingBatches(
        pairs,
        sources=datasets,
        tactile_cache_root=Path(str(tactile_cache["root"])).expanduser(),
        tactile_encoder_dir=tactile_encoder_dir,
        tactile_keys=tactile_keys,
        tactile_window=tactile_window,
        history_stride=history_stride,
        embedding_dim=_positive_int(model_config, "tactile_embedding_dim", 512),
        image_size=_positive_int(model_config, "tactile_image_size", 224),
        build_episode_baselines=True,
        return_raw_images=bool(
            getattr(model.config, "tactile_encoder_trainable", False)
        ),
        num_workers=int(training.get("num_workers", 8)),
        prefetch_batches=int(training.get("prefetch_batches", 8)),
        pipeline_prefetch=int(training.get("pipeline_prefetch", 4)),
        load_threads=int(training.get("load_threads", 8)),
        image_cache_size=int(training.get("image_cache_size", 8192)),
    )
    try:
        if conditioner.resnet_embedding_dim != int(model.config.resnet_embedding_dim):
            raise ValueError(
                f"Embedding cache dimension={conditioner.resnet_embedding_dim} does not match "
                f"checkpoint={model.config.resnet_embedding_dim}."
            )
        return EvaluationContext(
            pairs=pairs,
            conditioner=conditioner,
            model=model,
            gate_tau=gate_tau,
            gate_temperature=gate_temperature,
            rank_low_gate_threshold=rank_low_gate_threshold,
            rank_high_gate_threshold=rank_high_gate_threshold,
            default_num_steps=_positive_int(training, "validation_steps", 10),
            checkpoint_metadata=checkpoint_metadata,
            provenance=provenance,
        )
    except Exception:
        conditioner.close()
        raise


def evaluate_from_config(
    *,
    config_path: Path,
    checkpoint_dir: Path | None = None,
    output_dir: Path | None = None,
    split: str = "val",
    batch_size: int = 64,
    num_steps: int | None = None,
    solver: str = "euler",
    interventions: Iterable[str | Intervention] = DEFAULT_INTERVENTIONS,
    bootstrap_samples: int = 1000,
    bootstrap_seed: int = 0,
    allow_unverified_provenance: bool = False,
):
    """Load configured data, run fixed-base interventions, and write a Task-3 report."""

    if split not in {"train", "val"}:
        raise ValueError(f"split must be 'train' or 'val', got {split!r}")
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")
    if num_steps is not None and num_steps <= 0:
        raise ValueError(f"num_steps must be positive, got {num_steps}")
    if solver not in {"euler", "fireflow"}:
        raise ValueError(f"solver must be 'euler' or 'fireflow', got {solver!r}")

    resolved_output_dir = DEFAULT_OUTPUT_DIR if output_dir is None else Path(output_dir).expanduser()
    if resolved_output_dir.exists() and not resolved_output_dir.is_dir():
        raise ValueError(f"output_dir must be a directory path, got file: {resolved_output_dir}")
    context = load_evaluation_context(
        config_path=Path(config_path),
        checkpoint_dir=checkpoint_dir,
        allow_unverified_provenance=allow_unverified_provenance,
    )
    try:
        resolved_steps = num_steps or context.default_num_steps
        if resolved_steps <= 0:
            raise ValueError(f"num_steps must be positive, got {resolved_steps}")
        rows = evaluate_batches(
            batches=context.batches(split=split, batch_size=batch_size),
            baseline_fn=context.baselines,
            decode_fn=lambda x, tactile, state: context.decode(
                x, tactile, state, num_steps=resolved_steps, solver=solver
            ),
            tau=context.gate_tau,
            temperature=context.gate_temperature,
            interventions=interventions,
        )
        # Task 3 owns report formatting and output files. Keep this import local so
        # runner import and CLI help are usable while that module is developed.
        from modalities_eval.frs.reporting import write_report

        return write_report(
            rows,
            output_dir=resolved_output_dir,
            bootstrap_samples=bootstrap_samples,
            bootstrap_seed=bootstrap_seed,
            rank_low_gate_threshold=context.rank_low_gate_threshold,
            rank_high_gate_threshold=context.rank_high_gate_threshold,
            provenance=context.provenance,
        )
    finally:
        context.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="FRS training YAML configuration.")
    parser.add_argument("--checkpoint-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--split", choices=("train", "val"), default="val")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-steps", type=int, default=None)
    parser.add_argument("--solver", choices=("euler", "fireflow"), default="euler")
    parser.add_argument(
        "--interventions",
        nargs="+",
        default=[item.name for item in DEFAULT_INTERVENTIONS],
        help="One or more intervention names (default: all supported counterfactuals).",
    )
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--bootstrap-seed", type=int, default=0)
    parser.add_argument(
        "--allow-unverified-provenance",
        action="store_true",
        help=(
            "Allow evaluation when checkpoint compatibility is configuration-only and "
            "array/encoder content hashes cannot be verified."
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    evaluate_from_config(
        config_path=args.config,
        checkpoint_dir=args.checkpoint_dir,
        output_dir=args.output_dir,
        split=args.split,
        batch_size=args.batch_size,
        num_steps=args.num_steps,
        solver=args.solver,
        interventions=args.interventions,
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_seed=args.bootstrap_seed,
        allow_unverified_provenance=args.allow_unverified_provenance,
    )


if __name__ == "__main__":
    main()
