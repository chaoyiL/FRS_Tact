from __future__ import annotations

import argparse
import csv
import pathlib
from collections.abc import Sequence
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from utils.cache import atomic_write_json
from tactile_encoder.utils.checkpoint import load_checkpoint
from tactile_encoder.utils.clip_backend import CLIP_IMAGE_SIZE
from tactile_encoder.utils.clip_backend import ClipBackend
from tactile_encoder.utils.data import DataKeys
from tactile_encoder.utils.data import batches
from tactile_encoder.utils.data import build_future_records
from tactile_encoder.utils.data import resolve_data_keys
from tactile_encoder.utils.image_dataset import create_image_dataset
from tactile_encoder.utils.metrics import retrieval_metrics
from tactile_encoder.utils.model import TactileClipConfig
from tactile_encoder.utils.model import forward_embeddings
from tactile_encoder.utils.model import tactile_clip_config_from_dict


def _tree_to_jax(batch: dict[str, np.ndarray]) -> dict[str, jax.Array]:
    return {key: jnp.asarray(value) for key, value in batch.items()}


def make_embed_step(
    clip_model: Any,
    *,
    rgb_mask_patch_size: int,
    rgb_mask_ratio: float,
    config: TactileClipConfig,
):
    def step(params, frozen_clip_params, batch, key):
        embeddings, _ = forward_embeddings(
            clip_model,
            params,
            frozen_clip_params,
            batch,
            train=False,
            mask_key=key,
            rgb_mask_patch_size=rgb_mask_patch_size,
            rgb_mask_ratio=rgb_mask_ratio,
            config=config,
        )
        return embeddings["query"], embeddings["future"]

    return jax.jit(step)


def collect_retrieval_embeddings(
    *,
    clip_model: Any,
    params: dict[str, Any],
    frozen_clip_params: Any,
    dataset: Any,
    records,
    keys: DataKeys,
    batch_size: int,
    eval_mask_seed: int,
    rgb_mask_patch_size: int,
    rgb_mask_ratio: float,
    config: TactileClipConfig,
    tactile_history: int | None = None,
    history_stride: int = 5,
) -> dict[str, np.ndarray]:
    if tactile_history is None:
        tactile_history = int(config.tactile_history)
    embed_step = make_embed_step(
        clip_model,
        rgb_mask_patch_size=rgb_mask_patch_size,
        rgb_mask_ratio=rgb_mask_ratio,
        config=config,
    )
    query_parts: list[np.ndarray] = []
    future_parts: list[np.ndarray] = []
    dataset_indices: list[np.ndarray] = []
    future_indices: list[np.ndarray] = []
    episode_indices: list[np.ndarray] = []
    eval_key = jax.random.key(eval_mask_seed)
    for batch_number, batch_np in enumerate(
        batches(
            dataset,
            records,
            keys,
            batch_size=batch_size,
            shuffle=False,
            seed=0,
            image_size=CLIP_IMAGE_SIZE,
            tactile_history=tactile_history,
            history_stride=history_stride,
        )
    ):
        query, future = embed_step(
            params,
            frozen_clip_params,
            _tree_to_jax(batch_np),
            jax.random.fold_in(eval_key, batch_number),
        )
        query_parts.append(np.asarray(jax.device_get(query), dtype=np.float32))
        future_parts.append(np.asarray(jax.device_get(future), dtype=np.float32))
        dataset_indices.append(np.asarray(batch_np["dataset_index"], dtype=np.int64))
        future_indices.append(np.asarray(batch_np["future_dataset_index"], dtype=np.int64))
        episode_indices.append(np.asarray(batch_np["episode_index"], dtype=np.int64))
    return {
        "query": np.concatenate(query_parts, axis=0),
        "future": np.concatenate(future_parts, axis=0),
        "dataset_index": np.concatenate(dataset_indices, axis=0),
        "future_dataset_index": np.concatenate(future_indices, axis=0),
        "episode_index": np.concatenate(episode_indices, axis=0),
    }


def evaluate_records(
    *,
    clip_model: Any,
    params: dict[str, Any],
    frozen_clip_params: Any,
    dataset: Any,
    records,
    keys: DataKeys,
    batch_size: int,
    eval_mask_seed: int,
    rgb_mask_patch_size: int,
    rgb_mask_ratio: float,
    config: TactileClipConfig,
    tactile_history: int | None = None,
    history_stride: int = 5,
) -> tuple[dict[str, float | int], dict[str, np.ndarray]]:
    embeddings = collect_retrieval_embeddings(
        clip_model=clip_model,
        params=params,
        frozen_clip_params=frozen_clip_params,
        dataset=dataset,
        records=records,
        keys=keys,
        batch_size=batch_size,
        eval_mask_seed=eval_mask_seed,
        rgb_mask_patch_size=rgb_mask_patch_size,
        rgb_mask_ratio=rgb_mask_ratio,
        config=config,
        tactile_history=tactile_history,
        history_stride=history_stride,
    )
    metrics, ranks = retrieval_metrics(jnp.asarray(embeddings["query"]), jnp.asarray(embeddings["future"]))
    metric_dict: dict[str, float | int] = {
        "recall@1": metrics.recall_at_1,
        "recall@5": metrics.recall_at_5,
        "mean_rank": metrics.mean_rank,
        "sample_count": metrics.sample_count,
    }
    embeddings["rank"] = np.asarray(jax.device_get(ranks), dtype=np.int64)
    return metric_dict, embeddings


def write_retrieval_outputs(
    output_dir: pathlib.Path,
    metrics: dict[str, float | int],
    embeddings: dict[str, np.ndarray],
    *,
    top_k: int = 5,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(output_dir / "metrics.json", metrics)
    query = embeddings["query"]
    future = embeddings["future"]
    similarity = query @ future.T
    top_k = min(top_k, similarity.shape[1])
    top_indices = np.argsort(-similarity, axis=1)[:, :top_k]
    with (output_dir / "per_query.csv").open("w", newline="", encoding="utf-8") as file:
        fieldnames = [
            "query_position",
            "dataset_index",
            "future_dataset_index",
            "episode_index",
            "rank",
            "top_gallery_positions",
            "top_future_dataset_indices",
        ]
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in range(similarity.shape[0]):
            top = top_indices[row]
            writer.writerow(
                {
                    "query_position": row,
                    "dataset_index": int(embeddings["dataset_index"][row]),
                    "future_dataset_index": int(embeddings["future_dataset_index"][row]),
                    "episode_index": int(embeddings["episode_index"][row]),
                    "rank": int(embeddings["rank"][row]) + 1,
                    "top_gallery_positions": " ".join(str(int(index)) for index in top),
                    "top_future_dataset_indices": " ".join(
                        str(int(embeddings["future_dataset_index"][index])) for index in top
                    ),
                }
            )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate tactile ResNet18 + frozen CLIP future image retrieval."
    )
    parser.add_argument("--checkpoint-dir", type=pathlib.Path, required=True)
    parser.add_argument(
        "--dataset-repo-id",
        action="append",
        required=True,
        dest="dataset_repo_ids",
        help="LeRobot dataset repo id (repeatable).",
    )
    parser.add_argument(
        "--config-name",
        default=None,
        help="Optional run label (defaults to repo id).",
    )
    parser.add_argument("--output-dir", type=pathlib.Path, required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--split-seed", type=int, default=0)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--future-offset", type=int, default=1)
    parser.add_argument("--eval-mask-seed", type=int, default=0)
    parser.add_argument("--rgb-mask-patch-size", type=int, default=16)
    parser.add_argument("--rgb-mask-ratio", type=float, default=0.5)
    parser.add_argument(
        "--masked-rgb-key",
        default="",
        help="Optional RGB key used instead of each side's current camera for masking.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    params, metadata = load_checkpoint(args.checkpoint_dir)
    backend = ClipBackend.from_pretrained(metadata["clip_model_id"])
    dataset_info = create_image_dataset(
        args.dataset_repo_ids,
        image_size=CLIP_IMAGE_SIZE,
        config_name=args.config_name,
    )
    dataset = dataset_info.dataset
    record_set = build_future_records(
        dataset,
        future_offset=args.future_offset,
        val_fraction=args.val_fraction,
        split_seed=args.split_seed,
        frame_stride=args.frame_stride,
    )
    extra = metadata.get("extra_metadata") or {}
    keys = resolve_data_keys(
        masked_rgb_key=args.masked_rgb_key or extra.get("masked_rgb_key") or None,
    )
    config = tactile_clip_config_from_dict(metadata["tactile_clip_config"])
    history_stride = int(extra.get("history_stride") or args.frame_stride)
    metrics, embeddings = evaluate_records(
        clip_model=backend.model,
        params=params,
        frozen_clip_params=backend.params,
        dataset=dataset,
        records=record_set.split_records("val"),
        keys=keys,
        batch_size=args.batch_size,
        eval_mask_seed=args.eval_mask_seed,
        rgb_mask_patch_size=args.rgb_mask_patch_size,
        rgb_mask_ratio=args.rgb_mask_ratio,
        config=config,
        tactile_history=int(config.tactile_history),
        history_stride=history_stride,
    )
    write_retrieval_outputs(args.output_dir, metrics, embeddings)
    print(
        f"validation_samples={metrics['sample_count']} recall@1={metrics['recall@1']:.6f} "
        f"recall@5={metrics['recall@5']:.6f} mean_rank={metrics['mean_rank']:.2f}"
    )
    print(f"evaluation={args.output_dir}")


if __name__ == "__main__":
    main()
