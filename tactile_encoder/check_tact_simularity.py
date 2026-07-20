"""Check temporal cosine similarity of four tactile streams vs each stream's own first frame.

Loads a dataset + frozen tactile encoder, encodes each of the four tactile images
per frame through the shared ResNet backbone, L2-normalizes the vectors, then
for each stream plots cosine similarity of frame t against that stream's frame-0
vector (so every curve starts at 1.0).
"""

from __future__ import annotations

import argparse
import pathlib
from collections.abc import Sequence

import jax
import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from tactile_encoder.utils.checkpoint import load_tactile_encoder
from tactile_encoder.utils.image_dataset import create_image_dataset
from tactile_encoder.utils.model import encode_resnet18
from tactile_encoder.utils.model import tactile_clip_config_from_dict

TACTILE_KEYS = (
    "tactile_left_0",
    "tactile_right_0",
    "tactile_left_1",
    "tactile_right_1",
)
COLORS = ("#4C72B0", "#55A868", "#C44E52", "#8172B3")


def _l2_normalize(vectors: np.ndarray, *, eps: float = 1e-8) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=-1, keepdims=True)
    return vectors / np.maximum(norms, eps)


def _cosine_similarity(vectors: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """vectors [T, D], reference [D] -> [T] cosine similarities."""

    normalized = _l2_normalize(vectors)
    ref = _l2_normalize(reference[None, :])[0]
    return normalized @ ref


def encode_tactile_frames(
    *,
    bundle,
    images: np.ndarray,
    batch_size: int,
) -> np.ndarray:
    """Encode tactile images [T, H, W, C] -> [T, embedding_dim] via frozen ResNet."""

    config = tactile_clip_config_from_dict(bundle.metadata["tactile_clip_config"])
    resnet_params = bundle.params["tactile_resnet"]
    embeddings: list[np.ndarray] = []

    @jax.jit
    def encode_batch(batch: jax.Array) -> jax.Array:
        encoded, _ = encode_resnet18(
            resnet_params,
            batch,
            train=False,
            embedding_dim=config.embedding_dim,
        )
        return encoded

    for start in range(0, len(images), batch_size):
        batch = jnp.asarray(images[start : start + batch_size], dtype=jnp.float32)
        embeddings.append(np.asarray(encode_batch(batch), dtype=np.float32))
    return np.concatenate(embeddings, axis=0)


def collect_episode_tactile_images(
    dataset,
    *,
    episode_index: int,
    frame_stride: int,
    max_frames: int | None,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Return (frame_indices, {key: [T, H, W, C]})."""

    if frame_stride <= 0:
        raise ValueError(f"frame_stride must be positive, got {frame_stride}.")
    frame_indices = np.asarray(dataset.indices_for_episode(episode_index), dtype=np.int64)
    frame_indices = frame_indices[::frame_stride]
    if max_frames is not None:
        if max_frames <= 0:
            raise ValueError(f"max_frames must be positive when set, got {max_frames}.")
        frame_indices = frame_indices[:max_frames]
    if len(frame_indices) == 0:
        raise ValueError(f"Episode {episode_index} has no frames after stride/max_frames.")

    stacks: dict[str, list[np.ndarray]] = {key: [] for key in TACTILE_KEYS}
    for frame_index in frame_indices:
        images = dataset.get_images(int(frame_index), TACTILE_KEYS, as_float=True)
        for key in TACTILE_KEYS:
            stacks[key].append(np.asarray(images[key], dtype=np.float32))
    return frame_indices, {key: np.stack(parts, axis=0) for key, parts in stacks.items()}


def compute_similarities(
    embeddings_by_key: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """Cosine similarity of each stream/frame vs that stream's own first-frame vec."""

    return {
        key: _cosine_similarity(vectors, vectors[0])
        for key, vectors in embeddings_by_key.items()
    }


def plot_similarities(
    similarities: dict[str, np.ndarray],
    *,
    output_path: pathlib.Path,
    title: str,
) -> pathlib.Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axis = plt.subplots(figsize=(11, 5), constrained_layout=True)
    timesteps = np.arange(next(iter(similarities.values())).shape[0])
    for key, color in zip(TACTILE_KEYS, COLORS):
        axis.plot(
            timesteps,
            similarities[key],
            color=color,
            linewidth=1.8,
            label=key,
        )
    axis.axhline(1.0, color="#888888", linestyle=":", linewidth=1.0)
    axis.set_xlabel("frame index (after stride)")
    axis.set_ylabel("cosine similarity vs own t0")
    axis.set_ylim(-1.05, 1.05)
    axis.set_title(title)
    axis.grid(True, alpha=0.3)
    axis.legend(loc="best")
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def run_check(
    *,
    dataset_repo_id: str,
    tactile_encoder_dir: pathlib.Path,
    output_path: pathlib.Path,
    episode_index: int,
    frame_stride: int,
    max_frames: int | None,
    batch_size: int,
    image_cache_size: int,
) -> pathlib.Path:
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}.")

    bundle = load_tactile_encoder(tactile_encoder_dir)
    config = tactile_clip_config_from_dict(bundle.metadata["tactile_clip_config"])
    info = create_image_dataset(
        dataset_repo_id,
        image_size=int(config.tactile_image_size),
        cache_size=image_cache_size,
    )
    dataset = info.dataset
    if episode_index < 0 or episode_index >= dataset.num_episodes:
        raise ValueError(
            f"episode_index={episode_index} out of range "
            f"[0, {dataset.num_episodes - 1}]."
        )

    frame_indices, images_by_key = collect_episode_tactile_images(
        dataset,
        episode_index=episode_index,
        frame_stride=frame_stride,
        max_frames=max_frames,
    )
    print(
        f"episode={episode_index} frames={len(frame_indices)} "
        f"stride={frame_stride} image_size={config.tactile_image_size}"
    )

    embeddings_by_key: dict[str, np.ndarray] = {}
    for key in TACTILE_KEYS:
        embeddings_by_key[key] = encode_tactile_frames(
            bundle=bundle,
            images=images_by_key[key],
            batch_size=batch_size,
        )
        print(f"encoded {key}: {embeddings_by_key[key].shape}")

    similarities = compute_similarities(embeddings_by_key)
    title = (
        f"{dataset_repo_id} ep={episode_index} | "
        f"cos(v_t, v_0) per tactile stream"
    )
    path = plot_similarities(similarities, output_path=output_path, title=title)
    for key in TACTILE_KEYS:
        values = similarities[key]
        print(
            f"{key}: mean={float(np.mean(values)):.4f} "
            f"min={float(np.min(values)):.4f} max={float(np.max(values)):.4f}"
        )
    return path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Encode four tactile streams and plot cosine similarity of each frame "
            "against that stream's own first-frame embedding (curves start at 1.0)."
        )
    )
    parser.add_argument("--dataset-repo-id", type=str, required=True)
    parser.add_argument("--tactile-encoder-dir", type=pathlib.Path, required=True)
    parser.add_argument(
        "--output",
        type=pathlib.Path,
        default=pathlib.Path("tactile_encoder/outputs/tact_similarity.png"),
        help="Output PNG path.",
    )
    parser.add_argument("--episode-index", type=int, default=0)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Optional cap on frames after stride (default: all).",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--image-cache-size", type=int, default=512)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    path = run_check(
        dataset_repo_id=args.dataset_repo_id,
        tactile_encoder_dir=args.tactile_encoder_dir,
        output_path=args.output,
        episode_index=args.episode_index,
        frame_stride=args.frame_stride,
        max_frames=args.max_frames,
        batch_size=args.batch_size,
        image_cache_size=args.image_cache_size,
    )
    print(f"plot={path}")


if __name__ == "__main__":
    main()
