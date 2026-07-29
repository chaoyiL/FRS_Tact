from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np

Array = jax.Array


@dataclasses.dataclass(frozen=True)
class RetrievalMetrics:
    recall_at_1: float
    recall_at_5: float
    mean_rank: float
    sample_count: int


def l2_normalize(x: Array, *, eps: float = 1e-8) -> Array:
    return x / jnp.maximum(jnp.linalg.norm(x, axis=-1, keepdims=True), eps)


def ranks_from_similarity(similarity: Array, positive_indices: Array | None = None) -> Array:
    """Return zero-based ranks for each query's positive gallery item."""

    if similarity.ndim != 2:
        raise ValueError(f"Expected similarity [N, M], got {similarity.shape}.")
    num_queries, num_gallery = similarity.shape
    if positive_indices is None:
        if num_queries != num_gallery:
            raise ValueError("positive_indices is required when query and gallery counts differ.")
        positive_indices = jnp.arange(num_queries)
    positive_indices = jnp.asarray(positive_indices)
    if positive_indices.shape != (num_queries,):
        raise ValueError(
            f"positive_indices must have shape {(num_queries,)}, got {positive_indices.shape}."
        )
    if bool(jnp.any((positive_indices < 0) | (positive_indices >= num_gallery))):
        raise ValueError("positive_indices contains out-of-range gallery indices.")

    positive_scores = similarity[jnp.arange(num_queries), positive_indices]
    # Count strictly higher scores. Ties keep the positive at the best tied rank.
    return jnp.sum(similarity > positive_scores[:, None], axis=1)


def retrieval_metrics_from_ranks(ranks: Array) -> RetrievalMetrics:
    ranks_np = np.asarray(jax.device_get(ranks), dtype=np.int64)
    if ranks_np.ndim != 1:
        raise ValueError(f"Expected ranks [N], got {ranks_np.shape}.")
    if ranks_np.size == 0:
        raise ValueError("Cannot compute retrieval metrics for zero samples.")
    return RetrievalMetrics(
        recall_at_1=float(np.mean(ranks_np < 1)),
        recall_at_5=float(np.mean(ranks_np < 5)),
        mean_rank=float(np.mean(ranks_np + 1)),
        sample_count=int(ranks_np.size),
    )


def retrieval_metrics(
    query_embeddings: Array,
    gallery_embeddings: Array,
    positive_indices: Array | None = None,
) -> tuple[RetrievalMetrics, Array]:
    query_embeddings = l2_normalize(jnp.asarray(query_embeddings))
    gallery_embeddings = l2_normalize(jnp.asarray(gallery_embeddings))
    similarity = query_embeddings @ gallery_embeddings.T
    ranks = ranks_from_similarity(similarity, positive_indices)
    return retrieval_metrics_from_ranks(ranks), ranks


_SIDE_NAMES = {0: "left", 1: "right"}


def retrieval_metrics_by_side(
    query_embeddings: Array,
    gallery_embeddings: Array,
    side_id: Array | np.ndarray,
) -> tuple[dict[str, float | int], np.ndarray]:
    """Compute per-wrist square retrieval, then micro-average totals.

    Returns ``(metric_dict, ranks)`` where ``ranks`` aligns with the original
    query order (zero-based). Queries whose side has no peers receive rank 0.
    """

    query = np.asarray(jax.device_get(l2_normalize(jnp.asarray(query_embeddings))), dtype=np.float32)
    gallery = np.asarray(
        jax.device_get(l2_normalize(jnp.asarray(gallery_embeddings))), dtype=np.float32
    )
    sides = np.asarray(side_id, dtype=np.int64)
    if query.shape[0] != sides.shape[0] or gallery.shape[0] != sides.shape[0]:
        raise ValueError(
            f"query/gallery/side_id length mismatch: "
            f"{query.shape[0]}, {gallery.shape[0]}, {sides.shape[0]}"
        )

    ranks = np.zeros((query.shape[0],), dtype=np.int64)
    side_metrics: dict[int, RetrievalMetrics] = {}
    for side_value in sorted(np.unique(sides).tolist()):
        mask = sides == int(side_value)
        indices = np.flatnonzero(mask)
        if indices.size == 0:
            continue
        side_query = jnp.asarray(query[indices])
        side_gallery = jnp.asarray(gallery[indices])
        metrics, side_ranks = retrieval_metrics(side_query, side_gallery)
        ranks[indices] = np.asarray(jax.device_get(side_ranks), dtype=np.int64)
        side_metrics[int(side_value)] = metrics

    total = retrieval_metrics_from_ranks(jnp.asarray(ranks))
    metric_dict: dict[str, float | int] = {
        "recall@1": total.recall_at_1,
        "recall@5": total.recall_at_5,
        "mean_rank": total.mean_rank,
        "sample_count": total.sample_count,
    }
    for side_value, name in _SIDE_NAMES.items():
        if side_value not in side_metrics:
            metric_dict[f"recall@1_{name}"] = float("nan")
            metric_dict[f"recall@5_{name}"] = float("nan")
            metric_dict[f"mean_rank_{name}"] = float("nan")
            metric_dict[f"sample_count_{name}"] = 0
            continue
        metrics = side_metrics[side_value]
        metric_dict[f"recall@1_{name}"] = metrics.recall_at_1
        metric_dict[f"recall@5_{name}"] = metrics.recall_at_5
        metric_dict[f"mean_rank_{name}"] = metrics.mean_rank
        metric_dict[f"sample_count_{name}"] = metrics.sample_count
    return metric_dict, ranks

