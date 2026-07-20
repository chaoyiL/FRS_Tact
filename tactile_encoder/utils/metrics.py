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

