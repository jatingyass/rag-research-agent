"""
backend/retrieval/hybrid.py

Hybrid search: combines semantic (vector) + keyword (BM25) results
using Reciprocal Rank Fusion (RRF).

Why RRF?
  - No need to normalize scores from different scales
  - Stable and well-calibrated
  - Simple to implement, hard to beat in practice
  - Used by Elasticsearch, Cohere, and most production RAG systems

Formula: RRF(d) = Σ 1 / (k + rank(d))  where k=60 (standard constant)
"""
from __future__ import annotations
from collections import defaultdict

from langchain_core.documents import Document

from .vector_store import VectorStore
from .bm25_retriever import BM25Retriever
from ..core.config import get_settings


RRF_K = 60  # Standard constant — reduces sensitivity to top-rank outliers


def reciprocal_rank_fusion(
    ranked_lists: list[list[Document]],
    k: int = RRF_K,
) -> list[tuple[Document, float]]:
    """
    Fuse multiple ranked document lists using RRF.
    Returns list of (document, rrf_score) sorted by score descending.
    """
    scores: dict[str, float] = defaultdict(float)
    doc_map: dict[str, Document] = {}

    for ranked_list in ranked_lists:
        for rank, doc in enumerate(ranked_list, start=1):
            # Use chunk_id as unique key; fall back to content hash
            doc_id = doc.metadata.get("chunk_id") or str(hash(doc.page_content))
            scores[doc_id] += 1.0 / (k + rank)
            doc_map[doc_id] = doc

    # Sort by fused score
    sorted_ids = sorted(scores.keys(), key=lambda d: scores[d], reverse=True)
    return [(doc_map[doc_id], scores[doc_id]) for doc_id in sorted_ids]


class HybridRetriever:
    """
    Runs semantic and BM25 searches in parallel, fuses with RRF.
    Returns deduplicated, relevance-ranked results.
    """

    def __init__(self, vector_store: VectorStore, bm25: BM25Retriever):
        self._vector_store = vector_store
        self._bm25 = bm25
        self._settings = get_settings()

    def retrieve(
        self,
        query: str,
        k_semantic: int | None = None,
        k_bm25: int | None = None,
        k_final: int | None = None,
        filters: dict[str, str] | None = None,
    ) -> list[tuple[Document, float]]:
        """
        Full hybrid retrieval pipeline.

        Args:
            query: User's question
            k_semantic: How many semantic candidates to fetch
            k_bm25: How many BM25 candidates to fetch
            k_final: How many fused results to return (before re-ranking)

        Returns:
            List of (Document, rrf_score) sorted by relevance
        """
        k_sem = k_semantic or self._settings.top_k_semantic
        k_bm = k_bm25 or self._settings.top_k_bm25
        k_out = k_final or self._settings.top_k_final

        # Run both retrievers; filters applied to vector search only (BM25 is in-memory)
        semantic_results = self._vector_store.similarity_search(query, k=k_sem, filter=filters)
        bm25_results = self._bm25.search(query, k=k_bm)

        # Fuse with RRF
        fused = reciprocal_rank_fusion([semantic_results, bm25_results])

        # Add retrieval source metadata for debugging
        sem_ids = {d.metadata.get("chunk_id") for d in semantic_results}
        bm_ids = {d.metadata.get("chunk_id") for d in bm25_results}

        for doc, score in fused:
            doc_id = doc.metadata.get("chunk_id", "")
            in_sem = doc_id in sem_ids
            in_bm = doc_id in bm_ids
            doc.metadata["retrieval_source"] = (
                "both" if (in_sem and in_bm)
                else "semantic" if in_sem
                else "bm25"
            )
            doc.metadata["rrf_score"] = round(score, 6)

        return fused[:k_out]
