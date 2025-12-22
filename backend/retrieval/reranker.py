"""
backend/retrieval/reranker.py

Two-tier re-ranking strategy:
  1. CohereReranker  — cloud cross-encoder, best quality, needs API key
  2. LocalCrossEncoderReranker — local ms-marco model, no API key required
  3. Fallback        — RRF score order (if both unavailable)

Priority: Cohere > Local > Fallback
"""
from __future__ import annotations
from typing import Any

from langchain_core.documents import Document
from tenacity import retry, stop_after_attempt, wait_exponential

from ..core.config import get_settings
from ..core.logging import get_logger

logger = get_logger("reranker")


@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=20), reraise=True)
def _cohere_rerank(client: Any, model: str, query: str, documents: list[str], top_n: int):
    return client.rerank(
        model=model,
        query=query,
        documents=documents,
        top_n=top_n,
        return_documents=False,
    )


class LocalCrossEncoderReranker:
    """
    Local cross-encoder re-ranker using sentence-transformers.
    No API key needed — model downloads on first use (~22 MB).
    Uses ms-marco-MiniLM-L-6-v2 which scores query-document relevance
    on the same scale as Cohere (higher = more relevant).
    """

    _model = None  # class-level lazy singleton

    def _get_model(self):
        if LocalCrossEncoderReranker._model is None:
            from sentence_transformers import CrossEncoder
            logger.info("Loading local cross-encoder (ms-marco-MiniLM-L-6-v2)...")
            LocalCrossEncoderReranker._model = CrossEncoder(
                "cross-encoder/ms-marco-MiniLM-L-6-v2"
            )
            logger.info("Local cross-encoder ready")
        return LocalCrossEncoderReranker._model

    def rerank(
        self,
        query: str,
        docs: list[Document],
        top_n: int,
    ) -> list[Document]:
        if not docs:
            return []

        import math

        model = self._get_model()
        pairs = [(query, doc.page_content) for doc in docs]
        raw_scores = model.predict(pairs)

        # Sigmoid-normalise ms-marco scores to [0, 1] for consistent
        # display alongside Cohere scores (which are already in [0, 1]).
        def _sigmoid(x: float) -> float:
            return 1.0 / (1.0 + math.exp(-x))

        ranked = sorted(
            zip(docs, [_sigmoid(float(s)) for s in raw_scores]),
            key=lambda x: x[1],
            reverse=True,
        )

        result = []
        for doc, score in ranked[:top_n]:
            doc.metadata["relevance_score"] = round(score, 4)
            doc.metadata["reranked"] = True
            doc.metadata["reranker"] = "local-cross-encoder"
            result.append(doc)

        return result


class CohereReranker:
    """
    Re-ranks documents using Cohere's rerank API.
    Falls back to LocalCrossEncoderReranker if Cohere key not configured.
    """

    def __init__(self):
        self._settings = get_settings()
        self._client = None
        self._local: LocalCrossEncoderReranker | None = None
        self._init_client()

    def _init_client(self) -> None:
        if self._settings.cohere_api_key:
            try:
                import cohere
                self._client = cohere.Client(self._settings.cohere_api_key)
                logger.info("Cohere reranker initialised")
            except ImportError:
                logger.warning("cohere package not installed — trying local fallback")
        else:
            logger.info("No COHERE_API_KEY — using local cross-encoder fallback")
            self._local = LocalCrossEncoderReranker()

    def rerank(
        self,
        query: str,
        documents: list[tuple[Document, float]],
        top_n: int | None = None,
    ) -> list[Document]:
        top_n = top_n or self._settings.top_k_final
        docs = [d for d, _ in documents]

        if not docs:
            return []

        # Tier 1: Cohere cloud
        if self._client is not None:
            try:
                response = _cohere_rerank(
                    self._client,
                    self._settings.cohere_rerank_model,
                    query,
                    [doc.page_content for doc in docs],
                    top_n,
                )
                reranked = []
                for result in response.results:
                    doc = docs[result.index]
                    doc.metadata["relevance_score"] = round(result.relevance_score, 4)
                    doc.metadata["reranked"] = True
                    doc.metadata["reranker"] = "cohere"
                    reranked.append(doc)
                return reranked
            except Exception as e:
                logger.warning(f"Cohere rerank failed ({e}). Falling back to local.")

        # Tier 2: Local cross-encoder
        if self._local is None:
            self._local = LocalCrossEncoderReranker()
        try:
            return self._local.rerank(query, docs, top_n)
        except Exception as e:
            logger.warning(f"Local reranker failed ({e}). Using RRF order.")

        # Tier 3: RRF order
        for doc in docs[:top_n]:
            doc.metadata.setdefault("relevance_score", 0.5)
            doc.metadata["reranked"] = False
        return docs[:top_n]


_reranker_instance: CohereReranker | None = None


def get_reranker() -> CohereReranker:
    global _reranker_instance
    if _reranker_instance is None:
        _reranker_instance = CohereReranker()
    return _reranker_instance
