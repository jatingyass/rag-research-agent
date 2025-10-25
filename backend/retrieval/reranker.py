"""
backend/retrieval/reranker.py

Cohere re-ranking: takes hybrid search candidates and re-scores them
using a cross-encoder model (much more accurate than bi-encoder retrieval).

Why re-rank?
  - Retrieval (bi-encoder) = fast but coarse
  - Re-ranking (cross-encoder) = slow but precise
  - We run retrieval on 20-40 candidates, re-rank, keep top 6
  - This pattern is used by production RAG systems at Notion, Perplexity, etc.
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
    """Isolated Cohere API call so tenacity only retries the network hop."""
    return client.rerank(
        model=model,
        query=query,
        documents=documents,
        top_n=top_n,
        return_documents=False,
    )


class CohereReranker:
    """
    Re-ranks documents using Cohere's rerank API.
    Falls back to RRF-score ordering if Cohere key not configured.
    """

    def __init__(self):
        self._settings = get_settings()
        self._client = None
        self._init_client()

    def _init_client(self) -> None:
        if self._settings.cohere_api_key:
            try:
                import cohere
                self._client = cohere.Client(self._settings.cohere_api_key)
                logger.info("Cohere reranker initialised")
            except ImportError:
                logger.warning("cohere package not installed — using fallback ordering")

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

        if self._client is None:
            for doc in docs[:top_n]:
                doc.metadata.setdefault("relevance_score", 0.5)
                doc.metadata["reranked"] = False
            return docs[:top_n]

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
                reranked.append(doc)

            return reranked

        except Exception as e:
            logger.warning(f"Cohere rerank failed after retries ({e}). Using fallback.")
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
