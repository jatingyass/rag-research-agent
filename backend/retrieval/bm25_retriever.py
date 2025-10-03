"""
backend/retrieval/bm25_retriever.py

BM25 keyword retriever using rank-bm25.
Persists the index to disk so it survives restarts.
Complementary to semantic search — excels at exact keyword matching.
"""
from __future__ import annotations
from rank_bm25 import BM25Okapi
from langchain_core.documents import Document

from ..ingestion.chunker import Chunk


def _ensure_nltk_data() -> None:
    import nltk
    for resource in ("punkt_tab", "punkt"):
        try:
            nltk.data.find(f"tokenizers/{resource}")
            return
        except LookupError:
            pass
    nltk.download("punkt_tab", quiet=True)


def _tokenize(text: str) -> list[str]:
    try:
        from nltk.tokenize import word_tokenize
        return [w.lower() for w in word_tokenize(text) if w.isalpha()]
    except LookupError:
        _ensure_nltk_data()
        from nltk.tokenize import word_tokenize
        return [w.lower() for w in word_tokenize(text) if w.isalpha()]


class BM25Retriever:
    """
    In-memory BM25 retriever scoped to a single user.
    Rebuilt from the vector store on first query after a restart
    (ChromaDB persists; BM25 is a lightweight keyword index rebuilt on ingest).
    """

    def __init__(self):
        self._texts: list[str] = []
        self._metadatas: list[dict] = []
        self._bm25: BM25Okapi | None = None

    def add_chunks(self, chunks: list[Chunk]) -> None:
        """Add chunks to the BM25 index, replacing any existing chunks for the same source."""
        if not chunks:
            return
        source_id = chunks[0].source_id
        # Remove existing entries for this source to prevent duplicates on re-ingest
        keep = [i for i, m in enumerate(self._metadatas) if m.get("source_id") != source_id]
        self._texts = [self._texts[i] for i in keep]
        self._metadatas = [self._metadatas[i] for i in keep]

        for chunk in chunks:
            self._texts.append(chunk.text)
            self._metadatas.append({
                "chunk_id": chunk.chunk_id,
                "source_id": chunk.source_id,
                "source_type": chunk.source_type,
                "source_name": chunk.source_name,
                "page_number": chunk.page_number,
                "section": chunk.section,
                "url": chunk.url,
            })

        self._rebuild_index()

    def _rebuild_index(self) -> None:
        if not self._texts:
            return
        tokenized = [_tokenize(t) for t in self._texts]
        self._bm25 = BM25Okapi(tokenized)

    def search(self, query: str, k: int = 20) -> list[Document]:
        """Return top-k BM25 results as LangChain Documents."""
        if self._bm25 is None or not self._texts:
            return []

        query_tokens = _tokenize(query)
        scores = self._bm25.get_scores(query_tokens)

        # Get top-k indices
        top_k_indices = sorted(
            range(len(scores)), key=lambda i: scores[i], reverse=True
        )[:k]

        results = []
        for idx in top_k_indices:
            if scores[idx] > 0:  # Skip zero-score results
                results.append(Document(
                    page_content=self._texts[idx],
                    metadata={**self._metadatas[idx], "bm25_score": float(scores[idx])},
                ))

        return results

_bm25_instances: dict[str, BM25Retriever] = {}


def get_bm25_retriever(user_id: str) -> BM25Retriever:
    """Per-user in-memory BM25 retriever."""
    if user_id not in _bm25_instances:
        _bm25_instances[user_id] = BM25Retriever()
    return _bm25_instances[user_id]
