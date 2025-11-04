"""
backend/agents/research_agent.py

LangGraph-based research agent with:
  - Hybrid retrieval (semantic + BM25)
  - Cohere re-ranking
  - Citation-tracked answer generation
  - Query contextualization (handles follow-up questions)
  - Streaming support
"""
from __future__ import annotations
import json
from dataclasses import dataclass, field
from typing import Any, AsyncIterator

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from ..core.llm import get_llm, get_streaming_llm
from ..retrieval.vector_store import get_vector_store
from ..retrieval.bm25_retriever import get_bm25_retriever
from ..retrieval.hybrid import HybridRetriever
from ..retrieval.reranker import get_reranker
from ..core.config import get_settings
from .prompts import RESEARCH_SYSTEM_PROMPT, QUERY_CONTEXTUALIZATION_PROMPT


@dataclass
class Citation:
    source_name: str
    source_type: str
    page_number: int | None
    url: str | None
    chunk_id: str
    relevance_score: float
    retrieval_source: str  # "semantic" | "bm25" | "both"

    def to_dict(self) -> dict:
        return {
            "source_name": self.source_name,
            "source_type": self.source_type,
            "page_number": self.page_number,
            "url": self.url,
            "chunk_id": self.chunk_id,
            "relevance_score": self.relevance_score,
            "retrieval_source": self.retrieval_source,
        }


@dataclass
class ResearchResult:
    answer: str
    citations: list[Citation]
    context_chunks: list[str]  # Raw chunks used
    query: str
    contextualized_query: str  # After query expansion

    def to_dict(self) -> dict:
        return {
            "answer": self.answer,
            "citations": [c.to_dict() for c in self.citations],
            "query": self.query,
            "contextualized_query": self.contextualized_query,
            "num_chunks_retrieved": len(self.context_chunks),
        }


def _docs_to_citations(docs: list[Document]) -> list[Citation]:
    """Extract Citation objects from retrieved documents."""
    citations = []
    for doc in docs:
        m = doc.metadata
        citations.append(Citation(
            source_name=m.get("source_name", "Unknown"),
            source_type=m.get("source_type", "unknown"),
            page_number=m.get("page_number"),
            url=m.get("url"),
            chunk_id=m.get("chunk_id", ""),
            relevance_score=m.get("relevance_score", m.get("rrf_score", 0.0)),
            retrieval_source=m.get("retrieval_source", "semantic"),
        ))
    return citations


def _format_context(docs: list[Document]) -> str:
    """Format retrieved documents into a structured context block for the LLM."""
    parts = []
    for i, doc in enumerate(docs, start=1):
        m = doc.metadata
        source = m.get("source_name", "Unknown")
        page = f", Page {m['page_number']}" if m.get("page_number") else ""
        url = f"\nURL: {m['url']}" if m.get("url") else ""
        score = m.get("relevance_score", m.get("rrf_score", 0))

        header = f"[Document {i}] Source: {source}{page}{url} (relevance: {score:.3f})"
        parts.append(f"{header}\n{doc.page_content}")

    return "\n\n---\n\n".join(parts)


class ResearchAgent:
    """
    Full research RAG pipeline:
    1. Contextualize query (handle follow-ups)
    2. Hybrid retrieve (semantic + BM25 + RRF fusion)
    3. Re-rank with Cohere
    4. Generate cited answer with GPT-4o
    """

    def __init__(self, user_id: str):
        self._user_id = user_id
        self._settings = get_settings()
        self._llm = get_llm()
        self._vector_store = get_vector_store(user_id)
        self._bm25 = get_bm25_retriever(user_id)
        self._hybrid = HybridRetriever(self._vector_store, self._bm25)
        self._reranker = get_reranker()
        self._contextualize_prompt = ChatPromptTemplate.from_template(
            QUERY_CONTEXTUALIZATION_PROMPT
        )
        self._contextualize_chain = (
            self._contextualize_prompt | self._llm | StrOutputParser()
        )

    def _contextualize_query(
        self,
        query: str,
        chat_history: list[dict] | None = None,
    ) -> str:
        """Rewrite follow-up questions to standalone questions."""
        if not chat_history:
            return query

        history_str = "\n".join([
            f"{msg['role'].capitalize()}: {msg['content']}"
            for msg in chat_history[-6:]  # Last 3 turns
        ])

        try:
            return self._contextualize_chain.invoke({
                "chat_history": history_str,
                "question": query,
            })
        except Exception:
            return query  # Fallback to original query

    def query(
        self,
        query: str,
        chat_history: list[dict] | None = None,
        filters: dict[str, str] | None = None,
    ) -> ResearchResult:
        """
        Full synchronous query pipeline.
        Returns structured result with answer + citations.
        """
        # Step 1: Contextualize
        contextualized = self._contextualize_query(query, chat_history)

        # Step 2: Hybrid retrieve
        hybrid_results = self._hybrid.retrieve(
            query=contextualized,
            k_semantic=self._settings.top_k_semantic,
            k_bm25=self._settings.top_k_bm25,
            filters=filters,
        )

        # Step 3: Re-rank
        reranked_docs = self._reranker.rerank(
            query=contextualized,
            documents=hybrid_results,
            top_n=self._settings.top_k_final,
        )

        if not reranked_docs:
            return ResearchResult(
                answer=(
                    "I couldn't find any relevant information in the knowledge base "
                    "to answer your question. Please ensure relevant documents have "
                    "been ingested, or try rephrasing your question."
                ),
                citations=[],
                context_chunks=[],
                query=query,
                contextualized_query=contextualized,
            )

        # Step 4: Format context and generate answer with Gemini
        context = _format_context(reranked_docs)
        citations = _docs_to_citations(reranked_docs)

        messages = [
            SystemMessage(content=RESEARCH_SYSTEM_PROMPT),
            HumanMessage(content=(
                f"Context Documents:\n\n{context}\n\n"
                f"---\n\nResearch Question: {query}"
            )),
        ]

        response = self._llm.invoke(messages)
        answer = response.content

        return ResearchResult(
            answer=answer,
            citations=citations,
            context_chunks=[doc.page_content for doc in reranked_docs],
            query=query,
            contextualized_query=contextualized,
        )

    async def stream_query(
        self,
        query: str,
        chat_history: list[dict] | None = None,
        filters: dict[str, str] | None = None,
    ) -> AsyncIterator[str]:
        """
        Streaming version — yields answer tokens as they arrive.
        Yields special JSON markers for citations at the end.
        """
        streaming_llm = get_streaming_llm()

        # Steps 1–3 same as above
        contextualized = self._contextualize_query(query, chat_history)
        hybrid_results = self._hybrid.retrieve(query=contextualized, filters=filters)
        reranked_docs = self._reranker.rerank(
            query=contextualized,
            documents=hybrid_results,
        )

        if not reranked_docs:
            yield "I couldn't find relevant information. Please ingest documents first."
            return

        context = _format_context(reranked_docs)
        citations = _docs_to_citations(reranked_docs)

        messages = [
            SystemMessage(content=RESEARCH_SYSTEM_PROMPT),
            HumanMessage(content=(
                f"Context Documents:\n\n{context}\n\n"
                f"---\n\nResearch Question: {query}"
            )),
        ]

        # Stream answer tokens
        async for chunk in streaming_llm.astream(messages):
            if chunk.content:
                yield chunk.content

        # Send citations as a final structured marker
        citations_payload = {
            "__citations__": [c.to_dict() for c in citations],
            "__contextualized_query__": contextualized,
        }
        yield f"\n\n__CITATIONS_JSON__{json.dumps(citations_payload)}"


_agent_instances: dict[str, ResearchAgent] = {}


def get_research_agent(user_id: str) -> ResearchAgent:
    """Per-user research agent, lazily created and cached."""
    if user_id not in _agent_instances:
        _agent_instances[user_id] = ResearchAgent(user_id)
    return _agent_instances[user_id]
