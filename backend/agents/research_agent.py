"""
backend/agents/research_agent.py

LangGraph-based research agent with:
  - Query routing: classify as direct-answer vs retrieval-needed
  - Hybrid retrieval (semantic + BM25 + RRF fusion)
  - Cohere / local cross-encoder re-ranking
  - Citation-tracked answer generation
  - Query contextualization for follow-up questions
  - Streaming support
"""
from __future__ import annotations
import json
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Literal, TypedDict

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import END, StateGraph

from ..core.llm import get_llm, get_streaming_llm
from ..core.logging import get_logger
from ..retrieval.vector_store import get_vector_store
from ..retrieval.bm25_retriever import get_bm25_retriever
from ..retrieval.hybrid import HybridRetriever
from ..retrieval.reranker import get_reranker
from ..core.config import get_settings
from .prompts import RESEARCH_SYSTEM_PROMPT, QUERY_CONTEXTUALIZATION_PROMPT

logger = get_logger("research_agent")


# ── Data models ──────────────────────────────────────────────────────────────

@dataclass
class Citation:
    source_name: str
    source_type: str
    page_number: int | None
    url: str | None
    chunk_id: str
    relevance_score: float
    retrieval_source: str

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
    context_chunks: list[str]
    query: str
    contextualized_query: str
    route: str = "retrieve"

    def to_dict(self) -> dict:
        return {
            "answer": self.answer,
            "citations": [c.to_dict() for c in self.citations],
            "query": self.query,
            "contextualized_query": self.contextualized_query,
            "num_chunks_retrieved": len(self.context_chunks),
            "route": self.route,
        }


# ── LangGraph state ───────────────────────────────────────────────────────────

class AgentState(TypedDict):
    query: str
    contextualized_query: str
    chat_history: list[dict]
    filters: dict[str, str] | None
    hybrid_results: list          # list[tuple[Document, float]]
    reranked_docs: list           # list[Document]
    answer: str
    citations: list               # list[Citation]
    route: str                    # "direct" | "retrieve"


# ── Helper functions ──────────────────────────────────────────────────────────

def _docs_to_citations(docs: list[Document]) -> list[Citation]:
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


_ROUTE_PROMPT = ChatPromptTemplate.from_template(
    """Classify the following user query into exactly one category:

- "direct": The query is a greeting, a meta-question about the system (e.g.
  "what can you do?"), or something answerable from general knowledge without
  needing to search the knowledge base.
- "retrieve": The query requires looking up specific facts, documents, or
  data from the ingested knowledge base.

Query: {query}

Reply with only the single word: direct  OR  retrieve"""
)

_NO_DOCS_ANSWER = (
    "I couldn't find any relevant information in the knowledge base to answer "
    "your question. Please ensure relevant documents have been ingested, or "
    "try rephrasing your question."
)


# ── Node functions ────────────────────────────────────────────────────────────

def _node_contextualize(state: AgentState, llm, settings) -> AgentState:
    query = state["query"]
    chat_history = state.get("chat_history") or []
    if not chat_history:
        return {**state, "contextualized_query": query}

    history_str = "\n".join([
        f"{m['role'].capitalize()}: {m['content']}"
        for m in chat_history[-6:]
    ])
    prompt = ChatPromptTemplate.from_template(QUERY_CONTEXTUALIZATION_PROMPT)
    chain = prompt | llm | StrOutputParser()
    try:
        contextualized = chain.invoke({"chat_history": history_str, "question": query})
    except Exception:
        contextualized = query
    return {**state, "contextualized_query": contextualized}


def _node_route_query(state: AgentState, llm) -> AgentState:
    chain = _ROUTE_PROMPT | llm | StrOutputParser()
    try:
        route_raw = chain.invoke({"query": state["contextualized_query"]}).strip().lower()
        route: Literal["direct", "retrieve"] = "direct" if route_raw == "direct" else "retrieve"
    except Exception:
        route = "retrieve"
    logger.info(f"Query route: {route}")
    return {**state, "route": route}


def _node_retrieve(state: AgentState, hybrid: HybridRetriever, reranker, settings) -> AgentState:
    hybrid_results = hybrid.retrieve(
        query=state["contextualized_query"],
        k_semantic=settings.top_k_semantic,
        k_bm25=settings.top_k_bm25,
        filters=state.get("filters"),
    )
    reranked_docs = reranker.rerank(
        query=state["contextualized_query"],
        documents=hybrid_results,
        top_n=settings.top_k_final,
    )
    return {**state, "hybrid_results": hybrid_results, "reranked_docs": reranked_docs}


def _node_generate(state: AgentState, llm) -> AgentState:
    docs = state.get("reranked_docs") or []
    if not docs:
        return {**state, "answer": _NO_DOCS_ANSWER, "citations": []}

    context = _format_context(docs)
    citations = _docs_to_citations(docs)
    messages = [
        SystemMessage(content=RESEARCH_SYSTEM_PROMPT),
        HumanMessage(content=(
            f"Context Documents:\n\n{context}\n\n"
            f"---\n\nResearch Question: {state['query']}"
        )),
    ]
    response = llm.invoke(messages)
    return {**state, "answer": response.content, "citations": citations}


def _node_direct_answer(state: AgentState, llm) -> AgentState:
    messages = [
        SystemMessage(content=(
            "You are a helpful research assistant. Answer the user's question "
            "conversationally. If it's a greeting, respond warmly. "
            "If they ask what you can do, explain you help research documents."
        )),
        HumanMessage(content=state["query"]),
    ]
    response = llm.invoke(messages)
    return {**state, "answer": response.content, "citations": []}


# ── Agent class ───────────────────────────────────────────────────────────────

class ResearchAgent:
    """
    LangGraph-based research agent.
    Graph: contextualize → route_query → [retrieve → generate | direct_answer]
    """

    def __init__(self, user_id: str):
        self._user_id = user_id
        self._settings = get_settings()
        self._llm = get_llm()
        self._vector_store = get_vector_store(user_id)
        self._bm25 = get_bm25_retriever(user_id)
        self._hybrid = HybridRetriever(self._vector_store, self._bm25)
        self._reranker = get_reranker()
        self._graph = self._build_graph()

    def _build_graph(self):
        llm = self._llm
        hybrid = self._hybrid
        reranker = self._reranker
        settings = self._settings

        graph = StateGraph(AgentState)

        graph.add_node("contextualize", lambda s: _node_contextualize(s, llm, settings))
        graph.add_node("route_query", lambda s: _node_route_query(s, llm))
        graph.add_node("retrieve", lambda s: _node_retrieve(s, hybrid, reranker, settings))
        graph.add_node("generate", lambda s: _node_generate(s, llm))
        graph.add_node("direct_answer", lambda s: _node_direct_answer(s, llm))

        graph.set_entry_point("contextualize")
        graph.add_edge("contextualize", "route_query")
        graph.add_conditional_edges(
            "route_query",
            lambda s: s["route"],
            {"retrieve": "retrieve", "direct": "direct_answer"},
        )
        graph.add_edge("retrieve", "generate")
        graph.add_edge("generate", END)
        graph.add_edge("direct_answer", END)

        return graph.compile()

    def query(
        self,
        query: str,
        chat_history: list[dict] | None = None,
        filters: dict[str, str] | None = None,
    ) -> ResearchResult:
        initial: AgentState = {
            "query": query,
            "contextualized_query": query,
            "chat_history": chat_history or [],
            "filters": filters,
            "hybrid_results": [],
            "reranked_docs": [],
            "answer": "",
            "citations": [],
            "route": "retrieve",
        }
        final = self._graph.invoke(initial)
        return ResearchResult(
            answer=final["answer"],
            citations=final["citations"],
            context_chunks=[d.page_content for d in final.get("reranked_docs", [])],
            query=query,
            contextualized_query=final["contextualized_query"],
            route=final["route"],
        )

    async def stream_query(
        self,
        query: str,
        chat_history: list[dict] | None = None,
        filters: dict[str, str] | None = None,
    ) -> AsyncIterator[str]:
        streaming_llm = get_streaming_llm()

        # Run the graph synchronously for retrieval steps, then stream generation
        initial: AgentState = {
            "query": query,
            "contextualized_query": query,
            "chat_history": chat_history or [],
            "filters": filters,
            "hybrid_results": [],
            "reranked_docs": [],
            "answer": "",
            "citations": [],
            "route": "retrieve",
        }

        # Contextualize and route
        state = _node_contextualize(initial, self._llm, self._settings)
        state = _node_route_query(state, self._llm)

        if state["route"] == "direct":
            messages = [
                SystemMessage(content=(
                    "You are a helpful research assistant. Answer conversationally."
                )),
                HumanMessage(content=query),
            ]
            async for chunk in streaming_llm.astream(messages):
                if chunk.content:
                    yield chunk.content
            yield f"\n\n__CITATIONS_JSON__{json.dumps({'__citations__': [], '__contextualized_query__': state['contextualized_query']})}"
            return

        # Retrieve
        state = _node_retrieve(state, self._hybrid, self._reranker, self._settings)
        docs = state.get("reranked_docs") or []

        if not docs:
            yield _NO_DOCS_ANSWER
            return

        context = _format_context(docs)
        citations = _docs_to_citations(docs)
        messages = [
            SystemMessage(content=RESEARCH_SYSTEM_PROMPT),
            HumanMessage(content=(
                f"Context Documents:\n\n{context}\n\n"
                f"---\n\nResearch Question: {query}"
            )),
        ]

        async for chunk in streaming_llm.astream(messages):
            if chunk.content:
                yield chunk.content

        citations_payload = {
            "__citations__": [c.to_dict() for c in citations],
            "__contextualized_query__": state["contextualized_query"],
        }
        yield f"\n\n__CITATIONS_JSON__{json.dumps(citations_payload)}"


_agent_instances: dict[str, ResearchAgent] = {}


def get_research_agent(user_id: str = "default") -> ResearchAgent:
    if user_id not in _agent_instances:
        _agent_instances[user_id] = ResearchAgent(user_id)
    return _agent_instances[user_id]
