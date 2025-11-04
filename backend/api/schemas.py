"""
backend/api/schemas.py
All Pydantic request/response models.
"""
from __future__ import annotations
from typing import Any, Literal
from pydantic import BaseModel, Field, HttpUrl


# ── Ingestion ──────────────────────────────────────────────────────────────

class IngestURLRequest(BaseModel):
    url: str = Field(..., description="URL of the web page to ingest")
    tags: list[str] = Field(default_factory=list, description="Optional tags for filtering")


class IngestDatabaseRequest(BaseModel):
    db_url: str = Field(..., description="SQLAlchemy database URL")
    tables: list[str] = Field(
        default_factory=list,
        description="Specific tables to ingest. Empty = all tables.",
    )
    max_rows_per_table: int = Field(default=5000, ge=1, le=100000)


class IngestResponse(BaseModel):
    source_id: str
    source_name: str
    source_type: str
    num_chunks: int
    status: Literal["success", "error"]
    message: str = ""


# ── Sources ─────────────────────────────────────────────────────────────────

class SourceInfo(BaseModel):
    source_id: str
    source_name: str
    source_type: str
    num_chunks: int | None = None


class SourcesResponse(BaseModel):
    sources: list[SourceInfo]
    total: int


# ── Query ───────────────────────────────────────────────────────────────────

class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)
    chat_history: list[ChatMessage] = Field(default_factory=list)
    stream: bool = Field(default=False)
    filters: dict[str, str] | None = Field(
        default=None,
        description='Metadata filters applied at retrieval time. e.g. {"source_type": "pdf"}',
    )


class CitationInfo(BaseModel):
    source_name: str
    source_type: str
    page_number: int | None = None
    url: str | None = None
    chunk_id: str
    relevance_score: float
    retrieval_source: str


class QueryResponse(BaseModel):
    answer: str
    citations: list[CitationInfo]
    query: str
    contextualized_query: str
    num_chunks_retrieved: int
    model: str = "gemini-2.5-flash"


# ── Evaluation ──────────────────────────────────────────────────────────────

class EvalTestCase(BaseModel):
    question: str
    ground_truth: str


class EvalRequest(BaseModel):
    test_cases: list[EvalTestCase] = Field(..., min_length=1)


class EvalMetrics(BaseModel):
    faithfulness: float
    answer_relevancy: float
    context_precision: float
    context_recall: float
    overall_score: float
    num_test_cases: int


class EvalResponse(BaseModel):
    aggregate: EvalMetrics
    per_question: list[dict[str, Any]]
    status: Literal["success", "error"]


# ── Health ───────────────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status: str
    vector_db: str
    num_sources: int
    version: str = "1.0.0"
